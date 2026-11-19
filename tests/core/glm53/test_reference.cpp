// glm53/reference: the semantics oracle against golden activations from the
// pinned transformers reference (tools/glm53-golden), decoded one token at a
// time where the reference ran one causal prefill; plus the invariants of its
// building blocks (selection, routing, RoPE, norms) and of binding.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "core/glm53/config.hpp"
#include "core/glm53/reference.hpp"
#include "core/safetensors.hpp"

using namespace ds4n;
using namespace ds4n::glm53;

namespace {

/** The golden file: tensors as FP32 / integer copies, and its config. */
struct Golden {
  std::map<std::string, std::vector<float>> f32;
  std::map<std::string, std::vector<uint64_t>> shapes;
  std::map<std::string, std::vector<int64_t>> ints;
  std::map<std::string, std::string> metadata;

  TensorLookup lookup() const {
    return [this](const std::string& name) -> std::optional<TensorView> {
      const auto it = f32.find(name);
      if (it == f32.end()) return std::nullopt;
      return TensorView{it->second, shapes.at(name)};
    };
  }
};

const Golden& golden() {
  static const Golden g = [] {
    const std::string path = std::string(DS4N_SOURCE_DIR) + "/tests/data/glm53-golden.safetensors";
    std::ifstream f(path, std::ios::binary);
    REQUIRE_MESSAGE(f.good(), "cannot open " << path);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<std::byte> bytes(raw.size());
    std::memcpy(bytes.data(), raw.data(), raw.size());
    auto h = safetensors::parse_header(bytes);
    REQUIRE(h.is_ok());
    Golden out;
    out.metadata = h.value().metadata;
    for (const auto& [name, t] : h.value().tensors) {
      out.shapes[name] = t.shape;
      const std::byte* p = bytes.data() + t.offset;
      if (t.dtype == safetensors::Dtype::F32) {
        std::vector<float> v(t.bytes / 4);
        std::memcpy(v.data(), p, t.bytes);
        out.f32[name] = std::move(v);
      } else if (t.dtype == safetensors::Dtype::I64) {
        std::vector<int64_t> v(t.bytes / 8);
        std::memcpy(v.data(), p, t.bytes);
        out.ints[name] = std::move(v);
      } else if (t.dtype == safetensors::Dtype::I32) {
        std::vector<int32_t> v(t.bytes / 4);
        std::memcpy(v.data(), p, t.bytes);
        out.ints[name] = std::vector<int64_t>(v.begin(), v.end());
      }
    }
    return out;
  }();
  return g;
}

Config golden_config() {
  auto c = parse_config(golden().metadata.at("config_json"));
  REQUIRE_MESSAGE(c.is_ok(), (c.is_ok() ? "" : c.error().what));
  return c.value();
}

/** Largest |a - b| relative to the reference's scale: max|b| over the row, at least 1. */
double rel_err(std::span<const float> a, std::span<const float> b) {
  double worst = 0.0, scale = 1.0;
  for (float v : b) scale = std::max(scale, static_cast<double>(std::fabs(v)));
  for (size_t i = 0; i < a.size(); i++) worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]));
  return worst / scale;
}

std::vector<float> random_vector(std::mt19937& rng, size_t n, float sd = 1.0f) {
  std::normal_distribution<float> d(0.0f, sd);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

}  // namespace

// The reference ran one causal prefill in FP32; the oracle decodes token by
// token from its own caches. Agreement to ~1e-5 of each row's scale through all
// seven layers checks MLA (absorbed vs expanded), RoPE, the indexer, Shared
// Selections, routing and the Experts at once. The generator chose a seed whose
// every top-k decision has a relative margin above 1e-3, so no selection here
// sits on a tie FP32 noise could flip.
TEST_CASE("decoding token by token reproduces the reference's causal prefill") {
  const Golden& g = golden();
  const Config config = golden_config();
  const Dims& d = config.dims();
  auto bound = bind_model(config, g.lookup());
  REQUIRE_MESSAGE(bound.is_ok(), (bound.is_ok() ? "" : bound.error().what));
  const ModelWeights& weights = bound.value();

  const auto& ids = g.ints.at("input_ids");
  const auto H = static_cast<size_t>(d.hidden), V = static_cast<size_t>(d.vocab);
  constexpr double kTol = 2e-5;
  double worst_hidden = 0.0, worst_logits = 0.0, worst_latent = 0.0, worst_key = 0.0;
  int sparse_selections = 0;

  Session session = empty_session(config);
  for (size_t p = 0; p < ids.size(); p++) {
    auto step = decode_token(config, weights, session, static_cast<int32_t>(ids[p]));
    REQUIRE(step.is_ok());
    const TokenStep& s = step.value();

    for (int L = 0; L < d.layers; L++) {
      const auto& want = g.f32.at("hidden." + std::to_string(L));
      const double e = rel_err(s.layers[static_cast<size_t>(L)].out, std::span(want).subspan(p * H, H));
      worst_hidden = std::max(worst_hidden, e);
      CHECK_MESSAGE(e < kTol, "layer " << L << " position " << p << " error " << e);

      // The reference's top-k at a position below index_topk also lists masked
      // future positions; the attended set is the valid ones.
      const auto& topk = g.ints.at("topk." + std::to_string(L));
      const auto K = static_cast<size_t>(d.index_topk);
      std::vector<int32_t> want_sel;
      for (size_t k = 0; k < K; k++)
        if (topk[p * K + k] <= static_cast<int64_t>(p)) want_sel.push_back(static_cast<int32_t>(topk[p * K + k]));
      std::sort(want_sel.begin(), want_sel.end());
      CHECK_MESSAGE(s.layers[static_cast<size_t>(L)].selection == want_sel, "layer " << L << " position " << p);
      if (p + 1 > K) sparse_selections++;

      // The cache rows: the KV latent exactly as the reference normalizes it,
      // and on Full layers the index key's non-RoPE part (RoPE leaves it alone).
      const CacheRow& row = s.layers[static_cast<size_t>(L)].row;
      const auto KVL = static_cast<size_t>(d.kv_lora);
      const auto& latent = g.f32.at("latent." + std::to_string(L));
      const double el_row = rel_err(row.latent, std::span(latent).subspan(p * KVL, KVL));
      worst_latent = std::max(worst_latent, el_row);
      CHECK(el_row < kTol);
      if (config.indexer(L) == IndexerKind::Full) {
        const auto IHD = static_cast<size_t>(d.index_head_dim), R = static_cast<size_t>(d.rope);
        const auto& key = g.f32.at("index_key_norm." + std::to_string(L));
        const double ek = rel_err(std::span(row.index_key).subspan(R), std::span(key).subspan(p * IHD + R, IHD - R));
        worst_key = std::max(worst_key, ek);
        CHECK(ek < kTol);
      } else {
        CHECK(g.f32.count("index_key_norm." + std::to_string(L)) == 0);
      }
    }
    const double ef = rel_err(s.final_norm, std::span(g.f32.at("final_norm")).subspan(p * H, H));
    const double el = rel_err(s.logits, std::span(g.f32.at("logits")).subspan(p * V, V));
    worst_logits = std::max(worst_logits, el);
    CHECK(ef < kTol);
    CHECK(el < kTol);
    append(session, s);
  }
  CHECK(session.length() == static_cast<int64_t>(ids.size()));
  CHECK(sparse_selections > 0);  // DSA really dropped keys (16 positions, top-k 5)
  MESSAGE("worst relative error: hidden " << worst_hidden << ", logits " << worst_logits << ", latent "
                                          << worst_latent << ", index key " << worst_key);
}

// The real-weights run cannot hold the model: it binds one layer at a time and
// runs every token of a batch through begin_layer before finishing any, with
// the routed Experts supplied from outside. That order must change nothing.
TEST_CASE("layer-major decoding with per-layer binding is bit-identical to decode_token") {
  const Golden& g = golden();
  const Config config = golden_config();
  const Dims& d = config.dims();
  auto bound = bind_model(config, g.lookup());
  REQUIRE(bound.is_ok());
  const auto& ids = g.ints.at("input_ids");

  // Token-major, as decode_token runs.
  Session session = empty_session(config);
  std::vector<TokenStep> want;
  for (int64_t id : ids) {
    auto step = decode_token(config, bound.value(), session, static_cast<int32_t>(id));
    REQUIRE(step.is_ok());
    append(session, step.value());
    want.push_back(std::move(step).value());
  }

  // Layer-major: each layer bound alone, every token begun, then every token finished.
  std::vector<LayerCache> caches(static_cast<size_t>(d.layers));
  std::vector<std::vector<float>> x;
  for (int64_t id : ids) {
    const auto e = std::span(g.f32.at("model.embed_tokens.weight")).subspan(static_cast<size_t>(id) * d.hidden, d.hidden);
    x.emplace_back(e.begin(), e.end());
  }
  std::vector<std::vector<int32_t>> selection(ids.size());
  for (int L = 0; L < d.layers; L++) {
    const auto layer = bind_layer(config, L, g.lookup());
    REQUIRE(layer.is_ok());
    std::vector<LayerStart> starts;
    for (size_t p = 0; p < ids.size(); p++) {
      auto s = begin_layer(config, layer.value(), L, caches[static_cast<size_t>(L)], x[p], selection[p]);
      REQUIRE(s.is_ok());
      append(caches[static_cast<size_t>(L)], s.value().row);
      if (config.indexer(L) == IndexerKind::Full) selection[p] = s.value().selection;
      starts.push_back(std::move(s).value());
    }
    for (size_t p = 0; p < ids.size(); p++) {
      std::vector<Mlp> routed;
      for (int32_t e : starts[p].route.expert) routed.push_back(bound.value().experts(L)[static_cast<size_t>(e)]);
      auto step = finish_layer(config, layer.value(), std::move(starts[p]), routed);
      REQUIRE(step.is_ok());
      const LayerStep& w = want[p].layers[static_cast<size_t>(L)];
      CHECK(step.value().out == w.out);
      CHECK(step.value().row.latent == w.row.latent);
      CHECK(step.value().row.index_key == w.row.index_key);
      CHECK(step.value().selection == w.selection);
      CHECK(step.value().route.expert == w.route.expert);
      CHECK(step.value().route.weight == w.route.weight);
      x[p] = step.value().out;
    }
  }
  for (size_t L = 0; L < caches.size(); L++) {
    CHECK(caches[L].latent == session.layers[L].latent);
    CHECK(caches[L].index_keys == session.layers[L].index_keys);
  }
}

TEST_CASE("finish_layer takes exactly the routed Experts its Route names, in an Expert's shape") {
  const Golden& g = golden();
  const Config config = golden_config();
  const Dims& d = config.dims();
  auto bound = bind_model(config, g.lookup());
  REQUIRE(bound.is_ok());
  int sparse = 0;
  while (config.mlp(sparse) != MlpKind::Sparse) sparse++;
  const auto layer = bind_layer(config, sparse, g.lookup());
  REQUIRE(layer.is_ok());
  const std::vector<float> x(static_cast<size_t>(d.hidden), 0.25f);
  const LayerCache empty;
  const std::vector<int32_t> self{0};
  auto start = [&] { return begin_layer(config, layer.value(), sparse, empty, x, self).value(); };
  const auto s = start();
  REQUIRE(s.route.expert.size() == static_cast<size_t>(d.experts_per_token));
  CHECK(s.route.margin >= 0.0f);

  std::vector<Mlp> routed;
  for (int32_t e : s.route.expert) routed.push_back(bound.value().experts(sparse)[static_cast<size_t>(e)]);
  CHECK(finish_layer(config, layer.value(), start(), routed).is_ok());
  CHECK_FALSE(finish_layer(config, layer.value(), start(), std::span(routed).first(routed.size() - 1)).is_ok());
  CHECK_FALSE(finish_layer(config, layer.value(), start(), {}).is_ok());
  std::vector<Mlp> wrong(routed);
  std::swap(wrong.front().gate, wrong.front().down);  // a down projection where the gate goes
  CHECK_FALSE(finish_layer(config, layer.value(), start(), wrong).is_ok());

  // A Dense layer takes none.
  const auto dense = bind_layer(config, 0, g.lookup());
  REQUIRE(dense.is_ok());
  REQUIRE(config.mlp(0) == MlpKind::Dense);
  auto ds = begin_layer(config, dense.value(), 0, empty, x, {});
  REQUIRE(ds.is_ok());
  CHECK(ds.value().route.expert.empty());
  CHECK_FALSE(finish_layer(config, dense.value(), ds.value(), routed).is_ok());
  CHECK(finish_layer(config, dense.value(), std::move(ds).value(), {}).is_ok());

  CHECK_FALSE(bind_layer(config, d.layers, g.lookup()).is_ok());
  CHECK_FALSE(bind_layer(config, -1, g.lookup()).is_ok());
}

TEST_CASE("the Session caches index keys on Full layers only") {
  const Golden& g = golden();
  const Config config = golden_config();
  const Dims& d = config.dims();
  auto bound = bind_model(config, g.lookup());
  REQUIRE(bound.is_ok());
  Session session = empty_session(config);
  for (int t = 0; t < 3; t++) {
    auto step = decode_token(config, bound.value(), session, 5 + t);
    REQUIRE(step.is_ok());
    append(session, step.value());
  }
  for (int L = 0; L < d.layers; L++) {
    const LayerCache& c = session.layers[static_cast<size_t>(L)];
    CHECK(c.length == 3);
    CHECK(c.latent.size() == 3u * static_cast<size_t>(d.kv_lora));
    CHECK(c.k_rope.size() == 3u * static_cast<size_t>(d.rope));
    const size_t keys = config.indexer(L) == IndexerKind::Full ? 3u * static_cast<size_t>(d.index_head_dim) : 0u;
    CHECK(c.index_keys.size() == keys);
  }
}

TEST_CASE("decode rejects what it cannot run") {
  const Golden& g = golden();
  const Config config = golden_config();
  auto bound = bind_model(config, g.lookup());
  REQUIRE(bound.is_ok());
  const Session session = empty_session(config);
  CHECK(!decode_token(config, bound.value(), session, -1).is_ok());
  CHECK(!decode_token(config, bound.value(), session, config.dims().vocab).is_ok());

  int shared = 0;
  while (config.indexer(shared) != IndexerKind::Shared) shared++;
  const std::vector<float> x(static_cast<size_t>(config.dims().hidden), 0.5f);
  const LayerCache empty;
  CHECK(!decode_layer(config, bound.value(), shared, empty, x, {}).is_ok());
  const std::vector<int32_t> future{1};
  CHECK(!decode_layer(config, bound.value(), shared, empty, x, future).is_ok());
  CHECK(!decode_layer(config, bound.value(), config.dims().layers, empty, x, {}).is_ok());
}

TEST_CASE("bind_model names a missing tensor, a wrong shape and a misplaced indexer") {
  const Golden& g = golden();
  const Config config = golden_config();
  const auto base = g.lookup();

  auto missing = bind_model(config, [&](const std::string& n) {
    return n == "model.layers.3.self_attn.kv_b_proj.weight" ? std::nullopt : base(n);
  });
  REQUIRE(!missing.is_ok());
  CHECK(missing.error().what.find("kv_b_proj") != std::string::npos);

  auto wrong = bind_model(config, [&](const std::string& n) -> std::optional<TensorView> {
    auto t = base(n);
    if (t && n == "model.norm.weight") t->shape = {t->shape[0] / 2, 2};
    return t;
  });
  REQUIRE(!wrong.is_ok());
  CHECK(wrong.error().what.find("model.norm.weight") != std::string::npos);

  int shared = 0;
  while (config.indexer(shared) != IndexerKind::Shared) shared++;
  const std::string stray = "model.layers." + std::to_string(shared) + ".self_attn.indexer.wk.weight";
  auto misplaced = bind_model(config, [&](const std::string& n) {
    return n == stray ? base("model.layers.0.self_attn.indexer.wk.weight") : base(n);
  });
  REQUIRE(!misplaced.is_ok());
  CHECK(misplaced.error().what.find("shared") != std::string::npos);
}

TEST_CASE("top_k: k best, ascending, every pick at least every non-pick, ties to the lower index") {
  std::mt19937 rng(11);
  for (int trial = 0; trial < 400; trial++) {
    const int n = std::uniform_int_distribution<int>(1, 60)(rng);
    const int k = std::uniform_int_distribution<int>(1, 70)(rng);
    // Few distinct values, so ties are common.
    std::vector<float> s(static_cast<size_t>(n));
    for (auto& v : s) v = static_cast<float>(std::uniform_int_distribution<int>(-3, 3)(rng));
    const auto pick = top_k(s, k);
    REQUIRE(pick.size() == static_cast<size_t>(std::min(n, k)));
    CHECK(std::is_sorted(pick.begin(), pick.end()));
    CHECK(std::adjacent_find(pick.begin(), pick.end()) == pick.end());
    std::vector<bool> in(static_cast<size_t>(n), false);
    for (int32_t i : pick) in[static_cast<size_t>(i)] = true;
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        if (in[static_cast<size_t>(i)] && !in[static_cast<size_t>(j)]) {
          CHECK(s[static_cast<size_t>(i)] >= s[static_cast<size_t>(j)]);
          if (s[static_cast<size_t>(i)] == s[static_cast<size_t>(j)]) CHECK(i < j);
        }
  }
}

TEST_CASE("route: chooses by biased score, weighs by the unbiased one, weights sum to the scale") {
  const Config config = golden_config();
  Dims d = config.dims();
  d.experts = 32;
  d.experts_per_token = 5;
  std::mt19937 rng(3);
  for (int trial = 0; trial < 300; trial++) {
    const auto logits = random_vector(rng, 32, 2.0f);
    const auto bias = random_vector(rng, 32, 0.3f);
    const Route r = route(d, logits, bias);
    REQUIRE(r.expert.size() == 5u);
    CHECK(std::is_sorted(r.expert.begin(), r.expert.end()));
    auto sig = [&](int e) { return 1.0 / (1.0 + std::exp(-static_cast<double>(logits[static_cast<size_t>(e)]))); };
    std::vector<bool> chosen(32, false);
    for (int32_t e : r.expert) chosen[static_cast<size_t>(e)] = true;
    for (int a = 0; a < 32; a++)
      for (int b = 0; b < 32; b++)
        if (chosen[static_cast<size_t>(a)] && !chosen[static_cast<size_t>(b)])
          CHECK(sig(a) + bias[static_cast<size_t>(a)] >= sig(b) + bias[static_cast<size_t>(b)] - 1e-6);
    // The margin is the worst pick's biased score minus the best non-pick's.
    CHECK(r.margin >= 0.0f);
    int runner = -1;
    for (int e = 0; e < 32; e++)
      if (!chosen[static_cast<size_t>(e)] &&
          (runner < 0 || sig(e) + bias[static_cast<size_t>(e)] > sig(runner) + bias[static_cast<size_t>(runner)]))
        runner = e;
    double worst = 1e30;
    for (int32_t e : r.expert) worst = std::min(worst, sig(e) + bias[static_cast<size_t>(e)]);
    CHECK(r.margin == doctest::Approx(worst - (sig(runner) + bias[static_cast<size_t>(runner)])).epsilon(1e-5).scale(1.0));
    const double sum = std::accumulate(r.weight.begin(), r.weight.end(), 0.0);
    CHECK(sum == doctest::Approx(d.routed_scale).epsilon(1e-6));
    for (size_t k = 1; k < r.expert.size(); k++)
      CHECK(r.weight[k] / r.weight[0] == doctest::Approx(sig(r.expert[k]) / sig(r.expert[0])).epsilon(1e-5));

    // Relabelling the Experts relabels the choice (no id is privileged).
    std::vector<int> perm(32);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<float> pl(32), pb(32);
    for (int e = 0; e < 32; e++) {
      pl[static_cast<size_t>(perm[static_cast<size_t>(e)])] = logits[static_cast<size_t>(e)];
      pb[static_cast<size_t>(perm[static_cast<size_t>(e)])] = bias[static_cast<size_t>(e)];
    }
    const Route rp = route(d, pl, pb);
    std::vector<int32_t> mapped;
    for (int32_t e : r.expert) mapped.push_back(perm[static_cast<size_t>(e)]);
    std::sort(mapped.begin(), mapped.end());
    CHECK(rp.expert == mapped);
  }
}

// The reference rounds each angle pos * inv_freq to FP32 (and so does the
// oracle), so a rotation is off by up to half an ulp of its angle: about
// pos * 2^-24 radians for the fastest pair (inv_freq = 1). Relative-position
// invariance therefore holds to |q||k| times twice that (both vectors, both
// positions), which at long context is not small — position 2^20 is already
// ~0.06 rad. The GPU executor must round its angles the same way to match.
TEST_CASE("rope is an isometry and depends only on the relative position, to the FP32 angle's rounding") {
  std::mt19937 rng(5);
  auto norm = [](std::span<const float> v) { return std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0)); };
  for (int trial = 0; trial < 200; trial++) {
    const auto q = random_vector(rng, 64), k = random_vector(rng, 64);
    const int64_t p = std::uniform_int_distribution<int64_t>(0, 4000)(rng);
    const int64_t t = std::uniform_int_distribution<int64_t>(0, 4000)(rng);
    const int64_t shift = std::uniform_int_distribution<int64_t>(1, 4000)(rng);
    CHECK(norm(rope(q, p, 8000000.0f)) == doctest::Approx(norm(q)).epsilon(1e-5));
    auto dot_at = [&](int64_t a, int64_t b) {
      const auto x = rope(q, a, 8000000.0f), y = rope(k, b, 8000000.0f);
      return std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
    };
    const double max_pos = static_cast<double>(std::max(p, t) + shift);
    const double bound = norm(q) * norm(k) * (4.0 * max_pos * std::ldexp(1.0, -24) + 1e-6);
    CHECK(std::fabs(dot_at(p, t) - dot_at(p + shift, t + shift)) <= bound);
  }
  // Position 0 is the identity up to the de-interleaved layout.
  const std::vector<float> x{1, 2, 3, 4, 5, 6};
  CHECK(rope(x, 0, 10000.0f) == std::vector<float>{1, 3, 5, 2, 4, 6});
}

TEST_CASE("rms_norm is scale invariant and layer_norm is shift invariant") {
  std::mt19937 rng(9);
  for (int trial = 0; trial < 200; trial++) {
    const auto x = random_vector(rng, 48, 3.0f);
    const auto w = random_vector(rng, 48), b = random_vector(rng, 48);
    const float c = std::uniform_real_distribution<float>(0.5f, 4.0f)(rng);
    std::vector<float> cx(x), sx(x);
    for (auto& v : cx) v *= c;
    for (auto& v : sx) v += c;
    const auto a = rms_norm(x, w, 1e-12f), ac = rms_norm(cx, w, 1e-12f);
    const auto l = layer_norm(x, w, b, 1e-12f), ls = layer_norm(sx, w, b, 1e-12f);
    for (size_t i = 0; i < x.size(); i++) {
      CHECK(ac[i] == doctest::Approx(a[i]).epsilon(1e-5).scale(1.0));
      CHECK(ls[i] == doctest::Approx(l[i]).epsilon(1e-4).scale(1.0));
    }
  }
}

TEST_CASE("index_scores: ReLU per head, so a key no head likes scores zero, and scores are linear in the weights") {
  const Config config = golden_config();
  const Dims& d = config.dims();
  std::mt19937 rng(13);
  const auto ihd = static_cast<size_t>(d.index_head_dim), heads = static_cast<size_t>(d.index_heads);
  for (int trial = 0; trial < 100; trial++) {
    const auto q = random_vector(rng, heads * ihd);
    const auto keys = random_vector(rng, 7 * ihd);
    const auto w = random_vector(rng, heads);
    std::vector<float> w2(w);
    for (auto& v : w2) v *= 2.0f;
    const auto s = index_scores(d, q, w, keys), s2 = index_scores(d, q, w2, keys);
    REQUIRE(s.size() == 7u);
    for (size_t t = 0; t < 7; t++) CHECK(s2[t] == doctest::Approx(2.0 * s[t]).epsilon(1e-6).scale(1.0));
    // A key exactly opposite to every head's query: all dot products negative, ReLU zeroes them.
    std::vector<float> anti(ihd, 0.0f);
    for (size_t h = 0; h < heads; h++)
      for (size_t i = 0; i < ihd; i++) anti[i] -= q[h * ihd + i];
    bool all_negative = true;
    for (size_t h = 0; h < heads; h++)
      all_negative &= std::inner_product(anti.begin(), anti.end(), q.begin() + static_cast<std::ptrdiff_t>(h * ihd),
                                         0.0) < 0.0;
    if (all_negative) CHECK(index_scores(d, q, w, anti)[0] == 0.0f);
  }
}

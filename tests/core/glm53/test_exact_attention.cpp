// glm53/exact_attention: the attention side in the Reference configuration
// (docs/design/glm53-reference-configuration.md, "The attention side"). Held
// three ways, as exact_moe is: witnesses (the Semantics oracle on the same BF16
// weights agrees within the configuration's roundings), structure (what each
// GPU's partial reads, the RoPE layout, the Selection's ties) and the bits (a
// hash of a decode, kAttentionBits).
//
// The model is the released config with the real per-head widths (latent 512,
// RoPE 64, nope 192, values 256, index heads of 128) and fewer heads and
// positions, so a unit test can decode it: q_lora 2048 gives the 2-plane GEMVs,
// index_topk 80 over 100 positions gives Selections that drop positions and
// flash decoding of 2 splits.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <sstream>

#include "core/glm53/config.hpp"
#include "core/glm53/exact_attention.hpp"
#include "core/glm53/reference.hpp"
#include "core/mx_reference.hpp"
#include "exact_fixtures.hpp"

using namespace ds4n::glm53;
namespace mx = ds4n::mx;
using ds4n::glm53::test::bf16_all;
using ds4n::glm53::test::random_vector;

namespace {

constexpr int kPositions = 100;

Config test_config() {
  std::ifstream f(std::string(DS4N_SOURCE_DIR) + "/ref/glm-5.3/config.json");
  std::stringstream s;
  s << f.rdbuf();
  auto j = nlohmann::json::parse(s.str());
  j["hidden_size"] = 1024;
  j["num_attention_heads"] = 2;
  j["num_key_value_heads"] = 2;
  j["q_lora_rank"] = 2048;
  j["index_n_heads"] = 4;
  j["index_topk"] = 80;
  j["num_hidden_layers"] = 3;
  j["first_k_dense_replace"] = 1;
  j["intermediate_size"] = 1024;
  j["vocab_size"] = 96;
  j.erase("mlp_layer_types");
  j["indexer_types"] = {"full", "shared", "full"};
  auto c = parse_config(j.dump());
  REQUIRE(c.is_ok());
  return std::move(c).value();
}

/** Random BF16 values: a linear's entries ~ N(0, 1/sqrt(cols)), a norm's ~ N(1, 0.1). */
std::vector<float> weights(std::mt19937& rng, int rows, int cols) {
  return bf16_all(random_vector(rng, static_cast<size_t>(rows) * static_cast<size_t>(cols),
                                1.0f / std::sqrt(static_cast<float>(cols))));
}
std::vector<float> norm(std::mt19937& rng, int n) {
  auto v = random_vector(rng, static_cast<size_t>(n), 0.1f);
  for (auto& x : v) x = mx::round_bf16(1.0f + x);
  return v;
}

/** One layer's Backbone attention weights, owned, with a LayerWeights view (the MLP is a dense one). */
struct Layer {
  std::vector<float> in_norm, post_norm, q_a, q_b, kv_a, kv_b, o, q_a_norm, kv_a_norm;
  std::vector<float> wq_b, wk, wproj, k_norm_w, k_norm_b, gate, up, down;
  LayerWeights view(const Dims& d, bool full) const {
    AttentionWeights a{Matrix{q_a, d.q_lora, d.hidden},
                       Matrix{q_b, d.heads * d.qk_head(), d.q_lora},
                       Matrix{kv_a, d.kv_lora + d.rope, d.hidden},
                       Matrix{kv_b, d.heads * (d.nope + d.v_head), d.kv_lora},
                       Matrix{o, d.hidden, d.heads * d.v_head},
                       q_a_norm,
                       kv_a_norm,
                       std::nullopt};
    if (full)
      a.indexer = IndexerWeights{Matrix{wq_b, d.index_heads * d.index_head_dim, d.q_lora},
                                 Matrix{wk, d.index_head_dim, d.hidden}, Matrix{wproj, d.index_heads, d.hidden},
                                 k_norm_w, k_norm_b};
    return LayerWeights{in_norm, post_norm, a,
                        Mlp{Matrix{gate, d.intermediate, d.hidden}, Matrix{up, d.intermediate, d.hidden},
                            Matrix{down, d.hidden, d.intermediate}}};
  }
};

Layer random_layer(std::mt19937& rng, const Dims& d) {
  Layer l;
  l.in_norm = norm(rng, d.hidden);
  l.post_norm = norm(rng, d.hidden);
  l.q_a = weights(rng, d.q_lora, d.hidden);
  l.q_b = weights(rng, d.heads * d.qk_head(), d.q_lora);
  l.kv_a = weights(rng, d.kv_lora + d.rope, d.hidden);
  l.kv_b = weights(rng, d.heads * (d.nope + d.v_head), d.kv_lora);
  l.o = weights(rng, d.hidden, d.heads * d.v_head);
  l.q_a_norm = norm(rng, d.q_lora);
  l.kv_a_norm = norm(rng, d.kv_lora);
  // The indexer's queries a little larger, so its scores spread and its Selections differ from position order.
  l.wq_b = weights(rng, d.index_heads * d.index_head_dim, d.q_lora);
  for (auto& v : l.wq_b) v = mx::round_bf16(4.0f * v);
  l.wk = weights(rng, d.index_head_dim, d.hidden);
  l.wproj = weights(rng, d.index_heads, d.hidden);
  l.k_norm_w = norm(rng, d.index_head_dim);
  l.k_norm_b = bf16_all(random_vector(rng, static_cast<size_t>(d.index_head_dim), 0.1f));
  l.gate = weights(rng, d.intermediate, d.hidden);
  l.up = weights(rng, d.intermediate, d.hidden);
  l.down = weights(rng, d.hidden, d.intermediate);
  return l;
}

struct Fixture {
  Config config = test_config();
  std::mt19937 rng{20261120};
  Layer full = random_layer(rng, config.dims());    // layer 0 (Full)
  Layer shared = random_layer(rng, config.dims());  // layer 1 (Shared)
  LayerWeights full_w = full.view(config.dims(), true), shared_w = shared.view(config.dims(), false);
  /** The streams entering the layers: FP32 values, as the residual stream is (A1). */
  std::vector<std::vector<float>> xs;
  Fixture() {
    for (int t = 0; t < kPositions; t++) xs.push_back(random_vector(rng, static_cast<size_t>(config.dims().hidden), 1.0f));
  }
  exact::AttentionLayer layer(int i) const {
    auto a = exact::attention_layer(config, i, i == 0 ? full_w : shared_w);
    REQUIRE(a.is_ok());
    return std::move(a).value();
  }
};

double rel_rms(std::span<const float> got, std::span<const float> want) {
  double e = 0.0, n = 0.0;
  for (size_t i = 0; i < got.size(); i++) {
    e += (static_cast<double>(got[i]) - want[i]) * (static_cast<double>(got[i]) - want[i]);
    n += static_cast<double>(want[i]) * want[i];
  }
  return std::sqrt(e / n);
}

std::vector<float> diff(std::span<const float> a, std::span<const float> b) {
  std::vector<float> out(a.size());
  for (size_t i = 0; i < a.size(); i++) out[i] = a[i] - b[i];
  return out;
}

bool bf16_values(std::span<const float> v) {
  return std::all_of(v.begin(), v.end(), [](float x) { return mx::round_bf16(x) == x; });
}

/** FNV-1a over the bytes of `v`, continuing from `h`. */
template <class T>
uint64_t fnv1a(uint64_t h, std::span<const T> v) {
  for (const std::byte b : std::as_bytes(v)) h = (h ^ std::to_integer<uint64_t>(b)) * 0x100000001b3ull;
  return h;
}

}  // namespace

TEST_CASE("rope: the reference's layout at position 0, the Semantics oracle's rotation within a BF16 rounding") {
  std::mt19937 rng(1);
  const auto freqs = exact::rope_freqs(64, 8e6f);
  const auto x = bf16_all(random_vector(rng, 64, 1.0f));
  const auto at0 = exact::rope(x, 0, freqs);
  for (size_t i = 0; i < 32; i++) {
    CHECK(at0[i] == x[2 * i]);
    CHECK(at0[32 + i] == x[2 * i + 1]);
  }
  for (const int64_t pos : {1, 77, 4097, 200000}) {
    const auto got = exact::rope(x, pos, freqs);
    const auto want = rope(x, pos, 8e6f);
    CHECK(bf16_values(got));
    for (size_t i = 0; i < got.size(); i++) CHECK(std::abs(got[i] - want[i]) <= std::abs(want[i]) * 0x1p-8f + 1e-6f);
  }
}

TEST_CASE("index_key_norm and index_scores: the Semantics oracle's within the configuration's roundings") {
  const Dims d = test_config().dims();
  std::mt19937 rng(2);
  const auto x = bf16_all(random_vector(rng, 128, 3.0f));
  const auto w = norm(rng, 128);
  const auto b = bf16_all(random_vector(rng, 128, 0.1f));
  const auto got = exact::index_key_norm(x, w, b);
  CHECK(bf16_values(got));
  CHECK(rel_rms(got, layer_norm(x, w, b, kIndexKeyNormEps)) < 4e-3);

  const auto q = bf16_all(random_vector(rng, static_cast<size_t>(d.index_heads * d.index_head_dim), 1.0f));
  const auto hw = random_vector(rng, static_cast<size_t>(d.index_heads), 0.5f);
  const auto keys = bf16_all(random_vector(rng, 40 * static_cast<size_t>(d.index_head_dim), 1.0f));
  const auto s = exact::index_scores(d, q, hw, keys);
  const auto ref = index_scores(d, q, hw, keys);
  REQUIRE(s.size() == 40);
  for (size_t t = 0; t < s.size(); t++) CHECK(std::abs(s[t] - ref[t]) <= 1e-4f * (1.0f + std::abs(ref[t])));
}

TEST_CASE("latent_attention: a softmax over the keys within BF16 roundings; the value is the key's first part") {
  std::mt19937 rng(3);
  for (const size_t n : {1u, 17u, 64u, 65u, 150u}) {
    const auto q = bf16_all(random_vector(rng, 576, 0.3f));
    const auto rows = bf16_all(random_vector(rng, n * 576, 1.0f));
    std::vector<std::span<const float>> keys;
    for (size_t j = 0; j < n; j++) keys.push_back(std::span<const float>(rows).subspan(j * 576, 576));
    const auto got = exact::latent_attention(q, keys, 512, 1.0f / 16.0f);
    REQUIRE(got.size() == 512);
    CHECK(bf16_values(got));
    std::vector<double> p(n);
    for (size_t j = 0; j < n; j++) {
      double s = 0.0;
      for (size_t c = 0; c < 576; c++) s += static_cast<double>(q[c]) * keys[j][c];
      p[j] = s / 16.0;
    }
    const double m = *std::max_element(p.begin(), p.end());
    double z = 0.0;
    for (auto& v : p) z += (v = std::exp(v - m));
    std::vector<float> want(512);
    for (size_t c = 0; c < 512; c++) {
      double acc = 0.0;
      for (size_t j = 0; j < n; j++) acc += p[j] / z * keys[j][c];
      want[c] = static_cast<float>(acc);
    }
    CHECK(rel_rms(got, want) < 1e-2);
  }
}

TEST_CASE("attend: the Semantics oracle's attention on the same weights, cache and Selection") {
  Fixture f;
  const Dims& d = f.config.dims();
  for (const int i : {0, 1}) {
    const auto a = f.layer(i);
    const LayerWeights& w = i == 0 ? f.full_w : f.shared_w;
    LayerCache cache;
    double worst_out = 0.0, worst_row = 0.0;
    size_t selected = 0, agreed = 0;
    for (int t = 0; t < kPositions; t++) {
      std::vector<int32_t> given;  // a Shared layer: every other position, and this one
      for (int32_t p = 0; p <= t; p++)
        if (p % 2 == 0 || p == t) given.push_back(p);
      auto got = exact::begin_layer(a, cache, f.xs[static_cast<size_t>(t)], i == 0 ? std::span<const int32_t>{} : given);
      auto want = begin_layer(f.config, w, i, cache, f.xs[static_cast<size_t>(t)], i == 0 ? std::span<const int32_t>{} : given);
      REQUIRE(got.is_ok());
      REQUIRE(want.is_ok());
      const auto& g = got.value();
      const auto& r = want.value();
      CHECK(bf16_values(g.ffn_in));
      CHECK(bf16_values(g.row.latent));
      CHECK(bf16_values(g.row.k_rope));
      CHECK(bf16_values(g.row.index_key));
      worst_row = std::max({worst_row, rel_rms(g.row.latent, r.row.latent), rel_rms(g.row.k_rope, r.row.k_rope)});
      if (i == 1 || t < d.index_topk) {  // the same positions attended: compare the attention's output
        CHECK(g.selection == r.selection);
        worst_out = std::max(worst_out, rel_rms(diff(g.mid, f.xs[static_cast<size_t>(t)]),
                                                diff(r.mid, f.xs[static_cast<size_t>(t)])));
      } else {  // the Selection itself: FP32 MMA scores against double ones, near-ties may swap
        selected += g.selection.size();
        std::vector<int32_t> both;
        std::set_intersection(g.selection.begin(), g.selection.end(), r.selection.begin(), r.selection.end(),
                              std::back_inserter(both));
        agreed += both.size();
      }
      append(cache, g.row);
    }
    MESSAGE("layer " << i << ": worst attention output " << worst_out << ", worst cache row " << worst_row);
    CHECK(worst_out < 3e-2);
    CHECK(worst_row < 1e-2);
    if (i == 0) {
      MESSAGE("Selections agree on " << agreed << " of " << selected << " positions");
      REQUIRE(selected > 0);
      CHECK(static_cast<double>(agreed) / selected > 0.95);
    }
  }
}

TEST_CASE("attend: GPU g's o_proj partial reads only its own heads, and the Exchange is bf16(GPU 0 + GPU 1)") {
  Fixture f;
  const Dims& d = f.config.dims();
  const auto h = bf16_all(random_vector(f.rng, static_cast<size_t>(d.hidden), 1.0f));
  const std::vector<int32_t> self = {0};
  const auto before = exact::attend(f.layer(1), LayerCache{}, h, self);
  // Redraw the value rows of the last head (GPU 1's): GPU 0's partial must not move.
  const int last = d.heads - 1, per_head = d.nope + d.v_head;
  auto& kv_b = f.shared.kv_b;
  for (int v = 0; v < d.v_head; v++)
    for (int c = 0; c < d.kv_lora; c++)
      kv_b[static_cast<size_t>((last * per_head + d.nope + v) * d.kv_lora + c)] = mx::round_bf16(0.03f * (v - c % 7));
  const auto after = exact::attend(f.layer(1), LayerCache{}, h, self);
  REQUIRE(before.is_ok());
  REQUIRE(after.is_ok());
  CHECK(before.value().o.partial[0] == after.value().o.partial[0]);
  CHECK(before.value().o.partial[1] != after.value().o.partial[1]);
  const auto& o = after.value().o;
  for (size_t r = 0; r < o.out.size(); r++) CHECK(o.out[r] == mx::round_bf16(o.partial[0][r] + o.partial[1][r]));
}

TEST_CASE("attend: a Full layer keeps every position up to index_topk, then the best, ties to the lower position") {
  Fixture f;
  const Dims& d = f.config.dims();
  const auto a = f.layer(0);
  LayerCache cache;
  // The same input at every position before the last: equal keys, so the earlier positions tie.
  const auto h = bf16_all(random_vector(f.rng, static_cast<size_t>(d.hidden), 1.0f));
  for (int t = 0; t < d.index_topk; t++) {
    auto step = exact::attend(a, cache, h, {});
    REQUIRE(step.is_ok());
    std::vector<int32_t> all(static_cast<size_t>(t) + 1);
    std::iota(all.begin(), all.end(), 0);
    CHECK(step.value().selection == all);
    append(cache, step.value().row);
  }
  // Index keys differ only by their RoPE position, so scores differ; the Selection is top_k of them.
  auto step = exact::attend(a, cache, h, {});
  REQUIRE(step.is_ok());
  CHECK(step.value().selection.size() == static_cast<size_t>(d.index_topk));
  CHECK(std::is_sorted(step.value().selection.begin(), step.value().selection.end()));
}

TEST_CASE("attend: a cache row depends only on its input and position") {
  Fixture f;
  const auto a = f.layer(1);
  LayerCache one, other;
  for (int t = 0; t < 3; t++) {
    append(one, exact::attend(a, one, bf16_all(f.xs[static_cast<size_t>(t)]), std::vector<int32_t>{t}).value().row);
    append(other, exact::attend(a, other, bf16_all(f.xs[static_cast<size_t>(t + 10)]), std::vector<int32_t>{t}).value().row);
  }
  const auto h = bf16_all(f.xs[50]);
  const std::vector<int32_t> sel = {0, 3};
  const auto x = exact::attend(a, one, h, sel), y = exact::attend(a, other, h, sel);
  CHECK(x.value().row.latent == y.value().row.latent);
  CHECK(x.value().row.k_rope == y.value().row.k_rope);
  CHECK(x.value().o.out != y.value().o.out);  // what it attends differs
}

TEST_CASE("attention_layer and attend refuse what the configuration cannot run") {
  Fixture f;
  CHECK_FALSE(exact::attention_layer(f.config, 0, f.shared_w).is_ok());  // a Full layer without its indexer
  CHECK_FALSE(exact::attention_layer(f.config, 1, f.full_w).is_ok());    // a Shared layer with one
  CHECK_FALSE(exact::attention_layer(f.config, 3, f.full_w).is_ok());
  auto narrow = f.shared_w;
  narrow.attention.o.cols -= 8;
  CHECK_FALSE(exact::attention_layer(f.config, 1, narrow).is_ok());

  const auto a = f.layer(1);
  const auto h = bf16_all(f.xs[0]);
  CHECK_FALSE(exact::attend(a, LayerCache{}, h, {}).is_ok());                              // no Selection
  CHECK_FALSE(exact::attend(a, LayerCache{}, h, std::vector<int32_t>{1}).is_ok());         // past the position
  CHECK_FALSE(exact::attend(a, LayerCache{}, std::span(h).first(8), std::vector<int32_t>{0}).is_ok());
  LayerCache one;
  append(one, exact::attend(a, one, h, std::vector<int32_t>{0}).value().row);
  CHECK_FALSE(exact::attend(a, one, h, std::vector<int32_t>{1, 0}).is_ok());               // not ascending
  CHECK_FALSE(exact::attend(a, one, h, std::vector<int32_t>{1, 1}).is_ok());
  auto torn = one;
  torn.latent.pop_back();
  CHECK_FALSE(exact::attend(a, torn, h, std::vector<int32_t>{0}).is_ok());
  CHECK_FALSE(exact::attend(f.layer(0), one, h, {}).is_ok());                              // no index keys
  CHECK(exact::attend(a, one, h, std::vector<int32_t>{0, 1}).is_ok());
}

TEST_CASE("dense_mlp and logits: the Semantics oracle's within the configuration's roundings; the Exchange") {
  Fixture f;
  const Dims& d = f.config.dims();
  const auto x = bf16_all(random_vector(f.rng, static_cast<size_t>(d.hidden), 1.0f));
  const auto& mlp = std::get<Mlp>(f.full_w.mlp);
  const auto got = exact::dense_mlp(mlp, x);
  REQUIRE(got.is_ok());
  const auto& e = got.value();
  for (size_t r = 0; r < e.out.size(); r++) CHECK(e.out[r] == mx::round_bf16(e.partial[0][r] + e.partial[1][r]));
  const auto gate = linear(mlp.gate, x), up = linear(mlp.up, x);
  std::vector<float> hid(gate.size());
  for (size_t i = 0; i < hid.size(); i++) hid[i] = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
  CHECK(rel_rms(e.out, linear(mlp.down, hid)) < 1e-2);
  CHECK_FALSE(exact::dense_mlp(mlp, std::span(x).first(16)).is_ok());

  const auto head = weights(f.rng, d.vocab, d.hidden);
  const auto fnorm = norm(f.rng, d.hidden);
  const auto stream = random_vector(f.rng, static_cast<size_t>(d.hidden), 2.0f);
  const Matrix hm{head, d.vocab, d.hidden};
  const auto l = exact::logits(d, fnorm, hm, stream);
  REQUIRE(l.is_ok());
  CHECK(rel_rms(l.value(), linear(hm, rms_norm(stream, fnorm, d.rms_eps))) < 1e-2);
  CHECK_FALSE(exact::logits(d, std::span(fnorm).first(8), hm, stream).is_ok());
}

// The Reference configuration is its bits: plane counts, the MMA's k16 order,
// the flash-decoding splits and their combine, the indexer's head order, each
// rounding. The tolerance tests above cannot see a change to any of those; this
// hash of a 100-position decode through a Full and a Shared layer, plus the
// dense MLP, the head and the indexer's scores, sees all of them. A deliberate change to the
// configuration changes this constant and the design document with it.
constexpr uint64_t kAttentionBits = 0xbb8cd9fe99be6afbull;

TEST_CASE("the attention side's Reference bits") {
  Fixture f;
  const Dims& d = f.config.dims();
  const auto full = f.layer(0), shared = f.layer(1);
  LayerCache c0, c1;
  uint64_t h = 0xcbf29ce484222325ull;
  for (int t = 0; t < kPositions; t++) {
    const auto x = f.xs[static_cast<size_t>(t)];
    auto s0 = exact::begin_layer(full, c0, x, {});
    REQUIRE(s0.is_ok());
    auto a1 = exact::attend(shared, c1, bf16_all(s0.value().ffn_in), s0.value().selection);
    REQUIRE(a1.is_ok());
    for (const auto* v : {&s0.value().mid, &s0.value().ffn_in, &s0.value().row.latent, &s0.value().row.k_rope,
                          &s0.value().row.index_key, &a1.value().heads, &a1.value().o.partial[0],
                          &a1.value().o.partial[1], &a1.value().o.out})
      h = fnv1a(h, std::span<const float>(*v));
    h = fnv1a(h, std::span<const int32_t>(s0.value().selection));
    append(c0, s0.value().row);
    append(c1, a1.value().row);
  }
  const auto e = exact::dense_mlp(std::get<Mlp>(f.full_w.mlp), bf16_all(f.xs[7]));
  REQUIRE(e.is_ok());
  for (const auto* v : {&e.value().partial[0], &e.value().partial[1], &e.value().out}) h = fnv1a(h, std::span<const float>(*v));
  const auto head = weights(f.rng, d.vocab, d.hidden);
  const auto l = exact::logits(d, f.full.in_norm, Matrix{head, d.vocab, d.hidden}, f.xs[9]);
  REQUIRE(l.is_ok());
  h = fnv1a(h, std::span<const float>(l.value()));
  // The indexer's FP32 scores themselves: a Selection shows a changed score only at a near-tie.
  const auto ihd = static_cast<size_t>(d.index_head_dim), heads = static_cast<size_t>(d.index_heads);
  const auto iq = bf16_all(random_vector(f.rng, heads * ihd, 1.0f));
  const auto iw = random_vector(f.rng, heads, 0.5f);
  const auto ik = bf16_all(random_vector(f.rng, 64 * ihd, 1.0f));
  h = fnv1a(h, std::span<const float>(exact::index_scores(d, iq, iw, ik)));
  // The key norm over many keys: its BF16 output shows a one-ulp change of the variance (an FMA
  // or not) in about one value in 2^16, so 12,800 decoded key values are too few to see it.
  const auto kw = norm(f.rng, d.index_head_dim);
  const auto kb = bf16_all(random_vector(f.rng, ihd, 0.1f));
  for (int k = 0; k < 4096; k++) {
    const auto key = bf16_all(random_vector(f.rng, ihd, 0.5f + static_cast<float>(k % 13)));
    h = fnv1a(h, std::span<const float>(exact::index_key_norm(key, kw, kb)));
  }
  MESSAGE("attention bits: " << h);
  CHECK(h == kAttentionBits);
}

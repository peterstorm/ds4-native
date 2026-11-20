// glm53/exact_moe: GLM-5.3's MoE in the Reference configuration. The module is
// composed from DS4.1's hardware-proven definitions, so these tests do not
// restate it; they check it against independent witnesses: the checkpoint's
// values (dequantize_mxfp4), the Semantics oracle's FP32 mathematics at E4M3
// activations (within the Reference configuration's BF16 roundings), and the
// structure the GPUs rely on (a Half reads only its own rows; the order of the
// combine).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "core/glm53/exact_moe.hpp"
#include "core/reduce.hpp"
#include "exact_fixtures.hpp"

using namespace ds4n;
using namespace ds4n::glm53;
using namespace ds4n::glm53::test;

namespace {

/**
 * The design document's Expert written independently: double dot products on
 * the checkpoint's values, the whole Expert at once (no Halves, no split-K
 * planes, no MMA model), with the Reference configuration's roundings — gate
 * and up to BF16, bf16(weight * silu(gate) * up), E4M3 into down.
 */
std::vector<float> spec_expert(const Mlp& m, std::span<const float> x, float weight) {
  const auto in = e4m3_round_trip(x);
  const auto g = linear(m.gate, in), u = linear(m.up, in);
  std::vector<float> a(g.size());
  for (size_t i = 0; i < a.size(); i++) {
    const float gb = mx::round_bf16(g[i]), ub = mx::round_bf16(u[i]);
    a[i] = mx::round_bf16(weight * (gb / (1.0f + std::exp(-gb)) * ub));
  }
  return linear(m.down, e4m3_round_trip(a));
}

std::vector<float> scaled(std::vector<float> v, float s) {
  for (auto& x : v) x *= s;
  return v;
}

/** exact_moe against spec_expert: only accumulation order differs (MMA truncation vs double). */
constexpr double kSpecTolerance = 1e-3;

/**
 * Two independent E4M3 roundings of the same values: E4M3 keeps 3 mantissa
 * bits, about 2.6% RMS relative error each, so two differ by about 3.7%.
 */
constexpr double kTwoE4m3Roundings = 8e-2;

/**
 * FNV-1a of moe_token's outputs on RandomLayer(29), hidden 256, weight scales
 * 2^-22 .. 2^8: the Reference configuration's bits (see that test).
 */
constexpr uint64_t kReferenceBits = 0x20922a80e99434d8ull;

}  // namespace

TEST_CASE("router_planes: equal planes of at most 1024 values, each a multiple of 8") {
  CHECK(exact::router_planes(6144) == 6);  // GLM-5.3
  CHECK(exact::router_planes(5120) == 5);  // DS4.1's router plane width
  CHECK(exact::router_planes(2048) == 2);
  CHECK(exact::router_planes(1024) == 1);
  CHECK(exact::router_planes(64) == 1);
  CHECK_FALSE(exact::router_planes(1028).has_value());  // 2 planes of 514
  CHECK_FALSE(exact::router_planes(12).has_value());
  CHECK_FALSE(exact::router_planes(0).has_value());
}

// The codes are the engine's input, not their values (the MMA aligns products
// by the operands' exponents), so the conversion must keep every code and
// scale: each weight read back from the GGML blocks is the checkpoint's value.
TEST_CASE("fp4_expert keeps the checkpoint's codes: every weight reads back as dequantize_mxfp4's value") {
  for (unsigned seed = 0; seed < 8; seed++) {
    std::mt19937 rng(seed);
    const Packed p = random_packed(rng, 96, 128, 100, 150);
    const auto e = p.expert();
    const auto want = dequantized(p);
    auto read_back = [](std::span<const std::byte> row, size_t k) {
      const auto block = row.subspan(k / 32 * mx::kGgmlBlockBytes).first<mx::kGgmlBlockBytes>();
      return mx::ggml_block_element(block, k % 32) * mx::e8m0_value(std::to_integer<uint8_t>(block[0]));
    };
    bool same = true;
    for (size_t r = 0; r < 128; r++)
      for (size_t k = 0; k < 96; k++) {
        same &= read_back(e.gate_row(r), k) == want.gate[r * 96 + k];
        same &= read_back(e.up_row(r), k) == want.up[r * 96 + k];
      }
    for (size_t r = 0; r < 96; r++)
      for (size_t k = 0; k < 128; k++) same &= read_back(e.down_row(r), k) == want.down[r * 128 + k];
    CHECK(same);
  }
}

TEST_CASE("fp4_expert refuses widths the Halves cannot split and tensors of the wrong size") {
  std::mt19937 rng(1);
  const Packed p = random_packed(rng, 64, 64);
  CHECK(exact::fp4_expert(p.view(), 64, 64).is_ok());
  CHECK_FALSE(exact::fp4_expert(p.view(), 48, 64).is_ok());   // hidden not whole 32-blocks
  CHECK_FALSE(exact::fp4_expert(p.view(), 64, 32).is_ok());   // a Half's down K would be 16 values
  CHECK_FALSE(exact::fp4_expert(p.view(), 64, 128).is_ok());  // the tensors are for 64
  Packed short_up = p;
  short_up.up_codes.pop_back();
  const auto bad = exact::fp4_expert(short_up.view(), 64, 64);
  REQUIRE_FALSE(bad.is_ok());
  CHECK(bad.error().what.find("up_proj") != std::string::npos);
}

TEST_CASE("expert_input is the E4M3 quantization of the BF16 input, so BF16 rounding first changes nothing") {
  std::mt19937 rng(2);
  const auto x = random_vector(rng, 256, 3.0f);
  const auto in = exact::expert_input(x);
  REQUIRE(in.is_ok());
  CHECK(in.value().width() == 256);
  const auto want = mx::act_quant(bf16_all(x));
  REQUIRE(in.value().blocks().size() == want.size());
  for (size_t b = 0; b < want.size(); b++) {
    CHECK(in.value().blocks()[b].scale == want[b].scale);
    CHECK(in.value().blocks()[b].codes == want[b].codes);
  }
  const auto again = exact::expert_input(bf16_all(x));  // idempotent under the BF16 rounding
  for (size_t b = 0; b < want.size(); b++) CHECK(again.value().blocks()[b].codes == want[b].codes);
  CHECK_FALSE(exact::expert_input(std::vector<float>(33, 1.0f)).is_ok());
  CHECK_FALSE(exact::expert_input(std::vector<float>{}).is_ok());
}

TEST_CASE("router_logits: the fixed-plane BF16 GEMV of the BF16 input, within FP32 noise of the exact dot") {
  std::mt19937 rng(3);
  const int hidden = 2048, experts = 8;
  const auto w = bf16_all(random_vector(rng, static_cast<size_t>(hidden * experts), 0.05f));
  const Matrix router{w, experts, hidden};
  const auto x = random_vector(rng, hidden, 1.0f);
  const auto logits = exact::router_logits(router, x);
  REQUIRE(logits.is_ok());
  const auto xb = bf16_all(x);
  for (size_t e = 0; e < static_cast<size_t>(experts); e++) {
    const auto row = std::span<const float>(w).subspan(e * hidden, hidden);
    // The Reference configuration's definition: two planes of 1024, in order.
    CHECK(logits.value()[e] == reduce::bf16_gemv(row, xb, 2));
    double exact = 0.0, magnitude = 0.0;
    for (size_t k = 0; k < static_cast<size_t>(hidden); k++) {
      exact += static_cast<double>(row[k]) * xb[k];
      magnitude += std::fabs(static_cast<double>(row[k]) * xb[k]);
    }
    CHECK(std::fabs(logits.value()[e] - exact) <= 1e-6 * magnitude);
  }
  // Bits below the input's BF16 precision never reach the router.
  CHECK(exact::router_logits(router, xb).value() == logits.value());

  CHECK_FALSE(exact::router_logits(router, std::vector<float>(100, 1.0f)).is_ok());
  const std::vector<float> odd(1028 * 2, 0.0f);
  CHECK_FALSE(exact::router_logits(Matrix{odd, 2, 1028}, std::vector<float>(1028, 1.0f)).is_ok());
}

TEST_CASE("route agrees with the Semantics oracle's routing wherever the margin clears FP32 noise") {
  Dims d{};
  d.experts = 256;
  d.experts_per_token = 8;
  d.routed_scale = 2.5f;
  int compared = 0;
  for (unsigned seed = 0; seed < 300; seed++) {
    std::mt19937 rng(seed);
    const auto logits = random_vector(rng, 256, 2.0f);
    const auto bias = random_vector(rng, 256, 0.02f);
    const auto r = exact::route(d, logits, bias);
    REQUIRE(r.is_ok());
    const Route& got = r.value();
    REQUIRE(got.expert.size() == 8);
    CHECK(std::is_sorted(got.expert.begin(), got.expert.end()));
    CHECK(std::adjacent_find(got.expert.begin(), got.expert.end()) == got.expert.end());
    CHECK(got.margin >= 0.0f);
    double total = 0.0;
    for (float w : got.weight) total += w;
    CHECK(total == doctest::Approx(2.5).epsilon(1e-6));

    const Route want = glm53::route(d, logits, bias);
    if (want.margin <= 1e-5f) continue;
    compared++;
    CHECK(got.expert == want.expert);
    for (size_t k = 0; k < 8; k++) CHECK(got.weight[k] == doctest::Approx(want.weight[k]).epsilon(1e-6));
  }
  CHECK(compared > 250);
}

TEST_CASE("route: equal biased scores go to the lower Expert id, and a tie has margin 0") {
  Dims d{};
  d.experts = 16;
  d.experts_per_token = 4;
  d.routed_scale = 2.5f;
  const std::vector<float> logits(16, 0.3f), bias(16, 0.0f);
  const auto r = exact::route(d, logits, bias).value();
  CHECK(r.expert == std::vector<int32_t>{0, 1, 2, 3});
  CHECK(r.margin == 0.0f);
  for (float w : r.weight) CHECK(w == doctest::Approx(2.5 / 4).epsilon(1e-6));
  CHECK_FALSE(exact::route(d, std::vector<float>(15, 0.0f), bias).is_ok());
  d.experts_per_token = 17;
  CHECK_FALSE(exact::route(d, logits, bias).is_ok());
}

TEST_CASE("route with a resident bonus: never fewer resident Experts, weights still the chosen sigmoids'") {
  Dims d{};
  d.experts = 256;
  d.experts_per_token = 8;
  d.routed_scale = 2.5f;
  int changed = 0;
  for (unsigned seed = 0; seed < 300; seed++) {
    std::mt19937 rng(seed);
    const auto logits = random_vector(rng, 256, 2.0f);
    const auto bias = random_vector(rng, 256, 0.02f);
    std::vector<uint8_t> resident(256);
    for (auto& r : resident) r = static_cast<uint8_t>(rng() % 3 == 0);  // about a third resident
    const float bonus = std::uniform_real_distribution<float>(1e-4f, 0.2f)(rng);
    const Route plain = exact::route(d, logits, bias).value();
    const Route cc = exact::route(d, logits, bias, exact::ResidentBonus{bonus, resident}).value();
    auto resident_in = [&](const Route& r) {
      return std::ranges::count_if(r.expert, [&](int32_t e) { return resident[static_cast<size_t>(e)] != 0; });
    };
    CHECK(resident_in(cc) >= resident_in(plain));
    CHECK(std::is_sorted(cc.expert.begin(), cc.expert.end()));
    double total = 0.0;
    for (float w : cc.weight) total += w;
    CHECK(total == doctest::Approx(2.5).epsilon(1e-6));
    // Only the choice changes: on the same choice the weights are bit-identical.
    if (cc.expert == plain.expert)
      CHECK(cc.weight == plain.weight);
    else
      changed++;
  }
  CHECK(changed > 0);
}

TEST_CASE("route: an empty resident set changes nothing; a dominant bonus chooses only resident Experts") {
  Dims d{};
  d.experts = 256;
  d.experts_per_token = 8;
  d.routed_scale = 2.5f;
  std::mt19937 rng(7);
  const auto logits = random_vector(rng, 256, 2.0f);
  const auto bias = random_vector(rng, 256, 0.02f);
  const Route plain = exact::route(d, logits, bias).value();
  const std::vector<uint8_t> none(256, 0);
  const Route same = exact::route(d, logits, bias, exact::ResidentBonus{0.5f, none}).value();
  CHECK(same.expert == plain.expert);
  CHECK(same.weight == plain.weight);
  CHECK(same.margin == plain.margin);
  // Sigmoid + bias spans less than 1.1, so a bonus of 10 puts every resident Expert first.
  std::vector<uint8_t> twelve(256, 0);
  for (int e = 100; e < 112; e++) twelve[static_cast<size_t>(e)] = 1;
  const Route forced = exact::route(d, logits, bias, exact::ResidentBonus{10.0f, twelve}).value();
  for (int32_t e : forced.expert) CHECK(twelve[static_cast<size_t>(e)] == 1);
  CHECK_FALSE(exact::route(d, logits, bias, exact::ResidentBonus{0.1f, std::vector<uint8_t>(255, 0)}).is_ok());
  CHECK_FALSE(exact::route(d, logits, bias, exact::ResidentBonus{0.0f, none}).is_ok());
  CHECK_FALSE(exact::route(d, logits, bias, exact::ResidentBonus{-0.1f, none}).is_ok());
  CHECK_FALSE(exact::route(d, logits, bias, exact::ResidentBonus{std::nanf(""), none}).is_ok());
}

TEST_CASE("expert_halves: the two Halves add up to the Expert's output times its routing weight") {
  for (unsigned seed = 0; seed < 6; seed++) {
    std::mt19937 rng(100 + seed);
    const Packed p = random_packed(rng, 64, 128);
    const auto e = p.expert();
    const auto deq = dequantized(p);
    const auto x = bf16_all(random_vector(rng, 64, 1.0f));
    const float weight = 0.37f;
    const auto halves = exact::expert_halves(e, exact::expert_input(x).value(), weight);
    REQUIRE(halves.is_ok());
    const auto got = sum_halves(halves.value());
    const double spec = relative_error(got, spec_expert(deq.mlp(64, 128), x, weight));
    const double semantic = relative_error(got, semantic_expert(deq.mlp(64, 128), x, weight));
    CAPTURE(spec);
    CAPTURE(semantic);
    CHECK(spec < kSpecTolerance);
    CHECK(semantic < kTwoE4m3Roundings);
  }
}

// Decision 4: the weight goes in before down_proj (as DS4.1), where the pinned
// reference multiplies after. Down is linear, so only the roundings differ:
// down's input is E4M3-quantized after the weight instead of before, two
// different roundings of the same mathematics, and no further apart than that.
TEST_CASE("folding the weight before down_proj differs from weighting after only by down's E4M3 rounding") {
  double worst = 0.0;
  for (unsigned seed = 0; seed < 6; seed++) {
    std::mt19937 rng(200 + seed);
    const Packed p = random_packed(rng, 64, 128);
    const auto e = p.expert();
    const auto in = exact::expert_input(bf16_all(random_vector(rng, 64, 1.0f))).value();
    const auto folded = sum_halves(exact::expert_halves(e, in, 0.37f).value());
    const auto after = scaled(sum_halves(exact::expert_halves(e, in, 1.0f).value()), 0.37f);
    worst = std::max(worst, relative_error(folded, after));
  }
  CHECK(worst < kTwoE4m3Roundings);
  CHECK(worst > 0.0);  // not the same bits: the Reference configuration is not the reference's order
}

// DS4.1 clamps gate at +10 and up at +-10; GLM has no limit. With gate
// outputs far past 10 a clamp would shrink the output many-fold.
TEST_CASE("expert_halves does not clamp: large gate outputs pass through SwiGLU unchanged") {
  std::mt19937 rng(7);
  const Packed p = random_packed(rng, 64, 64, 126, 128);  // weights around 2.3: gate outputs in the tens
  const auto e = p.expert();
  const auto deq = dequantized(p);
  const auto x = bf16_all(random_vector(rng, 64, 1.0f));
  const auto gate = linear(deq.mlp(64, 64).gate, e4m3_round_trip(x));
  REQUIRE(*std::max_element(gate.begin(), gate.end()) > 30.0f);
  const auto got = sum_halves(exact::expert_halves(e, exact::expert_input(x).value(), 1.0f).value());
  CHECK(relative_error(got, spec_expert(deq.mlp(64, 64), x, 1.0f)) < kSpecTolerance);
}

// Tensor parallelism rests on this: GPU h holds only its Half's rows of gate
// and up and its K slice of down.
TEST_CASE("expert_halves: a Half reads only its own gate/up rows and its own K blocks of down") {
  std::mt19937 rng(8);
  const Packed p = random_packed(rng, 64, 128);
  const auto x = exact::expert_input(bf16_all(random_vector(rng, 64, 1.0f))).value();
  const auto before = exact::expert_halves(p.expert(), x, 1.0f).value();

  Packed changed = p;  // re-draw everything Half 1 owns: gate/up rows 64..127, down K 64..127
  const size_t row_bytes = 64 / 2, half_rows = 64;
  for (size_t r = half_rows; r < 128; r++)
    for (size_t b = 0; b < row_bytes; b++) {
      changed.gate_codes[r * row_bytes + b] = static_cast<std::byte>(rng());
      changed.up_codes[r * row_bytes + b] = static_cast<std::byte>(rng());
    }
  for (size_t r = 0; r < 64; r++)
    for (size_t b = 32; b < 64; b++) changed.down_codes[r * 64 + b] = static_cast<std::byte>(rng());
  const auto after = exact::expert_halves(changed.expert(), x, 1.0f).value();
  CHECK(after[0] == before[0]);
  CHECK(after[1] != before[1]);
}

TEST_CASE("expert_halves and half_hidden refuse an input of another width, half_hidden a third Half") {
  std::mt19937 rng(9);
  const auto e = random_packed(rng, 64, 64).expert();
  const auto wide = exact::expert_input(std::vector<float>(96, 1.0f)).value();
  CHECK_FALSE(exact::expert_halves(e, wide, 1.0f).is_ok());
  CHECK_FALSE(exact::half_hidden(e, 0, wide, 1.0f).is_ok());
  const auto x = exact::expert_input(std::vector<float>(64, 1.0f)).value();
  CHECK(exact::half_hidden(e, 1, x, 1.0f).value().size() == 32);  // a Half's rows: I/2
  CHECK_FALSE(exact::half_hidden(e, 2, x, 1.0f).is_ok());
}

// FP32 addition is not associative: 1e8 + 1 rounds back to 1e8. The combine's
// order is part of the Reference configuration: per GPU from zero, routed
// Halves in the given (ascending id) order, then the shared Half. Any other
// order of GPU 0's terms here gives 1, not 2.
TEST_CASE("combine adds per GPU from zero in Route order, then the shared Half, and rounds the exchange to BF16") {
  const std::vector<exact::Halves> routed = {{std::vector<float>{1e8f}, std::vector<float>{0.0f}},
                                             {std::vector<float>{-1e8f}, std::vector<float>{0.25f}},
                                             {std::vector<float>{1.0f}, std::vector<float>{0.5f}}};
  const exact::Halves shared = {std::vector<float>{1.0f}, std::vector<float>{1.0f / 3.0f}};
  const auto out = exact::combine(routed, shared);
  REQUIRE(out.is_ok());
  CHECK(out.value().partial[0] == std::vector<float>{2.0f});  // (((0 + 1e8) - 1e8) + 1) + 1; reversed or shared first: 1
  CHECK(out.value().partial[1] == std::vector<float>{0.75f + 1.0f / 3.0f});
  CHECK(out.value().out == std::vector<float>{mx::round_bf16(2.0f + (0.75f + 1.0f / 3.0f))});

  const auto no_routed = exact::combine({}, shared);  // a token routed nowhere still gets the shared expert
  CHECK(no_routed.value().out == std::vector<float>{mx::round_bf16(1.0f + 1.0f / 3.0f)});
  const std::vector<exact::Halves> wide = {{std::vector<float>{1.0f, 2.0f}, std::vector<float>{1.0f, 2.0f}}};
  CHECK_FALSE(exact::combine(wide, shared).is_ok());
  CHECK_FALSE(exact::combine({}, exact::Halves{std::vector<float>{1.0f}, std::vector<float>{1.0f, 2.0f}}).is_ok());
}

namespace {

/** FNV-1a over the bytes of `v`, continuing from `h`. */
template <class T>
uint64_t fnv1a(uint64_t h, std::span<const T> v) {
  for (const std::byte b : std::as_bytes(v)) h = (h ^ std::to_integer<uint64_t>(b)) * 0x100000001b3ull;
  return h;
}

}  // namespace

// The Reference configuration is its bits: the split-K planes, the router's
// planes, the order of every sum, each rounding. The tolerance tests above
// cannot see a change to any of those; this one sees all of them. A deliberate
// change to the Reference configuration changes this hash, and the design
// document with it (docs/design/glm53-reference-configuration.md). Hidden is
// 256 so both of gate/up's split-K planes hold blocks (4 of 8 each). Some
// choices move a value by one FP32 ulp before a BF16 rounding hides it (the
// split-K planes) or reorder a short sum (the routing weights' denominator),
// so they show only now and then: 512 tokens make ~330K BF16 roundings and
// 512 weight sums, enough that each such mutant changes some bit
// (tools/glm53-golden/exact-moe-mutation-probe.sh). The split-K planes show in
// the hidden fragments (down's E4M3 input hides nearly all of them from the
// output), and only when a sum truncates: E2M1 x E4M3 products carry at most 6
// significant bits, so blocks of nearby scales add exactly in FP32 in any
// order. Weight scales spread over 2^-22 .. 2^8, as a real Expert's blocks
// differ, make the MMA's 25-bit window cut.
TEST_CASE("moe_token's bits are the Reference configuration's: routes, weights, hidden fragments, partials, output") {
  Dims d = moe_dims();
  d.hidden = 256;
  RandomLayer l(29, d, 105, 135);
  uint64_t h = 0xcbf29ce484222325ull;
  for (int t = 0; t < 512; t++) {
    const auto x = bf16_all(random_vector(l.rng, static_cast<size_t>(l.d.hidden), 1.0f));
    const auto step = exact::moe_token(l.d, l.layer, l.experts, x);
    REQUIRE(step.is_ok());
    const auto& s = step.value();
    const auto in = exact::expert_input(x).value();
    for (size_t k = 0; k < s.route.expert.size(); k++)  // the hidden fragments, where the split-K planes show
      for (size_t half = 0; half < exact::kHalves; half++)
        h = fnv1a(h, std::span<const float>(
                         exact::half_hidden(l.experts[static_cast<size_t>(s.route.expert[k])], half, in, s.route.weight[k])
                             .value()));
    h = fnv1a(h, std::span<const int32_t>(s.route.expert));
    h = fnv1a(h, std::span<const float>(s.route.weight));
    h = fnv1a(h, std::span<const float>(s.output.partial[0]));
    h = fnv1a(h, std::span<const float>(s.output.partial[1]));
    h = fnv1a(h, std::span<const float>(s.output.out));
  }
  CHECK_MESSAGE(h == kReferenceBits, "Reference configuration bits (decimal): " << h);
}

TEST_CASE("moe_token: the specification's MoE, and the Semantics oracle's at E4M3 up to the roundings between them") {
  RandomLayer l(11);
  const Dims& d = l.d;
  int routes_compared = 0;
  for (int t = 0; t < 24; t++) {
    const auto x = bf16_all(random_vector(l.rng, static_cast<size_t>(d.hidden), 1.0f));
    const auto step = exact::moe_token(d, l.layer, l.experts, x);
    REQUIRE(step.is_ok());
    const Route want = glm53::route(d, linear(l.layer.router, x), l.bias);
    if (want.margin > 1e-5f) {
      routes_compared++;
      CHECK(step.value().route.expert == want.expert);
    }
    // Both witnesses' MoE on the exact Route's Experts and weights, summed in double.
    const Route& r = step.value().route;
    auto moe_of = [&](auto expert_fn) {
      std::vector<double> acc(static_cast<size_t>(d.hidden), 0.0);
      for (size_t k = 0; k < r.expert.size(); k++) {
        const auto y = expert_fn(l.deq[static_cast<size_t>(r.expert[k])].mlp(d.hidden, d.moe_intermediate), r.weight[k]);
        for (size_t i = 0; i < acc.size(); i++) acc[i] += static_cast<double>(y[i]);
      }
      const auto s = expert_fn(l.shared_deq.mlp(d.hidden, d.shared_intermediate()), 1.0f);
      std::vector<float> out(acc.size());
      for (size_t i = 0; i < out.size(); i++) out[i] = static_cast<float>(acc[i] + s[i]);
      return out;
    };
    const auto spec = bf16_all(moe_of([&](const Mlp& m, float w) { return spec_expert(m, x, w); }));  // the exchange
    const auto semantic = moe_of([&](const Mlp& m, float w) { return semantic_expert(m, x, w); });
    CHECK(relative_error(step.value().output.out, spec) < kSpecTolerance);
    CHECK(relative_error(step.value().output.out, semantic) < kTwoE4m3Roundings);
  }
  CHECK(routes_compared > 20);
  CHECK_FALSE(exact::moe_token(d, l.layer, std::span(l.experts).first(15), std::vector<float>(64, 1.0f)).is_ok());
}

// The Expert-major form (ds4n-glm53-oracle-run, the GPU's Grouped MoE): begin
// every token, run Experts in any grouping, finish each token.
TEST_CASE("begin_moe and finish_moe around expert_halves are moe_token, bit for bit; finish_moe wants one per slot") {
  RandomLayer l(13);
  for (int t = 0; t < 8; t++) {
    const auto x = bf16_all(random_vector(l.rng, static_cast<size_t>(l.d.hidden), 1.0f));
    const auto start = exact::begin_moe(l.d, l.layer, x);
    REQUIRE(start.is_ok());
    const auto& s = start.value();
    std::vector<exact::Halves> routed;
    for (size_t k = 0; k < s.route.expert.size(); k++)
      routed.push_back(exact::expert_halves(l.experts[static_cast<size_t>(s.route.expert[k])], s.input, s.route.weight[k]).value());
    const auto got = exact::finish_moe(l.layer, s, routed);
    REQUIRE(got.is_ok());
    const auto want = exact::moe_token(l.d, l.layer, l.experts, x).value();
    CHECK(s.route.expert == want.route.expert);
    CHECK(s.route.weight == want.route.weight);
    CHECK(got.value().out == want.output.out);
    routed.pop_back();
    CHECK_FALSE(exact::finish_moe(l.layer, s, routed).is_ok());
  }
}

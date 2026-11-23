#pragma once
// Fixtures for the Reference-configuration MoE tests (test_exact_moe,
// test_moe_roundings): random MXFP4 Experts as checkpoint tensors, their FP32
// values, a random Sparse layer, and the Semantics oracle's Expert at E4M3
// written independently of exact_moe.
#include <doctest/doctest.h>

#include <cmath>
#include <random>
#include <vector>

#include "core/glm53/dequant.hpp"
#include "core/glm53/exact_moe.hpp"
#include "core/glm53/precision.hpp"
#include "core/glm53/reference.hpp"
#include "core/mx_reference.hpp"

namespace ds4n::glm53::test {

inline std::vector<float> random_vector(std::mt19937& rng, size_t n, float sd) {
  std::normal_distribution<float> d(0.0f, sd);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

inline std::vector<float> bf16_all(std::vector<float> v) {
  for (auto& x : v) x = mx::round_bf16(x);
  return v;
}

inline std::vector<std::byte> random_codes(std::mt19937& rng, size_t n) {
  std::vector<std::byte> v(n);
  for (auto& b : v) b = static_cast<std::byte>(rng());
  return v;
}

inline std::vector<uint8_t> random_scales(std::mt19937& rng, size_t n, int lo, int hi) {
  std::vector<uint8_t> v(n);
  for (auto& b : v) b = static_cast<uint8_t>(std::uniform_int_distribution<int>(lo, hi)(rng));
  return v;
}

/** An Expert's checkpoint tensors (packed E2M1 codes, E8M0 scales), owned. */
struct Packed {
  int hidden, inter;
  std::vector<std::byte> gate_codes, up_codes, down_codes;
  std::vector<uint8_t> gate_scales, up_scales, down_scales;

  Sextuplet view() const { return {gate_codes, up_codes, down_codes, gate_scales, up_scales, down_scales}; }
  exact::Fp4Expert expert() const {
    auto e = exact::fp4_expert(view(), hidden, inter);
    REQUIRE(e.is_ok());
    return std::move(e).value();
  }
};

/** Random codes; scales in [lo, hi] (127 is 2^0), so weights are about 2.3 * 2^(scale - 127). */
inline Packed random_packed(std::mt19937& rng, int hidden, int inter, int lo = 119, int hi = 122) {
  const auto h = static_cast<size_t>(hidden), i = static_cast<size_t>(inter);
  return Packed{hidden,
                inter,
                random_codes(rng, i * h / 2),
                random_codes(rng, i * h / 2),
                random_codes(rng, h * i / 2),
                random_scales(rng, i * h / 32, lo, hi),
                random_scales(rng, i * h / 32, lo, hi),
                random_scales(rng, h * i / 32, lo, hi)};
}

inline std::span<const std::byte> bytes_of(const std::vector<uint8_t>& v) { return std::as_bytes(std::span(v)); }

/** The checkpoint's values in FP32, owned, with an Mlp view. */
struct Dequantized {
  std::vector<float> gate, up, down;
  Mlp mlp(int hidden, int inter) const {
    return Mlp{Matrix{gate, inter, hidden}, Matrix{up, inter, hidden}, Matrix{down, hidden, inter}};
  }
};

inline Dequantized dequantized(const Packed& p) {
  const auto h = static_cast<uint64_t>(p.hidden), i = static_cast<uint64_t>(p.inter);
  return Dequantized{dequantize_mxfp4(p.gate_codes, bytes_of(p.gate_scales), i, h).value(),
                     dequantize_mxfp4(p.up_codes, bytes_of(p.up_scales), i, h).value(),
                     dequantize_mxfp4(p.down_codes, bytes_of(p.down_scales), h, i).value()};
}

/**
 * The Semantics oracle's Expert at E4M3 activations, in FP32 with double dot
 * products (no BF16 roundings), with the routing weight folded in before
 * down_proj's E4M3 quantization as the Reference configuration folds it
 * (design, decision 4). Weighting after down instead rounds every down input
 * differently.
 */
inline std::vector<float> semantic_expert(const Mlp& m, std::span<const float> x, float weight = 1.0f) {
  const auto in = e4m3_round_trip(x);
  const auto g = linear(m.gate, in), u = linear(m.up, in);
  std::vector<float> a(g.size());
  for (size_t i = 0; i < a.size(); i++)
    a[i] = weight * (static_cast<float>(g[i] / (1.0 + std::exp(-static_cast<double>(g[i])))) * u[i]);
  return linear(m.down, e4m3_round_trip(a));
}

inline std::vector<float> sum_halves(const exact::Halves& h) {
  std::vector<float> out(h[0].size());
  for (size_t i = 0; i < out.size(); i++) out[i] = h[0][i] + h[1][i];
  return out;
}

inline Dims moe_dims() {
  Dims d{};
  d.hidden = 64;
  d.moe_intermediate = 64;
  d.experts = 16;
  d.experts_per_token = 4;
  d.shared_experts = 1;
  d.routed_scale = 2.5f;
  return d;
}

/** A random Sparse layer: its Experts' codes and their FP32 values, router and bias. */
struct RandomLayer {
  Dims d;
  std::mt19937 rng;
  int lo, hi;
  std::vector<Dequantized> deq;  // before `experts`: make_experts fills it
  std::vector<exact::Fp4Expert> experts;
  Packed shared_packed;
  exact::Fp4Expert shared;
  Dequantized shared_deq;
  std::vector<float> router_w, bias;
  exact::MoeLayer layer;

  /** Weight scales in [scale_lo, scale_hi] (E8M0; 127 is 2^0). */
  explicit RandomLayer(unsigned seed, Dims dims = moe_dims(), int scale_lo = 119, int scale_hi = 122)
      : d(dims),
        rng(seed),
        lo(scale_lo),
        hi(scale_hi),
        experts(make_experts()),
        shared_packed(random_packed(rng, d.hidden, d.shared_intermediate(), lo, hi)),
        shared(shared_packed.expert()),
        shared_deq(dequantized(shared_packed)),
        router_w(bf16_all(random_vector(rng, static_cast<size_t>(d.experts * d.hidden), 0.2f))),
        bias(random_vector(rng, static_cast<size_t>(d.experts), 0.01f)),
        layer{Matrix{router_w, d.experts, d.hidden}, bias, shared} {}
  RandomLayer(const RandomLayer&) = delete;  // `layer` views the members

 private:
  std::vector<exact::Fp4Expert> make_experts() {
    std::vector<exact::Fp4Expert> out;
    for (int e = 0; e < d.experts; e++) {
      const Packed p = random_packed(rng, d.hidden, d.moe_intermediate, lo, hi);
      out.push_back(p.expert());
      deq.push_back(dequantized(p));
    }
    return out;
  }
};

}  // namespace ds4n::glm53::test

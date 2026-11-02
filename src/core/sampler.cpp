#include "core/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

namespace ds4n::sample {

namespace {

uint64_t splitmix64(uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

/** The logits bias and penalties make (a copy only when they change something). */
std::vector<float> adjusted(std::span<const float> logits, const Settings& s, std::span<const int32_t> recent) {
  std::vector<float> out(logits.begin(), logits.end());
  for (const Bias& b : s.bias) out[static_cast<size_t>(b.token)] += b.bias;
  const Penalties& p = s.penalties;
  if (p.last_n > 0 && (p.repeat != 1.0f || p.frequency != 0.0f || p.presence != 0.0f)) {
    std::map<int32_t, int> counts;
    for (const int32_t t : recent.last(std::min(recent.size(), static_cast<size_t>(p.last_n))))
      if (t >= 0 && static_cast<size_t>(t) < out.size()) counts[t]++;
    for (const auto& [t, n] : counts) {
      float& l = out[static_cast<size_t>(t)];
      l = l <= 0.0f ? l * p.repeat : l / p.repeat;
      l -= static_cast<float>(n) * p.frequency + p.presence;
    }
  }
  return out;
}

/** Ids by logit, descending, ties to the lower id. */
struct Higher {
  std::span<const float> l;
  bool operator()(int32_t a, int32_t b) const { return l[a] != l[b] ? l[a] > l[b] : a < b; }
};

/** The `k` highest finite logits' ids (all finite ones for k = 0), descending. */
std::vector<int32_t> candidates(std::span<const float> l, int k) {
  const Higher higher{l};
  std::vector<int32_t> ids;
  if (k <= 0) {
    for (size_t i = 0; i < l.size(); i++)
      if (std::isfinite(l[i])) ids.push_back(static_cast<int32_t>(i));
    std::sort(ids.begin(), ids.end(), higher);
    return ids;
  }
  // One pass with a heap of the k best so far (its top the worst of them): most ids lose one compare.
  ids.reserve(static_cast<size_t>(k));
  for (size_t i = 0; i < l.size(); i++) {
    if (!std::isfinite(l[i])) continue;
    const auto id = static_cast<int32_t>(i);
    if (ids.size() < static_cast<size_t>(k)) {
      ids.push_back(id);
      std::push_heap(ids.begin(), ids.end(), higher);
    } else if (higher(id, ids.front())) {
      std::pop_heap(ids.begin(), ids.end(), higher);
      ids.back() = id;
      std::push_heap(ids.begin(), ids.end(), higher);
    }
  }
  std::sort_heap(ids.begin(), ids.end(), higher);
  return ids;
}

}  // namespace

Result<Sampling, SamplingError> make_sampling(Settings raw, int32_t vocab) {
  using R = Result<Sampling, SamplingError>;
  if (!std::isfinite(raw.temperature) || raw.temperature < 0.0f) return R::err({"temperature must be a number >= 0"});
  if (raw.top_k < 0) return R::err({"top_k must be >= 0"});
  if (!(raw.top_p > 0.0f && raw.top_p <= 1.0f)) return R::err({"top_p must be in (0, 1]"});
  if (!(raw.min_p >= 0.0f && raw.min_p <= 1.0f)) return R::err({"min_p must be in [0, 1]"});
  const Penalties& p = raw.penalties;
  if (p.last_n < 0) return R::err({"penalty_last_n must be >= 0"});
  if (!std::isfinite(p.repeat) || p.repeat <= 0.0f || !std::isfinite(p.frequency) || !std::isfinite(p.presence))
    return R::err({"penalties must be finite, the repeat penalty above 0"});
  for (const Bias& b : raw.bias) {
    if (b.token < 0 || b.token >= vocab) return R::err({"logit_bias names a token outside the vocabulary"});
    if (!std::isfinite(b.bias)) return R::err({"logit_bias values must be finite"});
  }
  return R::ok(Sampling(std::move(raw)));
}

double draw(uint64_t seed, uint64_t index) {
  return static_cast<double>(splitmix64(splitmix64(seed) + index) >> 11) * 0x1.0p-53;
}

Result<int32_t, SamplingError> pick(std::span<const float> logits, const Sampling& sampling, std::span<const int32_t> recent,
                                    uint64_t index) {
  using R = Result<int32_t, SamplingError>;
  if (logits.empty()) return R::err({"no logits"});
  if (std::any_of(logits.begin(), logits.end(), [](float v) { return std::isnan(v); })) return R::err({"a NaN logit"});
  const Settings& s = sampling.settings();
  const bool changed = !s.bias.empty() || s.penalties.last_n > 0;
  const std::vector<float> owned = changed ? adjusted(logits, s, recent) : std::vector<float>{};
  const std::span<const float> l = changed ? std::span<const float>(owned) : logits;

  if (sampling.greedy_pick()) {
    const auto best = std::max_element(l.begin(), l.end());  // the first of equal maxima: the lower id
    if (!std::isfinite(*best)) return R::err({"no finite logit"});
    return R::ok(static_cast<int32_t>(best - l.begin()));
  }

  std::vector<int32_t> ids = candidates(l, s.top_k);
  if (ids.empty()) return R::err({"no finite logit"});
  const double top = l[ids.front()];
  // top-p over the softmax of the candidates (untempered, as llama.cpp's chain): the shortest prefix
  // reaching top_p of their mass.
  if (s.top_p < 1.0f) {
    double total = 0.0;
    for (const int32_t id : ids) total += std::exp(l[id] - top);
    double mass = 0.0;
    size_t keep = 0;
    while (keep < ids.size() && mass < s.top_p * total) mass += std::exp(l[ids[keep++]] - top);
    ids.resize(std::max<size_t>(keep, 1));
  }
  // min-p: probability at least min_p times the top one's.
  if (s.min_p > 0.0f) {
    const double floor = top + std::log(static_cast<double>(s.min_p));
    ids.erase(std::find_if(ids.begin() + 1, ids.end(), [&](int32_t id) { return l[id] < floor; }), ids.end());
  }
  std::vector<double> weight(ids.size());
  for (size_t i = 0; i < ids.size(); i++) weight[i] = std::exp((l[ids[i]] - top) / s.temperature);
  const double total = std::accumulate(weight.begin(), weight.end(), 0.0);
  const double target = draw(s.seed, index) * total;
  double cumulative = 0.0;
  for (size_t i = 0; i < ids.size(); i++) {
    cumulative += weight[i];
    if (target < cumulative) return R::ok(ids[i]);
  }
  return R::ok(ids.back());  // rounding at the very top of the cumulative sum
}

}  // namespace ds4n::sample

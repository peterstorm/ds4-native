#pragma once
// One step's routing as the GPU kernels consume it: the distinct routed
// Experts in ascending id (which is also the combine order), the tokens that
// use each, and their routing weights. Plain data with C arrays so CUDA code
// shares this definition; built by moe::plan (CPU) and cuda::route (GPU).
#include <cstdint>

namespace ds4n::moe {

constexpr int kExperts = 384;
constexpr int kTopK = 6;
constexpr int kMaxTokens = 8;                 // decode and verify rows per step
constexpr int kMaxJobs = kTopK * kMaxTokens;  // distinct routed Experts in one step, at most

struct Plan {
  int32_t tokens;
  int32_t jobs;                        // distinct routed Experts; -1 when routing failed (non-finite logits)
  int32_t expert[kMaxJobs];            // ascending; entries past `jobs` are 0
  uint8_t token_mask[kMaxJobs];        // bit n set when token n routes to expert[j]
  float weight[kMaxJobs][kMaxTokens];  // routing weight, 0 where not routed
};

}  // namespace ds4n::moe

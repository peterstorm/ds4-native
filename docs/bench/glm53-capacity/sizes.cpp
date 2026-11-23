// Copies of the struct layouts at ds4-native 6866989 (moe_plan.hpp:33, mx_gemv.cuh:56-75, moe_groups.hpp:72)
#include <cstdint>
#include <cstdio>
constexpr int kMaxTokens = 8, kTopK = 8, kMaxJobs = kTopK * kMaxTokens;
struct Plan { int32_t tokens; int32_t jobs; int32_t expert[kMaxJobs]; uint8_t token_mask[kMaxJobs]; float weight[kMaxJobs][kMaxTokens]; };
enum class WeightFormat : int32_t { Mxfp4, Fp8 };
struct GemvJob { WeightFormat format; const uint8_t* weights; const uint8_t* act_codes; const uint8_t* act_scales; float* out; int rows; int blocks; int splits; int split_blocks; int groups = 1; };
struct TokenSlots { int32_t token, first, count; };
int main() { std::printf("Plan %zu GemvJob %zu TokenSlots %zu\n", sizeof(Plan), sizeof(GemvJob), sizeof(TokenSlots)); }

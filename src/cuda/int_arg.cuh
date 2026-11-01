#pragma once
// An integer kernel argument given by value, or read from device memory when
// the kernel runs. Kernels replayed in a CUDA graph keep their captured
// arguments, so per-step values (positions, cache lengths, ring slots) reach
// them through device memory; one-off launches can still pass values. A
// device argument carries a bound, which sizes the launch: kernels must give
// the same result for any grid big enough for the bound.
#include <cstdint>

namespace ds4n::cuda {

struct IntArg {
  const int64_t* device = nullptr;  // read at run time when set
  int64_t value = 0;                // the raw value when `device` is null
  int64_t divisor = 1;              // the argument is ceil(raw / divisor)
  int64_t raw_bound = 0;            // raw <= raw_bound

  static constexpr IntArg of(int64_t v) { return {nullptr, v, 1, v}; }
  static constexpr IntArg at(const int64_t* p, int64_t bound) { return {p, 0, 1, bound}; }
  /** ceil(ceil(raw / a) / b) = ceil(raw / (a b)) for positive a, b and raw >= 0. */
  constexpr IntArg ceil_div(int64_t d) const { return {device, value, divisor * d, raw_bound}; }
  /** The largest value the argument can take. */
  constexpr int64_t bound() const { return (raw_bound + divisor - 1) / divisor; }
  /** Entry r of a per-row array of such values (device), or the same value for every row. */
#ifdef __CUDACC__
  __host__ __device__
#endif
  constexpr IntArg row(int64_t r) const {
    return {device ? device + r : nullptr, value, divisor, raw_bound};
  }

#ifdef __CUDACC__
  __device__ __forceinline__ int64_t get() const {
    const int64_t v = device ? *device : value;
    return (v + divisor - 1) / divisor;
  }
#endif
};

static_assert(IntArg::of(17).ceil_div(8).bound() == 3 && IntArg::of(16).ceil_div(8).bound() == 2);

}  // namespace ds4n::cuda

// The card size Decoder::open plans against: cudaMemGetInfo's total, in bytes and MiB.
#include <cstdio>
#include <cuda_runtime.h>
int main() {
  size_t free_b = 0, total_b = 0;
  if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 1;
  std::printf("cudaMemGetInfo total %zu B (%.3f MiB), free %zu B\n", total_b, total_b / 1048576.0, free_b);
  return 0;
}

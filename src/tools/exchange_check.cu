// ds4n-exchange-check: the two-GPU exchange. Both GPUs produce changing FP32
// partials every round, exchange them and check on the device that the output
// is bf16(p0 + p1) on both, counting any mismatch (a race shows up as a stale
// inbox); the sizes alternate with decode-sized exchanges so both protocols share
// the inbox in turn. Then times exchanges of T tokens back to back and as CUDA graph replays.
//   ds4n-exchange-check [--rounds N] [--tokens T]
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "cuda/exchange.cuh"

using namespace ds4n;

namespace {

[[noreturn]] void die(const std::string& m) {
  std::fprintf(stderr, "ds4n-exchange-check: %s\n", m.c_str());
  std::exit(1);
}
void ck(cudaError_t e, const char* w) {
  if (e != cudaSuccess) die(std::string(w) + ": " + cudaGetErrorString(e));
}

__host__ __device__ float partial_value(int gpu, uint32_t round, size_t i) {
  const uint32_t h = static_cast<uint32_t>(i) * 2654435761u + round * 40503u + static_cast<uint32_t>(gpu) * 977u;
  return static_cast<float>(h % 20011u) * 1e-3f - 10.0f;
}

__global__ void produce(int gpu, const uint32_t* round, size_t n, float* partial) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) partial[i] = partial_value(gpu, *round, i);
}

__global__ void verify(uint32_t* round, size_t n, const __nv_bfloat16* out, unsigned long long* bad) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float want = __bfloat162float(__float2bfloat16_rn(__fadd_rn(partial_value(0, *round, i), partial_value(1, *round, i))));
    if (__bfloat162float(out[i]) != want) atomicAdd(bad, 1ull);
  }
}

__global__ void next_round(uint32_t* round) { ++*round; }

__global__ void produce_round(int gpu, uint32_t round, size_t n, float* partial) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) partial[i] = partial_value(gpu, round, i);
}

__global__ void verify_round(uint32_t round, size_t n, const __nv_bfloat16* out, unsigned long long* bad) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float want = __bfloat162float(__float2bfloat16_rn(__fadd_rn(partial_value(0, round, i), partial_value(1, round, i))));
    if (__bfloat162float(out[i]) != want) atomicAdd(bad, 1ull);
  }
}

struct Side {
  cudaStream_t stream;
  float* partial;
  __nv_bfloat16* out;
  uint32_t* round;
  unsigned long long* bad;
  float* burst_partial[2];
  __nv_bfloat16* burst_out;
  cuda::ExchangeLink link;
};

}  // namespace

int main(int argc, char** argv) {
  int rounds = 20000, tokens = 1;
  for (int i = 1; i < argc; i++) {
    const std::string s = argv[i];
    if (s == "--rounds" && i + 1 < argc) rounds = std::stoi(argv[++i]);
    else if (s == "--tokens" && i + 1 < argc) tokens = std::stoi(argv[++i]);
    else die("unknown argument " + s);
  }
  // Rows of 5120 values; the inbox holds a Round's largest exchange (8 Engram rows = 40 of these).
  constexpr int kMaxTokens = cuda::kExchangeMaxBlocks * cuda::kExchangeChunk / 5120;
  if (tokens < 1 || tokens > kMaxTokens) die("tokens must be 1.." + std::to_string(kMaxTokens));
  int count = 0;
  ck(cudaGetDeviceCount(&count), "device count");
  if (count < 2) die("needs two GPUs");
  const size_t n = static_cast<size_t>(tokens) * 5120;
  Side side[2];
  for (int g = 0; g < 2; g++) {
    ck(cudaSetDevice(g), "set device");
    int can = 0;
    ck(cudaDeviceCanAccessPeer(&can, g, 1 - g), "can access peer");
    if (!can) die("no peer access");
    ck(cudaDeviceEnablePeerAccess(1 - g, 0), "enable peer access");
    Side& s = side[g];
    ck(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking), "stream");
    ck(cudaMalloc(&s.partial, n * 4), "malloc");
    ck(cudaMalloc(&s.out, n * 2), "malloc");
    ck(cudaMalloc(&s.round, 4), "malloc");
    ck(cudaMalloc(&s.bad, 8), "malloc");
    ck(cudaMemset(s.round, 0, 4), "memset");
    ck(cudaMemset(s.bad, 0, 8), "memset");
    s.link.rank = g;
    ck(cudaMalloc(&s.link.inbox, cuda::kExchangeInboxFloats * 4), "malloc");
    ck(cudaMalloc(&s.link.flags, cuda::kExchangeMaxBlocks * 4), "malloc");
    ck(cudaMalloc(&s.link.sent, cuda::kExchangeMaxBlocks * 4), "malloc");
    ck(cudaMemset(s.link.flags, 0, cuda::kExchangeMaxBlocks * 4), "memset");
    ck(cudaMemset(s.link.sent, 0, cuda::kExchangeMaxBlocks * 4), "memset");
  }
  for (int g = 0; g < 2; g++) {
    side[g].link.peer_inbox = side[1 - g].link.inbox;
    side[g].link.peer_flags = side[1 - g].link.flags;
  }
  for (int g = 0; g < 2; g++) {
    ck(cudaSetDevice(g), "set device");
    ck(cudaDeviceSynchronize(), "sync");
  }

  // Correctness: produce, exchange, verify, advance -- every round, both GPUs in step. Every third
  // round is a decode-sized exchange and every third a Round's, so the two protocols (one push, or
  // scatter and gather by size) follow each other through the same inbox slots and flags.
  const unsigned grid = static_cast<unsigned>((n + 255) / 256);
  const size_t sizes[3] = {n, std::min<size_t>(n, 5120), std::min<size_t>(n, 8 * 5120)};
  for (int r = 0; r < rounds; r++)
    for (int g = 0; g < 2; g++) {
      Side& s = side[g];
      const size_t m = sizes[r % 3];
      ck(cudaSetDevice(g), "set device");
      produce<<<grid, 256, 0, s.stream>>>(g, s.round, m, s.partial);
      cuda::exchange_add(s.link, s.partial, m, s.out, s.stream);
      verify<<<grid, 256, 0, s.stream>>>(s.round, m, s.out, s.bad);
      next_round<<<1, 1, 0, s.stream>>>(s.round);
    }
  unsigned long long bad[2];
  for (int g = 0; g < 2; g++) {
    ck(cudaSetDevice(g), "set device");
    ck(cudaStreamSynchronize(side[g].stream), "sync");
    ck(cudaMemcpy(&bad[g], side[g].bad, 8, cudaMemcpyDeviceToHost), "bad");
  }
  std::printf("exchange %d token%s x %d rounds: %llu / %llu mismatching values on GPU 0 / 1\n", tokens,
              tokens > 1 ? "s" : "", rounds, bad[0], bad[1]);
  if (bad[0] || bad[1]) die("exchange produced wrong values");

  // Back to back: bursts of exchanges with no work between them (inbox parity keeps them apart),
  // every fourth decode-sized.
  constexpr int kBurst = 64, kBursts = 200;
  auto burst_size = [&](int i) { return i % 4 == 3 ? sizes[1] : n; };
  for (int g = 0; g < 2; g++) {
    Side& s = side[g];
    ck(cudaSetDevice(g), "set device");
    ck(cudaMalloc(&s.burst_partial[0], n * 4), "malloc");
    ck(cudaMalloc(&s.burst_partial[1], n * 4), "malloc");
    ck(cudaMalloc(&s.burst_out, n * 2 * kBurst), "malloc");
    produce_round<<<grid, 256, 0, s.stream>>>(g, 1000000u, n, s.burst_partial[0]);
    produce_round<<<grid, 256, 0, s.stream>>>(g, 1000001u, n, s.burst_partial[1]);
    ck(cudaMemset(s.bad, 0, 8), "memset");
  }
  for (int b = 0; b < kBursts; b++) {
    for (int g = 0; g < 2; g++) {
      Side& s = side[g];
      ck(cudaSetDevice(g), "set device");
      for (int i = 0; i < kBurst; i++)
        cuda::exchange_add(s.link, s.burst_partial[i % 2], burst_size(i), s.burst_out + static_cast<size_t>(i) * n, s.stream);
    }
    for (int g = 0; g < 2; g++) {
      Side& s = side[g];
      ck(cudaSetDevice(g), "set device");
      for (int i = 0; i < kBurst; i++)
        verify_round<<<grid, 256, 0, s.stream>>>(1000000u + i % 2, burst_size(i), s.burst_out + static_cast<size_t>(i) * n,
                                                 s.bad);
    }
  }
  for (int g = 0; g < 2; g++) {
    ck(cudaSetDevice(g), "set device");
    ck(cudaStreamSynchronize(side[g].stream), "sync");
    ck(cudaMemcpy(&bad[g], side[g].bad, 8, cudaMemcpyDeviceToHost), "bad");
  }
  std::printf("back-to-back bursts %d x %d: %llu / %llu mismatching values on GPU 0 / 1\n", kBursts, kBurst, bad[0],
              bad[1]);
  if (bad[0] || bad[1]) die("back-to-back exchanges produced wrong values");

  // Latency: exchanges back to back, then a graph of 100 exchanges replayed.
  auto time_exchanges = [&](bool graphs) {
    constexpr int kPerGraph = 100, kReplays = 50;
    cudaGraphExec_t exec[2];
    if (graphs)
      for (int g = 0; g < 2; g++) {
        ck(cudaSetDevice(g), "set device");
        cudaGraph_t graph;
        ck(cudaStreamBeginCapture(side[g].stream, cudaStreamCaptureModeThreadLocal), "capture");
        for (int i = 0; i < kPerGraph; i++) cuda::exchange_add(side[g].link, side[g].partial, n, side[g].out, side[g].stream);
        ck(cudaStreamEndCapture(side[g].stream, &graph), "end capture");
        ck(cudaGraphInstantiate(&exec[g], graph, 0), "instantiate");
      }
    cudaEvent_t a[2], b[2];
    for (int g = 0; g < 2; g++) {
      ck(cudaSetDevice(g), "set device");
      ck(cudaEventCreate(&a[g]), "event");
      ck(cudaEventCreate(&b[g]), "event");
      ck(cudaEventRecord(a[g], side[g].stream), "record");
    }
    for (int rep = 0; rep < kReplays; rep++)
      for (int g = 0; g < 2; g++) {
        ck(cudaSetDevice(g), "set device");
        if (graphs) ck(cudaGraphLaunch(exec[g], side[g].stream), "graph launch");
        else
          for (int i = 0; i < kPerGraph; i++) cuda::exchange_add(side[g].link, side[g].partial, n, side[g].out, side[g].stream);
      }
    float ms[2];
    for (int g = 0; g < 2; g++) {
      ck(cudaSetDevice(g), "set device");
      ck(cudaEventRecord(b[g], side[g].stream), "record");
      ck(cudaEventSynchronize(b[g]), "sync");
      ck(cudaEventElapsedTime(&ms[g], a[g], b[g]), "elapsed");
    }
    return std::max(ms[0], ms[1]) * 1000.0 / (kPerGraph * kReplays);
  };
  std::printf("exchange %d token%s (%zu KB per GPU): %.2f us launched back to back, %.2f us in CUDA graphs\n", tokens,
              tokens > 1 ? "s" : "", n * 4 / 1024, time_exchanges(false), time_exchanges(true));
  return 0;
}

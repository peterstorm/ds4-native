#include "cuda/glm53/grouped_moe.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <string>

#include "core/glm53/moe_plan.hpp"
#include "core/glm53/store.hpp"
#include "cuda/glm53/moe.cuh"
#include "cuda/moe.cuh"
#include "cuda/mx_gemv.cuh"

namespace ds4n::cuda::glm53 {
namespace {

namespace moe_plan = ds4n::glm53::moe_plan;

constexpr int kDim = static_cast<int>(ds4n::glm53::kHidden);
constexpr int kRows = static_cast<int>(ds4n::glm53::kHalfRows);
constexpr int kGateBlocks = static_cast<int>(ds4n::glm53::kGateUpBlocks);
constexpr int kDownBlocks = static_cast<int>(ds4n::glm53::kHalfDownBlocks);
constexpr int kSplits = exact::kExpertSplits;
constexpr int kBatch = 512;  // groups a pass of launches covers
constexpr float kNoLimit = std::numeric_limits<float>::infinity();  // GLM's SwiGLU: DS4.1's, unclamped

/** A CUDA failure inside run(); run() returns it as a MoeError. */
struct Failure {
  std::string what;
};

void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw Failure{std::string(what) + ": " + cudaGetErrorString(e)};
}

struct DeviceFree {
  void operator()(void* p) const { cudaFree(p); }
};
template <class T>
using DevicePtr = std::unique_ptr<T, DeviceFree>;

template <class T>
DevicePtr<T> alloc(size_t n) {
  void* p = nullptr;
  ck(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)), "cudaMalloc");
  return DevicePtr<T>(static_cast<T*>(p));
}

template <class T>
DevicePtr<T> upload(std::span<const T> v) {
  auto d = alloc<T>(v.size());
  ck(cudaMemcpy(d.get(), v.data(), v.size_bytes(), cudaMemcpyHostToDevice), "upload");
  return d;
}

template <class T>
std::vector<T> download(const T* d, size_t n) {
  std::vector<T> v(n);
  ck(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
  return v;
}

/** One Half of an Expert in VRAM: views into its Expert's upload. */
struct DeviceHalf {
  const uint8_t *gate, *up, *down;
};

struct DeviceExpert {
  DevicePtr<uint8_t> bytes;
  std::array<DeviceHalf, exact::kHalves> half;
};

DeviceExpert upload_expert(const exact::ExpertTiles& t) {
  DeviceExpert e{alloc<uint8_t>(t[0].bytes.size() + t[1].bytes.size()), {}};
  size_t at = 0;
  for (size_t h = 0; h < exact::kHalves; h++) {
    uint8_t* base = e.bytes.get() + at;
    ck(cudaMemcpy(base, t[h].bytes.data(), t[h].bytes.size(), cudaMemcpyHostToDevice), "upload Expert");
    e.half[h] = {base, base + t[h].up_offset(), base + t[h].down_offset()};
    at += t[h].bytes.size();
  }
  return e;
}

/** A batched GEMV launch living on the device. */
struct Launch {
  DevicePtr<GemvJob> jobs;
  DevicePtr<int> first;
  int n, warps;
};

Launch make_launch(const std::vector<GemvJob>& jobs) {
  std::vector<int> first;
  int warps = 0;
  for (const auto& j : jobs) {
    first.push_back(warps);
    warps += group_warps(j);
  }
  return {upload<GemvJob>(jobs), upload<int>(first), static_cast<int>(jobs.size()), warps};
}

}  // namespace

struct GroupedMoe::Impl {
  int device;
  DevicePtr<float> gate, up, down;
  DevicePtr<uint8_t> h_codes, h_scales, x_codes, x_scales;
  MoeBuffers buf() const {
    return {gate.get(), up.get(), h_codes.get(), h_scales.get(), down.get(), kRows, kSplits, kDim};
  }
};

GroupedMoe::GroupedMoe(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GroupedMoe::GroupedMoe(GroupedMoe&&) noexcept = default;
GroupedMoe& GroupedMoe::operator=(GroupedMoe&&) noexcept = default;
GroupedMoe::~GroupedMoe() = default;

Result<GroupedMoe, exact::MoeError> GroupedMoe::open(int device) {
  using R = Result<GroupedMoe, exact::MoeError>;
  try {
    ck(cudaSetDevice(device), "cudaSetDevice");
    auto impl = std::make_unique<Impl>();
    impl->device = device;
    const size_t plane = static_cast<size_t>(kSplits) * kRows * kMaxTokens;
    impl->gate = alloc<float>(kBatch * plane);
    impl->up = alloc<float>(kBatch * plane);
    impl->down = alloc<float>(static_cast<size_t>(kBatch) * kDim * kMaxTokens);
    impl->h_codes = alloc<uint8_t>(static_cast<size_t>(kBatch) * kDownBlocks * 256);
    impl->h_scales = alloc<uint8_t>(static_cast<size_t>(kBatch) * kDownBlocks * 8);
    impl->x_codes = alloc<uint8_t>(static_cast<size_t>(kBatch) * kGateBlocks * 256);
    impl->x_scales = alloc<uint8_t>(static_cast<size_t>(kBatch) * kGateBlocks * 8);
    return R::ok(GroupedMoe(std::move(impl)));
  } catch (const Failure& f) {
    return R::err({"GPU " + std::to_string(device) + ": " + f.what});
  }
}

Result<std::vector<exact::MoeOutput>, exact::MoeError> GroupedMoe::run(
    exact::Departure d, const exact::ExpertTiles& shared, std::span<const MoeToken> tokens,
    const std::map<int32_t, const exact::ExpertTiles*>& experts) {
  using R = Result<std::vector<exact::MoeOutput>, exact::MoeError>;
  if (!runs_on_gpu(d)) return R::err({"the GPU does not run " + std::string(exact::name_of(d))});
  if (tokens.empty()) return R::ok({});
  std::vector<ds4n::glm53::Route> routes;
  for (const MoeToken& t : tokens) {
    if (t.ffn_in.size() != static_cast<size_t>(kDim)) return R::err({"a token's input is not the hidden width"});
    routes.push_back(*t.route);
  }
  const auto grouped = moe_plan::group_tokens(routes);
  if (!grouped) return R::err({grouped.error().what});
  const moe::GroupPlan& gp = grouped.value();
  for (size_t i = 0; i < gp.routed; i++)
    if (!experts.contains(gp.groups[i].expert))
      return R::err({"Expert " + std::to_string(gp.groups[i].expert) + " has no tiles"});
  const bool weight_after_down = d == exact::Departure::WeightAfterDown;

  try {
    ck(cudaSetDevice(impl_->device), "cudaSetDevice");
    const MoeBuffers buf = impl_->buf();
    std::map<int32_t, DeviceExpert> resident;
    for (size_t i = 0; i < gp.routed; i++)
      if (!resident.contains(gp.groups[i].expert))
        resident.emplace(gp.groups[i].expert, upload_expert(*experts.at(gp.groups[i].expert)));
    const DeviceExpert shared_dev = upload_expert(shared);

    const size_t n_tok = tokens.size(), n_out = n_tok * kDim;
    std::vector<__nv_bfloat16> xb(n_out);
    for (size_t t = 0; t < n_tok; t++)
      for (size_t i = 0; i < static_cast<size_t>(kDim); i++) xb[t * kDim + i] = __float2bfloat16(tokens[t].ffn_in[i]);
    const auto d_x = upload<__nv_bfloat16>(xb);
    const std::array<DevicePtr<float>, exact::kHalves> y{alloc<float>(n_out), alloc<float>(n_out)};
    const int sb = split_blocks_for(kGateBlocks, kSplits);

    for (size_t h = 0; h < exact::kHalves; h++) {
      ck(cudaMemset(y[h].get(), 0, n_out * sizeof(float)), "memset");
      for (size_t at = 0; at < gp.groups.size(); at += kBatch) {
        const size_t count = std::min<size_t>(kBatch, gp.groups.size() - at);
        const auto stretch = std::span(gp.groups).subspan(at, count);
        std::vector<int32_t> token_of;
        std::vector<float> weight, swiglu_weight;  // per column; the SwiGLU's is 1 when the weight comes after down
        for (const moe::Group& g : stretch)
          for (int c = 0; c < kMaxTokens; c++) {
            token_of.push_back(g.token[c]);
            weight.push_back(g.weight[c]);
            swiglu_weight.push_back(weight_after_down ? (g.token[c] >= 0 ? 1.0f : 0.0f) : g.weight[c]);
          }
        std::vector<GemvJob> gate_up, down;
        for (const moe::ExpertRun& run : moe::expert_runs(stretch)) {
          const moe::Group& g = stretch[run.first];
          const int slot = static_cast<int>(run.first), groups = static_cast<int>(run.count);
          uint8_t* codes = impl_->x_codes.get() + run.first * kGateBlocks * 256;
          uint8_t* scales = impl_->x_scales.get() + run.first * kGateBlocks * 8;
          const auto hid = buf.hidden_of(slot);
          const DeviceHalf& w = g.expert == moe::kSharedExpert ? shared_dev.half[h] : resident.at(g.expert).half[h];
          gate_up.push_back({WeightFormat::Mxfp4, w.gate, codes, scales, buf.gate_of(slot), kRows, kGateBlocks, kSplits, sb,
                             groups});
          gate_up.push_back(
              {WeightFormat::Mxfp4, w.up, codes, scales, buf.up_of(slot), kRows, kGateBlocks, kSplits, sb, groups});
          down.push_back({WeightFormat::Mxfp4, w.down, hid.codes, hid.scales, buf.down_of(slot), kDim, kDownBlocks, 1,
                          kDownBlocks, groups});
        }
        const Launch gu = make_launch(gate_up), dn = make_launch(down);
        const moe::TokenColumns columns = moe::token_columns(stretch);
        const auto d_token_slots = upload<moe::TokenSlots>(columns.tokens);
        const auto d_slots = upload<int32_t>(columns.slots);
        const auto d_token_of = upload<int32_t>(token_of);
        const auto d_weight = upload<float>(weight);
        const auto d_swiglu_weight = upload<float>(swiglu_weight);
        const int n = static_cast<int>(count), n_tokens = static_cast<int>(columns.tokens.size());
        gather_quantize(d_x.get(), kGateBlocks, d_token_of.get(), n, impl_->x_codes.get(), impl_->x_scales.get(), nullptr);
        gemv_groups(gu.jobs.get(), gu.first.get(), gu.n, gu.warps, n, nullptr);
        swiglu_quantize_groups(d_token_of.get(), d_swiglu_weight.get(), n, buf, kNoLimit, nullptr);
        gemv_groups(dn.jobs.get(), dn.first.get(), dn.n, dn.warps, n, nullptr);
        if (weight_after_down)
          accumulate_tokens_weighted(d_token_slots.get(), n_tokens, d_slots.get(), d_weight.get(), buf, y[h].get(), nullptr);
        else
          accumulate_tokens(d_token_slots.get(), n_tokens, d_slots.get(), buf, y[h].get(), nullptr);
        ck(cudaGetLastError(), "launch");
        ck(cudaDeviceSynchronize(), "sweep");  // before this pass's buffers are freed
      }
    }

    const std::array<std::vector<float>, exact::kHalves> partial{download(y[0].get(), n_out), download(y[1].get(), n_out)};
    std::vector<float> out(n_out);
    if (d == exact::Departure::Fp32Exchange) {
      for (size_t i = 0; i < n_out; i++) out[i] = partial[0][i] + partial[1][i];
    } else {
      const auto d_out = alloc<__nv_bfloat16>(n_out);
      add_round_bf16(y[0].get(), y[1].get(), n_out, d_out.get(), nullptr);
      ck(cudaDeviceSynchronize(), "exchange");
      const auto b = download(d_out.get(), n_out);
      for (size_t i = 0; i < n_out; i++) out[i] = __bfloat162float(b[i]);
    }
    std::vector<exact::MoeOutput> result(n_tok);
    for (size_t t = 0; t < n_tok; t++) {
      const auto row = [&](const std::vector<float>& v) {
        return std::vector<float>(v.begin() + static_cast<std::ptrdiff_t>(t * kDim),
                                  v.begin() + static_cast<std::ptrdiff_t>((t + 1) * kDim));
      };
      result[t] = exact::MoeOutput{{row(partial[0]), row(partial[1])}, row(out)};
    }
    return R::ok(std::move(result));
  } catch (const Failure& f) {
    return R::err({"GPU " + std::to_string(impl_->device) + ": " + f.what});
  }
}

}  // namespace ds4n::cuda::glm53

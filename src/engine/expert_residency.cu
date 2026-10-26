#include "engine/expert_residency.cuh"

#include <algorithm>
#include <chrono>
#include <map>
#include <optional>
#include <span>

#include "core/expert_cache.hpp"
#include "core/store_layout.hpp"
#include "engine/device_layer.cuh"
#include "io/file.hpp"
#include "io/uring_reader.hpp"

namespace ds4n::engine {
namespace {

using Clock = std::chrono::steady_clock;
constexpr uint32_t kRecordsPerAllocation = 64;  // RAM tier records per pinned allocation (1.2 GB)
constexpr uint32_t kBatch = moe::kMaxJobs;          // most Loads one place() call can need
// Store reads go in 1 MiB chunks (one record of the store's ZFS dataset), each copied to its GPU
// as it lands, so the copies run under the reads and only the last chunk's copy is exposed.
constexpr uint64_t kReadChunk = store::kAlign;
constexpr uint32_t kChunksPerHalf = static_cast<uint32_t>(store::kHalfStride / kReadChunk);  // 9
constexpr unsigned kQueueDepth = kBatch * store::kHalves * kChunksPerHalf;
static_assert(store::kHalfStride % kReadChunk == 0);

[[noreturn]] void fail(const std::string& what) { throw ResidencyError(what); }

double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

}  // namespace

struct ExpertResidency::State {
  ResidencyConfig config;
  io::Fd store;
  io::UringReader reader;
  cache::ExpertCache vram, ram;
  std::array<uint8_t*, 2> pool{};  // per GPU: slots x kHalfPayloadBytes
  std::array<cudaStream_t, 2> copy{};
  std::vector<std::byte*> allocations;  // pinned RAM tier memory
  std::vector<std::byte*> records;  // per RAM tier Slot: one record (kRecordStride bytes)
  ResidencyStats stats;

  State(ResidencyConfig c, io::Fd fd, io::UringReader r, cache::ExpertCache v, cache::ExpertCache m)
      : config(std::move(c)), store(std::move(fd)), reader(std::move(r)), vram(std::move(v)), ram(std::move(m)) {}
  ~State();

  uint8_t* slot_ptr(int half, uint32_t slot) const {
    return pool[half] + static_cast<size_t>(slot) * store::kHalfPayloadBytes;
  }
  /** Both Halves of RAM tier record `record` into VRAM Slot `slot`, each on its GPU's copy stream. */
  void copy_record(uint32_t record, uint32_t slot);
  /** Fill the Loads' VRAM Slots (at most kBatch), returning when both GPUs' copies landed. */
  void load(std::span<const cache::Load> loads);
};

ExpertResidency::State::~State() {
  for (int h = 0; h < 2; h++) {
    cudaSetDevice(config.devices[h]);
    if (copy[h]) {
      cudaStreamSynchronize(copy[h]);
      cudaStreamDestroy(copy[h]);
    }
    if (pool[h]) cudaFree(pool[h]);
  }
  for (std::byte* a : allocations) cudaFreeHost(a);
}

void ExpertResidency::State::copy_record(uint32_t record, uint32_t slot) {
  for (int h = 0; h < 2; h++) {
    check(cudaSetDevice(config.devices[h]), "cudaSetDevice");
    check(cudaMemcpyAsync(slot_ptr(h, slot), records[record] + h * store::kHalfStride, store::kHalfPayloadBytes,
                          cudaMemcpyHostToDevice, copy[h]),
          "Half upload");
  }
}

void ExpertResidency::State::load(std::span<const cache::Load> loads) {
  std::vector<cache::ExpertKey> keys;
  std::map<cache::ExpertKey, uint32_t> slot_of;
  for (const auto& l : loads) {
    keys.push_back(l.key);
    slot_of[l.key] = l.slot;
  }
  auto placed = ram.plan(keys);
  if (!placed) fail("RAM tier: " + placed.error().what);
  for (const auto& h : placed.value().hits) {
    copy_record(h.slot, slot_of.at(h.key));
    stats.ram_hits++;
  }
  // Store reads land in the RAM tier buffers they took, chunk by chunk; each chunk's copy to its
  // GPU starts as it lands. Tag: key << 8 | chunk index within the record (Half-major).
  std::vector<io::ReadRequest> reads;
  for (const auto& l : placed.value().loads) {
    const store::ExpertId id{l.key / store::kExperts, l.key % store::kExperts};
    for (uint32_t c = 0; c < store::kHalves * kChunksPerHalf; c++)
      reads.push_back({store.get(), store::record_offset(id) + c * kReadChunk,
                       std::span(records[l.slot] + c * kReadChunk, kReadChunk), uint64_t{l.key} << 8 | c});
    stats.store_reads++;
  }
  if (!reads.empty()) {
    if (auto s = reader.submit(reads); !s) fail("Expert store: " + s.error().what);
    for (size_t landed = 0; landed < reads.size();) {
      auto done = reader.wait(1);
      if (!done) fail("Expert store: " + done.error().what);
      for (const auto& c : done.value()) {
        if (c.result != static_cast<int32_t>(kReadChunk))
          fail("Expert store: short or failed read (" + std::to_string(c.result) + ")");
        const auto key = static_cast<cache::ExpertKey>(c.tag >> 8);
        const auto chunk = static_cast<uint32_t>(c.tag & 0xFF);
        const int h = static_cast<int>(chunk / kChunksPerHalf);
        const uint64_t at = (chunk % kChunksPerHalf) * kReadChunk;  // within the Half
        const uint64_t bytes = std::min(kReadChunk, store::kHalfPayloadBytes - at);
        check(cudaSetDevice(config.devices[h]), "cudaSetDevice");
        check(cudaMemcpyAsync(slot_ptr(h, slot_of.at(key)) + at, records[*ram.slot_of(key)] + h * store::kHalfStride + at,
                              bytes, cudaMemcpyHostToDevice, copy[h]),
              "Half upload");
        landed++;
      }
    }
  }
  for (int h = 0; h < 2; h++) {
    check(cudaSetDevice(config.devices[h]), "cudaSetDevice");
    check(cudaStreamSynchronize(copy[h]), "Half upload");
  }
}

ExpertResidency::ExpertResidency(std::unique_ptr<State> state) : s_(std::move(state)) {}
ExpertResidency::~ExpertResidency() = default;

std::unique_ptr<ExpertResidency> ExpertResidency::open(const ResidencyConfig& config) {
  if (config.slots < kBatch || config.ram_records < kBatch)
    fail("the Expert cache and the RAM tier each need at least " + std::to_string(kBatch) + " Slots");
  auto fd = io::open_file(config.store_path, io::Access::ReadDirect);
  if (!fd) fail(fd.error().what);
  io::AlignedBuffer header(store::kHeaderBytes);
  if (auto r = io::pread_exact(fd.value(), header.span(), 0); !r) fail(r.error().what);
  const auto h = store::decode_header(header.span());
  if (!h) fail("Expert store header: " + h.error().what);
  auto reader = io::UringReader::create(kQueueDepth);
  if (!reader) fail(reader.error().what);
  auto vram = cache::ExpertCache::create(config.slots, config.prior, cache::Policy{});
  if (!vram) fail("Expert cache: " + vram.error().what);
  auto ram = cache::ExpertCache::empty(config.ram_records, cache::Policy{4096, false});
  if (!ram) fail("RAM tier: " + ram.error().what);
  auto s = std::make_unique<State>(config, std::move(fd).value(), std::move(reader).value(), std::move(vram).value(),
                                   std::move(ram).value());

  for (int g = 0; g < 2; g++) {
    check(cudaSetDevice(config.devices[g]), "cudaSetDevice");
    check(cudaMalloc(&s->pool[g], static_cast<size_t>(config.slots) * store::kHalfPayloadBytes), "Slot pool");
    check(cudaStreamCreateWithFlags(&s->copy[g], cudaStreamNonBlocking), "cudaStreamCreate");
  }
  // Portable: both GPUs copy from the same records at pinned speed.
  for (uint32_t first = 0; first < config.ram_records; first += kRecordsPerAllocation) {
    const uint32_t n = std::min(kRecordsPerAllocation, config.ram_records - first);
    void* p = nullptr;
    check(cudaHostAlloc(&p, static_cast<size_t>(n) * store::kRecordStride, cudaHostAllocPortable), "RAM tier");
    s->allocations.push_back(static_cast<std::byte*>(p));
    for (uint32_t i = 0; i < n; i++) s->records.push_back(static_cast<std::byte*>(p) + i * store::kRecordStride);
  }

  // Warm start: every Slot gets its Expert, kBatch records at a time.
  const auto t0 = Clock::now();
  const auto warm = s->vram.warm_start();
  for (size_t i = 0; i < warm.size(); i += kBatch)
    s->load(std::span(warm).subspan(i, std::min<size_t>(kBatch, warm.size() - i)));
  s->stats = {};
  s->stats.warm_ms = ms_since(t0);
  return std::unique_ptr<ExpertResidency>(new ExpertResidency(std::move(s)));
}

std::array<std::vector<const uint8_t*>, 2> ExpertResidency::place(int layer, const moe::Plan& plan) {
  if (plan.jobs < 1 || plan.jobs > moe::kMaxJobs) fail("place: the plan has no valid job count");
  if (layer < 0 || layer >= static_cast<int>(store::kLayers)) fail("place: layer out of range");
  std::vector<cache::ExpertKey> keys;
  for (int j = 0; j < plan.jobs; j++)
    keys.push_back(static_cast<cache::ExpertKey>(layer) * store::kExperts + static_cast<uint32_t>(plan.expert[j]));
  auto placed = s_->vram.plan(keys);
  if (!placed) fail("Expert cache: " + placed.error().what);
  s_->stats.layer_steps++;
  s_->stats.hits += placed.value().hits.size();
  if (!placed.value().loads.empty()) {
    const auto t0 = Clock::now();
    s_->load(placed.value().loads);
    s_->stats.exposed_ms += ms_since(t0);
  }
  std::array<std::vector<const uint8_t*>, 2> out;
  for (int h = 0; h < 2; h++)
    for (cache::ExpertKey k : keys) out[h].push_back(s_->slot_ptr(h, *s_->vram.slot_of(k)));
  return out;
}

const ResidencyStats& ExpertResidency::stats() const { return s_->stats; }

}  // namespace ds4n::engine

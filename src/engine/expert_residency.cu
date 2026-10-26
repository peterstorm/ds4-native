#include "engine/expert_residency.cuh"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <thread>

#include "core/expert_cache.hpp"
#include "core/miss_protocol.hpp"
#include "core/store_layout.hpp"
#include "engine/device_layer.cuh"
#include "io/file.hpp"
#include "io/uring_reader.hpp"

namespace ds4n::engine {
namespace {

using Clock = std::chrono::steady_clock;
constexpr uint32_t kRecordsPerAllocation = 64;  // RAM tier records per pinned allocation (1.2 GB)
constexpr uint32_t kBatch = moe::kMaxJobs;          // most Loads one step can need
// Store reads go in 1 MiB chunks (one record of the store's ZFS dataset), each copied to its GPU
// as it lands, so the copies run under the reads and only the last chunk's copy is exposed.
constexpr uint64_t kReadChunk = store::kAlign;
constexpr uint32_t kChunksPerHalf = static_cast<uint32_t>(store::kHalfStride / kReadChunk);  // 9
constexpr unsigned kQueueDepth = kBatch * store::kHalves * kChunksPerHalf;
static_assert(store::kHalfStride % kReadChunk == 0);
static_assert(moe::kResidencyKeys == cache::kKeys);
// The service spins while steps keep coming; after this long without one it polls with short sleeps.
constexpr auto kSpinWindow = std::chrono::milliseconds(2);
constexpr auto kIdlePoll = std::chrono::microseconds(20);

[[noreturn]] void fail(const std::string& what) { throw ResidencyError(what); }

double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

template <class T>
T* host_mapped(size_t n) {
  void* p = nullptr;
  engine::check(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocPortable | cudaHostAllocMapped), "cudaHostAlloc (mapped)");
  return static_cast<T*>(p);
}

}  // namespace

struct ExpertResidency::State {
  ResidencyConfig config;
  io::Fd store;
  io::UringReader reader;
  cache::ExpertCache vram, ram;
  std::array<uint8_t*, 2> pool{};  // per GPU: slots x kHalfPayloadBytes
  std::array<cudaStream_t, 2> copy{};
  std::vector<std::byte*> allocations;  // pinned RAM tier memory
  std::vector<std::byte*> records;      // per RAM tier Slot: one record (kRecordStride bytes)
  // The protocol pages (host-mapped, both GPUs).
  int32_t* table = nullptr;  // key -> Slot, -1 when not resident
  moe::Mailbox* mailbox = nullptr;
  moe::Board* board = nullptr;
  std::array<cuda::ExpertDirectory, 2> directories{};
  // The service.
  std::thread service;
  std::atomic<bool> stop{false};
  mutable std::mutex mutex;  // guards stats and failure
  ResidencyStats stats;
  std::string failure;

  State(ResidencyConfig c, io::Fd fd, io::UringReader r, cache::ExpertCache v, cache::ExpertCache m)
      : config(std::move(c)), store(std::move(fd)), reader(std::move(r)), vram(std::move(v)), ram(std::move(m)) {}
  ~State();

  uint8_t* slot_ptr(int half, uint32_t slot) const {
    return pool[half] + static_cast<size_t>(slot) * store::kHalfPayloadBytes;
  }
  cache::ExpertKey key_of(int32_t layer, int32_t expert) const {
    return static_cast<cache::ExpertKey>(layer % static_cast<int32_t>(config.key_layers)) * store::kExperts +
           static_cast<cache::ExpertKey>(expert);
  }
  /** Both Halves of RAM tier record `record` into VRAM Slot `slot`, each on its GPU's copy stream. */
  void copy_record(uint32_t record, uint32_t slot);
  /** Fill the Loads' VRAM Slots (at most kBatch), returning when both GPUs' copies landed. */
  void load(std::span<const cache::Load> loads, ResidencyStats& counts);
  /** Drain posted steps in order until stopped. */
  void serve();
  /** One posted step through the cache; Loads land before the table names them. */
  void serve_step(const moe::PostedStep& step, ResidencyStats& counts);
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
  for (void* p : {static_cast<void*>(table), static_cast<void*>(mailbox), static_cast<void*>(board)})
    if (p) cudaFreeHost(p);
}

void ExpertResidency::State::copy_record(uint32_t record, uint32_t slot) {
  for (int h = 0; h < 2; h++) {
    engine::check(cudaSetDevice(config.devices[h]), "cudaSetDevice");
    engine::check(cudaMemcpyAsync(slot_ptr(h, slot), records[record] + h * store::kHalfStride, store::kHalfPayloadBytes,
                          cudaMemcpyHostToDevice, copy[h]),
          "Half upload");
  }
}

void ExpertResidency::State::load(std::span<const cache::Load> loads, ResidencyStats& counts) {
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
    counts.ram_hits++;
  }
  // Store reads land in the RAM tier buffers they took, chunk by chunk; each chunk's copy to its
  // GPU starts as it lands. Tag: key << 8 | chunk index within the record (Half-major).
  std::vector<io::ReadRequest> reads;
  for (const auto& l : placed.value().loads) {
    const store::ExpertId id{l.key / store::kExperts, l.key % store::kExperts};
    for (uint32_t c = 0; c < store::kHalves * kChunksPerHalf; c++)
      reads.push_back({store.get(), store::record_offset(id) + c * kReadChunk,
                       std::span(records[l.slot] + c * kReadChunk, kReadChunk), uint64_t{l.key} << 8 | c});
    counts.store_reads++;
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
        engine::check(cudaSetDevice(config.devices[h]), "cudaSetDevice");
        engine::check(cudaMemcpyAsync(slot_ptr(h, slot_of.at(key)) + at, records[*ram.slot_of(key)] + h * store::kHalfStride + at,
                              bytes, cudaMemcpyHostToDevice, copy[h]),
              "Half upload");
        landed++;
      }
    }
  }
  for (int h = 0; h < 2; h++) {
    engine::check(cudaSetDevice(config.devices[h]), "cudaSetDevice");
    engine::check(cudaStreamSynchronize(copy[h]), "Half upload");
  }
}

void ExpertResidency::State::serve_step(const moe::PostedStep& step, ResidencyStats& counts) {
  counts.layer_steps++;
  if (step.jobs <= 0) return;  // failed routing: nothing to place (the GPUs run the shared Expert only)
  if (step.jobs > moe::kMaxJobs || step.layer < 0 || step.layer >= static_cast<int32_t>(store::kLayers))
    fail("posted step out of range (layer " + std::to_string(step.layer) + ", " + std::to_string(step.jobs) + " jobs)");
  std::vector<cache::ExpertKey> keys;
  for (int j = 0; j < step.jobs; j++) keys.push_back(key_of(step.layer, step.expert[j]));
  auto placed = vram.plan(keys);
  if (!placed) fail("Expert cache: " + placed.error().what);
  counts.hits += placed.value().hits.size();
  if (placed.value().loads.empty()) return;
  const auto t0 = Clock::now();
  load(placed.value().loads, counts);
  // The copies landed: now the table may name the new Slots (and forget the evicted Experts).
  for (const auto& l : placed.value().loads) {
    if (l.evicted) std::atomic_ref<int32_t>(table[*l.evicted]).store(-1, std::memory_order_relaxed);
    std::atomic_ref<int32_t>(table[l.key]).store(static_cast<int32_t>(l.slot), std::memory_order_relaxed);
  }
  counts.exposed_ms += ms_since(t0);
}

void ExpertResidency::State::serve() {
  uint64_t done = 0;
  auto last_step = Clock::now();
  try {
    while (!stop.load(std::memory_order_relaxed)) {
      const uint64_t posted = std::atomic_ref<uint64_t>(mailbox->posted).load(std::memory_order_acquire);
      if (posted == done) {
        if (Clock::now() - last_step < kSpinWindow) _mm_pause();
        else std::this_thread::sleep_for(kIdlePoll);
        continue;
      }
      ResidencyStats counts;
      for (uint64_t s = done + 1; s <= posted; s++) {
        const moe::PostedStep step = mailbox->ring[(s - 1) % moe::kMissRing];
        serve_step(step, counts);
        std::atomic_ref<uint64_t>(board->served).store(s, std::memory_order_release);
        done = s;
      }
      last_step = Clock::now();
      std::lock_guard lock(mutex);
      stats.layer_steps += counts.layer_steps;
      stats.hits += counts.hits;
      stats.ram_hits += counts.ram_hits;
      stats.store_reads += counts.store_reads;
      stats.exposed_ms += counts.exposed_ms;
    }
  } catch (const std::exception& e) {
    {
      std::lock_guard lock(mutex);
      failure = e.what();
    }
    std::atomic_ref<uint32_t>(board->failed).store(1, std::memory_order_release);
  }
}

ExpertResidency::ExpertResidency(std::unique_ptr<State> state) : s_(std::move(state)) {}

ExpertResidency::~ExpertResidency() {
  s_->stop.store(true);
  if (s_->service.joinable()) s_->service.join();
}

std::unique_ptr<ExpertResidency> ExpertResidency::open(const ResidencyConfig& config) {
  if (config.slots < kBatch || config.ram_records < kBatch)
    fail("the Expert cache and the RAM tier each need at least " + std::to_string(kBatch) + " Slots");
  if (config.key_layers < 1 || config.key_layers > store::kLayers) fail("key_layers must be in 1..40");
  auto fd = io::open_file(config.store_path, io::Access::ReadDirect);
  if (!fd) fail(fd.error().what);
  io::AlignedBuffer header(store::kHeaderBytes);
  if (auto r = io::pread_exact(fd.value(), header.span(), 0); !r) fail(r.error().what);
  const auto h = store::decode_header(header.span());
  if (!h) fail("Expert store header: " + h.error().what);
  auto reader = io::UringReader::create(kQueueDepth);
  if (!reader) fail(reader.error().what);
  // Without a prior, the warm start takes keys in order among those the layers can use.
  std::vector<uint32_t> prior = config.prior;
  if (prior.empty() && config.key_layers < store::kLayers) {
    prior.assign(cache::kKeys, 0);
    std::fill(prior.begin(), prior.begin() + config.key_layers * store::kExperts, 1u);
  }
  auto vram = cache::ExpertCache::create(config.slots, prior, cache::Policy{});
  if (!vram) fail("Expert cache: " + vram.error().what);
  auto ram = cache::ExpertCache::empty(config.ram_records, cache::Policy{4096, false});
  if (!ram) fail("RAM tier: " + ram.error().what);
  auto s = std::make_unique<State>(config, std::move(fd).value(), std::move(reader).value(), std::move(vram).value(),
                                   std::move(ram).value());

  for (int g = 0; g < 2; g++) {
    engine::check(cudaSetDevice(config.devices[g]), "cudaSetDevice");
    engine::check(cudaMalloc(&s->pool[g], static_cast<size_t>(config.slots) * store::kHalfPayloadBytes), "Slot pool");
    engine::check(cudaStreamCreateWithFlags(&s->copy[g], cudaStreamNonBlocking), "cudaStreamCreate");
  }
  // Portable: both GPUs copy from the same records at pinned speed.
  for (uint32_t first = 0; first < config.ram_records; first += kRecordsPerAllocation) {
    const uint32_t n = std::min(kRecordsPerAllocation, config.ram_records - first);
    void* p = nullptr;
    engine::check(cudaHostAlloc(&p, static_cast<size_t>(n) * store::kRecordStride, cudaHostAllocPortable), "RAM tier");
    s->allocations.push_back(static_cast<std::byte*>(p));
    for (uint32_t i = 0; i < n; i++) s->records.push_back(static_cast<std::byte*>(p) + i * store::kRecordStride);
  }

  // Warm start: every Slot gets its Expert, kBatch records at a time; then the table names them.
  const auto t0 = Clock::now();
  const auto warm = s->vram.warm_start();
  ResidencyStats ignored;
  for (size_t i = 0; i < warm.size(); i += kBatch)
    s->load(std::span(warm).subspan(i, std::min<size_t>(kBatch, warm.size() - i)), ignored);
  s->table = host_mapped<int32_t>(moe::kResidencyKeys);
  std::fill(s->table, s->table + moe::kResidencyKeys, -1);
  for (const auto& l : warm) s->table[l.key] = static_cast<int32_t>(l.slot);
  s->mailbox = host_mapped<moe::Mailbox>(1);
  s->board = host_mapped<moe::Board>(1);
  *s->mailbox = {};
  *s->board = {};
  s->stats.warm_ms = ms_since(t0);

  for (int g = 0; g < 2; g++) {
    engine::check(cudaSetDevice(config.devices[g]), "cudaSetDevice");
    cuda::ExpertDirectory& d = s->directories[g];
    void* p = nullptr;
    engine::check(cudaHostGetDevicePointer(&p, s->table, 0), "cudaHostGetDevicePointer");
    d.slot_of = static_cast<const int32_t*>(p);
    engine::check(cudaHostGetDevicePointer(&p, s->board, 0), "cudaHostGetDevicePointer");
    d.board = static_cast<const moe::Board*>(p);
    if (g == 0) {
      engine::check(cudaHostGetDevicePointer(&p, s->mailbox, 0), "cudaHostGetDevicePointer");
      d.mailbox = static_cast<moe::Mailbox*>(p);
    }
    d.pool = s->pool[g];
    d.slot_bytes = store::kHalfPayloadBytes;
    d.key_layers = static_cast<int>(config.key_layers);
  }
  State* raw = s.get();
  s->service = std::thread([raw] { raw->serve(); });
  return std::unique_ptr<ExpertResidency>(new ExpertResidency(std::move(s)));
}

cuda::ExpertDirectory ExpertResidency::directory(int rank) const { return s_->directories.at(rank); }

ResidencyStats ExpertResidency::stats() const {
  std::lock_guard lock(s_->mutex);
  return s_->stats;
}

void ExpertResidency::check() const {
  std::lock_guard lock(s_->mutex);
  if (!s_->failure.empty()) throw ResidencyError("Expert residency: " + s_->failure);
}

}  // namespace ds4n::engine

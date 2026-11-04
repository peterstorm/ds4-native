#include "engine/expert_residency.cuh"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <iterator>
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
  std::optional<cache::ExpertCache> scratch;  // a prompt sweep's Slots (config.sweep_slots), least recently used
  std::array<uint8_t*, 2> pool{};  // per GPU: (slots + sweep_slots) x kHalfPayloadBytes
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
  std::vector<cache::Access> accesses;  // with config.record_accesses
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
  /** One record to read from the store: into RAM tier buffer `buffer`, then (if set) VRAM Slot `slot`. */
  struct RecordRead {
    cache::ExpertKey key;
    uint32_t buffer;
    std::optional<uint32_t> slot;
  };
  /** Read records (at most kBatch) in 1 MiB chunks; each chunk bound for VRAM is copied to its GPU as it lands. */
  void read_records(std::span<const RecordRead> records_in);
  /**
   * Fill the Loads' VRAM Slots (at most kBatch), returning when both GPUs' copies landed. Decode
   * Loads go through the RAM tier (a store read lands in a record the tier's policy chose); a
   * sweep's only read it: RAM tier records are copied, the rest come from the store through the
   * staging buffers, so a prompt's one-off Loads never displace what the tier keeps for every prompt.
   */
  void load(std::span<const cache::Load> loads, bool sweep, ResidencyStats& counts);
  /** Wait until both GPUs' copy streams drained. */
  void sync_copies();
  /** Drain posted steps in order until stopped. */
  void serve();
  /** Posted step `s` through the cache; Loads land, and GPU 1 has routed the step, before the table changes. */
  void serve_step(const moe::PostedStep& step, uint64_t s, ResidencyStats& counts, std::vector<cache::Access>& seen);
  /** `keys` through the two regions (see the comment inside); the Loads, Slots in pool numbering. */
  std::vector<cache::Load> plan_keys(std::span<const cache::ExpertKey> keys, std::span<const cache::ExpertKey> hinted,
                                     bool sweep, ResidencyStats& counts);
  /** The table names the Loads' Slots and forgets what they evicted (their copies have landed). */
  void publish(std::span<const cache::Load> loads);
  /** A resident key's Slot in pool numbering. */
  uint32_t slot_of(cache::ExpertKey key) const;
  // Held by the service around each step and by place(): the caches, the reader and the copy
  // streams have one user at a time.
  std::mutex placement;
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

void ExpertResidency::State::load(std::span<const cache::Load> loads, bool sweep, ResidencyStats& counts) {
  std::vector<RecordRead> from_store;
  if (sweep) {
    for (const auto& l : loads)
      if (const auto record = ram.slot_of(l.key)) {
        copy_record(*record, l.slot);
        counts.ram_hits++;
      } else {
        from_store.push_back({l.key, config.ram_records + static_cast<uint32_t>(from_store.size()), l.slot});  // staging
      }
  } else {
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
    // Store reads land in the RAM tier buffers they took.
    for (const auto& l : placed.value().loads) from_store.push_back({l.key, l.slot, slot_of.at(l.key)});
  }
  counts.store_reads += from_store.size();
  read_records(from_store);
  sync_copies();
}

void ExpertResidency::State::sync_copies() {
  for (int h = 0; h < 2; h++) {
    engine::check(cudaSetDevice(config.devices[h]), "cudaSetDevice");
    engine::check(cudaStreamSynchronize(copy[h]), "Half upload");
  }
}

void ExpertResidency::State::read_records(std::span<const RecordRead> records_in) {
  // Tag: index into records_in << 8 | chunk index within the record (Half-major).
  std::vector<io::ReadRequest> reads;
  for (size_t i = 0; i < records_in.size(); i++) {
    const RecordRead& r = records_in[i];
    const store::ExpertId id{r.key / store::kExperts, r.key % store::kExperts};
    for (uint32_t c = 0; c < store::kHalves * kChunksPerHalf; c++)
      reads.push_back({store.get(), store::record_offset(id) + c * kReadChunk,
                       std::span(records[r.buffer] + c * kReadChunk, kReadChunk), uint64_t{i} << 8 | c});
  }
  if (!reads.empty()) {
    if (auto s = reader.submit(reads); !s) fail("Expert store: " + s.error().what);
    for (size_t landed = 0; landed < reads.size();) {
      auto done = reader.wait(1);
      if (!done) fail("Expert store: " + done.error().what);
      for (const auto& c : done.value()) {
        if (c.result != static_cast<int32_t>(kReadChunk))
          fail("Expert store: short or failed read (" + std::to_string(c.result) + ")");
        const RecordRead& r = records_in[c.tag >> 8];
        landed++;
        if (!r.slot) continue;
        const auto chunk = static_cast<uint32_t>(c.tag & 0xFF);
        const int h = static_cast<int>(chunk / kChunksPerHalf);
        const uint64_t at = (chunk % kChunksPerHalf) * kReadChunk;  // within the Half
        const uint64_t bytes = std::min(kReadChunk, store::kHalfPayloadBytes - at);
        engine::check(cudaSetDevice(config.devices[h]), "cudaSetDevice");
        engine::check(cudaMemcpyAsync(slot_ptr(h, *r.slot) + at, records[r.buffer] + h * store::kHalfStride + at, bytes,
                                      cudaMemcpyHostToDevice, copy[h]),
                      "Half upload");
      }
    }
  }
}

void ExpertResidency::State::serve_step(const moe::PostedStep& step, uint64_t s, ResidencyStats& counts,
                                        std::vector<cache::Access>& seen) {
  counts.layer_steps++;
  if (step.jobs <= 0) return;  // failed routing: nothing to place (the GPUs run the shared Expert only)
  if (step.jobs > moe::kMaxJobs || step.layer < 0 || step.layer >= static_cast<int32_t>(store::kLayers))
    fail("posted step out of range (layer " + std::to_string(step.layer) + ", " + std::to_string(step.jobs) + " jobs)");
  std::vector<cache::ExpertKey> keys;
  for (int j = 0; j < step.jobs; j++) keys.push_back(key_of(step.layer, step.expert[j]));
  if (step.near_count < 0 || step.near_count > moe::kMaxTokens * moe::kNear) fail("posted step with a bad near-miss count");
  std::vector<cache::ExpertKey> near;  // distinct, none needed
  for (int i = 0; i < step.near_count; i++) {
    if (step.near[i] < 0 || step.near[i] >= static_cast<int32_t>(store::kExperts)) fail("posted near-miss out of range");
    const cache::ExpertKey k = key_of(step.layer, step.near[i]);
    if (std::find(keys.begin(), keys.end(), k) == keys.end() && std::find(near.begin(), near.end(), k) == near.end())
      near.push_back(k);
  }
  if (config.record_accesses) seen.push_back({keys, near});
  const bool sweep = (step.flags & moe::kSweepStep) != 0;
  const auto loads = plan_keys(keys, config.hints && !sweep ? std::span<const cache::ExpertKey>(near)
                                                            : std::span<const cache::ExpertKey>{},
                               sweep, counts);
  if (loads.empty()) return;
  const auto t0 = Clock::now();
  load(loads, sweep, counts);
  // The table changes only once GPU 1 routed this step too: Cache-conditional routing reads it.
  const auto wait_start = Clock::now();
  while (std::atomic_ref<uint64_t>(mailbox->peer_routed).load(std::memory_order_acquire) < s) {
    if (Clock::now() - wait_start > std::chrono::seconds(10)) fail("GPU 1 never routed step " + std::to_string(s));
    _mm_pause();
  }
  publish(loads);
  counts.exposed_ms += ms_since(t0);
}

std::vector<cache::Load> ExpertResidency::State::plan_keys(std::span<const cache::ExpertKey> keys,
                                                           std::span<const cache::ExpertKey> hinted, bool sweep,
                                                           ResidencyStats& counts) {
  // Two regions: the decode Slots and the scratch Slots a prompt sweep's Misses take. A key lives in
  // at most one; where it lives it is a Hit. A sweep's other keys load into scratch, a decode
  // step's into the decode Slots, so a sweep never evicts what decode keeps -- nor reorders it: a
  // sweep's Hits in the decode Slots are not uses there, so they keep what decode used last (the
  // reply after a prompt routes like earlier replies, not like the prompt).
  sweep = sweep && scratch.has_value();
  std::vector<cache::ExpertKey> decode_keys, scratch_keys;
  for (const cache::ExpertKey k : keys) {
    const bool in_scratch = scratch && scratch->slot_of(k).has_value();
    const bool to_scratch = in_scratch || (sweep && !vram.slot_of(k).has_value());
    (to_scratch ? scratch_keys : decode_keys).push_back(k);
  }
  std::vector<cache::Load> loads;
  if (sweep) {
    counts.hits += decode_keys.size();  // all resident there
  } else {
    auto placed = vram.plan(decode_keys, hinted);
    if (!placed) fail("Expert cache: " + placed.error().what);
    counts.hits += placed.value().hits.size();
    loads = std::move(placed.value().loads);
  }
  if (!scratch_keys.empty()) {
    auto placed = scratch->plan(scratch_keys);
    if (!placed) fail("sweep Slots: " + placed.error().what);
    counts.hits += placed.value().hits.size();
    for (cache::Load l : placed.value().loads) {
      l.slot += config.slots;  // the scratch Slots follow the decode Slots in the pool
      loads.push_back(l);
    }
  }
  return loads;
}

void ExpertResidency::State::publish(std::span<const cache::Load> loads) {
  // The copies landed: now the table may name the new Slots (and forget the evicted Experts).
  for (const auto& l : loads) {
    if (l.evicted) std::atomic_ref<int32_t>(table[*l.evicted]).store(-1, std::memory_order_relaxed);
    std::atomic_ref<int32_t>(table[l.key]).store(static_cast<int32_t>(l.slot), std::memory_order_relaxed);
  }
}

uint32_t ExpertResidency::State::slot_of(cache::ExpertKey key) const {
  if (const auto s = vram.slot_of(key)) return *s;
  if (scratch)
    if (const auto s = scratch->slot_of(key)) return config.slots + *s;
  fail("an Expert just placed has no Slot (internal invariant broken)");
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
      std::vector<cache::Access> seen;
      for (uint64_t s = done + 1; s <= posted; s++) {
        const moe::PostedStep step = mailbox->ring[(s - 1) % moe::kMissRing];
        {
          std::lock_guard placing(placement);
          serve_step(step, s, counts, seen);
        }
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
      accesses.insert(accesses.end(), std::make_move_iterator(seen.begin()), std::make_move_iterator(seen.end()));
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
  if (config.sweep_slots != 0 && config.sweep_slots < kBatch)
    fail("the sweep Slots, when any, number at least " + std::to_string(kBatch));
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
  auto vram = cache::ExpertCache::create(config.slots, prior, config.vram_policy);
  if (!vram) fail("Expert cache: " + vram.error().what);
  // The RAM tier: Recent ranks by last load and starts empty; Popular ranks by decayed load count
  // and, with a prior, starts with the profile's next-hottest Experts after the VRAM set.
  const bool popular = config.ram_policy == RamPolicy::Popular;
  const cache::Policy ram_policy{4096, popular};
  std::vector<uint32_t> ram_prior;
  if (popular && !prior.empty()) {
    ram_prior = prior;
    for (const auto& l : vram.value().warm_start()) ram_prior[l.key] = 0;
  }
  auto ram = ram_prior.empty() ? cache::ExpertCache::empty(config.ram_records, ram_policy)
                               : cache::ExpertCache::create(config.ram_records, ram_prior, ram_policy);
  if (!ram) fail("RAM tier: " + ram.error().what);
  auto s = std::make_unique<State>(config, std::move(fd).value(), std::move(reader).value(), std::move(vram).value(),
                                   std::move(ram).value());
  if (config.sweep_slots > 0) {
    auto scratch = cache::ExpertCache::empty(config.sweep_slots, cache::Policy{4096, false});
    if (!scratch) fail("sweep Slots: " + scratch.error().what);
    s->scratch = std::move(scratch).value();
  }

  for (int g = 0; g < 2; g++) {
    engine::check(cudaSetDevice(config.devices[g]), "cudaSetDevice");
    engine::check(cudaMalloc(&s->pool[g], (static_cast<size_t>(config.slots) + config.sweep_slots) * store::kHalfPayloadBytes), "Slot pool");
    engine::check(cudaStreamCreateWithFlags(&s->copy[g], cudaStreamNonBlocking), "cudaStreamCreate");
  }
  // Portable: both GPUs copy from the same records at pinned speed. The RAM tier's records, then
  // kBatch staging buffers for a sweep's store reads.
  for (uint32_t first = 0; first < config.ram_records + kBatch; first += kRecordsPerAllocation) {
    const uint32_t n = std::min(kRecordsPerAllocation, config.ram_records + kBatch - first);
    void* p = nullptr;
    engine::check(cudaHostAlloc(&p, static_cast<size_t>(n) * store::kRecordStride, cudaHostAllocPortable), "RAM tier");
    s->allocations.push_back(static_cast<std::byte*>(p));
    for (uint32_t i = 0; i < n; i++) s->records.push_back(static_cast<std::byte*>(p) + i * store::kRecordStride);
  }

  // Warm start: every Slot gets its Expert, kBatch records at a time through the first RAM tier
  // buffers as plain staging (outside the RAM tier's policy); then a warm RAM tier is filled, and
  // the table names the Slots.
  const auto t0 = Clock::now();
  const auto warm = s->vram.warm_start();
  for (size_t i = 0; i < warm.size(); i += kBatch) {
    std::vector<State::RecordRead> batch;
    for (size_t j = i; j < std::min<size_t>(warm.size(), i + kBatch); j++)
      batch.push_back({warm[j].key, static_cast<uint32_t>(j - i), warm[j].slot});
    s->read_records(batch);
    s->sync_copies();
  }
  const auto ram_warm = s->ram.warm_start();
  for (size_t i = 0; i < ram_warm.size(); i += kBatch) {
    std::vector<State::RecordRead> batch;
    for (size_t j = i; j < std::min<size_t>(ram_warm.size(), i + kBatch); j++)
      batch.push_back({ram_warm[j].key, ram_warm[j].slot, std::nullopt});
    s->read_records(batch);
  }
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
    engine::check(cudaHostGetDevicePointer(&p, s->mailbox, 0), "cudaHostGetDevicePointer");
    d.mailbox = static_cast<moe::Mailbox*>(p);
    d.rank = g;
    d.route_bonus = config.route_bonus;
    d.pool = s->pool[g];
    d.slot_bytes = store::kHalfPayloadBytes;
    d.key_layers = static_cast<int>(config.key_layers);
    d.post_near = config.record_accesses || config.hints;
  }
  State* raw = s.get();
  s->service = std::thread([raw] { raw->serve(); });
  return std::unique_ptr<ExpertResidency>(new ExpertResidency(std::move(s)));
}

cuda::ExpertDirectory ExpertResidency::directory(int rank) const { return s_->directories.at(rank); }

std::vector<uint32_t> ExpertResidency::place(int layer, std::span<const int32_t> experts) {
  State& s = *s_;
  if (layer < 0 || layer >= static_cast<int>(store::kLayers)) fail("place: layer out of range");
  if (experts.empty() || experts.size() > kBatch) fail("place: 1 to 48 Experts at a time");
  std::vector<cache::ExpertKey> keys;
  for (const int32_t e : experts) {
    if (e < 0 || e >= static_cast<int32_t>(store::kExperts)) fail("place: Expert out of range");
    keys.push_back(s.key_of(layer, e));
  }
  ResidencyStats counts;
  std::vector<uint32_t> slots;
  {
    std::lock_guard placing(s.placement);
    const auto loads = s.plan_keys(keys, {}, true, counts);
    if (!loads.empty()) {
      const auto t0 = Clock::now();
      s.load(loads, true, counts);
      s.publish(loads);
      counts.exposed_ms += ms_since(t0);
    }
    for (const cache::ExpertKey k : keys) slots.push_back(s.slot_of(k));
  }
  std::lock_guard lock(s.mutex);
  s.stats.hits += counts.hits;
  s.stats.ram_hits += counts.ram_hits;
  s.stats.store_reads += counts.store_reads;
  s.stats.exposed_ms += counts.exposed_ms;
  if (s.config.record_accesses) s.accesses.push_back({keys, {}});
  return slots;
}

ResidencyStats ExpertResidency::stats() const {
  std::lock_guard lock(s_->mutex);
  return s_->stats;
}

std::vector<cache::Access> ExpertResidency::accesses() const {
  std::lock_guard lock(s_->mutex);
  return s_->accesses;
}

void ExpertResidency::check() const {
  std::lock_guard lock(s_->mutex);
  if (!s_->failure.empty()) throw ResidencyError("Expert residency: " + s_->failure);
}

}  // namespace ds4n::engine

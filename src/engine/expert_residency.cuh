#pragma once
// Imperative shell: Expert Halves in VRAM Slots on both GPUs, served on a Miss
// from the RAM tier or the Expert store -- the host side of the Miss protocol
// (core/miss_protocol.hpp). The GPUs resolve their Experts themselves
// (cuda/resolve.cuh): GPU 0 posts every layer step, and a GPU that meets a Miss
// waits until this service has served the step. A service thread drains the
// posted steps in order through core/expert_cache, so the cache sees every
// access, Hits included.
//
// Placement is core/expert_cache twice: the VRAM Slots (warm from construction,
// ranked by decayed pick count) and the RAM tier's pinned record buffers (empty
// at first, least recently used). A Slot index means the same Expert on both
// GPUs, GPU r holding Half r. A Load comes from its RAM tier record when there,
// otherwise from the store in 1 MiB O_DIRECT chunks into a RAM tier buffer, each
// chunk copied to its GPU as it lands. Only after both GPUs' copies landed does
// the table name the new Slots and the board mark the step served.
//
// Residency changes only while a step with Loads is being served, and both GPUs
// wait at that step, so they always resolve against the same table state, the
// one the cache had before the step. Evicting a Slot is safe: a GPU resolving
// step s has passed step s's attention exchange, so the peer finished step s-1's
// Experts, and the cache never evicts what step s needs. Residency never changes
// outputs (the Slot-count invariance gate).
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda/resolve.cuh"

namespace ds4n::engine {

/** A failed Expert store read or an impossible placement: thrown at the boundary. */
struct ResidencyError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct ResidencyConfig {
  std::string store_path;
  std::array<int, 2> devices{0, 1};  // device of Half 0, Half 1
  uint32_t slots = 0;                // VRAM Slots, at least moe::kMaxJobs
  uint32_t ram_records = 0;          // RAM tier record buffers (18 MiB pinned each), at least moe::kMaxJobs
  std::vector<uint32_t> prior;       // pick counts per Expert key for the warm start (empty: by key)
  // Experts of layer L are keyed (and read from the store) as layer L % key_layers: a test store
  // holding only the first few layers can serve every layer. 40 for a real store.
  uint32_t key_layers = 40;
};

/** Counters since open(); a Hit is a needed Expert found in a VRAM Slot. */
struct ResidencyStats {
  uint64_t layer_steps = 0, hits = 0, ram_hits = 0, store_reads = 0;
  double exposed_ms = 0.0;  // service time of steps with Loads (the GPUs wait that long)
  double warm_ms = 0.0;     // filling the Slots at open()
};

class ExpertResidency {
 public:
  /** Allocate the Slots, the RAM tier and the protocol pages, warm every Slot, then start serving. */
  static std::unique_ptr<ExpertResidency> open(const ResidencyConfig& config);
  ~ExpertResidency();  // stops the service
  ExpertResidency(const ExpertResidency&) = delete;
  ExpertResidency& operator=(const ExpertResidency&) = delete;

  /** GPU `rank`'s view for the resolve kernel (rank 0 posts the steps). */
  cuda::ExpertDirectory directory(int rank) const;

  /** A snapshot of the counters. */
  ResidencyStats stats() const;

  /** Throws the service's failure, if it failed (the GPUs then trap at their next Miss). */
  void check() const;

 private:
  struct State;
  explicit ExpertResidency(std::unique_ptr<State> state);
  std::unique_ptr<State> s_;
};

}  // namespace ds4n::engine

#pragma once
// Imperative shell: Expert Halves in VRAM Slots on both GPUs, served on a Miss
// from the RAM tier or the Expert store. Placement is core/expert_cache, twice:
// the VRAM Slots (warm from construction, ranked by recent pick counts) and the
// RAM tier's pinned record buffers (empty at first, least recently used). A
// Slot index means the same Expert on both GPUs, GPU r holding Half r.
// Residency never changes outputs: the kernels read the same Half bytes from
// whichever Slot holds them (the Slot-count invariance gate).
//
// place() serves one layer step. Hits return their Slots. Each Load is copied
// into the Slot it takes, from its RAM-tier record when there, otherwise from
// the store: one O_DIRECT read of the whole record (both Halves) into a RAM-tier
// buffer, so the RAM tier keeps the most recently loaded records. Each Half then
// crosses its own GPU's link on that GPU's copy stream, and place() returns once
// both GPUs' copies landed.
//
// Evicting a Slot is safe because of when the decoder calls place(): after both
// GPUs finished the previous layer's Experts (each GPU's phase A of this layer
// passed its exchange, which follows its phase B of the previous layer), and the
// cache never evicts an Expert the same call needs. place() allocates nothing and
// copies only from pinned memory, so it keeps the decoder's deadlock rule.
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/moe_plan.hpp"

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
  std::vector<uint32_t> prior;       // pick counts per Expert for the warm start (empty: by key)
};

/** Counters since open(); a Hit is a needed Expert found in a VRAM Slot. */
struct ResidencyStats {
  uint64_t layer_steps = 0, hits = 0, ram_hits = 0, store_reads = 0;
  double exposed_ms = 0.0;  // wall time inside place() calls that loaded anything
  double warm_ms = 0.0;     // filling the Slots at open()
};

class ExpertResidency {
 public:
  /** Allocate the Slots and the RAM tier, then fill every Slot with the warm-start Experts. */
  static std::unique_ptr<ExpertResidency> open(const ResidencyConfig& config);
  ~ExpertResidency();
  ExpertResidency(const ExpertResidency&) = delete;
  ExpertResidency& operator=(const ExpertResidency&) = delete;

  /**
   * Both GPUs' Half pointers for the plan's jobs, in plan order (engine::ExpertSlots),
   * valid until the next call.
   */
  std::array<std::vector<const uint8_t*>, 2> place(int layer, const moe::Plan& plan);

  const ResidencyStats& stats() const;

 private:
  struct State;
  explicit ExpertResidency(std::unique_ptr<State> state);
  std::unique_ptr<State> s_;
};

}  // namespace ds4n::engine

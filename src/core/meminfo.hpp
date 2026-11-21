#pragma once
// /proc/meminfo, parsed (pure): what a shell sizes host-memory pools from.
#include <cstdint>
#include <string>
#include <string_view>

#include "core/result.hpp"

namespace ds4n {

struct MemInfoError {
  std::string what;
};

/**
 * MemAvailable from the text of /proc/meminfo, in bytes: the kernel's estimate of memory
 * a new allocation can take without swapping. Missing, malformed, or not in kB is an error.
 */
Result<uint64_t, MemInfoError> mem_available_bytes(std::string_view meminfo);

}  // namespace ds4n

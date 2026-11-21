#include "core/meminfo.hpp"

#include <algorithm>
#include <charconv>

namespace ds4n {

Result<uint64_t, MemInfoError> mem_available_bytes(std::string_view meminfo) {
  using R = Result<uint64_t, MemInfoError>;
  constexpr std::string_view kKey = "MemAvailable:";
  size_t at = 0;
  while (at < meminfo.size()) {
    const size_t end = std::min(meminfo.find('\n', at), meminfo.size());
    std::string_view line = meminfo.substr(at, end - at);
    at = end + 1;
    if (!line.starts_with(kKey)) continue;
    line.remove_prefix(kKey.size());
    while (!line.empty() && line.front() == ' ') line.remove_prefix(1);
    uint64_t kib = 0;
    const char* last = line.data() + line.size();
    const auto [rest, ec] = std::from_chars(line.data(), last, kib);
    if (ec != std::errc{}) return R::err({"MemAvailable: not a number"});
    if (std::string_view(rest, static_cast<size_t>(last - rest)) != " kB") return R::err({"MemAvailable: expected kB"});
    if (kib > UINT64_MAX / 1024) return R::err({"MemAvailable: out of range"});
    return R::ok(kib * 1024);
  }
  return R::err({"no MemAvailable line"});
}

}  // namespace ds4n

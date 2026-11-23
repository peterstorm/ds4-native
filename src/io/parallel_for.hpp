#pragma once
// Imperative shell: a fixed pool of threads over an index range, for the tools
// that fan work out over Experts, layers or tokens (the oracle's run, the
// decode tool's --check replay). A work item reports failure as a message.
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace ds4n::io {

/** fn(i) for i in [0, n) on `threads` threads; the first error message wins. */
template <class Fn>
std::optional<std::string> parallel_for(size_t n, unsigned threads, Fn fn) {
  std::atomic<size_t> next{0};
  std::mutex m;
  std::optional<std::string> failure;
  auto worker = [&] {
    for (size_t i = next++; i < n; i = next++) {
      if (auto e = fn(i)) {
        const std::lock_guard lock(m);
        if (!failure) failure = std::move(e);
        return;
      }
    }
  };
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < std::min<size_t>(threads, std::max<size_t>(n, 1)); t++) pool.emplace_back(worker);
  for (auto& t : pool) t.join();
  return failure;
}

}  // namespace ds4n::io

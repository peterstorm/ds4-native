// core/meminfo: MemAvailable parsed from /proc/meminfo text, refusing what is not it.
#include <doctest/doctest.h>

#include <string>

#include "core/meminfo.hpp"

using namespace ds4n;

TEST_CASE("mem_available_bytes reads MemAvailable in kB as bytes") {
  const std::string text =
      "MemTotal:       96422008 kB\nMemFree:        54521432 kB\nMemAvailable:   64515256 kB\nBuffers:            1264 kB\n";
  CHECK(mem_available_bytes(text).value() == 64515256ull * 1024);
  CHECK(mem_available_bytes("MemAvailable: 0 kB").value() == 0);  // last line, no newline
}

TEST_CASE("mem_available_bytes refuses a missing, malformed or non-kB MemAvailable") {
  CHECK_FALSE(mem_available_bytes("").is_ok());
  CHECK_FALSE(mem_available_bytes("MemTotal: 5 kB\nMemAvailableX: 4 kB\n").is_ok());  // another key does not count
  CHECK_FALSE(mem_available_bytes("MemAvailable:   lots kB\n").is_ok());
  CHECK_FALSE(mem_available_bytes("MemAvailable:   12 MB\n").is_ok());
  CHECK_FALSE(mem_available_bytes("MemAvailable:   12\n").is_ok());
  CHECK_FALSE(mem_available_bytes("MemAvailable:   99999999999999999999 kB\n").is_ok());
}

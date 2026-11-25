#pragma once
// Engram copy (pure): the two Engram tables on a drive of their own, apart from the Expert store's,
// read by direct I/O (`--engram PATH`, a file or a whole partition). A copy is a 4 KiB header, then
// table 0's rows (layer 1), then table 1's (layer 14), each table starting 4 KiB aligned. What this
// decides: where each table lies, the header's bytes, the aligned window a row is read through, and
// which rows are compared with the GGUF at open. Writing, checking and reading a copy is
// engine/engram_source's.
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/engram.hpp"
#include "core/result.hpp"

namespace ds4n::engram {

/** Direct I/O's unit: every offset, length and buffer of a copy's reads is a multiple of it. */
inline constexpr uint64_t kCopyAlign = 4096;
inline constexpr uint64_t kCopyHeaderBytes = kCopyAlign;
inline constexpr uint32_t kCopyVersion = 1;
/** A table's rows at most (2^40): keeps every offset of a copy well inside 64 bits. */
inline constexpr uint64_t kCopyMaxRows = uint64_t{1} << 40;

/** `v` rounded up to a multiple of kCopyAlign. */
constexpr uint64_t copy_align_up(uint64_t v) { return (v + kCopyAlign - 1) / kCopyAlign * kCopyAlign; }

struct CopyError {
  std::string what;
};

/**
 * Where an Engram copy's tables lie. Made only by of() or parse_copy_header, so the tables follow the
 * header in order, each 4 KiB aligned and never overlapping, and the copy ends aligned.
 */
class CopyLayout {
 public:
  /** Tables of `rows[t]` rows (264 bytes each); every count must be 1 .. kCopyMaxRows. */
  static Result<CopyLayout, CopyError> of(std::array<uint64_t, kLayers> rows);

  uint64_t rows(int table) const { return rows_[table]; }
  /** Where row 0 of `table` starts in the copy. */
  uint64_t offset(int table) const { return offset_[table]; }
  /** The whole copy, header included: a multiple of kCopyAlign. */
  uint64_t bytes() const { return bytes_; }

  bool operator==(const CopyLayout&) const = default;

 private:
  CopyLayout() = default;
  std::array<uint64_t, kLayers> rows_{}, offset_{};
  uint64_t bytes_ = 0;
};

/** The copy's header: kCopyHeaderBytes bytes. */
std::vector<std::byte> encode_copy_header(const CopyLayout& layout);

/**
 * The layout a header records: its magic, version, row size and table count must be this build's,
 * and the offsets and size it records must be the ones its row counts give. `header` is
 * kCopyHeaderBytes.
 */
Result<CopyLayout, CopyError> parse_copy_header(std::span<const std::byte> header);

/** The aligned read holding one row: `bytes` (4 or 8 KiB) at `offset`, the row `skip` bytes in. */
struct CopyWindow {
  uint64_t offset;
  uint32_t bytes;
  uint32_t skip;
  bool operator==(const CopyWindow&) const = default;
};

/** Row `id` of `table` (0 .. kLayers - 1); `id` must be below layout.rows(table). */
CopyWindow copy_window(const CopyLayout& layout, int table, uint64_t id);

/**
 * The rows a check compares: 0, rows - 1 and evenly between, ascending and distinct, min(n, rows) of
 * them (`rows` >= 1, `n` >= 2).
 */
std::vector<uint64_t> sample_rows(uint64_t rows, unsigned n);

}  // namespace ds4n::engram

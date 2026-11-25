#include "engine/engram_source.hpp"

#include <algorithm>
#include <cstring>
#include <optional>

#include "core/gguf.hpp"

namespace ds4n::engine {
namespace {

using R = io::IoError;

template <class T>
Result<std::vector<T>, R> read_array(const io::Fd& fd, const gguf::Header& h, const std::string& key, uint32_t type) {
  using Out = Result<std::vector<T>, R>;
  const auto it = h.kv.find(key);
  if (it == h.kv.end()) return Out::err({"missing GGUF metadata " + key});
  const auto* ref = std::get_if<gguf::ArrayRef>(&it->second);
  if (ref == nullptr || ref->element_type != type) return Out::err({key + " is not the expected array type"});
  std::vector<std::byte> bytes(ref->count * sizeof(T));
  if (auto r = io::pread_exact(fd, bytes, ref->file_offset); !r) return Out::err(r.error());
  std::vector<T> out(ref->count);
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return Out::ok(std::move(out));
}

constexpr uint32_t kGgufUint32 = 4, kGgufUint64 = 10;
constexpr uint64_t kRow = engram::kRowBytes;
/** A staging slot: the largest window a row is read through (engram::copy_window). */
constexpr uint64_t kSlotBytes = 2 * engram::kCopyAlign;
/** What write_engram_copy moves at once. */
constexpr uint64_t kCopyChunk = uint64_t{64} << 20;

/**
 * The copy a GGUF's Engram tables of `rows` rows give: both tables present, and each row addressable by
 * the uint32 id the readers take. One rule for the copy written and the copy opened.
 */
Result<engram::CopyLayout, R> copy_layout_for(const std::array<uint64_t, engram::kLayers>& rows) {
  using Out = Result<engram::CopyLayout, R>;
  for (int t = 0; t < engram::kLayers; t++) {
    if (rows[t] == 0) return Out::err({"the GGUF has no Engram table " + std::to_string(t)});
    if (rows[t] > uint64_t{UINT32_MAX} + 1) return Out::err({"Engram table " + std::to_string(t) + " has more rows than a row id addresses"});
  }
  auto layout = engram::CopyLayout::of(rows);
  if (!layout) return Out::err({layout.error().what});
  return Out::ok(layout.value());
}

/**
 * Every one of `n` reads through `reader`, the queue kept full: make(i) gives read i (asked in order,
 * only when the queue has room), land(c) takes each completion. After the first failure (a submit, a
 * wait, or land refusing a completion) no read goes out, and those in flight complete before it
 * returns, so nothing lands in a buffer after its owner has moved on.
 */
template <class Make, class Land>
Result<size_t, R> pump(io::UringReader& reader, size_t n, Make make, Land land) {
  using Out = Result<size_t, R>;
  std::vector<io::ReadRequest> batch;
  size_t next = 0, landed = 0;
  std::optional<R> failed;
  while (!failed && landed < n) {
    const size_t room = std::min<size_t>(reader.depth() - reader.in_flight(), n - next);
    if (room > 0) {
      batch.clear();
      for (size_t k = 0; k < room; k++) batch.push_back(make(next + k));
      if (auto s = reader.submit(batch); !s) {
        failed = s.error();
        break;
      }
      next += room;
    }
    auto c = reader.wait(1);
    if (!c) {
      failed = c.error();
      break;
    }
    for (const auto& done : c.value()) {
      if (failed) continue;
      if (auto l = land(done); !l) failed = l.error();
      else landed++;
    }
  }
  while (failed && reader.in_flight() > 0)
    if (!reader.wait(1)) break;  // the ring itself failed: nothing more can be reaped
  if (failed) return Out::err(*failed);
  return Out::ok(landed);
}

}  // namespace

Result<engram::Layout, io::IoError> load_engram_layout(const std::string& gguf_path) {
  using Out = Result<engram::Layout, io::IoError>;
  auto header = io::read_gguf_header(gguf_path);
  if (!header) return Out::err(header.error());
  auto fd = io::open_file(gguf_path, io::Access::ReadBuffered);
  if (!fd) return Out::err(fd.error());
  const auto& h = header.value();
  const std::string p = "deepseek41.engram.";
  const std::string* encoding = gguf::find_string(h, p + "encoding");
  if (encoding == nullptr || *encoding != "e4m3_e8m0_32_row264") return Out::err({"unexpected engram encoding"});
  const uint64_t* vocab = gguf::find_uint(h, p + "compressed_vocab_size");
  const uint64_t* pad = gguf::find_uint(h, p + "pad_id");
  if (vocab == nullptr || pad == nullptr) return Out::err({"missing engram vocabulary metadata"});
  auto layer_ids = read_array<uint32_t>(fd.value(), h, p + "layer_ids", kGgufUint32);
  auto token_map = read_array<uint32_t>(fd.value(), h, p + "token_map", kGgufUint32);
  auto primes = read_array<uint32_t>(fd.value(), h, p + "primes", kGgufUint32);
  auto rows32 = read_array<uint32_t>(fd.value(), h, p + "rows", kGgufUint32);
  auto multipliers = read_array<uint64_t>(fd.value(), h, p + "multipliers", kGgufUint64);
  if (!layer_ids) return Out::err(layer_ids.error());
  if (!token_map) return Out::err(token_map.error());
  if (!primes) return Out::err(primes.error());
  if (!rows32) return Out::err(rows32.error());
  if (!multipliers) return Out::err(multipliers.error());
  if (layer_ids.value() != std::vector<uint32_t>{1, 14}) return Out::err({"engram layers are not 1 and 14"});
  const std::vector<uint64_t> rows(rows32.value().begin(), rows32.value().end());
  auto layout = engram::make_layout(std::move(token_map).value(), static_cast<uint32_t>(*vocab), static_cast<uint32_t>(*pad),
                                    primes.value(), multipliers.value(), rows);
  if (!layout) return Out::err({layout.error().what});
  return Out::ok(std::move(layout).value());
}

Result<EngramRows, io::IoError> EngramRows::open(const std::vector<std::string>& shards) {
  using Out = Result<EngramRows, io::IoError>;
  std::vector<io::Fd> fds;
  std::vector<gguf::Header> headers;
  for (const auto& path : shards) {
    auto header = io::read_gguf_header(path);
    if (!header) return Out::err(header.error());
    headers.push_back(std::move(header).value());
    auto fd = io::open_file(path, io::Access::ReadBuffered);
    if (!fd) return Out::err(fd.error());
    fds.push_back(std::move(fd).value());
  }
  std::array<Table, engram::kLayers> tables{};
  for (int t = 0; t < engram::kLayers; t++) {
    const std::string name = "blk." + std::to_string(t == 0 ? 1 : 14) + ".engram_embd.weight";
    for (size_t s = 0; s < headers.size() && tables[t].rows == 0; s++)
      for (const auto& info : headers[s].tensors) {
        if (info.name != name) continue;
        if (info.dims.size() != 2 || info.dims[0] != engram::kRowBytes) return Out::err({name + ": rows are not 264 bytes"});
        tables[t] = {s, headers[s].data_offset + info.offset, info.dims[1]};
        break;
      }
  }
  auto reader = io::UringReader::create(kBulkDepth, false);
  if (!reader) return Out::err(reader.error());
  return Out::ok(EngramRows(Gguf{std::move(fds), tables, std::move(reader).value()}));
}

Result<EngramRows, io::IoError> EngramRows::open_copy(const std::vector<std::string>& shards, const std::string& copy) {
  using Out = Result<EngramRows, io::IoError>;
  auto gguf = open(shards);
  if (!gguf) return gguf;
  std::array<uint64_t, engram::kLayers> rows{};
  for (int t = 0; t < engram::kLayers; t++) rows[t] = gguf.value().rows(t);
  auto expected = copy_layout_for(rows);
  if (!expected) return Out::err({"Engram copy " + copy + ": " + expected.error().what});
  const engram::CopyLayout& want = expected.value();

  auto fd = io::open_file(copy, io::Access::ReadDirect);
  if (!fd) return Out::err(fd.error());
  auto size = io::file_size(fd.value());
  if (!size) return Out::err(size.error());
  if (size.value() < want.bytes())
    return Out::err({"Engram copy " + copy + ": " + std::to_string(size.value()) + " bytes, the GGUF's tables need " +
                     std::to_string(want.bytes())});
  io::AlignedBuffer header(engram::kCopyHeaderBytes);
  if (auto r = io::pread_exact(fd.value(), header.span(), 0); !r) return Out::err({"Engram copy " + copy + ": " + r.error().what});
  auto layout = engram::parse_copy_header(header.span());
  if (!layout) return Out::err({"Engram copy " + copy + ": " + layout.error().what});
  if (!(layout.value() == want))
    return Out::err({"Engram copy " + copy + " is of other tables: rows " + std::to_string(layout.value().rows(0)) + " and " +
                     std::to_string(layout.value().rows(1)) + ", the GGUF's " + std::to_string(rows[0]) + " and " +
                     std::to_string(rows[1])});

  auto reader = io::UringReader::create(kBulkDepth, true);
  if (!reader) return Out::err(reader.error());
  EngramRows c(Copy{std::move(fd).value(), want, io::AlignedBuffer(kBulkDepth * kSlotBytes), std::move(reader).value()});

  for (int t = 0; t < engram::kLayers; t++) {
    std::vector<uint32_t> ids;
    for (uint64_t id : engram::sample_rows(rows[t], kCopySamples)) ids.push_back(static_cast<uint32_t>(id));
    auto from_gguf = gguf.value().read(t, ids);
    if (!from_gguf) return Out::err(from_gguf.error());
    auto from_copy = c.read(t, ids);
    if (!from_copy) return Out::err({"Engram copy " + copy + ": " + from_copy.error().what});
    for (size_t i = 0; i < ids.size(); i++)
      if (!std::equal(from_gguf.value().begin() + static_cast<ptrdiff_t>(i * kRow),
                      from_gguf.value().begin() + static_cast<ptrdiff_t>((i + 1) * kRow),
                      from_copy.value().begin() + static_cast<ptrdiff_t>(i * kRow)))
        return Out::err({"Engram copy " + copy + ": table " + std::to_string(t) + " row " + std::to_string(ids[i]) +
                         " differs from the GGUF's"});
  }
  return Out::ok(std::move(c));
}

uint64_t EngramRows::rows(int table) const {
  if (table < 0 || table >= engram::kLayers) return 0;
  if (const auto* g = std::get_if<Gguf>(&source_)) return g->tables[table].rows;
  if (const auto* c = std::get_if<Copy>(&source_)) return c->layout.rows(table);
  return 0;
}

Result<std::vector<uint8_t>, io::IoError> EngramRows::read(int table, std::span<const uint32_t> ids) const {
  using Out = Result<std::vector<uint8_t>, io::IoError>;
  std::vector<uint8_t> out(ids.size() * engram::kRowBytes);
  if (auto r = read_rows(table, ids, out); !r) return Out::err(r.error());
  return Out::ok(std::move(out));
}

Result<size_t, io::IoError> EngramRows::read_rows(int table, std::span<const uint32_t> ids, std::span<uint8_t> out) const {
  using Out = Result<size_t, io::IoError>;
  if (out.size() != ids.size() * engram::kRowBytes) return Out::err({"Engram rows: the output is not 264 bytes a row"});
  std::vector<RowRead> reads;
  reads.reserve(ids.size());
  for (size_t i = 0; i < ids.size(); i++) reads.push_back({table, ids[i], out.subspan(i * engram::kRowBytes, engram::kRowBytes)});
  return read_all(reads);
}

Result<std::vector<uint8_t>, io::IoError> EngramRows::read_token(const TokenRows& rows) const {
  return read_tokens(std::span(&rows, 1));
}

Result<std::vector<uint8_t>, io::IoError> EngramRows::read_tokens(std::span<const TokenRows> tokens) const {
  using Out = Result<std::vector<uint8_t>, io::IoError>;
  constexpr size_t kTokenBytes = static_cast<size_t>(engram::kLayers) * engram::kCols * engram::kRowBytes;
  std::vector<uint8_t> out(tokens.size() * kTokenBytes);
  std::vector<RowRead> reads;
  reads.reserve(tokens.size() * engram::kLayers * engram::kCols);
  for (size_t n = 0; n < tokens.size(); n++)
    for (int table = 0; table < engram::kLayers; table++)
      for (int i = 0; i < engram::kCols; i++) {
        const size_t at = n * kTokenBytes + (static_cast<size_t>(table) * engram::kCols + i) * engram::kRowBytes;
        reads.push_back({table, tokens[n][table][i], std::span(out).subspan(at, engram::kRowBytes)});
      }
  if (auto r = read_all(reads); !r) return Out::err(r.error());
  return Out::ok(std::move(out));
}

Result<size_t, io::IoError> EngramRows::read_all(std::span<const RowRead> reads) const {
  using Out = Result<size_t, io::IoError>;
  if (reads.empty()) return Out::ok(0);
  for (const auto& r : reads) {
    if (rows(r.table) == 0) return Out::err({"Engram table " + std::to_string(r.table) + " is not in the opened shards"});
    if (r.id >= rows(r.table)) return Out::err({"Engram row id out of range"});
  }
  if (auto* g = std::get_if<Gguf>(&source_)) {
    // Each row read where it lies in its shard, straight into the caller's buffer.
    const auto make = [&](size_t i) -> io::ReadRequest {
      const Table& t = g->tables[reads[i].table];
      return {g->fds[t.shard].get(), t.offset + uint64_t{reads[i].id} * kRow, std::as_writable_bytes(reads[i].dest), i};
    };
    const auto land = [](const io::Completion& c) -> Result<bool, R> {
      if (c.result != static_cast<int32_t>(kRow)) return Result<bool, R>::err({"Engram row read failed or short"});
      return Result<bool, R>::ok(true);
    };
    return pump(g->reader, reads.size(), make, land);
  }
  auto& c = std::get<Copy>(source_);  // with no source rows() is 0, so a read was refused above
  // Each row's aligned window lands in a free staging slot (the tag), and the row is copied out of it.
  // As many slots as the queue is deep: when the queue has room, a slot is free.
  std::vector<uint32_t> free(kBulkDepth);
  for (uint32_t s = 0; s < kBulkDepth; s++) free[s] = kBulkDepth - 1 - s;
  std::vector<size_t> row_of(kBulkDepth);
  std::vector<engram::CopyWindow> window_of(kBulkDepth);
  const auto make = [&](size_t i) -> io::ReadRequest {
    const uint32_t slot = free.back();
    free.pop_back();
    row_of[slot] = i;
    window_of[slot] = engram::copy_window(c.layout, reads[i].table, reads[i].id);
    return {c.fd.get(), window_of[slot].offset, c.staging.span().subspan(slot * kSlotBytes, window_of[slot].bytes), slot};
  };
  const auto land = [&](const io::Completion& done) -> Result<bool, R> {
    const auto slot = static_cast<uint32_t>(done.tag);
    const engram::CopyWindow& w = window_of[slot];
    if (done.result != static_cast<int32_t>(w.bytes))
      return Result<bool, R>::err({"Engram copy read failed or short at byte " + std::to_string(w.offset)});
    std::memcpy(reads[row_of[slot]].dest.data(), c.staging.data() + slot * kSlotBytes + w.skip, kRow);
    free.push_back(slot);
    return Result<bool, R>::ok(true);
  };
  return pump(c.reader, reads.size(), make, land);
}

Result<engram::CopyLayout, io::IoError> write_engram_copy(const std::vector<std::string>& shards, const std::string& copy,
                                                         const std::function<void(uint64_t, uint64_t)>& progress) {
  using Out = Result<engram::CopyLayout, io::IoError>;
  auto source = EngramRows::open(shards);
  if (!source) return Out::err(source.error());
  const auto& g = std::get<EngramRows::Gguf>(source.value().source_);
  std::array<uint64_t, engram::kLayers> rows{};
  for (int t = 0; t < engram::kLayers; t++) rows[t] = g.tables[t].rows;
  auto layout = copy_layout_for(rows);
  if (!layout) return Out::err(layout.error());
  const engram::CopyLayout& l = layout.value();

  auto out = io::open_file(copy, io::Access::WriteDirect);
  if (!out) return Out::err(out.error());
  auto device = io::is_block_device(out.value());
  if (!device) return Out::err(device.error());
  if (device.value()) {
    auto size = io::file_size(out.value());
    if (!size) return Out::err(size.error());
    if (size.value() < l.bytes())
      return Out::err({copy + ": a partition of " + std::to_string(size.value()) + " bytes, the copy needs " + std::to_string(l.bytes())});
  }
  const auto fail = [&](const std::string& what) { return Out::err({copy + ": " + what}); };

  // Any old header is zeroed first: until the new one is written, this is no copy open_copy takes.
  io::AlignedBuffer header(engram::kCopyHeaderBytes);
  if (auto w = io::pwrite_exact(out.value(), header.span(), 0); !w) return fail(w.error().what);
  if (auto s = io::sync_data(out.value()); !s) return fail(s.error().what);

  const uint64_t total = 2 * (rows[0] + rows[1]) * kRow;
  uint64_t done = 0;
  io::AlignedBuffer chunk(kCopyChunk), back(kCopyChunk);
  std::vector<std::byte> source_bytes(kCopyChunk);

  // The tables, chunk by chunk: read where they lie in the GGUF, written aligned (the last chunk of a
  // table padded with zeros up to the next table, or the copy's end).
  for (int t = 0; t < engram::kLayers; t++) {
    const auto& at = g.tables[t];
    const uint64_t bytes = rows[t] * kRow;
    for (uint64_t off = 0; off < bytes; off += kCopyChunk) {
      const uint64_t n = std::min(kCopyChunk, bytes - off);
      if (auto r = io::pread_exact(g.fds[at.shard], chunk.span().first(n), at.offset + off); !r) return Out::err(r.error());
      std::memset(chunk.data() + n, 0, engram::copy_align_up(n) - n);  // the padding (a table's last chunk only)
      if (auto w = io::pwrite_exact(out.value(), chunk.span().first(engram::copy_align_up(n)), l.offset(t) + off); !w) return fail(w.error().what);
      done += n;
      if (progress) progress(done, total);
    }
  }
  if (auto s = io::sync_data(out.value()); !s) return fail(s.error().what);

  // Every row compared: the copy read back by direct I/O against the GGUF read again.
  auto check = io::open_file(copy, io::Access::ReadDirect);
  if (!check) return Out::err(check.error());
  for (int t = 0; t < engram::kLayers; t++) {
    const auto& at = g.tables[t];
    const uint64_t bytes = rows[t] * kRow;
    for (uint64_t off = 0; off < bytes; off += kCopyChunk) {
      const uint64_t n = std::min(kCopyChunk, bytes - off);
      if (auto r = io::pread_exact(check.value(), back.span().first(engram::copy_align_up(n)), l.offset(t) + off); !r) return fail(r.error().what);
      if (auto r = io::pread_exact(g.fds[at.shard], std::span(source_bytes).first(n), at.offset + off); !r) return Out::err(r.error());
      if (std::memcmp(back.data(), source_bytes.data(), n) != 0)
        return fail("table " + std::to_string(t) + " reads back differently near row " + std::to_string(off / kRow));
      done += n;
      if (progress) progress(done, total);
    }
  }

  const auto encoded = engram::encode_copy_header(l);
  std::memcpy(header.data(), encoded.data(), encoded.size());
  if (auto w = io::pwrite_exact(out.value(), header.span(), 0); !w) return fail(w.error().what);
  if (auto s = io::sync_data(out.value()); !s) return fail(s.error().what);
  return Out::ok(l);
}

}  // namespace ds4n::engine

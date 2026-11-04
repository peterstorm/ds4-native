#pragma once
// Project Maya's engine protocol (pure; serve/server.py's StrataEngine, MIT): what ds4n-serve reads
// on stdin and writes on stdout. In: `GEN <max_new> [key=value ...] <id,id,...>` per request;
// `STOP` (the request in flight ends, "cancel"), `WRAP` (the server is stopping: a reply still
// thinking closes its reasoning) and `QUIT`, read on a thread of their own while a request runs.
// Out: INFO lines and `READY <context> stop` once; per request `T <id>` per written token, `PP <read>
// <total> <ms> <tok/s> <reused>` while the prompt is read, `STAT key=value ...` every 8 tokens, and
// `DONE <written> <prompt> <prompt ms> <decode ms> <finish> <drafts kept> <drafts offered> <reused>
// 0 0` -- or `ERR <message>` for a request that never ran. Keys this engine has no use for (Maya's
// GLM tuning keys, cvec) are skipped, as Maya's own engines skip them.
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "core/reply.hpp"
#include "core/result.hpp"
#include "core/sampler.hpp"

namespace ds4n::wire {

/** One GEN request: its prompt, how many tokens it may write, how it picks them. */
struct Generate {
  int64_t max_new;
  sample::Sampling sampling;
  std::optional<reply::ThinkBudget> think;
  // spec=N, Maya's verify window (the anchor and the drafts): whether this request drafts -- 1 (or 0)
  // is the plain token loop, 2 and up draft (the width stays adaptive); none: as the engine was started.
  std::optional<bool> spec;
  std::vector<int32_t> prompt;
};

struct Stop {};
struct Wrap {};
struct Quit {};

/** A stdin line. */
using Command = std::variant<Generate, Stop, Wrap, Quit>;

struct WireError {
  std::string what;
};

/**
 * A stdin line read for a vocabulary of `vocab` tokens: a GEN with a positive max_new, known keys
 * well formed (unknown ones skipped), every prompt id in the vocabulary, its seed `fallback_seed`
 * unless it names one (the server sends none for "any"); or a control line. GENI (pictures) is an
 * error: this engine has no vision tower.
 */
Result<Command, WireError> parse(std::string_view line, int32_t vocab, uint64_t fallback_seed);

/** The lines a request writes. */
std::string token_line(int32_t token);
std::string ready_line(int64_t context);

/** Prompt progress: `read` of `total` tokens after `ms`, `reused` of them from the Session. */
struct Progress {
  size_t read, total, reused;
  double ms;
};
std::string progress_line(const Progress& p);

/** A request's outcome (DONE). */
struct Done {
  int64_t written;
  size_t prompt;
  double prompt_ms, decode_ms;
  reply::Finish finish;
  int64_t drafts_kept, drafts_offered;
  size_t reused;
};
std::string done_line(const Done& d);

/** The Monitor's figures over a reply so far (STAT). */
struct Stat {
  int64_t written;
  double ms;               // since its first token
  double vram_hit;         // routed Experts found in VRAM Slots, of all needed
  double drafts;           // drafted tokens kept of those verified (negative: none drafted)
};
std::string stat_line(const Stat& s);

}  // namespace ds4n::wire

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
//
// Several requests at once (Maya's multi-request mode, as its server speaks it at b3fa99e:
// `_generate_multi`, `_route`; serve/fake_multi_engine.py is its reference engine): an engine with
// more than one Lane says `READY <context> stop multi=<lanes>`, and the server then sends up to that
// many requests at a time, each GEN naming itself with a key `rid=<id>`. A request that named a rid
// has it after the verb on each of its own lines -- `T <rid> <id>`, `PP <rid> ...`, `DONE <rid> ...`,
// `ERR <rid> <message>` -- the rest of the line as without one; `STOP <rid>` ends that request
// alone. STAT, INFO, NOTE and WRAP stay the engine's, untagged. A GEN that finds every Lane busy
// gets `ERR <rid> all <lanes> conversation states are busy`. With one Lane the server sends no rid
// and every line is as before.
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "core/lane.hpp"
#include "core/reply.hpp"
#include "core/result.hpp"
#include "core/sampler.hpp"

namespace ds4n::wire {

/** A request's id, chosen by the server (`rid=<id>` on its GEN). */
struct Rid {
  uint64_t id;
  friend bool operator==(const Rid&, const Rid&) = default;
};

/** One GEN request: its prompt, how many tokens it may write, how it picks them. */
struct Generate {
  int64_t max_new;
  sample::Sampling sampling;
  std::optional<reply::ThinkBudget> think;
  // spec=N, Maya's verify window (the anchor and the drafts): whether this request drafts -- 1 (or 0)
  // is the plain token loop, 2 and up draft (the width stays adaptive); none: as the engine was started.
  std::optional<bool> spec;
  std::vector<int32_t> prompt;
  std::optional<Rid> rid;  // what its lines are tagged with, when it named one
};

/** End a request, "cancel": the one named, or with no rid the one in flight. */
struct Stop {
  std::optional<Rid> rid;
};
struct Wrap {};
struct Quit {};

/** A stdin line. */
using Command = std::variant<Generate, Stop, Wrap, Quit>;

/** A line that is no command; `rid`: of the request it was for, when the line still named one (its ERR is that request's). */
struct WireError {
  std::string what;
  std::optional<Rid> rid = std::nullopt;
};

/**
 * A stdin line read for a vocabulary of `vocab` tokens: a GEN with a positive max_new, known keys
 * well formed (unknown ones skipped), every prompt id in the vocabulary, its seed `fallback_seed`
 * unless it names one (the server sends none for "any"); or a control line. GENI (pictures) is an
 * error: this engine has no vision tower.
 */
Result<Command, WireError> parse(std::string_view line, int32_t vocab, uint64_t fallback_seed);

/** The engine is ready: its context, that STOP is honoured, and -- with more than one Lane -- how many requests it runs at once. */
std::string ready_line(int64_t context, lane::LaneCount lanes);

/** The lines a request writes, tagged with `rid` when it named one. */
std::string token_line(std::optional<Rid> rid, int32_t token);

/** Prompt progress: `read` of `total` tokens after `ms`, `reused` of them from the Session. */
struct Progress {
  size_t read, total, reused;
  double ms;
};
std::string progress_line(std::optional<Rid> rid, const Progress& p);

/** A request's outcome (DONE). */
struct Done {
  int64_t written;
  size_t prompt;
  double prompt_ms, decode_ms;
  reply::Finish finish;
  int64_t drafts_kept, drafts_offered;
  size_t reused;
};
std::string done_line(std::optional<Rid> rid, const Done& d);

/** A request that could not run, or a line that was no command (ERR): `message` on one line. */
std::string error_line(std::optional<Rid> rid, std::string_view message);
/** The ERR of a GEN that found every one of `lanes` Lanes busy, in the server's words for it. */
std::string busy_line(Rid rid, lane::LaneCount lanes);

/** The Monitor's figures over a reply so far (STAT): the engine's line, never tagged. */
struct Stat {
  int64_t written;
  double ms;               // since its first token
  double vram_hit;         // routed Experts found in VRAM Slots, of all needed
  double drafts;           // drafted tokens kept of those verified (negative: none drafted)
};
std::string stat_line(const Stat& s);

}  // namespace ds4n::wire

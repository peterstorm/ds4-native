#pragma once
// Imperative shell: GEN requests (core/wire) on the whole model, several at once when the decoder
// has several Lanes, from any of the conversations the engine keeps.
//
// A request is admitted into a Lane (core/rewind plans which, and where its prompt resumes: the
// Lane's KV in place, another Lane's prefix borrowed, or a KV checkpoint taken back, with that
// state's rings or a Rewind point's restored over them). Then the engine runs pass after pass
// (step), each what core/shared_round composes from what the Lanes' requests do next:
//   - a prompt's long unread tail as a prompt sweep, the other Lanes pausing;
//   - one Lane writing alone, its step as an engine of one Lane runs it: a plain Round, or with
//     DSpark a drafted block verified in one Round;
//   - otherwise a Round of rows -- a Shared Round when of several Lanes -- every Lane at work with
//     rows in it: a writing Lane's next token, a reading Lane's next few prompt tokens.
// A prompt is split by a Rewind point where it leaves tokens other conversations share (a subagent's
// system prompt), and a Rewind point is taken at its end. The Reply follows: picks (core/sampler)
// under the Reply's rules (core/reply). A verify Round's rows are picked in turn, each at its own
// draw index, its drafted tokens kept while the pick at their row equals them: the Reply is the one
// plain steps would write, sampled or greedy.
//
// Mix invariance: a row computes the same in any Round, and a request's picks draw by its own
// count of written tokens, so its prompt-end logits and its Reply are the same bits whatever ran
// beside it.
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/engram.hpp"
#include "core/lane.hpp"
#include "core/reply.hpp"
#include "core/rewind.hpp"
#include "core/speculation.hpp"
#include "core/wire.hpp"
#include "engine/full_model.cuh"

namespace ds4n::engine {

struct ConversationConfig {
  // Snapshot buffers, each a WindowSnapshot (46 MB pinned per GPU). More than the decoder has Lanes
  // keeps every Lane's last prompt's end from the other Lanes' points (rewind::next_buffer).
  uint32_t rewind_points = 8;
  uint64_t checkpoint_budget = 6ull << 30;  // host bytes of KV checkpoints (0: none kept)
  rewind::Settings rewind{};               // when a Lane's Session is set aside, where a prompt is split
  size_t sweep_min = 64;                   // a prompt's unread tokens go as a prompt sweep from this many, else as rows of Rounds
  std::vector<int32_t> stops;              // end-of-turn tokens
};

/**
 * What a request reports as it runs, on the thread that steps the engine: each written token, the
 * prompt's progress, the Monitor's figures every 8 tokens, what happened to the kept conversations
 * (`note`), its prompt's next-token logits once it is read, and its end -- `done` with the DONE
 * figures, or `failed` (non-finite logits: its Lane's Session is forgotten, the others go on).
 */
struct ReplyEvents {
  std::function<void(int32_t)> token = [](int32_t) {};
  std::function<void(const wire::Progress&)> progress = [](const wire::Progress&) {};
  std::function<void(const wire::Stat&)> stat = [](const wire::Stat&) {};
  std::function<void(const std::string&)> note = [](const std::string&) {};
  std::function<void(std::span<const float>)> prompt_end = [](std::span<const float>) {};
  std::function<void(const wire::Done&)> done = [](const wire::Done&) {};
  std::function<void(const std::string&)> failed = [](const std::string&) {};
};

/** Why a GEN was not admitted. */
struct Refused {
  std::string what;
  bool busy = false;  // every Lane holds a request: more were sent at once than the engine has Lanes
};

class Conversation {
 public:
  /** On `model`'s decoder, with a Session for each of its Lanes. */
  Conversation(FullModel& model, ConversationConfig config);

  /**
   * Take one GEN (a prompt of 1 to kMaxCompressed - 1 tokens) into a Lane, its plan carried out
   * (a set-aside, a take-back or a borrow, the rings restored): the Lane names the request until its
   * `done` or `failed` event. Throws on a CUDA or I/O failure, as step() does.
   */
  Result<lane::LaneId, Refused> admit(wire::Generate g, ReplyEvents events);

  /**
   * The engine's next pass for the requests in flight (see the top of this file), first ending those
   * stop() named; their events fire from here. False when there was nothing to run. Throws on a CUDA
   * or I/O failure, after which every Lane's Session is forgotten and no request is in flight (the
   * checkpoints and Rewind points, host copies, stay).
   */
  bool step();

  /** Whether no request is in flight. */
  bool idle() const;
  /** End the request in `lane` ("cancel") before the next pass; no request there: nothing. */
  void stop(lane::LaneId lane);
  /** The request in `lane` closes its reasoning block, if one is open, with its next token. */
  void wrap(lane::LaneId lane);
  lane::LaneCount lanes() const { return sessions_.lanes(); }

  /**
   * One GEN from its admission to its end, stepping the engine (and whatever else is in flight):
   * the DONE figures. Throws std::invalid_argument when it is refused, std::runtime_error for
   * non-finite logits ("non-finite"), and as step() does.
   */
  wire::Done run(const wire::Generate& g, const ReplyEvents& events);

  /** Capture the Round shapes' graphs and DSpark's (a throwaway prompt), then forget everything kept. */
  void warm_up();

  /**
   * The GPUs' caches were used by someone else (no request in flight here): the next prompts read
   * from a kept state or position 0.
   */
  void forget_live();

 private:
  /** One Lane's rows of a Round: `tokens` at positions p0, p0 + 1, ... after History `h`. */
  struct Feed {
    lane::LaneId lane;
    int64_t p0;
    std::span<const int32_t> tokens;
    engram::History h;
  };
  /** A Round's or sweep's rows: every row's logits ([rows][vocab]; a sweep: its last row's) and the History after each row. */
  struct Fed {
    std::vector<float> logits;
    std::vector<engram::History> after;
    std::span<const float> row(size_t r) const;
  };
  /** A Reply under way. */
  struct Writing {
    reply::Reply r;
    std::optional<int32_t> forced;  // its next token, when the Reply's rules name it
    std::vector<int32_t> seen;      // what the penalties count: the prompt, then every written token
    int64_t kept = 0, offered = 0;  // drafted tokens
    int64_t p = 0;                  // where the Anchor is fed
    int32_t anchor = 0;             // the last written token
    double prompt_ms = 0.0;
    double t1_ms = 0.0;             // when the Reply began
    ResidencyStats before;
  };
  /** A request in its Lane: reading its prompt, then (`writing`) writing its Reply. */
  struct Request {
    wire::Generate g;
    ReplyEvents ev;
    size_t reused = 0;            // the prompt tokens its plan resumed
    size_t at = 0;                // prompt tokens in its Lane's caches
    std::optional<size_t> split;  // a Rewind point still to take there
    engram::History history;      // after its Lane's committed tokens
    double t0_ms = 0.0;
    std::string resumed, marked;  // for the prompt's log line: what its plan resumed from, where it took a Rewind point
    bool stop = false, wrap = false;
    std::optional<Writing> writing;
  };

  Request& request(lane::LaneId lane) { return *requests_.at(static_cast<size_t>(lane.index())); }
  reply::Limits limits(const Request& q) const { return {q.g.max_new, config_.stops, q.g.think}; }
  /** One Round of `feeds` (in Lane order), a Shared Round when of several Lanes. */
  Fed round(std::span<const Feed> feeds);
  Fed round(lane::LaneId lane, int64_t p0, std::span<const int32_t> tokens, engram::History h);
  /** A prompt sweep of `tokens` from p0 in `lane` after History `h`. */
  Fed sweep(lane::LaneId lane, int64_t p0, std::span<const int32_t> tokens, engram::History h);
  /** A Rewind point of `lane`'s rings, here and now. */
  void mark(lane::LaneId lane, bool prompt_end);
  /** Carry out `p`'s set-aside, its take-back or borrow, and its restore: then its Lane's Session stands at p.from. */
  void carry_out(const rewind::Plan& p, const ReplyEvents& ev);
  /** Drop the checkpoints `ids` (their stored caches freed). */
  void drop(const std::vector<uint64_t>& ids);

  // A pass, by what core/shared_round composed.
  void sweep_pass(lane::LaneId lane);
  void solo_pass(lane::LaneId lane);
  void rows_pass(std::span<const model::RoundRow> rows);
  /** `lane`'s request read `n` more prompt tokens: History `after` them, `last` the logits after the last. */
  void read(lane::LaneId lane, size_t n, const engram::History& after, std::span<const float> last);
  /** Its prompt is read: the Rewind point at its end, then the Reply's first token. */
  void begin_reply(lane::LaneId lane, std::span<const float> prompt_logits);
  /** The row of `lane`'s Anchor ran alone for it (`row` its logits): the Anchor committed, the next token written. */
  void step_written(lane::LaneId lane, std::span<const float> row, const engram::History& after);
  /** Write `tok`, to be fed at position `at`: how the Reply ended, when it ended with it. */
  std::optional<reply::Finish> write(Request& q, int32_t tok, int64_t at);
  /** The next token after logits `row`: the forced one, or the request's pick at its draw index. */
  int32_t next_token(const Request& q, std::span<const float> row) const;
  void stat(const Request& q) const;
  /** The request in `lane` ends: its DONE figures to `done`, its Lane free for the next. */
  void finish(lane::LaneId lane, reply::Finish how);
  /** What `pass` does for `lane`'s request, which on non-finite logits fails alone: `failed`, its Lane's Session forgotten. */
  void guarded(lane::LaneId lane, const std::function<void()>& pass);
  /** The caches' state is unknown: no request is in flight, no Lane has a Session. */
  void abandon();

  FullModel& m_;
  ConversationConfig config_;
  rewind::Sessions sessions_;
  std::vector<std::optional<Request>> requests_;  // by Lane: the request it holds
  std::map<uint64_t, StoredCaches> stored_;       // the checkpoints' caches, by id
  uint64_t next_id_ = 1;
  std::vector<WindowSnapshot> snapshots_;
  // DSpark's verify widths, measured across requests: greedy replies and sampled ones apart (the
  // confidence head predicts greedy acceptance; a sampled pick keeps fewer drafts).
  spec::Policy greedy_policy_ = spec::start(), sampled_policy_ = spec::start();
};

}  // namespace ds4n::engine

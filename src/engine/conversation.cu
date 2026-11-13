#include "engine/conversation.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include "core/sampler.hpp"
#include "core/shared_round.hpp"

namespace ds4n::engine {

namespace {

using Clock = std::chrono::steady_clock;

/** A prompt reading again at least this many tokens some state's KV held is named in a note: rings it lacked. */
constexpr int64_t kHeldNote = 2048;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
double now_ms() { return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count(); }

/** Rows' embeddings ([n][5120]) and Engram row ids, with the History after each row. */
struct Inputs {
  std::vector<float> embeddings;
  std::vector<EngramTokenRows> rows;
  std::vector<engram::History> after;
};

/** `tokens`, hashed in order after `h`, added to `in`. */
void add(Inputs& in, const FullModel& m, std::span<const int32_t> tokens, engram::History h) {
  in.embeddings.reserve(in.embeddings.size() + tokens.size() * model::kDim);
  for (const int32_t t : tokens) {
    const std::vector<float> e = m.embedding(t);
    in.embeddings.insert(in.embeddings.end(), e.begin(), e.end());
    auto step = engram::hash_token(m.layout, h, static_cast<uint32_t>(t), true);
    if (!step) throw std::runtime_error("Engram: " + step.error().what);
    h = step.value().next;
    in.rows.push_back(step.value().rows);
    in.after.push_back(h);
  }
}

/** A transition of the Sessions the shell drives correctly by construction (an error is a bug here). */
rewind::Sessions committed(rewind::Sessions s, lane::LaneId lane, std::span<const int32_t> fed, size_t kept) {
  auto next = rewind::wrote(std::move(s), lane, fed, kept);
  if (!next) throw std::logic_error("Conversation: " + next.error().what);
  return std::move(next).value();
}

std::string gb(uint64_t bytes) {
  char b[32];
  std::snprintf(b, sizeof b, "%.2f GB", static_cast<double>(bytes) / 1e9);
  return b;
}

std::vector<float> last_row(const std::vector<float>& all) {
  return {all.end() - model::kVocab, all.end()};
}

/**
 * Non-finite logits where a request picks its next token: that request's trouble alone (a row
 * computes the same whatever runs beside it), and the server's cue to retry it from token 0
 * (serve/server.py, on "non-finite").
 */
struct NonFinite : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/** The rings the decoder puts back with a checkpoint a plan takes: one overload a kind, so none goes unmapped. */
StoredRings stored(rewind::OwnRings) { return StoredRings::Own; }
StoredRings stored(rewind::TurnRings) { return StoredRings::Turn; }
StoredRings stored(rewind::PointRings) { return StoredRings::None; }  // the point's follow
StoredRings stored_rings(const rewind::FromCheckpoint& from) {
  return std::visit([](const auto& rings) { return stored(rings); }, from.rings);
}

/** What a plan resumes from, for the prompt's log line. */
std::string resumed_note(const rewind::Plan& p) {
  std::string note = rewind::taken(p) ? " (a checkpoint's)" : rewind::lender(p) ? " (another Lane's)" : "";
  const auto* from = std::get_if<rewind::FromCheckpoint>(&p.source);
  if (rewind::point_of(p)) note += " (a Rewind point's rings)";
  else if (from && std::holds_alternative<rewind::TurnRings>(from->rings)) note += " (its last prompt's end's rings)";
  return note;
}

}  // namespace

std::span<const float> Conversation::Fed::row(size_t r) const {
  return std::span<const float>(logits).subspan(r * model::kVocab, model::kVocab);
}

Conversation::Conversation(FullModel& model, ConversationConfig config)
    : m_(model),
      config_(std::move(config)),
      sessions_(m_.decoder->lanes()),
      requests_(static_cast<size_t>(m_.decoder->lanes().value())) {
  for (uint32_t i = 0; i < config_.rewind_points; i++) snapshots_.push_back(m_.decoder->window_snapshot());
}

Conversation::Fed Conversation::round(std::span<const Feed> feeds) {
  Inputs in;
  std::vector<model::RoundRow> rows;
  for (const Feed& f : feeds) {
    add(in, m_, f.tokens, f.h);
    for (size_t i = 0; i < f.tokens.size(); i++) rows.push_back({f.lane, f.p0 + static_cast<int64_t>(i)});
  }
  auto raw = m_.rows.read_tokens(in.rows);
  if (!raw) throw std::runtime_error("Engram rows: " + raw.error().what);
  m_.decoder->begin_round(rows, in.embeddings, raw.value());
  m_.decoder->decode_round();
  return {m_.decoder->logits(), std::move(in.after)};
}

Conversation::Fed Conversation::round(lane::LaneId lane, int64_t p0, std::span<const int32_t> tokens, engram::History h) {
  const Feed one[] = {{lane, p0, tokens, h}};
  return round(one);
}

Conversation::Fed Conversation::sweep(lane::LaneId lane, int64_t p0, std::span<const int32_t> tokens, engram::History h) {
  Inputs in;
  add(in, m_, tokens, h);
  const ExpertPlacer place = [&](int layer, std::span<const int32_t> experts) { return m_.residency->place(layer, experts); };
  const EngramReader read = [&](int table, std::span<const uint32_t> ids, std::span<uint8_t> out) {
    if (auto r = m_.rows.read_rows(table, ids, out); !r) throw std::runtime_error(r.error().what);
  };
  m_.residency->check();
  m_.decoder->prefill(p0, in.embeddings, in.rows, read, place, lane);
  return {last_row(m_.decoder->logits()), {in.after.back()}};
}

void Conversation::mark(lane::LaneId lane, bool prompt_end) {
  if (snapshots_.empty()) return;
  const uint32_t b = rewind::next_buffer(sessions_, lane, static_cast<uint32_t>(snapshots_.size()));
  m_.decoder->save(snapshots_[b], lane);
  sessions_ = rewind::marked(std::move(sessions_), lane, b, prompt_end);
}

void Conversation::drop(const std::vector<uint64_t>& ids) {
  for (const uint64_t id : ids) stored_.erase(id);
}

void Conversation::carry_out(const rewind::Plan& p, const ReplyEvents& ev) {
  if (p.set_aside) {  // the Lane's Session, whole, into host memory: another conversation takes its caches
    const int64_t n = sessions_.written(p.lane).committed;
    // In the middle of a turn, its last prompt's end's rings go with it: its next step re-renders the reply.
    const std::optional<uint32_t> turn = sessions_.turn_point(p.lane);
    const uint64_t bytes = m_.decoder->checkpoint_bytes(n) + (turn ? 2 * snapshots_[*turn].bytes() : 0);  // both GPUs'
    if (bytes <= config_.checkpoint_budget) {
      const auto t0 = Clock::now();
      const uint64_t id = next_id_++;
      StoredCaches c = m_.decoder->checkpoint(n, p.lane);
      if (turn) m_.decoder->keep_turn(c, snapshots_[*turn]);
      stored_.emplace(id, std::move(c));
      auto [next, superseded] = rewind::set_aside(std::move(sessions_), id, bytes, p);
      sessions_ = std::move(next);
      drop(superseded);
      ev.note("set aside a " + std::to_string(n) + "-token conversation (" + gb(bytes) + ", " +
              std::to_string(static_cast<int>(ms_since(t0))) + " ms); " + std::to_string(sessions_.checkpoints().size()) + " kept");
    }
  }
  if (const auto* from = std::get_if<rewind::FromCheckpoint>(&p.source)) {
    const auto t0 = Clock::now();
    const StoredCaches& c = stored_.at(from->id);
    m_.decoder->take_back(c, stored_rings(*from), p.lane);
    ev.note("took back a " + std::to_string(c.committed) + "-token conversation (" + gb(c.bytes()) + ", " +
            std::to_string(static_cast<int>(ms_since(t0))) + " ms)");
  }
  // Below the resume the lender's rows are final, whatever it goes on to write: it may be mid-request.
  if (const std::optional<lane::LaneId> from = rewind::lender(p)) m_.decoder->lend(*from, p.lane, p.from);
  if (const std::optional<uint32_t> b = rewind::point_of(p)) m_.decoder->restore(snapshots_[*b], p.lane);
  sessions_ = rewind::resumed(std::move(sessions_), p);
  auto [kept, dropped] = rewind::within(std::move(sessions_), config_.checkpoint_budget);
  sessions_ = std::move(kept);
  drop(dropped);
}

Result<lane::LaneId, Refused> Conversation::admit(wire::Generate g, ReplyEvents ev) {
  using R = Result<lane::LaneId, Refused>;
  if (static_cast<int64_t>(g.prompt.size()) >= kMaxCompressed)
    return R::err({"the prompt (" + std::to_string(g.prompt.size()) + " tokens) leaves no room in the context of " +
                   std::to_string(kMaxCompressed)});
  const double t0 = now_ms();
  const auto planned = rewind::plan(sessions_, g.prompt, config_.rewind);
  if (!planned) return R::err({rewind::what(planned.error()), planned.error() == rewind::PlanError::Busy});
  const rewind::Plan& p = planned.value();
  try {
    const int64_t held = rewind::kv_held(sessions_, g.prompt);
    if (held - p.from >= kHeldNote)
      ev.note("the KV held the prompt's first " + std::to_string(held) + " tokens, rings resume it from " + std::to_string(p.from));
    carry_out(p, ev);  // the Lane is busy from here until the request ends, however it ends
    const size_t from = static_cast<size_t>(p.from);
    auto h = engram::history_at(m_.layout, g.prompt, from);
    if (!h) throw std::runtime_error("Engram: " + h.error().what);
    ev.progress({from, g.prompt.size(), from, 0.0});
    Request q{std::move(g), std::move(ev), from, from, std::nullopt, h.value(), t0, resumed_note(p)};
    // Where the prompt leaves tokens other conversations share, a Rewind point: the next prompt that
    // shares them (a subagent's, under the same system prompt) resumes there.
    if (p.split) {
      q.split = static_cast<size_t>(*p.split);
      q.marked = " (a Rewind point at " + std::to_string(*p.split) + ")";
    }
    const bool split_here = q.split == from;
    slot(p.lane) = std::move(q);
    if (split_here) {  // nothing to read before it (--split-min 0): the point is the resume itself
      mark(p.lane, false);
      request(p.lane).split.reset();
    }
    return R::ok(p.lane);
  } catch (...) {
    abandon();
    throw;
  }
}

bool Conversation::idle() const {
  return std::none_of(requests_.begin(), requests_.end(), [](const auto& q) { return q.has_value(); });
}

void Conversation::stop(lane::LaneId lane) {
  if (auto& q = slot(lane)) q->stop = true;
}

void Conversation::wrap(lane::LaneId lane) {
  if (auto& q = slot(lane)) q->wrap = true;
}

void Conversation::abandon() {
  for (auto& q : requests_) q.reset();
  forget_live();
  for (const lane::LaneId id : lanes().ids()) sessions_ = rewind::released(std::move(sessions_), id);
}

void Conversation::forget_live() {
  for (const lane::LaneId id : lanes().ids()) sessions_ = rewind::forgot(std::move(sessions_), id);
}

bool Conversation::step() {
  try {
    const std::vector<lane::LaneId> ids = lanes().ids();
    for (const lane::LaneId id : ids)
      if (const auto& q = slot(id); q && q->stop) finish(id, reply::Finish::Cancel);

    std::vector<shared::LaneState> states;
    for (const auto& q : requests_) {
      if (!q) states.emplace_back(shared::Idle{});
      else if (q->writing) states.emplace_back(shared::Writing{q->writing->p});
      else states.emplace_back(shared::Reading{static_cast<int64_t>(q->at),
                                               static_cast<int64_t>(q->split.value_or(q->g.prompt.size()) - q->at)});
    }
    const size_t most = static_cast<size_t>(std::numeric_limits<int64_t>::max());
    const auto next = shared::compose(states, static_cast<int64_t>(std::min(config_.sweep_min, most)));
    if (!next) throw std::logic_error(std::string("Conversation: ") + shared::what(next.error()));

    struct Run {  // true: a pass ran
      Conversation& c;
      bool operator()(const shared::Rest&) const { return false; }
      bool operator()(const shared::Sweep& s) const { c.sweep_pass(s.lane); return true; }
      bool operator()(const shared::Solo& s) const { c.solo_pass(s.lane); return true; }
      bool operator()(const shared::Rows& r) const { c.rows_pass(r.rows); return true; }
    };
    return std::visit(Run{*this}, next.value());
  } catch (...) {
    abandon();  // the next prompts read from a kept state or position 0
    throw;
  }
}

void Conversation::guarded(lane::LaneId lane, const std::function<void()>& pass) {
  try {
    pass();
  } catch (const NonFinite& e) {
    auto& q = slot(lane);
    const auto failed = std::move(q->ev.failed);
    q.reset();
    sessions_ = rewind::released(rewind::forgot(std::move(sessions_), lane), lane);
    failed(e.what());
  }
}

void Conversation::sweep_pass(lane::LaneId lane) {
  const Request& q = request(lane);
  const size_t n = q.split.value_or(q.g.prompt.size()) - q.at;
  const Fed f = sweep(lane, static_cast<int64_t>(q.at), std::span(q.g.prompt).subspan(q.at, n), q.history);
  guarded(lane, [&] { read(lane, n, f.after.back(), f.logits); });
}

void Conversation::rows_pass(std::span<const model::RoundRow> rows) {
  // The Round's rows by Lane: a reading Lane's next prompt tokens, a writing Lane's Anchor.
  struct Part {
    lane::LaneId lane;
    size_t first, n;
  };
  std::vector<Part> parts;
  for (size_t r = 0; r < rows.size(); r++) {
    if (parts.empty() || parts.back().lane != rows[r].lane) parts.push_back({rows[r].lane, r, 0});
    parts.back().n++;
  }
  std::vector<Feed> feeds;
  for (const Part& part : parts) {
    const Request& q = request(part.lane);
    const std::span<const int32_t> tokens =
        q.writing ? std::span<const int32_t>(&q.writing->anchor, 1) : std::span(q.g.prompt).subspan(q.at, part.n);
    feeds.push_back({part.lane, rows[part.first].pos, tokens, q.history});
  }
  const Fed f = round(feeds);
  for (const Part& part : parts) {
    const size_t last = part.first + part.n - 1;
    guarded(part.lane, [&] {
      if (request(part.lane).writing) step_written(part.lane, f.row(last), f.after[last]);
      else read(part.lane, part.n, f.after[last], f.row(last));
    });
  }
}

void Conversation::read(lane::LaneId lane, size_t n, const engram::History& after, std::span<const float> last) {
  Request& q = request(lane);
  sessions_ = committed(std::move(sessions_), lane, std::span(q.g.prompt).subspan(q.at, n), n);
  q.history = after;
  q.at += n;
  q.ev.progress({q.at, q.g.prompt.size(), q.reused, now_ms() - q.t0_ms});
  if (q.split == q.at) {
    mark(lane, false);
    q.split.reset();
  }
  if (q.at == q.g.prompt.size()) begin_reply(lane, last);
}

void Conversation::begin_reply(lane::LaneId lane, std::span<const float> prompt_logits) {
  Request& q = request(lane);
  const size_t total = q.g.prompt.size();
  mark(lane, true);  // the next turn of a chat shares this prompt but its reply
  const double prompt_ms = now_ms() - q.t0_ms;
  std::fprintf(stderr, "ds4n-serve: prompt %zu tokens = %zu reused%s + %zu read%s in %.0f ms (%.1f tok/s), Lane %d\n", total,
               q.reused, q.resumed.c_str(), total - q.reused, q.marked.c_str(), prompt_ms,
               1000.0 * static_cast<double>(total - q.reused) / std::max(prompt_ms, 1e-9), lane.index());
  q.ev.prompt_end(prompt_logits);
  q.writing = Writing{.r = reply::start(limits(q)),
                      .seen = q.g.prompt,
                      .p = static_cast<int64_t>(total),
                      .prompt_ms = prompt_ms,
                      .t1_ms = now_ms(),
                      .before = m_.residency->stats()};
  Writing& w = *q.writing;
  w.anchor = next_token(q, prompt_logits);
  if (const auto end = write(q, w.anchor, w.p)) finish(lane, *end);
}

int32_t Conversation::next_token(const Request& q, std::span<const float> row) const {
  const Writing& w = *q.writing;
  if (w.forced) return *w.forced;
  const size_t last_n = static_cast<size_t>(q.g.sampling.settings().penalties.last_n);
  const std::span<const int32_t> recent = std::span(w.seen).last(std::min(w.seen.size(), last_n));
  const auto t = sample::pick(row, q.g.sampling, recent, static_cast<uint64_t>(w.r.written));
  if (!t) throw NonFinite("non-finite logits (" + t.error().what + ")");
  return t.value();
}

void Conversation::stat(const Request& q) const {
  const Writing& w = *q.writing;
  const ResidencyStats now = m_.residency->stats();
  const double hits = static_cast<double>(now.hits - w.before.hits);
  const double needed =
      hits + static_cast<double>((now.ram_hits - w.before.ram_hits) + (now.store_reads - w.before.store_reads));
  q.ev.stat({w.r.written, now_ms() - w.t1_ms, needed > 0 ? hits / needed : 1.0,
             w.offered > 0 ? static_cast<double>(w.kept) / static_cast<double>(w.offered) : -1.0});
}

std::optional<reply::Finish> Conversation::write(Request& q, int32_t tok, int64_t at) {
  Writing& w = *q.writing;
  q.ev.token(tok);
  w.seen.push_back(tok);
  const reply::Next n = reply::after(w.r, limits(q), tok, q.wrap, at < kMaxCompressed);
  w.r = n.reply;
  w.forced = n.forced;
  if (w.r.written % 8 == 0) stat(q);
  return n.finish;
}

void Conversation::step_written(lane::LaneId lane, std::span<const float> row, const engram::History& after) {
  Request& q = request(lane);
  Writing& w = *q.writing;
  const int32_t fed[] = {w.anchor};
  sessions_ = committed(std::move(sessions_), lane, fed, 1);
  q.history = after;
  w.p += 1;
  w.anchor = next_token(q, row);
  if (const auto end = write(q, w.anchor, w.p)) finish(lane, *end);
}

void Conversation::solo_pass(lane::LaneId lane) {
  guarded(lane, [&] {
    Request& q = request(lane);
    Writing& w = *q.writing;
    const bool drafting = m_.dspark && q.g.spec.value_or(true);
    spec::Policy& policy = q.g.sampling.greedy_pick() ? greedy_policy_ : sampled_policy_;
    if (!(drafting && !w.forced && spec::drafts(policy) && w.p + model::kVerifyRows <= kMaxCompressed)) {
      const auto t_round = Clock::now();
      const Fed f = round(lane, w.p, std::span<const int32_t>(&w.anchor, 1), q.history);
      if (drafting) policy = spec::after_plain(policy, ms_since(t_round));
      step_written(lane, f.row(0), f.after[0]);
      return;
    }
    // DSpark: a block after the Anchor, the Anchor and some of the block verified in one Round.
    const auto t_draft = Clock::now();
    const DraftBlock block = m_.decoder->draft(w.p, w.anchor, m_.embedding(w.anchor), lane);
    const double draft_ms = ms_since(t_draft);
    const int64_t left = std::max<int64_t>(1, q.g.max_new - w.r.written);
    const int n = static_cast<int>(std::min<int64_t>({spec::choose_rows(policy, block.confidence), left, kMaxCompressed - w.p}));
    std::vector<int32_t> fed = {w.anchor};
    fed.insert(fed.end(), block.tokens.begin(), block.tokens.begin() + (n - 1));
    const auto t_round = Clock::now();
    const Fed f = round(lane, w.p, fed, q.history);
    const double round_ms = ms_since(t_round);
    int i = 0;
    int32_t tok = next_token(q, f.row(0));
    std::optional<reply::Finish> end = write(q, tok, w.p + 1);
    while (!end && i + 1 < n && tok == fed[static_cast<size_t>(i) + 1]) {  // row i + 1 read the token picked here
      i++;
      tok = next_token(q, f.row(static_cast<size_t>(i)));
      end = write(q, tok, w.p + i + 1);
    }
    sessions_ = committed(std::move(sessions_), lane, fed, static_cast<size_t>(i) + 1);  // rows 0 .. i are committed
    q.history = f.after[static_cast<size_t>(i)];
    w.p += i + 1;
    w.anchor = tok;
    w.kept += i;
    w.offered += n - 1;
    policy = spec::after_draft(policy, block.confidence, n, i + 1, draft_ms, round_ms);
    if (end) finish(lane, *end);
  });
}

void Conversation::finish(lane::LaneId lane, reply::Finish how) {
  auto& held = slot(lane);
  const Request& q = *held;
  wire::Done d{0, q.g.prompt.size(), now_ms() - q.t0_ms, 0.0, how, 0, 0, q.reused};  // ended while reading its prompt
  if (q.writing) {
    const Writing& w = *q.writing;
    stat(q);
    const double decode_ms = now_ms() - w.t1_ms;
    const ResidencyStats after = m_.residency->stats();
    std::fprintf(stderr,
                 "ds4n-serve: %lld written in %.0f ms (%.2f tok/s), %s; drafts %lld of %lld kept; Loads %llu RAM tier + %llu "
                 "store (%.0f ms waited), %llu copied down, Lane %d\n",
                 static_cast<long long>(w.r.written), decode_ms, 1000.0 * static_cast<double>(w.r.written) / std::max(decode_ms, 1e-9),
                 reply::name(how), static_cast<long long>(w.kept), static_cast<long long>(w.offered),
                 static_cast<unsigned long long>(after.ram_hits - w.before.ram_hits),
                 static_cast<unsigned long long>(after.store_reads - w.before.store_reads), after.exposed_ms - w.before.exposed_ms,
                 static_cast<unsigned long long>(after.demoted - w.before.demoted), lane.index());
    d = {w.r.written, q.g.prompt.size(), w.prompt_ms, decode_ms, how, w.kept, w.offered, q.reused};
  }
  const auto done = std::move(held->ev.done);
  held.reset();
  sessions_ = rewind::released(std::move(sessions_), lane);
  done(d);
}

wire::Done Conversation::run(const wire::Generate& g, const ReplyEvents& events) {
  std::optional<wire::Done> done;
  std::optional<std::string> failed;
  ReplyEvents ev = events;
  ev.done = [&](const wire::Done& d) {
    events.done(d);
    done = d;
  };
  ev.failed = [&](const std::string& what) {
    events.failed(what);
    failed = what;
  };
  if (const auto lane = admit(g, std::move(ev)); !lane) throw std::invalid_argument(lane.error().what);
  while (!done && !failed)
    if (!step()) throw std::logic_error("Conversation: a request in flight and nothing to run");
  if (failed) throw std::runtime_error(*failed);
  return *done;
}

void Conversation::warm_up() {
  // A throwaway prompt through a sweep, then a Round of every width at both parities, then a draft.
  std::vector<int32_t> prompt;
  for (int i = 0; i < 600; i++) prompt.push_back(i == 0 ? 0 : (i * 7919 + 13) % 120000);
  run({1, sample::Sampling::greedy(), std::nullopt, false, prompt}, {});
  const std::vector<lane::LaneId> ids = lanes().ids();
  const std::span<const int32_t> some = std::span(prompt).subspan(1);
  int64_t p = static_cast<int64_t>(prompt.size());
  for (int rows = 1; rows <= model::kMaxRows; rows++)
    for (int parity = 0; parity < 2; parity++) {
      if (p % 2 != parity) p++;
      round(ids[0], p, some.first(static_cast<size_t>(rows)), {});
      p += rows;
    }
  if (m_.dspark) m_.decoder->draft(p, prompt[1], m_.embedding(prompt[1]), ids[0]);
  // Shared Rounds of one row a Lane, for every 2 .. all of the Lanes and every mix of rows that do and
  // do not complete a ratio-2 group (model::group_mask): what Lanes writing beside each other run. Each
  // Lane first holds positions 0 .. 3, so its row at 2 or 3 reads rows that were written. A reading
  // Lane's several rows beside them capture theirs on first use.
  for (size_t i = 1; i < ids.size(); i++) round(ids[i], 0, some.first(4), {});
  for (size_t k = 2; k <= ids.size(); k++)
    for (unsigned mix = 0; mix < (1u << k); mix++) {
      std::vector<Feed> feeds;
      for (size_t i = 0; i < k; i++) feeds.push_back({ids[i], 2 + ((mix >> i) & 1), some.first(1), {}});
      round(feeds);
    }
  m_.decoder->sync();
  const RoundGraphStats graphs = m_.decoder->round_graph_stats();
  std::fprintf(stderr, "ds4n-serve: Round graphs: %zu kept after the warm-up, %.0f MiB a GPU (%llu captured, %llu dropped for the budget)\n",
               graphs.kept, static_cast<double>(graphs.bytes) / (1 << 20), static_cast<unsigned long long>(graphs.captured),
               static_cast<unsigned long long>(graphs.dropped));
  sessions_ = rewind::Sessions(lanes());  // nothing the throwaway prompt left is kept
  stored_.clear();
}

}  // namespace ds4n::engine

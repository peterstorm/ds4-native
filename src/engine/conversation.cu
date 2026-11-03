#include "engine/conversation.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>

#include "core/reply.hpp"
#include "core/sampler.hpp"

namespace ds4n::engine {

namespace {

using Clock = std::chrono::steady_clock;

/** A prompt reading again at least this many tokens some state's KV held is named in a note: rings it lacked. */
constexpr int64_t kHeldNote = 2048;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
double now_ms() { return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count(); }

/** The tokens' embeddings ([n][5120]) and Engram row ids, hashed in order after `h`, with the History after each. */
struct Inputs {
  std::vector<float> embeddings;
  std::vector<EngramTokenRows> rows;
  std::vector<engram::History> after;
};

Inputs inputs(const FullModel& m, std::span<const int32_t> tokens, engram::History h) {
  Inputs in;
  in.embeddings.reserve(tokens.size() * model::kDim);
  for (const int32_t t : tokens) {
    const std::vector<float> e = m.embedding(t);
    in.embeddings.insert(in.embeddings.end(), e.begin(), e.end());
    auto step = engram::hash_token(m.layout, h, static_cast<uint32_t>(t), true);
    if (!step) throw std::runtime_error("Engram: " + step.error().what);
    h = step.value().next;
    in.rows.push_back(step.value().rows);
    in.after.push_back(h);
  }
  return in;
}

/** A transition of the Sessions the shell drives correctly by construction (an error is a bug here). */
rewind::Sessions committed(rewind::Sessions s, std::span<const int32_t> fed, size_t kept) {
  auto next = rewind::wrote(std::move(s), fed, kept);
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

/** A token is non-finite trouble for the server (serve/server.py retries the request from token 0 on "non-finite"). */
[[noreturn]] void non_finite(const std::string& what) { throw std::runtime_error("non-finite logits (" + what + ")"); }

/** The rings the decoder puts back with a checkpoint a plan takes: one overload a kind, so none goes unmapped. */
StoredRings stored(rewind::OwnRings) { return StoredRings::Own; }
StoredRings stored(rewind::TurnRings) { return StoredRings::Turn; }
StoredRings stored(rewind::PointRings) { return StoredRings::None; }  // the point's follow
StoredRings stored_rings(const rewind::FromCheckpoint& from) {
  return std::visit([](const auto& rings) { return stored(rings); }, from.rings);
}

/** Whose rings a plan resumes from, for the prompt's log line. */
const char* rings_note(const rewind::Plan& p) {
  if (rewind::point_of(p)) return " (a Rewind point's rings)";
  const auto* from = std::get_if<rewind::FromCheckpoint>(&p.source);
  return from && std::holds_alternative<rewind::TurnRings>(from->rings) ? " (its last prompt's end's rings)" : "";
}

}  // namespace

Conversation::Conversation(FullModel& model, ConversationConfig config) : m_(model), config_(std::move(config)) {
  for (uint32_t i = 0; i < config_.rewind_points; i++) snapshots_.push_back(m_.decoder->window_snapshot());
}

Conversation::Fed Conversation::round(int64_t p0, std::span<const int32_t> tokens, engram::History h) {
  Inputs in = inputs(m_, tokens, h);
  auto raw = m_.rows.read_tokens(in.rows);
  if (!raw) throw std::runtime_error("Engram rows: " + raw.error().what);
  m_.decoder->begin_round(p0, in.embeddings, raw.value());
  m_.decoder->decode_round();
  return {m_.decoder->logits(), std::move(in.after)};
}

Conversation::Fed Conversation::sweep(int64_t p0, std::span<const int32_t> tokens, engram::History h) {
  Inputs in = inputs(m_, tokens, h);
  const ExpertPlacer place = [&](int layer, std::span<const int32_t> experts) { return m_.residency->place(layer, experts); };
  const EngramReader read = [&](int table, std::span<const uint32_t> ids, std::span<uint8_t> out) {
    if (auto r = m_.rows.read_rows(table, ids, out); !r) throw std::runtime_error(r.error().what);
  };
  m_.residency->check();
  m_.decoder->prefill(p0, in.embeddings, in.rows, read, place);
  return {last_row(m_.decoder->logits()), {in.after.back()}};
}

bool Conversation::read_prompt(std::span<const int32_t> prompt, size_t from, size_t end, const ReplyEvents& ev,
                               double t0_ms) {
  const size_t total = prompt.size();
  if (end - from >= config_.sweep_min) {
    Fed f = sweep(static_cast<int64_t>(from), prompt.subspan(from, end - from), history_);
    sessions_ = committed(std::move(sessions_), prompt.subspan(from, end - from), end - from);
    history_ = f.after.back();
    if (end == total) prompt_logits_ = std::move(f.logits);
    return true;
  }
  for (size_t at = from; at < end;) {
    if (ev.stop && ev.stop->load()) return false;
    const size_t n = std::min<size_t>(model::kMaxRows, end - at);
    Fed f = round(static_cast<int64_t>(at), prompt.subspan(at, n), history_);
    sessions_ = committed(std::move(sessions_), prompt.subspan(at, n), n);
    history_ = f.after.back();
    at += n;
    if (at == total) prompt_logits_ = last_row(f.logits);
    ev.progress({at, total, from, now_ms() - t0_ms});
  }
  return true;
}

void Conversation::mark(bool prompt_end) {
  if (snapshots_.empty()) return;
  const uint32_t b = rewind::next_buffer(sessions_, static_cast<uint32_t>(snapshots_.size()));
  m_.decoder->save(snapshots_[b]);
  sessions_ = rewind::marked(std::move(sessions_), b, prompt_end);
}

void Conversation::drop(const std::vector<uint64_t>& ids) {
  for (const uint64_t id : ids) stored_.erase(id);
}

void Conversation::carry_out(const rewind::Plan& p, const ReplyEvents& ev) {
  if (p.set_aside) {  // the live Session, whole, into host memory: another conversation takes the GPUs
    const int64_t n = sessions_.written().committed;
    // In the middle of a turn, its last prompt's end's rings go with it: its next step re-renders the reply.
    const std::optional<uint32_t> turn = sessions_.turn_point();
    const uint64_t bytes = m_.decoder->checkpoint_bytes(n) + (turn ? 2 * snapshots_[*turn].bytes() : 0);  // both GPUs'
    if (bytes <= config_.checkpoint_budget) {
      const auto t0 = Clock::now();
      const uint64_t id = next_id_++;
      StoredCaches c = m_.decoder->checkpoint(n);
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
    m_.decoder->take_back(c, stored_rings(*from));
    ev.note("took back a " + std::to_string(c.committed) + "-token conversation (" + gb(c.bytes()) + ", " +
            std::to_string(static_cast<int>(ms_since(t0))) + " ms)");
  }
  if (const std::optional<uint32_t> b = rewind::point_of(p)) m_.decoder->restore(snapshots_[*b]);
  sessions_ = rewind::resumed(std::move(sessions_), p);
  auto [kept, dropped] = rewind::within(std::move(sessions_), config_.checkpoint_budget);
  sessions_ = std::move(kept);
  drop(dropped);
}

wire::Done Conversation::run(const wire::Generate& g, const ReplyEvents& ev) {
  if (g.prompt.empty() || static_cast<int64_t>(g.prompt.size()) >= kMaxCompressed)
    throw std::invalid_argument("a prompt of 1 to " + std::to_string(kMaxCompressed - 1) + " tokens");
  try {
    const double t0 = now_ms();
    const rewind::Plan p = rewind::plan(sessions_, g.prompt, config_.rewind).value();
    const int64_t held = rewind::kv_held(sessions_, g.prompt);
    if (held - p.from >= kHeldNote)
      ev.note("the KV held the prompt's first " + std::to_string(held) + " tokens, rings resume it from " + std::to_string(p.from));
    carry_out(p, ev);
    const size_t from = static_cast<size_t>(p.from), total = g.prompt.size();
    auto h = engram::history_at(m_.layout, g.prompt, from);
    if (!h) throw std::runtime_error("Engram: " + h.error().what);
    history_ = h.value();
    ev.progress({from, total, from, 0.0});
    // Where the prompt leaves tokens other conversations share, a Rewind point: the next prompt that
    // shares them (a subagent's, under the same system prompt) resumes there.
    size_t at = from;
    if (p.split) {
      const size_t split = static_cast<size_t>(*p.split);
      if (!read_prompt(g.prompt, at, split, ev, t0)) return {0, total, now_ms() - t0, 0.0, reply::Finish::Cancel, 0, 0, from};
      mark(false);
      at = split;
    }
    if (!read_prompt(g.prompt, at, total, ev, t0)) return {0, total, now_ms() - t0, 0.0, reply::Finish::Cancel, 0, 0, from};
    ev.progress({total, total, from, now_ms() - t0});
    mark(true);  // the next turn of a chat shares this prompt but its reply
    std::fprintf(stderr, "ds4n-serve: prompt %zu tokens = %zu reused%s%s + %zu read%s in %.0f ms (%.1f tok/s)\n", total, from,
                 rewind::taken(p) ? " (a checkpoint's)" : "", rings_note(p), total - from,
                 p.split ? (" (a Rewind point at " + std::to_string(*p.split) + ")").c_str() : "", now_ms() - t0,
                 1000.0 * static_cast<double>(total - from) / std::max(now_ms() - t0, 1e-9));
    return reply(g, ev, from, now_ms() - t0);
  } catch (...) {
    forget_live();  // the caches' state is unknown: the next prompt reads from a kept state or 0
    throw;
  }
}

wire::Done Conversation::reply(const wire::Generate& g, const ReplyEvents& ev, size_t reused, double prompt_ms) {
  const auto t1 = Clock::now();
  const ResidencyStats before = m_.residency->stats();
  const reply::Limits limits{g.max_new, config_.stops, g.think};
  reply::Reply r = reply::start(limits);
  std::optional<int32_t> forced;
  std::optional<reply::Finish> finish;
  std::vector<int32_t> seen = g.prompt;  // what the penalties count: the prompt, then every written token
  int64_t kept = 0, offered = 0;
  const bool drafting = m_.dspark && g.spec.value_or(true);
  spec::Policy& policy = g.sampling.greedy_pick() ? greedy_policy_ : sampled_policy_;

  auto pick = [&](std::span<const float> row) {
    const size_t last_n = static_cast<size_t>(g.sampling.settings().penalties.last_n);
    const std::span<const int32_t> recent = std::span(seen).last(std::min(seen.size(), last_n));
    auto t = sample::pick(row, g.sampling, recent, static_cast<uint64_t>(r.written));
    if (!t) non_finite(t.error().what);
    return t.value();
  };
  auto stat = [&] {
    const ResidencyStats now = m_.residency->stats();
    const double hits = static_cast<double>(now.hits - before.hits);
    const double needed = hits + static_cast<double>((now.ram_hits - before.ram_hits) + (now.store_reads - before.store_reads));
    ev.stat({r.written, ms_since(t1), needed > 0 ? hits / needed : 1.0,
             offered > 0 ? static_cast<double>(kept) / static_cast<double>(offered) : -1.0});
  };
  // Writes `tok`, to be fed at position `at`: whether the reply goes on.
  auto write = [&](int32_t tok, int64_t at) {
    ev.token(tok);
    seen.push_back(tok);
    const reply::Next n = reply::after(r, limits, tok, ev.wrap && ev.wrap->load(), at < kMaxCompressed);
    r = n.reply;
    forced = n.forced;
    finish = n.finish;
    if (r.written % 8 == 0) stat();
    return !finish;
  };
  auto next_of = [&](std::span<const float> row) { return forced ? *forced : pick(row); };

  int64_t p = static_cast<int64_t>(g.prompt.size());  // where the anchor is fed
  int32_t anchor = next_of(prompt_logits_);
  bool going = write(anchor, p);
  while (going) {
    if (ev.stop && ev.stop->load()) {
      finish = reply::Finish::Cancel;
      break;
    }
    const bool draft = drafting && !forced && spec::drafts(policy) && p + model::kVerifyRows <= kMaxCompressed;
    if (!draft) {
      const auto t_round = Clock::now();
      const std::vector<int32_t> fed = {anchor};
      Fed f = round(p, fed, history_);
      if (drafting) policy = spec::after_plain(policy, ms_since(t_round));
      sessions_ = committed(std::move(sessions_), fed, 1);
      history_ = f.after.back();
      p += 1;
      anchor = next_of(f.logits);
      going = write(anchor, p);
      continue;
    }
    // DSpark: a block after the anchor, the anchor and some of the block verified in one Round.
    const auto t_draft = Clock::now();
    const DraftBlock block = m_.decoder->draft(p, anchor, m_.embedding(anchor));
    const double draft_ms = ms_since(t_draft);
    const int64_t left = std::max<int64_t>(1, limits.max_new - r.written);
    const int n = static_cast<int>(std::min<int64_t>({spec::choose_rows(policy, block.confidence), left, kMaxCompressed - p}));
    std::vector<int32_t> fed = {anchor};
    fed.insert(fed.end(), block.tokens.begin(), block.tokens.begin() + (n - 1));
    const auto t_round = Clock::now();
    Fed f = round(p, fed, history_);
    const double round_ms = ms_since(t_round);
    auto row = [&](int i) { return std::span<const float>(f.logits).subspan(static_cast<size_t>(i) * model::kVocab, model::kVocab); };
    int i = 0;
    int32_t tok = next_of(row(0));
    going = write(tok, p + 1);
    while (going && i + 1 < n && tok == fed[static_cast<size_t>(i) + 1]) {  // row i + 1 read the token picked here
      i++;
      tok = next_of(row(i));
      going = write(tok, p + i + 1);
    }
    sessions_ = committed(std::move(sessions_), fed, static_cast<size_t>(i) + 1);  // rows 0 .. i are committed
    history_ = f.after[static_cast<size_t>(i)];
    p += i + 1;
    anchor = tok;
    kept += i;
    offered += n - 1;
    policy = spec::after_draft(policy, block.confidence, n, i + 1, draft_ms, round_ms);
  }
  stat();
  const double decode_ms = ms_since(t1);
  std::fprintf(stderr, "ds4n-serve: %lld written in %.0f ms (%.2f tok/s), %s; drafts %lld of %lld kept\n",
               static_cast<long long>(r.written), decode_ms, 1000.0 * static_cast<double>(r.written) / std::max(decode_ms, 1e-9),
               reply::name(finish.value_or(reply::Finish::Cancel)), static_cast<long long>(kept), static_cast<long long>(offered));
  return {r.written, g.prompt.size(), prompt_ms, decode_ms, finish.value_or(reply::Finish::Cancel), kept, offered, reused};
}

void Conversation::warm_up() {
  // A throwaway prompt through a sweep, then a Round of every width at both parities, then a draft.
  std::vector<int32_t> prompt;
  for (int i = 0; i < 600; i++) prompt.push_back(i == 0 ? 0 : (i * 7919 + 13) % 120000);
  wire::Generate g{1, sample::Sampling::greedy(), std::nullopt, false, prompt};
  run(g, {});
  int64_t p = static_cast<int64_t>(prompt.size());
  for (int rows = 1; rows <= model::kMaxRows; rows++)
    for (int parity = 0; parity < 2; parity++) {
      if (p % 2 != parity) p++;
      round(p, std::span(prompt).subspan(1, static_cast<size_t>(rows)), history_);
      p += rows;
    }
  if (m_.dspark) m_.decoder->draft(p, prompt[1], m_.embedding(prompt[1]));
  m_.decoder->sync();
  sessions_ = {};  // nothing the throwaway prompt left is kept
  stored_.clear();
}

}  // namespace ds4n::engine

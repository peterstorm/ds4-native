#include "engine/glm53/conversation.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <optional>
#include <stdexcept>

#include "core/reply.hpp"
#include "core/sampler.hpp"

namespace ds4n::engine::glm53 {

namespace {

namespace g53 = ds4n::glm53;

double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/** A Decoder failure: the engine is unsound from here (the caller starts it again). */
template <class T>
T must(Result<T, std::string> r, const char* what) {
  if (!r) throw std::runtime_error(std::string(what) + ": " + r.error());
  return std::move(r).value();
}

/** Logits that are not finite: the request fails alone. */
struct NonFinite {
  std::string what;
};

}  // namespace

Result<Conversation, std::string> Conversation::create(Decoder& decoder, ConversationConfig config) {
  using R = Result<Conversation, std::string>;
  if (decoder.length() != 0) return R::err("the Decoder's caches must start empty");
  auto s = g53::Sessions::create({decoder.capacity(), config.checkpoint_bytes, decoder.position_bytes(), config.min_lost});
  if (!s) return R::err(g53::what(s.error()));
  return R::ok(Conversation(decoder, std::move(config), std::move(s).value()));
}

void Conversation::resume(const g53::Resume& r, const ReplyEvents& ev) {
  if (r.save) {
    held_.emplace(*r.save, must(decoder_->save(decoder_->length()), "set the conversation aside"));
    ev.note("set aside a conversation of " + std::to_string(decoder_->length()) + " tokens");
  }
  if (r.restore) {
    must(decoder_->restore(held_.at(*r.restore), static_cast<int64_t>(r.keep)), "restore a conversation");
    ev.note("resumed a conversation set aside, at " + std::to_string(r.keep) + " tokens");
  } else {
    must(decoder_->rewind(static_cast<int64_t>(r.keep)), "rewind");
  }
  for (uint64_t id : r.freed) held_.erase(id);
}

Outcome Conversation::reply(const wire::Generate& g, const ReplyEvents& ev, const std::function<Interrupts()>& poll) {
  const double t0 = now_ms();
  auto planned = sessions_.plan(g.prompt);
  if (!planned) return Refused{g53::what(planned.error())};
  g53::Planned p = std::move(planned).value();
  resume(p.resume, ev);
  sessions_ = std::move(p.after);
  const size_t total = g.prompt.size(), reused = p.resume.keep;
  ev.progress({reused, total, reused, 0.0});

  // The prompt's rest, one sweep.
  const std::span<const int32_t> rest = std::span(g.prompt).subspan(reused);
  std::vector<float> logits = must(decoder_->prefill(rest), "the prompt");
  sessions_ = sessions_.fed(rest).value();  // within the capacity: plan() refused a prompt that is not
  const double prompt_ms = now_ms() - t0;
  ev.progress({total, total, reused, prompt_ms});
  std::fprintf(stderr, "ds4n-glm53-serve: prompt %zu tokens = %zu reused + %zu read in %.0f ms (%.1f tok/s)\n", total,
               reused, rest.size(), prompt_ms, 1000.0 * static_cast<double>(rest.size()) / std::max(prompt_ms, 1e-9));

  // A STOP that came while the prompt was read ends the request before it writes.
  const Interrupts during = poll();
  if (during.stop) return wire::Done{0, total, prompt_ms, 0.0, reply::Finish::Cancel, 0, 0, reused};

  // The reply.
  const reply::Limits limits{g.max_new, config_.stops, g.think};
  reply::Reply r = reply::start(limits);
  std::vector<int32_t> seen = g.prompt;  // what the penalties count: the prompt, then the reply
  std::optional<int32_t> forced;
  bool wrap = during.wrap;
  const double t1 = now_ms();
  const DecoderStats before = decoder_->stats();
  auto stat = [&] {
    const DecoderStats now = decoder_->stats();
    const auto picks = static_cast<double>(now.picks - before.picks);
    const auto misses = static_cast<double>(now.misses - before.misses);
    ev.stat({r.written, now_ms() - t1, picks > 0 ? (picks - misses) / picks : 1.0, -1.0});
  };
  auto pick = [&](std::span<const float> row) {
    if (ev.logits) ev.logits(row);
    if (forced) return *forced;
    const auto last_n = static_cast<size_t>(g.sampling.settings().penalties.last_n);
    const std::span<const int32_t> recent = std::span(seen).last(std::min(seen.size(), last_n));
    const auto t = sample::pick(row, g.sampling, recent, static_cast<uint64_t>(r.written));
    if (!t) throw NonFinite{"non-finite logits (" + t.error().what + ")"};
    return t.value();
  };
  std::optional<reply::Finish> finish;
  try {
    for (int32_t tok = pick(logits);;) {
      ev.token(tok);
      seen.push_back(tok);
      const Interrupts in = poll();
      wrap = wrap || in.wrap;
      const reply::Next next = reply::after(r, limits, tok, wrap, decoder_->length() < decoder_->capacity());
      r = next.reply;
      forced = next.forced;
      if (r.written % 8 == 0) stat();
      finish = next.finish;
      if (!finish && in.stop) finish = reply::Finish::Cancel;
      if (finish) break;
      const int32_t fed[] = {tok};
      logits = must(decoder_->round(fed), "a reply token");
      sessions_ = sessions_.fed(fed).value();  // reply::after ended the reply before the context ran out
      tok = pick(logits);
    }
  } catch (const NonFinite& e) {
    sessions_ = sessions_.reset();
    return Failed{e.what};
  }
  stat();
  const double decode_ms = now_ms() - t1;
  const DecoderStats after = decoder_->stats();
  std::fprintf(stderr,
               "ds4n-glm53-serve: %lld written in %.0f ms (%.2f tok/s), %s; misses %llu of %llu (%llu RAM tier, %llu store, "
               "%.0f ms waited)\n",
               static_cast<long long>(r.written), decode_ms, 1000.0 * static_cast<double>(r.written) / std::max(decode_ms, 1e-9),
               reply::name(*finish), static_cast<unsigned long long>(after.misses - before.misses),
               static_cast<unsigned long long>(after.picks - before.picks),
               static_cast<unsigned long long>(after.ram_hits - before.ram_hits),
               static_cast<unsigned long long>(after.store_reads - before.store_reads),
               1000.0 * (after.store_s - before.store_s));
  return wire::Done{r.written, total, prompt_ms, decode_ms, *finish, 0, 0, reused};
}

}  // namespace ds4n::engine::glm53

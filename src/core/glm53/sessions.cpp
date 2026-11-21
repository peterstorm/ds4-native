#include "core/glm53/sessions.hpp"

#include <algorithm>
#include <utility>

namespace ds4n::glm53 {

namespace {

/** How many leading tokens `a` and `b` share. */
size_t shared_prefix(std::span<const int32_t> a, std::span<const int32_t> b) {
  return static_cast<size_t>(std::ranges::mismatch(a, b).in1 - a.begin());
}

/** Whether `a` is where `b` starts (an empty `a` included). */
bool starts(std::span<const int32_t> a, std::span<const int32_t> b) {
  return a.size() <= b.size() && shared_prefix(a, b) == a.size();
}

}  // namespace

const char* what(SessionError e) {
  switch (e) {
    case SessionError::BadLimits: return "the context and a position's size must be positive";
    case SessionError::EmptyPrompt: return "the prompt is empty";
    case SessionError::PromptTooLong: return "the prompt leaves no room for a reply in the context";
    case SessionError::CacheFull: return "the context is full";
  }
  return "a session error";
}

Result<Sessions, SessionError> Sessions::create(const SessionLimits& limits) {
  using R = Result<Sessions, SessionError>;
  if (limits.capacity < 1 || limits.position_bytes < 1 || limits.min_lost < 1) return R::err(SessionError::BadLimits);
  return R::ok(Sessions(limits));
}

uint64_t Sessions::held_bytes() const {
  uint64_t tokens = 0;
  for (const Checkpoint& c : checkpoints_) tokens += c.tokens.size();
  return tokens * limits_.position_bytes;
}

Result<Planned, SessionError> Sessions::plan(std::span<const int32_t> prompt) const {
  using R = Result<Planned, SessionError>;
  if (prompt.empty()) return R::err(SessionError::EmptyPrompt);
  if (static_cast<int64_t>(prompt.size()) >= limits_.capacity) return R::err(SessionError::PromptTooLong);

  // The Checkpoint sharing the most, the most recently set aside on a tie; it is resumed from only
  // when it shares strictly more than the live Session.
  const size_t live_shared = shared_prefix(live_, prompt);
  size_t best = live_shared;
  std::optional<size_t> from;
  for (size_t i = 0; i < checkpoints_.size(); i++) {
    const size_t s = shared_prefix(checkpoints_[i].tokens, prompt);
    if (s > best || (from && s == best)) {
      best = s;
      from = i;
    }
  }

  Sessions after = *this;
  Resume r;
  r.keep = std::min(best, prompt.size() - 1);
  // Once the prompt is read the caches hold it: the live tokens past its share are dropped. Set the live
  // Session aside when that is min_lost or more, unless a Checkpoint already holds it all.
  const bool lost = live_.size() - live_shared >= limits_.min_lost;
  const bool held = std::ranges::any_of(checkpoints_, [&](const Checkpoint& c) { return starts(live_, c.tokens); });
  const uint64_t live_bytes = live_.size() * limits_.position_bytes;
  if (lost && !held && live_bytes <= limits_.budget_bytes) {
    r.save = after.next_id_++;
    // A Checkpoint the live Session starts with holds nothing more.
    std::erase_if(after.checkpoints_, [&](const Checkpoint& c) {
      if (!starts(c.tokens, live_)) return false;
      r.freed.push_back(c.id);
      return true;
    });
    after.checkpoints_.push_back({*r.save, live_});
  }
  if (from) {  // moves into the caches: its host copy goes once copied
    r.restore = checkpoints_[*from].id;
    r.freed.push_back(*r.restore);
    std::erase_if(after.checkpoints_, [&](const Checkpoint& c) { return c.id == *r.restore; });
  }
  while (after.held_bytes() > limits_.budget_bytes) {  // the least recently set aside first; never the one just saved
    r.freed.push_back(after.checkpoints_.front().id);
    after.checkpoints_.erase(after.checkpoints_.begin());
  }
  after.live_.assign(prompt.begin(), prompt.begin() + static_cast<std::ptrdiff_t>(r.keep));
  return R::ok(Planned{std::move(after), std::move(r)});
}

Result<Sessions, SessionError> Sessions::fed(std::span<const int32_t> tokens) const {
  using R = Result<Sessions, SessionError>;
  if (static_cast<int64_t>(live_.size() + tokens.size()) > limits_.capacity) return R::err(SessionError::CacheFull);
  Sessions after = *this;
  after.live_.insert(after.live_.end(), tokens.begin(), tokens.end());
  return R::ok(std::move(after));
}

Sessions Sessions::reset() const {
  Sessions after = *this;
  after.live_.clear();
  return after;
}

}  // namespace ds4n::glm53

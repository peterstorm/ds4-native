#include "core/session_mix.hpp"

#include <algorithm>
#include <cmath>

namespace ds4n::mix {
namespace {

constexpr uint32_t kLayerKeys = 384;  // an Expert key is layer * 384 + expert

void add_distinct(std::vector<cache::ExpertKey>& to, cache::ExpertKey k) {
  if (std::find(to.begin(), to.end(), k) == to.end()) to.push_back(k);
}

/** A Session's share of a Round: its tokens [first, first + rows) run, `commit` of them kept. */
struct Share {
  size_t session, first, rows, commit;
};

/** The Round's step at `layer`: the union of its rows' needs, and of their near-misses none of it needs. */
cache::Access round_step(std::span<const Traffic> sessions, std::span<const Share> shares, uint32_t layer) {
  cache::Access a;
  for (const Share& s : shares)
    for (size_t t = s.first; t < s.first + s.rows; t++)
      for (const cache::ExpertKey k : sessions[s.session].step(t, layer).needed) add_distinct(a.needed, k);
  for (const Share& s : shares)
    for (size_t t = s.first; t < s.first + s.rows; t++)
      for (const cache::ExpertKey k : sessions[s.session].step(t, layer).near)
        if (std::find(a.needed.begin(), a.needed.end(), k) == a.needed.end()) add_distinct(a.near, k);
  return a;
}

}  // namespace

Result<Traffic, MixError> Traffic::of(std::span<const cache::Access> steps) {
  using R = Result<Traffic, MixError>;
  if (steps.empty() || steps.size() % kLayerSteps != 0) return R::err({"traffic is whole tokens of 40 layer steps"});
  for (size_t i = 0; i < steps.size(); i++) {
    const auto& needed = steps[i].needed;
    const auto layer = static_cast<cache::ExpertKey>(i % kLayerSteps);
    if (needed.empty() || std::any_of(needed.begin(), needed.end(), [&](cache::ExpertKey k) { return k / kLayerKeys != layer; }))
      return R::err({"traffic's layer steps are out of layer order"});
  }
  return R::ok(Traffic(steps));
}

std::optional<Verify> Verify::make(uint32_t rows, double kept) {
  if (rows < 1 || rows > kRoundRows || !(kept >= 1.0 && kept <= static_cast<double>(rows))) return std::nullopt;
  return Verify(rows, kept);
}

Result<double, MixError> kept_tokens(std::span<const double> accept, uint32_t rows) {
  using R = Result<double, MixError>;
  if (rows < 1 || rows > accept.size() + 1) return R::err({"a verify width from 1 to one past the acceptances given"});
  if (std::any_of(accept.begin(), accept.end(), [](double p) { return !(p >= 0.0 && p <= 1.0); }))
    return R::err({"acceptances are probabilities"});
  double kept = 1.0, reach = 1.0;
  for (uint32_t i = 0; i + 1 < rows; i++) {
    reach *= accept[i];
    kept += reach;
  }
  return R::ok(kept);
}

Result<Mix, MixError> mix(std::span<const Traffic> sessions, Sharing sharing, Verify verify) {
  using R = Result<Mix, MixError>;
  const size_t n = sessions.size();
  if (n == 0) return R::err({"no Sessions to mix"});
  const Turns* turns = std::get_if<Turns>(&sharing);
  if (turns && turns->tokens == 0) return R::err({"a turn of at least one token"});
  if (!turns && n * verify.rows() > kRoundRows) return R::err({"a shared Round holds at most 8 rows"});

  std::vector<size_t> next(n, 0);     // per Session: its next token
  std::vector<double> credit(n, 0.0);  // per Session: the tokens its Rounds kept, before rounding down
  auto done = [&](size_t s) { return next[s] >= sessions[s].tokens(); };
  size_t owner = 0;    // Turns: the Session whose turn it is
  uint64_t taken = 0;  // ... and the tokens it committed in it
  Mix m;
  for (;;) {
    std::vector<size_t> in;
    if (turns) {
      for (size_t tried = 0; tried <= n && (done(owner) || taken >= turns->tokens); tried++) {
        owner = (owner + 1) % n;
        taken = 0;
      }
      if (!done(owner)) in.push_back(owner);
    } else {
      for (size_t s = 0; s < n; s++)
        if (!done(s)) in.push_back(s);
    }
    if (in.empty()) break;

    std::vector<Share> shares;
    Round round{static_cast<uint32_t>(in.size()), 0, 0};
    for (const size_t s : in) {
      const size_t left = sessions[s].tokens() - next[s];
      const double after = credit[s] + verify.kept();
      const auto kept = static_cast<size_t>(std::floor(after) - std::floor(credit[s]));  // >= 1: Verify keeps >= 1
      credit[s] = after;
      const Share share{s, next[s], std::min<size_t>(verify.rows(), left), std::min(kept, left)};
      shares.push_back(share);
      round.rows += static_cast<uint32_t>(share.rows);
      round.tokens += static_cast<uint32_t>(share.commit);
    }
    for (uint32_t layer = 0; layer < kLayerSteps; layer++) m.steps.push_back(round_step(sessions, shares, layer));
    for (const Share& s : shares) next[s.session] += s.commit;
    if (turns) taken += round.tokens;
    m.rounds.push_back(round);
  }
  return R::ok(std::move(m));
}

}  // namespace ds4n::mix

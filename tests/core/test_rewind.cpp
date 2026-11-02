#include <doctest/doctest.h>

#include <map>
#include <random>
#include <vector>

#include "core/rewind.hpp"

using namespace ds4n;
using namespace ds4n::rewind;

namespace {

uint64_t extend(uint64_t h, int32_t token) { return (h ^ static_cast<uint64_t>(token + 1)) * 0x100000001B3ull + 0x9E37ull; }

/** Hashes of the prefixes of `tokens` ending at each position. */
std::vector<uint64_t> prefixes(std::span<const int32_t> tokens) {
  std::vector<uint64_t> out;
  uint64_t h = 0;
  for (const int32_t t : tokens) out.push_back(h = extend(h, t));
  return out;
}

/**
 * The rings as the decoder fills them, independently of the rule: each slot remembers the position
 * whose write it holds and the tokens that write saw (a hash of the sequence up to there), so a slot
 * an old timeline or a discarded verify row left is told apart from the Session's own.
 */
struct Rings {
  struct Slot {
    int64_t position = -1;
    uint64_t prefix = 0;
    bool operator==(const Slot&) const = default;
  };
  std::vector<Slot> window = std::vector<Slot>(model::kRing), pool = std::vector<Slot>(model::kPoolRing);

  void write(int64_t p, uint64_t prefix) {
    window[static_cast<size_t>(p % model::kRing)] = {p, prefix};
    pool[static_cast<size_t>(p % model::kPoolRing)] = {p, prefix};
  }
  /**
   * Whether a row at p reads only the Session's own writes (`prefix[q]` its hash up to q): the
   * window and DSpark's context below p, and an odd p's pool slot.
   */
  bool exact_at(int64_t p, const std::vector<uint64_t>& prefix) const {
    auto own = [&](const Slot& s, int64_t q) { return s == Slot{q, prefix[static_cast<size_t>(q)]}; };
    for (int64_t q = std::max<int64_t>(0, p - model::kDraftContext); q < p; q++)
      if (!own(window[static_cast<size_t>(q % model::kRing)], q)) return false;
    return p % 2 == 0 || own(pool[static_cast<size_t>((p - 1) % model::kPoolRing)], p - 1);
  }
};

/** A Session with the rings the decoder would hold for it, the snapshot buffers, and its prefix hashes. */
struct Model {
  Session s;
  Rings rings;
  std::map<uint32_t, Rings> buffers;
  std::vector<uint64_t> hashes;  // of s.tokens()

  /** A Round or sweep at the committed end: the rings take every fed row, the Session its kept ones. */
  void feed(std::span<const int32_t> fed, size_t kept) {
    uint64_t h = hashes.empty() ? 0 : hashes.back();
    const int64_t p = s.written().committed;
    for (size_t i = 0; i < fed.size(); i++) {
      rings.write(p + static_cast<int64_t>(i), h = extend(h, fed[i]));
      if (i < kept) hashes.push_back(h);
    }
    s = wrote(std::move(s), fed, kept).value();
  }
  void mark(uint32_t buffers_held) {
    const uint32_t b = next_buffer(s, buffers_held);
    buffers[b] = rings;
    s = marked(std::move(s), b);
  }
  /** Plans `prompt`, checks the resume reads only the Session's own writes, applies it; returns it. */
  Resume resume(std::span<const int32_t> prompt) {
    const Resume r = plan(s, prompt).value();
    REQUIRE(r.from < static_cast<int64_t>(prompt.size()));
    const Rings& from = r.point ? buffers.at(r.point->buffer) : rings;
    REQUIRE(from.exact_at(r.from, prefixes(prompt)));
    if (r.point) rings = from;
    s = resumed(std::move(s), r);
    hashes.resize(static_cast<size_t>(r.from));
    REQUIRE(std::equal(s.tokens().begin(), s.tokens().end(), prompt.begin()));
    return r;
  }
};

std::vector<int32_t> tokens(std::mt19937& rng, size_t n) {
  std::vector<int32_t> v(n);
  for (auto& t : v) t = static_cast<int32_t>(rng() % 50);
  return v;
}

}  // namespace

TEST_CASE("resumes_at: within the rings' slack, an odd position also within the pool's") {
  CHECK(resumes_at({0, 0}, 0));
  CHECK(resumes_at({100, 100}, 100));
  CHECK_FALSE(resumes_at({100, 100}, 101));  // past what is committed
  CHECK(resumes_at({1000, 1000}, 1000 - kSlack));
  CHECK_FALSE(resumes_at({1000, 1000}, 1000 - kSlack - 2));
  CHECK(resumes_at({1000, 1000}, 1000 - 15));        // odd, the pool still holds the even one before
  CHECK_FALSE(resumes_at({1000, 1000}, 1000 - 17));  // odd, its pool slot was overwritten
  CHECK(resumes_at({1000, 1000}, 1000 - 18));        // even: no pool read
  CHECK(resumes_at({5, 70000}, 0));                  // a new Session reads nothing old
  CHECK(latest_resume({1000, 1000}, 983) == 982);    // odd past the pool: the even one below
  CHECK(latest_resume({1000, 1000}, 400) == 0);      // past the slack
  CHECK(latest_resume({1000, 1003}, 2000) == 1000);  // at most what is committed
}

TEST_CASE("a prompt continuing the Session reads only its new tokens; one equal to it, only its last") {
  std::mt19937 rng(1);
  Model m;
  m.feed(tokens(rng, 300), 300);
  std::vector<int32_t> prompt = m.s.tokens();
  CHECK(plan(m.s, prompt).value() == Resume{std::nullopt, 299});
  prompt.push_back(7);
  CHECK(plan(m.s, prompt).value() == Resume{std::nullopt, 300});
  CHECK_FALSE(plan(m.s, std::vector<int32_t>{}).is_ok());
}

TEST_CASE("a chat's next turn resumes at the Rewind point of its previous prompt's end") {
  std::mt19937 rng(2);
  Model m;
  const std::vector<int32_t> prompt = tokens(rng, 5000);
  m.feed(prompt, prompt.size());
  m.mark(4);
  for (int i = 0; i < 300; i++) {  // a long reasoning, in verify Rounds
    const auto fed = tokens(rng, 6);
    m.feed(fed, 1 + rng() % 6);
  }
  // The next turn re-renders the answer without its reasoning: it shares the prompt but its last token.
  std::vector<int32_t> next(prompt.begin(), prompt.end() - 1);
  for (const int32_t t : tokens(rng, 40)) next.push_back(t);
  const Resume r = m.resume(next);
  REQUIRE(r.point.has_value());
  CHECK(r.from == 4999);
  CHECK(m.s.points().empty());  // the point's position is past the resume: its rows get rewritten
}

TEST_CASE("next_buffer takes a free buffer, then the oldest point's") {
  Session s;
  CHECK(next_buffer(s, 2) == 0);
  s = marked(std::move(s), 0);
  CHECK(next_buffer(s, 2) == 1);
  s = marked(std::move(s), 1);
  CHECK(next_buffer(s, 2) == 0);
  s = marked(std::move(s), 0);  // retaken: now the newest
  CHECK(next_buffer(s, 2) == 1);
  CHECK(s.points().size() == 2);
}

TEST_CASE("wrote commits 1 to all of its rows") {
  CHECK_FALSE(wrote({}, std::vector<int32_t>{}, 0).is_ok());
  CHECK_FALSE(wrote({}, std::vector<int32_t>{1, 2}, 0).is_ok());
  CHECK_FALSE(wrote({}, std::vector<int32_t>{1, 2}, 3).is_ok());
  const Session s = wrote({}, std::vector<int32_t>{1, 2, 3}, 2).value();
  CHECK(s.tokens() == std::vector<int32_t>{1, 2});
  CHECK(s.written() == Written{2, 3});
}

TEST_CASE("property: every planned resume reads only the Session's own writes") {
  std::mt19937 rng(3);
  for (int trial = 0; trial < 40; trial++) {
    Model m;
    for (int turn = 0; turn < 12; turn++) {
      // A prompt: the Session's tokens (all, nearly all, a cut, or none), then new ones.
      std::vector<int32_t> prompt = m.s.tokens();
      const int shape = static_cast<int>(rng() % 4);
      if (shape == 1 && !prompt.empty()) prompt.resize(prompt.size() - 1 - rng() % std::min<size_t>(prompt.size(), 20));
      if (shape == 2 && !prompt.empty()) prompt.resize(rng() % prompt.size());
      if (shape == 3) prompt.clear();
      const size_t kept = prompt.size();  // tokens it shares with the Session (at least)
      for (const int32_t t : tokens(rng, 1 + rng() % (rng() % 3 ? 200 : 1500))) prompt.push_back(t);
      const Resume r = m.resume(prompt);
      // Near the live end (a reply's verify rows at most 5 past it): reused, an odd position perhaps less one.
      if (shape <= 1) CHECK(r.from + 1 >= static_cast<int64_t>(kept));
      // The rest of the prompt in sweeps of up to 4096, then (mostly) a Rewind point at its end.
      for (size_t at = static_cast<size_t>(r.from); at < prompt.size();) {
        const size_t n = std::min<size_t>(prompt.size() - at, 1 + rng() % 4096);
        m.feed(std::span<const int32_t>(prompt).subspan(at, n), n);
        at += n;
      }
      if (rng() % 3) m.mark(3);
      // A reply: plain steps and verify Rounds keeping some of their rows.
      for (int k = 0, steps = static_cast<int>(rng() % 400); k < steps; k++) {
        const size_t rows = 1 + rng() % 6;
        m.feed(tokens(rng, rows), 1 + rng() % rows);
      }
      for (const Point& p : m.s.points()) REQUIRE(p.at.committed <= static_cast<int64_t>(m.s.tokens().size()));
    }
  }
}

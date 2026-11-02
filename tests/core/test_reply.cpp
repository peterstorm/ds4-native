#include <doctest/doctest.h>

#include <vector>

#include "core/reply.hpp"

using namespace ds4n::reply;

namespace {

const std::vector<int32_t> kStops = {1, 128803};
constexpr int32_t kEnd = 128822;  // the reasoning block's end

/** A reply writing `tokens` in turn (forced tokens replacing the picks); its tokens and how it ended. */
std::pair<std::vector<int32_t>, std::optional<Finish>> run(const Limits& l, std::vector<int32_t> picks, bool wrap_at_3 = false) {
  Reply r = start(l);
  std::vector<int32_t> out;
  std::optional<int32_t> forced;
  for (size_t i = 0; i < picks.size(); i++) {
    const int32_t t = forced ? *forced : picks[i];
    out.push_back(t);
    const Next n = after(r, l, t, wrap_at_3 && i >= 2, true);
    if (n.finish) return {out, n.finish};
    r = n.reply;
    forced = n.forced;
  }
  return {out, std::nullopt};
}

}  // namespace

TEST_CASE("a reply ends at an end-of-turn token, or at max_new") {
  const Limits l{4, kStops, std::nullopt};
  CHECK(run(l, {5, 6, 1, 7}) == std::pair{std::vector<int32_t>{5, 6, 1}, std::optional{Finish::Stop}});
  CHECK(run(l, {5, 6, 7, 8, 9}) == std::pair{std::vector<int32_t>{5, 6, 7, 8}, std::optional{Finish::Length}});
  CHECK(run(l, {5, 6, 7, 1}).second == Finish::Stop);  // the last allowed token ending the turn is a stop
  CHECK(after(start(l), l, 5, false, false).finish == Finish::Length);  // the context is full
}

TEST_CASE("the thinking budget closes an open reasoning block with its end token, then the answer goes on") {
  const Limits l{100, kStops, ThinkBudget{3, kEnd}};
  const auto [tokens, finish] = run(l, {10, 11, 12, 13, 14, 15, 1});
  CHECK(tokens == std::vector<int32_t>{10, 11, 12, kEnd, 14, 15, 1});
  CHECK(finish == Finish::Stop);
  // A block the model closes itself is not forced again.
  CHECK(run(l, {10, kEnd, 12, 13, 14}).first == std::vector<int32_t>{10, kEnd, 12, 13, 14});
  // No budget: nothing forced, however long.
  CHECK(run({100, kStops, std::nullopt}, {10, 11, 12, 13, 14}).first == std::vector<int32_t>{10, 11, 12, 13, 14});
}

TEST_CASE("WRAP closes a reply's reasoning at its next token") {
  const Limits l{100, kStops, ThinkBudget{1000, kEnd}};
  CHECK(run(l, {10, 11, 12, 13, 14}, true).first == std::vector<int32_t>{10, 11, 12, kEnd, 14});
}

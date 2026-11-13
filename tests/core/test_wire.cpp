#include <doctest/doctest.h>

#include <optional>
#include <random>
#include <string>
#include <variant>

#include "core/wire.hpp"

using namespace ds4n;
using namespace ds4n::wire;

namespace {

constexpr int32_t kVocab = 129280;

Generate generate(std::string_view line, uint64_t seed = 99) {
  auto c = parse(line, kVocab, seed);
  REQUIRE(c.is_ok());
  REQUIRE(std::holds_alternative<Generate>(c.value()));
  return std::get<Generate>(c.value());
}

}  // namespace

TEST_CASE("a greedy GEN: max_new and the prompt, greedy when temperature is 0") {
  const Generate g = generate("GEN 64 temperature=0 0,128803,4,5");
  CHECK(g.max_new == 64);
  CHECK(g.prompt == std::vector<int32_t>{0, 128803, 4, 5});
  CHECK(g.sampling.greedy_pick());
  CHECK_FALSE(g.think.has_value());
  CHECK_FALSE(g.spec.has_value());
  CHECK(generate("GEN 8 1,2\r").prompt == std::vector<int32_t>{1, 2});
}

TEST_CASE("a sampled GEN as Maya's server writes it: its keys, unknown ones skipped") {
  const Generate g = generate(
      "GEN 32768 temperature=1.0 top_p=0.95 top_k=64 min_p=0.05 penalty_repeat=1.1 penalty_last_n=64 seed=42 "
      "think_budget=24576 think_end=128822 spec_cum=0.5 cvec=1 logit_bias=7:-100,9:2.5 0,1,2");
  const sample::Settings& s = g.sampling.settings();
  CHECK(s.temperature == 1.0f);
  CHECK(s.top_p == 0.95f);
  CHECK(s.top_k == 64);
  CHECK(s.min_p == 0.05f);
  CHECK(s.seed == 42);
  CHECK(s.penalties == sample::Penalties{64, 1.1f, 0.0f, 0.0f});
  CHECK(s.bias == std::vector<sample::Bias>{{7, -100.0f}, {9, 2.5f}});
  REQUIRE(g.think.has_value());
  CHECK(g.think->tokens == 24576);
  CHECK(g.think->end == 128822);
  CHECK(generate("GEN 4 temperature=0.7 1").sampling.settings().seed == 99);  // none named: the fallback
  CHECK(generate("GEN 4 spec=0 1").spec == false);
}

TEST_CASE("spec is Maya's verify window: 1 is the plain token loop, 2 and up draft") {
  // server.py's strata_tune {"spec": N} (1 <= N <= 16) becomes spec=N.
  CHECK(generate("GEN 4 spec=1 1").spec == false);
  CHECK(generate("GEN 4 spec=2 1").spec == true);
  CHECK(generate("GEN 4 spec=6 1").spec == true);
  CHECK(generate("GEN 4 spec=16 1").spec == true);
}

TEST_CASE("control lines, and what is refused") {
  CHECK(std::holds_alternative<Stop>(parse("STOP", kVocab, 0).value()));
  CHECK_FALSE(std::get<Stop>(parse("STOP", kVocab, 0).value()).rid.has_value());
  CHECK(std::holds_alternative<Wrap>(parse("WRAP", kVocab, 0).value()));
  CHECK(std::holds_alternative<Quit>(parse("QUIT", kVocab, 0).value()));
  for (const char* bad : {"GEN 0 1,2", "GEN -3 1", "GEN 4", "GEN x 1", "GEN 4 1,2 temperature=0", "GEN 4 1,,2",
                          "GEN 4 129280", "GEN 4 -1", "GEN 4 top_p=0 1", "GEN 4 temperature=hot 1",
                          "GEN 4 think_budget=10 1", "GEN 4 think_budget=10 think_end=129999 1", "GEN 4 logit_bias=3 1",
                          "GEN 4 logit_bias=999999:1 1", "GENI 4 pics.bin 1,2", "PAUSE", "PAUSE 3 keep", "STOP x", "STOP 3 4", "STOP -1", "GEN 4 rid=x 1", "GEN 4 rid=-2 1", ""})
    CHECK_FALSE(parse(bad, kVocab, 0).is_ok());
}

TEST_CASE("the lines a request writes, as before Lanes when it names no rid") {
  CHECK(token_line({}, 128803) == "T 128803\n");
  CHECK(ready_line(65536, lane::LaneCount()) == "READY 65536 stop\n");
  CHECK(progress_line({}, {600, 1000, 100, 250.0}) == "PP 600 1000 250.0 2000.0 100\n");
  CHECK(done_line({}, {12, 1000, 250.04, 80.0, reply::Finish::Stop, 7, 10, 100}) ==
        "DONE 12 1000 250.0 80.0 stop 7 10 100 0 0\n");
  CHECK(error_line({}, "non-finite logits (row 0)") == "ERR non-finite logits (row 0)\n");
  CHECK(stat_line({16, 160.0, 0.9512, 0.75}) == "STAT tok_s=100.00 ms_tok=10.00 vram_hit=0.9512 drafts=0.750\n");
  CHECK(stat_line({16, 160.0, 1.0, -1.0}) == "STAT tok_s=100.00 ms_tok=10.00 vram_hit=1.0000\n");
}

namespace {

/**
 * Maya's server routing a line to its request (serve/server.py `_route` and `_generate_multi` at
 * b3fa99e): a line starting `T `, `PP `, `DONE ` or `ERR ` whose second word is all digits is that
 * rid's, and the request reads it as the verb and the rest -- what an engine without rids writes.
 */
struct Routed {
  uint64_t rid;
  std::string line;
  bool operator==(const Routed&) const = default;
};
std::optional<Routed> routed(const std::string& line) {
  const size_t verb = line.find(' ');
  const size_t id_end = verb == std::string::npos ? verb : line.find(' ', verb + 1);
  if (id_end == std::string::npos) return std::nullopt;
  const std::string id = line.substr(verb + 1, id_end - verb - 1);
  if (id.empty() || id.find_first_not_of("0123456789") != std::string::npos) return std::nullopt;
  return Routed{std::stoull(id), line.substr(0, verb) + line.substr(id_end)};
}

}  // namespace

TEST_CASE("a GEN names its request with rid=, and STOP <rid> ends that one alone") {
  // As serve/server.py writes them: sampling_keys, then ` rid=<id>`, then the ids.
  const Generate g = generate("GEN 40 temperature=0 rid=7 0,1,2");
  CHECK(g.rid == Rid{7});
  CHECK(g.max_new == 40);
  CHECK(g.prompt == std::vector<int32_t>{0, 1, 2});
  CHECK_FALSE(generate("GEN 40 temperature=0 0,1,2").rid.has_value());
  CHECK(generate("GEN 4 rid=18446744073709551615 1").rid == Rid{UINT64_MAX});
  const Stop stop = std::get<Stop>(parse("STOP 7", kVocab, 0).value());
  CHECK(stop.rid == Rid{7});
}

TEST_CASE("a GEN that does not parse still says whose it was, so its ERR reaches that request") {
  for (const char* bad : {"GEN 0 rid=5 1,2", "GEN 4 rid=5 temperature=hot 1", "GEN 4 rid=5 129280", "GENI 4 rid=5 pics.bin 1,2"}) {
    const auto c = parse(bad, kVocab, 0);
    REQUIRE_FALSE(c.is_ok());
    CHECK(c.error().rid == Rid{5});
  }
  CHECK_FALSE(parse("GEN 0 1,2", kVocab, 0).error().rid.has_value());
  CHECK_FALSE(parse("GEN 4 rid=x 1", kVocab, 0).error().rid.has_value());  // no rid to answer to
  CHECK_FALSE(parse("NOPE rid=5", kVocab, 0).error().rid.has_value());
}

TEST_CASE("an engine with several Lanes says so in READY, and answers a GEN with no Lane free in the server's words") {
  CHECK(ready_line(262144, lane::LaneCount::of(4).value()) == "READY 262144 stop multi=4\n");
  CHECK(ready_line(262144, lane::LaneCount::of(1).value()) == "READY 262144 stop\n");  // one Lane: no multi=, no rids
  // serve/fake_multi_engine.py: f"ERR {rid} all {n} conversation states are busy"
  CHECK(busy_line(Rid{4}, lane::LaneCount::of(3).value()) == "ERR 4 all 3 conversation states are busy\n");
}

TEST_CASE("a request's lines carry its rid after the verb, as Maya's reference engine writes them") {
  // serve/fake_multi_engine.py: "T {rid} {t}", "PP {rid} {n} {n} 1.0 1000.0", "DONE {rid} {n} {p} 0.0 1.0 length 0 0 0".
  CHECK(token_line(Rid{3}, 1000) == "T 3 1000\n");
  CHECK(progress_line(Rid{3}, {40, 40, 0, 1.0}) == "PP 3 40 40 1.0 40000.0 0\n");
  CHECK(done_line(Rid{3}, {40, 40, 0.0, 1.0, reply::Finish::Length, 0, 0, 0}) == "DONE 3 40 40 0.0 1.0 length 0 0 0 0 0\n");
  CHECK(error_line(Rid{3}, "non-finite logits (row 0)") == "ERR 3 non-finite logits (row 0)\n");
  CHECK(error_line(Rid{3}, "two\nlines\r") == "ERR 3 two lines \n");  // one protocol line, whatever the message
  CHECK(token_line(Rid{0}, 5) == "T 0 5\n");                           // rid 0 is a rid
}

TEST_CASE("property: the server routes every tagged line to its rid and reads there the line an untagged engine writes") {
  std::mt19937_64 rng(31);
  for (int trial = 0; trial < 2000; trial++) {
    const Rid rid{trial % 7 == 0 ? rng() : rng() % 1000};
    const auto token = static_cast<int32_t>(rng() % kVocab);
    const Progress progress{rng() % 262144, 262144, rng() % 1000, static_cast<double>(rng() % 100000) / 10.0};
    const Done done{static_cast<int64_t>(rng() % 32768), rng() % 262144, static_cast<double>(rng() % 900000) / 10.0,
                    static_cast<double>(rng() % 900000) / 10.0, static_cast<reply::Finish>(rng() % 3),
                    static_cast<int64_t>(rng() % 5000), static_cast<int64_t>(rng() % 9000), rng() % 262144};
    CHECK(routed(token_line(rid, token)) == Routed{rid.id, token_line({}, token)});
    CHECK(routed(progress_line(rid, progress)) == Routed{rid.id, progress_line({}, progress)});
    CHECK(routed(done_line(rid, done)) == Routed{rid.id, done_line({}, done)});
    CHECK(routed(error_line(rid, "the prompt leaves no room")) == Routed{rid.id, error_line({}, "the prompt leaves no room")});
  }
  // An untagged ERR whose message starts with a number would be routed as that rid's: the engine never
  // writes one in multi-request mode (every ERR there is a request's, or names no number first).
  CHECK_FALSE(routed(error_line({}, "this engine reads no pictures")).has_value());
  CHECK_FALSE(routed(stat_line({16, 160.0, 1.0, -1.0})).has_value());
}

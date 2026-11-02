#include <doctest/doctest.h>

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

TEST_CASE("control lines, and what is refused") {
  CHECK(std::holds_alternative<Stop>(parse("STOP", kVocab, 0).value()));
  CHECK(std::holds_alternative<Stop>(parse("STOP 3", kVocab, 0).value()));
  CHECK(std::holds_alternative<Wrap>(parse("WRAP", kVocab, 0).value()));
  CHECK(std::holds_alternative<Quit>(parse("QUIT", kVocab, 0).value()));
  for (const char* bad : {"GEN 0 1,2", "GEN -3 1", "GEN 4", "GEN x 1", "GEN 4 1,2 temperature=0", "GEN 4 1,,2",
                          "GEN 4 129280", "GEN 4 -1", "GEN 4 top_p=0 1", "GEN 4 temperature=hot 1",
                          "GEN 4 think_budget=10 1", "GEN 4 think_budget=10 think_end=129999 1", "GEN 4 logit_bias=3 1",
                          "GEN 4 logit_bias=999999:1 1", "GENI 4 pics.bin 1,2", "PAUSE", ""})
    CHECK_FALSE(parse(bad, kVocab, 0).is_ok());
}

TEST_CASE("the lines a request writes") {
  CHECK(token_line(128803) == "T 128803\n");
  CHECK(ready_line(65536) == "READY 65536 stop\n");
  CHECK(progress_line({600, 1000, 100, 250.0}) == "PP 600 1000 250.0 2000.0 100\n");
  CHECK(done_line({12, 1000, 250.04, 80.0, reply::Finish::Stop, 7, 10, 100}) ==
        "DONE 12 1000 250.0 80.0 stop 7 10 100 0 0\n");
  CHECK(stat_line({16, 160.0, 0.9512, 0.75}) == "STAT tok_s=100.00 ms_tok=10.00 vram_hit=0.9512 drafts=0.750\n");
  CHECK(stat_line({16, 160.0, 1.0, -1.0}) == "STAT tok_s=100.00 ms_tok=10.00 vram_hit=1.0000\n");
}

// ds4n-glm53-serve-check: the server's conversations are exact. Every path a server takes to resume a
// prompt must give the logits a fresh run of the same tokens gives, bit for bit:
//
//   S1 (Decoder): save, rewind, restore. A prompt A then B against A + B swept at once from empty; A set
//      aside (save), the caches rewound and overwritten by another conversation C, A restored, then B
//      swept, and B fed token by token (round()); a restore of only a prefix of A, the rest re-read.
//   S2 (Conversation, greedy): conversation X's turn, another conversation Y, X's next turn (resumed
//      from its checkpoint), X's turn after that (resumed from the live caches), a STOP. Each turn's
//      logits rows and tokens against the same prompt run on emptied caches; each turn's reuse as the
//      plan says (X's next turn reuses its whole last turn but the reply's last token).
//
//   ds4n-glm53-serve-check --config CONFIG --backbone DIR --store FILE --prompt-file IDS [--capacity C]
//                          [--generate G] [--slots-per-layer S] [--ram-per-layer R]
//
// IDS: whitespace-separated token ids (a real text's, so routing is a real prompt's), at least 400.
// Exact routing (λ 0): under λ the routing of a reply depends on the residency, which a fresh run does
// not share. Any difference exits 1.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "core/glm53/config.hpp"
#include "core/sampler.hpp"
#include "core/wire.hpp"
#include "engine/glm53/conversation.hpp"
#include "engine/glm53/decoder.hpp"
#include "io/glm53_weights.hpp"

using namespace ds4n;
namespace eng = ds4n::engine::glm53;

namespace {

[[noreturn]] void die(const std::string& m) {
  std::fprintf(stderr, "ds4n-glm53-serve-check: %s\n", m.c_str());
  std::exit(1);
}

template <class T>
T must(Result<T, std::string> r, const std::string& what) {
  if (!r) die(what + ": " + r.error());
  return std::move(r).value();
}

using Tokens = std::vector<int32_t>;

Tokens cat(std::initializer_list<std::span<const int32_t>> parts) {
  Tokens t;
  for (auto p : parts) t.insert(t.end(), p.begin(), p.end());
  return t;
}

/** Values of `got` whose bits differ from `want`'s. */
size_t differing(std::span<const float> want, std::span<const float> got) {
  if (want.size() != got.size()) return std::max(want.size(), got.size());
  size_t bad = 0;
  for (size_t i = 0; i < want.size(); i++) bad += std::memcmp(&want[i], &got[i], sizeof(float)) != 0;
  return bad;
}

bool report(const char* gate, size_t bad, size_t of) {
  std::printf("  %-72s %s (%zu of %zu values differ)\n", gate, bad == 0 ? "exact" : "DIFFERS", bad, of);
  std::fflush(stdout);
  return bad == 0;
}

/** One request's record: every logits row picked from, the tokens written, the Done. */
struct Turn {
  std::vector<float> rows;
  Tokens written;
  wire::Done done{};
};

Turn run(eng::Conversation& c, const Tokens& prompt, int64_t max_new, int stop_after = -1) {
  Turn t;
  wire::Generate g{max_new, sample::Sampling::greedy(), std::nullopt, std::nullopt, prompt, std::nullopt};
  eng::ReplyEvents ev;
  ev.token = [&](int32_t tok) { t.written.push_back(tok); };
  ev.progress = [](const wire::Progress&) {};
  ev.stat = [](const wire::Stat&) {};
  ev.note = [](const std::string& n) { std::printf("    note: %s\n", n.c_str()); };
  ev.logits = [&](std::span<const float> row) { t.rows.insert(t.rows.end(), row.begin(), row.end()); };
  const auto poll = [&] { return eng::Interrupts{stop_after >= 0 && static_cast<int>(t.written.size()) >= stop_after, false}; };
  const eng::Outcome o = c.reply(g, ev, poll);
  if (const auto* d = std::get_if<wire::Done>(&o)) t.done = *d;
  else die("a request did not finish: " + (std::holds_alternative<eng::Refused>(o) ? std::get<eng::Refused>(o).what
                                                                                    : std::get<eng::Failed>(o).what));
  return t;
}

}  // namespace

int main(int argc, char** argv) {
  std::string config_path, backbone, store, prompt_file;
  eng::DecoderConfig dc;
  dc.capacity = 4096;
  int64_t generate = 6;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i], v = argv[i + 1];
    if (k == "--config") config_path = v;
    else if (k == "--backbone") backbone = v;
    else if (k == "--store") store = v;
    else if (k == "--prompt-file") prompt_file = v;
    else if (k == "--capacity") dc.capacity = std::stoi(v);
    else if (k == "--generate") generate = std::stoll(v);
    else if (k == "--slots-per-layer") dc.slots_per_layer = std::stoi(v);
    else if (k == "--ram-per-layer") dc.ram_tier = eng::RamTierFixed{std::stoi(v)};
    else die("unknown argument " + k);
  }
  if (config_path.empty() || backbone.empty() || store.empty() || prompt_file.empty())
    die("usage: --config CONFIG --backbone DIR --store FILE --prompt-file IDS [--capacity C] [--generate G]");
  Tokens text;
  {
    std::ifstream f(prompt_file);
    for (int32_t t; f >> t;) text.push_back(t);
  }
  if (text.size() < 400) die("the prompt file needs at least 400 token ids");
  std::ifstream cf(config_path);
  const auto config = glm53::parse_config(std::string((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>()));
  if (!config) die(config.error().what);
  const size_t V = static_cast<size_t>(config.value().dims().vocab);
  auto parts = io::BackboneParts::open(backbone);
  if (!parts) die(parts.error().what);
  auto opened = eng::Decoder::open(config.value(), parts.value(), {store}, dc);
  if (!opened) die(opened.error());
  eng::Decoder& dec = *opened.value();
  std::printf("open: %.0f s, %d Slots per layer, capacity %d, %.1f KB a position\n", dec.stats().load_s, dec.slots_per_layer(),
              dc.capacity, static_cast<double>(dec.position_bytes()) / 1e3);

  // Three pieces of real text: A and B continue each other, C is another conversation's.
  const std::span<const int32_t> all(text);
  const Tokens A(all.begin(), all.begin() + 200), B(all.begin() + 200, all.begin() + 300),
      C(all.begin() + 300, all.begin() + 400);
  bool ok = true;

  std::printf("S1: save, rewind, restore on the Decoder\n");
  {
    std::vector<float> ref;
    must(dec.prefill(cat({A, B}), nullptr, &ref), "A + B");
    const std::span<const float> ref_a = std::span<const float>(ref).subspan((A.size() - 1) * V, V);
    const std::span<const float> ref_b = std::span<const float>(ref).subspan(A.size() * V, B.size() * V);
    must(dec.rewind(0), "rewind");
    const std::vector<float> last_a = must(dec.prefill(A), "A");
    ok &= report("A swept alone: its last logits", differing(ref_a, last_a), V);
    const eng::KvRows saved = must(dec.save(dec.length()), "save A");
    must(dec.rewind(static_cast<int64_t>(A.size() / 2)), "rewind into A");
    must(dec.prefill(C), "C over A's second half");
    must(dec.restore(saved, saved.positions()), "restore A");
    std::vector<float> b;
    must(dec.prefill(B, nullptr, &b), "B after the restore");
    ok &= report("A restored, B swept: B's logits", differing(ref_b, b), ref_b.size());
    must(dec.restore(saved, saved.positions()), "restore A again");
    std::vector<float> fed;
    for (int32_t t : B) {
      const auto row = must(dec.round(std::span<const int32_t>(&t, 1)), "B fed");
      fed.insert(fed.end(), row.begin(), row.end());
    }
    ok &= report("A restored, B fed a token at a time: B's logits", differing(ref_b, fed), ref_b.size());
    const auto part = static_cast<int64_t>(A.size() / 3);
    must(dec.restore(saved, part), "restore A's first third");
    std::vector<float> rest;
    must(dec.prefill(cat({std::span(A).subspan(static_cast<size_t>(part)), B}), nullptr, &rest), "the rest of A, then B");
    ok &= report("A's first third restored, the rest of A and B swept: B's logits",
                 differing(ref_b, std::span<const float>(rest).subspan((A.size() - static_cast<size_t>(part)) * V)), ref_b.size());
    must(dec.rewind(0), "rewind");
  }

  std::printf("S2: a server's conversations, greedy, %lld tokens a reply\n", static_cast<long long>(generate));
  {
    eng::ConversationConfig cc;
    cc.stops = {154820, 154827, 154829};
    cc.checkpoint_bytes = 4'000'000'000;
    auto conv = must(eng::Conversation::create(dec, cc), "conversation");
    const Turn x1 = run(conv, A, generate);
    const Turn y1 = run(conv, C, generate);
    const Tokens p2 = cat({A, x1.written, B});
    const Turn x2 = run(conv, p2, generate);
    const Tokens p3 = cat({p2, x2.written, C});
    const Turn x3 = run(conv, p3, generate);
    const Turn stopped = run(conv, cat({p3, x3.written, B}), generate, 2);
    std::printf("  reuse: X %zu, Y %zu, X next %zu (of %zu), X after %zu (of %zu)\n", x1.done.reused, y1.done.reused,
                x2.done.reused, p2.size(), x3.done.reused, p3.size());
    ok &= report("X's next turn resumes from its checkpoint: its whole last turn but one token",
                 x2.done.reused == A.size() + x1.written.size() - 1 ? 0 : 1, 1);
    ok &= report("X's turn after resumes from the live caches", x3.done.reused == p2.size() + x2.written.size() - 1 ? 0 : 1, 1);
    ok &= report("a STOP ends the reply after the token it came with",
                 stopped.done.finish == reply::Finish::Cancel && stopped.written.size() == 2 ? 0 : 1, 1);
    // The same prompts on emptied caches: a new Conversation after a rewind.
    must(dec.rewind(0), "rewind");
    auto fresh = must(eng::Conversation::create(dec, cc), "fresh conversation");
    const Turn x2f = run(fresh, p2, generate);
    ok &= report("X's next turn: logits rows as on empty caches", differing(x2f.rows, x2.rows), x2f.rows.size());
    ok &= report("X's next turn: tokens as on empty caches", x2f.written == x2.written ? 0 : 1, 1);
    must(dec.rewind(0), "rewind");
    auto fresh3 = must(eng::Conversation::create(dec, cc), "fresh conversation");
    const Turn x3f = run(fresh3, p3, generate);
    ok &= report("X's turn after: logits rows as on empty caches", differing(x3f.rows, x3.rows), x3f.rows.size());
    ok &= report("X's turn after: tokens as on empty caches", x3f.written == x3.written ? 0 : 1, 1);
  }
  std::printf("%s\n", ok ? "PASS: every resumed conversation is bit-identical to a fresh run"
                         : "FAIL: a resumed conversation differs from a fresh run");
  return ok ? 0 : 1;
}

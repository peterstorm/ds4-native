// ds4n-glm53-serve: GLM-5.3 behind Project Maya's engine protocol (core/wire), so Maya's OpenAI-compatible
// server (serve/server.py, MIT) serves it on /v1 as it serves its own GLM engine: the server renders the
// chat (GLM-5's template, serve/glm5_chat_template.jinja), tokenizes, starts `ds4n-glm53-serve --serve
// ARGS...` and speaks GEN / T / DONE with it over stdin and stdout.
//
//   ds4n-glm53-serve --serve --config CONFIG --backbone DIR --store FILE[,FILE...] [--max-context N]
//                    [--lambda λ] [--eviction lru|counter:N] [--slots-per-layer S] [--slot-profile N,N,...]
//                    [--runtime-reserve-gib G] [--ram-per-layer R|auto] [--ram-reserve-gib G]
//                    [--checkpoint-gb G] [--stops ID,ID,...] [--wave W] [--cohort K] [--devices A,B]
//                    [--threads T] [--defer N]
//
// --max-context N (default 16384): positions every layer's cache holds, the context Maya reports (its
// dashboard restarts the engine with another). Each 1K positions take ~0.1 GB of VRAM per GPU (the
// caches, 95.2 KB a position): ~0.13 Expert Slots per layer. The Slots are the Ledger's
// (core/glm53/ledger): `ds4n-glm53-ledger --capacity N` prints them for any N without a GPU.
// --cohort K (default 16384, rounded down to whole waves): a longer prompt is read as a sweep per K tokens; the
// sweep's buffers take ~101 KB a token of min(K, N) on each GPU (16K: ~1.7 GB, ~2.2 Slots per layer).
// --lambda λ (default 0): cache-conditional routing for the reply's tokens; prompts route exactly.
// --eviction lru|counter:N (default lru): which Expert a full layer's Slots give up (core/glm53/layer_expert_cache).
// --slot-profile N,N,... (one count per Sparse layer): the Ledger's Slots in a fitted shape instead of an equal
// share (core/glm53/slot_profile), the total unchanged. --runtime-reserve-gib G: the VRAM the Ledger leaves
// outside its claims (default kDefaultRuntimeReserve); less is more Slots, while CUDA still fits in it.
// --checkpoint-gb G (default 4): host memory for conversations set aside (engine/glm53/conversation),
// 95.2 KB a token (Decoder::position_bytes; 4 GB hold ~42K tokens); with --ram-per-layer auto the RAM
// tier leaves it free beside --ram-reserve-gib (10).
// --stops (default GLM's turn ends: <|endoftext|>, <|user|>, <|observation|>).
// --defer N (default off): the reply's tokens defer each Miss outside their N heaviest Route slots one
// Sparse layer (core/glm53/deferral): lossy, opt-in; the oracle's -dfN streams measure its cost.
//
// One request at a time (READY <context> stop, no multi=): the server queues the rest. A chat's next
// turn and an agent's next step read only their new tokens; a conversation another one displaced comes
// back from its checkpoint. Stdout carries the protocol only (NOTE lines tell the server's log what
// happened to the conversations); everything else goes to stderr (the server's log). A Decoder failure
// fails the request in flight and ends the engine (exit 2): the server starts it again.
#include <poll.h>
#include <unistd.h>

#include <cerrno>
#include <charconv>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "core/glm53/config.hpp"
#include "core/glm53/layer_expert_cache.hpp"
#include "core/store_replicas.hpp"
#include "core/wire.hpp"
#include "engine/glm53/conversation.hpp"
#include "engine/glm53/decoder.hpp"
#include "io/glm53_weights.hpp"

using namespace ds4n;
namespace eng = ds4n::engine::glm53;

namespace {

[[noreturn]] void die(const std::string& m) {
  std::fprintf(stderr, "ds4n-glm53-serve: %s\n", m.c_str());
  std::exit(1);
}

void out(const std::string& line) {
  std::fputs(line.c_str(), stdout);
  std::fflush(stdout);
}

/**
 * Stdin's lines, read with read(2) (std::cin's lock would hold exit() up): take(true) waits for at
 * least one line or the end; take(false) takes only what has arrived.
 */
class Lines {
 public:
  std::vector<std::string> take(bool wait) {
    std::vector<std::string> lines;
    for (;;) {
      for (size_t nl; (nl = buf_.find('\n')) != std::string::npos;) {
        lines.push_back(buf_.substr(0, nl));
        buf_.erase(0, nl + 1);
      }
      if (!lines.empty() || ended_) {
        if (ended_ && !buf_.empty()) lines.push_back(std::exchange(buf_, {}));  // a last line without its newline
        return lines;
      }
      if (!wait) {
        pollfd p{0, POLLIN, 0};
        if (::poll(&p, 1, 0) <= 0) return lines;
      }
      char chunk[1 << 16];
      const ssize_t n = ::read(0, chunk, sizeof chunk);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) ended_ = true;
      else buf_.append(chunk, static_cast<size_t>(n));
      wait = false;  // something arrived: take what is complete
    }
  }
  bool ended() const { return ended_ && buf_.empty(); }

 private:
  std::string buf_;
  bool ended_ = false;
};

std::vector<int32_t> ids_of(const std::string& s, int32_t vocab, const char* what) {
  std::vector<int32_t> v;
  for (size_t at = 0; at < s.size();) {
    const size_t comma = std::min(s.find(',', at), s.size());
    int32_t id = -1;
    const auto [end, ec] = std::from_chars(s.data() + at, s.data() + comma, id);
    if (ec != std::errc{} || end != s.data() + comma || id < 0 || id >= vocab) die(std::string(what) + ": not a token id list: " + s);
    v.push_back(id);
    at = comma + 1;
  }
  return v;
}

/** The lines of a request (of `prompt` tokens) that ran and failed: its ERR, and a DONE so the server's turn for it ends. */
std::string failed_lines(std::optional<wire::Rid> rid, size_t prompt, const std::string& what) {
  return wire::error_line(rid, what) + wire::done_line(rid, {0, prompt, 0.0, 0.0, reply::Finish::Cancel, 0, 0, 0});
}

template <class... F>
struct Overloaded : F... {
  using F::operator()...;
};
template <class... F>
Overloaded(F...) -> Overloaded<F...>;

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGPIPE, SIG_IGN);  // a server already gone must not end us mid-write
  std::string config_path, backbone, stops, slot_profile;
  std::optional<int> defer;  // --defer N
  std::vector<std::string> stores;
  eng::DecoderConfig dc;
  dc.capacity = 16384;
  dc.ram_tier = eng::RamTierFit{};
  double checkpoint_gb = 4.0;
  uint64_t reserve_gib = 10;
  bool serve = false;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    if (a == "--serve") {
      serve = true;
      continue;
    }
    if (i + 1 >= argc) die("missing value for " + a);
    const std::string v = argv[++i];
    if (a == "--config") config_path = v;
    else if (a == "--backbone") backbone = v;
    else if (a == "--store")
      for (std::string& p : replicas::paths_named(v)) stores.push_back(std::move(p));
    else if (a == "--max-context") dc.capacity = std::stoi(v);
    else if (a == "--lambda") dc.route_bonus = std::stof(v);
    else if (a == "--eviction") {
      auto e = glm53::parse_eviction(v);
      if (!e) die(e.error().what);
      dc.eviction = e.value();
    }
    else if (a == "--slots-per-layer") dc.slots_per_layer = std::stoi(v);
    else if (a == "--slot-profile") slot_profile = v;  // parsed against the config below
    else if (a == "--defer") defer = std::stoi(v);  // made a Deferral against the config below
    else if (a == "--runtime-reserve-gib") {
      const double g = std::stod(v);
      if (!(g >= 0.0)) die("--runtime-reserve-gib must be >= 0");
      dc.runtime_reserve_bytes = static_cast<uint64_t>(g * static_cast<double>(1ull << 30));
    }
    else if (a == "--ram-per-layer") dc.ram_tier = v == "auto" ? eng::RamTierSize{eng::RamTierFit{}} : eng::RamTierFixed{std::stoi(v)};
    else if (a == "--ram-reserve-gib") reserve_gib = std::stoull(v);
    else if (a == "--checkpoint-gb") checkpoint_gb = std::stod(v);
    else if (a == "--stops") stops = v;
    else if (a == "--wave") dc.wave = std::stoi(v);
    else if (a == "--cohort") dc.cohort = std::stoi(v);
    else if (a == "--threads") dc.threads = static_cast<unsigned>(std::stoul(v));
    else if (a == "--devices") {
      if (std::sscanf(v.c_str(), "%d,%d", &dc.devices[0], &dc.devices[1]) != 2) die("--devices A,B");
    }
    // Maya's server adds this for its engines on more than one GPU: the two GPUs split every layer.
    else if (a == "--layer-split") {
      if (v != "auto") die("--layer-split " + v + ": ds4-native splits every layer across the two GPUs");
    }
    else die("unknown argument " + a);
  }
  if (!serve || config_path.empty() || backbone.empty() || stores.empty())
    die("usage: ds4n-glm53-serve --serve --config CONFIG --backbone DIR --store FILE[,FILE...] [--max-context N] [--lambda λ] "
        "[--eviction lru|counter:N] [--slots-per-layer S] [--slot-profile N,N,...] [--runtime-reserve-gib G] [--ram-per-layer R|auto] [--ram-reserve-gib G] [--checkpoint-gb G] [--stops ID,...] "
        "[--wave W] [--cohort K] [--devices A,B] [--threads T] [--defer N]");
  if (!(checkpoint_gb >= 0.0)) die("--checkpoint-gb must be >= 0");
  const auto checkpoint_bytes = static_cast<uint64_t>(checkpoint_gb * 1e9);
  // The RAM tier is pinned at open, sized to the memory available then: the checkpoints' budget stays free beside the reserve.
  if (auto* fit = std::get_if<eng::RamTierFit>(&dc.ram_tier)) fit->reserve_bytes = (reserve_gib << 30) + checkpoint_bytes;

  std::ifstream cf(config_path);
  if (!cf) die("cannot read " + config_path);
  const auto config = glm53::parse_config(std::string((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>()));
  if (!config) die(config.error().what);
  if (!slot_profile.empty()) {
    auto p = glm53::SlotProfile::parse(slot_profile, config.value());
    if (!p) die("--slot-profile: " + p.error().what);
    dc.slot_profile = std::move(p).value();
  }
  if (defer) {
    auto rule = glm53::Deferral::create(*defer, config.value().dims().experts_per_token);
    if (!rule) die("--defer: " + rule.error().what);
    dc.deferral = rule.value();
  }
  const int32_t vocab = config.value().dims().vocab;
  int sparse_layers = 0;
  for (int L = 0; L < config.value().dims().layers; L++) sparse_layers += config.value().mlp(L) == glm53::MlpKind::Sparse;
  auto parts = io::BackboneParts::open(backbone);
  if (!parts) die(parts.error().what);
  auto decoder = eng::Decoder::open(config.value(), parts.value(), stores, dc);
  if (!decoder) die(decoder.error());
  eng::Decoder& dec = *decoder.value();
  eng::ConversationConfig cc;
  cc.stops = stops.empty() ? std::vector<int32_t>{154820, 154827, 154829} : ids_of(stops, vocab, "--stops");
  cc.checkpoint_bytes = checkpoint_bytes;
  auto made = eng::Conversation::create(dec, cc);
  if (!made) die(made.error());
  eng::Conversation conversation = std::move(made).value();

  char info[400];
  std::snprintf(info, sizeof info,
                "INFO engine=ds4-native engine_kind=ds4-native model=glm-5.3 context=%d vram_slots=%d ram_records=%d "
                "lambda=%g checkpoint_gb=%.1f load_s=%.0f logit_bias=1 lanes=1\n",
                dc.capacity, dec.slots_per_layer() * sparse_layers, dec.ram_per_layer() * sparse_layers, static_cast<double>(dc.route_bonus),
                checkpoint_gb, dec.stats().load_s);
  out(info);
  out(wire::ready_line(dc.capacity, lane::LaneCount::of(1).value()));
  std::fprintf(stderr,
               "ds4n-glm53-serve: ready in %.0f s: context %d, %d Slots per Sparse layer (%s), RAM tier %d per layer, λ %g, "
               "eviction %s, deferral %s, checkpoints %.1f GB (%.1f KB a token)\n",
               dec.stats().load_s, dc.capacity, dec.slots_per_layer(), dc.slot_profile ? "profile" : "equal",
               dec.ram_per_layer(), static_cast<double>(dc.route_bonus),
               glm53::to_string(dc.eviction).c_str(),
               dc.deferral ? ("protect " + std::to_string(dc.deferral->protect())).c_str() : "none", checkpoint_gb,
               static_cast<double>(dec.position_bytes()) / 1e3);
  for (size_t g = 0; g < 2; g++)  // the Ledger's account (core/glm53/ledger): ds4n-glm53-ledger predicts it
    std::fprintf(stderr, "ds4n-glm53-serve: vram GPU %zu: card %.2f GB, claims %.2f GB, free after open %.2f GB\n", g,
                 dec.vram().total[g] / 1e9, dec.vram().plan.claims[g].total() / 1e9, dec.vram().free_after_open[g] / 1e9);

  Lines in;
  std::mt19937_64 seeds(std::random_device{}());  // a GEN naming no seed samples with a fresh one
  std::deque<wire::Generate> waiting;            // GENs that arrived while another ran: they take turns
  bool wrapping = false, quit = false;           // WRAP: the server is stopping; replies close their reasoning
  // A control line; a GEN waits its turn. Returns whether it asked the request in flight to stop.
  const auto control = [&](const std::string& line) {
    const auto parsed = wire::parse(line, vocab, seeds());
    if (!parsed) {
      out(wire::error_line(parsed.error().rid, parsed.error().what));
      return false;
    }
    return std::visit(Overloaded{[&](const wire::Generate& g) {
                                   waiting.push_back(g);
                                   return false;
                                 },
                                 [&](const wire::Stop&) { return true; },
                                 [&](const wire::Wrap&) {
                                   wrapping = true;
                                   return false;
                                 },
                                 [&](const wire::Quit&) {
                                   quit = true;
                                   return true;
                                 }},
                      parsed.value());
  };
  while (!quit) {
    if (waiting.empty()) {
      if (in.ended()) break;
      for (const std::string& line : in.take(true)) control(line);  // STOPs with nothing in flight end nothing
      continue;
    }
    const wire::Generate g = std::move(waiting.front());
    waiting.pop_front();
    eng::ReplyEvents ev;
    ev.token = [&](int32_t t) { out(wire::token_line(g.rid, t)); };
    ev.progress = [&](const wire::Progress& p) { out(wire::progress_line(g.rid, p)); };
    ev.stat = [](const wire::Stat& s) { out(wire::stat_line(s)); };
    ev.note = [](const std::string& note) {
      std::fprintf(stderr, "ds4n-glm53-serve: %s\n", note.c_str());
      out("NOTE sessions: " + note + "\n");
    };
    const auto interrupts = [&] {
      eng::Interrupts i;
      for (const std::string& line : in.take(false)) i.stop = control(line) || i.stop;
      i.stop = i.stop || in.ended();  // the server is gone: nobody reads the reply
      i.wrap = wrapping;
      return i;
    };
    try {
      const eng::Outcome o = conversation.reply(g, ev, interrupts);
      std::visit(Overloaded{[&](const wire::Done& d) { out(wire::done_line(g.rid, d)); },
                            [&](const eng::Refused& r) { out(wire::error_line(g.rid, r.what)); },
                            [&](const eng::Failed& f) {
                              std::fprintf(stderr, "ds4n-glm53-serve: %s\n", f.what.c_str());
                              out(failed_lines(g.rid, g.prompt.size(), f.what));
                            }},
                 o);
    } catch (const std::exception& e) {
      // The caches and Slots may no longer agree: the request fails, and the server starts the engine again.
      std::fprintf(stderr, "ds4n-glm53-serve: %s\n", e.what());
      out(failed_lines(g.rid, g.prompt.size(), e.what()));
      return 2;
    }
  }
  return 0;
}

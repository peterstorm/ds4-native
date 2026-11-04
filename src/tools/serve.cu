// ds4n-serve: the whole model behind Project Maya's engine protocol (core/wire), so Maya's
// OpenAI-compatible server (serve/server.py, MIT) serves it on /v1 as it serves its own engines:
// the server renders the chat (DeepSeek-V4.1's encoding), tokenizes, starts `ds4n-serve --serve
// ARGS...` and speaks GEN / T / DONE with it over stdin and stdout.
//
//   ds4n-serve --serve --backbone B --store S --gguf G [--dspark DIR] [--slots N] [--ram-records R]
//              [--prior PROFILE] [--ram-policy recent|popular|victim] [--rewind-points N] [--sweep-min N]
//              [--checkpoint-gb G] [--keep-min N] [--split-min N] [--stops 1,128803]
//
// One request at a time (READY ... stop: no multi, no PAUSE), from any of the conversations it keeps
// (engine/conversation): an agent's tool loop and a chat's next turn read only their new tokens; a
// conversation another one took the GPUs from comes back from its KV checkpoint in host memory
// (--checkpoint-gb of them, the least recently used dropped first); a subagent's first prompt
// resumes where it leaves the system prompt it shares with its parent. Stdout carries the protocol
// only (NOTE lines tell the server's log what happened to the conversations); everything else goes
// to stderr (the server's log).
#include <unistd.h>

#include <cerrno>
#include <charconv>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <deque>
#include <mutex>
#include <random>
#include <string>
#include <thread>

#include "core/wire.hpp"
#include "engine/conversation.cuh"
#include "engine/device_layer.cuh"

using namespace ds4n;

namespace {

[[noreturn]] void die(const std::string& m) {
  std::fprintf(stderr, "ds4n-serve: %s\n", m.c_str());
  std::exit(1);
}

void out(const std::string& line) {
  std::fputs(line.c_str(), stdout);
  std::fflush(stdout);
}

/**
 * Stdin on a thread of its own, so a request in flight sees STOP and WRAP: those set flags; GEN and
 * QUIT (and lines that do not parse) queue for the main thread. read(2), not std::cin: glibc's exit()
 * waits on stdin's lock, which a getline waiting for input holds.
 */
struct Input {
  using Line = Result<wire::Command, wire::WireError>;
  std::mutex mu;
  std::condition_variable cv;
  std::deque<Line> lines;
  bool eof = false;
  std::atomic<bool> stop{false}, wrap{false};

  void start() {
    std::thread([this] {
      std::mt19937_64 seeds(std::random_device{}());  // a GEN naming no seed samples with a fresh one
      std::string buf;
      char chunk[1 << 16];
      for (;;) {
        const size_t nl = buf.find('\n');
        if (nl == std::string::npos) {
          const ssize_t n = ::read(0, chunk, sizeof chunk);
          if (n < 0 && errno == EINTR) continue;
          if (n > 0) {
            buf.append(chunk, static_cast<size_t>(n));
            continue;
          }
          if (buf.empty()) break;
          buf.push_back('\n');  // a last line without its newline
          continue;
        }
        Line line = wire::parse(std::string_view(buf).substr(0, nl), model::kVocab, seeds());
        buf.erase(0, nl + 1);
        if (line && std::holds_alternative<wire::Stop>(line.value())) {
          stop = true;
          continue;
        }
        if (line && std::holds_alternative<wire::Wrap>(line.value())) {
          wrap = true;
          continue;
        }
        if (line && std::holds_alternative<wire::Quit>(line.value())) stop = true;  // ends a request in flight too
        std::lock_guard lock(mu);
        lines.push_back(std::move(line));
        cv.notify_one();
      }
      std::lock_guard lock(mu);
      eof = true;
      stop = true;
      cv.notify_one();
    }).detach();
  }
  /** The next queued line; none at the end of input. */
  std::optional<Line> next() {
    std::unique_lock lock(mu);
    cv.wait(lock, [&] { return !lines.empty() || eof; });
    if (lines.empty()) return std::nullopt;
    Line l = std::move(lines.front());
    lines.pop_front();
    return l;
  }
};

/** --stops: token ids, comma separated. */
std::vector<int32_t> ids_of(const std::string& s) {
  std::vector<int32_t> v;
  for (size_t at = 0; at < s.size();) {
    const size_t comma = std::min(s.find(',', at), s.size());
    int32_t id = -1;
    const auto [end, ec] = std::from_chars(s.data() + at, s.data() + comma, id);
    if (ec != std::errc{} || end != s.data() + comma || id < 0 || id >= model::kVocab) die("--stops: not a token id list: " + s);
    v.push_back(id);
    at = comma + 1;
  }
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGPIPE, SIG_IGN);  // a server already gone must not end us mid-write
  engine::FullModelConfig mc;
  engine::ConversationConfig cc;
  std::string stops;
  bool serve = false;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    if (a == "--serve") {
      serve = true;
      continue;
    }
    if (i + 1 >= argc) die("missing value for " + a);
    const std::string v = argv[++i];
    if (a == "--backbone") mc.backbone = v;
    else if (a == "--store") mc.store = v;
    else if (a == "--gguf") mc.gguf = v;
    else if (a == "--dspark") mc.dspark = v;
    else if (a == "--slots") mc.slots = static_cast<uint32_t>(std::stoul(v));
    else if (a == "--ram-records") mc.ram_records = static_cast<uint32_t>(std::stoul(v));
    else if (a == "--prior") mc.prior = v;
    else if (a == "--ram-policy") {
      const auto policy = engine::ram_policy_named(v);
      if (!policy) die("--ram-policy takes recent, popular or victim: " + v);
      mc.ram_policy = *policy;
    }
    else if (a == "--rewind-points") cc.rewind_points = static_cast<uint32_t>(std::stoul(v));
    else if (a == "--sweep-min") cc.sweep_min = std::stoul(v);
    else if (a == "--checkpoint-gb") cc.checkpoint_budget = static_cast<uint64_t>(std::stod(v) * 1e9);
    else if (a == "--keep-min") cc.rewind.keep_min = std::stoul(v);
    else if (a == "--split-min") cc.rewind.split_min = std::stoul(v);
    else if (a == "--stops") stops = v;
    // Maya's server adds these for its own engines: the context is the executor's, and the two GPUs
    // split every layer (tensor parallel), not the layers.
    else if (a == "--max-context") {}
    else if (a == "--layer-split") {
      if (v != "auto") die("--layer-split " + v + ": ds4-native splits every layer across the two GPUs");
    }
    else die("unknown argument " + a);
  }
  if (!serve || mc.backbone.empty() || mc.store.empty() || mc.gguf.empty())
    die("usage: ds4n-serve --serve --backbone B --store S --gguf G [--dspark DIR] [--slots N] [--ram-records R]");

  engine::FullModel m = [&] {
    try {
      return engine::load_full_model(mc);
    } catch (const std::exception& e) {
      die(e.what());
    }
  }();
  cc.stops = stops.empty() ? std::vector<int32_t>{m.eos, 128803} : ids_of(stops);  // EOS, <｜User｜>
  engine::Conversation conversation(m, cc);
  try {
    conversation.warm_up();
  } catch (const std::exception& e) {
    die(std::string("warm-up: ") + e.what());
  }

  char info[320];
  std::snprintf(info, sizeof info,
                "INFO engine=ds4-native engine_kind=ds4-native context=%lld vram_slots=%u ram_records=%u dspark=%d "
                "rewind_points=%u checkpoint_gb=%.1f logit_bias=1\n",
                static_cast<long long>(engine::kMaxCompressed), mc.slots, mc.ram_records, m.dspark ? 1 : 0, cc.rewind_points,
                static_cast<double>(cc.checkpoint_budget) / 1e9);
  out(info);
  out(wire::ready_line(engine::kMaxCompressed));
  std::fprintf(stderr, "ds4n-serve: ready (GEN <max_new> [keys] <ids>; STOP, WRAP, QUIT)\n");

  Input in;
  in.start();
  const engine::ReplyEvents events{[](int32_t t) { out(wire::token_line(t)); },
                                   [](const wire::Progress& p) { out(wire::progress_line(p)); },
                                   [](const wire::Stat& s) { out(wire::stat_line(s)); },
                                   [](const std::string& n) {
                                     std::fprintf(stderr, "ds4n-serve: %s\n", n.c_str());
                                     out("NOTE sessions: " + n + "\n");
                                   },
                                   &in.stop, &in.wrap};
  while (const std::optional<Input::Line> line = in.next()) {
    if (!*line) {
      out("ERR " + line->error().what + "\n");
      continue;
    }
    if (std::holds_alternative<wire::Quit>(line->value())) break;
    const auto* g = std::get_if<wire::Generate>(&line->value());
    if (g == nullptr) continue;
    in.stop = false;  // a STOP from before this request is not for it
    if (static_cast<int64_t>(g->prompt.size()) >= engine::kMaxCompressed) {
      out("ERR the prompt (" + std::to_string(g->prompt.size()) + " tokens) leaves no room in the context of " +
          std::to_string(engine::kMaxCompressed) + "\n");
      continue;
    }
    try {
      out(wire::done_line(conversation.run(*g, events)));
    } catch (const std::exception& e) {
      const std::string what = e.what();
      std::fprintf(stderr, "ds4n-serve: %s\n", what.c_str());
      out("ERR " + what + "\n");
      out(wire::done_line({0, g->prompt.size(), 0.0, 0.0, reply::Finish::Cancel, 0, 0, 0}));
      // Non-finite logits leave the engine sound (the Session starts over); a CUDA or I/O failure may
      // not: end, and the server starts the engine again.
      if (what.find("non-finite") == std::string::npos) return 2;
    }
  }
  m.decoder->sync();  // nothing in flight reads the Expert cache once it closes
  return 0;
}

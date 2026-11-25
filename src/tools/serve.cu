// ds4n-serve: the whole model behind Project Maya's engine protocol (core/wire), so Maya's
// OpenAI-compatible server (serve/server.py, MIT) serves it on /v1 as it serves its own engines:
// the server renders the chat (DeepSeek-V4.1's encoding), tokenizes, starts `ds4n-serve --serve
// ARGS...` and speaks GEN / T / DONE with it over stdin and stdout.
//
//   ds4n-serve --serve --backbone B --store S[,S...] --gguf G [--dspark DIR] [--slots N] [--ram-records R]
//              [--lanes N] [--prior PROFILE] [--ram-policy recent|popular|victim] [--rewind-points N]
//              [--sweep-min N] [--checkpoint-gb G] [--keep-min N] [--split-min N] [--stops 1,128803]
//              [--sweep-slots N] [--round-graphs-mb N] [--replica-depth N] [--engram PATH]
//
// --store S,S,... (or --store repeated): Store replicas, copies of the Expert store on separate
// drives; a step's reads spread over them, the faster drive taking more (--replica-depth N, default
// 16: each replica's reads in flight at most).
// --engram PATH: Engram rows from an Engram copy (ds4n-engram-copy) on a drive of its own, read by
// direct I/O, instead of from the GGUF; checked against the GGUF's tables at open.
// --sweep-slots N (default 512, at least 384): the Slots prompt sweeps load into, VRAM that --slots
// cannot use; fewer of them leave room for that many more --slots.
// --round-graphs-mb N (default 512): the device memory a GPU's captured Round graphs may hold
// together; past it the least recently launched are destroyed and captured again when needed
// (core/round_graphs). VRAM that --slots must leave free.
//
// Requests from any of the conversations it keeps (engine/conversation): an agent's tool loop and a
// chat's next turn read only their new tokens; a conversation another one took a Lane from comes
// back from its KV checkpoint in host memory (--checkpoint-gb of them, the least recently used
// dropped first); a subagent's first prompt resumes where it leaves the system prompt it shares with
// its parent. With --lanes N (default 1) up to N requests run at once, their rows sharing Rounds
// (READY ... stop multi=N; each Lane's caches take ~730 MiB a GPU: give --slots that many fewer, ~78 a
// Lane); with one Lane requests take turns, as the server expects of an engine without multi. Stdout
// carries the protocol only (NOTE lines tell the server's log what happened to the conversations);
// everything else goes to stderr (the server's log).
#include <unistd.h>

#include <cerrno>
#include <charconv>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <deque>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>

#include "core/store_replicas.hpp"
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
 * Stdin on a thread of its own, its lines queued in order for the main thread, which takes them
 * between the engine's passes: a STOP, a WRAP or another GEN reaches requests in flight. read(2), not
 * std::cin: glibc's exit() waits on stdin's lock, which a getline waiting for input holds.
 */
struct Input {
  using Line = Result<wire::Command, wire::WireError>;
  std::mutex mu;
  std::condition_variable cv;
  std::deque<Line> lines;
  bool eof = false;

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
        std::lock_guard lock(mu);
        lines.push_back(std::move(line));
        cv.notify_one();
      }
      std::lock_guard lock(mu);
      eof = true;
      cv.notify_one();
    }).detach();
  }
  /** The next queued line, waiting for one when `wait`; none: nothing is queued (waiting: the input ended). */
  std::optional<Line> next(bool wait) {
    std::unique_lock lock(mu);
    if (wait) cv.wait(lock, [&] { return !lines.empty() || eof; });
    if (lines.empty()) return std::nullopt;
    Line l = std::move(lines.front());
    lines.pop_front();
    return l;
  }
  /** Whether the input ended and every line of it was taken. */
  bool ended() {
    std::lock_guard lock(mu);
    return eof && lines.empty();
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

/** A request in flight: its Lane, what its lines are tagged with, its prompt's length. */
struct Flight {
  lane::LaneId lane;
  std::optional<wire::Rid> rid;
  size_t prompt;
};

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
  engine::FullModelConfig mc;
  engine::ConversationConfig cc;
  std::string stops;
  bool serve = false, points_given = false;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    if (a == "--serve") {
      serve = true;
      continue;
    }
    if (i + 1 >= argc) die("missing value for " + a);
    const std::string v = argv[++i];
    if (a == "--backbone") mc.backbone = v;
    else if (a == "--store") mc.stores = replicas::paths_named(v, std::move(mc.stores));
    else if (a == "--replica-depth") mc.replica_depth = static_cast<uint32_t>(std::stoul(v));
    else if (a == "--gguf") mc.gguf = v;
    else if (a == "--engram") mc.engram = v;
    else if (a == "--dspark") mc.dspark = v;
    else if (a == "--slots") mc.slots = static_cast<uint32_t>(std::stoul(v));
    else if (a == "--ram-records") mc.ram_records = static_cast<uint32_t>(std::stoul(v));
    else if (a == "--sweep-slots") mc.sweep_slots = static_cast<uint32_t>(std::stoul(v));
    else if (a == "--round-graphs-mb") mc.round_graph_bytes = std::stoull(v) << 20;
    else if (a == "--lanes") {
      const auto lanes = lane::LaneCount::of(std::stoll(v));
      if (!lanes) die("--lanes: " + lanes.error().what);
      mc.lanes = lanes.value();
    }
    else if (a == "--prior") mc.prior = v;
    else if (a == "--ram-policy") {
      const auto policy = engine::ram_policy_named(v);
      if (!policy) die("--ram-policy takes recent, popular or victim: " + v);
      mc.ram_policy = *policy;
    }
    else if (a == "--rewind-points") {
      cc.rewind_points = static_cast<uint32_t>(std::stoul(v));
      points_given = true;
    }
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
  if (!serve || mc.backbone.empty() || mc.stores.empty() || mc.gguf.empty())
    die("usage: ds4n-serve --serve --backbone B --store S[,S...] --gguf G [--engram PATH] [--dspark DIR] [--slots N] [--ram-records R] [--lanes N]");
  // More Rewind points than Lanes: no Lane's last prompt's end is lost to another Lane's points
  // (rewind::next_buffer), so each Session's next turn resumes there.
  const auto lanes = static_cast<uint32_t>(mc.lanes.value());
  if (!points_given) cc.rewind_points = std::max(cc.rewind_points, lanes + 1);
  else if (lanes > 1 && cc.rewind_points != 0 && cc.rewind_points <= lanes)  // a Lane's points spare the other Lanes' turns
    die("--rewind-points " + std::to_string(cc.rewind_points) + ": with " + std::to_string(lanes) + " Lanes at least " +
        std::to_string(lanes + 1) + ", or 0 for none");

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
                "rewind_points=%u checkpoint_gb=%.1f logit_bias=1 lanes=%u\n",
                static_cast<long long>(engine::kMaxCompressed), mc.slots, mc.ram_records, m.dspark ? 1 : 0, cc.rewind_points,
                static_cast<double>(cc.checkpoint_budget) / 1e9, lanes);
  out(info);
  out(wire::ready_line(engine::kMaxCompressed, mc.lanes));  // one Lane: no multi=, and so no rids
  std::fprintf(stderr, "ds4n-serve: ready, %u Lane(s) (GEN <max_new> [keys] <ids>; STOP, WRAP, QUIT)\n", lanes);

  Input in;
  in.start();
  std::map<uint64_t, Flight> flights;  // the requests in flight, by the order they were admitted in
  uint64_t admitted = 0;
  std::deque<wire::Generate> waiting;  // GENs naming no rid that found every Lane busy: they take turns
  bool wrapping = false;               // the server is stopping: replies close their reasoning
  bool quit = false;

  // Admits `g`, or answers why not; false: it named no rid and every Lane is busy, so it waits its turn.
  const auto admit = [&](const wire::Generate& g) {
    const uint64_t n = admitted++;
    engine::ReplyEvents ev;
    ev.token = [rid = g.rid](int32_t t) { out(wire::token_line(rid, t)); };
    ev.progress = [rid = g.rid](const wire::Progress& p) { out(wire::progress_line(rid, p)); };
    ev.stat = [](const wire::Stat& s) { out(wire::stat_line(s)); };
    ev.note = [](const std::string& note) {
      std::fprintf(stderr, "ds4n-serve: %s\n", note.c_str());
      out("NOTE sessions: " + note + "\n");
    };
    ev.done = [&flights, n, rid = g.rid](const wire::Done& d) {
      out(wire::done_line(rid, d));
      flights.erase(n);
    };
    // Non-finite logits leave the engine sound (that Lane's Session starts over): the others go on.
    ev.failed = [&flights, n, rid = g.rid, prompt = g.prompt.size()](const std::string& what) {
      std::fprintf(stderr, "ds4n-serve: %s\n", what.c_str());
      out(failed_lines(rid, prompt, what));
      flights.erase(n);
    };
    // A failure while it is taken in (a checkpoint, the GPUs) ends the engine like any other, and this
    // request is not in flight yet: it gets its own answer first.
    const auto lane = [&] {
      try {
        return conversation.admit(g, std::move(ev));
      } catch (const std::exception& e) {
        out(failed_lines(g.rid, g.prompt.size(), e.what()));
        throw;
      }
    }();
    if (!lane) {
      if (!lane.error().busy) out(wire::error_line(g.rid, lane.error().what));
      else if (g.rid) out(wire::busy_line(*g.rid, mc.lanes));
      else return false;
      return true;
    }
    flights.emplace(n, Flight{lane.value(), g.rid, g.prompt.size()});
    if (wrapping) conversation.wrap(lane.value());
    return true;
  };
  const auto command = Overloaded{
      [&](const wire::Generate& g) {
        if (!waiting.empty() || !admit(g)) waiting.push_back(g);
      },
      [&](const wire::Stop& s) {  // the request it names, or naming none every one in flight
        for (const auto& [n, f] : flights)
          if (!s.rid || f.rid == s.rid) conversation.stop(f.lane);
      },
      [&](const wire::Wrap&) {
        wrapping = true;
        for (const auto& [n, f] : flights) conversation.wrap(f.lane);
      },
      [&](const wire::Quit&) { quit = true; },
  };

  try {
    while (!quit) {
      while (!waiting.empty() && admit(waiting.front())) waiting.pop_front();
      // Every line that arrived, then one pass; with nothing in flight, wait for a line.
      while (!quit) {
        const std::optional<Input::Line> line = in.next(conversation.idle());
        if (!line) break;
        if (*line) std::visit(command, line->value());
        else out(wire::error_line(line->error().rid, line->error().what));
      }
      if (in.ended()) quit = true;
      if (quit)  // requests in flight end ("cancel"), each with its DONE
        for (const auto& [n, f] : flights) conversation.stop(f.lane);
      conversation.step();
    }
  } catch (const std::exception& e) {
    // A CUDA or I/O failure may leave the engine unsound: every request in flight fails, and the
    // server starts the engine again.
    std::fprintf(stderr, "ds4n-serve: %s\n", e.what());
    for (const auto& [n, f] : flights) out(failed_lines(f.rid, f.prompt, e.what()));
    return 2;
  }
  m.decoder->sync();  // nothing in flight reads the Expert cache once it closes
  return 0;
}

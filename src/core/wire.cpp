#include "core/wire.hpp"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <type_traits>

namespace ds4n::wire {

namespace {

using R = Result<Command, WireError>;

template <class T>
std::optional<T> number(std::string_view s) {
  T v{};
  const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
  if (ec != std::errc{} || end != s.data() + s.size()) return std::nullopt;
  return v;
}

/** `s` split at `sep` (an empty piece where two meet). */
std::vector<std::string_view> split(std::string_view s, char sep) {
  std::vector<std::string_view> out;
  for (size_t at = 0;;) {
    const size_t next = s.find(sep, at);
    out.push_back(s.substr(at, next == std::string_view::npos ? std::string_view::npos : next - at));
    if (next == std::string_view::npos) return out;
    at = next + 1;
  }
}

/** A GEN line's keys as read so far (unchecked: parse_generate checks them once all are in). */
struct Keys {
  sample::Settings sampling;
  std::optional<uint64_t> seed;
  std::optional<bool> spec;
  std::optional<int64_t> think_budget;
  std::optional<int32_t> think_end;
  std::optional<uint64_t> rid;
};

/** A key=value into `k` (unknown keys skipped); false when the value is malformed. */
bool read_key(Keys& k, std::string_view key, std::string_view value) {
  auto set = [&](auto& field) {
    const auto v = number<std::remove_reference_t<decltype(field)>>(value);
    if (v) field = *v;
    return v.has_value();
  };
  auto set_optional = [&](auto& field) {
    field = number<typename std::remove_reference_t<decltype(field)>::value_type>(value);
    return field.has_value();
  };
  sample::Settings& s = k.sampling;
  if (key == "temperature") return set(s.temperature);
  if (key == "top_p") return set(s.top_p);
  if (key == "top_k") return set(s.top_k);
  if (key == "min_p") return set(s.min_p);
  if (key == "seed") return set_optional(k.seed);
  if (key == "penalty_last_n") return set(s.penalties.last_n);
  if (key == "penalty_repeat") return set(s.penalties.repeat);
  if (key == "penalty_freq") return set(s.penalties.frequency);
  if (key == "penalty_present") return set(s.penalties.presence);
  if (key == "think_budget") return set_optional(k.think_budget);
  if (key == "think_end") return set_optional(k.think_end);
  if (key == "rid") return set_optional(k.rid);
  if (key == "spec") {
    const auto v = number<int>(value);
    if (v) k.spec = *v >= 2;
    return v.has_value();
  }
  if (key == "logit_bias") {
    for (const std::string_view entry : split(value, ',')) {
      const size_t colon = entry.find(':');
      if (colon == std::string_view::npos) return false;
      const auto id = number<int32_t>(entry.substr(0, colon));
      const auto bias = number<float>(entry.substr(colon + 1));
      if (!id || !bias) return false;
      s.bias.push_back({*id, *bias});
    }
    return true;
  }
  return true;  // a key for another engine
}

R parse_generate(std::string_view rest, int32_t vocab, uint64_t fallback_seed) {
  const std::vector<std::string_view> words = [&] {
    std::vector<std::string_view> w;
    for (const std::string_view piece : split(rest, ' '))
      if (!piece.empty()) w.push_back(piece);
    return w;
  }();
  if (words.size() < 2) return R::err({"expected: GEN <max_new> [key=value ...] <id,id,...>"});
  const auto max_new = number<int64_t>(words.front());
  if (!max_new || *max_new <= 0) return R::err({"max_new must be a positive number"});
  Keys k;
  for (size_t i = 1; i + 1 < words.size(); i++) {
    const size_t eq = words[i].find('=');
    if (eq == std::string_view::npos) return R::err({"the token ids must come last"});
    const std::string_view key = words[i].substr(0, eq), value = words[i].substr(eq + 1);
    if (!read_key(k, key, value)) return R::err({"malformed " + std::string(key)});
  }
  std::vector<int32_t> prompt;
  for (const std::string_view id : split(words.back(), ',')) {
    const auto t = number<int32_t>(id);
    if (!t || *t < 0 || *t >= vocab) return R::err({"token id '" + std::string(id) + "' is not in the vocabulary"});
    prompt.push_back(*t);
  }
  if (k.think_budget.has_value() != k.think_end.has_value()) return R::err({"think_budget and think_end go together"});
  std::optional<reply::ThinkBudget> think;
  if (k.think_budget && *k.think_budget > 0) {
    if (*k.think_end < 0 || *k.think_end >= vocab) return R::err({"think_end is not in the vocabulary"});
    think = reply::ThinkBudget{*k.think_budget, *k.think_end};
  }
  k.sampling.seed = k.seed.value_or(fallback_seed);
  auto sampling = sample::make_sampling(std::move(k.sampling), vocab);
  if (!sampling) return R::err({sampling.error().what});
  const std::optional<Rid> rid = k.rid ? std::optional(Rid{*k.rid}) : std::nullopt;
  return R::ok(Generate{*max_new, std::move(sampling).value(), think, k.spec, std::move(prompt), rid});
}

std::string formatted(const char* format, auto... args) {
  char buf[256];
  const int n = std::snprintf(buf, sizeof buf, format, args...);
  return std::string(buf, n < 0 ? 0 : std::min<size_t>(static_cast<size_t>(n), sizeof buf - 1));
}

/** The rid a GEN or GENI line names among its words, if it does: a line that does not parse still says whose it was. */
std::optional<Rid> rid_of(std::string_view rest) {
  for (const std::string_view word : split(rest, ' '))
    if (word.starts_with("rid="))
      if (const auto id = number<uint64_t>(word.substr(4))) return Rid{*id};
  return std::nullopt;
}

/** `verb`, then the rid when there is one: how each of a request's lines begins. */
std::string head(const char* verb, std::optional<Rid> rid) {
  return rid ? formatted("%s %llu", verb, static_cast<unsigned long long>(rid->id)) : std::string(verb);
}

}  // namespace

Result<Command, WireError> parse(std::string_view line, int32_t vocab, uint64_t fallback_seed) {
  if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
  if (line == "STOP") return R::ok(Stop{});
  if (line.starts_with("STOP ")) {
    const auto id = number<uint64_t>(line.substr(5));
    if (!id) return R::err({"STOP names a request by its rid, a number", std::nullopt});
    return R::ok(Stop{Rid{*id}});
  }
  if (line == "WRAP") return R::ok(Wrap{});
  if (line == "QUIT") return R::ok(Quit{});
  if (line.starts_with("GENI "))
    return R::err({"this engine reads no pictures (it has no vision tower)", rid_of(line.substr(5))});
  if (line.starts_with("GEN ")) {
    R command = parse_generate(line.substr(4), vocab, fallback_seed);
    if (!command) return R::err({command.error().what, rid_of(line.substr(4))});
    return command;
  }
  return R::err({"unsupported command '" + std::string(line.substr(0, line.find(' '))) + "' (GEN, STOP, WRAP and QUIT)", std::nullopt});
}

std::string ready_line(int64_t context, lane::LaneCount lanes) {
  if (lanes.value() == 1) return formatted("READY %lld stop\n", static_cast<long long>(context));
  return formatted("READY %lld stop multi=%d\n", static_cast<long long>(context), lanes.value());
}

std::string token_line(std::optional<Rid> rid, int32_t token) { return head("T", rid) + formatted(" %d\n", token); }

std::string progress_line(std::optional<Rid> rid, const Progress& p) {
  const double rate = p.ms > 0 ? 1000.0 * static_cast<double>(p.read - p.reused) / p.ms : 0.0;
  return head("PP", rid) + formatted(" %zu %zu %.1f %.1f %zu\n", p.read, p.total, p.ms, rate, p.reused);
}

std::string done_line(std::optional<Rid> rid, const Done& d) {
  return head("DONE", rid) + formatted(" %lld %zu %.1f %.1f %s %lld %lld %zu 0 0\n", static_cast<long long>(d.written), d.prompt,
                                       d.prompt_ms, d.decode_ms, reply::name(d.finish), static_cast<long long>(d.drafts_kept),
                                       static_cast<long long>(d.drafts_offered), d.reused);
}

std::string error_line(std::optional<Rid> rid, std::string_view message) {
  std::string line = head("ERR", rid) + " ";
  for (const char c : message) line.push_back(c == '\n' || c == '\r' ? ' ' : c);
  return line + "\n";
}

std::string busy_line(Rid rid, lane::LaneCount lanes) {
  return error_line(rid, "all " + std::to_string(lanes.value()) + " conversation states are busy");
}

std::string stat_line(const Stat& s) {
  const double per_token = s.written > 0 ? s.ms / static_cast<double>(s.written) : 0.0;
  std::string line = formatted("STAT tok_s=%.2f ms_tok=%.2f vram_hit=%.4f", s.ms > 0 ? 1000.0 * s.written / s.ms : 0.0,
                               per_token, s.vram_hit);
  if (s.drafts >= 0) line += formatted(" drafts=%.3f", s.drafts);
  return line + "\n";
}

}  // namespace ds4n::wire

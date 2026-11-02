#include "core/reply.hpp"

#include <algorithm>

namespace ds4n::reply {

Next after(Reply r, const Limits& l, int32_t token, bool wrap, bool room) {
  Next n{{r.written + 1, r.thinking}, std::nullopt, std::nullopt};
  if (std::find(l.stops.begin(), l.stops.end(), token) != l.stops.end()) {
    n.finish = Finish::Stop;
    return n;
  }
  if (n.reply.written >= l.max_new || !room) {
    n.finish = Finish::Length;
    return n;
  }
  if (r.thinking && l.think) {
    const bool closed = token == l.think->end;
    if (closed || n.reply.written >= l.think->tokens || wrap) {
      n.reply.thinking = false;
      if (!closed) n.forced = l.think->end;
    }
  }
  return n;
}

}  // namespace ds4n::reply

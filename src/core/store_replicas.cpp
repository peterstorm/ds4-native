#include "core/store_replicas.hpp"

#include <stdexcept>

namespace ds4n::replicas {

std::optional<size_t> next_replica(std::span<const uint32_t> in_flight, uint32_t depth) {
  std::optional<size_t> best;
  for (size_t r = 0; r < in_flight.size(); r++)
    if (in_flight[r] < depth && (!best || in_flight[r] < in_flight[*best])) best = r;
  return best;
}

Spread::Spread(size_t reads, size_t replicas, uint32_t depth)
    : depth_(depth), in_flight_(replicas, 0), replica_of_(reads, 0), state_(reads, Read::Unsent) {
  if (replicas == 0) throw std::invalid_argument("a Spread needs at least one replica");
  if (depth == 0) throw std::invalid_argument("a Spread needs a depth of at least one");
}

std::vector<Send> Spread::send() {
  std::vector<Send> out;
  while (!failed_ && sent_ < state_.size()) {
    const auto r = next_replica(in_flight_, depth_);
    if (!r) break;
    out.push_back({sent_, *r});
    replica_of_[sent_] = *r;
    state_[sent_] = Read::InFlight;
    in_flight_[*r]++;
    sent_++;
  }
  return out;
}

void Spread::refuse(std::span<const Send> sends) {
  for (const Send& s : sends) {
    if (state_.at(s.read) != Read::InFlight || s.read + sends.size() < sent_)
      throw std::logic_error("refused a read that is not among the last ones sent");
    state_[s.read] = Read::Unsent;
    in_flight_[s.replica]--;
  }
  sent_ -= sends.size();
  failed_ = true;
}

Landing Spread::complete(size_t read, bool whole) {
  if (state_.at(read) != Read::InFlight) throw std::logic_error("a read completed that was not in flight");
  state_[read] = Read::Completed;
  in_flight_[replica_of_[read]]--;
  done_++;
  if (whole) return failed_ ? Landing::Drop : Landing::Land;
  if (failed_) return Landing::Drop;
  failed_ = true;
  return Landing::Fail;
}

bool Spread::settled() const { return done_ == sent_ && (failed_ || sent_ == state_.size()); }

std::vector<std::string> paths_named(std::string_view arg, std::vector<std::string> named) {
  std::vector<std::string> out = std::move(named);
  while (!arg.empty()) {
    const size_t comma = arg.find(',');
    const std::string_view part = arg.substr(0, comma);
    if (!part.empty()) out.emplace_back(part);
    if (comma == std::string_view::npos) break;
    arg.remove_prefix(comma + 1);
  }
  return out;
}

}  // namespace ds4n::replicas

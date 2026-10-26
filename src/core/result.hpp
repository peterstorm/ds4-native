#pragma once
// Either for the functional core: a value or a domain error. Core code returns
// errors as values and never throws; the shell decides what an error means.
#include <cstddef>
#include <utility>
#include <variant>

namespace ds4n {

template <class T, class E>
class Result {
 public:
  static Result ok(T value) { return Result(std::in_place_index<0>, std::move(value)); }
  static Result err(E error) { return Result(std::in_place_index<1>, std::move(error)); }

  bool is_ok() const { return v_.index() == 0; }
  explicit operator bool() const { return is_ok(); }

  const T& value() const& { return std::get<0>(v_); }
  T& value() & { return std::get<0>(v_); }
  T&& value() && { return std::get<0>(std::move(v_)); }
  const E& error() const& { return std::get<1>(v_); }

 private:
  template <std::size_t I, class... A>
  explicit Result(std::in_place_index_t<I> tag, A&&... args) : v_(tag, std::forward<A>(args)...) {}
  std::variant<T, E> v_;
};

}  // namespace ds4n

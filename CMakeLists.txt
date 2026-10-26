cmake_minimum_required(VERSION 3.28)
project(ds4_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Functional core: pure C++, no CUDA, no I/O. Everything here is unit-testable on CPU.
add_library(ds4n_core STATIC
  src/core/gguf.cpp
  src/core/store_layout.cpp
  src/core/ledger.cpp
  src/core/expert_cache.cpp
  src/core/mx_reference.cpp
)
target_include_directories(ds4n_core PUBLIC src)
target_compile_options(ds4n_core PRIVATE -Wall -Wextra -Wpedantic -Werror)

# Imperative shell: files and io_uring.
find_library(URING_LIB uring REQUIRED)
add_library(ds4n_io STATIC
  src/io/file.cpp
  src/io/uring_reader.cpp
)
target_link_libraries(ds4n_io PUBLIC ds4n_core ${URING_LIB})
target_compile_options(ds4n_io PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(ds4n-store-build src/tools/store_build.cpp)
target_link_libraries(ds4n-store-build PRIVATE ds4n_io)
add_executable(ds4n-readbench src/tools/read_bench.cpp)
target_link_libraries(ds4n-readbench PRIVATE ds4n_io)

enable_testing()
find_package(doctest REQUIRED)
add_executable(ds4n_tests
  tests/main.cpp
  tests/core/test_gguf.cpp
  tests/core/test_store_layout.cpp
  tests/core/test_ledger.cpp
  tests/core/test_expert_cache.cpp
  tests/core/test_stats.cpp
  tests/core/test_mx_reference.cpp
)
target_link_libraries(ds4n_tests PRIVATE ds4n_core doctest::doctest)
target_compile_options(ds4n_tests PRIVATE -Wall -Wextra)
add_test(NAME core COMMAND ds4n_tests)

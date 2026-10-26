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
  src/core/mx_tiles.cpp
  src/core/hc_reference.cpp
  src/core/attn_reference.cpp
  src/core/engram.cpp
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

# GPU kernels (sm_120a: block-scaled MMA needs the arch-specific target).
option(DS4N_CUDA "Build CUDA kernels and GPU tools" ON)
if(DS4N_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_ARCHITECTURES OFF)
  set(DS4N_CUDA_FLAGS -gencode=arch=compute_120a,code=sm_120a -lineinfo -Xcompiler=-Wall)
  add_library(ds4n_cuda STATIC src/cuda/mx_gemv.cu src/cuda/glue.cu src/cuda/attn.cu)
  target_compile_options(ds4n_cuda PRIVATE ${DS4N_CUDA_FLAGS})
  target_link_libraries(ds4n_cuda PUBLIC ds4n_core)
  add_executable(ds4n-mx-check src/tools/mx_check.cu)
  target_compile_options(ds4n-mx-check PRIVATE ${DS4N_CUDA_FLAGS})
  target_link_libraries(ds4n-mx-check PRIVATE ds4n_cuda ds4n_io)
  find_package(CUDAToolkit REQUIRED)
  add_executable(ds4n-attn-check src/tools/attn_check.cu)
  target_compile_options(ds4n-attn-check PRIVATE ${DS4N_CUDA_FLAGS})
  target_link_libraries(ds4n-attn-check PRIVATE ds4n_cuda)
  add_executable(ds4n-glue-check src/tools/glue_check.cu)
  target_compile_options(ds4n-glue-check PRIVATE ${DS4N_CUDA_FLAGS})
  target_link_libraries(ds4n-glue-check PRIVATE ds4n_cuda)
  add_executable(ds4n-fp8-check src/tools/fp8_check.cu)
  target_compile_options(ds4n-fp8-check PRIVATE ${DS4N_CUDA_FLAGS})
  target_link_libraries(ds4n-fp8-check PRIVATE ds4n_cuda)
  add_executable(ds4n-miss-bench src/tools/miss_bench.cu)
  target_compile_options(ds4n-miss-bench PRIVATE ${DS4N_CUDA_FLAGS})
  target_link_libraries(ds4n-miss-bench PRIVATE ds4n_io CUDA::cudart)
endif()

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
  tests/core/test_mx_tiles.cpp
  tests/core/test_hc_reference.cpp
  tests/core/test_attn_reference.cpp
  tests/core/test_engram.cpp
)
target_link_libraries(ds4n_tests PRIVATE ds4n_core doctest::doctest)
target_compile_options(ds4n_tests PRIVATE -Wall -Wextra)
add_test(NAME core COMMAND ds4n_tests)

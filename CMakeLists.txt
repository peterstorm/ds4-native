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
  src/core/moe_reference.cpp
  src/core/safetensors.cpp
  src/core/glm53/store.cpp
  src/core/glm53/backbone.cpp
  src/core/glm53/config.cpp
  src/core/glm53/reference.cpp
  src/core/glm53/dequant.cpp
  src/core/glm53/precision.cpp
  src/core/glm53/exact_attention.cpp
  src/core/glm53/exact_moe.cpp
  src/core/glm53/moe_roundings.cpp
  src/core/glm53/moe_plan.cpp
  src/core/glm53/expert_tiles.cpp
  src/core/glm53/layer_expert_cache.cpp
  src/core/glm53/victim_tier.cpp
  src/core/glm53/sessions.cpp
  src/core/glm53/cohort.cpp
  src/core/glm53/ledger.cpp
  src/core/glm53/slot_profile.cpp
  src/core/glm53/deferral.cpp
  src/core/meminfo.cpp
  src/core/glm53/pruning.cpp
  src/core/indexer.cpp
  src/core/compressor.cpp
  src/core/layer_reference.cpp
  src/core/layer_weights.cpp
  src/core/compare.cpp
  src/core/cache_replay.cpp
  src/core/slot_map.cpp
  src/core/round_graphs.cpp
  src/core/session_mix.cpp
  src/core/shared_round.cpp
  src/core/speculation.cpp
  src/core/moe_groups.cpp
  src/core/row_index.cpp
  src/core/rewind.cpp
  src/core/sampler.cpp
  src/core/reply.cpp
  src/core/wire.cpp
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

# Engine shell (host side): loading Backbone layers.
add_library(ds4n_engine STATIC src/engine/host_layer.cpp src/engine/engram_source.cpp)
target_link_libraries(ds4n_engine PUBLIC ds4n_io)
target_compile_options(ds4n_engine PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(ds4n-store-build src/tools/store_build.cpp)
target_link_libraries(ds4n-store-build PRIVATE ds4n_io)
# GLM-5.3 shell: Backbone parts, the Expert store, and reading both back as FP32 weights. Its own library,
# so ds4n_io (linked by the DS4.1 engine) is unchanged.
add_library(ds4n_glm53_io STATIC src/io/glm53_backbone.cpp src/io/glm53_store.cpp src/io/glm53_weights.cpp)
target_link_libraries(ds4n_glm53_io PUBLIC ds4n_io)
target_compile_options(ds4n_glm53_io PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_executable(ds4n-glm53-readbench src/tools/glm53_read_bench.cpp)
target_link_libraries(ds4n-glm53-readbench PRIVATE ds4n_glm53_io)
add_executable(ds4n-glm53-store-build src/tools/glm53_store_build.cpp)
target_link_libraries(ds4n-glm53-store-build PRIVATE ds4n_glm53_io)
add_executable(ds4n-glm53-backbone-extract src/tools/glm53_backbone_extract.cpp)
target_link_libraries(ds4n-glm53-backbone-extract PRIVATE ds4n_glm53_io)
add_executable(ds4n-glm53-oracle-run src/tools/glm53_oracle_run.cpp)
target_link_libraries(ds4n-glm53-oracle-run PRIVATE ds4n_glm53_io)
add_executable(ds4n-readbench src/tools/read_bench.cpp)
target_link_libraries(ds4n-readbench PRIVATE ds4n_io)
add_executable(ds4n-cache-study src/tools/cache_study.cpp)
target_link_libraries(ds4n-cache-study PRIVATE ds4n_core)

# GPU kernels (sm_120a: block-scaled MMA needs the arch-specific target).
option(DS4N_CUDA "Build CUDA kernels and GPU tools" ON)
if(DS4N_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_ARCHITECTURES OFF)
  set(DS4N_CUDA_FLAGS -gencode=arch=compute_120a,code=sm_120a -lineinfo -Xcompiler=-Wall)
  add_library(ds4n_cuda STATIC src/cuda/mx_gemv.cu src/cuda/glue.cu src/cuda/attn.cu src/cuda/moe.cu src/cuda/exchange.cu src/cuda/indexer.cu src/cuda/select.cu src/cuda/compressor.cu src/cuda/engram.cu src/cuda/planes.cu src/cuda/resolve.cu src/cuda/fused.cu src/cuda/dspark.cu src/cuda/chunk.cu src/cuda/glm53/moe.cu src/cuda/glm53/grouped_moe.cu src/cuda/glm53/attention.cu src/cuda/glm53/indexer.cu src/cuda/glm53/attention_layer.cu)
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
  add_executable(ds4n-indexer-check src/tools/indexer_check.cu)
  target_compile_options(ds4n-indexer-check PRIVATE ${DS4N_CUDA_FLAGS})
  target_link_libraries(ds4n-indexer-check PRIVATE ds4n_cuda)
  add_executable(ds4n-compressor-check src/tools/compressor_check.cu)
  target_compile_options(ds4n-compressor-check PRIVATE ${DS4N_CUDA_FLAGS})
  target_link_libraries(ds4n-compressor-check PRIVATE ds4n_cuda ds4n_io)
  add_executable(ds4n-engram-check src/tools/engram_check.cu)
  target_compile_options(ds4n-engram-check PRIVATE ${DS4N_CUDA_FLAGS})
  target_link_libraries(ds4n-engram-check PRIVATE ds4n_cuda ds4n_io)
  add_library(ds4n_engine_cuda STATIC src/engine/device_layer.cu src/engine/decoder.cu src/engine/decoder_sweep.cu
                                    src/engine/expert_residency.cu src/engine/full_model.cu src/engine/conversation.cu)
  target_compile_options(ds4n_engine_cuda PRIVATE ${DS4N_CUDA_FLAGS})
  target_link_libraries(ds4n_engine_cuda PUBLIC ds4n_cuda ds4n_engine)
  add_executable(ds4n-layer-check src/tools/layer_check.cu)
  target_compile_options(ds4n-layer-check PRIVATE ${DS4N_CUDA_FLAGS})
  target_link_libraries(ds4n-layer-check PRIVATE ds4n_engine_cuda)
  add_executable(ds4n-model-run src/tools/model_run.cu)
  target_compile_options(ds4n-model-run PRIVATE ${DS4N_CUDA_FLAGS})
  target_link_libraries(ds4n-model-run PRIVATE ds4n_engine_cuda)
  add_executable(ds4n-serve src/tools/serve.cu)
  target_compile_options(ds4n-serve PRIVATE ${DS4N_CUDA_FLAGS})
  target_link_libraries(ds4n-serve PRIVATE ds4n_engine_cuda)
  add_executable(ds4n-serve-check src/tools/serve_check.cu)
  target_compile_options(ds4n-serve-check PRIVATE ${DS4N_CUDA_FLAGS})
  target_link_libraries(ds4n-serve-check PRIVATE ds4n_engine_cuda)
  add_executable(ds4n-exchange-check src/tools/exchange_check.cu)
  target_compile_options(ds4n-exchange-check PRIVATE ${DS4N_CUDA_FLAGS})
  target_link_libraries(ds4n-exchange-check PRIVATE ds4n_cuda)
  add_executable(ds4n-det-check src/tools/det_check.cu)
  target_compile_options(ds4n-det-check PRIVATE ${DS4N_CUDA_FLAGS})
  target_include_directories(ds4n-det-check PRIVATE src)
  add_executable(ds4n-moe-check src/tools/moe_check.cu)
  target_compile_options(ds4n-moe-check PRIVATE ${DS4N_CUDA_FLAGS})
  target_link_libraries(ds4n-moe-check PRIVATE ds4n_cuda ds4n_io)
  add_executable(ds4n-glm53-attn-check src/tools/glm53_attn_check.cu)
  target_compile_options(ds4n-glm53-attn-check PRIVATE ${DS4N_CUDA_FLAGS})
  target_link_libraries(ds4n-glm53-attn-check PRIVATE ds4n_cuda ds4n_glm53_io)
  add_executable(ds4n-glm53-moe-check src/tools/glm53_moe_check.cu)
  target_compile_options(ds4n-glm53-moe-check PRIVATE ${DS4N_CUDA_FLAGS})
  target_link_libraries(ds4n-glm53-moe-check PRIVATE ds4n_cuda ds4n_glm53_io)
  # GLM-5.3's engine: the decoder on two GPUs, and its gate.
  add_library(ds4n_glm53_engine STATIC src/engine/glm53/decoder.cu src/engine/glm53/conversation.cu)
  target_compile_options(ds4n_glm53_engine PRIVATE ${DS4N_CUDA_FLAGS})
  target_link_libraries(ds4n_glm53_engine PUBLIC ds4n_cuda ds4n_glm53_io ds4n_io)
  add_executable(ds4n-glm53-decode src/tools/glm53_decode.cu)
  target_compile_options(ds4n-glm53-decode PRIVATE ${DS4N_CUDA_FLAGS})
  target_link_libraries(ds4n-glm53-decode PRIVATE ds4n_glm53_engine)
  add_executable(ds4n-glm53-ledger src/tools/glm53_ledger.cu)
  target_compile_options(ds4n-glm53-ledger PRIVATE ${DS4N_CUDA_FLAGS})
  target_link_libraries(ds4n-glm53-ledger PRIVATE ds4n_glm53_engine)
  add_executable(ds4n-glm53-serve src/tools/glm53_serve.cu)
  target_compile_options(ds4n-glm53-serve PRIVATE ${DS4N_CUDA_FLAGS})
  target_link_libraries(ds4n-glm53-serve PRIVATE ds4n_glm53_engine)
  add_executable(ds4n-glm53-serve-check src/tools/glm53_serve_check.cu)
  target_compile_options(ds4n-glm53-serve-check PRIVATE ${DS4N_CUDA_FLAGS})
  target_link_libraries(ds4n-glm53-serve-check PRIVATE ds4n_glm53_engine)
  # The oracle tool gains its GPU path (--gpu) in a CUDA build.
  target_link_libraries(ds4n-glm53-oracle-run PRIVATE ds4n_cuda CUDA::cudart)
  target_compile_definitions(ds4n-glm53-oracle-run PRIVATE DS4N_GLM53_GPU)
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
  tests/core/test_moe_reference.cpp
  tests/core/test_det_math.cpp
  tests/core/test_safetensors.cpp
  tests/core/glm53/test_store.cpp
  tests/core/glm53/test_backbone.cpp
  tests/core/glm53/test_config.cpp
  tests/core/glm53/test_reference.cpp
  tests/core/glm53/test_dequant.cpp
  tests/core/glm53/test_precision.cpp
  tests/core/glm53/test_exact_attention.cpp
  tests/core/glm53/test_exact_moe.cpp
  tests/core/glm53/test_moe_roundings.cpp
  tests/core/glm53/test_moe_plan.cpp
  tests/core/glm53/test_expert_tiles.cpp
  tests/core/glm53/test_layer_expert_cache.cpp
  tests/core/glm53/test_victim_tier.cpp
  tests/core/glm53/test_sessions.cpp
  tests/core/glm53/test_cohort.cpp
  tests/core/glm53/test_ledger.cpp
  tests/core/glm53/test_slot_profile.cpp
  tests/core/glm53/test_deferral.cpp
  tests/core/glm53/test_latent_split.cpp
  tests/core/test_meminfo.cpp
  tests/core/glm53/test_pruning.cpp
  tests/core/test_indexer.cpp
  tests/core/test_compressor.cpp
  tests/core/test_compare.cpp
  tests/core/test_lane.cpp
  tests/core/test_round.cpp
  tests/core/test_cache_replay.cpp
  tests/core/test_slot_map.cpp
  tests/core/test_round_graphs.cpp
  tests/core/test_session_mix.cpp
  tests/core/test_shared_round.cpp
  tests/core/test_speculation.cpp
  tests/core/test_moe_groups.cpp
  tests/core/test_row_index.cpp
  tests/core/test_decoder_tail.cpp
  tests/core/test_rewind.cpp
  tests/core/test_sampler.cpp
  tests/core/test_reply.cpp
  tests/core/test_wire.cpp
)
target_link_libraries(ds4n_tests PRIVATE ds4n_core doctest::doctest)
target_compile_options(ds4n_tests PRIVATE -Wall -Wextra)
# Tests that read checked-in reference files (ref/, tests/data/) find them from here.
target_compile_definitions(ds4n_tests PRIVATE DS4N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME core COMMAND ds4n_tests)

# I/O-boundary tests (temp files): the GLM-5.3 Backbone part writer and verifier.
add_executable(ds4n_glm53_io_tests tests/main.cpp tests/io/test_glm53_backbone.cpp tests/io/test_glm53_store.cpp
               tests/io/test_glm53_weights.cpp)
target_link_libraries(ds4n_glm53_io_tests PRIVATE ds4n_glm53_io doctest::doctest)
target_compile_options(ds4n_glm53_io_tests PRIVATE -Wall -Wextra)
add_test(NAME glm53_io COMMAND ds4n_glm53_io_tests)

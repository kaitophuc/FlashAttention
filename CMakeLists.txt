cmake_minimum_required(VERSION 3.24)
project(flash_attention_scratch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Reasonable defaults for Ampere/Hopper class GPUs; override via -DCMAKE_CUDA_ARCHITECTURES=...
  set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)
find_package(benchmark QUIET)
find_package(Git QUIET)

option(FA_AUTO_FETCH_BENCHMARK "Automatically fetch Google Benchmark when missing" ON)

if(NOT benchmark_FOUND AND FA_AUTO_FETCH_BENCHMARK)
  if(GIT_FOUND)
    message(STATUS "Google Benchmark not found. Attempting auto-fetch from source...")
    set(_fa_bench_src_dir "${CMAKE_BINARY_DIR}/_deps/googlebenchmark-src")
    if(NOT EXISTS "${_fa_bench_src_dir}/CMakeLists.txt")
      file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/_deps")
      execute_process(
        COMMAND "${GIT_EXECUTABLE}" clone --depth 1 --branch v1.8.4 https://github.com/google/benchmark.git googlebenchmark-src
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/_deps"
        RESULT_VARIABLE _fa_bench_clone_result
        OUTPUT_QUIET
        ERROR_VARIABLE _fa_bench_clone_error
      )
    else()
      set(_fa_bench_clone_result 0)
    endif()

    if(_fa_bench_clone_result EQUAL 0 AND EXISTS "${_fa_bench_src_dir}/CMakeLists.txt")
      set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
      set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
      add_subdirectory(
        "${_fa_bench_src_dir}"
        "${CMAKE_BINARY_DIR}/_deps/googlebenchmark-build"
        EXCLUDE_FROM_ALL
      )
      message(STATUS "Google Benchmark fetched and enabled.")
    else()
      message(WARNING "Google Benchmark auto-fetch failed: ${_fa_bench_clone_error}")
    endif()
  else()
    message(WARNING "Google Benchmark not found and git is unavailable; cannot auto-fetch.")
  endif()
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/flash_attn_api.cu")
  add_library(flashattn_ops SHARED
    src/ops/flash_attn_api.cu
  )

  target_include_directories(flashattn_ops
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/src
  )

  target_compile_options(flashattn_ops
    PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  )

  set_target_properties(flashattn_ops PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME flashattn_ops
  )
  target_link_libraries(flashattn_ops PUBLIC CUDA::cudart)
endif()

if(TARGET flashattn_ops AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke_main.cpp")
  add_executable(flashattn_smoke tests/smoke_main.cpp)
  target_link_libraries(flashattn_smoke PRIVATE flashattn_ops)
endif()

add_executable(tensor_tests
  tests/test_tensor.cu
  src/tensor/src/cu_stream.cu
)
target_include_directories(tensor_tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/include
)
target_link_libraries(tensor_tests PRIVATE CUDA::cudart GTest::gtest_main)

if(benchmark_FOUND OR TARGET benchmark::benchmark OR TARGET benchmark)
  add_executable(tensor_bandwidth_bench
    bench/tensor_bandwidth_bench.cu
    src/tensor/src/cu_stream.cu
  )
  target_include_directories(tensor_bandwidth_bench
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/include
  )
  if(TARGET benchmark::benchmark)
    target_link_libraries(tensor_bandwidth_bench PRIVATE CUDA::cudart benchmark::benchmark)
  else()
    target_link_libraries(tensor_bandwidth_bench PRIVATE CUDA::cudart benchmark)
  endif()
else()
  message(WARNING "Google Benchmark unavailable: tensor_bandwidth_bench target is disabled.")
endif()

enable_testing()
if(TARGET flashattn_smoke)
  add_test(NAME smoke COMMAND flashattn_smoke)
endif()
include(GoogleTest)
gtest_discover_tests(tensor_tests)

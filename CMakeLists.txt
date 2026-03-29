cmake_minimum_required(VERSION 3.24)
project(flash_attention_scratch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Reasonable defaults for Ampere/Hopper/Blackwell class GPUs; override via -DCMAKE_CUDA_ARCHITECTURES=...
  set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90 120)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Git QUIET)
option(FA_BUILD_TESTS "Build C++ tests" ON)
option(FA_BUILD_BENCHMARKS "Build benchmarks" ON)
option(FA_BUILD_PYTHON "Build Python extension module" OFF)
option(FA_BUILD_PYTHON_TEST_TARGET "Add CMake target for Python API tests" ON)

if(FA_BUILD_TESTS)
  find_package(GTest QUIET)
endif()

if(FA_BUILD_BENCHMARKS)
  find_package(benchmark QUIET)
endif()

if(FA_BUILD_TESTS)
  if(GTest_FOUND)
    message(STATUS "GoogleTest found on system. Skipping download.")
  else()
    if(NOT GIT_FOUND)
      message(FATAL_ERROR "GoogleTest missing and git is unavailable; cannot download GoogleTest.")
    endif()

    message(STATUS "GoogleTest not found. Downloading from source...")
    set(_fa_gtest_src_dir "${CMAKE_BINARY_DIR}/_deps/googletest-src")
    if(NOT EXISTS "${_fa_gtest_src_dir}/CMakeLists.txt")
      file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/_deps")
      execute_process(
        COMMAND "${GIT_EXECUTABLE}" clone --depth 1 --branch v1.14.0 https://github.com/google/googletest.git googletest-src
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/_deps"
        RESULT_VARIABLE _fa_gtest_clone_result
        OUTPUT_QUIET
        ERROR_VARIABLE _fa_gtest_clone_error
      )
    else()
      set(_fa_gtest_clone_result 0)
    endif()

    if(NOT _fa_gtest_clone_result EQUAL 0 OR NOT EXISTS "${_fa_gtest_src_dir}/CMakeLists.txt")
      message(FATAL_ERROR "Failed to download GoogleTest: ${_fa_gtest_clone_error}")
    endif()

    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    add_subdirectory(
      "${_fa_gtest_src_dir}"
      "${CMAKE_BINARY_DIR}/_deps/googletest-build"
      EXCLUDE_FROM_ALL
    )

    if(NOT TARGET GTest::gtest_main)
      message(FATAL_ERROR "GoogleTest downloaded but target GTest::gtest_main is unavailable.")
    endif()
    message(STATUS "GoogleTest downloaded and enabled.")
  endif()
endif()

add_library(flashattention_common INTERFACE)
target_include_directories(flashattention_common
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

option(FA_AUTO_FETCH_BENCHMARK "Automatically fetch Google Benchmark when missing" ON)

if(FA_BUILD_BENCHMARKS AND NOT benchmark_FOUND AND FA_AUTO_FETCH_BENCHMARK)
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
  target_link_libraries(flashattn_ops PUBLIC flashattention_common CUDA::cudart)
endif()

if(TARGET flashattn_ops AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke_main.cpp")
  add_executable(flashattn_smoke tests/smoke_main.cpp)
  target_link_libraries(flashattn_smoke PRIVATE flashattn_ops)
endif()

# Auto-discover core sources once and build a reusable library.
file(GLOB_RECURSE FA_ALL_SRC_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

set(FA_CORE_SOURCES)
set(FA_SRC_MAIN_SOURCES)
foreach(_src IN LISTS FA_ALL_SRC_SOURCES)
  if(_src MATCHES ".*/ops/flash_attn_api\\.cu$")
    continue()
  endif()

  file(READ "${_src}" _src_head LIMIT 16384)
  if(_src_head MATCHES "int[ \t\r\n]+main[ \t\r\n]*\\(")
    list(APPEND FA_SRC_MAIN_SOURCES "${_src}")
  else()
    list(APPEND FA_CORE_SOURCES "${_src}")
  endif()
endforeach()

if(FA_CORE_SOURCES)
  add_library(flashattention_core STATIC ${FA_CORE_SOURCES})
  set_target_properties(flashattention_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(flashattention_core PUBLIC flashattention_common CUDA::cudart CUDA::cublas CUDA::cublasLt)
endif()

set(FA_TEST_SOURCES)
if(FA_BUILD_TESTS)
  # Auto-discover tests and create one executable per test file.
  file(GLOB_RECURSE FA_TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp"
  )
endif()

if(FA_BUILD_TESTS)
  foreach(_test_src IN LISTS FA_TEST_SOURCES)
    file(RELATIVE_PATH _test_rel "${CMAKE_CURRENT_SOURCE_DIR}/tests" "${_test_src}")
    string(REPLACE "/" "_" _test_target "${_test_rel}")
    string(REPLACE ".cu" "" _test_target "${_test_target}")
    string(REPLACE ".cpp" "" _test_target "${_test_target}")
    set(_test_target "fa_test_${_test_target}")

    add_executable(${_test_target} "${_test_src}")
    target_include_directories(${_test_target}
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/include
    )
    if(TARGET flashattention_core)
      target_link_libraries(${_test_target} PRIVATE flashattention_core)
    endif()
    target_link_libraries(${_test_target} PRIVATE GTest::gtest_main)
  endforeach()
endif()

# Auto-discover src files with main() and create one executable per file.
foreach(_src_main IN LISTS FA_SRC_MAIN_SOURCES)
  file(RELATIVE_PATH _src_rel "${CMAKE_CURRENT_SOURCE_DIR}/src" "${_src_main}")
  string(REPLACE "/" "_" _src_target "${_src_rel}")
  string(REPLACE ".cu" "" _src_target "${_src_target}")
  string(REPLACE ".cpp" "" _src_target "${_src_target}")
  set(_src_target "src_${_src_target}")

  add_executable(${_src_target} "${_src_main}")
  if(TARGET flashattention_core)
    target_link_libraries(${_src_target} PRIVATE flashattention_core)
  endif()
  target_link_libraries(${_src_target} PRIVATE flashattention_common CUDA::cudart CUDA::cublas CUDA::cublasLt)
endforeach()

if(FA_BUILD_BENCHMARKS AND (benchmark_FOUND OR TARGET benchmark::benchmark OR TARGET benchmark))
  add_executable(tensor_bandwidth_bench
    bench/tensor_bandwidth_bench.cu
    src/cu_stream.cu
  )
  target_include_directories(tensor_bandwidth_bench
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  if(TARGET benchmark::benchmark)
    target_link_libraries(tensor_bandwidth_bench PRIVATE flashattention_common CUDA::cudart benchmark::benchmark)
  else()
    target_link_libraries(tensor_bandwidth_bench PRIVATE flashattention_common CUDA::cudart benchmark)
  endif()

  add_executable(linear_forward_bench
    bench/linear_forward_bench.cu
    src/cu_stream.cu
    src/ops_linear.cu
  )
  target_include_directories(linear_forward_bench
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  if(TARGET benchmark::benchmark)
    target_link_libraries(linear_forward_bench PRIVATE flashattention_common CUDA::cudart CUDA::cublas CUDA::cublasLt benchmark::benchmark)
  else()
    target_link_libraries(linear_forward_bench PRIVATE flashattention_common CUDA::cudart CUDA::cublas CUDA::cublasLt benchmark)
  endif()

  add_executable(linear_backward_bench
    bench/linear_backward_bench.cu
    src/cu_stream.cu
    src/ops_linear.cu
  )
  target_include_directories(linear_backward_bench
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  if(TARGET benchmark::benchmark)
    target_link_libraries(linear_backward_bench PRIVATE flashattention_common CUDA::cudart CUDA::cublas CUDA::cublasLt benchmark::benchmark)
  else()
    target_link_libraries(linear_backward_bench PRIVATE flashattention_common CUDA::cudart CUDA::cublas CUDA::cublasLt benchmark)
  endif()

  add_executable(linear_forward_backward_bench
    bench/linear_forward_backward_bench.cu
    src/cu_stream.cu
    src/ops_linear.cu
  )
  target_include_directories(linear_forward_backward_bench
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  if(TARGET benchmark::benchmark)
    target_link_libraries(linear_forward_backward_bench PRIVATE flashattention_common CUDA::cudart CUDA::cublas CUDA::cublasLt benchmark::benchmark)
  else()
    target_link_libraries(linear_forward_backward_bench PRIVATE flashattention_common CUDA::cudart CUDA::cublas CUDA::cublasLt benchmark)
  endif()
elseif(FA_BUILD_BENCHMARKS)
  message(WARNING "Google Benchmark unavailable: benchmark targets are disabled.")
endif()

if(FA_BUILD_PYTHON_TEST_TARGET)
  add_custom_target(fa_python_tests
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_python_tests.sh"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Run ktorch Python API tests"
    USES_TERMINAL
  )
endif()

if(FA_BUILD_TESTS)
  enable_testing()
  if(TARGET flashattn_smoke)
    add_test(NAME smoke COMMAND flashattn_smoke)
  endif()
  include(GoogleTest)
  foreach(_test_src IN LISTS FA_TEST_SOURCES)
    file(RELATIVE_PATH _test_rel "${CMAKE_CURRENT_SOURCE_DIR}/tests" "${_test_src}")
    string(REPLACE "/" "_" _test_target "${_test_rel}")
    string(REPLACE ".cu" "" _test_target "${_test_target}")
    string(REPLACE ".cpp" "" _test_target "${_test_target}")
    set(_test_target "fa_test_${_test_target}")
    if(TARGET "${_test_target}")
      set(_fa_discovered_tests)
      gtest_discover_tests(${_test_target}
        NO_PRETTY_VALUES
        NO_PRETTY_TYPES
        TEST_LIST _fa_discovered_tests
      )
      foreach(_one_test IN LISTS _fa_discovered_tests)
        set(_labels "all")
        if(_one_test MATCHES "^LinearForward\\.")
          list(APPEND _labels "linear")
        endif()
        if(_one_test MATCHES "Rejects|SweepAllCases|Invariant|Numeric|SingleStream")
          list(APPEND _labels "smoke")
        endif()
        if(_one_test MATCHES "Stress")
          list(APPEND _labels "stress")
        endif()
        set_tests_properties(${_one_test} PROPERTIES LABELS "${_labels}")
      endforeach()
    endif()
  endforeach()
endif()

if(FA_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)

  set(FA_PYTHON_BINDING_SOURCES
    bindings/python/module.cpp
    bindings/python/py_dtype_device.cpp
    bindings/python/py_tensor.cpp
    bindings/python/py_ops_linear.cpp
    bindings/python/py_ops_layernorm.cpp
    bindings/python/py_ops_relu.cpp
    bindings/python/py_ops_softmax.cpp
    bindings/python/py_ops_classification.cpp
    bindings/python/py_ops_optimizer.cpp
  )

  pybind11_add_module(ktorch_cpp MODULE ${FA_PYTHON_BINDING_SOURCES})
  target_include_directories(ktorch_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings/python
  )
  target_link_libraries(ktorch_cpp PRIVATE flashattention_core CUDA::cudart CUDA::cublas CUDA::cublasLt)
  set_target_properties(ktorch_cpp PROPERTIES
    OUTPUT_NAME _C
  )

  install(TARGETS ktorch_cpp LIBRARY DESTINATION ktorch)
endif()

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

add_executable(flashattn_smoke tests/smoke_main.cpp)
target_link_libraries(flashattn_smoke PRIVATE flashattn_ops)

enable_testing()
add_test(NAME smoke COMMAND flashattn_smoke)

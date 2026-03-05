#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr) do {                               \
    cudaError_t _e = (expr);                                \
    if (_e != cudaSuccess) {                                \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",      \
            __FILE__, __LINE__, cudaGetErrorString(_e));    \
        std::abort();                                       \
    }                                                       \
} while(0)

#define CUDA_KERNEL_CHECK() do {                            \
    CUDA_CHECK(cudaGetLastError());                         \
    CUDA_CHECK(cudaDeviceSynchronize());                    \
} while(0)

enum class Device {CPU, CUDA};
enum class DType {F16, BF16, F32, I32, U8};
constexpr size_t dtype_size(DType dtype) {
    // Return the size in bytes for each dtype.
    switch (dtype) {
        case DType::F16: return 2;
        case DType::BF16: return 2;
        case DType::F32: return 4;
        case DType::I32: return 4;
        case DType::U8: return 1;
        default: return 0; // Should not happen.
    }
}
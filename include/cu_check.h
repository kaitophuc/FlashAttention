#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

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

inline const char* cublas_status_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

#define CUBLAS_CHECK(expr) do {                                                      \
    cublasStatus_t _s = (expr);                                                      \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                               \
        std::fprintf(stderr, "cuBLAS error %s:%d: %s\n",                             \
            __FILE__, __LINE__, cublas_status_string(_s));                           \
        std::abort();                                                                 \
    }                                                                                 \
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

template <typename>
struct always_false : std::false_type {};

template <typename T>
constexpr DType get_dtype() {
    if constexpr (std::is_same_v<T, float>) {
        return DType::F32;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return DType::U8;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return DType::I32;
    } else if constexpr (std::is_same_v<T, __half>) {
        return DType::F16;
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return DType::BF16;
    } else {
        static_assert(always_false<T>::value, "Unsupported type for get_dtype");
    }
}

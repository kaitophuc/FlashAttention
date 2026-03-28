#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include "cu_check.h"
#include "cu_stream.h"

inline bool can_use_async_pool_cached() {
    static const bool ok = [] {
        // Check if the current CUDA runtime supports async memory operations.
        // And memory pool API is available (CUDA 11.2+).
        #if defined(CUDART_VERSION)
            int dev;
            if (cudaGetDevice(&dev) != cudaSuccess) {
                return false;
            }

            int supported = 0;
            if (cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, dev) != cudaSuccess) {
                return false;   
            }
            if (!supported) {
                return false;
            }

            cudaMemPool_t pool = nullptr;
            if (cudaDeviceGetDefaultMemPool(&pool, dev) != cudaSuccess) {
                return false;
            }
            if (pool == nullptr) {
                return false;
            }

            auto threshold = __UINT64_MAX__;
            if (cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold) != cudaSuccess) {
                return false;
            }
            return true;
        #else
            return false;
        #endif
    }();
    return ok;
}

// A simple allocator that manages a single contiguous block of memory on the GPU.
inline auto allocate_device(size_t bytes) -> void* {
    void* ptr;
    if (can_use_async_pool_cached()) {
        auto current_stream = cudaStream_t(0);//get_current_stream();
        CUDA_CHECK(cudaMallocAsync(&ptr, bytes, current_stream));
    } else {
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
    }
    return ptr;
}

// Explicitly specify stream for allocation.
inline auto allocate_device(size_t bytes, Stream& stream) -> void* {
    if (stream.s != cudaStream_t(0)) {
        throw std::invalid_argument("allocator.h: Stream argument should be the default stream at this phase.");
    }
    void* ptr;
    if (can_use_async_pool_cached()) {
        CUDA_CHECK(cudaMallocAsync(&ptr, bytes, stream.s));
    } else {
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
    }
    return ptr;
}

// A simple deallocator for GPU memory.
inline void deallocate_device(void* ptr) {
    if (can_use_async_pool_cached()) {
        auto current_stream = cudaStream_t(0);//get_current_stream();
        CUDA_CHECK(cudaFreeAsync(ptr, current_stream));
    } else {
        CUDA_CHECK(cudaFree(ptr));
    }
}

// Explicitly specify stream for deallocation.
inline void deallocate_device(void* ptr, Stream& stream) {
    if (stream.s != cudaStream_t(0)) {
        throw std::invalid_argument("allocator.h: Stream argument should be the default stream at this phase.");
    }
    if (can_use_async_pool_cached()) {
        CUDA_CHECK(cudaFreeAsync(ptr, stream.s));
    } else {
        CUDA_CHECK(cudaFree(ptr));
    }
}

// A simple allocator that manages pinned host memory.
inline auto allocate_host(size_t bytes) -> void* {
    void* ptr;
    CUDA_CHECK(cudaMallocHost(&ptr, bytes));
    return ptr;
}

inline void deallocate_host(void* ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
}

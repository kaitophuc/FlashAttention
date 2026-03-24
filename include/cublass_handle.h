#pragma once

#include <cublas_v2.h>

#include <mutex>
#include <utility>

#include "cu_stream.h"

struct CublasHandle {
    cublasHandle_t handle{};
    mutable std::mutex mu_;

    CublasHandle() {
        CUBLAS_CHECK(cublasCreate(&handle));
    }

    ~CublasHandle() {
        if (handle != nullptr) {
            CUBLAS_CHECK(cublasDestroy(handle));
        }
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    void bind_current_stream(int device = -1) {
        cudaStream_t stream = get_current_stream(device);
        bind_stream(stream);
    }

    void bind_stream(const Stream& stream) {
        bind_stream(stream.s);
    }

    void bind_stream(cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mu_);
        CUBLAS_CHECK(cublasSetStream(handle, stream));
    }

    template <typename Fn>
    decltype(auto) with_bound_stream(cudaStream_t stream, Fn&& fn) {
        std::lock_guard<std::mutex> lock(mu_);
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        return std::forward<Fn>(fn)(handle);
    }

    cublasHandle_t get() const {
        return handle;
    }
};

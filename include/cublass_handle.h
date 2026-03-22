#pragma once

#include <cublas_v2.h>
#include "cu_stream.h"

struct CublasHandle {
    cublasHandle_t handle{};

    CublasHandle() {
        cublasCreate(&handle);
    }

    ~CublasHandle() {
        if (handle != nullptr) {
            cublasDestroy(handle);
        }
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    void bind_current_stream(int device = -1) {
        cudaStream_t stream = get_current_stream(device);
        cublasSetStream(handle, stream);
    }

    void bind_stream(const Stream& stream) {
        bind_stream(stream.s);
    }

    void bind_stream(cudaStream_t stream) {
        cublasSetStream(handle, stream);
    }

    cublasHandle_t get() const {
        return handle;
    }
};
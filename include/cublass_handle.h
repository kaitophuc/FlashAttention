#pragma once

#include <cublas_v2.h>

#include <mutex>
#include <utility>

#include "cu_stream.h"

struct CublasHandle {
    cublasHandle_t handle{};

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

    cublasHandle_t get() const {
        return handle;
    }
};

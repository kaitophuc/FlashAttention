#pragma once

#include <cublas_v2.h>
#include <cublasLt.h>

#include <mutex>
#include <utility>

#include "cu_stream.h"

struct CublasHandle {
    cublasHandle_t handle{};
    cublasLtHandle_t lt_handle{};

    CublasHandle() {
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasLtCreate(&lt_handle));
    }

    ~CublasHandle() {
        if (handle != nullptr) {
            CUBLAS_CHECK(cublasDestroy(handle));
        }
        if (lt_handle != nullptr) {
            CUBLAS_CHECK(cublasLtDestroy(lt_handle));
        }
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t get() const {
        return handle;
    }

    cublasLtHandle_t get_lt() const {
        return lt_handle;
    }
};

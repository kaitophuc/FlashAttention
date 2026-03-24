#pragma once

#include <cuda_runtime_api.h>

namespace fa_test {

inline bool cuda_available() {
    int device_count = 0;
    const cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

}  // namespace fa_test

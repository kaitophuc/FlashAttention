#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// dtype codes for API boundary.
// 0 -> fp16 (half), 1 -> bf16 (nv_bfloat16)
enum fa_dtype {
  FA_DTYPE_FP16 = 0,
  FA_DTYPE_BF16 = 1,
};

// Naive scaled dot-product attention forward pass.
// Inputs/outputs are expected as contiguous [B, H, N, D] on CUDA device.
// Constraints for milestone 0:
// - N <= 2048
// - D in {64, 128}
// Returns cudaSuccess on success, or an error code.
int fa_sdpa_forward(
    const void* q,
    const void* k,
    const void* v,
    void* o,
    int B,
    int H,
    int N,
    int D,
    int dtype,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

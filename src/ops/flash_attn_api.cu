#include "ops/flash_attn_api.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cmath>

template <typename T>
__device__ __forceinline__ float to_float(T x);

template <>
__device__ __forceinline__ float to_float<half>(half x) {
  return __half2float(x);
}

template <>
__device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 x) {
  return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float x);

template <>
__device__ __forceinline__ half from_float<half>(float x) {
  return __float2half_rn(x);
}

template <>
__device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}

template <typename T>
__global__ void sdpa_forward_naive_kernel(
    const T* q,
    const T* k,
    const T* v,
    T* o,
    int B,
    int H,
    int N,
    int D,
    float scale) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = blockIdx.y;
  const int bh = blockIdx.z;

  if (d >= D) {
    return;
  }

  const int b = bh / H;
  const int h = bh % H;
  if (b >= B) {
    return;
  }

  const size_t q_base = (((size_t)b * H + h) * N + n) * D;

  float max_score = -CUDART_INF_F;
  for (int j = 0; j < N; ++j) {
    const size_t k_base = (((size_t)b * H + h) * N + j) * D;
    float dot = 0.0f;
    for (int t = 0; t < D; ++t) {
      dot += to_float(q[q_base + t]) * to_float(k[k_base + t]);
    }
    const float score = dot * scale;
    if (score > max_score) {
      max_score = score;
    }
  }

  float denom = 0.0f;
  float out_val = 0.0f;
  for (int j = 0; j < N; ++j) {
    const size_t k_base = (((size_t)b * H + h) * N + j) * D;
    const size_t v_base = (((size_t)b * H + h) * N + j) * D;

    float dot = 0.0f;
    for (int t = 0; t < D; ++t) {
      dot += to_float(q[q_base + t]) * to_float(k[k_base + t]);
    }
    const float w = __expf(dot * scale - max_score);
    denom += w;
    out_val += w * to_float(v[v_base + d]);
  }

  out_val = out_val / denom;
  o[q_base + d] = from_float<T>(out_val);
}

template <typename T>
static int launch_sdpa_forward(
    const void* q,
    const void* k,
    const void* v,
    void* o,
    int B,
    int H,
    int N,
    int D,
    cudaStream_t stream) {
  const int threads = 128;
  const dim3 block(threads);
  const dim3 grid((D + threads - 1) / threads, N, B * H);
  const float scale = 1.0f / sqrtf(static_cast<float>(D));

  sdpa_forward_naive_kernel<T><<<grid, block, 0, stream>>>(
      static_cast<const T*>(q),
      static_cast<const T*>(k),
      static_cast<const T*>(v),
      static_cast<T*>(o),
      B,
      H,
      N,
      D,
      scale);

  return static_cast<int>(cudaGetLastError());
}

extern "C" int fa_sdpa_forward(
    const void* q,
    const void* k,
    const void* v,
    void* o,
    int B,
    int H,
    int N,
    int D,
    int dtype,
    cudaStream_t stream) {
  if (!q || !k || !v || !o) {
    return static_cast<int>(cudaErrorInvalidDevicePointer);
  }
  if (B <= 0 || H <= 0 || N <= 0 || D <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (N > 2048) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (!(D == 64 || D == 128)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  if (dtype == FA_DTYPE_FP16) {
    return launch_sdpa_forward<half>(q, k, v, o, B, H, N, D, stream);
  }
  if (dtype == FA_DTYPE_BF16) {
    return launch_sdpa_forward<nv_bfloat16>(q, k, v, o, B, H, N, D, stream);
  }

  return static_cast<int>(cudaErrorInvalidValue);
}

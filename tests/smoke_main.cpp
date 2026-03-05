#include "ops/flash_attn_api.h"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

int main() {
  constexpr int B = 1;
  constexpr int H = 1;
  constexpr int N = 4;
  constexpr int D = 64;

  const size_t elements = static_cast<size_t>(B) * H * N * D;
  const size_t bytes = elements * sizeof(unsigned short);  // fp16 payload size

  void* q = nullptr;
  void* k = nullptr;
  void* v = nullptr;
  void* o = nullptr;

  cudaError_t err = cudaMalloc(&q, bytes);
  if (err != cudaSuccess) {
    std::cerr << "cudaMalloc(q) failed: " << cudaGetErrorString(err) << "\n";
    return 1;
  }
  err = cudaMalloc(&k, bytes);
  if (err != cudaSuccess) {
    std::cerr << "cudaMalloc(k) failed: " << cudaGetErrorString(err) << "\n";
    return 1;
  }
  err = cudaMalloc(&v, bytes);
  if (err != cudaSuccess) {
    std::cerr << "cudaMalloc(v) failed: " << cudaGetErrorString(err) << "\n";
    return 1;
  }
  err = cudaMalloc(&o, bytes);
  if (err != cudaSuccess) {
    std::cerr << "cudaMalloc(o) failed: " << cudaGetErrorString(err) << "\n";
    return 1;
  }

  err = cudaMemset(q, 0, bytes);
  if (err != cudaSuccess) return 1;
  err = cudaMemset(k, 0, bytes);
  if (err != cudaSuccess) return 1;
  err = cudaMemset(v, 0, bytes);
  if (err != cudaSuccess) return 1;

  int api_err = fa_sdpa_forward(q, k, v, o, B, H, N, D, FA_DTYPE_FP16, nullptr);
  if (api_err != static_cast<int>(cudaSuccess)) {
    std::cerr << "fa_sdpa_forward failed with code " << api_err << "\n";
    return 1;
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(err) << "\n";
    return 1;
  }

  cudaFree(q);
  cudaFree(k);
  cudaFree(v);
  cudaFree(o);

  std::cout << "smoke test passed\n";
  return 0;
}

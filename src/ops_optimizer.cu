#include "ops.h"

#include <cmath>
#include <cstddef>

namespace {

template <int BLOCK_SIZE>
__global__ void sgd_update_kernel(float* __restrict__ param,
                                  const float* __restrict__ grad,
                                  size_t numel,
                                  float lr) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * BLOCK_SIZE + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * BLOCK_SIZE;
    for (size_t i = idx; i < numel; i += stride) {
        param[i] -= lr * grad[i];
    }
}

}  // namespace

void sgd_update_(Tensor& param, const Tensor& grad, float lr, Stream* stream) {
    if (stream == nullptr) {
        throw std::invalid_argument("ops_optimizer.cu: sgd_update_: Stream pointer cannot be null.");
    }
    if (stream->s != cudaStream_t(0)) {
        throw std::invalid_argument("ops_optimizer.cu: sgd_update_: Only the default stream is supported at this phase.");
    }
    if (!std::isfinite(lr)) {
        throw std::invalid_argument("ops_optimizer.cu: sgd_update_: lr must be finite.");
    }
    if (param.shape_ != grad.shape_) {
        throw std::invalid_argument("ops_optimizer.cu: sgd_update_: param and grad must have the same shape.");
    }
    if (param.dtype_ != DType::F32 || grad.dtype_ != DType::F32) {
        throw std::invalid_argument("ops_optimizer.cu: sgd_update_: Only float32 tensors are supported.");
    }
    if (param.device_ != Device::CUDA || grad.device_ != Device::CUDA) {
        throw std::invalid_argument("ops_optimizer.cu: sgd_update_: Only CUDA tensors are supported.");
    }

    const size_t numel = param.numel();
    if (numel == 0) {
        return;
    }

    constexpr int kBlockSize = 256;
    const int grid = static_cast<int>((numel + kBlockSize - 1) / kBlockSize);
    sgd_update_kernel<kBlockSize><<<grid, kBlockSize, 0, stream->s>>>(
        static_cast<float*>(param.data_),
        static_cast<const float*>(grad.data_),
        numel,
        lr);
}

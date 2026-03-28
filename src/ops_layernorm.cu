#include "ops.h"

#include <cstdint>

namespace {

template <int BLOCK_SIZE>
__global__ void layernorm_forward_kernel(
    const float* __restrict__ X,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ Y,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    int m,
    int n,
    float eps) {
    const int row = blockIdx.x;
    if (row >= m) {
        return;
    }

    const int row_offset = row * n;

    float thread_sum = 0.0f;
    for (int col = threadIdx.x; col < n; col += BLOCK_SIZE) {
        thread_sum += X[row_offset + col];
    }

    __shared__ float smem_sum[BLOCK_SIZE];
    smem_sum[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem_sum[threadIdx.x] += smem_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float mu = smem_sum[0] / static_cast<float>(n);

    float thread_var_sum = 0.0f;
    for (int col = threadIdx.x; col < n; col += BLOCK_SIZE) {
        const float centered = X[row_offset + col] - mu;
        thread_var_sum += centered * centered;
    }

    __shared__ float smem_var[BLOCK_SIZE];
    smem_var[threadIdx.x] = thread_var_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem_var[threadIdx.x] += smem_var[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float inv_std = rsqrtf(smem_var[0] / static_cast<float>(n) + eps);

    if (threadIdx.x == 0) {
        mean[row] = mu;
        rstd[row] = inv_std;
    }

    for (int col = threadIdx.x; col < n; col += BLOCK_SIZE) {
        const float x_hat = (X[row_offset + col] - mu) * inv_std;
        Y[row_offset + col] = x_hat * gamma[col] + beta[col];
    }
}

template <int BLOCK_SIZE>
__global__ void layernorm_backward_dx_kernel(
    const float* __restrict__ dY,
    const float* __restrict__ X,
    const float* __restrict__ gamma,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ dX,
    int m,
    int n) {
    const int row = blockIdx.x;
    if (row >= m) {
        return;
    }

    const int row_offset = row * n;
    const float mu = mean[row];
    const float inv_std = rstd[row];

    float thread_sum_dyg = 0.0f;
    float thread_sum_dyg_xhat = 0.0f;
    for (int col = threadIdx.x; col < n; col += BLOCK_SIZE) {
        const float x_hat = (X[row_offset + col] - mu) * inv_std;
        const float dyg = dY[row_offset + col] * gamma[col];
        thread_sum_dyg += dyg;
        thread_sum_dyg_xhat += dyg * x_hat;
    }

    __shared__ float smem_dyg[BLOCK_SIZE];
    __shared__ float smem_dyg_xhat[BLOCK_SIZE];
    smem_dyg[threadIdx.x] = thread_sum_dyg;
    smem_dyg_xhat[threadIdx.x] = thread_sum_dyg_xhat;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem_dyg[threadIdx.x] += smem_dyg[threadIdx.x + stride];
            smem_dyg_xhat[threadIdx.x] += smem_dyg_xhat[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float sum_dyg = smem_dyg[0];
    const float sum_dyg_xhat = smem_dyg_xhat[0];
    const float inv_n = 1.0f / static_cast<float>(n);

    for (int col = threadIdx.x; col < n; col += BLOCK_SIZE) {
        const float x_hat = (X[row_offset + col] - mu) * inv_std;
        const float dyg = dY[row_offset + col] * gamma[col];
        const float inner = static_cast<float>(n) * dyg - sum_dyg - x_hat * sum_dyg_xhat;
        dX[row_offset + col] = inv_std * inv_n * inner;
    }
}

template <int BLOCK_SIZE>
__global__ void layernorm_backward_param_grads_kernel(
    const float* __restrict__ dY,
    const float* __restrict__ X,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int m,
    int n,
    bool needs_dgamma,
    bool needs_dbeta) {
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int total = m * n;
    if (idx >= total) {
        return;
    }

    const int row = idx / n;
    const int col = idx % n;

    const float dy = dY[idx];
    const float x_hat = (X[idx] - mean[row]) * rstd[row];

    if (needs_dgamma) {
        atomicAdd(&dgamma[col], dy * x_hat);
    }
    if (needs_dbeta) {
        atomicAdd(&dbeta[col], dy);
    }
}

}  // namespace

LayerNormResults layernorm_forward(const Tensor& X,
                                   const Tensor& gamma,
                                   const Tensor& beta,
                                   float eps,
                                   Stream* stream) {
    if (stream == nullptr) {
        throw std::invalid_argument("ops_layernorm.cu: Layernorm_forward: Stream pointer cannot be null.");
    }
    if (stream->s != cudaStream_t(0)) {
        throw std::invalid_argument("ops_layernorm.cu: Layernorm_forward: Only the default stream is supported at this phase.");
    }
    if (eps <= 0.0f) {
        throw std::invalid_argument("ops_layernorm.cu: Layernorm_forward: eps must be greater than 0.");
    }
    if (X.dtype_ != DType::F32 || gamma.dtype_ != DType::F32 || beta.dtype_ != DType::F32) {
        throw std::invalid_argument("ops_layernorm.cu: Layernorm_forward: Only float32 tensors are supported.");
    }
    if (X.device_ != Device::CUDA || gamma.device_ != Device::CUDA || beta.device_ != Device::CUDA) {
        throw std::invalid_argument("ops_layernorm.cu: Layernorm_forward: Only CUDA tensors are supported.");
    }
    if (X.shape_.size() != 2) {
        throw std::invalid_argument("ops_layernorm.cu: Layernorm_forward: X must be a 2D tensor [m, n].");
    }
    if (gamma.shape_.size() != 1 || beta.shape_.size() != 1) {
        throw std::invalid_argument("ops_layernorm.cu: Layernorm_forward: gamma and beta must be 1D tensors [n].");
    }
    if (gamma.shape_[0] != X.shape_[1] || beta.shape_[0] != X.shape_[1]) {
        throw std::invalid_argument("ops_layernorm.cu: Layernorm_forward: gamma and beta sizes must match X last dimension.");
    }

    const int64_t m64 = X.shape_[0];
    const int64_t n64 = X.shape_[1];
    if (m64 <= 0 || n64 <= 0) {
        throw std::invalid_argument("ops_layernorm.cu: Layernorm_forward: Input dimensions must be greater than zero.");
    }

    const int m = static_cast<int>(m64);
    const int n = static_cast<int>(n64);

    Tensor Y = Tensor::empty(X.shape_, X.dtype_, X.device_, *stream);
    Tensor mean = Tensor::empty({m64}, X.dtype_, X.device_, *stream);
    Tensor rstd = Tensor::empty({m64}, X.dtype_, X.device_, *stream);

    constexpr int block_size = 256;
    layernorm_forward_kernel<block_size><<<m, block_size, 0, stream->s>>>(
        static_cast<const float*>(X.data_),
        static_cast<const float*>(gamma.data_),
        static_cast<const float*>(beta.data_),
        static_cast<float*>(Y.data_),
        static_cast<float*>(mean.data_),
        static_cast<float*>(rstd.data_),
        m,
        n,
        eps);

    return LayerNormResults{std::move(Y), LayerNormCtx{&X, &gamma, std::move(mean), std::move(rstd), eps, m64, n64}};
}

LayerNormGrads layernorm_backward(const Tensor& dY,
                                  const LayerNormCtx& ctx,
                                  bool needs_dX,
                                  bool needs_dgamma,
                                  bool needs_dbeta,
                                  Stream* stream) {
    if (stream == nullptr) {
        throw std::invalid_argument("ops_layernorm.cu: Layernorm_backward: Stream pointer cannot be null.");
    }
    if (stream->s != cudaStream_t(0)) {
        throw std::invalid_argument("ops_layernorm.cu: Layernorm_backward: Only the default stream is supported at this phase.");
    }
    if (ctx.X == nullptr || ctx.gamma == nullptr) {
        throw std::invalid_argument("ops_layernorm.cu: Layernorm_backward: ctx.X and ctx.gamma cannot be null.");
    }
    if (ctx.eps <= 0.0f) {
        throw std::invalid_argument("ops_layernorm.cu: Layernorm_backward: ctx.eps must be greater than 0.");
    }

    const Tensor& X = *ctx.X;
    const Tensor& gamma = *ctx.gamma;
    const Tensor& mean = ctx.mean;
    const Tensor& rstd = ctx.rstd;

    if (dY.dtype_ != DType::F32 || X.dtype_ != DType::F32 || gamma.dtype_ != DType::F32 ||
        mean.dtype_ != DType::F32 || rstd.dtype_ != DType::F32) {
        throw std::invalid_argument("ops_layernorm.cu: Layernorm_backward: Only float32 tensors are supported.");
    }
    if (dY.device_ != Device::CUDA || X.device_ != Device::CUDA || gamma.device_ != Device::CUDA ||
        mean.device_ != Device::CUDA || rstd.device_ != Device::CUDA) {
        throw std::invalid_argument("ops_layernorm.cu: Layernorm_backward: Only CUDA tensors are supported.");
    }
    if (X.shape_.size() != 2 || dY.shape_.size() != 2) {
        throw std::invalid_argument("ops_layernorm.cu: Layernorm_backward: X and dY must be 2D tensors [m, n].");
    }
    if (gamma.shape_.size() != 1 || mean.shape_.size() != 1 || rstd.shape_.size() != 1) {
        throw std::invalid_argument("ops_layernorm.cu: Layernorm_backward: gamma, mean, and rstd must be 1D tensors.");
    }
    if (dY.shape_ != X.shape_) {
        throw std::invalid_argument("ops_layernorm.cu: Layernorm_backward: dY shape must match X shape.");
    }
    if (X.shape_[0] != ctx.m || X.shape_[1] != ctx.n) {
        throw std::invalid_argument("ops_layernorm.cu: Layernorm_backward: Context shape must match X shape.");
    }
    if (gamma.shape_[0] != ctx.n || mean.shape_[0] != ctx.m || rstd.shape_[0] != ctx.m) {
        throw std::invalid_argument("ops_layernorm.cu: Layernorm_backward: Context tensor shapes are invalid.");
    }

    const int m = static_cast<int>(ctx.m);
    const int n = static_cast<int>(ctx.n);

    std::optional<Tensor> dX;
    if (needs_dX) {
        dX = Tensor::empty(X.shape_, X.dtype_, X.device_, *stream);
    }

    std::optional<Tensor> dgamma;
    if (needs_dgamma) {
        dgamma = Tensor::zeros({ctx.n}, gamma.dtype_, gamma.device_, *stream);
    }

    std::optional<Tensor> dbeta;
    if (needs_dbeta) {
        dbeta = Tensor::zeros({ctx.n}, gamma.dtype_, gamma.device_, *stream);
    }

    if (needs_dX) {
        constexpr int block_size = 256;
        layernorm_backward_dx_kernel<block_size><<<m, block_size, 0, stream->s>>>(
            static_cast<const float*>(dY.data_),
            static_cast<const float*>(X.data_),
            static_cast<const float*>(gamma.data_),
            static_cast<const float*>(mean.data_),
            static_cast<const float*>(rstd.data_),
            static_cast<float*>(dX->data_),
            m,
            n);
    }

    if (needs_dgamma || needs_dbeta) {
        constexpr int block_size = 256;
        const int total = m * n;
        const int grid_size = (total + block_size - 1) / block_size;

        float* dgamma_ptr = needs_dgamma ? static_cast<float*>(dgamma->data_) : nullptr;
        float* dbeta_ptr = needs_dbeta ? static_cast<float*>(dbeta->data_) : nullptr;

        layernorm_backward_param_grads_kernel<block_size><<<grid_size, block_size, 0, stream->s>>>(
            static_cast<const float*>(dY.data_),
            static_cast<const float*>(X.data_),
            static_cast<const float*>(mean.data_),
            static_cast<const float*>(rstd.data_),
            dgamma_ptr,
            dbeta_ptr,
            m,
            n,
            needs_dgamma,
            needs_dbeta);
    }

    return LayerNormGrads{std::move(dX), std::move(dgamma), std::move(dbeta), needs_dX, needs_dgamma, needs_dbeta};
}

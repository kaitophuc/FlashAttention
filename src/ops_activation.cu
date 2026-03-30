#include "ops.h"

#include <cstdint>

template <int BLOCK_SIZE>
__global__ void relu_forward_scalar_kernel(
    const float* __restrict__ X,
    float* __restrict__ Y,
    int total_elements) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        const float x = X[idx];
        Y[idx] = x > 0.0f ? x : 0.0f;
    }
}

template <int BLOCK_SIZE>
__global__ void relu_forward_vec4_kernel(
    const float* __restrict__ X,
    float* __restrict__ Y,
    int total_elements,
    int total_vec4_chunks) {
    const int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx < total_vec4_chunks) {
        const int base = vec_idx * 4;
        if (base + 3 < total_elements) {
            const float4 x_reg = reinterpret_cast<const float4*>(X)[vec_idx];
            float4 y_reg;
            y_reg.x = x_reg.x > 0.0f ? x_reg.x : 0.0f;
            y_reg.y = x_reg.y > 0.0f ? x_reg.y : 0.0f;
            y_reg.z = x_reg.z > 0.0f ? x_reg.z : 0.0f;
            y_reg.w = x_reg.w > 0.0f ? x_reg.w : 0.0f;
            reinterpret_cast<float4*>(Y)[vec_idx] = y_reg;
            return;
        }

        const int remaining = total_elements - base;  // 1..3 for the tail chunk.
        if (remaining >= 1) {
            const float x = X[base];
            Y[base] = x > 0.0f ? x : 0.0f;
        }
        if (remaining >= 2) {
            const float x = X[base + 1];
            Y[base + 1] = x > 0.0f ? x : 0.0f;
        }
        if (remaining >= 3) {
            const float x = X[base + 2];
            Y[base + 2] = x > 0.0f ? x : 0.0f;
        }
    }
}

template <int BLOCK_SIZE>
__global__ void relu_backward_scalar_kernel(
    const float* __restrict__ dY,
    const float* __restrict__ X,
    float* __restrict__ dX,
    int total_elements) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        const float dy = dY[idx];
        const float x = X[idx];
        dX[idx] = x > 0.0f ? dy : 0.0f;
    }
}

template <int BLOCK_SIZE>
__global__ void relu_backward_vec4_kernel(
    const float* __restrict__ dY,
    const float* __restrict__ X,
    float* __restrict__ dX,
    int total_elements,
    int total_vec4_chunks) {
    const int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx < total_vec4_chunks) {
        const int base = vec_idx * 4;
        if (base + 3 < total_elements) {
            const float4 dy_reg = reinterpret_cast<const float4*>(dY)[vec_idx];
            const float4 x_reg = reinterpret_cast<const float4*>(X)[vec_idx];
            float4 dx_reg;
            dx_reg.x = x_reg.x > 0.0f ? dy_reg.x : 0.0f;
            dx_reg.y = x_reg.y > 0.0f ? dy_reg.y : 0.0f;
            dx_reg.z = x_reg.z > 0.0f ? dy_reg.z : 0.0f;
            dx_reg.w = x_reg.w > 0.0f ? dy_reg.w : 0.0f;
            reinterpret_cast<float4*>(dX)[vec_idx] = dx_reg;
            return;
        }

        const int remaining = total_elements - base;  // 1..3 for the tail chunk.
        if (remaining >= 1) {
            const float dy = dY[base];
            const float x = X[base];
            dX[base] = x > 0.0f ? dy : 0.0f;
        }
        if (remaining >= 2) {
            const float dy = dY[base + 1];
            const float x = X[base + 1];
            dX[base + 1] = x > 0.0f ? dy : 0.0f;
        }
        if (remaining >= 3) {
            const float dy = dY[base + 2];
            const float x = X[base + 2];
            dX[base + 2] = x > 0.0f ? dy : 0.0f;
        }
    }
}

ReluResults relu_forward(const Tensor& X, Stream& stream) {
    // Check input shapes and dtypes.
    assert_non_default_stream(stream.s, "ops_activation.cu: Relu_forward");
    if (X.dtype_ != DType::F32) {
        throw std::invalid_argument("ops_activation.cu: Relu_forward: Only float32 tensors are supported.");
    }
    if (X.device_ != Device::CUDA) {
        throw std::invalid_argument("ops_activation.cu: Relu_forward: Only CUDA tensors are supported.");
    }
    if (X.shape_.size() != 2) {
        throw std::invalid_argument("ops_activation.cu: Relu_forward: X must be a 2D tensor.");
    }
    
    // Allocate output tensor.
    Tensor Y = Tensor::empty(X.shape_, X.dtype_, X.device_, stream);

    // Launch CUDA kernel to compute ReLU.
    const int total_elements = X.numel();
    const int block_size = 256;
    const float* x_ptr = static_cast<const float*>(X.data_);
    float* y_ptr = static_cast<float*>(Y.data_);

    const bool aligned_for_vec4 =
        ((reinterpret_cast<uintptr_t>(x_ptr) | reinterpret_cast<uintptr_t>(y_ptr)) &
         (alignof(float4) - 1)) == 0;

    if (aligned_for_vec4 && total_elements >= 4 && (total_elements % 4 == 0)) {
        const int total_vec4_chunks = (total_elements + 3) / 4;
        const int vec_grid_size = (total_vec4_chunks + block_size - 1) / block_size;

        relu_forward_vec4_kernel<block_size><<<vec_grid_size, block_size, 0, stream.s>>>(
            x_ptr,
            y_ptr,
            total_elements,
            total_vec4_chunks);
    } else {
        const int grid_size = (total_elements + block_size - 1) / block_size;
        relu_forward_scalar_kernel<block_size><<<grid_size, block_size, 0, stream.s>>>(
            x_ptr, y_ptr, total_elements);
    }

    return ReluResults{std::move(Y), ReluCtx{&X}};
}

ReluGrads relu_backward(const Tensor& dY, const ReluCtx& ctx, Stream& stream) {
    // Check input shapes and dtypes.
    assert_non_default_stream(stream.s, "ops_activation.cu: Relu_backward");
    if (ctx.X == nullptr) {
        throw std::invalid_argument("ops_activation.cu: Relu_backward: ctx.X cannot be null.");
    }
    if (ctx.X->shape_.size() != 2) {
        throw std::invalid_argument("ops_activation.cu: Relu_backward: ctx.X must be a 2D tensor.");
    }
    if (dY.shape_.size() != 2) {
        throw std::invalid_argument("ops_activation.cu: Relu_backward: dY must be a 2D tensor.");
    }
    if (dY.dtype_ != DType::F32) {
        throw std::invalid_argument("ops_activation.cu: Relu_backward: Only float32 tensors are supported.");
    }
    if (ctx.X->dtype_ != DType::F32) {
        throw std::invalid_argument("ops_activation.cu: Relu_backward: ctx.X must be float32.");
    }
    if (dY.dtype_ != ctx.X->dtype_) {
        throw std::invalid_argument("ops_activation.cu: Relu_backward: dY and ctx.X dtypes must match.");
    }
    if (dY.device_ != Device::CUDA || ctx.X->device_ != Device::CUDA) {
        throw std::invalid_argument("ops_activation.cu: Relu_backward: Only CUDA tensors are supported.");
    }
    if (dY.device_ != ctx.X->device_) {
        throw std::invalid_argument("ops_activation.cu: Relu_backward: dY and ctx.X must be on the same device.");
    }
    if (dY.shape_ != ctx.X->shape_) {
        throw std::invalid_argument("ops_activation.cu: Relu_backward: dY shape must match original input X shape.");
    }

    // Allocate output tensor for dX.
    Tensor dX = Tensor::empty(ctx.X->shape_, ctx.X->dtype_, ctx.X->device_, stream);

    // Launch CUDA kernel to compute dX.
    const int total_elements = dY.numel();
    const int block_size = 256;
    const float* dy_ptr = static_cast<const float*>(dY.data_);
    const float* x_ptr = static_cast<const float*>(ctx.X->data_);
    float* dx_ptr = static_cast<float*>(dX.data_);

    const bool aligned_for_vec4 =
        ((reinterpret_cast<uintptr_t>(dy_ptr) | reinterpret_cast<uintptr_t>(x_ptr) |
          reinterpret_cast<uintptr_t>(dx_ptr)) &
         (alignof(float4) - 1)) == 0;

    if (aligned_for_vec4 && total_elements >= 4 && (total_elements % 4 == 0)) {
        const int total_vec4_chunks = (total_elements + 3) / 4;
        const int vec_grid_size = (total_vec4_chunks + block_size - 1) / block_size;

        relu_backward_vec4_kernel<block_size><<<vec_grid_size, block_size, 0, stream.s>>>(
            dy_ptr,
            x_ptr,
            dx_ptr,
            total_elements,
            total_vec4_chunks);
    } else {
        const int grid_size = (total_elements + block_size - 1) / block_size;
        relu_backward_scalar_kernel<block_size><<<grid_size, block_size, 0, stream.s>>>(
            dy_ptr, x_ptr, dx_ptr, total_elements);
    }

    return ReluGrads{std::move(dX)};
}

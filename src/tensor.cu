#include "tensor.h"

Tensor Tensor::matmul(const Tensor& other) const {
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::invalid_argument("tensor.h: matmul requires 2D tensors");
    }
    if (shape_[1] != other.shape_[0]) {
        throw std::invalid_argument("tensor.h: matmul dimension mismatch");
    }
    if (dtype_ != other.dtype_) {
        throw std::invalid_argument("tensor.h: matmul requires tensors of the same data type");
    }
    if (device_ != other.device_) {
        throw std::invalid_argument("tensor.h: matmul requires tensors on the same device");
    }
    if (device_ == Device::CUDA) {
        throw std::invalid_argument("tensor.h: matmul on CUDA device requires a stream and cuBLAS handle");
    }
    // For CPU tensors, we can implement a simple matmul (not optimized)
    Tensor result = Tensor::empty({shape_[0], other.shape_[1]}, dtype_, device_);
    switch (dtype_) {
        case DType::F32:
        case DType::F16:
        case DType::I32:
        case DType::U8:
            break;
        default:
            throw std::invalid_argument("tensor.h: unsupported data type for matmul");
    }
    // Naive matmul implementation (not optimized)
    for (int64_t i = 0; i < result.shape_[0]; ++i) {
        for (int64_t j = 0; j < result.shape_[1]; ++j) {
            double sum = 0.0;
            for (int64_t k = 0; k < shape_[1]; ++k) {
                double a = 0.0;
                double b = 0.0;
                switch (dtype_) {
                    case DType::F32:
                        a = static_cast<float*>(data_)[i * shape_[1] + k];
                        b = static_cast<float*>(other.data_)[k * other.shape_[1] + j];
                        sum += a * b;
                        break;
                    case DType::F16: {
                        const __half a_h = static_cast<const __half*>(data_)[i * shape_[1] + k];
                        const __half b_h = static_cast<const __half*>(other.data_)[k * other.shape_[1] + j];
                        a = static_cast<double>(__half2float(a_h));
                        b = static_cast<double>(__half2float(b_h));
                        sum += a * b;
                        break;
                    }
                    case DType::I32:
                        a = static_cast<int32_t*>(data_)[i * shape_[1] + k];
                        b = static_cast<int32_t*>(other.data_)[k * other.shape_[1] + j];
                        sum += a * b;
                        break;
                    case DType::U8:
                        a = static_cast<uint8_t*>(data_)[i * shape_[1] + k];
                        b = static_cast<uint8_t*>(other.data_)[k * other.shape_[1] + j];
                        sum += a * b;
                        break;
                    default:
                        throw std::invalid_argument("tensor.h: unsupported data type for matmul");
                }
            }
            switch (result.dtype_) {
                case DType::F32:
                    static_cast<float*>(result.data_)[i * result.shape_[1] + j] = sum;
                    break;
                case DType::F16:
                    static_cast<__half*>(result.data_)[i * result.shape_[1] + j] =
                        __float2half_rn(static_cast<float>(sum));
                    break;
                case DType::I32:
                    static_cast<int32_t*>(result.data_)[i * result.shape_[1] + j] = static_cast<int32_t>(sum);
                    break;
                case DType::U8:
                    static_cast<uint8_t*>(result.data_)[i * result.shape_[1] + j] = static_cast<uint8_t>(sum);
                    break;
                default:
                    throw std::invalid_argument("tensor.h: unsupported data type for matmul");
            }
        }
    }
    return result;   
}

Tensor Tensor::matmul(const Tensor& other, Stream& stream, CublasHandle& cublas_handle) const {
    // Check dimensions and data types
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::invalid_argument("tensor.h: matmul requires 2D tensors");
    }
    if (shape_[1] != other.shape_[0]) {
        throw std::invalid_argument("tensor.h: matmul dimension mismatch");
    }
    if (dtype_ != other.dtype_) {
        throw std::invalid_argument("tensor.h: matmul requires tensors of the same data type");
    }
    if (stream.s != cudaStream_t(0)) {
        throw std::invalid_argument("tensor.h: matmul requires a default stream");
    }
    if (this->device_ != other.device_) {
        throw std::invalid_argument("tensor.h: matmul requires tensors on the same device");
    }
    if (this->device_ != Device::CUDA) {
        throw std::invalid_argument("tensor.h: matmul with CublasHandle requires tensors on CUDA device");
    }
    if (dtype_ != DType::F32 && dtype_ != DType::F16) {
        throw std::invalid_argument("tensor.h: matmul with CublasHandle only supports F32 and F16 data types");
    }

    int64_t m_out = shape_[0];
    int64_t n_out = other.shape_[1];
    int64_t k_out = shape_[1];
    std::vector<int64_t> output_shape = {m_out, n_out};
    Tensor result = Tensor::empty(output_shape, dtype_, device_, stream);

    // Perform matrix multiplication Y = X * other.
    // Row-major Y(m x n) = X(m x k) * other(k x n) mapped to column-major:
    // Y_col(n x m) = other_col(n x k) * X_col(k x m)
    cublasOperation_t opA = CUBLAS_OP_N;
    cublasOperation_t opB = CUBLAS_OP_N;
    const void* A = other.data_;
    const void* B = data_;
    void* C = result.data_;
    // For column-major GEMM:
    // A = other_col with logical shape (n_out x k_out) -> lda >= n_out
    // B = this_col  with logical shape (k_out x m_out) -> ldb >= k_out
    int lda = static_cast<int>(n_out);
    int ldb = static_cast<int>(k_out);
    int ldc = static_cast<int>(n_out);
    cublasStatus_t status;
    if (dtype_ == DType::F32) {
        float alpha = 1.0f;
        float beta = 0.0f;
        status = cublasSgemm(cublas_handle.handle, opA, opB, n_out, m_out, k_out, &alpha, static_cast<const float*>(A), lda, static_cast<const float*>(B), ldb, &beta, static_cast<float*>(C), ldc);
    } else { // DType::F16
        __half alpha = __float2half(1.0f);
        __half beta = __float2half(0.0f);
        status = cublasHgemm(cublas_handle.handle, opA, opB, n_out, m_out, k_out, &alpha, static_cast<const __half*>(A), lda, static_cast<const __half*>(B), ldb, &beta, static_cast<__half*>(C), ldc);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("tensor.h: cublas gemm failed with error code " + std::to_string(status));
    }
    return result;
}

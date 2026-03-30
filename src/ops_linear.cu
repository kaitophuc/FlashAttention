#include "ops.h"

#include <algorithm>
#include <cstddef>

/*__global__
void add_bias(float* __restrict__ Y, const float* __restrict__ b, int m, int n) {
    const int total_elements = m * n;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int col = idx % n;
        Y[idx] += b[col];
    }
}*/

template <int BLOCK_X, int BLOCK_Y, int ROWS_PER_THREAD>
__global__ void compute_db_kernel(const float* __restrict__ dY,
    float* __restrict__ db,
    int M,
    int N) {
    const int col = blockIdx.y * BLOCK_X + threadIdx.x;
    const bool col_in_range = (col < N);

    const int row_tile_start = blockIdx.x * (BLOCK_Y * ROWS_PER_THREAD);
    const int row_tile_end = min(M, row_tile_start + BLOCK_Y * ROWS_PER_THREAD);

    float sum = 0.0f;
    // Each (threadIdx.x, threadIdx.y) walks rows in its lane.
    if (col_in_range) {
        for (int row = row_tile_start + threadIdx.y; row < row_tile_end; row += BLOCK_Y) {
            sum += dY[static_cast<size_t>(row) * N + col];
        }
    }

    // Reduce across threadIdx.y for each fixed threadIdx.x (column).
    __shared__ float smem[BLOCK_Y][BLOCK_X];
    smem[threadIdx.y][threadIdx.x] = sum;
    __syncthreads();

    for (int stride = BLOCK_Y / 2; stride > 0; stride >>= 1) {
        if (threadIdx.y < stride) {
            smem[threadIdx.y][threadIdx.x] += smem[threadIdx.y + stride][threadIdx.x];
        }
        __syncthreads();
    }

    // One writer per column per block.
    if (col_in_range && threadIdx.y == 0) {
        atomicAdd(&db[col], smem[0][threadIdx.x]);
    }
}

// Using cublasLt for better performance and to leverage epilogue fusion for bias addition when available.
/*
LinearResults linear_forward(const Tensor& X, const Tensor& W, const Tensor* b, Stream& stream, CublasHandle& handle) {
    // Check input shapes and dtypes.
    // At this phase, assume all tensors are in Float32 for simplicity. In a full implementation, we would handle different dtypes and possibly mixed precision.
    // currently disable CPU support, so we can assume all tensors are on CUDA device.
    assert_non_default_stream(stream.s, "ops_linear.cu: Linear_forward");
    if (X.dtype_ != DType::F32 || W.dtype_ != DType::F32) {
        throw std::invalid_argument("ops_linear.cu: Linear_forward: Only float32 tensors are supported.");
    }
    if (X.shape_.size() != 2 || W.shape_.size() != 2) {
        throw std::invalid_argument("ops_linear.cu: Linear_forward: X and W must be 2D tensors.");
    }
    if (X.dtype_ != W.dtype_) {
        throw std::invalid_argument("ops_linear.cu: Linear_forward: X and W dtypes must match.");
    }
    if (b != nullptr) {
        if (b->shape_.size() != 1 || b->shape_[0] != W.shape_[0]) {
            throw std::invalid_argument("ops_linear.cu: Linear_forward: Bias b must be a 1D tensor with shape matching output features.");
        }
        if (b->dtype_ != X.dtype_) {
            throw std::invalid_argument("ops_linear.cu: Linear_forward: Bias b dtype must match X and W.");
        }
    }
    if (X.device_ != W.device_) {
        throw std::invalid_argument("ops_linear.cu: Linear_forward: X and W must be on the same device.");
    }
    if (b != nullptr && b->device_ != X.device_) {
        throw std::invalid_argument("ops_linear.cu: Linear_forward: Bias b must be on the same device as X and W.");
    }
    if (X.shape_[1] != W.shape_[1]) {
        throw std::invalid_argument("ops_linear.cu: Linear_forward: Inner dimensions of X and W must match.");
    }
    if (X.device_ != Device::CUDA || W.device_ != Device::CUDA) {
        throw std::invalid_argument("ops_linear.cu: Linear_forward: Only CUDA tensors are supported.");
    }

    // Create output tensor Y.
    std::vector<int64_t> y_shape = {X.shape_[0], W.shape_[0]};
    Tensor Y = Tensor::empty(y_shape, X.dtype_, X.device_, stream);
    
    // Perform matrix multiplication Y = X * W^T.
    // Row-major Y(m x n) = X(m x k) * W(n x k)^T mapped to column-major:
    // Y_col(n x m) = W_col(k x n)^T * X_col(k x m)
    cublasOperation_t opA = CUBLAS_OP_T;
    cublasOperation_t opB = CUBLAS_OP_N;
    int m_out = static_cast<int>(W.shape_[0]);
    int n_out = static_cast<int>(X.shape_[0]);
    int k_out = static_cast<int>(X.shape_[1]);
    int la = static_cast<int>(W.shape_[1]);
    int lb = static_cast<int>(X.shape_[1]);
    int lc = static_cast<int>(W.shape_[0]);
    float alpha = 1.0f;
    float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle.get(), opA, opB, m_out, n_out, k_out,
                                &alpha,
                                static_cast<float*>(W.data_), la,
                                static_cast<float*>(X.data_), lb,
                                &beta,
                                static_cast<float*>(Y.data_), lc));

    if (b != nullptr) {
        const int m = static_cast<int>(Y.shape_[0]);
        const int n = static_cast<int>(Y.shape_[1]);
        const int total_elements = m * n;
        
        int min_grid = 0, block = 0;
        cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, add_bias, 0, 0);

        int grid = (total_elements + block - 1) / block;

        add_bias<<<grid, block, 0, stream.s>>>(static_cast<float*>(Y.data_), static_cast<const float*>(b->data_), m, n);
    }
    return LinearResults{std::move(Y), LinearCtx{&X, &W, b != nullptr, X.shape_[0], W.shape_[0], X.shape_[1]}};
}
*/


LinearResults linear_forward(const Tensor& X, const Tensor& W, const Tensor* b, Stream& stream, CublasHandle& handle) {
    // Check input shapes and dtypes.
    // At this phase, assume all tensors are in Float32 for simplicity. In a full implementation, we would handle different dtypes and possibly mixed precision.
    // currently disable CPU support, so we can assume all tensors are on CUDA device.
    assert_non_default_stream(stream.s, "ops_linear.cu: Linear_forward");
    if (X.dtype_ != DType::F32 || W.dtype_ != DType::F32) {
        throw std::invalid_argument("ops_linear.cu: Linear_forward: Only float32 tensors are supported.");
    }
    if (X.shape_.size() != 2 || W.shape_.size() != 2) {
        throw std::invalid_argument("ops_linear.cu: Linear_forward: X and W must be 2D tensors.");
    }
    if (X.dtype_ != W.dtype_) {
        throw std::invalid_argument("ops_linear.cu: Linear_forward: X and W dtypes must match.");
    }
    if (b != nullptr) {
        if (b->shape_.size() != 1 || b->shape_[0] != W.shape_[0]) {
            throw std::invalid_argument("ops_linear.cu: Linear_forward: Bias b must be a 1D tensor with shape matching output features.");
        }
        if (b->dtype_ != X.dtype_) {
            throw std::invalid_argument("ops_linear.cu: Linear_forward: Bias b dtype must match X and W.");
        }
    }
    if (X.device_ != W.device_) {
        throw std::invalid_argument("ops_linear.cu: Linear_forward: X and W must be on the same device.");
    }
    if (b != nullptr && b->device_ != X.device_) {
        throw std::invalid_argument("ops_linear.cu: Linear_forward: Bias b must be on the same device as X and W.");
    }
    if (X.shape_[1] != W.shape_[1]) {
        throw std::invalid_argument("ops_linear.cu: Linear_forward: Inner dimensions of X and W must match.");
    }
    if (X.device_ != Device::CUDA || W.device_ != Device::CUDA) {
        throw std::invalid_argument("ops_linear.cu: Linear_forward: Only CUDA tensors are supported.");
    }

    // Create output tensor Y.
    const int64_t m = X.shape_[0];
    const int64_t n = W.shape_[0];
    const int64_t k = X.shape_[1];
    Tensor Y = Tensor::empty({m, n}, X.dtype_, X.device_, stream);

    cublasLtMatmulDesc_t operation_desc;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&operation_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLAS_CHECK(cublasSetStream(handle.get(), stream.s));
    
    // Perform matrix multiplication Y = X * W^T.
    // Map row-major tensors into column-major descriptors:
    // Y_col(n x m) = W_col(k x n)^T * X_col(k x m)
    cublasOperation_t opA = CUBLAS_OP_T;
    cublasOperation_t opB = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    if (b != nullptr) {
        // If bias is present, we can fuse the bias addition into the matmul using cublasLt with an epilogue.
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
        void* bias_ptr = b->data_;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
    }

    cublasLtMatrixLayout_t A_desc, B_desc, C_desc, D_desc;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_32F, k, n, k)); // W as col-major [k, n]
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_32F, k, m, k)); // X as col-major [k, m]
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&C_desc, CUDA_R_32F, n, m, n)); // Y as col-major [n, m]
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&D_desc, CUDA_R_32F, n, m, n)); // Y as col-major [n, m]

    const size_t workspace_size = 1 << 22; // 4 MiB workspace for cublasLt (tuning may be needed for larger matrices)
    void* workspace = allocate_device(workspace_size, stream);

    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

    cublasLtMatmulHeuristicResult_t heuristic_result{};
    int heuristic_count = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle.get_lt(),
        operation_desc,
        A_desc,
        B_desc,
        C_desc,
        D_desc,
        pref,
        1,
        &heuristic_result,
        &heuristic_count));

    if (heuristic_count == 0) {
        deallocate_device(workspace, stream);
        CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
        CUBLAS_CHECK(cublasLtMatmulDescDestroy(operation_desc));
        CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(A_desc));
        CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(B_desc));
        CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(C_desc));
        CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(D_desc));
        throw std::runtime_error("No suitable cublasLt matmul algorithm found.");
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(handle.get_lt(),
        operation_desc,
        &alpha,
        W.data_, A_desc,
        X.data_, B_desc,
        &beta,
        Y.data_, C_desc,
        Y.data_, D_desc,
        &heuristic_result.algo,
        workspace, workspace_size,
        stream.s));

    deallocate_device(workspace, stream);
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(operation_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(A_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(B_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(C_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(D_desc));

    return LinearResults{std::move(Y), LinearCtx{&X, &W, b != nullptr, X.shape_[0], W.shape_[0], X.shape_[1]}};
}

LinearGrads linear_backward(const Tensor& dY, const LinearCtx& ctx, bool needs_dX, bool needs_dW, bool needs_db, Stream& stream, CublasHandle& handle) {
    // Check input shapes and dtypes.
    assert_non_default_stream(stream.s, "ops_linear.cu: Linear_backward");
    if (ctx.X == nullptr || ctx.W == nullptr) {
        throw std::invalid_argument("ops_linear.cu: Linear_backward: LinearCtx contains null tensor pointers.");
    }
    if (dY.dtype_ != DType::F32) {
        throw std::invalid_argument("ops_linear.cu: Linear_backward: Only float32 tensors are supported.");
    }
    if (dY.shape_.size() != 2) {
        throw std::invalid_argument("ops_linear.cu: Linear_backward: dY must be a 2D tensor.");
    }
    if (dY.dtype_ != ctx.X->dtype_ || dY.dtype_ != ctx.W->dtype_) {
        throw std::invalid_argument("ops_linear.cu: Linear_backward: dY dtype must match X and W.");
    }
    if (dY.device_ != ctx.X->device_ || dY.device_ != ctx.W->device_) {
        throw std::invalid_argument("ops_linear.cu: Linear_backward: dY must be on the same device as X and W.");
    }
    if (dY.shape_[0] != ctx.m || dY.shape_[1] != ctx.n) {
        throw std::invalid_argument("ops_linear.cu: Linear_backward: dY shape must match output shape of forward pass.");
    }
    if (ctx.has_bias && needs_db && (ctx.n <= 0)) {
        throw std::invalid_argument("ops_linear.cu: Linear_backward: Invalid shape for bias gradient.");
    }
    if (ctx.k <= 0) {
        throw std::invalid_argument("ops_linear.cu: Linear_backward: Invalid inner dimension k in context.");
    }

    std::optional<Tensor> dX;
    if (needs_dX) {
        dX = Tensor::empty(ctx.X->shape_, ctx.X->dtype_, ctx.X->device_, stream);
    }
    std::optional<Tensor> dW;
    if (needs_dW) {
        dW = Tensor::empty(ctx.W->shape_, ctx.W->dtype_, ctx.W->device_, stream);
    }
    std::optional<Tensor> db;
    if (ctx.has_bias && needs_db) {
        db = Tensor::empty({ctx.n}, ctx.X->dtype_, ctx.X->device_, stream);
    }

    // Compute dX = dY * W
    if (needs_dX) {
        // Row-major dX(m x k) = dY(m x n) * W(n x k) mapped to column-major:
        // dX_col(k x m) = W_col(k x n) * dY_col(n x m)
        cublasOperation_t opA = CUBLAS_OP_N;
        cublasOperation_t opB = CUBLAS_OP_N;
        int m_out = static_cast<int>(ctx.W->shape_[1]);
        int n_out = static_cast<int>(dY.shape_[0]);
        int k_out = static_cast<int>(dY.shape_[1]);
        int la = static_cast<int>(ctx.W->shape_[1]);
        int lb = static_cast<int>(dY.shape_[1]);
        int lc = static_cast<int>(ctx.W->shape_[1]);

        float alpha = 1.0f;
        float beta = 0.0f;
        CUBLAS_CHECK(cublasSetStream(handle.get(), stream.s));
        CUBLAS_CHECK(cublasSgemm(handle.get(), opA, opB, m_out, n_out, k_out,
                                    &alpha,
                                    static_cast<const float*>(ctx.W->data_), la,
                                    static_cast<const float*>(dY.data_), lb,
                                    &beta,
                                    static_cast<float*>(dX.value().data_), lc));
    }
    
    // Compute dW = dY^T * X
    if (needs_dW) {
        // Row-major dW(n x k) = dY(m x n)^T * X(m x k) mapped to column-major:
        // dW_col(k x n) = X_col(k x m) * dY_col(n x m)^T
        cublasOperation_t opA = CUBLAS_OP_N;
        cublasOperation_t opB = CUBLAS_OP_T;
        int m_out = static_cast<int>(ctx.X->shape_[1]);
        int n_out = static_cast<int>(dY.shape_[1]);
        int k_out = static_cast<int>(ctx.X->shape_[0]);
        int la = static_cast<int>(ctx.X->shape_[1]);
        int lb = static_cast<int>(dY.shape_[1]);
        int lc = static_cast<int>(ctx.X->shape_[1]);
        float alpha = 1.0f;
        float beta = 0.0f;
        CUBLAS_CHECK(cublasSetStream(handle.get(), stream.s));
        CUBLAS_CHECK(cublasSgemm(handle.get(), opA, opB, m_out, n_out, k_out,
                                    &alpha,
                                    static_cast<const float*>(ctx.X->data_), la,
                                    static_cast<const float*>(dY.data_), lb,
                                    &beta,
                                    static_cast<float*>(dW.value().data_), lc));
    }

    // Compute db = sum(dY, dim=0)
    if (ctx.has_bias && needs_db) {
        constexpr int BLOCK_X = 128;
        constexpr int BLOCK_Y = 8;
        constexpr int ROWS_PER_THREAD = 4;

        const int m = static_cast<int>(dY.shape_[0]);
        const int n = static_cast<int>(dY.shape_[1]);

        dim3 block(BLOCK_X, BLOCK_Y);
        const int rows_per_block = BLOCK_Y * ROWS_PER_THREAD;
        int grid_x = (m + rows_per_block - 1) / rows_per_block;
        int grid_y = (n + BLOCK_X - 1) / BLOCK_X;
        dim3 grid(grid_x, grid_y);

        db.value().zero_(stream); // Ensure db is zeroed before accumulation

        compute_db_kernel<BLOCK_X, BLOCK_Y, ROWS_PER_THREAD><<<grid, block, 0, stream.s>>>(
            static_cast<const float*>(dY.data_),
            static_cast<float*>(db.value().data_),
            m,
            n);
    } 
        
    return LinearGrads{
        std::move(dX),
        std::move(dW),
        std::move(db),
        needs_dX,
        needs_dW,
        (ctx.has_bias && needs_db)};
}

#pragma once

#include <cstdint>

#include "tensor.h"
#include "cublass_handle.h"
#include <stdexcept>
#include <utility>
#include <optional>
#include <cublasLt.h>

struct LinearCtx {
    const Tensor* X;
    const Tensor* W;
    bool has_bias;
    int64_t m, n, k;
};    
struct LinearGrads {
    std::optional<Tensor> dX;
    std::optional<Tensor> dW;
    std::optional<Tensor> db; // Optional, may be empty if b was not provided.
    bool has_dX;
    bool has_dW;
    bool has_db;
};

struct LinearResults {
    Tensor Y;
    LinearCtx ctx;
};

LinearResults linear_forward(const Tensor& X, const Tensor& W, const Tensor* b, Stream* stream, CublasHandle& cublas_handle); //const
LinearGrads linear_backward(const Tensor& dY, const LinearCtx& ctx, bool needs_dX, bool needs_dW, bool needs_db, Stream* stream, CublasHandle& cublas_handle); //const


struct LayerNormGrads {
    std::optional<Tensor> dX;
    std::optional<Tensor> dgamma;
    std::optional<Tensor> dbeta;
    bool has_dX;
    bool has_dgamma;
    bool has_dbeta;
};
struct LayerNormCtx {
    const Tensor* X;
    const Tensor* gamma;
    Tensor mean;
    Tensor rstd;
    float eps;
    int64_t m;
    int64_t n;
};

struct LayerNormResults {
    Tensor Y;
    LayerNormCtx ctx;
};
LayerNormResults layernorm_forward(const Tensor& X, const Tensor& gamma, const Tensor& beta, float eps, Stream* stream); //const
LayerNormGrads layernorm_backward(const Tensor& dY, const LayerNormCtx& ctx, bool needs_dX, bool needs_dgamma, bool needs_dbeta, Stream* stream); //const

struct ReluGrads {
    Tensor dX;
};
struct ReluCtx {
    const Tensor* X;
};
struct ReluResults {
    Tensor Y;
    ReluCtx ctx;
};
ReluResults relu_forward(const Tensor& X, Stream* stream); //const
ReluGrads relu_backward(const Tensor& dY, const ReluCtx& ctx, Stream* stream); //const

struct DropoutCtx {
    Tensor mask;
    float p;
    bool training;
};

struct DropoutResults {
    Tensor Y;
    DropoutCtx ctx;
};

DropoutResults dropout_forward(const Tensor& X, float p, bool training, uint64_t seed, Stream* stream);
Tensor dropout_backward(const Tensor& dY, const DropoutCtx& ctx, Stream* stream);

struct SoftmaxGrads {
    Tensor dX;
};
Tensor softmax_forward(const Tensor& X, Stream* stream);
SoftmaxGrads softmax_backward(const Tensor& dY, const Tensor& Y, Stream* stream);

struct SoftmaxCrossEntropyCtx {
    const Tensor* labels;
    Tensor probs;
    int64_t m;
    int64_t n;
};

struct SoftmaxCrossEntropyResults {
    Tensor loss;
    SoftmaxCrossEntropyCtx ctx;
};

struct SoftmaxCrossEntropyGrads {
    Tensor dX;
};

SoftmaxCrossEntropyResults softmax_cross_entropy_forward(const Tensor& logits, const Tensor& labels, Stream* stream);
SoftmaxCrossEntropyGrads softmax_cross_entropy_backward(const SoftmaxCrossEntropyCtx& ctx, Stream* stream);

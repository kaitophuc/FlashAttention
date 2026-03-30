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
inline LinearResults linear_forward(const Tensor& X, const Tensor& W, const Tensor* b = nullptr) {
    Stream stream = current_stream();
    return linear_forward(X, W, b, &stream, current_cublas_handle());
}
inline LinearGrads linear_backward(const Tensor& dY, const LinearCtx& ctx, bool needs_dX = true, bool needs_dW = true, bool needs_db = true) {
    Stream stream = current_stream();
    return linear_backward(dY, ctx, needs_dX, needs_dW, needs_db, &stream, current_cublas_handle());
}


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
inline LayerNormResults layernorm_forward(const Tensor& X, const Tensor& gamma, const Tensor& beta, float eps) {
    Stream stream = current_stream();
    return layernorm_forward(X, gamma, beta, eps, &stream);
}
inline LayerNormGrads layernorm_backward(const Tensor& dY, const LayerNormCtx& ctx, bool needs_dX = true, bool needs_dgamma = true, bool needs_dbeta = true) {
    Stream stream = current_stream();
    return layernorm_backward(dY, ctx, needs_dX, needs_dgamma, needs_dbeta, &stream);
}

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
inline ReluResults relu_forward(const Tensor& X) {
    Stream stream = current_stream();
    return relu_forward(X, &stream);
}
inline ReluGrads relu_backward(const Tensor& dY, const ReluCtx& ctx) {
    Stream stream = current_stream();
    return relu_backward(dY, ctx, &stream);
}

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
inline DropoutResults dropout_forward(const Tensor& X, float p, bool training, uint64_t seed) {
    Stream stream = current_stream();
    return dropout_forward(X, p, training, seed, &stream);
}
inline Tensor dropout_backward(const Tensor& dY, const DropoutCtx& ctx) {
    Stream stream = current_stream();
    return dropout_backward(dY, ctx, &stream);
}

struct SoftmaxGrads {
    Tensor dX;
};
Tensor softmax_forward(const Tensor& X, Stream* stream);
SoftmaxGrads softmax_backward(const Tensor& dY, const Tensor& Y, Stream* stream);
inline Tensor softmax_forward(const Tensor& X) {
    Stream stream = current_stream();
    return softmax_forward(X, &stream);
}
inline SoftmaxGrads softmax_backward(const Tensor& dY, const Tensor& Y) {
    Stream stream = current_stream();
    return softmax_backward(dY, Y, &stream);
}

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
inline SoftmaxCrossEntropyResults softmax_cross_entropy_forward(const Tensor& logits, const Tensor& labels) {
    Stream stream = current_stream();
    return softmax_cross_entropy_forward(logits, labels, &stream);
}
inline SoftmaxCrossEntropyGrads softmax_cross_entropy_backward(const SoftmaxCrossEntropyCtx& ctx) {
    Stream stream = current_stream();
    return softmax_cross_entropy_backward(ctx, &stream);
}

Tensor classification_correct_count(const Tensor& logits, const Tensor& labels, Stream* stream);
inline Tensor classification_correct_count(const Tensor& logits, const Tensor& labels) {
    Stream stream = current_stream();
    return classification_correct_count(logits, labels, &stream);
}

void sgd_update_(Tensor& param, const Tensor& grad, float lr, Stream* stream);
inline void sgd_update_(Tensor& param, const Tensor& grad, float lr) {
    Stream stream = current_stream();
    sgd_update_(param, grad, lr, &stream);
}

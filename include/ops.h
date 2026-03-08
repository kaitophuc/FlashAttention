#pragma once

#include <cstdint>

#include "tensor.h"
#include "cublass_handle.h"
#include <stdexcept>
#include <utility>

struct LinearCtx {
    const Tensor* X;
    const Tensor* W;
};    
struct LinearGrads {
    Tensor dX;
    Tensor dW;
    Tensor db; // Optional, may be empty if b was not provided.
};

struct LinearResults {
    Tensor Y;
    LinearCtx ctx;
};

LinearResults linear_forward(const Tensor& X, const Tensor& W, const Tensor* b, Stream* stream, CublasHandle& cublas_handle); //const
LinearGrads linear_backward(const Tensor& dY, const Tensor& X, const Tensor& W, const Tensor* b, Stream* stream); //const


struct LayerNormGrads {
    Tensor dX;
    Tensor dgamma;
    Tensor dbeta;
};
struct LayerNormCtx {
    Tensor Y;
    Tensor gamma;
    Tensor mean;
    Tensor rstd;
};

struct LayerNormResults {
    Tensor Y;
    LayerNormCtx ctx;
};
LayerNormResults layernorm_forward(const Tensor& X, const Tensor& gamma, const Tensor& beta, float eps, Tensor* mean, Tensor* rstd, Stream* stream); //const
LayerNormGrads layernorm_backward(const Tensor& dY, const Tensor& X, const Tensor& gamma, const Tensor& beta, const Tensor& mean, const Tensor& rstd, float eps, Stream* stream); //const

struct GeluGrads {
    Tensor dX;
};
struct GeluCtx {
    Tensor X;
};
struct GeluResults {
    Tensor Y;
    GeluCtx ctx;
};
GeluResults gelu_forward(const Tensor& X, Stream* stream); //const
GeluGrads gelu_backward(const Tensor& dY, const Tensor& X, Stream* stream); //const

struct DropoutCtx {
    Tensor mask;
    float p;
    bool training;
};

struct DropoutResults {
    Tensor Y;
    DropoutCtx ctx;
};

DropoutResults dropout_forward(
    const Tensor& X,
    float p,
    bool training,
    uint64_t seed,
    Stream* stream);
Tensor dropout_backward(const Tensor& dY, const DropoutCtx& ctx, Stream* stream);

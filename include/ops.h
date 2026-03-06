#pragma once

#include "tensor.h"

Tensor linear_forward(const Tensor& X, const Tensor& W, const Tensor* b, Stream* stream); //const
Tensor gelu_forward(const Tensor& X, Stream* stream); //const
Tensor layernorm_forward(const Tensor& X, const Tensor& gamma, const Tensor& beta, float eps, Tensor* mean, Tensor* rstd, Stream* stream); //const

struct LinearGrads {
    Tensor dX;
    Tensor dW;
    Tensor db; // Optional, may be empty if b was not provided.
};

LinearGrads linear_backward(const Tensor& dY, const Tensor& X, const Tensor& W, const Tensor* b, Stream* stream); //const

struct LayerNormGrads {
    Tensor dX;
    Tensor dgamma;
    Tensor dbeta;
};

LayerNormGrads layernorm_backward(const Tensor& dY, const Tensor& X, const Tensor& gamma, const Tensor& beta, const Tensor& mean, const Tensor& rstd, float eps, Stream* stream); //const

struct GeluGrads {
    Tensor dX;
};

GeluGrads gelu_backward(const Tensor& dY, const Tensor& X, Stream* stream); //const

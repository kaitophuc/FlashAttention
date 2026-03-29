#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cublass_handle.h"
#include "ops.h"
#include "tensor.h"

namespace py = pybind11;

struct LinearContextPy {
    LinearCtx ctx{};
    std::shared_ptr<Tensor> x;
    std::shared_ptr<Tensor> w;
};

struct LayerNormContextPy {
    LayerNormCtx ctx;
    std::shared_ptr<Tensor> x;
    std::shared_ptr<Tensor> gamma;

    LayerNormContextPy(LayerNormCtx&& ctx_in,
                       std::shared_ptr<Tensor> x_in,
                       std::shared_ptr<Tensor> gamma_in)
        : ctx(std::move(ctx_in)), x(std::move(x_in)), gamma(std::move(gamma_in)) {
        ctx.X = x.get();
        ctx.gamma = gamma.get();
    }
};

struct ReluContextPy {
    ReluCtx ctx{};
    std::shared_ptr<Tensor> x;
};

struct SoftmaxCrossEntropyContextPy {
    SoftmaxCrossEntropyCtx ctx;
    std::shared_ptr<Tensor> labels;

    SoftmaxCrossEntropyContextPy(SoftmaxCrossEntropyCtx&& ctx_in,
                                 std::shared_ptr<Tensor> labels_in)
        : ctx(std::move(ctx_in)), labels(std::move(labels_in)) {
        ctx.labels = labels.get();
    }
};

struct LinearGradsPy {
    std::shared_ptr<Tensor> dX;
    std::shared_ptr<Tensor> dW;
    std::shared_ptr<Tensor> db;
    bool has_dX{false};
    bool has_dW{false};
    bool has_db{false};
};

struct LayerNormGradsPy {
    std::shared_ptr<Tensor> dX;
    std::shared_ptr<Tensor> dgamma;
    std::shared_ptr<Tensor> dbeta;
    bool has_dX{false};
    bool has_dgamma{false};
    bool has_dbeta{false};
};

inline Stream& py_default_stream() {
    static Stream stream(cudaStream_t(0));
    return stream;
}

inline CublasHandle& py_default_cublas_handle() {
    static CublasHandle handle;
    return handle;
}

inline std::shared_ptr<Tensor> make_tensor_shared(Tensor&& t) {
    return std::make_shared<Tensor>(std::move(t));
}

inline std::shared_ptr<Tensor> optional_tensor_to_shared(std::optional<Tensor>&& t) {
    if (!t.has_value()) {
        return nullptr;
    }
    return make_tensor_shared(std::move(*t));
}

inline std::shared_ptr<Tensor> make_tensor(const std::vector<int64_t>& shape, DType dtype, Device device) {
    if (device == Device::CUDA) {
        return make_tensor_shared(Tensor(shape, dtype, device, py_default_stream()));
    }
    return make_tensor_shared(Tensor(shape, dtype, device));
}

inline std::shared_ptr<Tensor> tensor_empty(const std::vector<int64_t>& shape, DType dtype, Device device) {
    if (device == Device::CUDA) {
        return make_tensor_shared(Tensor::empty(shape, dtype, device, py_default_stream()));
    }
    return make_tensor_shared(Tensor::empty(shape, dtype, device));
}

inline std::shared_ptr<Tensor> tensor_zeros(const std::vector<int64_t>& shape, DType dtype, Device device) {
    if (device == Device::CUDA) {
        return make_tensor_shared(Tensor::zeros(shape, dtype, device, py_default_stream()));
    }
    return make_tensor_shared(Tensor::zeros(shape, dtype, device));
}

inline std::shared_ptr<Tensor> tensor_clone(const Tensor& src, Device device) {
    if (device == Device::CUDA) {
        return make_tensor_shared(src.clone(device, py_default_stream()));
    }
    return make_tensor_shared(src.clone(device));
}

inline void tensor_copy_from_list_float(Tensor& dst, const std::vector<float>& values) {
    if (dst.dtype_ != DType::F32) {
        throw std::invalid_argument("ktorch: copy_from_list_float only supports F32 tensors.");
    }
    dst.copy_from(values);
    if (dst.device_ == Device::CUDA) {
        py_default_stream().synchronize();
    }
}

inline void tensor_copy_from_list_int32(Tensor& dst, const std::vector<int32_t>& values) {
    if (dst.dtype_ != DType::I32) {
        throw std::invalid_argument("ktorch: copy_from_list_int32 only supports I32 tensors.");
    }
    dst.copy_from(values);
    if (dst.device_ == Device::CUDA) {
        py_default_stream().synchronize();
    }
}

inline std::vector<float> tensor_to_list_float(const Tensor& src) {
    if (src.dtype_ != DType::F32) {
        throw std::invalid_argument("ktorch: to_list_float only supports F32 tensors.");
    }
    return src.to_vector<float>();
}

void bind_dtype_device(py::module_& m);
void bind_tensor(py::module_& m);
void bind_linear(py::module_& m);
void bind_layernorm(py::module_& m);
void bind_relu(py::module_& m);
void bind_softmax(py::module_& m);
void bind_softmax_cross_entropy(py::module_& m);

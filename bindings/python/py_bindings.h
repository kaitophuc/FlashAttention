#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
#include <cstring>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
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

inline Stream py_current_stream() {
    return current_stream();
}

inline CublasHandle& py_default_cublas_handle() {
    thread_local CublasHandle handle;
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
        Stream stream = py_current_stream();
        return make_tensor_shared(Tensor(shape, dtype, device, stream));
    }
    return make_tensor_shared(Tensor(shape, dtype, device));
}

inline std::shared_ptr<Tensor> tensor_empty(const std::vector<int64_t>& shape, DType dtype, Device device) {
    if (device == Device::CUDA) {
        Stream stream = py_current_stream();
        return make_tensor_shared(Tensor::empty(shape, dtype, device, stream));
    }
    return make_tensor_shared(Tensor::empty(shape, dtype, device));
}

inline std::shared_ptr<Tensor> tensor_zeros(const std::vector<int64_t>& shape, DType dtype, Device device) {
    if (device == Device::CUDA) {
        Stream stream = py_current_stream();
        return make_tensor_shared(Tensor::zeros(shape, dtype, device, stream));
    }
    return make_tensor_shared(Tensor::zeros(shape, dtype, device));
}

inline std::shared_ptr<Tensor> tensor_random_uniform(const std::vector<int64_t>& shape,
                                                     float low,
                                                     float high,
                                                     uint64_t seed,
                                                     DType dtype,
                                                     Device device) {
    if (device == Device::CUDA) {
        Stream stream = py_current_stream();
        return make_tensor_shared(Tensor::random_uniform(shape, low, high, seed, dtype, device, stream));
    }
    return make_tensor_shared(Tensor::random_uniform(shape, low, high, seed, dtype, device));
}

inline std::shared_ptr<Tensor> tensor_clone(const Tensor& src, Device device) {
    if (device == Device::CUDA) {
        Stream stream = py_current_stream();
        return make_tensor_shared(src.clone(device, stream));
    }
    return make_tensor_shared(src.clone(device));
}

inline void tensor_copy_from_list_float(Tensor& dst, const std::vector<float>& values) {
    if (dst.dtype_ != DType::F32) {
        throw std::invalid_argument("ktorch: copy_from_list_float only supports F32 tensors.");
    }
    dst.copy_from(values);
    if (dst.device_ == Device::CUDA) {
        Stream stream = py_current_stream();
        stream.synchronize();
    }
}

inline void tensor_copy_from_list_int32(Tensor& dst, const std::vector<int32_t>& values) {
    if (dst.dtype_ != DType::I32) {
        throw std::invalid_argument("ktorch: copy_from_list_int32 only supports I32 tensors.");
    }
    dst.copy_from(values);
    if (dst.device_ == Device::CUDA) {
        Stream stream = py_current_stream();
        stream.synchronize();
    }
}

inline std::shared_ptr<Tensor> tensor_from_list_float(const std::vector<int64_t>& shape,
                                                      const std::vector<float>& values,
                                                      Device device) {
    auto out = tensor_empty(shape, DType::F32, device);
    tensor_copy_from_list_float(*out, values);
    return out;
}

inline std::shared_ptr<Tensor> tensor_from_list_int32(const std::vector<int64_t>& shape,
                                                      const std::vector<int32_t>& values,
                                                      Device device) {
    auto out = tensor_empty(shape, DType::I32, device);
    tensor_copy_from_list_int32(*out, values);
    return out;
}

inline std::vector<float> tensor_to_list_float(const Tensor& src) {
    if (src.dtype_ != DType::F32) {
        throw std::invalid_argument("ktorch: to_list_float only supports F32 tensors.");
    }
    return src.to_vector<float>();
}

inline py::array ensure_c_contiguous_dtype(const py::object& obj, const py::dtype& expected_dtype, const char* fn_name) {
    py::array arr = py::array::ensure(obj);
    if (!arr) {
        throw std::invalid_argument(std::string(fn_name) + ": expected an object exposing the buffer protocol.");
    }
    if (!arr.dtype().is(expected_dtype)) {
        throw std::invalid_argument(std::string(fn_name) + ": dtype mismatch.");
    }
    const bool is_c_contiguous = (arr.flags() & py::array::c_style) == py::array::c_style;
    if (!is_c_contiguous) {
        throw std::invalid_argument(std::string(fn_name) + ": input must be C-contiguous.");
    }
    return arr;
}

inline std::vector<py::ssize_t> shape_to_py(const std::vector<int64_t>& shape) {
    std::vector<py::ssize_t> out;
    out.reserve(shape.size());
    for (int64_t d : shape) {
        out.push_back(static_cast<py::ssize_t>(d));
    }
    return out;
}

inline void tensor_copy_from_buffer_float(Tensor& dst, const py::object& obj) {
    if (dst.dtype_ != DType::F32) {
        throw std::invalid_argument("ktorch: copy_from_buffer_float only supports F32 tensors.");
    }
    py::array arr = ensure_c_contiguous_dtype(obj, py::dtype::of<float>(), "ktorch.copy_from_buffer_float");
    if (static_cast<size_t>(arr.size()) != dst.numel_) {
        throw std::invalid_argument("ktorch.copy_from_buffer_float: input element count must match tensor.numel().");
    }

    if (dst.device_ == Device::CUDA) {
        Stream stream = py_current_stream();
        CUDA_CHECK(cudaMemcpyAsync(dst.data_, arr.data(), dst.nbytes_, cudaMemcpyHostToDevice, stream.s));
        stream.synchronize();
    } else {
        std::memcpy(dst.data_, arr.data(), dst.nbytes_);
    }
}

inline void tensor_copy_from_buffer_int32(Tensor& dst, const py::object& obj) {
    if (dst.dtype_ != DType::I32) {
        throw std::invalid_argument("ktorch: copy_from_buffer_int32 only supports I32 tensors.");
    }
    py::array arr = ensure_c_contiguous_dtype(obj, py::dtype::of<int32_t>(), "ktorch.copy_from_buffer_int32");
    if (static_cast<size_t>(arr.size()) != dst.numel_) {
        throw std::invalid_argument("ktorch.copy_from_buffer_int32: input element count must match tensor.numel().");
    }

    if (dst.device_ == Device::CUDA) {
        Stream stream = py_current_stream();
        CUDA_CHECK(cudaMemcpyAsync(dst.data_, arr.data(), dst.nbytes_, cudaMemcpyHostToDevice, stream.s));
        stream.synchronize();
    } else {
        std::memcpy(dst.data_, arr.data(), dst.nbytes_);
    }
}

inline py::array_t<float> tensor_to_numpy_float(const Tensor& src) {
    if (src.dtype_ != DType::F32) {
        throw std::invalid_argument("ktorch: to_numpy_float only supports F32 tensors.");
    }
    py::array_t<float> out(shape_to_py(src.shape_));
    if (src.device_ == Device::CUDA) {
        Stream stream = py_current_stream();
        CUDA_CHECK(cudaMemcpyAsync(out.mutable_data(), src.data_, src.nbytes_, cudaMemcpyDeviceToHost, stream.s));
        stream.synchronize();
    } else {
        std::memcpy(out.mutable_data(), src.data_, src.nbytes_);
    }
    return out;
}

inline py::array_t<int32_t> tensor_to_numpy_int32(const Tensor& src) {
    if (src.dtype_ != DType::I32) {
        throw std::invalid_argument("ktorch: to_numpy_int32 only supports I32 tensors.");
    }
    py::array_t<int32_t> out(shape_to_py(src.shape_));
    if (src.device_ == Device::CUDA) {
        Stream stream = py_current_stream();
        CUDA_CHECK(cudaMemcpyAsync(out.mutable_data(), src.data_, src.nbytes_, cudaMemcpyDeviceToHost, stream.s));
        stream.synchronize();
    } else {
        std::memcpy(out.mutable_data(), src.data_, src.nbytes_);
    }
    return out;
}

inline float tensor_item_float(const Tensor& src) {
    if (src.dtype_ != DType::F32) {
        throw std::invalid_argument("ktorch: item_float only supports F32 tensors.");
    }
    if (src.numel_ != 1) {
        throw std::invalid_argument("ktorch: item_float requires a tensor with exactly one element.");
    }
    float out = 0.0f;
    if (src.device_ == Device::CUDA) {
        Stream stream = py_current_stream();
        CUDA_CHECK(cudaMemcpyAsync(&out, src.data_, sizeof(float), cudaMemcpyDeviceToHost, stream.s));
        stream.synchronize();
    } else {
        std::memcpy(&out, src.data_, sizeof(float));
    }
    return out;
}

inline int32_t tensor_item_int32(const Tensor& src) {
    if (src.dtype_ != DType::I32) {
        throw std::invalid_argument("ktorch: item_int32 only supports I32 tensors.");
    }
    if (src.numel_ != 1) {
        throw std::invalid_argument("ktorch: item_int32 requires a tensor with exactly one element.");
    }
    int32_t out = 0;
    if (src.device_ == Device::CUDA) {
        Stream stream = py_current_stream();
        CUDA_CHECK(cudaMemcpyAsync(&out, src.data_, sizeof(int32_t), cudaMemcpyDeviceToHost, stream.s));
        stream.synchronize();
    } else {
        std::memcpy(&out, src.data_, sizeof(int32_t));
    }
    return out;
}

void bind_dtype_device(py::module_& m);
void bind_stream(py::module_& m);
void bind_tensor(py::module_& m);
void bind_linear(py::module_& m);
void bind_layernorm(py::module_& m);
void bind_relu(py::module_& m);
void bind_softmax(py::module_& m);
void bind_softmax_cross_entropy(py::module_& m);
void bind_classification(py::module_& m);
void bind_optimizer(py::module_& m);

#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
#include <cstring>

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
    if (dst.is_read_only()) {
        throw std::invalid_argument("ktorch.copy_from_list_float: destination tensor is read-only.");
    }
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
    if (dst.is_read_only()) {
        throw std::invalid_argument("ktorch.copy_from_list_int32: destination tensor is read-only.");
    }
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

inline std::vector<int32_t> tensor_to_list_int32(const Tensor& src) {
    if (src.dtype_ != DType::I32) {
        throw std::invalid_argument("ktorch: to_list_int32 only supports I32 tensors.");
    }
    return src.to_vector<int32_t>();
}

inline py::object ensure_torch_tensor(const py::object& obj, const char* fn_name) {
    try {
        py::module_ torch = py::module_::import("torch");
        py::object torch_tensor_type = torch.attr("Tensor");
        if (!py::isinstance(obj, torch_tensor_type)) {
            throw std::invalid_argument(std::string(fn_name) + ": expected a torch.Tensor.");
        }
    } catch (const py::error_already_set&) {
        throw std::runtime_error(std::string(fn_name) + ": torch is required.");
    }
    return obj;
}

inline uint64_t torch_tensor_version(const py::object& t, const char* fn_name) {
    try {
        return t.attr("_version").cast<uint64_t>();
    } catch (const py::error_already_set&) {
        throw std::runtime_error(std::string(fn_name) + ": unable to read torch tensor _version.");
    }
}

inline std::shared_ptr<void> make_py_owner_token(const py::object& owner_obj) {
    auto* p = new py::object(owner_obj);
    return std::shared_ptr<void>(p, [](void* raw) {
        py::gil_scoped_acquire gil;
        delete static_cast<py::object*>(raw);
    });
}

inline bool tensor_validate_torch_borrow_version(const Tensor& src) {
    if (src.borrow_source_ != Tensor::BorrowSource::Torch) {
        return true;
    }
    if (src.external_owner_ == nullptr) {
        throw std::runtime_error("ktorch: borrowed torch tensor missing owner handle.");
    }
    auto* owner = static_cast<py::object*>(src.external_owner_.get());
    if (owner == nullptr) {
        throw std::runtime_error("ktorch: invalid borrowed torch owner handle.");
    }
    const uint64_t now = torch_tensor_version(*owner, "ktorch.tensor_validate_torch_borrow_version");
    return now == src.borrow_version_;
}

inline std::shared_ptr<Tensor> tensor_from_torch_borrow_cpu(const py::object& obj,
                                                            bool require_contiguous,
                                                            bool require_pinned) {
    py::object t = ensure_torch_tensor(obj, "ktorch.from_torch_borrow_cpu");
    const std::string src_device_type = py::str(t.attr("device").attr("type"));
    if (src_device_type != "cpu") {
        throw std::invalid_argument("ktorch.from_torch_borrow_cpu: only CPU torch tensors are supported.");
    }
    if (require_contiguous && !t.attr("is_contiguous")().cast<bool>()) {
        throw std::invalid_argument("ktorch.from_torch_borrow_cpu: input tensor must be contiguous.");
    }
    if (require_pinned && !t.attr("is_pinned")().cast<bool>()) {
        throw std::invalid_argument("ktorch.from_torch_borrow_cpu: input tensor must be pinned (require_pinned=True).");
    }

    const py::object dtype_obj = t.attr("dtype");
    py::module_ torch = py::module_::import("torch");
    DType dtype;
    if (py::bool_(dtype_obj.is(torch.attr("float32")))) {
        dtype = DType::F32;
    } else if (py::bool_(dtype_obj.is(torch.attr("int32")))) {
        dtype = DType::I32;
    } else {
        throw std::invalid_argument("ktorch.from_torch_borrow_cpu: only torch.float32 and torch.int32 are supported.");
    }

    py::tuple shape_t = t.attr("shape");
    std::vector<int64_t> shape;
    shape.reserve(shape_t.size());
    for (py::handle d : shape_t) {
        shape.push_back(py::cast<int64_t>(d));
    }

    py::tuple stride_t = t.attr("stride")();
    std::vector<int64_t> strides;
    strides.reserve(stride_t.size());
    for (py::handle d : stride_t) {
        strides.push_back(py::cast<int64_t>(d));
    }

    void* ptr = reinterpret_cast<void*>(t.attr("data_ptr")().cast<uintptr_t>());
    const uint64_t version = torch_tensor_version(t, "ktorch.from_torch_borrow_cpu");
    std::shared_ptr<void> owner = make_py_owner_token(t);
    return make_tensor_shared(Tensor::borrowed_external(
        ptr,
        std::move(shape),
        std::move(strides),
        dtype,
        Device::CPU,
        true,
        Tensor::BorrowSource::Torch,
        version,
        std::move(owner)));
}

inline void tensor_copy_from_tensor(Tensor& dst,
                                    const Tensor& src,
                                    Stream stream,
                                    bool strict_immutability = true) {
    assert_non_default_stream(stream.s, "ktorch.Tensor.copy_from");
    if (!src.is_contiguous() || !dst.is_contiguous()) {
        throw std::invalid_argument("ktorch.Tensor.copy_from: src and dst must be contiguous.");
    }
    if (strict_immutability && src.borrow_source_ == Tensor::BorrowSource::Torch) {
        if (!tensor_validate_torch_borrow_version(src)) {
            throw std::invalid_argument("ktorch.Tensor.copy_from: borrowed torch tensor was mutated before copy.");
        }
    }

    dst.copy_from(src, stream);

    if (strict_immutability && src.borrow_source_ == Tensor::BorrowSource::Torch) {
        if (!tensor_validate_torch_borrow_version(src)) {
            throw std::invalid_argument("ktorch.Tensor.copy_from: borrowed torch tensor was mutated during copy submission.");
        }
    }
}

inline void tensor_copy_from_torch_float(Tensor& dst, const py::object& obj) {
    if (dst.is_read_only()) {
        throw std::invalid_argument("ktorch.copy_from_torch_float: destination tensor is read-only.");
    }
    if (dst.dtype_ != DType::F32) {
        throw std::invalid_argument("ktorch: copy_from_torch_float only supports F32 tensors.");
    }
    py::object t = ensure_torch_tensor(obj, "ktorch.copy_from_torch_float");
    if (!t.attr("is_contiguous")().cast<bool>()) {
        throw std::invalid_argument("ktorch.copy_from_torch_float: input tensor must be contiguous.");
    }
    if (t.attr("numel")().cast<size_t>() != dst.numel_) {
        throw std::invalid_argument("ktorch.copy_from_torch_float: input element count must match tensor.numel().");
    }
    const std::string src_device_type = py::str(t.attr("device").attr("type"));
    const void* src_ptr = reinterpret_cast<const void*>(t.attr("data_ptr")().cast<uintptr_t>());

    if (dst.device_ == Device::CUDA) {
        Stream stream = py_current_stream();
        if (src_device_type == "cuda") {
            CUDA_CHECK(cudaMemcpyAsync(dst.data_, src_ptr, dst.nbytes_, cudaMemcpyDeviceToDevice, stream.s));
        } else if (src_device_type == "cpu") {
            CUDA_CHECK(cudaMemcpyAsync(dst.data_, src_ptr, dst.nbytes_, cudaMemcpyHostToDevice, stream.s));
        } else {
            throw std::invalid_argument("ktorch.copy_from_torch_float: unsupported source torch device.");
        }
        stream.synchronize();
    } else {
        if (src_device_type == "cpu") {
            std::memcpy(dst.data_, src_ptr, dst.nbytes_);
        } else if (src_device_type == "cuda") {
            Stream stream = py_current_stream();
            CUDA_CHECK(cudaMemcpyAsync(dst.data_, src_ptr, dst.nbytes_, cudaMemcpyDeviceToHost, stream.s));
            stream.synchronize();
        } else {
            throw std::invalid_argument("ktorch.copy_from_torch_float: unsupported source torch device.");
        }
    }
}

inline void tensor_copy_from_torch_int32(Tensor& dst, const py::object& obj) {
    if (dst.is_read_only()) {
        throw std::invalid_argument("ktorch.copy_from_torch_int32: destination tensor is read-only.");
    }
    if (dst.dtype_ != DType::I32) {
        throw std::invalid_argument("ktorch: copy_from_torch_int32 only supports I32 tensors.");
    }
    py::object t = ensure_torch_tensor(obj, "ktorch.copy_from_torch_int32");
    if (!t.attr("is_contiguous")().cast<bool>()) {
        throw std::invalid_argument("ktorch.copy_from_torch_int32: input tensor must be contiguous.");
    }
    if (t.attr("numel")().cast<size_t>() != dst.numel_) {
        throw std::invalid_argument("ktorch.copy_from_torch_int32: input element count must match tensor.numel().");
    }
    const std::string src_device_type = py::str(t.attr("device").attr("type"));
    const void* src_ptr = reinterpret_cast<const void*>(t.attr("data_ptr")().cast<uintptr_t>());

    if (dst.device_ == Device::CUDA) {
        Stream stream = py_current_stream();
        if (src_device_type == "cuda") {
            CUDA_CHECK(cudaMemcpyAsync(dst.data_, src_ptr, dst.nbytes_, cudaMemcpyDeviceToDevice, stream.s));
        } else if (src_device_type == "cpu") {
            CUDA_CHECK(cudaMemcpyAsync(dst.data_, src_ptr, dst.nbytes_, cudaMemcpyHostToDevice, stream.s));
        } else {
            throw std::invalid_argument("ktorch.copy_from_torch_int32: unsupported source torch device.");
        }
        stream.synchronize();
    } else {
        if (src_device_type == "cpu") {
            std::memcpy(dst.data_, src_ptr, dst.nbytes_);
        } else if (src_device_type == "cuda") {
            Stream stream = py_current_stream();
            CUDA_CHECK(cudaMemcpyAsync(dst.data_, src_ptr, dst.nbytes_, cudaMemcpyDeviceToHost, stream.s));
            stream.synchronize();
        } else {
            throw std::invalid_argument("ktorch.copy_from_torch_int32: unsupported source torch device.");
        }
    }
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

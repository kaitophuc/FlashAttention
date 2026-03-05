#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#include <cstring>

#include "allocator.h"
#include "cu_check.h"

struct Tensor {
    void* data_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;  // Element strides (PyTorch style).
    DType dtype_;
    Device device_;
    size_t numel_;
    size_t nbytes_;

    Tensor(std::vector<int64_t> shape, DType dtype, Device device, Stream& stream) {
        // Check for valid shape and dtype.
        if (shape.empty()) {
            throw std::invalid_argument("Shape cannot be empty.");
        }
        for (auto dim : shape) {
            if (dim <= 0) {
                throw std::invalid_argument("Shape dimensions must be positive.");
            }
        }
        if (dtype == DType::F16 || dtype == DType::BF16 || dtype == DType::F32 || dtype == DType::I32 || dtype == DType::U8) {
            dtype_ = dtype;
        } else {
            throw std::invalid_argument("Unsupported dtype.");
        }
        if (device == Device::CPU || device == Device::CUDA) {
            device_ = device;
        } else {
            throw std::invalid_argument("Unsupported device.");
        }
        shape_ = std::move(shape);
        // Compute strides (C-contiguous).
        strides_.resize(shape_.size());
        numel_ = 1;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            strides_[i] = numel_;
            numel_ *= shape_[i];
        }
        nbytes_ = numel_ * dtype_size(dtype_);
        if (device_ == Device::CUDA) {
            data_ = allocate_device(nbytes_, stream);
        } else {
            data_ = allocate_host(nbytes_);
        }
    }

    ~Tensor() {
        if (data_ != nullptr) {
            if (device_ == Device::CUDA) {
                deallocate_device(data_);
            } else {
                deallocate_host(data_);
            }
        }
    }

    // Disable copying
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
    static Tensor empty(std::vector<int64_t> shape, DType dtype, Device device, Stream& stream) {
        return Tensor(std::move(shape), dtype, device, stream);
    }

    size_t numel() const {
        return numel_;
    }

    size_t nbytes() const {
        return nbytes_;
    }

    template <typename T>
    const T* data() const {
        if (!check_dtype_match<T>()) {
            throw std::runtime_error("Requested dtype does not match tensor dtype.");
        }
        return static_cast<const T*>(data_);
    }

    bool is_contiguous() const {
        // Check if the tensor is contiguous in memory.
        size_t expected_stride = 1;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            if (strides_[i] != expected_stride) {
                return false;
            }
            expected_stride *= shape_[i];
        }
        return true;
    }

    void view(std::vector<int64_t> new_shape) {
        // Reshape the tensor without changing the underlying data.
        size_t new_numel = 1;
        for (auto dim : new_shape) {
            if (dim <= 0) {
                throw std::invalid_argument("Shape dimensions must be positive.");
            }
            new_numel *= dim;
        }
        if (new_numel != numel_) {
            throw std::invalid_argument("New shape must have the same number of elements.");
        }
        shape_ = std::move(new_shape);
        // Recompute strides for the new shape (C-contiguous).
        strides_.resize(shape_.size());
        size_t stride = 1;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    }   

    Tensor clone(Device device, Stream& stream) const {
        // Implementation for cloning the tensor.
        Tensor cloned(shape_, dtype_, device, stream);
        // Copy data from this tensor to the cloned tensor.
        cudaMemcpyKind kind;
        if (device_ == Device::CUDA && device == Device::CUDA) {
            kind = cudaMemcpyDeviceToDevice;
        } else if (device_ == Device::CUDA && device == Device::CPU) {
            kind = cudaMemcpyDeviceToHost;
        } else if (device_ == Device::CPU && device == Device::CUDA) {
            kind = cudaMemcpyHostToDevice;
        } else {
            kind = cudaMemcpyHostToHost;
        }

        if (device_ == Device::CUDA || device == Device::CUDA) {
            CUDA_CHECK(cudaMemcpyAsync(cloned.data_, data_, nbytes_, kind, stream.s));
        } else {
            std::memcpy(cloned.data_, data_, nbytes_);
        }
        return cloned;
    }

    void copy_from(const Tensor& other, Stream& stream) {
        // Implementation for copying data from another tensor.
    }

private:
    template <typename T>
    bool check_dtype_match() const {
        if constexpr (std::is_same_v<T, float>) {
            return dtype_ == DType::F32;
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            return dtype_ == DType::U8;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return dtype_ == DType::I32;
        } else if constexpr (std::is_same_v<T, __half>) {
            return dtype_ == DType::F16;
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            return dtype_ == DType::BF16;
        } else {
            return false;
        }
    }

};

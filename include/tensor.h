#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <cstring>

#include "allocator.h"
#include "cu_check.h"
#include "cublass_handle.h"

struct Tensor {
    void* data_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;  // Element strides (PyTorch style).
    DType dtype_;
    Device device_;
    size_t numel_;
    size_t nbytes_;

    void checkValid(const std::vector<int64_t>& shape, DType dtype, Device device) const {
        if (shape.empty()) {
            throw std::invalid_argument("tensor.h: Shape cannot be empty.");
        }
        for (auto dim : shape) {
            if (dim <= 0) {
                throw std::invalid_argument("tensor.h: Shape dimensions must be positive.");
            }
        }
        switch (dtype) {
            case DType::F16:
            case DType::BF16:
            case DType::F32:
            case DType::I32:
            case DType::U8:
                break;
            default:
                throw std::invalid_argument("tensor.h: Unsupported dtype.");
        }
        switch (device) {
            case Device::CPU:
            case Device::CUDA:
                break;
            default:
                throw std::invalid_argument("tensor.h: Unsupported device.");
        }
    }

    Tensor(std::vector<int64_t> shape, DType dtype, Device device, Stream& stream) {
        // Check for valid shape and dtype.
        if (device != Device::CUDA) {
            throw std::invalid_argument("tensor.h: Stream argument is only valid for CUDA tensors.");
        }
        if (stream.s != cudaStream_t(0)) {
            throw std::invalid_argument("tensor.h: Stream argument should be the default stream at this phase.");
        }
        checkValid(shape, dtype, device);
        dtype_ = dtype;
        device_ = device;
        shape_ = std::move(shape);
        // Compute strides (C-contiguous).
        strides_.resize(shape_.size());
        numel_ = 1;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            strides_[i] = numel_;
            numel_ *= shape_[i];
        }
        nbytes_ = numel_ * dtype_size(dtype_);
        
        data_ = allocate_device(nbytes_, stream);
    }

    Tensor(std::vector<int64_t> shape, DType dtype, Device device) {
        // Constructor when stream is not provided. Use the current stream for CUDA tensors.
        checkValid(shape, dtype, device);
        dtype_ = dtype;
        device_ = device;
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
            Stream current(cudaStream_t(0)); //get_current_stream());
            data_ = allocate_device(nbytes_, current);
        } else {
            data_ = allocate_host(nbytes_);
        }
    }

    ~Tensor() {
        release();
    }

    Tensor(Tensor&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          shape_(std::move(other.shape_)),
          strides_(std::move(other.strides_)),
          dtype_(other.dtype_),
          device_(other.device_),
          numel_(other.numel_),
          nbytes_(other.nbytes_) {
        other.numel_ = 0;
        other.nbytes_ = 0;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            release();
            data_ = std::exchange(other.data_, nullptr);
            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
            dtype_ = other.dtype_;
            device_ = other.device_;
            numel_ = other.numel_;
            nbytes_ = other.nbytes_;
            other.numel_ = 0;
            other.nbytes_ = 0;
        }
        return *this;
    }

    // Disable copying
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    static Tensor empty(std::vector<int64_t> shape, DType dtype, Device device, Stream& stream) {
        return Tensor(std::move(shape), dtype, device, stream);
    }

    static Tensor empty(std::vector<int64_t> shape, DType dtype, Device device) {
        if (device == Device::CUDA) {
            Stream current(cudaStream_t(0)); //get_current_stream());
            return Tensor(std::move(shape), dtype, device, current);
        }
        return Tensor(std::move(shape), dtype, device);
    }

    static Tensor zeros(std::vector<int64_t> shape, DType dtype, Device device, Stream& stream) {
        if (device != Device::CUDA) {
            throw std::invalid_argument("tensor.h: Stream argument is only valid for CUDA tensors.");
        }
        if (stream.s != cudaStream_t(0)) {
            throw std::invalid_argument("tensor.h: Stream argument should be the default stream at this phase.");
        }
        Tensor out(std::move(shape), dtype, device, stream);
        out.zero_(stream);
        return out;
    }

    static Tensor zeros(std::vector<int64_t> shape, DType dtype, Device device) {
        Tensor out = empty(std::move(shape), dtype, device);
        out.zero_();
        return out;
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

    template <typename T>
    const std::vector<T> to_vector(Stream& stream) const {
        if (!check_dtype_match<T>()) {
            throw std::runtime_error("Requested dtype does not match tensor dtype.");
        }
        if (stream.s != cudaStream_t(0)) {
            throw std::invalid_argument("tensor.h: Stream argument should be the default stream at this phase.");
        }
        if (device_ != Device::CUDA) {
            throw std::invalid_argument("tensor.h: Stream argument is only valid for CUDA tensors.");
        }
        std::vector<T> host_data(numel_);
        CUDA_CHECK(cudaMemcpyAsync(host_data.data(), data_, nbytes_, cudaMemcpyDeviceToHost, stream.s));
        stream.synchronize();
        return host_data;
    }

    template <typename T>
    const std::vector<T> to_vector() const {
        if (!check_dtype_match<T>()) {
            throw std::runtime_error("Requested dtype does not match tensor dtype.");
        }
        if (device_ == Device::CUDA) {
            Stream current(cudaStream_t(0)); //get_current_stream());
            return to_vector<T>(current);
        } else {
            return std::vector<T>(data<T>(), data<T>() + numel_);
        }
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
                throw std::invalid_argument("tensor.h: Shape dimensions must be positive.");
            }
            new_numel *= dim;
        }
        if (new_numel != numel_) {
            throw std::invalid_argument("tensor.h: New shape must have the same number of elements.");
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
        if (device != Device::CUDA) {
            throw std::invalid_argument("tensor.h: Stream argument is only valid for CUDA tensors.");
        }
        if (stream.s != cudaStream_t(0)) {
            throw std::invalid_argument("tensor.h: Stream argument should be the default stream at this phase.");
        }
        Tensor cloned(shape_, dtype_, device, stream);
        // Copy data from this tensor to the cloned tensor.
        cudaMemcpyKind kind;
        if (device_ == Device::CPU) {
            kind = cudaMemcpyHostToDevice;      // CPU -> CUDA
        } else if (device_ == Device::CUDA) {
            kind = cudaMemcpyDeviceToDevice;    // CUDA -> CUDA
        } else {
            throw std::invalid_argument("tensor.h: Unsupported source device.");
        }

        CUDA_CHECK(cudaMemcpyAsync(cloned.data_, data_, nbytes_, kind, stream.s));
        return cloned;
    }

    Tensor clone(Device device) const {
        if (device_ == Device::CPU && device == Device::CPU) {
            Tensor cloned(shape_, dtype_, Device::CPU);
            std::memcpy(cloned.data_, data_, nbytes_);
            return cloned;
        }

        if (device_ == Device::CUDA && device == Device::CPU) {
            Tensor cloned(shape_, dtype_, Device::CPU);
            Stream current(cudaStream_t(0)); //get_current_stream());
            CUDA_CHECK(cudaMemcpyAsync(cloned.data_, data_, nbytes_, cudaMemcpyDeviceToHost, current.s)); //get_current_stream()));
            current.synchronize();
            return cloned;
        }

        if (device == Device::CUDA) {
            Stream current(cudaStream_t(0)); //get_current_stream());
            Tensor cloned = clone(device, current);   // existing stream overload
            current.synchronize();
            return cloned;
        }

        throw std::invalid_argument("tensor.h: Unsupported device combination for cloning.");
    }


    void zero_(Stream& stream) {
        if (stream.s != cudaStream_t(0)) {
            throw std::invalid_argument("tensor.h: Stream argument should be the default stream at this phase.");
        }
        if (device_ != Device::CUDA) {
            throw std::invalid_argument("tensor.h: Stream argument is only valid for CUDA tensors.");
        }
        CUDA_CHECK(cudaMemsetAsync(data_, 0, nbytes_, stream.s));
    }

    void zero_() {
        if (device_ == Device::CUDA) {
            CUDA_CHECK(cudaMemsetAsync(data_, 0, nbytes_, cudaStream_t(0))); //get_current_stream()));
        } else {
            std::memset(data_, 0, nbytes_);
        }
    }

    void copy_from(const Tensor& src, Stream& stream) {
        if (stream.s != cudaStream_t(0)) {
            throw std::invalid_argument("tensor.h: Stream argument should be the default stream at this phase.");
        }
        if (shape_ != src.shape_) {
            throw std::invalid_argument("tensor.h: Source and destination tensors must have the same shape.");
        }
        if (dtype_ != src.dtype_) {
            throw std::invalid_argument("tensor.h: Source and destination tensors must have the same dtype.");
        }
        if (device_ == Device::CPU && src.device_ == Device::CPU) {
            throw std::invalid_argument("tensor.h: Stream argument is only valid for CUDA tensors.");
        }
        cudaMemcpyKind kind;
        if (device_ == Device::CUDA && src.device_ == Device::CUDA) {
            kind = cudaMemcpyDeviceToDevice;
        } else if (device_ == Device::CUDA && src.device_ == Device::CPU) {
            kind = cudaMemcpyHostToDevice;
        } else if (device_ == Device::CPU && src.device_ == Device::CUDA) {
            kind = cudaMemcpyDeviceToHost;
        } else {
            throw std::invalid_argument("tensor.h: Unsupported device combination for copying.");
        }
        CUDA_CHECK(cudaMemcpyAsync(data_, src.data_, nbytes_, kind, stream.s));
        stream.synchronize();
    }

    template <typename T>
    void copy_from(const std::vector<T>& src, Stream& stream) {
        if (stream.s != cudaStream_t(0)) {
            throw std::invalid_argument("tensor.h: Stream argument should be the default stream at this phase.");
        }
        if (!check_dtype_match<T>()) {
            throw std::runtime_error("Requested dtype does not match tensor dtype.");
        }
        if (src.size() != numel_) {
            throw std::invalid_argument("tensor.h: Source vector size must match the number of elements in the tensor.");
        }
        if (device_ == Device::CPU) {
            throw std::invalid_argument("tensor.h: Stream argument is only valid for CUDA tensors.");
        }
        CUDA_CHECK(cudaMemcpyAsync(data_, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice, stream.s));
        stream.synchronize();
    }

    template <typename T>
    void copy_from(const std::vector<T>& src) {
        if (!check_dtype_match<T>()) {
            throw std::runtime_error("Requested dtype does not match tensor dtype.");
        }
        if (src.size() != numel_) {
            throw std::invalid_argument("tensor.h: Source vector size must match the number of elements in the tensor.");
        }
        if (device_ == Device::CUDA) {
            CUDA_CHECK(cudaMemcpyAsync(data_, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice, cudaStream_t(0))); //get_current_stream()));
        } else {
            std::memcpy(data_, src.data(), src.size() * sizeof(T));
        }
    }

    Tensor matmul(const Tensor& other) const;
    Tensor matmul(const Tensor& other, Stream& stream, CublasHandle& cublas_handle) const;

private:
    void release() noexcept {
        if (data_ != nullptr) {
            if (device_ == Device::CUDA) {
                deallocate_device(data_);
            } else {
                deallocate_host(data_);
            }
        }
        data_ = nullptr;
    }

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

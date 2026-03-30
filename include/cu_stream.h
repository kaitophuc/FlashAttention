#pragma once

#include <cuda_runtime.h>

#include <stdexcept>

#include "cu_check.h"

constexpr int kStreamPoolSize = 32;

inline void assert_non_default_stream(cudaStream_t stream, const char* where) {
    if (stream == cudaStream_t(0)) {
        throw std::invalid_argument(std::string(where) +
                                    ": default stream is banned. Use current_stream()/StreamGuard or pooled streams.");
    }
}

cudaStream_t current_stream_raw(int device = -1);
void set_current_stream_raw(cudaStream_t stream, int device = -1);
cudaStream_t stream_from_pool_raw(int idx, int device = -1);
cudaStream_t next_stream_raw(int device = -1);
int stream_pool_size(int device = -1);

struct Stream {
    cudaStream_t s;
    bool owns_;

    Stream() : s(current_stream_raw()), owns_(false) {}

    explicit Stream(cudaStream_t stream, bool validate_non_default = false)
        : s(stream), owns_(false) {
        if (validate_non_default) {
            assert_non_default_stream(s, "cu_stream.h: Stream");
        }
    }

    ~Stream() {
        if (owns_ && s != nullptr) {
            cudaStreamDestroy(s);
        }
    }

    void synchronize() const {
        assert_non_default_stream(s, "cu_stream.h: Stream::synchronize");
        CUDA_CHECK(cudaStreamSynchronize(s));
    }

    Stream(const Stream&) = default;
    Stream& operator=(const Stream&) = default;
};

inline Stream current_stream(int device = -1) {
    return Stream(current_stream_raw(device), false);
}

inline void set_current_stream(const Stream& stream, int device = -1) {
    set_current_stream_raw(stream.s, device);
}

inline Stream stream_from_pool(int idx, int device = -1) {
    return Stream(stream_from_pool_raw(idx, device), false);
}

inline Stream next_stream(int device = -1) {
    return Stream(next_stream_raw(device), false);
}

struct StreamGuard {
    int device_;
    Stream prev_stream_;

    explicit StreamGuard(const Stream& stream, int device = -1)
        : device_(device), prev_stream_(current_stream(device)) {
        set_current_stream(stream, device_);
    }

    explicit StreamGuard(cudaStream_t stream, int device = -1)
        : device_(device), prev_stream_(current_stream(device)) {
        set_current_stream(Stream(stream, true), device_);
    }

    ~StreamGuard() {
        set_current_stream(prev_stream_, device_);
    }

    StreamGuard(const StreamGuard&) = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;
};

struct Event {
    cudaEvent_t e;
    Event() {
        CUDA_CHECK(cudaEventCreate(&e));
    }

    ~Event() {
        cudaEventDestroy(e);
    }

    void synchronize() {
        CUDA_CHECK(cudaEventSynchronize(e));
    }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

inline void record(Event& event, const Stream& stream) {
    assert_non_default_stream(stream.s, "cu_stream.h: record");
    CUDA_CHECK(cudaEventRecord(event.e, stream.s));
}

inline void wait(const Stream& stream, Event& event) {
    assert_non_default_stream(stream.s, "cu_stream.h: wait");
    CUDA_CHECK(cudaStreamWaitEvent(stream.s, event.e, 0));
}

inline float elapsed_time(Event& start, Event& end) {
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start.e, end.e));
    return ms;
}

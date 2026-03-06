#pragma once

#include <cuda_runtime.h>
#include "cu_check.h"

// A simple wrapper around cudaStream_t for RAII management.
struct Stream {
    cudaStream_t s;
    bool owns_;

    Stream () : owns_(true) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    }

    explicit Stream(cudaStream_t stream) : s(stream), owns_(false) {}

    ~Stream () {
        if (owns_ && s != nullptr) {
            cudaStreamDestroy(s);
        }
    }

    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(s));
    }

    // Prevent copying
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

struct Event {
    cudaEvent_t e;
    Event () {
        CUDA_CHECK(cudaEventCreate(&e));
    }
    
    ~Event () {
        cudaEventDestroy(e);
    }

    void synchronize() {
        CUDA_CHECK(cudaEventSynchronize(e));
    }

    // Prevent copying
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

inline void record(Event& event, Stream& stream) {
    CUDA_CHECK(cudaEventRecord(event.e, stream.s));
}

inline void wait(Stream& stream, Event& event) {
    CUDA_CHECK(cudaStreamWaitEvent(stream.s, event.e, 0));
}

inline float elapsed_time(Event& start, Event& end) {
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start.e, end.e));
    return ms;
}

cudaStream_t get_current_stream(int device = -1);
void set_current_stream(cudaStream_t stream, int device = -1);

struct StreamGuard {
    cudaStream_t stream_;
    cudaStream_t prev_stream_;
    int device_;

    StreamGuard(Stream& stream, int device = -1) : stream_(stream.s), device_(device) {
        // Save the previous stream and set the new stream.
        prev_stream_ = get_current_stream(device_);
        set_current_stream(stream_, device_);
    }

    StreamGuard(cudaStream_t stream, int device = -1) : stream_(stream), device_(device) {
        // Save the previous stream and set the new stream.
        prev_stream_ = get_current_stream(device_);
        set_current_stream(stream_, device_);
    }

    ~StreamGuard() {
        // Reset to the previous stream.
        set_current_stream(prev_stream_, device_);
    }

    // Prevent copying
    StreamGuard(const StreamGuard&) = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;
};

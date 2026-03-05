#include "cu_stream.h"
#include <map>

// Thread-local storage for current streams per device
namespace {
    thread_local std::map<int, cudaStream_t> current_streams;
}

cudaStream_t get_current_stream(int device) {
    if (device == -1) {
        CUDA_CHECK(cudaGetDevice(&device));
    }
    
    auto it = current_streams.find(device);
    if (it != current_streams.end()) {
        return it->second;
    }
    
    // Default to stream 0 (default stream)
    return cudaStream_t(0);
}

void set_current_stream(cudaStream_t stream, int device) {
    if (device == -1) {
        CUDA_CHECK(cudaGetDevice(&device));
    }
    
    current_streams[device] = stream;
}
#include "cu_stream.h"

#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

struct DevicePool {
    std::vector<cudaStream_t> streams;
    size_t next_lane = 0;
};

std::mutex g_pool_mutex;
std::unordered_map<int, DevicePool> g_device_pools;

thread_local std::unordered_map<int, cudaStream_t> tls_current_streams;

int resolve_device(int device) {
    if (device >= 0) {
        return device;
    }
    int current = -1;
    CUDA_CHECK(cudaGetDevice(&current));
    return current;
}

DevicePool& get_or_create_pool_locked(int device) {
    auto it = g_device_pools.find(device);
    if (it != g_device_pools.end()) {
        return it->second;
    }

    DevicePool pool;
    pool.streams.reserve(kStreamPoolSize);
    for (int i = 0; i < kStreamPoolSize; ++i) {
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        pool.streams.push_back(stream);
    }

    auto [inserted_it, inserted] = g_device_pools.emplace(device, std::move(pool));
    (void)inserted;
    return inserted_it->second;
}

cudaStream_t get_thread_initial_stream(int device) {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    DevicePool& pool = get_or_create_pool_locked(device);
    const size_t lane = pool.next_lane % static_cast<size_t>(kStreamPoolSize);
    pool.next_lane += 1;
    return pool.streams[lane];
}

}  // namespace

cudaStream_t current_stream_raw(int device) {
    const int dev = resolve_device(device);
    auto it = tls_current_streams.find(dev);
    if (it != tls_current_streams.end()) {
        return it->second;
    }

    cudaStream_t stream = get_thread_initial_stream(dev);
    tls_current_streams.emplace(dev, stream);
    return stream;
}

void set_current_stream_raw(cudaStream_t stream, int device) {
    assert_non_default_stream(stream, "cu_stream.cu: set_current_stream_raw");
    const int dev = resolve_device(device);
    tls_current_streams[dev] = stream;
}

cudaStream_t stream_from_pool_raw(int idx, int device) {
    const int dev = resolve_device(device);
    if (idx < 0 || idx >= kStreamPoolSize) {
        throw std::out_of_range("cu_stream.cu: stream_from_pool_raw index out of range.");
    }

    std::lock_guard<std::mutex> lock(g_pool_mutex);
    DevicePool& pool = get_or_create_pool_locked(dev);
    return pool.streams[static_cast<size_t>(idx)];
}

cudaStream_t next_stream_raw(int device) {
    const int dev = resolve_device(device);
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    DevicePool& pool = get_or_create_pool_locked(dev);
    const size_t lane = pool.next_lane % static_cast<size_t>(kStreamPoolSize);
    pool.next_lane += 1;
    return pool.streams[lane];
}

int stream_pool_size(int device) {
    const int dev = resolve_device(device);
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    (void)get_or_create_pool_locked(dev);
    return kStreamPoolSize;
}

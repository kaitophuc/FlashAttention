#include "cu_stream.h"
#include "general.h"
#include "tensor.h"

#include <gtest/gtest.h>

#include <future>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

__global__ void fill_constant_kernel(float* out, int n, float value) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

__global__ void add_constant_kernel(float* out, int n, float value) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] += value;
    }
}

TEST(StreamRuntime, CurrentStreamIsAlwaysNonDefault) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream s = current_stream();
    EXPECT_NE(s.s, cudaStream_t(0));
}

TEST(StreamRuntime, SetCurrentStreamRejectsDefaultStream) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream default_stream(cudaStream_t(0));
    EXPECT_THROW(set_current_stream(default_stream), std::invalid_argument);
}

TEST(StreamRuntime, StreamGuardRestoresPreviousStream) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream original = current_stream();
    Stream other = next_stream();
    EXPECT_NE(other.s, cudaStream_t(0));

    {
        StreamGuard guard(other);
        Stream active = current_stream();
        EXPECT_EQ(active.s, other.s);
    }

    Stream restored = current_stream();
    EXPECT_EQ(restored.s, original.s);
}

TEST(StreamRuntime, TensorCudaConstructorRejectsDefaultStream) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream default_stream(cudaStream_t(0));
    EXPECT_THROW((void)Tensor({8}, DType::F32, Device::CUDA, default_stream), std::invalid_argument);
}

TEST(MultiStream, StreamFromPoolBoundsAndUniqueness) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    const int pool_n = stream_pool_size();
    EXPECT_EQ(pool_n, kStreamPoolSize);

    std::unordered_set<cudaStream_t> unique;
    for (int i = 0; i < pool_n; ++i) {
        Stream s = stream_from_pool(i);
        EXPECT_NE(s.s, cudaStream_t(0));
        unique.insert(s.s);
    }
    EXPECT_EQ(static_cast<int>(unique.size()), pool_n);

    EXPECT_THROW((void)stream_from_pool(-1), std::out_of_range);
    EXPECT_THROW((void)stream_from_pool(pool_n), std::out_of_range);
}

TEST(MultiStream, NextStreamRotationYieldsPoolMembersAndWraps) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    const int pool_n = stream_pool_size();
    std::unordered_set<cudaStream_t> pool_members;
    for (int i = 0; i < pool_n; ++i) {
        pool_members.insert(stream_from_pool(i).s);
    }

    std::unordered_set<cudaStream_t> observed;
    for (int i = 0; i < pool_n * 2; ++i) {
        Stream s = next_stream();
        EXPECT_NE(s.s, cudaStream_t(0));
        EXPECT_TRUE(pool_members.count(s.s) > 0);
        observed.insert(s.s);
    }

    EXPECT_EQ(static_cast<int>(observed.size()), pool_n);
}

TEST(MultiStream, CurrentStreamIsThreadLocalPerDevice) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream main_stream = stream_from_pool(0);
    set_current_stream(main_stream);
    EXPECT_EQ(current_stream().s, main_stream.s);

    std::promise<cudaStream_t> worker_promise;
    std::future<cudaStream_t> worker_future = worker_promise.get_future();

    std::thread worker([&worker_promise]() {
        Stream worker_stream = stream_from_pool(1);
        set_current_stream(worker_stream);
        worker_promise.set_value(current_stream().s);
    });
    worker.join();

    const cudaStream_t worker_value = worker_future.get();
    EXPECT_EQ(current_stream().s, main_stream.s);
    EXPECT_NE(worker_value, main_stream.s);
}

TEST(MultiStream, EventWaitEstablishesCrossStreamOrdering) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream producer = stream_from_pool(2);
    Stream consumer = stream_from_pool(3);
    Tensor t({128}, DType::F32, Device::CUDA, producer);

    constexpr int kN = 128;
    constexpr int kBlock = 128;
    fill_constant_kernel<<<1, kBlock, 0, producer.s>>>(static_cast<float*>(t.data_), kN, 3.0f);

    Event ready;
    record(ready, producer);
    wait(consumer, ready);

    add_constant_kernel<<<1, kBlock, 0, consumer.s>>>(static_cast<float*>(t.data_), kN, 2.0f);
    consumer.synchronize();

    const std::vector<float> got = t.to_vector<float>(consumer);
    ASSERT_EQ(got.size(), static_cast<size_t>(kN));
    for (float v : got) {
        EXPECT_FLOAT_EQ(v, 5.0f);
    }
}

TEST(MultiStreamStress, StreamGuardRepeatedHandoffsRestoreOriginalStream) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    const int pool_n = stream_pool_size();
    Stream original = current_stream();

    for (int i = 0; i < 1000; ++i) {
        Stream a = stream_from_pool(i % pool_n);
        Stream b = stream_from_pool((i + 7) % pool_n);

        {
            StreamGuard g1(a);
            EXPECT_EQ(current_stream().s, a.s);
            {
                StreamGuard g2(b);
                EXPECT_EQ(current_stream().s, b.s);
            }
            EXPECT_EQ(current_stream().s, a.s);
        }

        EXPECT_EQ(current_stream().s, original.s);
    }
}

}  // namespace

#include "cu_stream.h"
#include "general.h"
#include "tensor.h"

#include <gtest/gtest.h>

namespace {

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

}  // namespace

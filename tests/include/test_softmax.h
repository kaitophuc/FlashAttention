#pragma once

#include "ops.h"
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace fa_test {

constexpr uint32_t kSoftmaxSeedBase = 29221u;

inline uint32_t MixSeed(uint32_t base, int a, int b = 0, int c = 0, int d = 0) {
    uint32_t h = base ^ 0x9e3779b9u;
    const uint32_t vals[4] = {
        static_cast<uint32_t>(a),
        static_cast<uint32_t>(b),
        static_cast<uint32_t>(c),
        static_cast<uint32_t>(d),
    };
    for (uint32_t v : vals) {
        h ^= v + 0x9e3779b9u + (h << 6u) + (h >> 2u);
    }
    return h;
}

inline int GetEnvInt(const char* name, int default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return default_value;
    }
    const long v = std::strtol(raw, nullptr, 10);
    return static_cast<int>(v);
}

inline bool LongStressEnabled() {
    return GetEnvInt("FA_ENABLE_LONG_STRESS", 0) == 1;
}

enum class Dist { Uniform, NearZero, MixedLarge };

inline const char* DistName(Dist d) {
    switch (d) {
        case Dist::Uniform: return "uniform";
        case Dist::NearZero: return "nearzero";
        case Dist::MixedLarge: return "mixedlarge";
    }
    return "unknown";
}

struct SoftmaxCase {
    int m, n;
    Dist dist;
    float lo, hi;
    float abs_tol;
    float rel_tol;
    int iters;
    const char* name;
};

struct SoftmaxEdgeCase {
    const char* name;
    int m;
    int n;
    float abs_tol;
    float rel_tol;
    std::vector<float> x;
};

inline std::vector<float> SampleUniformVector(size_t n, float lo, float hi, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> out(n);
    for (float& v : out) {
        v = dist(rng);
    }
    return out;
}

inline std::vector<SoftmaxCase> BuildForwardCases() {
    const std::vector<std::tuple<int, int, const char*>> shapes = {
        {1, 1, "tiny_11"},
        {1, 7, "tiny_17"},
        {5, 1, "tiny_51"},
        {13, 37, "odd_1337"},
        {37, 53, "prime_3753"},
        {64, 96, "mid_6496"},
        {128, 256, "big_128256"},
        {200, 300, "med_200300"},
        {512, 768, "large_512768"},
    };

    std::vector<SoftmaxCase> out;
    for (const auto& s : shapes) {
        const int m = std::get<0>(s);
        const int n = std::get<1>(s);
        const char* name = std::get<2>(s);
        const long long volume = static_cast<long long>(m) * n;

        int iters = 4;
        if (volume < 1000000LL) {
            iters = 10;
        } else if (volume < 100000000LL) {
            iters = 5;
        }

        out.push_back({m, n, Dist::Uniform, -1.0f, 1.0f, 2e-5f, 2e-5f, iters, name});
        out.push_back({m, n, Dist::NearZero, -1e-6f, 1e-6f, 2e-5f, 2e-5f, iters, name});
        out.push_back({m, n, Dist::MixedLarge, -100.0f, 100.0f, 5e-5f, 5e-5f, iters, name});
    }
    return out;
}

inline std::vector<SoftmaxCase> BuildBackwardCases() {
    return BuildForwardCases();
}

inline std::vector<SoftmaxCase> BuildForwardBackwardCases() {
    const std::vector<std::tuple<int, int, const char*>> shapes = {
        {3, 4, "tiny_34"},
        {13, 37, "odd_1337"},
        {64, 96, "mid_6496"},
        {128, 256, "big_128256"},
        {200, 300, "med_200300"},
    };

    std::vector<SoftmaxCase> out;
    for (const auto& s : shapes) {
        const int m = std::get<0>(s);
        const int n = std::get<1>(s);
        const char* name = std::get<2>(s);
        const long long volume = static_cast<long long>(m) * n;

        int iters = 3;
        if (volume < 1000000LL) {
            iters = 6;
        } else if (volume < 100000000LL) {
            iters = 4;
        }

        out.push_back({m, n, Dist::Uniform, -1.0f, 1.0f, 3e-5f, 3e-5f, iters, name});
        out.push_back({m, n, Dist::NearZero, -1e-6f, 1e-6f, 3e-5f, 3e-5f, iters, name});
        out.push_back({m, n, Dist::MixedLarge, -100.0f, 100.0f, 7e-5f, 7e-5f, iters, name});
    }
    return out;
}

inline std::vector<float> reference_softmax_forward(const std::vector<float>& x, int m, int n) {
    if (static_cast<int>(x.size()) != m * n) {
        throw std::invalid_argument("test_softmax.h: reference_softmax_forward: invalid input size");
    }

    std::vector<float> y(static_cast<size_t>(m) * n, 0.0f);
    for (int row = 0; row < m; ++row) {
        const size_t offset = static_cast<size_t>(row) * n;

        float row_max = x[offset];
        for (int col = 1; col < n; ++col) {
            row_max = std::max(row_max, x[offset + static_cast<size_t>(col)]);
        }

        double row_sum = 0.0;
        for (int col = 0; col < n; ++col) {
            const double ex = std::exp(static_cast<double>(x[offset + static_cast<size_t>(col)] - row_max));
            y[offset + static_cast<size_t>(col)] = static_cast<float>(ex);
            row_sum += ex;
        }

        const double inv = 1.0 / row_sum;
        for (int col = 0; col < n; ++col) {
            y[offset + static_cast<size_t>(col)] =
                static_cast<float>(static_cast<double>(y[offset + static_cast<size_t>(col)]) * inv);
        }
    }

    return y;
}

inline std::vector<float> reference_softmax_backward(const std::vector<float>& dY,
                                                     const std::vector<float>& y,
                                                     int m,
                                                     int n) {
    if (static_cast<int>(dY.size()) != m * n || static_cast<int>(y.size()) != m * n) {
        throw std::invalid_argument("test_softmax.h: reference_softmax_backward: invalid input size");
    }

    std::vector<float> dX(static_cast<size_t>(m) * n, 0.0f);
    for (int row = 0; row < m; ++row) {
        const size_t offset = static_cast<size_t>(row) * n;

        double dot = 0.0;
        for (int col = 0; col < n; ++col) {
            const size_t idx = offset + static_cast<size_t>(col);
            dot += static_cast<double>(y[idx]) * static_cast<double>(dY[idx]);
        }

        for (int col = 0; col < n; ++col) {
            const size_t idx = offset + static_cast<size_t>(col);
            dX[idx] = static_cast<float>(static_cast<double>(y[idx]) * (static_cast<double>(dY[idx]) - dot));
        }
    }

    return dX;
}

}  // namespace fa_test

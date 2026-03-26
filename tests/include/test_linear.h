#pragma once

#include "ops.h"
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace fa_test {

constexpr uint32_t kLinearSeedBase = 14406u;

inline uint32_t MixSeed(uint32_t base, int a, int b = 0, int c = 0, int d = 0) {
    // Integer hash mix for stable cross-platform testcase seeds.
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

inline bool DeterministicBool(uint32_t seed) {
    return (seed & 1u) == 0u;
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

inline std::string ShapeString(int m, int n, int k) {
    std::ostringstream os;
    os << "m=" << m << " n=" << n << " k=" << k;
    return os.str();
}

enum class Dist { Uniform, NearZero, MixedLarge };

struct LinearCase {
    int m, n, k;
    bool with_bias;
    Dist dist;
    float lo, hi;
    float abs_tol;
    float rel_tol;
    int iters;
    const char* name;
};

inline const char* DistName(Dist d) {
    switch (d) {
        case Dist::Uniform: return "uniform";
        case Dist::NearZero: return "nearzero";
        case Dist::MixedLarge: return "mixedlarge";
    }
    return "unknown";
}

inline std::vector<LinearCase> BuildCases() {
    const std::vector<std::tuple<int, int, int, const char*>> shapes = {
        {1, 1, 1, "tiny_111"},
        {1, 7, 3, "tiny_173"},
        {5, 1, 9, "tiny_519"},
        {37, 53, 29, "prime_375329"},
        {200, 300, 400, "med_200300400"},
        {512, 768, 1024, "large_5127681024"},
    };

    std::vector<LinearCase> out;
    for (const auto& s : shapes) {
        const int m = std::get<0>(s);
        const int n = std::get<1>(s);
        const int k = std::get<2>(s);
        const char* name = std::get<3>(s);

        const int iters = (m * n * k < 1000000) ? 10 : 3;
        for (bool with_bias : {false, true}) {
            out.push_back({m, n, k, with_bias, Dist::Uniform, -1.0f, 1.0f, 1e-4f, 1e-4f, iters, name});
            out.push_back({m, n, k, with_bias, Dist::NearZero, -1e-6f, 1e-6f, 1e-7f, 1e-4f, iters, name});
            out.push_back({m, n, k, with_bias, Dist::MixedLarge, -100.0f, 100.0f, 1.5e-1f, 1e-3f, iters, name});
        }
    }
    return out;
}

template <typename T>
struct HostLinearInputs {
    int m, n, k;
    bool with_bias;

    Tensor x_h;
    Tensor w_h;
    std::optional<Tensor> b_h;

    std::vector<T> x_ref;
    std::vector<T> w_ref;
    std::vector<T> b_ref;

    std::mt19937 rng;

    HostLinearInputs(int m_, int n_, int k_, bool with_bias_ = false,
                     T start = static_cast<T>(-1), T end = static_cast<T>(1),
                     uint32_t seed = 14406u)
        : m(m_),
          n(n_),
          k(k_),
          with_bias(with_bias_),
          x_h({m_, k_}, get_dtype<T>(), Device::CPU),
          w_h({n_, k_}, get_dtype<T>(), Device::CPU),
          rng(seed) {
        std::uniform_real_distribution<T> dist(start, end);

        x_ref.resize(static_cast<size_t>(m) * k);
        w_ref.resize(static_cast<size_t>(n) * k);
        for (auto& v : x_ref) v = dist(rng);
        for (auto& v : w_ref) v = dist(rng);

        x_h.copy_from(x_ref);
        w_h.copy_from(w_ref);

        if (with_bias) {
            b_ref.resize(static_cast<size_t>(n));
            for (auto& v : b_ref) v = dist(rng);
            b_h.emplace(std::vector<int64_t>{n}, get_dtype<T>(), Device::CPU);
            b_h->copy_from(b_ref);
        }
    }
};

struct QueuedLinearJob {
    int m, n, k;
    float abs_tol, rel_tol;

    Tensor x_d;
    Tensor w_d;
    std::optional<Tensor> b_d;
    Tensor y_h;
    std::vector<float> expected;
};

struct ShapeCfg {
    int m, n, k;
    float lo, hi;
    float abs_tol, rel_tol;
    bool with_bias;
};

inline std::vector<float> reference_linear(const std::vector<float>& x,
                                           const std::vector<float>& w,
                                           const std::vector<float>* b,
                                           int m,
                                           int k,
                                           int n);

inline std::vector<float> reference_linear(const std::vector<float>& x,
                                           const std::vector<float>& w,
                                           const std::vector<float>* b,
                                           int m,
                                           int k,
                                           int n) {
    std::vector<float> y(static_cast<size_t>(m) * static_cast<size_t>(n), 0.0f);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int t = 0; t < k; ++t) {
                acc += static_cast<double>(x[static_cast<size_t>(i) * k + t]) *
                       static_cast<double>(w[static_cast<size_t>(j) * k + t]);
            }
            if (b != nullptr) {
                acc += static_cast<double>((*b)[j]);
            }
            y[static_cast<size_t>(i) * n + j] = static_cast<float>(acc);
        }
    }
    return y;
}

static void ValidateQueuedJobs(const std::vector<QueuedLinearJob>& jobs) {
    std::vector<std::string> failures;
    for (size_t job_idx = 0; job_idx < jobs.size(); ++job_idx) {
        const auto& j = jobs[job_idx];
        auto* y_ptr = static_cast<const float*>(j.y_h.data_);

        int fail_count = 0;
        float worst_abs_error = 0.0f;
        float worst_tol = 0.0f;
        int worst_i = -1;

        const int total = j.m * j.n;
        for (int i = 0; i < total; ++i) {
            const float e = j.expected[i];
            const float got = y_ptr[i];
            const float abs_error = std::fabs(got - e);
            const float tol = j.abs_tol + j.rel_tol * std::fabs(e);
            if (abs_error > tol) {
                ++fail_count;
                if (abs_error > worst_abs_error) {
                    worst_abs_error = abs_error;
                    worst_tol = tol;
                    worst_i = i;
                }
            }
        }

        if (fail_count > 0) {
            std::ostringstream os;
            os << "Job " << job_idx << " failed: " << fail_count << "/" << total
               << " elements exceeded tolerance. Worst error at index " << worst_i
               << ": got " << y_ptr[worst_i] << ", expected " << j.expected[worst_i]
               << ", abs error " << worst_abs_error << ", tol " << worst_tol;
            failures.push_back(os.str());
        }
    }

    if (!failures.empty()) {
        std::ostringstream all;
        all << "Queued linear validation failed in " << failures.size() << " job(s):\n";
        for (const auto& f : failures) all << "  " << f << "\n";
        ADD_FAILURE() << all.str();
    }
}

}  // namespace fa_test

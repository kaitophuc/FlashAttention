#pragma once

#include "ops.h"
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace fa_test {

constexpr uint32_t kReluSeedBase = 18577u;

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

enum class Dist { Uniform, NearZero, MixedLarge };

inline const char* DistName(Dist d) {
    switch (d) {
        case Dist::Uniform: return "uniform";
        case Dist::NearZero: return "nearzero";
        case Dist::MixedLarge: return "mixedlarge";
    }
    return "unknown";
}

struct ReluCase {
    int m, n;
    Dist dist;
    float lo, hi;
    float abs_tol;
    float rel_tol;
    int iters;
    const char* name;
};

struct ForwardEdgeCase {
    const char* name;
    int m;
    int n;
    float abs_tol;
    float rel_tol;
    std::vector<float> x;
};

struct ReluShapeCfg {
    int m, n;
    float lo, hi;
    float abs_tol;
    float rel_tol;
};

struct QueuedReluForwardJob {
    int m, n;
    float abs_tol, rel_tol;
    Tensor x_d;
    Tensor y_h;
    std::vector<float> expected;
};

struct QueuedReluBackwardJob {
    int m, n;
    float abs_tol, rel_tol;
    Tensor x_d;
    Tensor dY_d;
    Tensor dX_h;
    std::vector<float> expected;
};

inline std::vector<ReluCase> BuildForwardCases() {
    const std::vector<std::tuple<int, int, const char*>> shapes = {
        {1, 1, "tiny_11"},
        {1, 7, "tiny_17"},
        {5, 1, "tiny_51"},
        {37, 53, "prime_3753"},
        {64, 96, "mid_6496"},
        {128, 256, "big_128256"},
        {200, 300, "med_200300"},
        {512, 768, "large_512768"},
        {768, 1024, "xlarge_7681024"},
    };

    std::vector<ReluCase> out;
    for (const auto& s : shapes) {
        const int m = std::get<0>(s);
        const int n = std::get<1>(s);
        const char* name = std::get<2>(s);

        const long long volume = static_cast<long long>(m) * n;
        int iters = 4;
        if (volume < 1000000LL) {
            iters = 12;
        } else if (volume < 100000000LL) {
            iters = 6;
        }

        out.push_back({m, n, Dist::Uniform, -1.0f, 1.0f, 1e-6f, 1e-6f, iters, name});
        out.push_back({m, n, Dist::NearZero, -1e-6f, 1e-6f, 1e-7f, 1e-6f, iters, name});
        out.push_back({m, n, Dist::MixedLarge, -100.0f, 100.0f, 1e-5f, 1e-6f, iters, name});
    }
    return out;
}

inline std::vector<ReluCase> BuildBackwardCases() {
    return BuildForwardCases();
}

inline std::vector<ReluCase> BuildForwardBackwardCases() {
    const std::vector<std::tuple<int, int, const char*>> shapes = {
        {3, 4, "tiny_34"},
        {13, 37, "odd_1337"},
        {64, 96, "mid_6496"},
        {128, 256, "big_128256"},
        {200, 300, "med_200300"},
        {512, 768, "large_512768"},
    };

    std::vector<ReluCase> out;
    for (const auto& s : shapes) {
        const int m = std::get<0>(s);
        const int n = std::get<1>(s);
        const char* name = std::get<2>(s);

        const long long volume = static_cast<long long>(m) * n;
        int iters = 3;
        if (volume < 1000000LL) {
            iters = 7;
        } else if (volume < 100000000LL) {
            iters = 4;
        }

        out.push_back({m, n, Dist::Uniform, -1.0f, 1.0f, 1e-6f, 1e-6f, iters, name});
        out.push_back({m, n, Dist::NearZero, -1e-6f, 1e-6f, 1e-7f, 1e-6f, iters, name});
        out.push_back({m, n, Dist::MixedLarge, -100.0f, 100.0f, 1e-5f, 1e-6f, iters, name});
    }
    return out;
}

inline std::vector<float> SampleUniformVector(size_t n, float lo, float hi, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> out(n);
    for (float& v : out) {
        v = dist(rng);
    }
    return out;
}

inline std::vector<float> reference_relu_forward(const std::vector<float>& x) {
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
    return y;
}

inline std::vector<float> reference_relu_backward(const std::vector<float>& dY,
                                                  const std::vector<float>& x) {
    if (dY.size() != x.size()) {
        throw std::invalid_argument("test_relu.h: reference_relu_backward requires dY and x to have same size");
    }
    std::vector<float> dX(dY.size());
    for (size_t i = 0; i < dY.size(); ++i) {
        dX[i] = x[i] > 0.0f ? dY[i] : 0.0f;
    }
    return dX;
}

inline void ValidateQueuedForwardJobs(const std::vector<QueuedReluForwardJob>& jobs) {
    std::vector<std::string> failures;
    for (size_t job_idx = 0; job_idx < jobs.size(); ++job_idx) {
        const auto& j = jobs[job_idx];
        const std::vector<float> got = j.y_h.to_vector<float>();
        if (got.size() != j.expected.size()) {
            std::ostringstream os;
            os << "Forward job " << job_idx << " size mismatch got=" << got.size()
               << " expected=" << j.expected.size();
            failures.push_back(os.str());
            continue;
        }

        int fail_count = 0;
        float worst_abs_error = 0.0f;
        float worst_tol = 0.0f;
        int worst_i = -1;

        for (size_t i = 0; i < got.size(); ++i) {
            const float e = j.expected[i];
            const float abs_error = std::fabs(got[i] - e);
            const float tol = j.abs_tol + j.rel_tol * std::fabs(e);
            if (abs_error > tol) {
                ++fail_count;
                if (abs_error > worst_abs_error) {
                    worst_abs_error = abs_error;
                    worst_tol = tol;
                    worst_i = static_cast<int>(i);
                }
            }
        }

        if (fail_count > 0) {
            std::ostringstream os;
            os << "Forward job " << job_idx << " failed: " << fail_count << "/" << got.size()
               << " elements exceeded tolerance. Worst idx=" << worst_i
               << " got=" << got[static_cast<size_t>(worst_i)]
               << " expected=" << j.expected[static_cast<size_t>(worst_i)]
               << " abs error=" << worst_abs_error
               << " tol=" << worst_tol;
            failures.push_back(os.str());
        }
    }

    if (!failures.empty()) {
        std::ostringstream all;
        all << "Queued ReLU forward validation failed in " << failures.size() << " job(s):\n";
        for (const auto& f : failures) {
            all << "  " << f << "\n";
        }
        ADD_FAILURE() << all.str();
    }
}

inline void ValidateQueuedBackwardJobs(const std::vector<QueuedReluBackwardJob>& jobs) {
    std::vector<std::string> failures;
    for (size_t job_idx = 0; job_idx < jobs.size(); ++job_idx) {
        const auto& j = jobs[job_idx];
        const std::vector<float> got = j.dX_h.to_vector<float>();
        if (got.size() != j.expected.size()) {
            std::ostringstream os;
            os << "Backward job " << job_idx << " size mismatch got=" << got.size()
               << " expected=" << j.expected.size();
            failures.push_back(os.str());
            continue;
        }

        int fail_count = 0;
        float worst_abs_error = 0.0f;
        float worst_tol = 0.0f;
        int worst_i = -1;

        for (size_t i = 0; i < got.size(); ++i) {
            const float e = j.expected[i];
            const float abs_error = std::fabs(got[i] - e);
            const float tol = j.abs_tol + j.rel_tol * std::fabs(e);
            if (abs_error > tol) {
                ++fail_count;
                if (abs_error > worst_abs_error) {
                    worst_abs_error = abs_error;
                    worst_tol = tol;
                    worst_i = static_cast<int>(i);
                }
            }
        }

        if (fail_count > 0) {
            std::ostringstream os;
            os << "Backward job " << job_idx << " failed: " << fail_count << "/" << got.size()
               << " elements exceeded tolerance. Worst idx=" << worst_i
               << " got=" << got[static_cast<size_t>(worst_i)]
               << " expected=" << j.expected[static_cast<size_t>(worst_i)]
               << " abs error=" << worst_abs_error
               << " tol=" << worst_tol;
            failures.push_back(os.str());
        }
    }

    if (!failures.empty()) {
        std::ostringstream all;
        all << "Queued ReLU backward validation failed in " << failures.size() << " job(s):\n";
        for (const auto& f : failures) {
            all << "  " << f << "\n";
        }
        ADD_FAILURE() << all.str();
    }
}

}  // namespace fa_test

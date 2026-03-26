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

struct LinearBackwardCase {
    int m, n, k;
    bool with_bias;
    Dist dist;
    float lo, hi;
    float abs_tol;
    float rel_tol;
    int iters;
    const char* name;
};

struct LinearForwardBackwardCase {
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
        {64, 96, 80, "mid_649680"},
        {128, 256, 192, "big_128256192"},
        {200, 300, 400, "med_200300400"},
        {512, 768, 1024, "large_5127681024"},
        {768, 1024, 512, "xlarge_7681024512"},
    };

    std::vector<LinearCase> out;
    for (const auto& s : shapes) {
        const int m = std::get<0>(s);
        const int n = std::get<1>(s);
        const int k = std::get<2>(s);
        const char* name = std::get<3>(s);

        const long long volume = static_cast<long long>(m) * n * k;
        int iters = 3;
        if (volume < 1000000LL) {
            iters = 12;
        } else if (volume < 100000000LL) {
            iters = 5;
        }
        for (bool with_bias : {false, true}) {
            out.push_back({m, n, k, with_bias, Dist::Uniform, -1.0f, 1.0f, 1e-4f, 1e-4f, iters, name});
            out.push_back({m, n, k, with_bias, Dist::NearZero, -1e-6f, 1e-6f, 1e-7f, 1e-4f, iters, name});
            out.push_back({m, n, k, with_bias, Dist::MixedLarge, -100.0f, 100.0f, 1.5e-1f, 1e-3f, iters, name});
        }
    }
    return out;
}

inline std::vector<LinearBackwardCase> BuildBackwardCases() {
    const std::vector<std::tuple<int, int, int, const char*>> shapes = {
        {1, 1, 1, "tiny_111"},
        {1, 7, 3, "tiny_173"},
        {5, 1, 9, "tiny_519"},
        {37, 53, 29, "prime_375329"},
        {64, 96, 80, "mid_649680"},
        {128, 256, 192, "big_128256192"},
        {200, 300, 400, "med_200300400"},
        {512, 768, 1024, "large_5127681024"},
        {768, 1024, 512, "xlarge_7681024512"},
    };

    std::vector<LinearBackwardCase> out;
    for (const auto& s : shapes) {
        const int m = std::get<0>(s);
        const int n = std::get<1>(s);
        const int k = std::get<2>(s);
        const char* name = std::get<3>(s);
        const long long volume = static_cast<long long>(m) * n * k;
        int iters = 3;
        if (volume < 1000000LL) {
            iters = 12;
        } else if (volume < 100000000LL) {
            iters = 5;
        }

        for (bool with_bias : {false, true}) {
            out.push_back({m, n, k, with_bias, Dist::Uniform, -1.0f, 1.0f, 1e-4f, 1e-4f, iters, name});
            out.push_back({m, n, k, with_bias, Dist::NearZero, -1e-6f, 1e-6f, 1e-7f, 1e-4f, iters, name});
            out.push_back({m, n, k, with_bias, Dist::MixedLarge, -100.0f, 100.0f, 1.5e-1f, 1e-3f, iters, name});
        }
    }
    return out;
}

inline std::vector<LinearForwardBackwardCase> BuildForwardBackwardCases() {
    const std::vector<std::tuple<int, int, int, const char*>> shapes = {
        {3, 4, 2, "tiny_342"},
        {13, 37, 19, "odd_133719"},
        {64, 96, 80, "mid_649680"},
        {128, 256, 192, "big_128256192"},
        {200, 300, 400, "med_200300400"},
        {512, 768, 1024, "large_5127681024"},
    };

    std::vector<LinearForwardBackwardCase> out;
    for (const auto& s : shapes) {
        const int m = std::get<0>(s);
        const int n = std::get<1>(s);
        const int k = std::get<2>(s);
        const char* name = std::get<3>(s);
        const long long volume = static_cast<long long>(m) * n * k;
        int iters = 2;
        if (volume < 1000000LL) {
            iters = 6;
        } else if (volume < 100000000LL) {
            iters = 3;
        }

        for (bool with_bias : {false, true}) {
            out.push_back({m, n, k, with_bias, Dist::Uniform, -1.0f, 1.0f, 1e-4f, 1e-4f, iters, name});
            out.push_back({m, n, k, with_bias, Dist::NearZero, -1e-6f, 1e-6f, 1e-7f, 1e-4f, iters, name});
            out.push_back({m, n, k, with_bias, Dist::MixedLarge, -100.0f, 100.0f, 2e-1f, 2e-3f, iters, name});
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

struct ForwardEdgeCase {
    const char* name;
    int m;
    int n;
    int k;
    bool with_bias;
    float abs_tol;
    float rel_tol;
    std::vector<float> x;
    std::vector<float> w;
    std::vector<float> b;
};

struct BackwardShapeCfg {
    int m, n, k;
    float lo, hi;
    float abs_tol, rel_tol;
    bool with_bias;
};

struct QueuedLinearBackwardJob {
    int m, n, k;
    bool with_bias;
    float abs_tol, rel_tol;

    Tensor x_d;
    Tensor w_d;
    std::optional<Tensor> b_d;
    Tensor dY_d;

    std::optional<Tensor> dX_h;
    std::optional<Tensor> dW_h;
    std::optional<Tensor> db_h;

    std::vector<float> dX_expected;
    std::vector<float> dW_expected;
    std::vector<float> db_expected;
};

inline std::vector<float> reference_linear(const std::vector<float>& x,
                                           const std::vector<float>& w,
                                           const std::vector<float>* b,
                                           int m,
                                           int k,
                                           int n);

inline void reference_linear_backward(const std::vector<float>& x,
                                      const std::vector<float>& w,
                                      const std::vector<float>& dY,
                                      int m,
                                      int k,
                                      int n,
                                      std::vector<float>& dX,
                                      std::vector<float>& dW,
                                      std::vector<float>& db);

inline std::vector<float> SampleUniformVector(size_t n, float lo, float hi, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> out(n);
    for (float& v : out) v = dist(rng);
    return out;
}

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

inline void reference_linear_backward(const std::vector<float>& x,
                                      const std::vector<float>& w,
                                      const std::vector<float>& dY,
                                      int m,
                                      int k,
                                      int n,
                                      std::vector<float>& dX,
                                      std::vector<float>& dW,
                                      std::vector<float>& db) {
    dX.assign(static_cast<size_t>(m) * k, 0.0f);
    dW.assign(static_cast<size_t>(n) * k, 0.0f);
    db.assign(static_cast<size_t>(n), 0.0f);

    for (int i = 0; i < m; ++i) {
        for (int t = 0; t < k; ++t) {
            double acc = 0.0;
            for (int j = 0; j < n; ++j) {
                acc += static_cast<double>(dY[static_cast<size_t>(i) * n + j]) *
                       static_cast<double>(w[static_cast<size_t>(j) * k + t]);
            }
            dX[static_cast<size_t>(i) * k + t] = static_cast<float>(acc);
        }
    }

    for (int j = 0; j < n; ++j) {
        for (int t = 0; t < k; ++t) {
            double acc = 0.0;
            for (int i = 0; i < m; ++i) {
                acc += static_cast<double>(dY[static_cast<size_t>(i) * n + j]) *
                       static_cast<double>(x[static_cast<size_t>(i) * k + t]);
            }
            dW[static_cast<size_t>(j) * k + t] = static_cast<float>(acc);
        }
    }

    for (int j = 0; j < n; ++j) {
        double acc = 0.0;
        for (int i = 0; i < m; ++i) {
            acc += static_cast<double>(dY[static_cast<size_t>(i) * n + j]);
        }
        db[static_cast<size_t>(j)] = static_cast<float>(acc);
    }
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

static void ValidateQueuedBackwardJobs(const std::vector<QueuedLinearBackwardJob>& jobs) {
    std::vector<std::string> failures;
    for (size_t job_idx = 0; job_idx < jobs.size(); ++job_idx) {
        const auto& j = jobs[job_idx];
        if (!j.dX_h.has_value() || !j.dW_h.has_value()) {
            std::ostringstream os;
            os << "Backward job " << job_idx << " missing dX or dW buffers";
            failures.push_back(os.str());
            continue;
        }

        auto check_tensor = [&](const Tensor& t,
                                const std::vector<float>& expected,
                                const char* label) {
            const std::vector<float> got = t.to_vector<float>();
            if (got.size() != expected.size()) {
                std::ostringstream os;
                os << "Backward job " << job_idx << " " << label << " size mismatch: got "
                   << got.size() << " expected " << expected.size();
                failures.push_back(os.str());
                return;
            }

            int fail_count = 0;
            float worst_abs_error = 0.0f;
            float worst_tol = 0.0f;
            int worst_i = -1;
            for (size_t i = 0; i < got.size(); ++i) {
                const float e = expected[i];
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
                os << "Backward job " << job_idx << " " << label << " failed: " << fail_count
                   << "/" << got.size() << " elements exceeded tolerance. Worst idx " << worst_i
                   << " got=" << got[static_cast<size_t>(worst_i)]
                   << " expected=" << expected[static_cast<size_t>(worst_i)]
                   << " abs error=" << worst_abs_error << " tol=" << worst_tol;
                failures.push_back(os.str());
            }
        };

        check_tensor(j.dX_h.value(), j.dX_expected, "dX");
        check_tensor(j.dW_h.value(), j.dW_expected, "dW");

        if (j.with_bias) {
            if (!j.db_h.has_value()) {
                std::ostringstream os;
                os << "Backward job " << job_idx << " missing db buffer";
                failures.push_back(os.str());
            } else {
                check_tensor(j.db_h.value(), j.db_expected, "db");
            }
        }
    }

    if (!failures.empty()) {
        std::ostringstream all;
        all << "Queued backward linear validation failed in " << failures.size() << " check(s):\n";
        for (const auto& f : failures) all << "  " << f << "\n";
        ADD_FAILURE() << all.str();
    }
}

}  // namespace fa_test

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

constexpr uint32_t kLayerNormSeedBase = 21937u;

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

struct LayerNormCase {
    int m, n;
    Dist dist;
    float lo, hi;
    float abs_tol;
    float rel_tol;
    int iters;
    const char* name;
};

struct LayerNormEdgeCase {
    const char* name;
    int m;
    int n;
    float abs_tol;
    float rel_tol;
    std::vector<float> x;
    std::vector<float> gamma;
    std::vector<float> beta;
};

struct LayerNormRefForward {
    std::vector<float> y;
    std::vector<float> mean;
    std::vector<float> rstd;
    std::vector<float> xhat;
};

struct LayerNormRefBackward {
    std::vector<float> dX;
    std::vector<float> dgamma;
    std::vector<float> dbeta;
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

inline std::vector<LayerNormCase> BuildForwardCases() {
    const std::vector<std::tuple<int, int, const char*>> shapes = {
        {1, 1, "tiny_11"},
        {1, 7, "tiny_17"},
        {5, 1, "tiny_51"},
        {13, 37, "odd_1337"},
        {64, 96, "mid_6496"},
        {128, 256, "big_128256"},
        {200, 300, "med_200300"},
        {256, 512, "trans_256512"},
        {128, 1024, "trans_1281024"},
    };

    std::vector<LayerNormCase> out;
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

        out.push_back({m, n, Dist::Uniform, -1.0f, 1.0f, 1e-4f, 1e-4f, iters, name});
        out.push_back({m, n, Dist::NearZero, -1e-6f, 1e-6f, 2e-5f, 2e-4f, iters, name});
        out.push_back({m, n, Dist::MixedLarge, -100.0f, 100.0f, 1e-3f, 2e-3f, iters, name});
    }
    return out;
}

inline std::vector<LayerNormCase> BuildBackwardCases() {
    return BuildForwardCases();
}

inline std::vector<LayerNormCase> BuildForwardBackwardCases() {
    const std::vector<std::tuple<int, int, const char*>> shapes = {
        {3, 4, "tiny_34"},
        {13, 37, "odd_1337"},
        {64, 96, "mid_6496"},
        {128, 256, "big_128256"},
        {256, 512, "trans_256512"},
    };

    std::vector<LayerNormCase> out;
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

        out.push_back({m, n, Dist::Uniform, -1.0f, 1.0f, 2e-4f, 2e-4f, iters, name});
        out.push_back({m, n, Dist::NearZero, -1e-6f, 1e-6f, 3e-5f, 3e-4f, iters, name});
        out.push_back({m, n, Dist::MixedLarge, -100.0f, 100.0f, 2e-3f, 3e-3f, iters, name});
    }
    return out;
}

inline LayerNormRefForward reference_layernorm_forward(const std::vector<float>& x,
                                                       const std::vector<float>& gamma,
                                                       const std::vector<float>& beta,
                                                       int m,
                                                       int n,
                                                       float eps) {
    if (static_cast<int>(x.size()) != m * n) {
        throw std::invalid_argument("reference_layernorm_forward: x has invalid size");
    }
    if (static_cast<int>(gamma.size()) != n || static_cast<int>(beta.size()) != n) {
        throw std::invalid_argument("reference_layernorm_forward: gamma/beta have invalid size");
    }

    LayerNormRefForward out;
    out.y.assign(static_cast<size_t>(m) * n, 0.0f);
    out.mean.assign(static_cast<size_t>(m), 0.0f);
    out.rstd.assign(static_cast<size_t>(m), 0.0f);
    out.xhat.assign(static_cast<size_t>(m) * n, 0.0f);

    for (int row = 0; row < m; ++row) {
        double sum = 0.0;
        for (int col = 0; col < n; ++col) {
            sum += static_cast<double>(x[static_cast<size_t>(row) * n + col]);
        }
        const double mu = sum / static_cast<double>(n);

        double var_sum = 0.0;
        for (int col = 0; col < n; ++col) {
            const double centered = static_cast<double>(x[static_cast<size_t>(row) * n + col]) - mu;
            var_sum += centered * centered;
        }
        const double var = var_sum / static_cast<double>(n);
        const double inv_std = 1.0 / std::sqrt(var + static_cast<double>(eps));

        out.mean[static_cast<size_t>(row)] = static_cast<float>(mu);
        out.rstd[static_cast<size_t>(row)] = static_cast<float>(inv_std);

        for (int col = 0; col < n; ++col) {
            const double xh = (static_cast<double>(x[static_cast<size_t>(row) * n + col]) - mu) * inv_std;
            out.xhat[static_cast<size_t>(row) * n + col] = static_cast<float>(xh);
            out.y[static_cast<size_t>(row) * n + col] =
                static_cast<float>(xh * static_cast<double>(gamma[col]) + static_cast<double>(beta[col]));
        }
    }

    return out;
}

inline LayerNormRefBackward reference_layernorm_backward(const std::vector<float>& dY,
                                                         const std::vector<float>& x,
                                                         const std::vector<float>& gamma,
                                                         const std::vector<float>& mean,
                                                         const std::vector<float>& rstd,
                                                         int m,
                                                         int n,
                                                         bool needs_dX,
                                                         bool needs_dgamma,
                                                         bool needs_dbeta) {
    if (static_cast<int>(dY.size()) != m * n || static_cast<int>(x.size()) != m * n) {
        throw std::invalid_argument("reference_layernorm_backward: x/dY have invalid size");
    }
    if (static_cast<int>(gamma.size()) != n || static_cast<int>(mean.size()) != m || static_cast<int>(rstd.size()) != m) {
        throw std::invalid_argument("reference_layernorm_backward: gamma/mean/rstd have invalid size");
    }

    LayerNormRefBackward out;
    if (needs_dX) {
        out.dX.assign(static_cast<size_t>(m) * n, 0.0f);
    }
    if (needs_dgamma) {
        out.dgamma.assign(static_cast<size_t>(n), 0.0f);
    }
    if (needs_dbeta) {
        out.dbeta.assign(static_cast<size_t>(n), 0.0f);
    }

    if (needs_dgamma || needs_dbeta) {
        for (int row = 0; row < m; ++row) {
            const double mu = static_cast<double>(mean[static_cast<size_t>(row)]);
            const double inv_std = static_cast<double>(rstd[static_cast<size_t>(row)]);
            for (int col = 0; col < n; ++col) {
                const size_t idx = static_cast<size_t>(row) * n + col;
                const double dy = static_cast<double>(dY[idx]);
                const double xh = (static_cast<double>(x[idx]) - mu) * inv_std;
                if (needs_dgamma) {
                    out.dgamma[static_cast<size_t>(col)] += static_cast<float>(dy * xh);
                }
                if (needs_dbeta) {
                    out.dbeta[static_cast<size_t>(col)] += static_cast<float>(dy);
                }
            }
        }
    }

    if (needs_dX) {
        const double inv_n = 1.0 / static_cast<double>(n);
        for (int row = 0; row < m; ++row) {
            const double mu = static_cast<double>(mean[static_cast<size_t>(row)]);
            const double inv_std = static_cast<double>(rstd[static_cast<size_t>(row)]);

            double sum_dyg = 0.0;
            double sum_dyg_xhat = 0.0;
            for (int col = 0; col < n; ++col) {
                const size_t idx = static_cast<size_t>(row) * n + col;
                const double xh = (static_cast<double>(x[idx]) - mu) * inv_std;
                const double dyg = static_cast<double>(dY[idx]) * static_cast<double>(gamma[col]);
                sum_dyg += dyg;
                sum_dyg_xhat += dyg * xh;
            }

            for (int col = 0; col < n; ++col) {
                const size_t idx = static_cast<size_t>(row) * n + col;
                const double xh = (static_cast<double>(x[idx]) - mu) * inv_std;
                const double dyg = static_cast<double>(dY[idx]) * static_cast<double>(gamma[col]);
                const double inner = static_cast<double>(n) * dyg - sum_dyg - xh * sum_dyg_xhat;
                out.dX[idx] = static_cast<float>(inv_std * inv_n * inner);
            }
        }
    }

    return out;
}

}  // namespace fa_test

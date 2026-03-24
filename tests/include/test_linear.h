#pragma once

#include "ops.h"
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
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

    Stream stream;
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
          stream(),
          x_h({m_, k_}, get_dtype<T>(), Device::CPU, stream),
          w_h({n_, k_}, get_dtype<T>(), Device::CPU, stream),
          rng(seed) {
        std::uniform_real_distribution<T> dist(start, end);

        x_ref.resize(static_cast<size_t>(m) * k);
        w_ref.resize(static_cast<size_t>(n) * k);
        for (auto& v : x_ref) v = dist(rng);
        for (auto& v : w_ref) v = dist(rng);

        x_h.copy_from(x_ref, stream);
        w_h.copy_from(w_ref, stream);

        if (with_bias) {
            b_ref.resize(static_cast<size_t>(n));
            for (auto& v : b_ref) v = dist(rng);
            b_h.emplace(std::vector<int64_t>{n}, get_dtype<T>(), Device::CPU, stream);
            b_h->copy_from(b_ref, stream);
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

enum class HandlePattern {SharedHandle, PerStreamHandle,};

enum class HostLaunchPattern {SingleThreadRoundRobin, MultiThreaded,};

struct MatrixJob {
    fa_test::QueuedLinearJob job;
    std::string tag;
};

struct MatrixStats {
    int jobs_total = 0;
    int jobs_failed = 0;
    int elements_total = 0;
    int elements_failed = 0;

    float worst_abs_error = 0.0f;
    float worst_tol = 0.0f;
    int worst_job_idx = -1;
    int worst_element_idx = -1;
    std::string worst_job_tag;

    std::vector<std::string> sample_failures;
    std::vector<std::string> launch_errors;
};

inline std::vector<float> reference_linear(const std::vector<float>& x,
                                           const std::vector<float>& w,
                                           const std::vector<float>* b,
                                           int m,
                                           int k,
                                           int n);

inline const char* HandlePatternName(HandlePattern p) {
    switch (p) {
        case HandlePattern::SharedHandle: return "shared_handle";
        case HandlePattern::PerStreamHandle: return "per_stream_handle";
    }
    return "unknown";
}

inline const char* HostLaunchPatternName(HostLaunchPattern p) {
    switch (p) {
        case HostLaunchPattern::SingleThreadRoundRobin: return "single_thread_round_robin";
        case HostLaunchPattern::MultiThreaded: return "multi_threaded";
    }
    return "unknown";
}

inline MatrixStats MergeStats(const MatrixStats& a, const MatrixStats& b) {
    MatrixStats out;
    out.jobs_total = a.jobs_total + b.jobs_total;
    out.jobs_failed = a.jobs_failed + b.jobs_failed;
    out.elements_total = a.elements_total + b.elements_total;
    out.elements_failed = a.elements_failed + b.elements_failed;

    out.worst_abs_error = a.worst_abs_error;
    out.worst_tol = a.worst_tol;
    out.worst_job_idx = a.worst_job_idx;
    out.worst_element_idx = a.worst_element_idx;
    out.worst_job_tag = a.worst_job_tag;

    if (b.worst_abs_error > out.worst_abs_error) {
        out.worst_abs_error = b.worst_abs_error;
        out.worst_tol = b.worst_tol;
        out.worst_job_idx = b.worst_job_idx;
        out.worst_element_idx = b.worst_element_idx;
        out.worst_job_tag = b.worst_job_tag;
    }

    out.sample_failures = a.sample_failures;
    out.sample_failures.insert(out.sample_failures.end(), b.sample_failures.begin(), b.sample_failures.end());

    out.launch_errors = a.launch_errors;
    out.launch_errors.insert(out.launch_errors.end(), b.launch_errors.begin(), b.launch_errors.end());

    return out;
}

inline std::string StatsToString(const MatrixStats& s, const std::string& scenario_name) {
    std::ostringstream os;
    os << "scenario=" << scenario_name
       << " jobs_failed=" << s.jobs_failed << "/" << s.jobs_total
       << " elems_failed=" << s.elements_failed << "/" << s.elements_total
       << " worst_abs_error=" << s.worst_abs_error
       << " worst_tol=" << s.worst_tol
       << " worst_job_idx=" << s.worst_job_idx
       << " worst_element_idx=" << s.worst_element_idx
       << " worst_job_tag=" << s.worst_job_tag
       << " launch_errors=" << s.launch_errors.size();

    if (!s.sample_failures.empty()) {
        os << "\nsample_failures:\n";
        const int limit = static_cast<int>(std::min<size_t>(s.sample_failures.size(), 8));
        for (int i = 0; i < limit; ++i) {
            os << "  " << s.sample_failures[static_cast<size_t>(i)] << "\n";
        }
    }
    if (!s.launch_errors.empty()) {
        os << "launch_errors:\n";
        const int limit = static_cast<int>(std::min<size_t>(s.launch_errors.size(), 8));
        for (int i = 0; i < limit; ++i) {
            os << "  " << s.launch_errors[static_cast<size_t>(i)] << "\n";
        }
    }
    return os.str();
}

inline MatrixStats ValidateMatrixJobs(const std::vector<MatrixJob>& jobs) {
    MatrixStats out;
    out.jobs_total = static_cast<int>(jobs.size());

    for (size_t job_idx = 0; job_idx < jobs.size(); ++job_idx) {
        const auto& rec = jobs[job_idx];
        const auto& j = rec.job;
        auto* y_ptr = static_cast<const float*>(j.y_h.data_);

        int fail_count = 0;
        float worst_abs = 0.0f;
        float worst_tol = 0.0f;
        int worst_i = -1;

        const int total = j.m * j.n;
        out.elements_total += total;

        for (int i = 0; i < total; ++i) {
            const float e = j.expected[static_cast<size_t>(i)];
            const float got = y_ptr[i];
            const float abs_err = std::fabs(got - e);
            const float tol = j.abs_tol + j.rel_tol * std::fabs(e);

            if (abs_err > tol) {
                ++fail_count;
                if (abs_err > worst_abs) {
                    worst_abs = abs_err;
                    worst_tol = tol;
                    worst_i = i;
                }
            }
        }

        if (fail_count > 0) {
            ++out.jobs_failed;
            out.elements_failed += fail_count;
            if (worst_abs > out.worst_abs_error) {
                out.worst_abs_error = worst_abs;
                out.worst_tol = worst_tol;
                out.worst_job_idx = static_cast<int>(job_idx);
                out.worst_element_idx = worst_i;
                out.worst_job_tag = rec.tag;
            }

            if (out.sample_failures.size() < 12) {
                std::ostringstream one;
                one << "job=" << job_idx
                    << " tag=" << rec.tag
                    << " fail_count=" << fail_count << "/" << total
                    << " worst_elem=" << worst_i
                    << " worst_abs=" << worst_abs
                    << " worst_tol=" << worst_tol;
                out.sample_failures.push_back(one.str());
            }
        }
    }

    return out;
}

inline MatrixJob BuildOneJob(const ShapeCfg& cfg, uint32_t seed, Stream& stream, CublasHandle& handle, const std::string& tag) {
    HostLinearInputs<float> inputs(cfg.m, cfg.n, cfg.k, cfg.with_bias, cfg.lo, cfg.hi, seed);
    
    Tensor x_d = inputs.x_h.clone(Device::CUDA, stream);
    Tensor w_d = inputs.w_h.clone(Device::CUDA, stream);
    std::optional<Tensor> b_d;
    const Tensor* b_d_ptr = nullptr;
    if (cfg.with_bias) {
        if (!inputs.b_h.has_value()) {
            throw std::runtime_error("Expected host bias tensor for with_bias=true");
        }
        b_d.emplace(inputs.b_h->clone(Device::CUDA, stream));
        b_d_ptr = &b_d.value();
    }

    LinearResults out = linear_forward(x_d, w_d, b_d_ptr, &stream, handle);
    Tensor y_h = out.Y.clone(Device::CPU, stream);

    const std::vector<float>* b_ref = cfg.with_bias ? &inputs.b_ref : nullptr;
    const std::vector<float> expected = reference_linear(inputs.x_ref, inputs.w_ref, b_ref, cfg.m, cfg.k, cfg.n);

    return MatrixJob{QueuedLinearJob{cfg.m, cfg.n, cfg.k, cfg.abs_tol, cfg.rel_tol, std::move(x_d), std::move(w_d), std::move(b_d), std::move(y_h), std::move(expected)}, tag};
}

inline MatrixStats RunConcurrencyScenario(HandlePattern handle_pattern,
                                          HostLaunchPattern launch_pattern,
                                          int num_streams,
                                          int rounds,
                                          uint32_t seed_base) {
    const std::array<ShapeCfg, 5> cfgs = {{
        {64, 96, 128, -1.0f,   1.0f,   1e-4f,   1e-4f, true},
        {31, 47, 29,  -1.0f,   1.0f,   1e-4f,   1e-4f, false},
        {128, 72, 96, -100.0f, 100.0f, 1.5e-1f, 1e-3f, true},
        {200, 300, 400, -1.0f,  1.0f,   1e-4f,   1e-4f, true},
        {96, 80, 64,  -1e-6f,  1e-6f,  1e-7f,   1e-4f, true},
    }};

    std::vector<std::unique_ptr<Stream>> streams;
    streams.reserve(num_streams);
    for (int s = 0; s < num_streams; ++s) {
        streams.emplace_back(std::make_unique<Stream>());
    }

    std::vector<std::unique_ptr<CublasHandle>> per_stream_handles;
    std::unique_ptr<CublasHandle> shared_handle;
    if (handle_pattern == HandlePattern::PerStreamHandle) {
        per_stream_handles.reserve(num_streams);
        for (int s = 0; s < num_streams; ++s) {
            per_stream_handles.emplace_back(std::make_unique<CublasHandle>());
        }
    } else {
        shared_handle = std::make_unique<CublasHandle>();
    }

    std::vector<MatrixJob> jobs;
    jobs.reserve(static_cast<size_t>(num_streams) * static_cast<size_t>(rounds));
    std::vector<std::string> launch_errors;

    if (launch_pattern == HostLaunchPattern::SingleThreadRoundRobin) {
        for (int round = 0; round < rounds; ++round) {
            for (int s = 0; s < num_streams; ++s) {
                const ShapeCfg& c = cfgs[static_cast<size_t>((round + s) % static_cast<int>(cfgs.size()))];
                const uint32_t seed = seed_base + static_cast<uint32_t>(round * 1000 + s);

                CublasHandle& handle =
                    (handle_pattern == HandlePattern::SharedHandle) ? *shared_handle : *per_stream_handles[s];

                std::ostringstream tag;
                tag << "round=" << round
                    << " stream=" << s
                    << " seed=" << seed
                    << " " << ShapeString(c.m, c.n, c.k)
                    << " hp=" << HandlePatternName(handle_pattern)
                    << " lp=" << HostLaunchPatternName(launch_pattern);

                jobs.push_back(BuildOneJob(c, seed, *streams[s], handle, tag.str()));
            }
        }
    } else {
        std::mutex jobs_mu;
        std::mutex err_mu;
        std::vector<std::thread> workers;
        workers.reserve(num_streams);

        std::atomic<int> ready_count{0};
        std::atomic<bool> start_flag{false};

        for (int s = 0; s < num_streams; ++s) {
            workers.emplace_back([&, s]() {
                std::vector<MatrixJob> local_jobs;
                local_jobs.reserve(static_cast<size_t>(rounds));

                ready_count.fetch_add(1, std::memory_order_release);
                while (!start_flag.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }

                try {
                    for (int round = 0; round < rounds; ++round) {
                        const ShapeCfg& c = cfgs[static_cast<size_t>((round + s) % static_cast<int>(cfgs.size()))];
                        const uint32_t seed = seed_base + static_cast<uint32_t>(s * 100000 + round);

                        CublasHandle& handle =
                            (handle_pattern == HandlePattern::SharedHandle) ? *shared_handle : *per_stream_handles[s];

                        std::ostringstream tag;
                        tag << "round=" << round
                            << " stream=" << s
                            << " seed=" << seed
                            << " " << ShapeString(c.m, c.n, c.k)
                            << " hp=" << HandlePatternName(handle_pattern)
                            << " lp=" << HostLaunchPatternName(launch_pattern);

                        local_jobs.push_back(BuildOneJob(c, seed, *streams[s], handle, tag.str()));
                    }
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(err_mu);
                    std::ostringstream err;
                    err << "thread stream=" << s << " exception: " << e.what();
                    launch_errors.push_back(err.str());
                } catch (...) {
                    std::lock_guard<std::mutex> lock(err_mu);
                    std::ostringstream err;
                    err << "thread stream=" << s << " unknown exception";
                    launch_errors.push_back(err.str());
                }

                std::lock_guard<std::mutex> lock(jobs_mu);
                for (auto& j : local_jobs) {
                    jobs.push_back(std::move(j));
                }
            });
        }

        while (ready_count.load(std::memory_order_acquire) < num_streams) {
            std::this_thread::yield();
        }
        start_flag.store(true, std::memory_order_release);

        for (auto& t : workers) {
            t.join();
        }
    }

    for (auto& s : streams) {
        s->synchronize();
    }

    MatrixStats stats = ValidateMatrixJobs(jobs);
    stats.launch_errors = std::move(launch_errors);
    return stats;
}

inline MatrixStats RunAcrossStreamCounts(HandlePattern handle_pattern,
                                         HostLaunchPattern launch_pattern,
                                         const std::vector<int>& stream_counts,
                                         int rounds_per_stream_count,
                                         uint32_t seed_base) {
    MatrixStats agg;
    for (int num_streams : stream_counts) {
        MatrixStats one = RunConcurrencyScenario(handle_pattern,
                                                 launch_pattern,
                                                 num_streams,
                                                 rounds_per_stream_count,
                                                 seed_base + static_cast<uint32_t>(num_streams * 1000000));
        agg = MergeStats(agg, one);
    }
    return agg;
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

// Copyright (c) 2025 - present KMX Systems. All rights reserved.
/// @file perf-test.cpp
/// @brief Integration performance benchmarks for kmx::fft using std::chrono.
#include <algorithm>
#include <array>
#include <chrono>
#include <complex>
#include <cstdio>
#include <kmx/fft.hpp>
#include <numbers>
#include <random>
// Select Engine based on macros
#if defined(KMX_FFT_USE_DYNAMIC_ENGINE)
template <typename T>
using tested_engine = kmx::fft::dynamic_engine<std::complex<T>>;
#define KMX_FFT_ENGINE_NAME "Dynamic Engine"
#elif defined(KMX_FFT_ENABLE_AVX2)
#include <kmx/fft_backend/avx2.hpp>
template <typename T>
using tested_engine = kmx::fft::engine<std::complex<T>, kmx::fft::backend::avx2<std::complex<T>>>;
#define KMX_FFT_ENGINE_NAME "AVX2"
#elif defined(KMX_FFT_ENABLE_OPENCL)
#include <kmx/fft_backend/opencl.hpp>
template <typename T>
using tested_engine = kmx::fft::engine<std::complex<T>, kmx::fft::backend::opencl<std::complex<T>>>;
#define KMX_FFT_ENGINE_NAME "OpenCL"
#else
template <typename T>
using tested_engine = kmx::fft::engine<std::complex<T>, kmx::fft::backend::software<std::complex<T>>>;
#define KMX_FFT_ENGINE_NAME "Software"
#endif

#if defined(KMX_FFT_USE_DYNAMIC_ENGINE)
template <typename Plan>
const char* plan_strategy(const Plan& plan) { return kmx::fft::backend::to_string(plan.routed_id); }
template <typename Engine>
const char* engine_strategy(const Engine& eng) { return kmx::fft::backend::to_string(eng.get_last_strategy()); }
#else
template <typename Plan>
const char* plan_strategy(const Plan&) { return "avx2"; }
template <typename Engine>
const char* engine_strategy(const Engine&) { return ""; }
#endif

namespace
{
    using clock_t = std::chrono::steady_clock;

    /// Returns elapsed microseconds for a single call to fn.
    template <typename Fn>
    [[nodiscard]] double measure_us(Fn&& fn)
    {
        const auto t0 = clock_t::now();
        fn();
        const auto t1 = clock_t::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    /// Runs fn `reps` times after one warm-up call, returns median µs.
    template <typename Fn>
    [[nodiscard]] double bench_median_us(Fn&& fn, const int reps = 51)
    {
        fn(); // warm-up (also populates caches)
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(reps));
        for (int i = 0; i < reps; ++i)
            samples.push_back(measure_us(fn));
        std::nth_element(samples.begin(), samples.begin() + reps / 2, samples.end());
        return samples[static_cast<std::size_t>(reps / 2)];
    }

    template <typename T>
    std::vector<std::complex<T>> make_random_signal(const std::size_t n, const unsigned seed = 42u)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<T> dist(T{-1}, T{1});
        std::vector<std::complex<T>> v(n);
        for (auto& c : v)
            c = {dist(rng), dist(rng)};
        return v;
    }

    void print_header()
    {
        std::printf("\n%-40s %10s %10s %10s %10s\n", "Benchmark", "N", "Cold µs", "Warm µs (med)", "Strategy");
        std::printf("%-40s %10s %10s %10s %10s\n", "---", "---", "---", "---", "---");
    }

    void print_row(const char* name, const std::size_t n, const double cold_us, const double warm_us, const char* strategy = "")
    {
        std::printf("%-40s %10zu %10.1f %10.1f [%8s]\n", name, n, cold_us, warm_us, strategy);
    }

    // -------------------------------------------------------------------------
    // 1-D benchmarks
    // -------------------------------------------------------------------------
    template <typename T>
    void bench_1d(const std::size_t n, const char* label)
    {
        auto signal = make_random_signal<T>(n);
        tested_engine<T> eng;

        // Cold: engine constructed fresh, cache empty → triggers twiddle/Bluestein setup
        tested_engine<T> eng_cold;
        const double cold_us = measure_us([&]
        {
            auto s = signal;
            const kmx::tensor::view<std::complex<T>> v{std::span<std::complex<T>>(s)};
            eng_cold.transform_1d(v);
        });

        const double warm_us = bench_median_us([&]
        {
            auto s = signal;
            const kmx::tensor::view<std::complex<T>> v{std::span<std::complex<T>>(s)};
            eng.transform_1d(v);
        });

        print_row(label, n, cold_us, warm_us, engine_strategy(eng));
    }

    // -------------------------------------------------------------------------
    // 1-D roundtrip (forward + inverse)
    // -------------------------------------------------------------------------
    template <typename T>
    void bench_roundtrip(const std::size_t n, const char* label)
    {
        auto signal = make_random_signal<T>(n);
        tested_engine<T> eng;
        // one warm-up to populate caches for both forward & inverse
        {
            auto s = signal;
            const kmx::tensor::view<std::complex<T>> v{std::span<std::complex<T>>(s)};
            eng.transform_1d(v, false);
            eng.transform_1d(v, true);
        }

        const double warm_us = bench_median_us([&]
        {
            auto s = signal;
            const kmx::tensor::view<std::complex<T>> v{std::span<std::complex<T>>(s)};
            eng.transform_1d(v, false);
            eng.transform_1d(v, true);
        });

        print_row(label, n, 0.0, warm_us, engine_strategy(eng));
    }

    // -------------------------------------------------------------------------
    // 2-D benchmarks
    // -------------------------------------------------------------------------
    template <typename T>
    void bench_2d(const std::size_t rows, const std::size_t cols, const char* label)
    {
        const std::size_t n = rows * cols;
        auto signal = make_random_signal<T>(n);

        const std::array<std::size_t, 2> shape_arr{rows, cols};
        const kmx::tensor::dimension_shape shape(shape_arr);

        tested_engine<T> eng_cold;
        const double cold_us = measure_us([&]
        {
            auto s = signal;
            const kmx::tensor::view<std::complex<T>> v{std::span<std::complex<T>>(s), shape};
            eng_cold.transform_2d(v);
        });

        tested_engine<T> eng;
        const double warm_us = bench_median_us([&]
        {
            auto s = signal;
            const kmx::tensor::view<std::complex<T>> v{std::span<std::complex<T>>(s), shape};
            eng.transform_2d(v);
        });

        print_row(label, n, cold_us, warm_us, engine_strategy(eng));
    }

    // -------------------------------------------------------------------------
    // Out-of-place vs in-place comparison (1-D)
    // -------------------------------------------------------------------------
    template <typename T>
    void bench_1d_out_of_place(const std::size_t n, const char* label)
    {
        const auto signal = make_random_signal<T>(n);
        std::vector<std::complex<T>> output(n);

        tested_engine<T> eng;
        {
            // warm up caches
            const kmx::tensor::view<const std::complex<T>> in{std::span<const std::complex<T>>(signal)};
            const kmx::tensor::view<std::complex<T>> out{std::span<std::complex<T>>(output)};
            eng.transform_1d(in, out);
        }

        const double warm_us = bench_median_us([&]
        {
            const kmx::tensor::view<const std::complex<T>> in{std::span<const std::complex<T>>(signal)};
            const kmx::tensor::view<std::complex<T>> out{std::span<std::complex<T>>(output)};
            eng.transform_1d(in, out);
        });

        print_row(label, n, 0.0, warm_us, engine_strategy(eng));
    }
} // anonymous namespace

int main()
{
    std::printf("kmx::fft — performance integration test [%s]\n", KMX_FFT_ENGINE_NAME);
    std::printf("Median over 51 repetitions after one warm-up. Cold = first call with empty cache.\n");

    // ------------------------------------------------------------------
    // 1-D double — power-of-2
    // ------------------------------------------------------------------
    print_header();
    bench_1d<double>(8,     "1D double  N=8     (pow2)");
    bench_1d<double>(16,    "1D double  N=16    (pow2)");
    bench_1d<double>(32,    "1D double  N=32    (pow2)");
    bench_1d<double>(64,    "1D double  N=64    (pow2)");
    bench_1d<double>(256,   "1D double  N=256   (pow2)");
    bench_1d<double>(1024,  "1D double  N=1024  (pow2)");
    bench_1d<double>(4096,  "1D double  N=4096  (pow2)");
    bench_1d<double>(16384, "1D double  N=16384 (pow2)");

    // ------------------------------------------------------------------
    // 1-D double — non-power-of-2 (Bluestein)
    // ------------------------------------------------------------------
    bench_1d<double>(5,    "1D double  N=5     (Bluestein)");
    bench_1d<double>(11,   "1D double  N=11    (Bluestein)");
    bench_1d<double>(15,   "1D double  N=15    (Bluestein)");
    bench_1d<double>(100,  "1D double  N=100   (Bluestein)");
    bench_1d<double>(1000, "1D double  N=1000  (Bluestein)");
    bench_1d<double>(1500, "1D double  N=1500  (Bluestein)");

    // ------------------------------------------------------------------
    // 1-D float — power-of-2
    // ------------------------------------------------------------------
    bench_1d<float>(1024,  "1D float   N=1024  (pow2)");
    bench_1d<float>(4096,  "1D float   N=4096  (pow2)");
    bench_1d<float>(16384, "1D float   N=16384 (pow2)");

    // ------------------------------------------------------------------
    // 1-D roundtrip (forward + inverse)
    // ------------------------------------------------------------------
    std::printf("\n--- roundtrip (forward + inverse, warm) ---\n");
    bench_roundtrip<double>(1024,  "roundtrip double N=1024");
    bench_roundtrip<double>(4096,  "roundtrip double N=4096");
    bench_roundtrip<float> (4096,  "roundtrip float  N=4096");

    // ------------------------------------------------------------------
    // 1-D out-of-place
    // ------------------------------------------------------------------
    std::printf("\n--- out-of-place (warm) ---\n");
    bench_1d_out_of_place<double>(1024,  "out-of-place double N=1024");
    bench_1d_out_of_place<double>(4096,  "out-of-place double N=4096");

    // ------------------------------------------------------------------
    // 2-D double
    // ------------------------------------------------------------------
    std::printf("\n--- 2D transforms ---\n");
    print_header();
    bench_2d<double>(32,  32,  "2D double  32x32");
    bench_2d<double>(64,  64,  "2D double  64x64");
    bench_2d<double>(128, 128, "2D double  128x128");
    bench_2d<double>(256, 256, "2D double  256x256");
    bench_2d<double>(1024, 1024, "2D double  1024x1024");
    bench_2d<double>(2048, 2048, "2D double  2048x2048");
    bench_2d<float> (128, 128, "2D float   128x128");
    bench_2d<float> (256, 256, "2D float   256x256");

    std::printf("\nDone.\n");
}

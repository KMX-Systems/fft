// Standalone FFTW3 vs kmx::fft comparison benchmark.
// Compile: g++ -O3 -march=native -flto -std=c++26 -funroll-loops
//          fftw_compare.cpp -lfftw3 -lfftw3f -I source/library/inc -o fftw_compare
#include <algorithm>
#include <array>
#include <chrono>
#include <complex>
#include <cstdlib>
#include <cstdio>
#include <fftw3.h>
#include <kmx/fft.hpp>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

// Select Engine based on macros
#if defined(KMX_FFT_ENABLE_AVX2)
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

using steady_clock = std::chrono::steady_clock;

enum class bench_mode {
    all,
    one_d_double,
    one_d_float,
    two_d_double,
};

struct bench_options {
    bench_mode mode = bench_mode::all;
    int reps = 51;
    int threads = 0;
    bool pin_threads = false;
    std::optional<std::size_t> n;
    std::optional<std::size_t> rows;
    std::optional<std::size_t> cols;
};

[[nodiscard]] static const char* mode_name(const bench_mode mode)
{
    switch (mode) {
    case bench_mode::all: return "all";
    case bench_mode::one_d_double: return "1d-double";
    case bench_mode::one_d_float: return "1d-float";
    case bench_mode::two_d_double: return "2d-double";
    }
    return "unknown";
}

[[noreturn]] static void usage(const char* argv0)
{
    std::fprintf(stderr,
                 "Usage: %s [--mode all|1d-double|1d-float|2d-double] [--n N] [--rows R --cols C] [--reps N] [--threads N] [--pin]\n",
                 argv0);
    std::exit(2);
}

[[nodiscard]] static std::size_t parse_size_arg(const char* value, const char* name)
{
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (value == end || (end && *end != '\0')) {
        std::fprintf(stderr, "Invalid value for %s: %s\n", name, value);
        std::exit(2);
    }
    return static_cast<std::size_t>(parsed);
}

[[nodiscard]] static int parse_int_arg(const char* value, const char* name)
{
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (value == end || (end && *end != '\0')) {
        std::fprintf(stderr, "Invalid value for %s: %s\n", name, value);
        std::exit(2);
    }
    return static_cast<int>(parsed);
}

[[nodiscard]] static bench_options parse_args(const int argc, char** argv)
{
    bench_options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", name);
                usage(argv[0]);
            }
            return argv[++i];
        };

        if (arg == "--mode") {
            const std::string_view value = require_value("--mode");
            if (value == "all") options.mode = bench_mode::all;
            else if (value == "1d-double") options.mode = bench_mode::one_d_double;
            else if (value == "1d-float") options.mode = bench_mode::one_d_float;
            else if (value == "2d-double") options.mode = bench_mode::two_d_double;
            else usage(argv[0]);
        } else if (arg == "--n") {
            options.n = parse_size_arg(require_value("--n"), "--n");
        } else if (arg == "--rows") {
            options.rows = parse_size_arg(require_value("--rows"), "--rows");
        } else if (arg == "--cols") {
            options.cols = parse_size_arg(require_value("--cols"), "--cols");
        } else if (arg == "--reps") {
            options.reps = parse_int_arg(require_value("--reps"), "--reps");
        } else if (arg == "--threads") {
            options.threads = parse_int_arg(require_value("--threads"), "--threads");
        } else if (arg == "--pin") {
            options.pin_threads = true;
        } else if (arg == "--help") {
            usage(argv[0]);
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            usage(argv[0]);
        }
    }

    if (options.reps <= 0) {
        std::fprintf(stderr, "--reps must be > 0\n");
        std::exit(2);
    }
    if (options.threads < 0) {
        std::fprintf(stderr, "--threads must be >= 0\n");
        std::exit(2);
    }
    if ((options.mode == bench_mode::one_d_double || options.mode == bench_mode::one_d_float) && !options.n.has_value()) {
        std::fprintf(stderr, "--n is required for 1D focused runs\n");
        std::exit(2);
    }
    if (options.mode == bench_mode::two_d_double && (!options.rows.has_value() || !options.cols.has_value())) {
        std::fprintf(stderr, "--rows and --cols are required for 2D focused runs\n");
        std::exit(2);
    }
    return options;
}

static void configure_runtime(const bench_options& options)
{
#if defined(_OPENMP)
    if (options.threads > 0)
        omp_set_num_threads(options.threads);
#endif
    if (options.pin_threads) {
        setenv("OMP_PROC_BIND", "true", 1);
        setenv("OMP_PLACES", "cores", 1);
    }
}

template <typename Fn>
[[nodiscard]] double measure_us(Fn&& fn)
{
    const auto t0 = steady_clock::now();
    fn();
    const auto t1 = steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

template <typename Fn>
[[nodiscard]] double bench_median_us(Fn&& fn, const int reps = 51)
{
    fn(); // warm-up
    std::vector<double> s;
    s.reserve(static_cast<std::size_t>(reps));
    for (int i = 0; i < reps; ++i)
        s.push_back(measure_us(fn));
    std::nth_element(s.begin(), s.begin() + reps / 2, s.end());
    return s[static_cast<std::size_t>(reps / 2)];
}

// -------------------------------------------------------------------------
// FFTW helpers
// -------------------------------------------------------------------------
struct FftwPlanD
{
    fftw_plan p{nullptr};
    ~FftwPlanD() { if (p) fftw_destroy_plan(p); }
};

struct FftwPlanF
{
    fftwf_plan p{nullptr};
    ~FftwPlanF() { if (p) fftwf_destroy_plan(p); }
};

// -------------------------------------------------------------------------
// 1-D double comparison
// -------------------------------------------------------------------------
void compare_1d_double(const std::size_t n, const int reps)
{
    // Shared input
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<std::complex<double>> src(n);
    for (auto& c : src) c = {dist(rng), dist(rng)};

    std::vector<fftw_complex> fw_in(n), fw_out(n);
    for (std::size_t i = 0; i < n; ++i) { fw_in[i][0] = src[i].real(); fw_in[i][1] = src[i].imag(); }

    // --- FFTW MEASURE ---
    FftwPlanD plan_meas;
    plan_meas.p = fftw_plan_dft_1d(static_cast<int>(n), fw_in.data(), fw_out.data(), FFTW_FORWARD, FFTW_MEASURE);
    fftw_execute(plan_meas.p);
    const double fftw_meas_us = bench_median_us([&]{ fftw_execute(plan_meas.p); }, reps);

    // --- kmx::fft ---
    tested_engine<double> eng;
    // warm-up + cache fill
    {
        auto s = src;
        const kmx::tensor::view<std::complex<double>> v{std::span<std::complex<double>>(s)};
        eng.transform_1d(v);
    }
    const double kmx_us = bench_median_us([&]
    {
        auto s = src;
        const kmx::tensor::view<std::complex<double>> v{std::span<std::complex<double>>(s)};
        eng.transform_1d(v);
    }, reps);

#if defined(KMX_FFT_ENABLE_AVX2)
    auto plan = eng.make_plan(n);
    {
        auto s = src;
        const kmx::tensor::view<std::complex<double>> v{std::span<std::complex<double>>(s)};
        plan.execute(v);
    }
    const double kmx_plan_us = bench_median_us([&]
    {
        auto s = src;
        const kmx::tensor::view<std::complex<double>> v{std::span<std::complex<double>>(s)};
        plan.execute(v);
    }, reps);

    std::printf("1D double  N=%-6zu  fftw/MEASURE=%7.1f µs  kmx=%7.1f µs  kmx/plan=%7.1f µs  ratio(plan/MEASURE)=%5.1fx\n",
                n, fftw_meas_us, kmx_us, kmx_plan_us, kmx_plan_us / fftw_meas_us);
    return;
#endif

    std::printf("1D double  N=%-6zu  fftw/MEASURE=%7.1f µs  kmx=%7.1f µs  ratio(kmx/MEASURE)=%5.1fx\n",
                n, fftw_meas_us, kmx_us, kmx_us / fftw_meas_us);
}

// -------------------------------------------------------------------------
// 1-D float comparison
// -------------------------------------------------------------------------
void compare_1d_float(const std::size_t n, const int reps)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<std::complex<float>> src(n);
    for (auto& c : src) c = {dist(rng), dist(rng)};

    std::vector<fftwf_complex> fw_in(n), fw_out(n);
    for (std::size_t i = 0; i < n; ++i) { fw_in[i][0] = src[i].real(); fw_in[i][1] = src[i].imag(); }

    FftwPlanF plan_meas;
    plan_meas.p = fftwf_plan_dft_1d(static_cast<int>(n), fw_in.data(), fw_out.data(), FFTW_FORWARD, FFTW_MEASURE);
    fftwf_execute(plan_meas.p);
    const double fftw_meas_us = bench_median_us([&]{ fftwf_execute(plan_meas.p); }, reps);

    tested_engine<float> eng;
    {
        auto s = src;
        const kmx::tensor::view<std::complex<float>> v{std::span<std::complex<float>>(s)};
        eng.transform_1d(v);
    }
    const double kmx_us = bench_median_us([&]
    {
        auto s = src;
        const kmx::tensor::view<std::complex<float>> v{std::span<std::complex<float>>(s)};
        eng.transform_1d(v);
    }, reps);

#if defined(KMX_FFT_ENABLE_AVX2)
    auto plan = eng.make_plan(n);
    {
        auto s = src;
        const kmx::tensor::view<std::complex<float>> v{std::span<std::complex<float>>(s)};
        plan.execute(v);
    }
    const double kmx_plan_us = bench_median_us([&]
    {
        auto s = src;
        const kmx::tensor::view<std::complex<float>> v{std::span<std::complex<float>>(s)};
        plan.execute(v);
    }, reps);

    std::printf("1D float   N=%-6zu  fftw/MEASURE=%7.1f µs  kmx=%7.1f µs  kmx/plan=%7.1f µs  ratio(plan/MEASURE)=%5.1fx\n",
                n, fftw_meas_us, kmx_us, kmx_plan_us, kmx_plan_us / fftw_meas_us);
    return;
#endif

    std::printf("1D float   N=%-6zu  fftw/MEASURE=%7.1f µs  kmx=%7.1f µs  ratio(kmx/MEASURE)=%5.1fx\n",
                n, fftw_meas_us, kmx_us, kmx_us / fftw_meas_us);
}

// -------------------------------------------------------------------------
// 2-D double comparison
// -------------------------------------------------------------------------
void compare_2d_double(const std::size_t rows, const std::size_t cols, const int reps)
{
    const std::size_t n = rows * cols;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<std::complex<double>> src(n);
    for (auto& c : src) c = {dist(rng), dist(rng)};

    std::vector<fftw_complex> fw_in(n), fw_out(n);
    for (std::size_t i = 0; i < n; ++i) { fw_in[i][0] = src[i].real(); fw_in[i][1] = src[i].imag(); }

    FftwPlanD plan_meas;
    plan_meas.p = fftw_plan_dft_2d(static_cast<int>(rows), static_cast<int>(cols),
                                   fw_in.data(), fw_out.data(), FFTW_FORWARD, FFTW_MEASURE);
    fftw_execute(plan_meas.p);
    const double fftw_meas_us = bench_median_us([&]{ fftw_execute(plan_meas.p); }, reps);

    const std::array<std::size_t, 2> shape_arr{rows, cols};
    const kmx::tensor::dimension_shape shape(shape_arr);
    tested_engine<double> eng;
    {
        auto s = src;
        const kmx::tensor::view<std::complex<double>> v{std::span<std::complex<double>>(s), shape};
        eng.transform_2d(v);
    }
    const double kmx_us = bench_median_us([&]
    {
        auto s = src;
        const kmx::tensor::view<std::complex<double>> v{std::span<std::complex<double>>(s), shape};
        eng.transform_2d(v);
    }, reps);

#if defined(KMX_FFT_ENABLE_AVX2)
    auto plan = eng.make_plan_2d(rows, cols);
    {
        auto s = src;
        const kmx::tensor::view<std::complex<double>> v{std::span<std::complex<double>>(s), shape};
        plan.execute(v);
    }
    const double kmx_plan_us = bench_median_us([&]
    {
        auto s = src;
        const kmx::tensor::view<std::complex<double>> v{std::span<std::complex<double>>(s), shape};
        plan.execute(v);
    }, reps);

    std::printf("2D double  %zux%-6zu  fftw/MEASURE=%7.1f µs  kmx=%7.1f µs  kmx/plan=%7.1f µs  ratio(plan/MEASURE)=%5.1fx\n",
                rows, cols, fftw_meas_us, kmx_us, kmx_plan_us, kmx_plan_us / fftw_meas_us);
    return;
#endif

    std::printf("2D double  %zux%-6zu  fftw/MEASURE=%7.1f µs  kmx=%7.1f µs  ratio(kmx/MEASURE)=%5.1fx\n",
                rows, cols, fftw_meas_us, kmx_us, kmx_us / fftw_meas_us);
}

int main(int argc, char** argv)
{
    const bench_options options = parse_args(argc, argv);
    configure_runtime(options);

    std::printf("kmx::fft [%s] vs FFTW3 (v3.3.10) — release build, -O3 -march=native -flto\n", KMX_FFT_ENGINE_NAME);
    std::printf("Median over %d reps. FFTW timings exclude plan construction. mode=%s", options.reps, mode_name(options.mode));
    if (options.threads > 0)
        std::printf(" threads=%d", options.threads);
    if (options.pin_threads)
        std::printf(" pin=on");
    std::printf("\n\n");

    if (options.mode == bench_mode::one_d_double) {
        compare_1d_double(*options.n, options.reps);
        return 0;
    }
    if (options.mode == bench_mode::one_d_float) {
        compare_1d_float(*options.n, options.reps);
        return 0;
    }
    if (options.mode == bench_mode::two_d_double) {
        compare_2d_double(*options.rows, *options.cols, options.reps);
        return 0;
    }

    std::printf("=== 1-D double (pow-2) ===\n");
    for (const std::size_t n : {32, 64, 128, 256, 512, 1024, 4096, 16384, 65536})
        compare_1d_double(n, options.reps);

    std::printf("\n=== 1-D double (non-power-of-2 / Bluestein) ===\n");
    for (const std::size_t n : {100, 1000, 1500})
        compare_1d_double(n, options.reps);

    std::printf("\n=== 1-D float (pow-2) ===\n");
    for (const std::size_t n : {1024, 4096, 16384})
        compare_1d_float(n, options.reps);

    std::printf("\n=== 2-D double ===\n");
    for (const auto [r, c] : std::array<std::pair<std::size_t,std::size_t>, 7>{{{32,32},{64,64},{128,128},{128,256},{256,128},{256,256},{512,512}}})
        compare_2d_double(r, c, options.reps);

    return 0;
}

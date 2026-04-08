// Copyright (c) 2026 - present KMX Systems. All rights reserved.

#include <array>
#include <complex>
#include <immintrin.h>
#include <numbers>
#include <span>

#include <kmx/fft_backend/avx2.hpp>

namespace kmx::fft::backend::avx2_detail {
namespace {

constexpr double kInvSqrt2 = 0.707106781186547524400844362104849039;
constexpr double kCosPi8   = 0.923879532511286756128183189396788286;
constexpr double kSinPi8   = 0.382683432365089771728459984030398867;
constexpr double kCos2Pi5  = 0.309016994374947424102293417182819059;
constexpr double kCos4Pi5  = -0.809016994374947424102293417182819059;
constexpr double kSin2Pi5  = 0.951056516295153572116439333379382143;
constexpr double kSin4Pi5  = 0.587785252292473129168705954639072769;

[[nodiscard]] inline std::complex<double> imag_rot(const std::complex<double>& z, double imag_sign) noexcept {
    return {-imag_sign * z.imag(), imag_sign * z.real()};
}

// FMA complex multiply for two complex doubles packed in one __m256d
[[nodiscard]] inline __m256d cmul_pd_local(__m256d a, __m256d b) noexcept {
    __m256d a_re   = _mm256_unpacklo_pd(a, a);
    __m256d a_im   = _mm256_unpackhi_pd(a, a);
    __m256d b_shuf = _mm256_shuffle_pd(b, b, 0x5);
    return _mm256_fmaddsub_pd(a_re, b, _mm256_mul_pd(a_im, b_shuf));
}

void fft4_core(std::complex<double>* x, bool inverse) noexcept {
    const std::complex<double> a0 = x[0] + x[2];
    const std::complex<double> a1 = x[0] - x[2];
    const std::complex<double> a2 = x[1] + x[3];
    const std::complex<double> d  = x[1] - x[3];
    const std::complex<double> a3 = inverse
        ? std::complex<double>(d.imag(), -d.real())
        : std::complex<double>(-d.imag(), d.real());
    x[0] = a0 + a2;
    x[1] = a1 + a3;
    x[2] = a0 - a2;
    x[3] = a1 - a3;
}

template <std::size_t N, bool Inverse>
[[nodiscard]] const std::array<std::complex<double>, N / 2>& fixed_twiddles() noexcept {
    static const std::array<std::complex<double>, N / 2> table = [] {
        std::array<std::complex<double>, N / 2> out{};
        constexpr double sign = Inverse ? 1.0 : -1.0;
        constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;
        for (std::size_t k = 0; k < N / 2; ++k) {
            const double theta = two_pi * static_cast<double>(k) / static_cast<double>(N);
            out[k] = {std::cos(theta), sign * std::sin(theta)};
        }
        return out;
    }();
    return table;
}

template <std::size_t N, bool Inverse>
[[nodiscard]] const std::array<std::complex<double>, N>& full_twiddles() noexcept {
    static const std::array<std::complex<double>, N> table = [] {
        std::array<std::complex<double>, N> out{};
        constexpr double sign = Inverse ? 1.0 : -1.0;
        constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;
        for (std::size_t k = 0; k < N; ++k) {
            const double theta = two_pi * static_cast<double>(k) / static_cast<double>(N);
            out[k] = {std::cos(theta), sign * std::sin(theta)};
        }
        return out;
    }();
    return table;
}

template <std::size_t N>
void codelet_fixed_core(std::complex<double>* x, bool inverse) noexcept {
    if constexpr (N == 2) {
        const std::complex<double> a = x[0];
        const std::complex<double> b = x[1];
        x[0] = a + b;
        x[1] = a - b;
    } else if constexpr (N == 4) {
        fft4_core(x, !inverse);
    } else {
        std::array<std::complex<double>, N / 2> even{};
        std::array<std::complex<double>, N / 2> odd{};

        // AVX2 deinterleave: even[i] = x[2i], odd[i] = x[2i+1]
        // Process 2 complex-double pairs at a time (4 complex doubles = 8 doubles = 2 AVX2 loads)
        {
            const double* xp = reinterpret_cast<const double*>(x);
            double* ep = reinterpret_cast<double*>(even.data());
            double* op = reinterpret_cast<double*>(odd.data());
            std::size_t i = 0;
            for (; i + 1 < N / 2; i += 2) {
                // v0 = [x[2i].re, x[2i].im, x[2i+1].re, x[2i+1].im]
                // v1 = [x[2i+2].re, x[2i+2].im, x[2i+3].re, x[2i+3].im]
                __m256d v0 = _mm256_loadu_pd(xp + i * 4);
                __m256d v1 = _mm256_loadu_pd(xp + i * 4 + 4);
                // Deinterleave: even = low 128-bit halves of v0, v1; odd = high halves
                __m256d ev = _mm256_permute2f128_pd(v0, v1, 0x20);
                __m256d od = _mm256_permute2f128_pd(v0, v1, 0x31);
                _mm256_storeu_pd(ep + i * 2, ev);
                _mm256_storeu_pd(op + i * 2, od);
            }
            for (; i < N / 2; ++i) {
                even[i] = x[2 * i];
                odd[i]  = x[2 * i + 1];
            }
        }

        codelet_fixed_core<N / 2>(even.data(), inverse);
        codelet_fixed_core<N / 2>(odd.data(), inverse);

        const auto& tw = inverse ? fixed_twiddles<N, true>() : fixed_twiddles<N, false>();

        // AVX2 butterfly combine: x[k] = even[k] + odd[k]*tw[k]
        {
            const double* ep  = reinterpret_cast<const double*>(even.data());
            const double* op  = reinterpret_cast<const double*>(odd.data());
            const double* tp  = reinterpret_cast<const double*>(tw.data());
            double* xlo = reinterpret_cast<double*>(x);
            double* xhi = reinterpret_cast<double*>(x + N / 2);
            constexpr std::size_t half = N / 2;
            std::size_t k = 0;
            for (; k + 1 < half; k += 2) {
                __m256d e  = _mm256_loadu_pd(ep + k * 2);
                __m256d o  = _mm256_loadu_pd(op + k * 2);
                __m256d tw2 = _mm256_loadu_pd(tp + k * 2);
                __m256d t  = cmul_pd_local(o, tw2);
                _mm256_storeu_pd(xlo + k * 2, _mm256_add_pd(e, t));
                _mm256_storeu_pd(xhi + k * 2, _mm256_sub_pd(e, t));
            }
            for (; k < half; ++k) {
                const std::complex<double> t = odd[k] * tw[k];
                x[k]          = even[k] + t;
                x[k + half]   = even[k] - t;
            }
        }
    }
}

template <std::size_t N>
void codelet_fixed(double* __restrict__ d, bool inverse) noexcept {
    auto* x = reinterpret_cast<std::complex<double>*>(d);
    codelet_fixed_core<N>(x, inverse);
    if (inverse) {
        const double scale = 1.0 / static_cast<double>(N);
        for (auto& v : std::span<std::complex<double>>(x, N)) v *= scale;
    }
}

void codelet_3_impl(std::complex<double>* x, bool inverse) noexcept {
    const auto& tw = inverse ? full_twiddles<3, true>() : full_twiddles<3, false>();
    const std::complex<double> y0 = x[0] + x[1] + x[2];
    const std::complex<double> y1 = x[0] + x[1] * tw[1] + x[2] * tw[2];
    const std::complex<double> y2 = x[0] + x[1] * tw[2] + x[2] * tw[1];
    x[0] = y0;
    x[1] = y1;
    x[2] = y2;
    if (inverse) {
        for (auto& v : std::span<std::complex<double>>(x, 3)) v *= (1.0 / 3.0);
    }
}

void codelet_5_impl(std::complex<double>* x, bool inverse) noexcept {
    const auto& tw = inverse ? full_twiddles<5, true>() : full_twiddles<5, false>();
    const std::complex<double> y0 = x[0] + x[1] + x[2] + x[3] + x[4];
    const std::complex<double> y1 = x[0] + x[1] * tw[1] + x[2] * tw[2] + x[3] * tw[3] + x[4] * tw[4];
    const std::complex<double> y2 = x[0] + x[1] * tw[2] + x[2] * tw[4] + x[3] * tw[1] + x[4] * tw[3];
    const std::complex<double> y3 = x[0] + x[1] * tw[3] + x[2] * tw[1] + x[3] * tw[4] + x[4] * tw[2];
    const std::complex<double> y4 = x[0] + x[1] * tw[4] + x[2] * tw[3] + x[3] * tw[2] + x[4] * tw[1];
    x[0] = y0;
    x[1] = y1;
    x[2] = y2;
    x[3] = y3;
    x[4] = y4;
    if (inverse) {
        for (auto& v : std::span<std::complex<double>>(x, 5)) v *= 0.2;
    }
}

template <std::size_t N>
void smooth_fixed_core(std::complex<double>* x, bool inverse) noexcept {
    if constexpr (N == 1) {
        return;
    } else if constexpr (N == 2) {
        codelet_2(reinterpret_cast<double*>(x), inverse);
    } else if constexpr (N == 3) {
        codelet_3_impl(x, inverse);
    } else if constexpr (N == 4) {
        codelet_4(reinterpret_cast<double*>(x), inverse);
    } else if constexpr (N == 5) {
        codelet_5_impl(x, inverse);
    } else if constexpr (N == 8) {
        codelet_8(reinterpret_cast<double*>(x), inverse);
    } else if constexpr (N == 16) {
        codelet_16(reinterpret_cast<double*>(x), inverse);
    } else {
        constexpr std::size_t radix = (N % 5u) == 0u ? 5u : ((N % 3u) == 0u ? 3u : 2u);
        constexpr std::size_t M = N / radix;
        // Reuse per-thread scratch to avoid repeated large stack allocation/zero-fill
        // in special smooth kernels (notably N=1000 and N=1500).
        static thread_local std::array<std::complex<double>, N> scratch;

        if constexpr (radix == 2u) {
            for (std::size_t q = 0; q < M; ++q) {
                scratch[q] = x[q * 2u];
                scratch[M + q] = x[q * 2u + 1u];
            }
        } else if constexpr (radix == 3u) {
            for (std::size_t q = 0; q < M; ++q) {
                scratch[q] = x[q * 3u];
                scratch[M + q] = x[q * 3u + 1u];
                scratch[2u * M + q] = x[q * 3u + 2u];
            }
        } else {
            for (std::size_t q = 0; q < M; ++q) {
                scratch[q] = x[q * 5u];
                scratch[M + q] = x[q * 5u + 1u];
                scratch[2u * M + q] = x[q * 5u + 2u];
                scratch[3u * M + q] = x[q * 5u + 3u];
                scratch[4u * M + q] = x[q * 5u + 4u];
            }
        }

        for (std::size_t j = 0; j < radix; ++j)
            smooth_fixed_core<M>(scratch.data() + j * M, inverse);

        const auto& tw_n = inverse ? full_twiddles<N, true>() : full_twiddles<N, false>();
        const double inv_radix = 1.0 / static_cast<double>(radix);

        if constexpr (radix == 2u) {
            const auto* s0 = scratch.data();
            const auto* s1 = scratch.data() + M;
            for (std::size_t q = 0; q < M; ++q) {
                const std::complex<double> v0 = s0[q];
                const std::complex<double> v1 = s1[q] * tw_n[q];
                const std::complex<double> y0 = v0 + v1;
                const std::complex<double> y1 = v0 - v1;
                x[q] = inverse ? y0 * inv_radix : y0;
                x[q + M] = inverse ? y1 * inv_radix : y1;
            }
        } else if constexpr (radix == 3u) {
            const auto& tw_r = inverse ? full_twiddles<3, true>() : full_twiddles<3, false>();
            const auto* s0 = scratch.data();
            const auto* s1 = scratch.data() + M;
            const auto* s2 = scratch.data() + 2u * M;
            for (std::size_t q = 0; q < M; ++q) {
                const std::complex<double> v0 = s0[q];
                const std::complex<double> v1 = s1[q] * tw_n[q];
                const std::complex<double> v2 = s2[q] * tw_n[2u * q];
                const std::complex<double> y0 = v0 + v1 + v2;
                const std::complex<double> y1 = v0 + v1 * tw_r[1] + v2 * tw_r[2];
                const std::complex<double> y2 = v0 + v1 * tw_r[2] + v2 * tw_r[1];
                x[q] = inverse ? y0 * inv_radix : y0;
                x[q + M] = inverse ? y1 * inv_radix : y1;
                x[q + 2u * M] = inverse ? y2 * inv_radix : y2;
            }
        } else {
            const auto* s0 = scratch.data();
            const auto* s1 = scratch.data() + M;
            const auto* s2 = scratch.data() + 2u * M;
            const auto* s3 = scratch.data() + 3u * M;
            const auto* s4 = scratch.data() + 4u * M;
            const double imag_sign = inverse ? 1.0 : -1.0;
            for (std::size_t q = 0; q < M; ++q) {
                const std::complex<double> v0 = s0[q];
                const std::complex<double> v1 = s1[q] * tw_n[q];
                const std::complex<double> v2 = s2[q] * tw_n[2u * q];
                const std::complex<double> v3 = s3[q] * tw_n[3u * q];
                const std::complex<double> v4 = s4[q] * tw_n[4u * q];
                const std::complex<double> a14 = v1 + v4;
                const std::complex<double> d14 = v1 - v4;
                const std::complex<double> a23 = v2 + v3;
                const std::complex<double> d23 = v2 - v3;

                const std::complex<double> base14 = a14 * kCos2Pi5 + a23 * kCos4Pi5;
                const std::complex<double> base23 = a14 * kCos4Pi5 + a23 * kCos2Pi5;
                const std::complex<double> u1 = d14 * kSin2Pi5 + d23 * kSin4Pi5;
                const std::complex<double> u2 = d14 * kSin4Pi5 - d23 * kSin2Pi5;
                const std::complex<double> r1 = imag_rot(u1, imag_sign);
                const std::complex<double> r2 = imag_rot(u2, imag_sign);

                const std::complex<double> y0 = v0 + a14 + a23;
                const std::complex<double> y1 = v0 + base14 + r1;
                const std::complex<double> y2 = v0 + base23 + r2;
                const std::complex<double> y3 = v0 + base23 - r2;
                const std::complex<double> y4 = v0 + base14 - r1;
                x[q] = inverse ? y0 * inv_radix : y0;
                x[q + M] = inverse ? y1 * inv_radix : y1;
                x[q + 2u * M] = inverse ? y2 * inv_radix : y2;
                x[q + 3u * M] = inverse ? y3 * inv_radix : y3;
                x[q + 4u * M] = inverse ? y4 * inv_radix : y4;
            }
        }
    }
}

} // namespace

void codelet_2(double* __restrict__ d, bool inverse) noexcept {
    const double r0 = d[0], i0 = d[1], r1 = d[2], i1 = d[3];
    if (!inverse) {
        d[0]=r0+r1; d[1]=i0+i1; d[2]=r0-r1; d[3]=i0-i1;
    } else {
        d[0]=(r0+r1)*0.5; d[1]=(i0+i1)*0.5; d[2]=(r0-r1)*0.5; d[3]=(i0-i1)*0.5;
    }
}

void codelet_4(double* __restrict__ d, bool inverse) noexcept {
    double r0=d[0],i0=d[1],r1=d[2],i1=d[3],r2=d[4],i2=d[5],r3=d[6],i3=d[7];
    double a0=r0+r2,b0=i0+i2,a1=r0-r2,b1=i0-i2;
    double a2=r1+r3,b2=i1+i3,a3=r1-r3,b3=i1-i3;
    if (!inverse) {
        d[0]=a0+a2; d[1]=b0+b2;
        d[2]=a1+b3; d[3]=b1-a3;
        d[4]=a0-a2; d[5]=b0-b2;
        d[6]=a1-b3; d[7]=b1+a3;
    } else {
        d[0]=(a0+a2)*0.25; d[1]=(b0+b2)*0.25;
        d[2]=(a1-b3)*0.25; d[3]=(b1+a3)*0.25;
        d[4]=(a0-a2)*0.25; d[5]=(b0-b2)*0.25;
        d[6]=(a1+b3)*0.25; d[7]=(b1-a3)*0.25;
    }
}

void codelet_8(double* __restrict__ d, bool inverse) noexcept {
    auto* x = reinterpret_cast<std::complex<double>*>(d);
    std::array<std::complex<double>, 4> even{x[0], x[2], x[4], x[6]};
    std::array<std::complex<double>, 4> odd{x[1], x[3], x[5], x[7]};
    fft4_core(even.data(), !inverse);
    fft4_core(odd.data(), !inverse);

    const double imag_sign = inverse ? 1.0 : -1.0;
    const std::array<std::complex<double>, 4> tw{
        std::complex<double>(1.0, 0.0),
        std::complex<double>(kInvSqrt2, imag_sign * kInvSqrt2),
        std::complex<double>(0.0, imag_sign),
        std::complex<double>(-kInvSqrt2, imag_sign * kInvSqrt2),
    };

    for (std::size_t k = 0; k < 4; ++k) {
        const std::complex<double> t = odd[k] * tw[k];
        x[k] = even[k] + t;
        x[k + 4] = even[k] - t;
    }

    if (inverse) {
        for (auto& v : std::span<std::complex<double>>(x, 8)) v *= 0.125;
    }
}

void codelet_16(double* __restrict__ d, bool inverse) noexcept {
    auto* x = reinterpret_cast<std::complex<double>*>(d);
    std::array<std::complex<double>, 8> even{};
    std::array<std::complex<double>, 8> odd{};
    for (std::size_t i = 0; i < 8; ++i) {
        even[i] = x[2 * i];
        odd[i] = x[2 * i + 1];
    }

    codelet_8(reinterpret_cast<double*>(even.data()), inverse);
    codelet_8(reinterpret_cast<double*>(odd.data()), inverse);
    if (inverse) {
        for (auto& v : even) v *= 8.0;
        for (auto& v : odd) v *= 8.0;
    }

    const double imag_sign = inverse ? 1.0 : -1.0;
    const std::array<std::complex<double>, 8> tw{
        std::complex<double>(1.0, 0.0),
        std::complex<double>(kCosPi8, imag_sign * kSinPi8),
        std::complex<double>(kInvSqrt2, imag_sign * kInvSqrt2),
        std::complex<double>(kSinPi8, imag_sign * kCosPi8),
        std::complex<double>(0.0, imag_sign),
        std::complex<double>(-kSinPi8, imag_sign * kCosPi8),
        std::complex<double>(-kInvSqrt2, imag_sign * kInvSqrt2),
        std::complex<double>(-kCosPi8, imag_sign * kSinPi8),
    };

    for (std::size_t k = 0; k < 8; ++k) {
        const std::complex<double> t = odd[k] * tw[k];
        x[k] = even[k] + t;
        x[k + 8] = even[k] - t;
    }

    if (inverse) {
        for (auto& v : std::span<std::complex<double>>(x, 16)) v *= 0.0625;
    }
}

void codelet_32(double* __restrict__ d, bool inverse) noexcept {
    codelet_fixed<32>(d, inverse);
}

void codelet_64(double* __restrict__ d, bool inverse) noexcept {
    codelet_fixed<64>(d, inverse);
}

void codelet_128(double* __restrict__ d, bool inverse) noexcept {
    codelet_fixed<128>(d, inverse);
}

void codelet_256(double* __restrict__ d, bool inverse) noexcept {
    codelet_fixed<256>(d, inverse);
}

void codelet_512(double* __restrict__ d, bool inverse) noexcept {
    codelet_fixed<512>(d, inverse);
}

void codelet_1024(double* __restrict__ d, bool inverse) noexcept {
    using cd = std::complex<double>;
    auto* x = reinterpret_cast<cd*>(d);

    // Thread-local workspace avoids 2×8 KB top-level stack allocation
    // (safe: codelet_fixed_core does not call back into codelet_1024)
    static thread_local std::array<cd, 512> even_buf, odd_buf;

    // SIMD deinterleave: even_buf[i] = x[2i], odd_buf[i] = x[2i+1]
    {
        const double* xp = reinterpret_cast<const double*>(x);
        double* ep = reinterpret_cast<double*>(even_buf.data());
        double* op = reinterpret_cast<double*>(odd_buf.data());
        for (std::size_t i = 0; i < 512; i += 2) {
            __m256d v0 = _mm256_loadu_pd(xp + i * 4);
            __m256d v1 = _mm256_loadu_pd(xp + i * 4 + 4);
            _mm256_storeu_pd(ep + i * 2, _mm256_permute2f128_pd(v0, v1, 0x20));
            _mm256_storeu_pd(op + i * 2, _mm256_permute2f128_pd(v0, v1, 0x31));
        }
    }

    codelet_fixed_core<512>(even_buf.data(), inverse);
    codelet_fixed_core<512>(odd_buf.data(), inverse);

    // SIMD butterfly combine using precomputed twiddles
    const auto& tw = inverse ? fixed_twiddles<1024, true>() : fixed_twiddles<1024, false>();
    {
        const double* ep  = reinterpret_cast<const double*>(even_buf.data());
        const double* op  = reinterpret_cast<const double*>(odd_buf.data());
        const double* tp  = reinterpret_cast<const double*>(tw.data());
        double* xlo = reinterpret_cast<double*>(x);
        double* xhi = reinterpret_cast<double*>(x + 512);
        for (std::size_t k = 0; k < 512; k += 2) {
            __m256d e   = _mm256_loadu_pd(ep + k * 2);
            __m256d o   = _mm256_loadu_pd(op + k * 2);
            __m256d tw2 = _mm256_loadu_pd(tp + k * 2);
            __m256d t   = cmul_pd_local(o, tw2);
            _mm256_storeu_pd(xlo + k * 2, _mm256_add_pd(e, t));
            _mm256_storeu_pd(xhi + k * 2, _mm256_sub_pd(e, t));
        }
    }

    if (inverse) {
        const __m256d s = _mm256_set1_pd(1.0 / 1024.0);
        double* xp = d;
        for (std::size_t i = 0; i < 1024; i += 2, xp += 4)
            _mm256_storeu_pd(xp, _mm256_mul_pd(_mm256_loadu_pd(xp), s));
    }
}

void smooth_100(double* __restrict__ d, bool inverse) noexcept {
    smooth_fixed_core<100>(reinterpret_cast<std::complex<double>*>(d), inverse);
}

void smooth_1000(double* __restrict__ d, bool inverse) noexcept {
    smooth_fixed_core<1000>(reinterpret_cast<std::complex<double>*>(d), inverse);
}

void smooth_1500(double* __restrict__ d, bool inverse) noexcept {
    smooth_fixed_core<1500>(reinterpret_cast<std::complex<double>*>(d), inverse);
}

} // namespace kmx::fft::backend::avx2_detail
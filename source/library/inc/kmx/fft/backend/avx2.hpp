// Copyright (c) 2026 - present KMX Systems. All rights reserved.
/// @file avx2.hpp
/// @brief AVX2 + FMA + OpenMP accelerated FFT backend for kmx::fft.
/// Techniques used:
///   - Plan object: pre-allocated aligned work buffer, pre-packed SIMD twiddle table
///   - Zero-allocation hot path via plan::execute()
///   - FMA complex multiply (fmaddsub)
///   - Small-N codelets for N=2,4,8 (zero twiddle multiplications)
///   - Stockham ping-pong (eliminates bit-reversal)
///   - Mixed radix-4 / radix-2 butterflies
///   - Six-step cache-oblivious large-N FFT with OpenMP parallelism
///   - Tiled 2D column pass with explicit software prefetch
///   - AVX2 path used inside Bluestein for non-pow-2 sizes
#pragma once
#ifndef PCH
    #include <immintrin.h>
    #include <omp.h>
    #include <array>
    #include <complex>
    #include <vector>
    #include <concepts>
    #include <cmath>
    #include <bit>
    #include <numbers>
    #include <memory>
    #include <mutex>
    #include <shared_mutex>
    #include <unordered_map>
    #include <kmx/tensor.hpp>
#endif

#include "concepts.hpp"
#include "software.hpp"

namespace kmx::fft::backend {

namespace avx2_detail {
void codelet_2(double* d, bool inverse) noexcept;
void codelet_4(double* d, bool inverse) noexcept;
void codelet_8(double* d, bool inverse) noexcept;
void codelet_16(double* d, bool inverse) noexcept;
void codelet_32(double* d, bool inverse) noexcept;
void codelet_64(double* d, bool inverse) noexcept;
void codelet_128(double* d, bool inverse) noexcept;
void codelet_256(double* d, bool inverse) noexcept;
void codelet_512(double* d, bool inverse) noexcept;
void codelet_1024(double* d, bool inverse) noexcept;
void prime_small_kernel_selector(std::size_t n, bool inverse) noexcept;
void smooth_100(double* d, bool inverse) noexcept;
void smooth_1000(double* d, bool inverse) noexcept;
void smooth_1500(double* d, bool inverse) noexcept;
}

// 32-byte aligned allocator
template <typename T>
struct aligned32_allocator {
    using value_type = T;
    T* allocate(std::size_t n) {
        void* p = nullptr;
        if (posix_memalign(&p, 32, n * sizeof(T)) != 0)
            throw std::bad_alloc{};
        return static_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t) noexcept { free(p); }
    bool operator==(const aligned32_allocator&) const noexcept { return true; }
    bool operator!=(const aligned32_allocator&) const noexcept { return false; }
};

template <ComplexNumber _ComplexNumber>
class avx2 {
public:
    using allocator_type = aligned32_allocator<_ComplexNumber>;
    using value_type     = typename _ComplexNumber::value_type;

    using aligned_vec = std::vector<_ComplexNumber, aligned32_allocator<_ComplexNumber>>;

    struct plan_2d;

private:
    software<_ComplexNumber> fallback_engine_;

    static constexpr bool is_complex_double = std::same_as<_ComplexNumber, std::complex<double>>;
    static constexpr bool is_complex_float  = std::same_as<_ComplexNumber, std::complex<float>>;

    // Parallelise stages when total transform size exceeds this
    static constexpr std::size_t kParallelThresh = 16384;
    // Use six-step above this size
    static constexpr std::size_t kSixStepThresh  = 16384;
    // For smaller 2D sizes the software backend still wins on overhead.
    static constexpr std::size_t k2DAvx2Thresh = 16384;
    // Tile width for 2D column pass (number of columns processed together)
    static constexpr std::size_t kColTile = 8;

    // FMA complex multiply
    [[nodiscard]] static __m256d cmul_pd(__m256d a, __m256d b) noexcept {
        __m256d a_re   = _mm256_unpacklo_pd(a, a);
        __m256d a_im   = _mm256_unpackhi_pd(a, a);
        __m256d b_shuf = _mm256_shuffle_pd(b, b, 0x5);
        return _mm256_fmaddsub_pd(a_re, b, _mm256_mul_pd(a_im, b_shuf));
    }
    [[nodiscard]] static __m256 cmul_ps(__m256 a, __m256 b) noexcept {
        __m256 a_re   = _mm256_moveldup_ps(a);
        __m256 a_im   = _mm256_movehdup_ps(a);
        __m256 b_shuf = _mm256_shuffle_ps(b, b, 0xB1);
        return _mm256_fmaddsub_ps(a_re, b, _mm256_mul_ps(a_im, b_shuf));
    }

    // Pre-packed twiddle table
    // For each stage s (where stage_len = 1<<s), we pack twiddles contiguously
    // in the order they are accessed: tw_packed[s][k] = W^(k * (n >> (s+1)))
    // This eliminates all strided loads from the hot butterfly loops.
    struct packed_twiddles {
        // per-stage packed tables; stage s covers half_len = 1<<s twiddles
        std::vector<aligned_vec> stages;  // stages[s] has (1<<s) entries
        std::size_t n = 0;
    };

    static packed_twiddles make_packed_twiddles(std::size_t n, bool inverse) {
        packed_twiddles pt;
        pt.n = n;
        const int log2n = std::countr_zero(n);
        pt.stages.resize(log2n);
        const value_type sign = inverse ? value_type(1) : value_type(-1);
        const value_type two_pi_over_n = value_type(2) * std::numbers::pi_v<value_type> * sign / static_cast<value_type>(n);
        for (int s = 0; s < log2n; ++s) {
            const std::size_t half_len = std::size_t(1) << s;
            const std::size_t tw_step  = n >> (s + 1);   // = n / (2 * half_len)
            pt.stages[s].resize(half_len);
            for (std::size_t k = 0; k < half_len; ++k) {
                const value_type theta = two_pi_over_n * static_cast<value_type>(k * tw_step);
                pt.stages[s][k] = _ComplexNumber(std::cos(theta), std::sin(theta));
            }
        }
        return pt;
    }

    // Global packed-twiddle cache keyed by (n, inverse)
    struct ptw_cache_t {
        mutable std::shared_mutex mtx;
        std::unordered_map<std::size_t, packed_twiddles> fwd, inv;

        const packed_twiddles& get(std::size_t n, bool inverse) {
            auto& map = inverse ? inv : fwd;
            { std::shared_lock lk(mtx); auto it = map.find(n); if (it != map.end()) return it->second; }
            auto pt = make_packed_twiddles(n, inverse);
            std::unique_lock lk(mtx);
            auto [it, _] = map.emplace(n, std::move(pt));
            return it->second;
        }
    } ptw_cache_;

    // Simple full-table twiddle cache (for six-step twiddle multiply step)
    struct full_tw_cache_t {
        mutable std::shared_mutex mtx;
        std::unordered_map<std::size_t, aligned_vec> fwd, inv;
        const aligned_vec& get(std::size_t n, bool inverse) {
            auto& map = inverse ? inv : fwd;
            { std::shared_lock lk(mtx); auto it = map.find(n); if (it != map.end()) return it->second; }
            aligned_vec tw(n);
            const value_type sign = inverse ? value_type(1) : value_type(-1);
            const value_type base = value_type(2) * std::numbers::pi_v<value_type> * sign / static_cast<value_type>(n);
            for (std::size_t k = 0; k < n; ++k) {
                const value_type t = base * static_cast<value_type>(k);
                tw[k] = _ComplexNumber(std::cos(t), std::sin(t));
            }
            std::unique_lock lk(mtx);
            auto [it, _] = map.emplace(n, std::move(tw));
            return it->second;
        }
    } full_tw_cache_;

    struct bluestein_cache_entry {
        std::size_t m = 0;
        aligned_vec chirp;
        aligned_vec h_fft;
    };

    struct bluestein_cache_t {
        mutable std::shared_mutex mtx;
        std::unordered_map<std::size_t, bluestein_cache_entry> fwd, inv;

        const bluestein_cache_entry& get(avx2& engine, std::size_t n, bool inverse) {
            auto& map = inverse ? inv : fwd;
            {
                std::shared_lock lk(mtx);
                auto it = map.find(n);
                if (it != map.end()) return it->second;
            }

            bluestein_cache_entry entry;
            entry.m = std::bit_ceil(2 * n - 1);
            entry.chirp.resize(n);

            const value_type sign = inverse ? value_type(1) : value_type(-1);
            const value_type pi_over_n = std::numbers::pi_v<value_type> / static_cast<value_type>(n);
            for (std::size_t k = 0; k < n; ++k) {
                const std::size_t k2 = (k * k) % (2 * n);
                const value_type theta = sign * pi_over_n * static_cast<value_type>(k2);
                entry.chirp[k] = _ComplexNumber(std::cos(theta), std::sin(theta));
            }

            aligned_vec h_pad(entry.m, _ComplexNumber{});
            h_pad[0] = std::conj(entry.chirp[0]);
            for (std::size_t k = 1; k < n; ++k) {
                h_pad[k] = std::conj(entry.chirp[k]);
                h_pad[entry.m - k] = std::conj(entry.chirp[k]);
            }

            aligned_vec work;
            const auto& pt_m = engine.ptw_cache_.get(entry.m, false);
            engine.execute_pow2_raw(h_pad.data(), entry.m, work, pt_m, false);
            entry.h_fft = std::move(h_pad);

            std::unique_lock lk(mtx);
            auto [it, _] = map.emplace(n, std::move(entry));
            return it->second;
        }
    } bluestein_cache_;

    // Persistent member work buffer
    mutable aligned_vec work_buf_;
    mutable aligned_vec bluestein_buf_;

    enum class smooth_plan_kind : std::uint8_t {
        pow2,
        radix2,
        radix3,
        radix5,
        special100,
        special1000,
        special1500,
    };

    struct smooth_plan_node {
        std::size_t n = 0;
        std::size_t m = 0;
        smooth_plan_kind kind = smooth_plan_kind::radix2;
        const packed_twiddles* pt = nullptr;
        const aligned_vec* tw_n = nullptr;
        const aligned_vec* tw_r = nullptr;
        std::array<std::shared_ptr<smooth_plan_node>, 5> children{};

        void execute(avx2& engine, _ComplexNumber* data, _ComplexNumber* scratch, bool inverse) const {
            switch (kind) {
            case smooth_plan_kind::special100:
                avx2_detail::smooth_100(reinterpret_cast<double*>(data), inverse);
                return;
            case smooth_plan_kind::special1000:
                avx2_detail::smooth_1000(reinterpret_cast<double*>(data), inverse);
                return;
            case smooth_plan_kind::special1500:
                avx2_detail::smooth_1500(reinterpret_cast<double*>(data), inverse);
                return;
            case smooth_plan_kind::pow2:
                engine.execute_pow2_presized(data, n, scratch, *pt, inverse);
                return;
            case smooth_plan_kind::radix2:
            case smooth_plan_kind::radix3:
            case smooth_plan_kind::radix5:
                break;
            }

            const std::size_t radix = kind == smooth_plan_kind::radix2 ? 2u : (kind == smooth_plan_kind::radix3 ? 3u : 5u);
            if (radix == 2u) {
                for (std::size_t q = 0; q < m; ++q) {
                    scratch[q] = data[q * 2u];
                    scratch[m + q] = data[q * 2u + 1u];
                }
            } else if (radix == 3u) {
                for (std::size_t q = 0; q < m; ++q) {
                    scratch[q] = data[q * 3u];
                    scratch[m + q] = data[q * 3u + 1u];
                    scratch[2u * m + q] = data[q * 3u + 2u];
                }
            } else {
                for (std::size_t q = 0; q < m; ++q) {
                    scratch[q] = data[q * 5u];
                    scratch[m + q] = data[q * 5u + 1u];
                    scratch[2u * m + q] = data[q * 5u + 2u];
                    scratch[3u * m + q] = data[q * 5u + 3u];
                    scratch[4u * m + q] = data[q * 5u + 4u];
                }
            }

            for (std::size_t j = 0; j < radix; ++j)
                children[j]->execute(engine, scratch + j * m, data + j * m, inverse);

            const value_type inv_radix = value_type(1) / static_cast<value_type>(radix);
            if (radix == 2u) {
                const _ComplexNumber* s0 = scratch;
                const _ComplexNumber* s1 = scratch + m;
                for (std::size_t q = 0; q < m; ++q) {
                    const _ComplexNumber v0 = s0[q];
                    const _ComplexNumber v1 = s1[q] * (*tw_n)[q];
                    const _ComplexNumber y0 = v0 + v1;
                    const _ComplexNumber y1 = v0 - v1;
                    data[q] = inverse ? y0 * inv_radix : y0;
                    data[q + m] = inverse ? y1 * inv_radix : y1;
                }
                return;
            }

            if (radix == 3u) {
                const _ComplexNumber* s0 = scratch;
                const _ComplexNumber* s1 = scratch + m;
                const _ComplexNumber* s2 = scratch + 2u * m;
                for (std::size_t q = 0; q < m; ++q) {
                    const _ComplexNumber v0 = s0[q];
                    const _ComplexNumber v1 = s1[q] * (*tw_n)[q];
                    const _ComplexNumber v2 = s2[q] * (*tw_n)[2u * q];
                    const _ComplexNumber y0 = v0 + v1 + v2;
                    const _ComplexNumber y1 = v0 + v1 * (*tw_r)[1] + v2 * (*tw_r)[2];
                    const _ComplexNumber y2 = v0 + v1 * (*tw_r)[2] + v2 * (*tw_r)[1];
                    data[q] = inverse ? y0 * inv_radix : y0;
                    data[q + m] = inverse ? y1 * inv_radix : y1;
                    data[q + 2u * m] = inverse ? y2 * inv_radix : y2;
                }
                return;
            }

            const _ComplexNumber* s0 = scratch;
            const _ComplexNumber* s1 = scratch + m;
            const _ComplexNumber* s2 = scratch + 2u * m;
            const _ComplexNumber* s3 = scratch + 3u * m;
            const _ComplexNumber* s4 = scratch + 4u * m;
            for (std::size_t q = 0; q < m; ++q) {
                const _ComplexNumber v0 = s0[q];
                const _ComplexNumber v1 = s1[q] * (*tw_n)[q];
                const _ComplexNumber v2 = s2[q] * (*tw_n)[2u * q];
                const _ComplexNumber v3 = s3[q] * (*tw_n)[3u * q];
                const _ComplexNumber v4 = s4[q] * (*tw_n)[4u * q];
                const _ComplexNumber y0 = v0 + v1 + v2 + v3 + v4;
                const _ComplexNumber y1 = v0 + v1 * (*tw_r)[1] + v2 * (*tw_r)[2] + v3 * (*tw_r)[3] + v4 * (*tw_r)[4];
                const _ComplexNumber y2 = v0 + v1 * (*tw_r)[2] + v2 * (*tw_r)[4] + v3 * (*tw_r)[1] + v4 * (*tw_r)[3];
                const _ComplexNumber y3 = v0 + v1 * (*tw_r)[3] + v2 * (*tw_r)[1] + v3 * (*tw_r)[4] + v4 * (*tw_r)[2];
                const _ComplexNumber y4 = v0 + v1 * (*tw_r)[4] + v2 * (*tw_r)[3] + v3 * (*tw_r)[2] + v4 * (*tw_r)[1];
                data[q] = inverse ? y0 * inv_radix : y0;
                data[q + m] = inverse ? y1 * inv_radix : y1;
                data[q + 2u * m] = inverse ? y2 * inv_radix : y2;
                data[q + 3u * m] = inverse ? y3 * inv_radix : y3;
                data[q + 4u * m] = inverse ? y4 * inv_radix : y4;
            }
        }
    };

    [[nodiscard]] std::shared_ptr<smooth_plan_node> build_smooth_plan(std::size_t n, bool inverse) {
        auto node = std::make_shared<smooth_plan_node>();
        node->n = n;
        if constexpr (is_complex_double) {
            if (n == 100u) {
                node->kind = smooth_plan_kind::special100;
                return node;
            }
            if (n == 1000u) {
                node->kind = smooth_plan_kind::special1000;
                return node;
            }
            if (n == 1500u) {
                node->kind = smooth_plan_kind::special1500;
                return node;
            }
        }
        if (std::has_single_bit(n)) {
            node->kind = smooth_plan_kind::pow2;
            node->pt = &ptw_cache_.get(n, inverse);
            return node;
        }

        const std::size_t radix = smooth_radix_235(n);
        node->m = n / radix;
        node->kind = radix == 2u ? smooth_plan_kind::radix2 : (radix == 3u ? smooth_plan_kind::radix3 : smooth_plan_kind::radix5);
        node->tw_n = &full_tw_cache_.get(n, inverse);
        if (radix == 3u || radix == 5u)
            node->tw_r = &full_tw_cache_.get(radix, inverse);
        for (std::size_t j = 0; j < radix; ++j)
            node->children[j] = build_smooth_plan(node->m, inverse);
        return node;
    }

    struct smooth_plan_cache_t {
        mutable std::shared_mutex mtx;
        std::unordered_map<std::size_t, std::shared_ptr<smooth_plan_node>> fwd, inv;

        const std::shared_ptr<smooth_plan_node>& get(avx2& engine, std::size_t n, bool inverse) {
            auto& map = inverse ? inv : fwd;
            {
                std::shared_lock lk(mtx);
                auto it = map.find(n);
                if (it != map.end()) return it->second;
            }

            auto node = engine.build_smooth_plan(n, inverse);
            std::unique_lock lk(mtx);
            auto [it, _] = map.emplace(n, std::move(node));
            return it->second;
        }
    } smooth_plan_cache_;

    struct plan_2d_cache_t {
        mutable std::shared_mutex mtx;
        std::unordered_map<std::uint64_t, std::shared_ptr<plan_2d>> fwd, inv;

        [[nodiscard]] static std::uint64_t key(std::size_t rows, std::size_t cols) noexcept {
            return (static_cast<std::uint64_t>(rows) << 32u) ^ static_cast<std::uint64_t>(cols);
        }

        const std::shared_ptr<plan_2d>& get(avx2& engine, std::size_t rows, std::size_t cols, bool inverse) {
            auto& map = inverse ? inv : fwd;
            const std::uint64_t cache_key = key(rows, cols);
            {
                std::shared_lock lk(mtx);
                auto it = map.find(cache_key);
                if (it != map.end()) return it->second;
            }

            auto plan_ptr = std::make_shared<plan_2d>(&engine, rows, cols, inverse);
            std::unique_lock lk(mtx);
            auto [it, _] = map.emplace(cache_key, std::move(plan_ptr));
            return it->second;
        }
    } plan_2d_cache_;

    // Stockham stage with pre-packed contiguous twiddles
    // stage_idx = log2(stage_len); pt.stages[stage_idx] is contiguous [0..half_len-1]
    static void stockham_stage_packed_pd(const _ComplexNumber* __restrict__ src,
                                               _ComplexNumber* __restrict__ dst,
                                         int stage_idx, std::size_t n,
                                         const packed_twiddles& pt) noexcept {
        const std::size_t half  = std::size_t(1) << stage_idx;
        const std::size_t full  = half * 2;
        const std::size_t groups = n / full;
        const _ComplexNumber* tw = pt.stages[stage_idx].data();

#pragma omp parallel for schedule(static) if(n >= kParallelThresh * 2)
        for (std::ptrdiff_t g = 0; g < static_cast<std::ptrdiff_t>(groups); ++g) {
            const _ComplexNumber* su = src + g * half;
            const _ComplexNumber* sv = src + g * half + n / 2;
                  _ComplexNumber* du = dst + g * full;
                  _ComplexNumber* dv = dst + g * full + half;
            std::size_t k = 0;
            for (; k + 1 < half; k += 2) {
                // Twiddles always aligned (from aligned_vec); src/dst may not be.
                const double* t = reinterpret_cast<const double*>(tw + k);
                __m256d w = _mm256_load_pd(t);
                __m256d u = _mm256_loadu_pd(reinterpret_cast<const double*>(su + k));
                __m256d v = _mm256_loadu_pd(reinterpret_cast<const double*>(sv + k));
                _mm_prefetch(reinterpret_cast<const char*>(tw + k + 8), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(su + k + 8), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(sv + k + 8), _MM_HINT_T0);
                __m256d vw = cmul_pd(v, w);
                _mm256_storeu_pd(reinterpret_cast<double*>(du + k), _mm256_add_pd(u, vw));
                _mm256_storeu_pd(reinterpret_cast<double*>(dv + k), _mm256_sub_pd(u, vw));
            }
            for (; k < half; ++k) {
                const _ComplexNumber ww = tw[k];
                const _ComplexNumber u  = su[k];
                const _ComplexNumber vw = sv[k] * ww;
                du[k] = u + vw;  dv[k] = u - vw;
            }
        }
    }

    static void stockham_stage_packed_ps(const _ComplexNumber* __restrict__ src,
                                               _ComplexNumber* __restrict__ dst,
                                         int stage_idx, std::size_t n,
                                         const packed_twiddles& pt) noexcept {
        const std::size_t half  = std::size_t(1) << stage_idx;
        const std::size_t full  = half * 2;
        const std::size_t groups = n / full;
        const _ComplexNumber* tw = pt.stages[stage_idx].data();

#pragma omp parallel for schedule(static) if(n >= kParallelThresh * 2)
        for (std::ptrdiff_t g = 0; g < static_cast<std::ptrdiff_t>(groups); ++g) {
            const _ComplexNumber* su = src + g * half;
            const _ComplexNumber* sv = src + g * half + n / 2;
                  _ComplexNumber* du = dst + g * full;
                  _ComplexNumber* dv = dst + g * full + half;
            std::size_t k = 0;
            for (; k + 3 < half; k += 4) {
                const float* t = reinterpret_cast<const float*>(tw + k);
                __m256 w  = _mm256_load_ps(t);  // twiddles always aligned
                __m256 u  = _mm256_loadu_ps(reinterpret_cast<const float*>(su + k));
                __m256 v  = _mm256_loadu_ps(reinterpret_cast<const float*>(sv + k));
                _mm_prefetch(reinterpret_cast<const char*>(tw + k + 8), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(su + k + 8), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(sv + k + 8), _MM_HINT_T0);
                __m256 vw = cmul_ps(v, w);
                _mm256_storeu_ps(reinterpret_cast<float*>(du + k), _mm256_add_ps(u, vw));
                _mm256_storeu_ps(reinterpret_cast<float*>(dv + k), _mm256_sub_ps(u, vw));
            }
            for (; k < half; ++k) {
                const _ComplexNumber ww = tw[k];
                const _ComplexNumber u  = su[k];
                const _ComplexNumber vw = sv[k] * ww;
                du[k] = u + vw;  dv[k] = u - vw;
            }
        }
    }

    // Radix-4 stage (double, in-place, packed twiddles)
    static void radix4_stage_packed_pd(_ComplexNumber* data, std::size_t n,
                                        int stage_idx,   // stage_idx for a *quarter-len* pass
                                        const packed_twiddles& pt) noexcept {
        // len = 4 * (1 << stage_idx) but we reuse the packed twiddle for stage_idx
        // (W^k for k=0..(quarter-1)), w2=W^(2k), w3=W^(3k) computed from w1
        const std::size_t quarter = std::size_t(1) << stage_idx;
        const std::size_t len = quarter * 4;
        const _ComplexNumber* tw = pt.stages[stage_idx].data(); // w^k for k=0..quarter-1

#pragma omp parallel for schedule(static) if(n >= kParallelThresh * 2)
        for (std::ptrdiff_t base = 0; base < static_cast<std::ptrdiff_t>(n);
             base += static_cast<std::ptrdiff_t>(len)) {
            std::size_t k = 0;
            for (; k + 1 < quarter; k += 2) {
                const double* ta = reinterpret_cast<const double*>(tw + k);
                __m256d w1 = _mm256_load_pd(ta);  // twiddles always aligned
                __m256d w2 = cmul_pd(w1, w1);
                __m256d w3 = cmul_pd(w2, w1);
                double* p0 = reinterpret_cast<double*>(&data[base + k]);
                double* p1 = reinterpret_cast<double*>(&data[base + k + quarter]);
                double* p2 = reinterpret_cast<double*>(&data[base + k + 2 * quarter]);
                double* p3 = reinterpret_cast<double*>(&data[base + k + 3 * quarter]);
                __m256d x0 = _mm256_loadu_pd(p0);
                __m256d x1 = cmul_pd(_mm256_loadu_pd(p1), w1);
                __m256d x2 = cmul_pd(_mm256_loadu_pd(p2), w2);
                __m256d x3 = cmul_pd(_mm256_loadu_pd(p3), w3);
                __m256d t0 = _mm256_add_pd(x0, x2);
                __m256d t1 = _mm256_sub_pd(x0, x2);
                __m256d t2 = _mm256_add_pd(x1, x3);
                __m256d d_  = _mm256_sub_pd(x1, x3);
                __m256d d_shuf = _mm256_shuffle_pd(d_, d_, 0x5);
                static const __m256d sign_mask = _mm256_setr_pd(-0.0, 0.0, -0.0, 0.0);
                __m256d t3 = _mm256_xor_pd(d_shuf, sign_mask);
                _mm256_storeu_pd(p0, _mm256_add_pd(t0, t2));
                _mm256_storeu_pd(p1, _mm256_add_pd(t1, t3));
                _mm256_storeu_pd(p2, _mm256_sub_pd(t0, t2));
                _mm256_storeu_pd(p3, _mm256_sub_pd(t1, t3));
            }
            for (; k < quarter; ++k) {
                const _ComplexNumber w1  = tw[k];
                const _ComplexNumber w2  = w1 * w1;
                const _ComplexNumber w3  = w2 * w1;
                const _ComplexNumber x0  = data[base + k];
                const _ComplexNumber x1  = data[base + k + quarter]     * w1;
                const _ComplexNumber x2  = data[base + k + 2 * quarter] * w2;
                const _ComplexNumber x3  = data[base + k + 3 * quarter] * w3;
                const _ComplexNumber t0  = x0 + x2;
                const _ComplexNumber t1  = x0 - x2;
                const _ComplexNumber t2  = x1 + x3;
                const _ComplexNumber d_  = x1 - x3;
                const _ComplexNumber t3  = _ComplexNumber(d_.imag(), -d_.real());
                data[base + k]             = t0 + t2;
                data[base + k + quarter]   = t1 + t3;
                data[base + k + 2*quarter] = t0 - t2;
                data[base + k + 3*quarter] = t1 - t3;
            }
        }
    }

    static void radix4_stage_packed_ps(_ComplexNumber* data, std::size_t n,
                                       int stage_idx,
                                       const packed_twiddles& pt) noexcept {
        const std::size_t quarter = std::size_t(1) << stage_idx;
        const std::size_t len = quarter * 4;
        const _ComplexNumber* tw = pt.stages[stage_idx].data();

#pragma omp parallel for schedule(static) if(n >= kParallelThresh * 2)
        for (std::ptrdiff_t base = 0; base < static_cast<std::ptrdiff_t>(n);
             base += static_cast<std::ptrdiff_t>(len)) {
            std::size_t k = 0;
            for (; k + 3 < quarter; k += 4) {
                const float* ta = reinterpret_cast<const float*>(tw + k);
                __m256 w1 = _mm256_load_ps(ta);
                __m256 w2 = cmul_ps(w1, w1);
                __m256 w3 = cmul_ps(w2, w1);
                float* p0 = reinterpret_cast<float*>(&data[base + k]);
                float* p1 = reinterpret_cast<float*>(&data[base + k + quarter]);
                float* p2 = reinterpret_cast<float*>(&data[base + k + 2 * quarter]);
                float* p3 = reinterpret_cast<float*>(&data[base + k + 3 * quarter]);
                __m256 x0 = _mm256_loadu_ps(p0);
                __m256 x1 = cmul_ps(_mm256_loadu_ps(p1), w1);
                __m256 x2 = cmul_ps(_mm256_loadu_ps(p2), w2);
                __m256 x3 = cmul_ps(_mm256_loadu_ps(p3), w3);
                __m256 t0 = _mm256_add_ps(x0, x2);
                __m256 t1 = _mm256_sub_ps(x0, x2);
                __m256 t2 = _mm256_add_ps(x1, x3);
                __m256 d_ = _mm256_sub_ps(x1, x3);
                __m256 d_shuf = _mm256_shuffle_ps(d_, d_, 0xB1);
                static const __m256 sign_mask = _mm256_setr_ps(0.0f, -0.0f, 0.0f, -0.0f,
                                                               0.0f, -0.0f, 0.0f, -0.0f);
                __m256 t3 = _mm256_xor_ps(d_shuf, sign_mask);
                _mm256_storeu_ps(p0, _mm256_add_ps(t0, t2));
                _mm256_storeu_ps(p1, _mm256_add_ps(t1, t3));
                _mm256_storeu_ps(p2, _mm256_sub_ps(t0, t2));
                _mm256_storeu_ps(p3, _mm256_sub_ps(t1, t3));
            }
            for (; k < quarter; ++k) {
                const _ComplexNumber w1 = tw[k];
                const _ComplexNumber w2 = w1 * w1;
                const _ComplexNumber w3 = w2 * w1;
                const _ComplexNumber x0 = data[base + k];
                const _ComplexNumber x1 = data[base + k + quarter] * w1;
                const _ComplexNumber x2 = data[base + k + 2 * quarter] * w2;
                const _ComplexNumber x3 = data[base + k + 3 * quarter] * w3;
                const _ComplexNumber t0 = x0 + x2;
                const _ComplexNumber t1 = x0 - x2;
                const _ComplexNumber t2 = x1 + x3;
                const _ComplexNumber d_ = x1 - x3;
                const _ComplexNumber t3 = _ComplexNumber(d_.imag(), -d_.real());
                data[base + k] = t0 + t2;
                data[base + k + quarter] = t1 + t3;
                data[base + k + 2 * quarter] = t0 - t2;
                data[base + k + 3 * quarter] = t1 - t3;
            }
        }
    }

    // Core Stockham (double) using pre-packed twiddles
    static void scale_inplace(_ComplexNumber* data, std::size_t n, value_type scale) noexcept {
        if constexpr (is_complex_double) {
            const __m256d s4 = _mm256_set1_pd(double(scale));
            std::size_t i = 0;
            double* dp = reinterpret_cast<double*>(data);
            for (; i + 1 < n; i += 2, dp += 4)
                _mm256_storeu_pd(dp, _mm256_mul_pd(_mm256_loadu_pd(dp), s4));
            for (; i < n; ++i) data[i] *= scale;
        } else {
            const __m256 s8 = _mm256_set1_ps(float(scale));
            std::size_t i = 0;
            float* fp = reinterpret_cast<float*>(data);
            for (; i + 3 < n; i += 4, fp += 8)
                _mm256_storeu_ps(fp, _mm256_mul_ps(_mm256_loadu_ps(fp), s8));
            for (; i < n; ++i) data[i] *= scale;
        }
    }

    static void execute_pow2_presized(_ComplexNumber* raw, std::size_t n,
                                      _ComplexNumber* scratch,
                                      const packed_twiddles& pt,
                                      bool inverse) noexcept {
        if (n <= 1) return;

        if constexpr (is_complex_double) {
            if (n ==  2) { avx2_detail::codelet_2(reinterpret_cast<double*>(raw), inverse); return; }
            if (n ==  4) { avx2_detail::codelet_4(reinterpret_cast<double*>(raw), inverse); return; }
            if (n ==  8) { avx2_detail::codelet_8(reinterpret_cast<double*>(raw), inverse); return; }
            if (n == 16) { avx2_detail::codelet_16(reinterpret_cast<double*>(raw), inverse); return; }
            if (n == 32) { avx2_detail::codelet_32(reinterpret_cast<double*>(raw), inverse); return; }
            if (n == 64) { avx2_detail::codelet_64(reinterpret_cast<double*>(raw), inverse); return; }
            if (n == 128) { avx2_detail::codelet_128(reinterpret_cast<double*>(raw), inverse); return; }
            if (n == 256) { avx2_detail::codelet_256(reinterpret_cast<double*>(raw), inverse); return; }
            if (n == 512) { avx2_detail::codelet_512(reinterpret_cast<double*>(raw), inverse); return; }
            if (n == 1024) { avx2_detail::codelet_1024(reinterpret_cast<double*>(raw), inverse); return; }
        }

        _ComplexNumber* a = raw;
        _ComplexNumber* b = scratch;
        const int log2n = static_cast<int>(std::countr_zero(n));
        int s = 0;
        while (s + 1 < log2n) {
            if constexpr (is_complex_double)
                stockham_stage_packed_pd(a, b, s, n, pt);
            else if constexpr (is_complex_float)
                stockham_stage_packed_ps(a, b, s, n, pt);
            std::swap(a, b);
            ++s;
            if (s < log2n && s + 1 < log2n) {
                if constexpr (is_complex_double) {
                    radix4_stage_packed_pd(a, n, s - 1, pt);
                }
                    ++s;
            }
        }
        if (s < log2n) {
            if constexpr (is_complex_double)
                stockham_stage_packed_pd(a, b, s, n, pt);
            else if constexpr (is_complex_float)
                stockham_stage_packed_ps(a, b, s, n, pt);
            std::swap(a, b);
        }
        if (a != raw)
            std::copy_n(scratch, n, raw);
        if (inverse)
            scale_inplace(raw, n, value_type(1) / static_cast<value_type>(n));
    }

    static void execute_pow2_raw(_ComplexNumber* raw, std::size_t n,
                                 aligned_vec& work,
                                 const packed_twiddles& pt,
                                 bool inverse) noexcept {
        if (n <= 1) return;
        if constexpr (is_complex_double) {
            if (n == 2u) { avx2_detail::codelet_2(reinterpret_cast<double*>(raw), inverse); return; }
            if (n == 4u) { avx2_detail::codelet_4(reinterpret_cast<double*>(raw), inverse); return; }
            if (n == 8u) { avx2_detail::codelet_8(reinterpret_cast<double*>(raw), inverse); return; }
            if (n == 16u) { avx2_detail::codelet_16(reinterpret_cast<double*>(raw), inverse); return; }
            if (n == 32u) { avx2_detail::codelet_32(reinterpret_cast<double*>(raw), inverse); return; }
            if (n == 64u) { avx2_detail::codelet_64(reinterpret_cast<double*>(raw), inverse); return; }
            if (n == 128u) { avx2_detail::codelet_128(reinterpret_cast<double*>(raw), inverse); return; }
            if (n == 256u) { avx2_detail::codelet_256(reinterpret_cast<double*>(raw), inverse); return; }
            if (n == 512u) { avx2_detail::codelet_512(reinterpret_cast<double*>(raw), inverse); return; }
            if (n == 1024u) { avx2_detail::codelet_1024(reinterpret_cast<double*>(raw), inverse); return; }
        }
        work.resize(n);
        execute_pow2_presized(raw, n, work.data(), pt, inverse);
    }

    static void run_stockham_double(aligned_vec& data, aligned_vec& work,
                                    const packed_twiddles& pt, bool inverse) noexcept {
        execute_pow2_raw(data.data(), data.size(), work, pt, inverse);
    }

    [[nodiscard]] static std::size_t smooth_radix_235(std::size_t n) noexcept {
        if ((n % 5u) == 0u) return 5u;
        if ((n % 3u) == 0u) return 3u;
        if ((n % 2u) == 0u) return 2u;
        return 0u;
    }

    [[nodiscard]] static bool is_smooth_235(std::size_t n) noexcept {
        if (n == 0u) return false;
        while ((n % 2u) == 0u) n /= 2u;
        while ((n % 3u) == 0u) n /= 3u;
        while ((n % 5u) == 0u) n /= 5u;
        return n == 1u;
    }

    [[nodiscard]] static bool prefer_small_2d_avx2(std::size_t rows, std::size_t cols) noexcept {
        return is_complex_double && std::has_single_bit(rows) && std::has_single_bit(cols) && rows <= 512u && cols <= 512u;
    }

    void smooth_235_presized(_ComplexNumber* data,
                             std::size_t n,
                             _ComplexNumber* scratch,
                             bool inverse) {
        if (n <= 1u) return;
        if constexpr (is_complex_double) {
            if (n == 100u) { avx2_detail::smooth_100(reinterpret_cast<double*>(data), inverse); return; }
            if (n == 1000u) { avx2_detail::smooth_1000(reinterpret_cast<double*>(data), inverse); return; }
            if (n == 1500u) { avx2_detail::smooth_1500(reinterpret_cast<double*>(data), inverse); return; }
        }
        if (std::has_single_bit(n)) {
            const auto& pt = ptw_cache_.get(n, inverse);
            execute_pow2_presized(data, n, scratch, pt, inverse);
            return;
        }

        const std::size_t radix = smooth_radix_235(n);
        if (radix == 0u) {
            tensor::view<_ComplexNumber> v(std::span<_ComplexNumber>{data, n});
            bluestein_avx2(v, inverse);
            return;
        }

        const std::size_t m = n / radix;

        if (radix == 2u) {
            for (std::size_t q = 0; q < m; ++q) {
                scratch[q] = data[q * 2u];
                scratch[m + q] = data[q * 2u + 1u];
            }
        } else if (radix == 3u) {
            for (std::size_t q = 0; q < m; ++q) {
                scratch[q] = data[q * 3u];
                scratch[m + q] = data[q * 3u + 1u];
                scratch[2u * m + q] = data[q * 3u + 2u];
            }
        } else {
            for (std::size_t q = 0; q < m; ++q) {
                scratch[q] = data[q * 5u];
                scratch[m + q] = data[q * 5u + 1u];
                scratch[2u * m + q] = data[q * 5u + 2u];
                scratch[3u * m + q] = data[q * 5u + 3u];
                scratch[4u * m + q] = data[q * 5u + 4u];
            }
        }

        for (std::size_t j = 0; j < radix; ++j)
            smooth_235_presized(scratch + j * m, m, data + j * m, inverse);

        const auto& tw_n = full_tw_cache_.get(n, inverse);
        const value_type inv_radix = value_type(1) / static_cast<value_type>(radix);

        if (radix == 2u) {
            const _ComplexNumber* s0 = scratch;
            const _ComplexNumber* s1 = scratch + m;
            for (std::size_t q = 0; q < m; ++q) {
                const _ComplexNumber v0 = s0[q];
                const _ComplexNumber v1 = s1[q] * tw_n[q];
                const _ComplexNumber y0 = v0 + v1;
                const _ComplexNumber y1 = v0 - v1;
                data[q] = inverse ? y0 * inv_radix : y0;
                data[q + m] = inverse ? y1 * inv_radix : y1;
            }
            return;
        }

        if (radix == 3u) {
            const auto& tw_r = full_tw_cache_.get(3u, inverse);
            const _ComplexNumber* s0 = scratch;
            const _ComplexNumber* s1 = scratch + m;
            const _ComplexNumber* s2 = scratch + 2u * m;
            for (std::size_t q = 0; q < m; ++q) {
                const _ComplexNumber v0 = s0[q];
                const _ComplexNumber v1 = s1[q] * tw_n[q];
                const _ComplexNumber v2 = s2[q] * tw_n[2u * q];
                const _ComplexNumber y0 = v0 + v1 + v2;
                const _ComplexNumber y1 = v0 + v1 * tw_r[1] + v2 * tw_r[2];
                const _ComplexNumber y2 = v0 + v1 * tw_r[2] + v2 * tw_r[1];
                data[q] = inverse ? y0 * inv_radix : y0;
                data[q + m] = inverse ? y1 * inv_radix : y1;
                data[q + 2 * m] = inverse ? y2 * inv_radix : y2;
            }
            return;
        }

        const auto& tw_r = full_tw_cache_.get(5u, inverse);
        const _ComplexNumber* s0 = scratch;
        const _ComplexNumber* s1 = scratch + m;
        const _ComplexNumber* s2 = scratch + 2u * m;
        const _ComplexNumber* s3 = scratch + 3u * m;
        const _ComplexNumber* s4 = scratch + 4u * m;
        for (std::size_t q = 0; q < m; ++q) {
            const _ComplexNumber v0 = s0[q];
            const _ComplexNumber v1 = s1[q] * tw_n[q];
            const _ComplexNumber v2 = s2[q] * tw_n[2u * q];
            const _ComplexNumber v3 = s3[q] * tw_n[3u * q];
            const _ComplexNumber v4 = s4[q] * tw_n[4u * q];
            const _ComplexNumber y0 = v0 + v1 + v2 + v3 + v4;
            const _ComplexNumber y1 = v0 + v1 * tw_r[1] + v2 * tw_r[2] + v3 * tw_r[3] + v4 * tw_r[4];
            const _ComplexNumber y2 = v0 + v1 * tw_r[2] + v2 * tw_r[4] + v3 * tw_r[1] + v4 * tw_r[3];
            const _ComplexNumber y3 = v0 + v1 * tw_r[3] + v2 * tw_r[1] + v3 * tw_r[4] + v4 * tw_r[2];
            const _ComplexNumber y4 = v0 + v1 * tw_r[4] + v2 * tw_r[3] + v3 * tw_r[2] + v4 * tw_r[1];
            data[q] = inverse ? y0 * inv_radix : y0;
            data[q + m] = inverse ? y1 * inv_radix : y1;
            data[q + 2 * m] = inverse ? y2 * inv_radix : y2;
            data[q + 3 * m] = inverse ? y3 * inv_radix : y3;
            data[q + 4 * m] = inverse ? y4 * inv_radix : y4;
        }
    }

    void smooth_235_raw(_ComplexNumber* data, std::size_t n, aligned_vec& scratch, bool inverse) {
        scratch.resize(n);
        smooth_235_presized(data, n, scratch.data(), inverse);
    }

    static void transpose_blocked(const _ComplexNumber* src,
                                  std::size_t rows,
                                  std::size_t cols,
                                  _ComplexNumber* dst) {
        static constexpr std::size_t kTransposeTile = 32;
#pragma omp parallel for schedule(static) collapse(2)
        for (std::ptrdiff_t r0 = 0; r0 < static_cast<std::ptrdiff_t>(rows); r0 += static_cast<std::ptrdiff_t>(kTransposeTile)) {
            for (std::ptrdiff_t c0 = 0; c0 < static_cast<std::ptrdiff_t>(cols); c0 += static_cast<std::ptrdiff_t>(kTransposeTile)) {
                const std::size_t r_end = std::min<std::size_t>(rows, static_cast<std::size_t>(r0) + kTransposeTile);
                const std::size_t c_end = std::min<std::size_t>(cols, static_cast<std::size_t>(c0) + kTransposeTile);
                for (std::size_t r = static_cast<std::size_t>(r0); r < r_end; ++r)
                    for (std::size_t c = static_cast<std::size_t>(c0); c < c_end; ++c)
                        dst[c * rows + r] = src[r * cols + c];
            }
        }
    }

    // Six-step (double)
    void six_step_double_raw_impl(_ComplexNumber* data, std::size_t n,
                                  aligned_vec& scratch, bool inverse,
                                  const aligned_vec& tw_n,
                                  const packed_twiddles& ptC,
                                  const packed_twiddles& ptR) {
        const int half_log  = std::countr_zero(n) / 2;
        const std::size_t C = std::size_t(1) << half_log;
        const std::size_t R = n / C;

        // Step 1: row FFTs (each row length C)
#pragma omp parallel
        {
            aligned_vec local_work;
#pragma omp for schedule(static)
            for (std::ptrdiff_t r = 0; r < static_cast<std::ptrdiff_t>(R); ++r)
                execute_pow2_raw(data + static_cast<std::size_t>(r) * C, C, local_work, ptC, inverse);
        }

        // Step 2: twiddle multiply
#pragma omp parallel for schedule(static) collapse(2)
        for (std::ptrdiff_t r = 1; r < static_cast<std::ptrdiff_t>(R); ++r)
            for (std::ptrdiff_t c = 1; c < static_cast<std::ptrdiff_t>(C); ++c)
                data[r*C + c] *= tw_n[(r * c) % n];

        // Step 3: transpose R×C → C×R into scratch
        scratch.resize(n);
        transpose_blocked(data, R, C, scratch.data());

        // Step 4: column FFTs (each length R)
#pragma omp parallel
        {
            aligned_vec local_work;
#pragma omp for schedule(static)
            for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(C); ++c)
                execute_pow2_raw(scratch.data() + static_cast<std::size_t>(c) * R, R, local_work, ptR, inverse);
        }

        // Step 5: transpose back C×R → R×C
        transpose_blocked(scratch.data(), C, R, data);
    }

    void six_step_double_raw(_ComplexNumber* data, std::size_t n,
                             aligned_vec& scratch, bool inverse) {
        const int half_log  = std::countr_zero(n) / 2;
        const std::size_t C = std::size_t(1) << half_log;
        const std::size_t R = n / C;
        const auto& ptC  = ptw_cache_.get(C, inverse);
        const auto& ptR  = ptw_cache_.get(R, inverse);
        const auto& tw_n = full_tw_cache_.get(n, inverse);
        six_step_double_raw_impl(data, n, scratch, inverse, tw_n, ptC, ptR);
    }

    void six_step_double(aligned_vec& data, bool inverse) {
        six_step_double_raw(data.data(), data.size(), work_buf_, inverse);
    }

    // Top-level dispatch
    void run_avx2(aligned_vec& buf, const packed_twiddles& pt, bool inverse) {
        if (buf.size() >= kSixStepThresh && is_complex_double)
            six_step_double(buf, inverse);
        else
            execute_pow2_raw(buf.data(), buf.size(), work_buf_, pt, inverse);
    }

    // Bluestein via AVX2 inner FFT
    // Non-power-of-2 sizes: pads to M (next pow-2 ≥ 2N-1), runs AVX2 for the
    // inner power-of-2 FFTs rather than falling to the software backend.
    static void bluestein_execute(tensor::view<_ComplexNumber> v,
                                  const bluestein_cache_entry& cache,
                                  aligned_vec& a_pad,
                                  aligned_vec& work,
                                  const packed_twiddles& pt_m,
                                  const packed_twiddles& pt_m_inv,
                                  bool inverse) {
        const std::size_t ni = v.size();
        const std::size_t m = cache.m;

        a_pad.resize(m);
        std::fill(a_pad.begin(), a_pad.end(), _ComplexNumber{});
        for (std::size_t k = 0; k < ni; ++k)
            a_pad[k] = v[k] * cache.chirp[k];

        execute_pow2_raw(a_pad.data(), m, work, pt_m, false);

        for (std::size_t i = 0; i < m; ++i)
            a_pad[i] *= cache.h_fft[i];

        execute_pow2_raw(a_pad.data(), m, work, pt_m_inv, true);

        for (std::size_t k = 0; k < ni; ++k)
            v[k] = a_pad[k] * cache.chirp[k];

        if (inverse) {
            const value_type sc = value_type(1) / static_cast<value_type>(ni);
            for (std::size_t k = 0; k < ni; ++k) v[k] *= sc;
        }
    }

    void bluestein_avx2(tensor::view<_ComplexNumber> v, bool inverse) {
        const auto& cache = bluestein_cache_.get(*this, v.size(), inverse);
        const auto& pt_m = ptw_cache_.get(cache.m, false);
        const auto& pt_m_inv = ptw_cache_.get(cache.m, true);
        bluestein_execute(v, cache, bluestein_buf_, work_buf_, pt_m, pt_m_inv, inverse);
    }

public:
    // Plan object
    // A plan pre-computes all twiddles and pre-allocates the work buffer.
    // plan::execute() is the zero-allocation hot path.
    struct plan {
        enum class pinned_pow2_codelet : std::uint8_t {
            none,
            n2,
            n4,
            n8,
            n16,
            n32,
            n64,
            n128,
            n256,
            n512,
            n1024,
        };

        avx2*          engine_;
        std::size_t    n_;
        bool           inverse_;
        bool           is_pow2_;
        bool           use_six_step_;
        pinned_pow2_codelet pinned_codelet_ = pinned_pow2_codelet::none;
        const packed_twiddles* pt_ = nullptr;
        const packed_twiddles* pt_bluestein_fwd_ = nullptr;
        const packed_twiddles* pt_bluestein_inv_ = nullptr;
        const packed_twiddles* pt_rows_ = nullptr;
        const packed_twiddles* pt_cols_ = nullptr;
        const bluestein_cache_entry* bluestein_ = nullptr;
        const aligned_vec* full_tw_ = nullptr;
        std::shared_ptr<smooth_plan_node> smooth_plan_;
        aligned_vec    work_;
        aligned_vec    signal_buf_;

        // Called by avx2::make_plan()
        plan(avx2* eng, std::size_t n, bool inv)
            : engine_(eng), n_(n), inverse_(inv),
              is_pow2_(std::has_single_bit(n)),
              use_six_step_(is_pow2_ && is_complex_double && n >= kSixStepThresh)
        {
            if (is_pow2_) {
                pt_ = &eng->ptw_cache_.get(n, inv);
                if constexpr (is_complex_double)
                    avx2_detail::prime_small_kernel_selector(n, inv);
                if constexpr (is_complex_double) {
                    switch (n) {
                    case 2u: pinned_codelet_ = pinned_pow2_codelet::n2; break;
                    case 4u: pinned_codelet_ = pinned_pow2_codelet::n4; break;
                    case 8u: pinned_codelet_ = pinned_pow2_codelet::n8; break;
                    case 16u: pinned_codelet_ = pinned_pow2_codelet::n16; break;
                    case 32u: pinned_codelet_ = pinned_pow2_codelet::n32; break;
                    case 64u: pinned_codelet_ = pinned_pow2_codelet::n64; break;
                    case 128u: pinned_codelet_ = pinned_pow2_codelet::n128; break;
                    case 256u: pinned_codelet_ = pinned_pow2_codelet::n256; break;
                    case 512u: pinned_codelet_ = pinned_pow2_codelet::n512; break;
                    case 1024u: pinned_codelet_ = pinned_pow2_codelet::n1024; break;
                    default: break;
                    }
                }
                work_.resize(n);
                if (use_six_step_) {
                    const int half_log = std::countr_zero(n) / 2;
                    const std::size_t cols = std::size_t(1) << half_log;
                    const std::size_t rows = n / cols;
                    pt_cols_ = &eng->ptw_cache_.get(cols, inv);
                    pt_rows_ = &eng->ptw_cache_.get(rows, inv);
                    full_tw_ = &eng->full_tw_cache_.get(n, inv);
                }
            } else if (is_complex_double && is_smooth_235(n)) {
                smooth_plan_ = eng->smooth_plan_cache_.get(*eng, n, inv);
                work_.resize(n);
            } else {
                bluestein_ = &eng->bluestein_cache_.get(*eng, n, inv);
                pt_bluestein_fwd_ = &eng->ptw_cache_.get(bluestein_->m, false);
                pt_bluestein_inv_ = &eng->ptw_cache_.get(bluestein_->m, true);
                work_.resize(bluestein_->m);
                signal_buf_.resize(bluestein_->m);
            }
        }

        [[nodiscard]] std::size_t work_size() const noexcept {
            if (is_pow2_) return n_;
            if (bluestein_) return bluestein_->m;
            return n_;
        }

        [[nodiscard]] std::size_t signal_size() const noexcept {
            return (is_pow2_ || !bluestein_) ? std::size_t {0} : bluestein_->m;
        }

        void execute_raw_with_scratch(_ComplexNumber* data,
                                      aligned_vec& work,
                                      aligned_vec* signal_buf = nullptr) {
            if (n_ == 0) return;
            if (is_pow2_) {
                if constexpr (is_complex_double) {
                    if (use_six_step_) {
                        engine_->six_step_double_raw_impl(data, n_, work, inverse_, *full_tw_, *pt_cols_, *pt_rows_);
                        return;
                    }
                    switch (pinned_codelet_) {
                    case pinned_pow2_codelet::n2: avx2_detail::codelet_2(reinterpret_cast<double*>(data), inverse_); return;
                    case pinned_pow2_codelet::n4: avx2_detail::codelet_4(reinterpret_cast<double*>(data), inverse_); return;
                    case pinned_pow2_codelet::n8: avx2_detail::codelet_8(reinterpret_cast<double*>(data), inverse_); return;
                    case pinned_pow2_codelet::n16: avx2_detail::codelet_16(reinterpret_cast<double*>(data), inverse_); return;
                    case pinned_pow2_codelet::n32: avx2_detail::codelet_32(reinterpret_cast<double*>(data), inverse_); return;
                    case pinned_pow2_codelet::n64: avx2_detail::codelet_64(reinterpret_cast<double*>(data), inverse_); return;
                    case pinned_pow2_codelet::n128: avx2_detail::codelet_128(reinterpret_cast<double*>(data), inverse_); return;
                    case pinned_pow2_codelet::n256: avx2_detail::codelet_256(reinterpret_cast<double*>(data), inverse_); return;
                    case pinned_pow2_codelet::n512: avx2_detail::codelet_512(reinterpret_cast<double*>(data), inverse_); return;
                    case pinned_pow2_codelet::n1024: avx2_detail::codelet_1024(reinterpret_cast<double*>(data), inverse_); return;
                    case pinned_pow2_codelet::none: break;
                    }
                }
                engine_->execute_pow2_presized(data, n_, work.data(), *pt_, inverse_);
            } else if constexpr (is_complex_double) {
                if (smooth_plan_) {
                    smooth_plan_->execute(*engine_, data, work.data(), inverse_);
                    return;
                }
            } else {
                aligned_vec& signal = signal_buf ? *signal_buf : signal_buf_;
                avx2::bluestein_execute(tensor::view<_ComplexNumber>(std::span<_ComplexNumber>{data, n_}), *bluestein_, signal, work, *pt_bluestein_fwd_, *pt_bluestein_inv_, inverse_);
                return;
            }

            if (bluestein_) {
                aligned_vec& signal = signal_buf ? *signal_buf : signal_buf_;
                avx2::bluestein_execute(tensor::view<_ComplexNumber>(std::span<_ComplexNumber>{data, n_}), *bluestein_, signal, work, *pt_bluestein_fwd_, *pt_bluestein_inv_, inverse_);
            }
        }

        void execute_with_scratch(tensor::view<_ComplexNumber> v_inout,
                                  aligned_vec& work,
                                  aligned_vec* signal_buf = nullptr) {
            if (n_ == 0 || v_inout.size() != n_) return;
            if (work.size() < work_size()) work.resize(work_size());
            if (signal_size() != 0u) {
                aligned_vec& signal = signal_buf ? *signal_buf : signal_buf_;
                if (signal.size() < signal_size()) signal.resize(signal_size());
            }
            execute_raw_with_scratch(v_inout.data().data(), work, signal_buf);
        }

        void execute(tensor::view<_ComplexNumber> v_inout) {
            execute_with_scratch(v_inout, work_, is_pow2_ ? nullptr : &signal_buf_);
        }
    };

    struct plan_2d {
        avx2* engine_;
        std::size_t rows_;
        std::size_t cols_;
        std::size_t total_;
        bool inverse_;
        plan row_plan_;
        plan col_plan_;
        std::vector<aligned_vec> row_work_pool_;
        std::vector<aligned_vec> row_signal_pool_;
        std::vector<aligned_vec> col_work_pool_;
        std::vector<aligned_vec> col_signal_pool_;
        std::vector<aligned_vec> tile_pool_;

        plan_2d(avx2* eng, std::size_t rows, std::size_t cols, bool inv)
            : engine_(eng),
              rows_(rows),
              cols_(cols),
              total_(rows * cols),
              inverse_(inv),
              row_plan_(eng, cols, inv),
              col_plan_(eng, rows, inv) {
            const std::size_t threads = std::max(1, omp_get_max_threads());

            row_work_pool_.resize(threads);
            col_work_pool_.resize(threads);
            for (std::size_t tid = 0; tid < threads; ++tid) {
                row_work_pool_[tid].resize(row_plan_.work_size());
                col_work_pool_[tid].resize(col_plan_.work_size());
            }

            if (row_plan_.signal_size() != 0) {
                row_signal_pool_.resize(threads);
                for (std::size_t tid = 0; tid < threads; ++tid)
                    row_signal_pool_[tid].resize(row_plan_.signal_size());
            }

            if (col_plan_.signal_size() != 0) {
                col_signal_pool_.resize(threads);
                for (std::size_t tid = 0; tid < threads; ++tid)
                    col_signal_pool_[tid].resize(col_plan_.signal_size());
            }

            if (rows_ > 1u)
                tile_pool_.assign(threads, aligned_vec(rows_ * kColTile));

#pragma omp parallel num_threads(static_cast<int>(threads))
            {
                const std::size_t tid = static_cast<std::size_t>(omp_get_thread_num());
                auto touch = [] (aligned_vec& buf) {
                    if (!buf.empty())
                        std::fill(buf.begin(), buf.end(), _ComplexNumber{});
                };
                touch(row_work_pool_[tid]);
                touch(col_work_pool_[tid]);
                if (!row_signal_pool_.empty()) touch(row_signal_pool_[tid]);
                if (!col_signal_pool_.empty()) touch(col_signal_pool_[tid]);
                if (!tile_pool_.empty()) touch(tile_pool_[tid]);
            }
        }

        void execute(tensor::view<_ComplexNumber> v_inout) {
            if (v_inout.rank() != 2u) return;
            const auto shape = v_inout.shape();
            if (shape[0u] != rows_ || shape[1u] != cols_) return;

            if (total_ < k2DAvx2Thresh && !prefer_small_2d_avx2(rows_, cols_)) {
                engine_->fallback_engine_.transform_2d(v_inout, inverse_);
                return;
            }

            const auto raw = v_inout.data();

#pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                aligned_vec* row_signal = row_signal_pool_.empty() ? nullptr : &row_signal_pool_[static_cast<std::size_t>(tid)];
                aligned_vec* col_signal = col_signal_pool_.empty() ? nullptr : &col_signal_pool_[static_cast<std::size_t>(tid)];
                aligned_vec& row_work = row_work_pool_[static_cast<std::size_t>(tid)];
                aligned_vec& col_work = col_work_pool_[static_cast<std::size_t>(tid)];
                aligned_vec& tile = tile_pool_[static_cast<std::size_t>(tid)];

#pragma omp for schedule(static)
                for (std::ptrdiff_t r = 0; r < static_cast<std::ptrdiff_t>(rows_); ++r) {
                    row_plan_.execute_raw_with_scratch(raw.data() + static_cast<std::size_t>(r) * cols_, row_work, row_signal);
                }

#pragma omp for schedule(static)
                for (std::ptrdiff_t c0 = 0; c0 < static_cast<std::ptrdiff_t>(cols_);
                     c0 += static_cast<std::ptrdiff_t>(kColTile)) {
                    const std::size_t nc = std::min(kColTile, cols_ - static_cast<std::size_t>(c0));

                    for (std::size_t r = 0; r < rows_; ++r) {
                        _mm_prefetch(reinterpret_cast<const char*>(&raw[r * cols_ + static_cast<std::size_t>(c0) + kColTile]),
                                     _MM_HINT_T1);
                        for (std::size_t tc = 0; tc < nc; ++tc)
                            tile[tc * rows_ + r] = raw[r * cols_ + static_cast<std::size_t>(c0) + tc];
                    }

                    for (std::size_t tc = 0; tc < nc; ++tc) {
                        col_plan_.execute_raw_with_scratch(tile.data() + tc * rows_, col_work, col_signal);
                    }

                    for (std::size_t r = 0; r < rows_; ++r)
                        for (std::size_t tc = 0; tc < nc; ++tc)
                            raw[r * cols_ + static_cast<std::size_t>(c0) + tc] = tile[tc * rows_ + r];
                }
            }
        }
    };

    plan make_plan(std::size_t n, bool inverse = false) {
        // Warm up twiddle caches while we're at it
        if (std::has_single_bit(n)) {
            ptw_cache_.get(n, inverse);
        }
        return plan{this, n, inverse};
    }

    plan_2d make_plan_2d(std::size_t rows, std::size_t cols, bool inverse = false) {
        return plan_2d{this, rows, cols, inverse};
    }

    // Standard transform_1d / transform_2d
    immediate_awaitable transform_1d(tensor::view<_ComplexNumber> v_inout, bool inverse) {
        const std::size_t n = v_inout.size();
        if (!std::has_single_bit(n)) {
            if constexpr (is_complex_double || is_complex_float) {
                if constexpr (is_complex_double) {
                    if (is_smooth_235(n)) {
                        const auto& smooth_plan = smooth_plan_cache_.get(*this, n, inverse);
                        work_buf_.resize(n);
                        smooth_plan->execute(*this, v_inout.data().data(), work_buf_.data(), inverse);
                        return immediate_awaitable{};
                    }
                }
                bluestein_avx2(v_inout, inverse);
                return immediate_awaitable{};
            }
            return fallback_engine_.transform_1d(v_inout, inverse);
        }

        if (n >= kSixStepThresh && is_complex_double) {
            six_step_double_raw(v_inout.data().data(), n, work_buf_, inverse);
            return immediate_awaitable{};
        }
        const auto& pt = ptw_cache_.get(n, inverse);
        if constexpr (is_complex_float) {
            execute_pow2_raw(v_inout.data().data(), n, work_buf_, pt, inverse);
            return immediate_awaitable{};
        }
        execute_pow2_raw(v_inout.data().data(), n, work_buf_, pt, inverse);
        return immediate_awaitable{};
    }

    template <typename _InputComplexNumber>
    immediate_awaitable transform_1d(const tensor::view<_InputComplexNumber>& v_in,
                                     tensor::view<_ComplexNumber> v_out, bool inverse) {
        if (reinterpret_cast<const void*>(v_in.data().data()) !=
            reinterpret_cast<const void*>(v_out.data().data()))
            std::copy(v_in.data().begin(), v_in.data().end(), v_out.data().begin());
        return transform_1d(v_out, inverse);
    }

    // 2D transform
    immediate_awaitable transform_2d(tensor::view<_ComplexNumber> v_inout, bool inverse) {
        if (v_inout.rank() == 1u) return transform_1d(v_inout, inverse);
        if (v_inout.rank() != 2u) [[unlikely]]
            return fallback_engine_.transform_2d(v_inout, inverse);

        const tensor::dimension_shape shape = v_inout.shape();
        const std::size_t rows = shape[0u];
        const std::size_t cols = shape[1u];
        const auto raw = v_inout.data();
        const std::size_t total = rows * cols;

        // For small 2D sizes the scalar software engine has lower per-call overhead
        // than the Stockham setup cost. Delegate until 128×128 (16384 elements).
        if (total < k2DAvx2Thresh && prefer_small_2d_avx2(rows, cols)) {
            plan_2d_cache_.get(*this, rows, cols, inverse)->execute(v_inout);
            return immediate_awaitable{};
        }

        if (total < k2DAvx2Thresh && !prefer_small_2d_avx2(rows, cols))
            return fallback_engine_.transform_2d(v_inout, inverse);

        const bool col_pow2 = std::has_single_bit(cols);
        const bool row_pow2 = std::has_single_bit(rows);

        // Acquire twiddle references ONCE — one shared_lock each, zero inside the loops.
        const packed_twiddles* ptC = col_pow2 ? &ptw_cache_.get(cols, inverse) : nullptr;
        const packed_twiddles* ptR = row_pow2 ? &ptw_cache_.get(rows, inverse) : nullptr;

        // Small / sequential path: pre-allocate scratch once, zero per-row alloc
        if (total < kParallelThresh && !prefer_small_2d_avx2(rows, cols)) {
            // Pre-size work_buf_ to cover both row and column transforms.
            // All subsequent work.resize(n) calls inside run_stockham_double will be
            // no-ops as long as n <= max(rows, cols).
            work_buf_.resize(std::max(rows, cols));

            // Row passes — zero-copy: use row pointer directly as ping-pong buffer A.
            // loadu/storeu in Stockham stages handle any alignment.
            if (cols > 1u) {
                for (std::size_t r = 0; r < rows; ++r) {
                    _ComplexNumber* rp = raw.data() + r * cols;
                    if (col_pow2) {
                        execute_pow2_raw(rp, cols, work_buf_, *ptC, inverse);
                    } else {
                        tensor::view<_ComplexNumber> rv(std::span<_ComplexNumber>{rp, cols});
                        fallback_engine_.transform_1d(rv, inverse);
                    }
                }
            }

            // Column passes — col_buf allocated ONCE, reused for every column
            if (rows > 1u) {
                aligned_vec col_buf(rows);
                auto xform_col = [&]() {
                    if (row_pow2) {
                        if constexpr (is_complex_double) {
                            if (rows ==  2) { avx2_detail::codelet_2(reinterpret_cast<double*>(col_buf.data()), inverse); return; }
                            if (rows ==  4) { avx2_detail::codelet_4(reinterpret_cast<double*>(col_buf.data()), inverse); return; }
                        }
                        run_stockham_double(col_buf, work_buf_, *ptR, inverse);
                    } else {
                        tensor::view<_ComplexNumber> cv(std::span<_ComplexNumber>{col_buf.data(), rows});
                        fallback_engine_.transform_1d(cv, inverse);
                    }
                };
                for (std::size_t c = 0; c < cols; ++c) {
                    for (std::size_t r = 0; r < rows; ++r) col_buf[r] = raw[r * cols + c];
                    xform_col();
                    for (std::size_t r = 0; r < rows; ++r) raw[r * cols + c] = col_buf[r];
                }
            }
            return immediate_awaitable{};
        }

        // Large / parallel path: tiled column pass with OpenMP
        // Row passes
#pragma omp parallel
        {
            aligned_vec local_work;
#pragma omp for schedule(static)
            for (std::ptrdiff_t r = 0; r < static_cast<std::ptrdiff_t>(rows); ++r) {
                _ComplexNumber* rp = raw.data() + static_cast<std::size_t>(r) * cols;
                if (col_pow2)
                    execute_pow2_raw(rp, cols, local_work, *ptC, inverse);
                else {
                    tensor::view<_ComplexNumber> rv(std::span<_ComplexNumber>{rp, cols});
                    fallback_engine_.transform_1d(rv, inverse);
                }
            }
        }

        // Column passes — tiled and transposed so each column FFT runs on a
        // contiguous span inside the per-thread tile buffer.
        if (rows > 1u) {
#pragma omp parallel
            {
                aligned_vec tile(rows * kColTile);
                aligned_vec lw;

#pragma omp for schedule(static)
                for (std::ptrdiff_t c0 = 0; c0 < static_cast<std::ptrdiff_t>(cols);
                     c0 += static_cast<std::ptrdiff_t>(kColTile)) {
                    const std::size_t nc = std::min(kColTile, cols - static_cast<std::size_t>(c0));

                    for (std::size_t r = 0; r < rows; ++r) {
                        _mm_prefetch(reinterpret_cast<const char*>(&raw[r * cols + c0 + kColTile]),
                                     _MM_HINT_T1);
                        for (std::size_t tc = 0; tc < nc; ++tc)
                            tile[tc * rows + r] = raw[r * cols + c0 + tc];
                    }

                    for (std::size_t tc = 0; tc < nc; ++tc) {
                        _ComplexNumber* cp = tile.data() + tc * rows;
                        if (row_pow2)
                            execute_pow2_raw(cp, rows, lw, *ptR, inverse);
                        else {
                            tensor::view<_ComplexNumber> cv(std::span<_ComplexNumber>{cp, rows});
                            fallback_engine_.transform_1d(cv, inverse);
                        }
                    }

                    for (std::size_t r = 0; r < rows; ++r)
                        for (std::size_t tc = 0; tc < nc; ++tc)
                            raw[r * cols + c0 + tc] = tile[tc * rows + r];
                }
            }
        }
        return immediate_awaitable{};
    }

    template <typename _InputComplexNumber>
    immediate_awaitable transform_2d(const tensor::view<_InputComplexNumber>& v_in,
                                     tensor::view<_ComplexNumber> v_out, bool inverse) {
        if (reinterpret_cast<const void*>(v_in.data().data()) !=
            reinterpret_cast<const void*>(v_out.data().data()))
            std::copy(v_in.data().begin(), v_in.data().end(), v_out.data().begin());
        return transform_2d(v_out, inverse);
    }
};

} // namespace kmx::fft::backend

// Copyright (c) 2026 - present KMX Systems. All rights reserved.
/// @file fft_router.hpp
/// @brief Provides a runtime FFT strategy router facade.
#pragma once

#include <concepts>
#include <optional>
#include <kmx/tensor.hpp>
#include <kmx/fft_backend/concepts.hpp>
#include <kmx/fft_backend/capabilities.hpp>
#include <kmx/fft_backend/software.hpp>

// Include statically enabled backends
#ifdef KMX_FFT_ENABLE_AVX2
#include <kmx/fft_backend/avx2.hpp>
#endif

#ifdef KMX_FFT_ENABLE_OPENCL
#include <kmx/fft_backend/opencl.hpp>
#endif

namespace kmx::fft
{

/// @brief Runtime facade engine that routes FFT operations to the best backend.
/// @tparam _ComplexNumber The complex number type (e.g. std::complex<double>).
template <ComplexNumber _ComplexNumber>
class dynamic_engine
{
public:
    using complex_number_type = _ComplexNumber;
    using float_type = typename complex_number_type::value_type;
    using internal_index_type = tensor::index;
    using allocator_type = std::allocator<complex_number_type>; // Or pick a suitable one

    dynamic_engine()
    {
        // Initialize capabilities and check health once.
        cap_sw_ = {
            .id = backend::backend_id::software, .enabled = true,
            .supports_1d_pow2 = true, .supports_1d_smooth = true, .supports_2d = true,
            .supports_float = true, .supports_double = true
        };

        #ifdef KMX_FFT_ENABLE_AVX2
        cap_avx2_ = {
            .id = backend::backend_id::avx2,
            .enabled = __builtin_cpu_supports("avx2") > 0, // Rough runtime health check
            .supports_1d_pow2 = true, .supports_1d_smooth = true, .supports_2d = true,
            .supports_float = true, .supports_double = true
        };
        #else
        cap_avx2_ = { .id = backend::backend_id::avx2, .enabled = false };
        #endif

        #ifdef KMX_FFT_ENABLE_OPENCL
        // Determine OpenCL availability (would do real init/health check here, for now assume true if compiled)
        cap_ocl_ = {
            .id = backend::backend_id::opencl, .enabled = true, 
            .supports_1d_pow2 = true, .supports_1d_smooth = true, 
            .supports_2d = false, // OpenCL 2D explicitly marked unsupported
            .supports_float = true, .supports_double = true
        };
        #else
        cap_ocl_ = { .id = backend::backend_id::opencl, .enabled = false };
        #endif
        
        // AVX512 placeholder
        cap_avx512_ = { .id = backend::backend_id::avx512, .enabled = false };
    }

    // Helper to determine route
    backend::backend_id route_1d(std::size_t n, bool is_pow2) const {
        if (is_pow2) {
            if (n <= 16) return backend::backend_id::software; // Software overhead is lower for <= 16
            if (cap_avx2_.enabled) return backend::backend_id::avx2;
            if (n >= 4096 && cap_ocl_.enabled) return backend::backend_id::opencl; // Large size OpenCL logic
            return backend::backend_id::software;
        } else {
            if (n <= 31) return backend::backend_id::software; // Simple loops faster than bluestein SIMD overhead
            if (cap_avx2_.enabled) return backend::backend_id::avx2; // Good for exact smooth sizes
            return backend::backend_id::software;
        }
    }

    backend::backend_id route_2d(std::size_t rows, std::size_t cols) const {
        (void)rows; (void)cols;
        if (cap_avx2_.enabled) return backend::backend_id::avx2;
        return backend::backend_id::software;
    }

    bool is_power_of_two(std::size_t n) const { return (n & (n - 1)) == 0; }

    decltype(auto) transform_1d(tensor::view<complex_number_type> tensor_obj, bool inverse = false)
    {
        last_routed_ = route_1d(tensor_obj.size(), is_power_of_two(tensor_obj.size()));
        return dispatch_1d(tensor_obj, inverse);
    }
    
    template <typename _InputComplexNumber>
    decltype(auto) transform_1d(const tensor::view<_InputComplexNumber>& input_view, tensor::view<complex_number_type> output_view, bool inverse = false)
    {
        last_routed_ = route_1d(input_view.size(), is_power_of_two(input_view.size()));
        return dispatch_1d_oop(input_view, output_view, inverse);
    }

    decltype(auto) transform_2d(tensor::view<complex_number_type> tensor_obj, bool inverse = false)
    {
        last_routed_ = route_2d(tensor_obj.shape()[0], tensor_obj.shape()[1]);
        return dispatch_2d(tensor_obj, inverse);
    }
    
    template <typename _InputComplexNumber>
    decltype(auto) transform_2d(const tensor::view<_InputComplexNumber>& input_view, tensor::view<complex_number_type> output_view, bool inverse = false)
    {
        last_routed_ = route_2d(input_view.shape()[0], input_view.shape()[1]);
        return dispatch_2d_oop(input_view, output_view, inverse);
    }

    struct dynamic_plan {
        dynamic_engine* engine;
        backend::backend_id routed_id;
        
        #ifdef KMX_FFT_ENABLE_AVX2
        std::optional<typename backend::avx2<complex_number_type>::plan> avx2_plan;
        #endif

        bool inverse;

        void execute(tensor::view<complex_number_type> v) {
            #ifdef KMX_FFT_ENABLE_AVX2
            if (routed_id == backend::backend_id::avx2 && avx2_plan) {
                avx2_plan->execute(v);
                return;
            }
            #endif
            engine->transform_1d(v, inverse);
        }
    };

    auto make_plan(const std::size_t n, const bool inverse = false)
    {
        backend::backend_id routed = route_1d(n, is_power_of_two(n));
        dynamic_plan plan{this, routed, 
        #ifdef KMX_FFT_ENABLE_AVX2
            std::nullopt, 
        #endif
            inverse};

        #ifdef KMX_FFT_ENABLE_AVX2
        if (routed == backend::backend_id::avx2) {
            plan.avx2_plan = backend_avx2_.make_plan(n, inverse);
        }
        #endif
        return plan;
    }

    struct dynamic_plan_2d {
        dynamic_engine* engine;
        backend::backend_id routed_id;

        #ifdef KMX_FFT_ENABLE_AVX2
        std::optional<typename backend::avx2<complex_number_type>::plan_2d> avx2_plan;
        #endif

        bool inverse;
        
        void execute(tensor::view<complex_number_type> v) {
            #ifdef KMX_FFT_ENABLE_AVX2
            if (routed_id == backend::backend_id::avx2 && avx2_plan) {
                avx2_plan->execute(v);
                return;
            }
            #endif
            engine->transform_2d(v, inverse);
        }
    };

    auto make_plan_2d(const std::size_t rows, const std::size_t cols, const bool inverse = false)
    {
        backend::backend_id routed = route_2d(rows, cols);
        dynamic_plan_2d plan{this, routed,
        #ifdef KMX_FFT_ENABLE_AVX2
            std::nullopt, 
        #endif
            inverse};

        #ifdef KMX_FFT_ENABLE_AVX2
        if (routed == backend::backend_id::avx2) {
            plan.avx2_plan = backend_avx2_.make_plan_2d(rows, cols, inverse);
        }
        #endif
        return plan;
    }

    backend::backend_id get_last_strategy() const { return last_routed_; }

private:
    backend::software<complex_number_type> backend_sw_;
    #ifdef KMX_FFT_ENABLE_AVX2
    backend::avx2<complex_number_type> backend_avx2_;
    #endif
    #ifdef KMX_FFT_ENABLE_OPENCL
    backend::opencl<complex_number_type> backend_ocl_;
    #endif

    backend::capabilities cap_sw_;
    backend::capabilities cap_avx2_;
    backend::capabilities cap_avx512_;
    backend::capabilities cap_ocl_;

    backend::backend_id last_routed_ = backend::backend_id::software;

    decltype(auto) dispatch_1d(tensor::view<complex_number_type> tensor_obj, bool inverse) {
        #ifdef KMX_FFT_ENABLE_AVX2
        if (last_routed_ == backend::backend_id::avx2) return backend_avx2_.transform_1d(tensor_obj, inverse);
        #endif
        #ifdef KMX_FFT_ENABLE_OPENCL
        if (last_routed_ == backend::backend_id::opencl) return backend_ocl_.transform_1d(tensor_obj, inverse);
        #endif
        return backend_sw_.transform_1d(tensor_obj, inverse);
    }

    template <typename _InputComplexNumber>
    decltype(auto) dispatch_1d_oop(const tensor::view<_InputComplexNumber>& input_view, tensor::view<complex_number_type> output_view, bool inverse) {
        #ifdef KMX_FFT_ENABLE_AVX2
        if (last_routed_ == backend::backend_id::avx2) return backend_avx2_.transform_1d(input_view, output_view, inverse);
        #endif
        #ifdef KMX_FFT_ENABLE_OPENCL
        if (last_routed_ == backend::backend_id::opencl) return backend_ocl_.transform_1d(input_view, output_view, inverse);
        #endif
        return backend_sw_.transform_1d(input_view, output_view, inverse);
    }

    decltype(auto) dispatch_2d(tensor::view<complex_number_type> tensor_obj, bool inverse) {
        #ifdef KMX_FFT_ENABLE_AVX2
        if (last_routed_ == backend::backend_id::avx2) return backend_avx2_.transform_2d(tensor_obj, inverse);
        #endif
        return backend_sw_.transform_2d(tensor_obj, inverse);
    }
    
    template <typename _InputComplexNumber>
    decltype(auto) dispatch_2d_oop(const tensor::view<_InputComplexNumber>& input_view, tensor::view<complex_number_type> output_view, bool inverse) {
        #ifdef KMX_FFT_ENABLE_AVX2
        if (last_routed_ == backend::backend_id::avx2) return backend_avx2_.transform_2d(input_view, output_view, inverse);
        #endif
        return backend_sw_.transform_2d(input_view, output_view, inverse);
    }
};

} // namespace kmx::fft
// Copyright (c) 2025 - present KMX Systems. All rights reserved.
/// @file fft.hpp
/// @brief Provides 1D and 2D Fast Fourier Transform implementations using tensor views via an engine class.
#pragma once
#ifndef PCH
    #include <concepts>
    #include <kmx/tensor.hpp>
    #include <kmx/fft_backend/concepts.hpp>
    #include <kmx/fft_backend/software.hpp>
#endif

namespace kmx::fft
{

/// @brief Configurable engine that delegates FFT operations to a designated backend.
/// @tparam _ComplexNumber The complex number type (e.g. std::complex<double>).
/// @tparam Backend The specific hardware or software execution backend.
template <ComplexNumber _ComplexNumber, backend::FftBackend<_ComplexNumber> Backend = backend::software<_ComplexNumber>>
class engine
{
public:
    /// @brief Complex number type.
    using complex_number_type = _ComplexNumber;
    
    /// @brief Underlying float type.
    using float_type = typename complex_number_type::value_type;
    
    /// @brief Internal index type used for tensor iteration.
    using internal_index_type = tensor::index;
    
    /// @brief Expose the allocator type required by the FftBackend concept.
    using allocator_type = typename Backend::allocator_type;
    
    /// @brief Associated backend type.
    using backend_type = Backend;

    engine() = default;

    engine(const engine&) = delete;
    engine(engine&&) = delete;
    engine& operator=(const engine&) = delete;
    engine& operator=(engine&&) = delete;
    ~engine() = default;

    decltype(auto) transform_1d(tensor::view<complex_number_type> tensor_obj, bool inverse = false)
    {
        return backend_.transform_1d(tensor_obj, inverse);
    }

    template <typename _InputComplexNumber>
    decltype(auto) transform_1d(const tensor::view<_InputComplexNumber>& input_view, tensor::view<complex_number_type> output_view, bool inverse = false)
    {
        return backend_.transform_1d(input_view, output_view, inverse);
    }

    decltype(auto) transform_2d(tensor::view<complex_number_type> tensor_obj, bool inverse = false)
    {
        return backend_.transform_2d(tensor_obj, inverse);
    }

    template <typename _InputComplexNumber>
    decltype(auto) transform_2d(const tensor::view<_InputComplexNumber>& input_view, tensor::view<complex_number_type> output_view, bool inverse = false)
    {
        return backend_.transform_2d(input_view, output_view, inverse);
    }

    template <typename B = backend_type>
        requires requires(B& backend) { backend.make_plan(std::size_t {}, false); }
    decltype(auto) make_plan(const std::size_t n, const bool inverse = false)
    {
        return backend_.make_plan(n, inverse);
    }

    template <typename B = backend_type>
        requires requires(B& backend) { backend.make_plan_2d(std::size_t {}, std::size_t {}, false); }
    decltype(auto) make_plan_2d(const std::size_t rows, const std::size_t cols, const bool inverse = false)
    {
        return backend_.make_plan_2d(rows, cols, inverse);
    }

private:
    backend_type backend_ {};
};

} // namespace kmx::fft

#include <kmx/fft_router.hpp>

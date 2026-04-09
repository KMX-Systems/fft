// Copyright (c) 2025 - present KMX Systems. All rights reserved.
/// @file concepts.hpp
/// @brief Defines the FftBackend concept for kmx::fft::engine template parameterization.
#pragma once

#include <concepts>
#include <coroutine>
#include <kmx/tensor.hpp>

namespace kmx::fft
{

template <typename T>
concept ComplexNumber = requires(T a, T b, typename T::value_type v) {
    typename T::value_type;
    { a + b } -> std::same_as<T>;
    { a - b } -> std::same_as<T>;
    { a * b } -> std::same_as<T>;
    { a / b } -> std::same_as<T>;
    { a *= b } -> std::same_as<T&>;
    { a /= b } -> std::same_as<T&>;
    { std::conj(a) } -> std::convertible_to<T>;
} && std::floating_point<typename T::value_type>;

namespace backend
{

// Awaitable concept for C++20 coroutines
template <typename T>
concept Awaitable = requires(T a) {
    { a.await_ready() } -> std::convertible_to<bool>;
    { a.await_suspend(std::noop_coroutine()) } -> std::same_as<void>;
    { a.await_resume() } -> std::same_as<void>;
};

/// @brief Concept defining the required interface for an FFT computation backend.
template <typename B, typename ComplexType>
concept FftBackend = requires(B backend, tensor::view<ComplexType> v_inout, const tensor::view<const ComplexType> v_in, tensor::view<ComplexType> v_out, bool inverse) {
    requires ComplexNumber<ComplexType>;

    // The backend must expose an allocator type for tensors
    typename B::allocator_type;

    // 1D In-place transform (must return an Awaitable)
    { backend.transform_1d(v_inout, inverse) } -> Awaitable;

    // 1D Out-of-place transform (must return an Awaitable)
    { backend.transform_1d(v_in, v_out, inverse) } -> Awaitable;

    // 2D In-place transform (must return an Awaitable)
    { backend.transform_2d(v_inout, inverse) } -> Awaitable;

    // 2D Out-of-place transform (must return an Awaitable)
    { backend.transform_2d(v_in, v_out, inverse) } -> Awaitable;
};

// Simple immediate awaitable for synchronous backends
struct immediate_awaitable {
    constexpr bool await_ready() const noexcept { return true; }
    constexpr void await_suspend(auto /*h*/) const noexcept {}
    constexpr void await_resume() const noexcept {}
};

} // namespace backend
} // namespace kmx::fft

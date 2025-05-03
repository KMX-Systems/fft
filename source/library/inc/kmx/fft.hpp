// Copyright (c) 2025 - present KMX Systems. All rights reserved.
/// @file fft.hpp
/// @brief Provides 1D and 2D Fast Fourier Transform implementations using tensor views via an engine class.
#pragma once
#ifndef PCH
    #include <algorithm>
    #include <bit>
    #include <cmath>
    #include <complex>
    #include <functional>
    #include <kmx/tensor.hpp>
    #include <mutex>
    #include <numbers>
    #include <shared_mutex>
    #include <stdexcept>
    #include <unordered_map>
    #include <vector>
#endif

/// @brief KMX library namespace.
namespace kmx::fft // Nested namespace syntax
{
    // Forward declare engine for use in cache classes if needed
    template <typename _ComplexNumber>
    class engine;

    /// @internal
    /// @brief Internal implementation details for FFT algorithms.
    namespace internal
    {
        /// @brief Internal alias for complex number type. Cleaner than repeating std::complex<float_type>.
        template <typename float_type>
        using complex_t = std::complex<float_type>;

        /// @brief Checks if a number is a power of 2. `n=0` is not a power of 2.
        [[nodiscard]] constexpr bool is_power_of_2(const std::size_t n) noexcept
        {
            // return std::has_single_bit(n); // Alternative using C++20 <bit> header
            return (n > 0u) && ((n & (n - 1u)) == 0u);
        }

        /// @brief Reorders elements in a view based on bit-reversed indices (in-place).
        /// Required for Cooley-Tukey algorithm. Assumes view is 1D.
        /// @tparam _ComplexNumber The complex number type.
        /// @param signal_view The non-const 1D view to reorder.
        template <typename _ComplexNumber>
        void bit_reverse_reorder(tensor::view<_ComplexNumber> signal_view) noexcept
        {
            static_assert(!tensor::view<_ComplexNumber>::is_const_type, "bit_reverse_reorder requires a non-const view");
            // Assume signal_view is rank 1, checked by callers if necessary.
            const std::size_t n = signal_view.size();
            if (n <= 1u)
                return; // No reordering needed for size 0 or 1

            // std::bit_width(n-1) gives the number of bits needed to represent indices up to n-1.
            const unsigned int num_bits = std::bit_width(n - 1u);

            // Iterate only up to n/2, swapping ensures full reorder.
            for (std::size_t i = 1u; i < n; ++i) // Start from 1, 0 stays at 0
            {
                std::size_t reversed_i = 0u;
                std::size_t temp_i = i;
                // Efficient bit reversal loop
                for (unsigned int k = 0u; k < num_bits; ++k)
                {
                    reversed_i = (reversed_i << 1u) | (temp_i & 1u);
                    temp_i >>= 1u;
                }

                // Swap only if the reversed index is greater to avoid double swaps
                if (reversed_i > i)
                    std::swap(signal_view[i], signal_view[reversed_i]);
            }
        }

        // Twiddle Factor Cache Class
        /// @internal
        /// @brief Manages caching of precomputed twiddle factors for Cooley-Tukey FFT. Thread-safe using shared_mutex.
        /// @tparam _ComplexNumber The complex number type used in FFT.
        template <typename _ComplexNumber>
        class twiddle_factor_cache
        {
            // Type alias for the template argument.
            using complex_number_type = _ComplexNumber;

            using float_type = typename _ComplexNumber::value_type;

            // Cache entry holds the computed factors
            struct cache_entry
            {
                std::vector<complex_number_type> factors;
            };

            using cache_map = std::unordered_map<std::size_t, cache_entry>;

            cache_map cache_;
            // Use shared_mutex for read/write locking
            mutable std::shared_mutex mutex_; // Mutable to allow locking in const member functions if needed (not strictly needed here)

        public:
            /// @brief Gets or computes twiddle factors for size n.
            /// @param n FFT size (must be power of 2 for Cooley-Tukey using these).
            /// @param inverse True for inverse FFT factors, false for forward.
            /// @return Const reference to the vector of twiddle factors.
            /// @throws std::bad_alloc If memory allocation fails.
            const std::vector<complex_number_type>& get(const std::size_t n, const bool inverse) // noexcept(false) implied
            {
                // Use top bit of size_t to store the inverse flag in the key.
                constexpr std::size_t inverse_flag_bit = (1ull << (sizeof(std::size_t) * 8u - 1u));
                const std::size_t cache_key = inverse ? (n | inverse_flag_bit) : n;

                // --- Double-Checked Locking Pattern ---

                // 1. First check (read lock): Try to find the key under a shared lock.
                {
                    std::shared_lock lock(mutex_); // Acquire shared lock (multiple readers allowed)
                    auto it = cache_.find(cache_key);
                    if (it != cache_.end())
                    {
                        return it->second.factors; // Found: return cached data
                    }
                } // Shared lock released here

                // 2. Prepare for potential write: Compute factors *outside* any lock.
                //    This allows other threads to proceed with reads or even compute the same factors
                //    if they also miss the cache concurrently.
                std::vector<complex_number_type> computed_factors;
                computed_factors.reserve(n); // Reserve capacity

                const float_type sign_for_angle = inverse ? float_type {1.0} : float_type {-1.0};
                if (n > 0u)
                {
                    const float_type base_theta = float_type {2.0} * std::numbers::pi_v<float_type> / static_cast<float_type>(n);
                    for (std::size_t k = 0u; k < n; ++k)
                    {
                        const float_type theta = sign_for_angle * base_theta * static_cast<float_type>(k);
                        computed_factors.emplace_back(std::cos(theta), std::sin(theta));
                    }
                }

                // 3. Second check and potential insert (write lock): Acquire an exclusive lock.
                {
                    std::unique_lock lock(mutex_); // Acquire exclusive lock (only one writer allowed)
                    // Need to check *again* in case another thread computed and inserted
                    // the factors between the shared lock release and acquiring the unique lock.
                    auto [it, inserted] = cache_.try_emplace(cache_key, cache_entry {std::move(computed_factors)});
                    // 'it' points to the element (either newly inserted or the one found by try_emplace)
                    // 'inserted' tells us if the emplace actually happened (useful for debugging, not strictly needed here)

                    // Return reference to the factors vector (either newly inserted or the one already present)
                    return it->second.factors;
                } // Unique lock released here
            }
        };

        // Bluestein Factor Cache Class
        /// @internal
        /// @brief Manages caching of precomputed chirp and filter factors for Bluestein's FFT algorithm. Thread-safe using shared_mutex.
        /// @tparam _ComplexNumber The complex number type used in FFT.
        template <typename _ComplexNumber>
        class bluestein_factor_cache
        {
        public:
            // Type alias for the template argument.
            using complex_number_type = _ComplexNumber;

            // Type alias for the recursive FFT function needed internally
            using transform_func = std::function<void(tensor::view<complex_number_type>, bool)>;

            // Cache entry holds necessary precomputed data
            struct cache_entry
            {
                std::vector<complex_number_type> chirp;      // W^-(k^2/2) or W^(k^2/2)
                std::vector<complex_number_type> filter_fft; // FFT of the Bluestein filter sequence
                std::size_t original_n;                      // Original FFT size N
                std::size_t padded_m;                        // Padded size M (power of 2 >= 2N-1)
            };

        private:
            using float_type = typename complex_number_type::value_type;
            using cache_map = std::unordered_map<std::size_t, cache_entry>;
            cache_map cache_;
            // Use shared_mutex for read/write locking
            mutable std::shared_mutex mutex_; // Mutable to allow locking in const member functions if needed (not strictly needed here)

        public:
            /// @brief Gets or computes Bluestein factors for size n.
            /// @param n FFT size (arbitrary).
            /// @param inverse True for inverse FFT, false for forward.
            /// @param recursive_transform Function to compute FFT (used for filter).
            /// @return Const reference to the cache entry containing factors.
            /// @throws std::bad_alloc If memory allocation fails. std::logic_error on inconsistency.
            const cache_entry& get(const std::size_t n, const bool inverse,
                                   const transform_func& recursive_transform) // noexcept(false) implied
            {
                // Use top bit for inverse flag in cache key
                constexpr std::size_t inverse_flag_bit = (1ull << (sizeof(std::size_t) * 8u - 1u));
                const std::size_t cache_key = inverse ? (n | inverse_flag_bit) : n;
                // Calculate required padded size M (must be power of 2 >= 2N-1)
                const std::size_t m = (n == 0u) ? 0u : std::bit_ceil(2u * n - 1u); // Handles n=0

                // --- Double-Checked Locking Pattern ---

                // 1. First check (read lock): Try to find the key under a shared lock.
                {
                    std::shared_lock lock(mutex_); // Acquire shared lock
                    auto it = cache_.find(cache_key);
                    if (it != cache_.end())
                    {
                        // Sanity check: Ensure cached padded size matches current requirement
                        if (it->second.padded_m != m) [[unlikely]]
                            throw std::logic_error("Bluestein cache inconsistency: padded size mismatch.");

                        return it->second; // Found and consistent: return cached data
                    }
                } // Shared lock released here

                // 2. Prepare for potential write: Compute factors *outside* any lock.
                cache_entry computed_entry;
                computed_entry.original_n = n;
                computed_entry.padded_m = m;

                if (n > 0u) // Avoid computations for N=0
                {
                    // Compute chirp sequence: chirp[k] = exp(sign * i * pi * k^2 / N)
                    computed_entry.chirp.resize(n); // Resize only once
                    const float_type pi_val = std::numbers::pi_v<float_type>;
                    const float_type pi_over_n = pi_val / static_cast<float_type>(n);
                    const float_type sign = inverse ? float_type {1.0} : float_type {-1.0}; // Sign depends on forward/inverse

                    for (std::size_t k = 0u; k < n; ++k)
                    {
                        // Use unsigned long long for k*k to avoid potential overflow for large k before casting to float
                        unsigned long long k_squared = static_cast<unsigned long long>(k) * k;
                        float_type theta =
                            sign * pi_over_n *
                            static_cast<float_type>(k_squared %
                                                    (2 * n)); // Use modulo 2N property if needed, although direct calc is likely fine
                        computed_entry.chirp[k] = {std::cos(theta), std::sin(theta)};
                    }

                    // Compute Bluestein filter sequence h[k] = conj(chirp[k]) = exp(-sign * i * pi * k^2 / N)
                    // Padded to size M.
                    std::vector<_ComplexNumber> filter_vec(m, {float_type {0.0}, float_type {0.0}});
                    filter_vec[0u] = std::conj(computed_entry.chirp[0u]); // h[0] = conj(chirp[0]) = 1

                    for (std::size_t k = 1u; k < n; ++k) // Indices k = 1 to N-1
                    {
                        _ComplexNumber filter_term = std::conj(computed_entry.chirp[k]); // h[k]
                        filter_vec[k] = filter_term;                                     // Positive index k
                        if (m > k)                           // Avoid wrap-around if m == k (only happens if m=1, n=1)
                            filter_vec[m - k] = filter_term; // Negative index -k wraps around to M-k
                    }

                    // Compute FFT of the filter sequence using the provided recursive transform
                    tensor::view<_ComplexNumber> filter_view(filter_vec);
                    recursive_transform(filter_view, false);           // Always use forward FFT for filter
                    computed_entry.filter_fft = std::move(filter_vec); // Store the FFT'd filter
                }
                else // Handle N=0 case: empty factors
                {
                    computed_entry.chirp.clear();
                    computed_entry.filter_fft.clear(); // M is 0, so filter FFT is empty
                }

                // 3. Second check and potential insert (write lock): Acquire an exclusive lock.
                {
                    std::unique_lock lock(mutex_); // Acquire exclusive lock
                    // Need to check *again* in case another thread computed and inserted
                    // the factors between the shared lock release and acquiring the unique lock.
                    // Also re-verify consistency if found.
                    auto it = cache_.find(cache_key);
                    if (it != cache_.end())
                    {
                        // Entry exists, check consistency again before returning
                        if (it->second.padded_m != m) [[unlikely]]
                        {
                            // This indicates a logic error or hash collision with different M requirements, which shouldn't happen with the
                            // current key scheme. Depending on desired behavior, could overwrite or throw. Throwing is safer.
                            throw std::logic_error("Bluestein cache inconsistency detected during write lock: padded size mismatch.");
                        }
                        // It exists and is consistent, return it. The computed_entry will be discarded.
                        return it->second;
                    }
                    else
                    {
                        // Entry doesn't exist, insert the computed one
                        auto [insert_it, inserted] = cache_.emplace(cache_key, std::move(computed_entry));
                        // Return the newly inserted entry
                        return insert_it->second;
                    }
                    // Alternative using try_emplace if consistency check is primary concern:
                    // auto [it, inserted] = cache_.try_emplace(cache_key, std::move(computed_entry));
                    // // Check consistency *after* emplace attempt
                    // if (it->second.padded_m != m) [[unlikely]] {
                    //     // Handle inconsistency - maybe remove the bad entry? Or throw.
                    //     throw std::logic_error("Bluestein cache inconsistency after computation: padded size mismatch.");
                    // }
                    // return it->second;
                } // Unique lock released here
            }
        };

    } // namespace internal

    /// @brief Main class for performing FFT operations using cached factors and appropriate algorithms.
    /// Provides thread-safe 1D and 2D FFT/IFFT capabilities via tensor views.
    /// The engine itself is designed to be used safely by multiple threads concurrently,
    /// leveraging internally thread-safe caches.
    /// @tparam _ComplexNumber The complex number type (e.g., `std::complex<float>`, `std::complex<double>`).
    ///         Must have a `value_type` member defining the underlying float type.
    template <typename _ComplexNumber>
    class engine
    {
    public:
        using complex_number_type = _ComplexNumber;
        using float_type = typename _ComplexNumber::value_type; // Ensure _ComplexNumber provides this
        // Define the index type used internally, matching the default of tensor::view
        using internal_index_type = tensor::index;

    private:
        // Caches for precomputed factors (thread-safe internally via shared_mutex)
        internal::twiddle_factor_cache<complex_number_type> twiddle_cache_;
        internal::bluestein_factor_cache<complex_number_type> bluestein_cache_;

        // Private FFT Implementation Methods

        /// @brief Performs in-place Cooley-Tukey Radix-2 FFT. Assumes n is power of 2.
        /// Does NOT perform the final 1/N scaling for inverse transforms.
        void transform_power_of_2(tensor::view<complex_number_type> signal_view, const bool inverse)
        {
            static_assert(!tensor::view<complex_number_type>::is_const_type, "transform_power_of_2 requires a non-const view");
            const std::size_t n = signal_view.size();
            // Assume n > 1 and is power of 2 (checked by caller)

            internal::bit_reverse_reorder(signal_view); // In-place reordering

            // Get twiddle factors (thread-safe access to cache)
            const std::vector<complex_number_type>& twiddle_factors_n = twiddle_cache_.get(n, inverse);

            // Iterative Cooley-Tukey butterfly stages
            for (std::size_t segment_len = 2u; segment_len <= n; segment_len <<= 1u) // segment_len = 2, 4, 8, ..., N
            {
                const std::size_t half_segment_len = segment_len >> 1u; // segment_len / 2
                const std::size_t twiddle_step = n / segment_len;       // Step size in the twiddle factor table

                // Process segments within the signal
                for (std::size_t offset = 0u; offset < n; offset += segment_len)
                {
                    // Process butterflies within the segment
                    for (std::size_t k = 0u; k < half_segment_len; ++k)
                    {
                        // Twiddle factor W_segmentLen^k (or conj for inverse)
                        // Index into precomputed table: k * (N / segmentLen)
                        const complex_number_type& twiddle = twiddle_factors_n[k * twiddle_step];

                        const std::size_t even_idx = offset + k;
                        const std::size_t odd_idx = even_idx + half_segment_len;

                        // Perform butterfly: Use temporary to avoid overwriting signal_view[even_idx] prematurely
                        const _ComplexNumber odd_val_times_twiddle = signal_view[odd_idx] * twiddle;
                        const _ComplexNumber even_val_temp = signal_view[even_idx];

                        signal_view[even_idx] = even_val_temp + odd_val_times_twiddle;
                        signal_view[odd_idx] = even_val_temp - odd_val_times_twiddle;
                    }
                }
            }
        }

        /// @brief Performs in-place Bluestein's algorithm FFT for arbitrary size N.
        /// Does NOT perform the final 1/N scaling for inverse transforms.
        void transform_bluestein(tensor::view<_ComplexNumber> signal_view, const bool inverse)
        {
            static_assert(!tensor::view<_ComplexNumber>::is_const_type, "transform_bluestein requires a non-const view");
            const std::size_t n = signal_view.size();
            if (n <= 1u)
                return; // Base case: FFT of size 0 or 1 is identity

            // Lambda function for the recursive FFT calls needed by the cache
            // Captures 'this' to call back into the engine's core transform method.
            // Ensure this lambda has the correct signature expected by bluestein_factor_cache.
            typename internal::bluestein_factor_cache<_ComplexNumber>::transform_func recursive_transform =
                [this](tensor::view<_ComplexNumber> v, bool inv) { this->transform_1d_core(v, inv); };

            // Get Bluestein factors (thread-safe access to cache)
            const auto& factors = bluestein_cache_.get(n, inverse, recursive_transform);
            const std::size_t m = factors.padded_m; // Padded size M (power of 2)

            // Use cached factors
            const auto& chirp = factors.chirp;
            const auto& filter_fft = factors.filter_fft;

            // Check if factors are valid (m should not be 0 if n > 0)
            if (n > 0 && m == 0) [[unlikely]]
            {
                throw std::logic_error("Bluestein algorithm error: Padded size M is zero for non-zero N.");
            }
            // Also check if filter_fft size matches m
            if (filter_fft.size() != m) [[unlikely]]
            {
                throw std::logic_error("Bluestein algorithm error: Cached filter FFT size mismatch.");
            }

            // Allocate temporary padded vector
            // Note: This allocation happens per-call. Could be optimized with a thread-local buffer pool if this is a bottleneck.
            std::vector<_ComplexNumber> a_padded_vec(m, _ComplexNumber {float_type {0.0}, float_type {0.0}});

            // Step 1: Multiply input by chirp: a[k] = signal[k] * chirp[k]
            auto signal_data = signal_view.data(); // Get span/iterator to underlying data
            for (std::size_t k = 0u; k < n; ++k)
                a_padded_vec[k] = signal_data[k] * chirp[k];
            // The rest of a_padded_vec (from n to m-1) remains zero.

            // Step 2: Compute FFT of the padded sequence 'a' (using power-of-2 FFT since m is power of 2)
            tensor::view<_ComplexNumber> a_padded_view(a_padded_vec);
            // Crucially, use transform_power_of_2 directly IF m is guaranteed power-of-2 and > 1
            // Or just call transform_1d_core which will dispatch correctly.
            if (m > 1u)
            {                                                  // Only transform if size > 1
                this->transform_1d_core(a_padded_view, false); // Forward FFT of 'a'
            }

            // Step 3: Element-wise multiply FFT(a) with FFT(filter)
            tensor::view<const _ComplexNumber> filter_fft_view(filter_fft);
            if (a_padded_view.size() != filter_fft_view.size()) [[unlikely]] // Should already be checked by cache getter
                throw std::logic_error("Bluestein size mismatch between padded data and filter FFT");

            // Use std::ranges::transform or a simple loop
            auto a_padded_data = a_padded_view.data();     // Get span
            auto filter_fft_data = filter_fft_view.data(); // Get span
            for (std::size_t i = 0; i < m; ++i)
            {
                a_padded_data[i] *= filter_fft_data[i];
            }
            // Or: std::ranges::transform(a_padded_data, filter_fft_data, a_padded_data.begin(), std::multiplies<>{});

            // Step 4: Compute Inverse FFT of the product (without 1/M scaling yet)
            if (m > 1u)
            {                                                 // Only transform if size > 1
                this->transform_1d_core(a_padded_view, true); // Inverse FFT
            }

            // Step 5: Scale the result of IFFT by 1/M (apply the IFFT scaling for the convolution here)
            if (m > 0u)
            {
                const float_type inv_m_float = float_type {1.0} / static_cast<float_type>(m);
                const _ComplexNumber inv_m_complex = _ComplexNumber {inv_m_float, float_type {0.0}};
                auto result_data = a_padded_view.data();
                for (auto& val: result_data) // Using span simplifies loop
                    val *= inv_m_complex;
            }

            // Step 6: Multiply by chirp again and store final result back in original signal_view
            auto final_result_data = a_padded_view.data();
            for (std::size_t k = 0u; k < n; ++k)
                signal_data[k] = final_result_data[k] * chirp[k];
        }

        /// @brief Core 1D FFT/IFFT logic dispatcher (in-place). Selects algorithm.
        /// Does NOT perform the final 1/N scaling for inverse transforms.
        /// @param tensor_obj Non-const 1D tensor view.
        /// @param inverse True for IFFT, false for FFT.
        void transform_1d_core(tensor::view<_ComplexNumber> tensor_obj, bool inverse)
        {
            static_assert(!tensor::view<_ComplexNumber>::is_const_type, "transform_1d_core requires a non-const view");
            const std::size_t n = tensor_obj.size();

            if (n > 1u)
            {
                // Dispatch based on whether n is a power of 2
                if (internal::is_power_of_2(n))
                    transform_power_of_2(tensor_obj, inverse);
                else
                    transform_bluestein(tensor_obj, inverse);
            }
            // Base cases N=0 and N=1: do nothing (identity transform)
        }

    public:
        /// @brief Default constructor. Initializes internal caches.
        engine() = default;

        // Prevent copying and moving
        engine(const engine&) = delete;
        engine(engine&&) = delete;
        engine& operator=(const engine&) = delete;
        engine& operator=(engine&&) = delete;
        ~engine() = default; // Default destructor is fine

        // Public 1D Transform Methods

        /// @brief Performs in-place 1D FFT or IFFT on a tensor view.
        /// Includes the final 1/N scaling factor for inverse transforms.
        /// @param tensor_obj A non-const, rank-1 tensor view.
        /// @param inverse If `true`, performs Inverse FFT (IFFT), otherwise Forward FFT (FFT). Defaults to `false`.
        /// @throws std::invalid_argument If `tensor_obj` is not rank 1.
        void transform_1d(tensor::view<_ComplexNumber> tensor_obj, bool inverse = false)
        {
            static_assert(!tensor::view<_ComplexNumber>::is_const_type, "In-place transform requires a non-const view");
            const std::size_t n = tensor_obj.size();

            const std::size_t r = tensor_obj.rank();
            if (r != 1u) [[unlikely]]
                throw std::invalid_argument("engine::transform_1d (in-place) requires rank 1. Found rank " + std::to_string(r));

            if (n > 0u) // Avoid division by zero or processing empty views
            {
                transform_1d_core(tensor_obj, inverse); // Core computation

                // Apply final 1/N scaling *only* for inverse transforms and if N > 0.
                if (inverse)
                {
                    const float_type inv_n_float = float_type {1.0} / static_cast<float_type>(n);
                    const _ComplexNumber inv_n_complex = _ComplexNumber {inv_n_float, float_type {0.0}};
                    auto data = tensor_obj.data(); // Get span
                    for (auto& val: data)
                        val *= inv_n_complex;
                }
            }
        }

        /// @brief Performs out-of-place 1D FFT or IFFT.
        /// Includes the final 1/N scaling factor for inverse transforms.
        /// @tparam _InputComplexNumber The complex type of the input view (can be const).
        /// @param input_view A const or non-const, rank-1 tensor view for the input signal.
        /// @param output_view A non-const, rank-1 tensor view for the output. Must have the same shape as input.
        /// @param inverse If `true`, performs Inverse FFT (IFFT), otherwise Forward FFT (FFT). Defaults to `false`.
        /// @throws std::invalid_argument If shapes mismatch, ranks are not 1, or output view is const.
        template <typename _InputComplexNumber>
        void transform_1d(const tensor::view<_InputComplexNumber>& input_view, tensor::view<_ComplexNumber> output_view,
                          bool inverse = false)
        {
            // Check compatibility of float types
            static_assert(std::is_same_v<typename tensor::view<_InputComplexNumber>::non_const_value_type::value_type, float_type>,
                          "Input view float type must match engine float type for out-of-place transform.");
            // Check output view is non-const
            static_assert(!tensor::view<_ComplexNumber>::is_const_type, "Output view for out-of-place transform cannot be const");

            if (input_view.shape() != output_view.shape()) [[unlikely]]
                throw std::invalid_argument("engine::transform_1d (out-of-place): Input and output shapes must match.");

            if (input_view.rank() != 1u) [[unlikely]]
                throw std::invalid_argument("engine::transform_1d (out-of-place) requires input view rank 1. Found rank " +
                                            std::to_string(input_view.rank()));
            // No need to check output rank if shapes match and input rank is 1.

            // Copy input to output, then perform in-place transform on output
            if (!input_view.empty()) // Check for non-empty before copying/transforming
            {
                // Use std::ranges::copy for potentially better efficiency/vectorization
                std::ranges::copy(input_view.data(), output_view.data().begin());
                this->transform_1d(output_view, inverse); // Perform in-place on output buffer
            }
            // If empty, do nothing (output remains unchanged/empty as per shape match)
        }

        // Public 2D Transform Methods

        /// @brief Performs in-place 2D FFT or IFFT on a tensor view using the row-column method.
        /// Includes the final 1/(rows*cols) scaling factor for inverse transforms.
        /// @param tensor_obj A non-const, rank-1 or rank-2 tensor view. If rank 1, treats as 1D FFT.
        /// @param inverse If `true`, performs Inverse FFT (IFFT), otherwise Forward FFT (FFT). Defaults to `false`.
        /// @throws std::invalid_argument If `tensor_obj` has rank other than 1 or 2.
        void transform_2d(tensor::view<_ComplexNumber> tensor_obj, bool inverse = false)
        {
            static_assert(!tensor::view<_ComplexNumber>::is_const_type, "In-place 2D transform requires a non-const view");
            const std::size_t current_rank = tensor_obj.rank();

            if (tensor_obj.empty())
                return; // Nothing to do for empty tensors

            // Delegate to 1D transform if rank is 1
            if (current_rank == 1u)
            {
                this->transform_1d(tensor_obj, inverse); // Handles scaling correctly for 1D
                return;                                  // Done
            }
            else if (current_rank != 2u) [[unlikely]]
            {
                throw std::invalid_argument("engine::transform_2d (in-place) input view must have rank 1 or 2. Found rank " +
                                            std::to_string(current_rank));
            }

            // Proceed with Rank 2
            const tensor::dimension_shape current_shape = tensor_obj.shape();
            const std::size_t rows = current_shape[0u];
            const std::size_t cols = current_shape[1u];

            // If either dimension is 0 or 1, FFT along that dimension is trivial.
            // The 1/N scaling logic in transform_1d handles N=1 correctly (multiplies by 1.0).

            // Step 1: Perform 1D FFT/IFFT on each row
            if (cols > 0u) // Avoid loop if cols is 0
            {
                for (std::size_t r = 0u; r < rows; ++r)
                {
                    auto row_t_view = tensor_obj.row_view(r);
                    // Apply 1D transform (includes 1/cols scaling if inverse and cols > 0)
                    // Note: transform_1d handles scaling internally based on 'inverse' flag and row_t_view.size() (which is 'cols')
                    this->transform_1d(row_t_view, inverse);
                }
            }

            // Step 2: Perform 1D FFT/IFFT on each column
            if (rows > 0u) // Avoid loop if rows is 0
            {
                // Need a temporary buffer for columns as they are not contiguous in memory.
                // This buffer can be reused for all columns.
                // Optimization: If rows == 1, this loop is skipped anyway. If rows > 1, allocate.
                std::vector<_ComplexNumber> temp_col_vec(rows); // Allocate buffer of size 'rows'
                tensor::view<_ComplexNumber> temp_col_view(temp_col_vec);

                for (std::size_t c = 0u; c < cols; ++c)
                {
                    // Copy column 'c' to the temporary buffer
                    // This copy is necessary because tensor columns are strided.
                    for (std::size_t r = 0u; r < rows; ++r)
                        // Use multi-dimensional index accessor
                        temp_col_vec[r] = tensor_obj[internal_index_type {r, c}];

                    // Perform 1D transform on the temporary column buffer
                    // Note: transform_1d handles scaling (1/rows) internally if inverse=true and rows > 0.
                    this->transform_1d(temp_col_view, inverse);

                    // Copy the transformed column back from the buffer
                    for (std::size_t r = 0u; r < rows; ++r)
                        // Use multi-dimensional index accessor
                        tensor_obj[internal_index_type {r, c}] = temp_col_view[r];
                }
            }

            // Overall scaling check for inverse FFT:
            // - If inverse=true, row transforms applied scaling of 1/cols (if cols > 0).
            // - If inverse=true, col transforms applied scaling of 1/rows (if rows > 0).
            // - Total scaling applied = (1/cols) * (1/rows) = 1/(rows*cols), which is correct for 2D IFFT.
            // - Handles cases where rows or cols are 0 or 1 correctly due to scaling logic in transform_1d.
        }

        /// @brief Performs out-of-place 2D FFT or IFFT using the row-column method.
        /// Includes the final 1/(rows*cols) scaling factor for inverse transforms.
        /// @tparam _InputComplexNumber The complex type of the input view (can be const).
        /// @param input_view A const or non-const, rank-1 or rank-2 tensor view for the input signal.
        /// @param output_view A non-const, rank-1 or rank-2 tensor view for the output. Must have the same shape as input.
        /// @param inverse If `true`, performs Inverse FFT (IFFT), otherwise Forward FFT (FFT). Defaults to `false`.
        /// @throws std::invalid_argument If shapes mismatch, ranks are invalid, or output view is const.
        template <typename _InputComplexNumber>
        void transform_2d(const tensor::view<_InputComplexNumber>& input_view, tensor::view<_ComplexNumber> output_view,
                          bool inverse = false)
        {
            static_assert(std::is_same_v<typename tensor::view<_InputComplexNumber>::non_const_value_type::value_type, float_type>,
                          "Input view float type must match engine float type for out-of-place 2D transform.");
            static_assert(!tensor::view<_ComplexNumber>::is_const_type, "Output view for out-of-place 2D transform cannot be const");

            if (input_view.shape() != output_view.shape()) [[unlikely]]
                throw std::invalid_argument("engine::transform_2d (out-of-place): Input and output shapes must match.");

            const std::size_t current_rank = input_view.rank();
            if (current_rank != 1u && current_rank != 2u) [[unlikely]]
                throw std::invalid_argument("engine::transform_2d (out-of-place) input view must have rank 1 or 2. Found rank " +
                                            std::to_string(current_rank));
            // No need to check output rank due to shape match check.

            if (!input_view.empty()) // Check for non-empty before copying/transforming
            {
                // Copy input to output
                std::ranges::copy(input_view.data(), output_view.data().begin());
                // Perform in-place transform on the output buffer
                this->transform_2d(output_view, inverse);
            }
            // If empty, do nothing (output remains unchanged/empty).
        }

    }; // class engine
} // namespace kmx::fft

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
    #include <stdexcept>
    #include <unordered_map>
    #include <vector>
#endif

/// @brief KMX library namespace.
namespace kmx::fft // Nested namespace syntax
{
    // Forward declare engine for use in cache classes if needed
    template <typename complex_number_type>
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
        /// @tparam complex_number_type The complex number type.
        /// @param signal_view The non-const 1D view to reorder.
        template <typename complex_number_type>
        void bit_reverse_reorder(tensor::view<complex_number_type> signal_view) noexcept
        {
            static_assert(!tensor::view<complex_number_type>::is_const_type, "bit_reverse_reorder requires a non-const view");
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
        /// @brief Manages caching of precomputed twiddle factors for Cooley-Tukey FFT. Thread-safe.
        /// @tparam complex_number_type The complex number type used in FFT.
        template <typename complex_number_type>
        class twiddle_factor_cache
        {
            using float_type = typename complex_number_type::value_type;
            // Cache entry holds the computed factors
            struct cache_entry
            {
                std::vector<complex_number_type> factors;
            };

            using cache_map = std::unordered_map<std::size_t, cache_entry>;

            cache_map cache_;
            std::mutex mutex_; // Mutex protecting the cache map

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

                // Double-checked locking pattern for efficiency
                // First check (read): without lock
                // Note: Reading cache_ might race, but finding an element is okay. Missing it leads to lock.
                // Consider std::shared_mutex if read contention is high and C++17+ is available.
                {
                    // Scoped lock for read access (if using shared_mutex, use std::shared_lock)
                    // std::shared_lock lock(mutex_); // C++17+ alternative
                    // For std::mutex, we still need exclusive lock for find/read for safety if map rehashes etc.
                    // Let's stick to simple lock_guard for now.
                    std::lock_guard lock(mutex_);
                    auto it = cache_.find(cache_key);
                    if (it != cache_.end())
                        return it->second.factors;
                } // Read lock released

                // Factors not found, compute them (outside the lock initially)
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

                // Second check and insertion: Lock again (exclusive)
                {
                    std::lock_guard lock(mutex_);
                    // Use try_emplace: inserts only if key doesn't exist, avoids moving vector if already inserted by another thread.
                    auto [it, inserted] = cache_.try_emplace(cache_key, cache_entry {std::move(computed_factors)});
                    // Return reference to the factors vector (either newly inserted or the one found by try_emplace)
                    return it->second.factors;
                }
            }
        };

        // Bluestein Factor Cache Class
        /// @internal
        /// @brief Manages caching of precomputed chirp and filter factors for Bluestein's FFT algorithm. Thread-safe.
        /// @tparam complex_number_type The complex number type used in FFT.
        template <typename complex_number_type>
        class bluestein_factor_cache
        {
        public:
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
            std::mutex mutex_; // Mutex protecting the cache map

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

                // Double-checked locking pattern
                {
                    std::lock_guard lock(mutex_); // Lock for read access
                    auto it = cache_.find(cache_key);
                    if (it != cache_.end())
                    {
                        // Sanity check: Ensure cached padded size matches current requirement
                        if (it->second.padded_m != m) [[unlikely]]
                            throw std::logic_error("Bluestein cache inconsistency: padded size mismatch.");

                        return it->second; // Return existing entry
                    }
                } // Read lock released

                // Factors not found, compute them (outside the lock initially)
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
                        unsigned long long k_squared = static_cast<unsigned long long>(k) * k;
                        float_type theta = sign * pi_over_n * static_cast<float_type>(k_squared);
                        computed_entry.chirp[k] = {std::cos(theta), std::sin(theta)};
                    }

                    // Compute Bluestein filter sequence h[k] = conj(chirp[k]) = exp(-sign * i * pi * k^2 / N)
                    // Padded to size M.
                    std::vector<complex_number_type> filter_vec(m, {float_type {0.0}, float_type {0.0}});
                    filter_vec[0u] = std::conj(computed_entry.chirp[0u]); // h[0] = conj(chirp[0]) = 1

                    for (std::size_t k = 1u; k < n; ++k) // Indices k = 1 to N-1
                    {
                        complex_number_type filter_term = std::conj(computed_entry.chirp[k]); // h[k]
                        filter_vec[k] = filter_term;                                          // Positive index k
                        if (m > k)
                            filter_vec[m - k] = filter_term; // Negative index -k wraps around to M-k
                    }

                    // Compute FFT of the filter sequence using the provided recursive transform
                    tensor::view<complex_number_type> filter_view(filter_vec);
                    recursive_transform(filter_view, false);           // Always use forward FFT for filter
                    computed_entry.filter_fft = std::move(filter_vec); // Store the FFT'd filter
                }
                else // Handle N=0 case: empty factors
                {
                    computed_entry.chirp.clear();
                    computed_entry.filter_fft.clear(); // M is 0, so filter FFT is empty
                }

                // Second check and insertion: Lock again (exclusive)
                {
                    std::lock_guard lock(mutex_);
                    // Use try_emplace
                    auto [it, inserted] = cache_.try_emplace(cache_key, std::move(computed_entry));
                    // Sanity check padded size again after potential race
                    if (it->second.padded_m != m) [[unlikely]]
                        throw std::logic_error("Bluestein cache inconsistency after computation: padded size mismatch.");

                    return it->second;
                }
            }
        };

    } // namespace internal

    /// @brief Main class for performing FFT operations using cached factors and appropriate algorithms.
    /// Provides thread-safe 1D and 2D FFT/IFFT capabilities via tensor views.
    /// @tparam _ComplexNumber The complex number type (e.g., `std::complex<float>`, `std::complex<double>`).
    ///         Must have a `value_type` member defining the underlying float type.
    template <typename _ComplexNumber>
    class engine
    {
    public:
        using complex_number_type = _ComplexNumber;
        using float_type = typename complex_number_type::value_type; // Ensure complex_number_type provides this
        // Define the index type used internally, matching the default of tensor::view
        using internal_index_type = kmx::tensor::index;

    private:
        // Caches for precomputed factors (thread-safe internally)
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

            // Get twiddle factors (potentially computes and caches them)
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
                        const complex_number_type odd_val_times_twiddle = signal_view[odd_idx] * twiddle;
                        const complex_number_type even_val_temp = signal_view[even_idx];

                        signal_view[even_idx] = even_val_temp + odd_val_times_twiddle;
                        signal_view[odd_idx] = even_val_temp - odd_val_times_twiddle;
                    }
                }
            }
        }

        /// @brief Performs in-place Bluestein's algorithm FFT for arbitrary size N.
        /// Does NOT perform the final 1/N scaling for inverse transforms.
        void transform_bluestein(tensor::view<complex_number_type> signal_view, const bool inverse)
        {
            static_assert(!tensor::view<complex_number_type>::is_const_type, "transform_bluestein requires a non-const view");
            const std::size_t n = signal_view.size();
            if (n <= 1u)
                return; // Base case: FFT of size 0 or 1 is identity

            // Lambda function for the recursive FFT calls needed by the cache
            typename internal::bluestein_factor_cache<complex_number_type>::transform_func recursive_transform =
                [this](tensor::view<complex_number_type> v, bool inv) { this->transform_1d_core(v, inv); };

            // Get Bluestein factors (computes/caches if needed)
            const auto& factors = bluestein_cache_.get(n, inverse, recursive_transform);
            const std::size_t m = factors.padded_m; // Padded size M (power of 2)

            // Use cached factors
            const auto& chirp = factors.chirp;
            const auto& filter_fft = factors.filter_fft;

            // Allocate temporary padded vector
            std::vector<complex_number_type> a_padded_vec(m, complex_number_type {float_type {0.0}, float_type {0.0}});

            // Step 1: Multiply input by chirp: a[k] = signal[k] * chirp[k]
            auto signal_data = signal_view.data();
            for (std::size_t k = 0u; k < n; ++k)
                a_padded_vec[k] = signal_data[k] * chirp[k];

            // Step 2: Compute FFT of the padded sequence 'a'
            tensor::view<complex_number_type> a_padded_view(a_padded_vec);
            this->transform_1d_core(a_padded_view, false); // Forward FFT of 'a'

            // Step 3: Element-wise multiply FFT(a) with FFT(filter)
            tensor::view<const complex_number_type> filter_fft_view(filter_fft);
            if (a_padded_view.size() != filter_fft_view.size()) [[unlikely]]
                throw std::logic_error("Bluestein size mismatch between padded data and filter FFT");

            std::ranges::transform(a_padded_view.data(), filter_fft_view.data(), a_padded_view.data().begin(), std::multiplies<> {});

            // Step 4: Compute Inverse FFT of the product (without 1/M scaling yet)
            this->transform_1d_core(a_padded_view, true); // Inverse FFT

            // Step 5: Scale the result of IFFT by 1/M
            if (m > 0u)
            {
                const float_type inv_m_float = float_type {1.0} / static_cast<float_type>(m);
                const complex_number_type inv_m_complex = complex_number_type {inv_m_float, float_type {0.0}};
                auto result_data = a_padded_view.data();
                for (auto& val: result_data)
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
        void transform_1d_core(tensor::view<complex_number_type> tensor_obj, bool inverse)
        {
            static_assert(!tensor::view<complex_number_type>::is_const_type, "transform_1d_core requires a non-const view");
            const std::size_t n = tensor_obj.size();

            if (n > 1u)
            {
                // Dispatch based on whether n is a power of 2
                if (internal::is_power_of_2(n))
                    transform_power_of_2(tensor_obj, inverse);
                else
                    transform_bluestein(tensor_obj, inverse);
            }
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
        void transform_1d(tensor::view<complex_number_type> tensor_obj, bool inverse = false)
        {
            static_assert(!tensor::view<complex_number_type>::is_const_type, "In-place transform requires a non-const view");
            const std::size_t n = tensor_obj.size();
            if (n > 0u)
            {
                const std::size_t r = tensor_obj.rank();
                if (r != 1u) [[unlikely]]
                    throw std::invalid_argument("engine::transform_1d (in-place) requires rank 1. Found rank " + std::to_string(r));

                transform_1d_core(tensor_obj, inverse); // Core computation

                // Apply final 1/N scaling *only* for inverse transforms.
                if (inverse)
                {
                    const float_type inv_n_float = float_type {1.0} / static_cast<float_type>(n);
                    const complex_number_type inv_n_complex = complex_number_type {inv_n_float, float_type {0.0}};
                    auto data = tensor_obj.data();
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
        void transform_1d(const tensor::view<_InputComplexNumber>& input_view, tensor::view<complex_number_type> output_view,
                          bool inverse = false)
        {
            static_assert(std::is_same_v<typename tensor::view<_InputComplexNumber>::non_const_value_type::value_type, float_type>,
                          "Input view float type must match engine float type for out-of-place transform.");
            static_assert(!tensor::view<complex_number_type>::is_const_type, "Output view for out-of-place transform cannot be const");

            if (input_view.shape() != output_view.shape()) [[unlikely]]
                throw std::invalid_argument("engine::transform_1d (out-of-place): Input and output shapes must match.");

            if (input_view.rank() != 1u) [[unlikely]]
                throw std::invalid_argument("engine::transform_1d (out-of-place) requires input view rank 1. Found rank " +
                                            std::to_string(input_view.rank()));
            if (!input_view.empty())
            {
                std::ranges::copy(input_view.data(), output_view.data().begin());
                this->transform_1d(output_view, inverse); // Perform in-place on output buffer
            }
        }

        // Public 2D Transform Methods

        /// @brief Performs in-place 2D FFT or IFFT on a tensor view using the row-column method.
        /// Includes the final 1/(rows*cols) scaling factor for inverse transforms.
        /// @param tensor_obj A non-const, rank-1 or rank-2 tensor view. If rank 1, treats as 1D FFT.
        /// @param inverse If `true`, performs Inverse FFT (IFFT), otherwise Forward FFT (FFT). Defaults to `false`.
        /// @throws std::invalid_argument If `tensor_obj` has rank other than 1 or 2.
        void transform_2d(tensor::view<complex_number_type> tensor_obj, bool inverse = false)
        {
            static_assert(!tensor::view<complex_number_type>::is_const_type, "In-place 2D transform requires a non-const view");
            const std::size_t current_rank = tensor_obj.rank();

            if (tensor_obj.empty())
                return;

            // Delegate to 1D transform if rank is 1
            if (current_rank == 1u)
                this->transform_1d(tensor_obj, inverse); // Handles scaling correctly for 1D
            else
            {
                if (current_rank != 2u) [[unlikely]]
                    throw std::invalid_argument("engine::transform_2d (in-place) input view must have rank 1 or 2. Found rank " +
                                                std::to_string(current_rank));

                const tensor::dimension_shape current_shape = tensor_obj.shape();
                const std::size_t rows = current_shape[0u];
                const std::size_t cols = current_shape[1u];

                // Step 1: Perform 1D FFT/IFFT on each row
                if (cols > 1u) // Optimization: Skip if cols = 0 or 1
                    for (std::size_t r = 0u; r < rows; ++r)
                    {
                        auto row_t_view = tensor_obj.row_view(r);
                        // Apply 1D transform (includes 1/cols scaling if inverse)
                        // Note: This transform_1d *does* apply scaling if inverse=true.
                        this->transform_1d(row_t_view, inverse);
                    }

                // Step 2: Perform 1D FFT/IFFT on each column
                if (rows > 1u) // Optimization: Skip if rows = 0 or 1
                {
                    // Need a temporary buffer for non-contiguous columns
                    std::vector<complex_number_type> temp_col_vec(rows);
                    tensor::view<complex_number_type> temp_col_view(temp_col_vec);

                    for (std::size_t c = 0u; c < cols; ++c)
                    {
                        // Copy column to temp buffer using multi-dimensional index
                        for (std::size_t r = 0u; r < rows; ++r)
                            // FIX: Use the fully qualified or aliased index type
                            temp_col_vec[r] = tensor_obj[internal_index_type {r, c}];

                        // Perform 1D transform on the temporary column buffer
                        // Note: This transform_1d *does* apply scaling (1/rows) if inverse=true.
                        this->transform_1d(temp_col_view, inverse);

                        // Copy transformed column back
                        for (std::size_t r = 0u; r < rows; ++r)
                            // FIX: Use the fully qualified or aliased index type
                            tensor_obj[internal_index_type {r, c}] = temp_col_view[r];
                    }
                }
                // Overall scaling for inverse:
                // Rows transformed with 1/cols scaling.
                // Columns transformed with 1/rows scaling.
                // Total scaling = (1/cols) * (1/rows) = 1/(rows*cols), which is correct.
            }
        }

        /// @brief Performs out-of-place 2D FFT or IFFT using the row-column method.
        /// Includes the final 1/(rows*cols) scaling factor for inverse transforms.
        /// @tparam _InputComplexNumber The complex type of the input view (can be const).
        /// @param input_view A const or non-const, rank-1 or rank-2 tensor view for the input signal.
        /// @param output_view A non-const, rank-1 or rank-2 tensor view for the output. Must have the same shape as input.
        /// @param inverse If `true`, performs Inverse FFT (IFFT), otherwise Forward FFT (FFT). Defaults to `false`.
        /// @throws std::invalid_argument If shapes mismatch, ranks are invalid, or output view is const.
        template <typename _InputComplexNumber>
        void transform_2d(const tensor::view<_InputComplexNumber>& input_view, tensor::view<complex_number_type> output_view,
                          bool inverse = false)
        {
            static_assert(std::is_same_v<typename tensor::view<_InputComplexNumber>::non_const_value_type::value_type, float_type>,
                          "Input view float type must match engine float type for out-of-place 2D transform.");
            static_assert(!tensor::view<complex_number_type>::is_const_type, "Output view for out-of-place 2D transform cannot be const");

            if (input_view.shape() != output_view.shape()) [[unlikely]]
                throw std::invalid_argument("engine::transform_2d (out-of-place): Input and output shapes must match.");

            const std::size_t current_rank = input_view.rank();
            if (current_rank != 1u && current_rank != 2u) [[unlikely]]
                throw std::invalid_argument("engine::transform_2d (out-of-place) input view must have rank 1 or 2. Found rank " +
                                            std::to_string(current_rank));

            if (!input_view.empty())
            {
                std::ranges::copy(input_view.data(), output_view.data().begin());
                this->transform_2d(output_view, inverse); // Perform in-place on output buffer
            }
        }

    }; // class engine
} // namespace kmx::fft

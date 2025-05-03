// Copyright (c) 2025 - present KMX Systems. All rights reserved.
/// @file fft_test.cpp
/// @brief Unit tests for the FFT implementation using Catch2 and engine.
#define CATCH_CONFIG_MAIN
#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <complex>
#include <kmx/fft.hpp>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/// @brief Contains helper functions and classes for FFT unit tests.
namespace kmx::fft::test_helpers // Use a nested namespace for test helpers
{
    // Use using-declaration for convenience within this namespace
    using kmx::tensor::dimension_shape;
    using kmx::tensor::index; // Make index type available for matcher if needed explicitly
    using kmx::tensor::view;

    /// @brief Checks if two std::complex numbers are approximately equal within tolerances.
    /// Uses a robust relative/absolute comparison method.
    /// @tparam float_type The underlying floating-point type.
    /// @param a First std::complex number.
    /// @param b Second std::complex number.
    /// @param rel_tolerance Relative tolerance for comparison.
    /// @param abs_tolerance Absolute tolerance for comparison (important near zero).
    /// @return `true` if numbers are close, `false` otherwise. Handles non-finite values by returning `false`.
    template <typename float_type>
    [[nodiscard]] bool complex_close(const std::complex<float_type>& a, const std::complex<float_type>& b, float_type rel_tolerance,
                                     float_type abs_tolerance) noexcept
    {
        // Non-finite numbers are never close to anything, including themselves if NaN
        if (!std::isfinite(a.real()) || !std::isfinite(a.imag()) || !std::isfinite(b.real()) || !std::isfinite(b.imag()))
        {
            return false;
        }

        // Check absolute difference first (handles comparison near zero)
        const float_type diff_real = std::abs(a.real() - b.real());
        const float_type diff_imag = std::abs(a.imag() - b.imag());
        if (diff_real <= abs_tolerance && diff_imag <= abs_tolerance)
        {
            return true;
        }

        // Check relative difference for larger numbers
        // Use max of absolute values as the reference scale
        const float_type norm_a = std::max(std::abs(a.real()), std::abs(a.imag()));
        const float_type norm_b = std::max(std::abs(b.real()), std::abs(b.imag()));
        const float_type max_norm = std::max(norm_a, norm_b);

        // Note: Avoid division by zero if max_norm is very small (handled by abs_tolerance check)
        // If max_norm is exactly zero, only the abs_tolerance check matters.
        if (max_norm == float_type {0.0})
        {
            return diff_real <= abs_tolerance && diff_imag <= abs_tolerance;
        }

        return diff_real <= rel_tolerance * max_norm && diff_imag <= rel_tolerance * max_norm;
    }

    /// @brief Checks if two tensor views are element-wise approximately equal.
    /// Compares shapes first, then compares elements using `complex_close`.
    /// @tparam T1 Element type of the first view (can be const).
    /// @tparam T2 Element type of the second view (can be const).
    /// @tparam IndexType The index type used by the views.
    /// @param t1 First tensor view.
    /// @param t2 Second tensor view.
    /// @param rel_tolerance Relative tolerance passed to `complex_close`.
    /// @return `true` if shapes match and all corresponding elements are close, `false` otherwise.
    template <typename T1, typename T2, typename IndexType = kmx::tensor::index>
    [[nodiscard]] bool tensor_close(const view<T1, IndexType>& t1, const view<T2, IndexType>& t2,
                                    typename view<T1, IndexType>::non_const_value_type::value_type rel_tolerance) noexcept
    {
        if (t1.shape() != t2.shape())
        {
            // std::cerr << "Shape mismatch: t1=" << t1.shape_str() << " vs t2=" << t2.shape_str() << std::endl; // Need a shape string
            // helper
            return false;
        }
        // If shapes match and one is empty, the other must be too.
        if (t1.empty())
        {
            return true; // Both must be empty
        }

        // Determine float type and absolute tolerance
        using float_type = typename view<T1, IndexType>::non_const_value_type::value_type;
        // Absolute tolerance slightly larger than epsilon accounts for minor accumulated errors.
        const float_type abs_tol = std::numeric_limits<float_type>::epsilon() * float_type {100.0};

        // Iterate using flat index for efficiency (already checked shapes match)
        const std::size_t total_size = t1.size();
        for (std::size_t i = 0u; i < total_size; ++i)
        {
            // Access using flat index operator[] which includes bounds check in our implementation
            if (!complex_close(t1[i], t2[i], rel_tolerance, abs_tol))
            {
                // std::cerr << "Mismatch at flat index " << i << ": " << t1[i] << " vs " << t2[i]
                //           << " (RelTol=" << rel_tolerance << ", AbsTol=" << abs_tol << ")" << std::endl;
                return false;
            }
        }
        return true;
    }

    /// @brief Custom Catch2 matcher for asserting tensor closeness.
    /// @tparam ComplexType The std::complex element type *without* const qualifier.
    /// @tparam IndexType The index type used by the view.
    template <typename ComplexType, typename IndexType = kmx::tensor::index>
    class tensor_close_matcher: public Catch::Matchers::MatcherBase<view<ComplexType, IndexType>> // Matches non-const view by default
    {
        // Store expected view as const, allowing comparison with both const and non-const actual views.
        view<const ComplexType, IndexType> m_expected;
        typename ComplexType::value_type m_rel_tolerance;

    public:
        /// @brief Constructor.
        /// @param expected The expected tensor view (pass as const or non-const, stored as const).
        /// @param rel_tolerance The relative tolerance for comparison.
        tensor_close_matcher(const view<const ComplexType, IndexType>& expected, typename ComplexType::value_type rel_tolerance):
            m_expected(expected),
            m_rel_tolerance(rel_tolerance)
        {
        }

        // Match against a non-const actual view (primary overload)
        bool match(const view<ComplexType, IndexType>& actual) const override
        {
            // tensor_close handles const vs non-const comparison correctly
            return tensor_close(actual, m_expected, m_rel_tolerance);
        }

        // Overload to allow matching directly against a const actual view
        bool match(const view<const ComplexType, IndexType>& actual) const // Overload for const view
        {
            return tensor_close(actual, m_expected, m_rel_tolerance);
        }

        /// @brief Describes the matcher for Catch2 output.
        /// @return A description string.
        std::string describe() const override
        {
            std::ostringstream ss;
            // Get the shape object via the public accessor
            const auto& expected_shape = m_expected.shape(); // Assuming view::shape() returns dimension_shape

            ss << "is element-wise close to expected tensor (shape [";
            // Manually format the shape dimensions
            bool first_dim = true;
            // Assuming dimension_shape is iterable or has begin()/end()
            for (const auto& dim_size: expected_shape)
            {
                if (!first_dim)
                    ss << ", ";
                ss << dim_size;
                first_dim = false;
            }
            ss << "], size " << m_expected.size() << ") within relative tolerance " << m_rel_tolerance;
            return ss.str();
        }
    };

    /// @brief Factory function to create a `tensor_close_matcher` instance. Simplifies test syntax.
    /// @tparam T The element type of the view (can be const or non-const complex).
    /// @tparam IndexType The index type used by the view.
    /// @param expected The expected tensor view (can be const or non-const).
    /// @param tolerance The relative tolerance.
    /// @return A `tensor_close_matcher` object configured for the non-const version of T.
    template <typename T, typename IndexType = kmx::tensor::index>
    auto is_tensor_close(const view<T, IndexType>& expected, typename view<T, IndexType>::non_const_value_type::value_type tolerance)
    {
        // Deduce the non-const complex type for the matcher template argument
        using NonConstComplexType = typename view<T, IndexType>::non_const_value_type;
        // Create a const view to pass to the matcher constructor, ensuring consistency
        view<const NonConstComplexType, IndexType> const_expected_view(expected);
        return tensor_close_matcher<NonConstComplexType, IndexType>(const_expected_view, tolerance);
    }

    /// @brief Fills a vector with random std::complex numbers using a deterministic seed.
    /// @tparam complex_type The std::complex number type.
    /// @param vec The vector to fill (will be modified).
    /// @param seed_offset An offset added to a base seed for variation across tests.
    /// @param real_only If true, sets imaginary parts to zero.
    template <typename complex_type>
    void fill_random(std::vector<complex_type>& vec, unsigned int seed_offset = 0, bool real_only = false) noexcept
    {
        if (vec.empty())
            return; // Nothing to fill

        using float_type = typename complex_type::value_type;
        // Use a fixed base seed plus offset for reproducible test runs
        static constexpr unsigned int base_seed = 12345u;
        std::mt19937 gen(base_seed + seed_offset); // Mersenne Twister engine
        // Distribution between -1.0 and 1.0 for real and imaginary parts
        std::uniform_real_distribution<float_type> dist(float_type {-1.0}, float_type {1.0});

        // Fill the vector efficiently
        for (auto& val: vec)
        {
            val = complex_type {dist(gen), (real_only ? float_type {0.0} : dist(gen))};
        }
    }

} // namespace kmx::fft::test_helpers

// --- Test Cases ---

/// @brief Tests the forward and inverse 1D FFT for identity property (FFT(IFFT(x)) == x) using engine.
TEST_CASE("fft_1d_forward_inverse_identity", "[fft][d1][engine]")
{
    using complex_d = std::complex<double>; // Use double for higher precision test
    using namespace kmx::tensor;
    using namespace kmx::fft::test_helpers;
    const double tolerance = std::numeric_limits<double>::epsilon() * 10000.0; // Relative tolerance based on epsilon

    // Create FFT engine instance (caches live for the duration of this test case)
    kmx::fft::engine<complex_d> engine;

    // Test various sizes, including powers of 2, primes, and composites
    const std::vector<std::size_t> test_sizes = {1u,  2u,  3u,  4u,  5u,  6u,   7u,   8u,   13u,  15u,  16u,   17u,  19u,
                                                 24u, 31u, 32u, 60u, 64u, 127u, 128u, 255u, 256u, 512u, 1000u, 1024u};

    for (const std::size_t n: test_sizes)
    {
        // Use DYNAMIC_SECTION for better reporting in Catch2 output
        DYNAMIC_SECTION("size_n_" << n)
        {
            // Allocate data buffers
            std::vector<complex_d> signal_data(n);
            std::vector<complex_d> spectrum_data(n);
            std::vector<complex_d> reconstructed_data(n);

            // Fill with reproducible random data
            fill_random(signal_data, static_cast<unsigned int>(n));
            // Keep a copy of the original signal for comparison
            const std::vector<complex_d> original_data_copy = signal_data;

            // Create views (use explicit 1D constructor where appropriate)
            // Const view for original data and signal input
            view<const complex_d> original_view(original_data_copy);
            view<const complex_d> signal_view(signal_data);
            // Non-const views for outputs
            view<complex_d> spectrum_view(spectrum_data);
            view<complex_d> reconstructed_view(reconstructed_data);

            // --- Out-of-place Test: FFT -> IFFT ---
            REQUIRE_NOTHROW(engine.transform_1d(signal_view, spectrum_view, false));       // Forward FFT
            REQUIRE_NOTHROW(engine.transform_1d(spectrum_view, reconstructed_view, true)); // Inverse FFT
            // Verify that the reconstructed signal matches the original within tolerance
            REQUIRE_THAT(reconstructed_view, is_tensor_close(original_view, tolerance));

            // --- In-place Test: FFT -> IFFT ---
            // Create a non-const view of the original signal data buffer for in-place modification
            view<complex_d> signal_inplace_view(signal_data);
            REQUIRE_NOTHROW(engine.transform_1d(signal_inplace_view, false)); // Forward FFT (in-place)
            REQUIRE_NOTHROW(
                engine.transform_1d(signal_inplace_view, true)); // Inverse FFT (in-place)
                                                                 // Verify that the modified buffer now matches the original data
            REQUIRE_THAT(signal_inplace_view, is_tensor_close(original_view, tolerance));
        }
    }
}

namespace kmx::fft
{

    /// @brief Tests the forward and inverse 2D FFT for identity property using engine.
    TEST_CASE("fft_2d_forward_inverse_identity", "[fft][d2][engine]")
    {
        using complex_f = std::complex<float>; // Use float for variety
        using namespace kmx::tensor;
        using namespace kmx::fft::test_helpers;
        const float tolerance = std::numeric_limits<float>::epsilon() * 5000.0f; // Tolerance for float, potentially higher due to 2D errors

        // Create FFT engine instance
        kmx::fft::engine<complex_f> engine;

        // Test various dimensions (Rows x Cols)
        std::vector<std::pair<std::size_t, std::size_t>> test_dims = {
            {1, 1},   {2, 2},  {4, 4}, {8, 8}, {16, 16}, // Powers of 2 squares
            {3, 5},   {7, 4},  {4, 7},                   // Mixed prime/power-of-2 sizes
            {5, 1},   {1, 5},                            // Effectively 1D cases handled by 2D transform
            {17, 8},  {8, 17},                           // Prime x Power of 2
            {6, 10},  {15, 9},                           // Composite x Composite
            {32, 32}, {64, 63}                           // Larger dimensions
        };

        for (const auto& dims: test_dims)
        {
            const std::size_t rows = dims.first;
            const std::size_t cols = dims.second;
            DYNAMIC_SECTION("size_" << rows << "x" << cols)
            {
                // Skip degenerate cases (already tested in edge cases)
                if (rows == 0u || cols == 0u)
                    continue;

                const std::size_t total_size = rows * cols;
                // Allocate data buffers
                std::vector<complex_f> signal_data(total_size);
                std::vector<complex_f> spectrum_data(total_size);
                std::vector<complex_f> reconstructed_data(total_size);

                // Fill with reproducible random data (use dimensions in seed offset)
                fill_random(signal_data, static_cast<unsigned int>(rows * 1000 + cols));
                const std::vector<complex_f> original_data_copy = signal_data;

                // Create shape and views
                // Use std::array for shape storage - ensures lifetime
                const std::array<std::size_t, 2> shape_arr = {rows, cols};
                const dimension_shape shape_view(shape_arr); // Create shape span from array

                // Create const views for inputs/original data
                view<const complex_f> original_view(original_data_copy, shape_view);
                view<const complex_f> signal_view(signal_data, shape_view);
                // Create non-const views for outputs
                view<complex_f> spectrum_view(spectrum_data, shape_view);
                view<complex_f> reconstructed_view(reconstructed_data, shape_view);

                // --- Out-of-place Test: FFT -> IFFT ---
                REQUIRE_NOTHROW(engine.transform_2d(signal_view, spectrum_view, false));       // Forward 2D FFT
                REQUIRE_NOTHROW(engine.transform_2d(spectrum_view, reconstructed_view, true)); // Inverse 2D FFT
                // Verify identity
                REQUIRE_THAT(reconstructed_view, is_tensor_close(original_view, tolerance));

                // --- In-place Test: FFT -> IFFT ---
                // Create non-const view for in-place operation
                view<complex_f> signal_inplace_view(signal_data, shape_view);
                REQUIRE_NOTHROW(engine.transform_2d(signal_inplace_view, false)); // Forward 2D FFT (in-place)
                REQUIRE_NOTHROW(engine.transform_2d(signal_inplace_view, true));  // Inverse 2D FFT (in-place)
                // Verify identity
                REQUIRE_THAT(signal_inplace_view, is_tensor_close(original_view, tolerance));
            }
        }
    }

    /// @brief Tests edge cases for 1D and 2D FFT using engine.
    TEST_CASE("fft_edge_cases", "[fft][edge][engine]")
    {
        using complex_d = std::complex<double>;
        using namespace kmx::tensor;
        using namespace kmx::fft::test_helpers;
        // Use a tight tolerance for exact results where expected (like size 1 or empty)
        const double tolerance = std::numeric_limits<double>::epsilon() * 10.0;

        // Create engine instance
        kmx::fft::engine<complex_d> engine;

        SECTION("empty_1d")
        {
            std::vector<complex_d> data;     // Empty vector
            std::vector<complex_d> out_data; // Empty vector for output

            // Create an explicitly rank 1, size 0 view using explicit shape
            const std::array<std::size_t, 1> shape_1d_empty_arr = {0};
            const dimension_shape shape_1d_empty(shape_1d_empty_arr);
            view<complex_d> view_mut_1d(data, shape_1d_empty);
            view<const complex_d> view_const_1d(data, shape_1d_empty);
            view<complex_d> out_view_1d(out_data, shape_1d_empty);

            // Create rank 1, size 0 view using 1D constructor with empty span
            view<complex_d> view_mut_1d_deduced(data);
            view<const complex_d> view_const_1d_deduced(data);
            view<complex_d> out_view_1d_deduced(out_data);

            // Verify properties of explicitly shaped view
            REQUIRE(view_mut_1d.empty());
            REQUIRE(view_mut_1d.rank() == 1);
            REQUIRE(view_mut_1d.size() == 0);
            REQUIRE(view_mut_1d.shape().size() == 1);
            REQUIRE(!view_mut_1d.shape().empty()); // Shape should exist even if size is 0
            REQUIRE(view_mut_1d.shape()[0] == 0);

            // Verify properties of deduced view
            REQUIRE(view_mut_1d_deduced.empty());
            REQUIRE(view_mut_1d_deduced.rank() == 1); // Check constructor logic
            REQUIRE(view_mut_1d_deduced.size() == 0);
            REQUIRE(view_mut_1d_deduced.shape().size() == 1);
            REQUIRE(!view_mut_1d_deduced.shape().empty());
            REQUIRE(view_mut_1d_deduced.shape()[0] == 0);

            // Test transforms on empty views (should be no-ops)
            REQUIRE_NOTHROW(engine.transform_1d(view_mut_1d, false));
            REQUIRE(view_mut_1d.empty()); // Should remain empty and unchanged
            REQUIRE_NOTHROW(engine.transform_1d(view_const_1d, out_view_1d, false));
            REQUIRE(out_view_1d.empty()); // Output should be empty

            // Test deduced views too
            REQUIRE_NOTHROW(engine.transform_1d(view_mut_1d_deduced, true)); // Inverse
            REQUIRE(view_mut_1d_deduced.empty());
            REQUIRE_NOTHROW(engine.transform_1d(view_const_1d_deduced, out_view_1d_deduced, true)); // Inverse
            REQUIRE(out_view_1d_deduced.empty());
        }

        SECTION("empty_2d_shape_0xN")
        {
            std::vector<complex_d> data;                         // Empty
            std::vector<complex_d> out_data;                     // Empty
            const std::array<std::size_t, 2> shape_arr = {0, 5}; // 0 rows, 5 columns
            const dimension_shape shape_0xN(shape_arr);

            view<complex_d> view_mut(data, shape_0xN);
            view<const complex_d> view_const(data, shape_0xN);
            view<complex_d> out_view(out_data, shape_0xN);

            REQUIRE(view_mut.empty());
            REQUIRE(view_mut.rank() == 2);
            REQUIRE(view_mut.size() == 0);

            // Test 2D transforms (should be no-ops)
            REQUIRE_NOTHROW(engine.transform_2d(view_mut, false));
            REQUIRE(view_mut.empty());
            REQUIRE_NOTHROW(engine.transform_2d(view_const, out_view, true)); // Inverse
            REQUIRE(out_view.empty());
        }

        SECTION("empty_2d_shape_Nx0")
        {
            std::vector<complex_d> data;                         // Empty
            std::vector<complex_d> out_data;                     // Empty
            const std::array<std::size_t, 2> shape_arr = {5, 0}; // 5 rows, 0 columns
            const dimension_shape shape_Nx0(shape_arr);

            view<complex_d> view_mut(data, shape_Nx0);
            view<const complex_d> view_const(data, shape_Nx0);
            view<complex_d> out_view(out_data, shape_Nx0);

            REQUIRE(view_mut.empty());
            REQUIRE(view_mut.rank() == 2);
            REQUIRE(view_mut.size() == 0);

            // Test 2D transforms (should be no-ops)
            REQUIRE_NOTHROW(engine.transform_2d(view_mut, true)); // Inverse
            REQUIRE(view_mut.empty());
            REQUIRE_NOTHROW(engine.transform_2d(view_const, out_view, false));
            REQUIRE(out_view.empty());
        }

        SECTION("size_1_1d")
        {
            // FFT/IFFT of size 1 is the identity operation (up to scaling for IFFT)
            std::vector<complex_d> data = {{1.2, -3.4}};
            const std::vector<complex_d> data_orig = data; // Store original value
            std::vector<complex_d> out_data(1);

            // Create views using 1D constructor
            view<complex_d> view_mut(data);
            view<const complex_d> view_const(data);
            view<const complex_d> orig_view(data_orig);
            view<complex_d> out_view(out_data);

            REQUIRE(view_mut.size() == 1);
            REQUIRE(view_mut.rank() == 1);

            // In-place forward FFT (should be identity)
            REQUIRE_NOTHROW(engine.transform_1d(view_mut, false));
            REQUIRE_THAT(view_mut, is_tensor_close(orig_view, tolerance));

            // Restore data for inverse test
            view_mut[0] = data_orig[0];
            // In-place inverse FFT (should be identity * (1/1) = identity)
            REQUIRE_NOTHROW(engine.transform_1d(view_mut, true));
            REQUIRE_THAT(view_mut, is_tensor_close(orig_view, tolerance));

            // Out-of-place forward FFT
            REQUIRE_NOTHROW(engine.transform_1d(view_const, out_view, false));
            REQUIRE_THAT(out_view, is_tensor_close(orig_view, tolerance));

            // Out-of-place inverse FFT
            REQUIRE_NOTHROW(engine.transform_1d(view_const, out_view, true));
            REQUIRE_THAT(out_view, is_tensor_close(orig_view, tolerance));
        }

        SECTION("size_1x1_2d")
        {
            // FFT/IFFT of size 1x1 is identity (up to scaling)
            std::vector<complex_d> data = {{5.6, 7.8}};
            const std::vector<complex_d> data_orig = data;
            std::vector<complex_d> out_data(1);

            const std::array<std::size_t, 2> shape_arr = {1u, 1u};
            const dimension_shape shape_1x1(shape_arr);

            view<complex_d> view_mut(data, shape_1x1);
            view<const complex_d> view_const(data, shape_1x1);
            view<const complex_d> orig_view(data_orig, shape_1x1);
            view<complex_d> out_view(out_data, shape_1x1);

            REQUIRE(view_mut.size() == 1);
            REQUIRE(view_mut.rank() == 2);

            // In-place forward FFT (should be identity)
            REQUIRE_NOTHROW(engine.transform_2d(view_mut, false));
            REQUIRE_THAT(view_mut, is_tensor_close(orig_view, tolerance));

            // Restore data for inverse test
            view_mut[0] = data_orig[0];
            // In-place inverse FFT (should be identity * (1/(1*1)) = identity)
            REQUIRE_NOTHROW(engine.transform_2d(view_mut, true));
            REQUIRE_THAT(view_mut, is_tensor_close(orig_view, tolerance));

            // Out-of-place forward FFT
            REQUIRE_NOTHROW(engine.transform_2d(view_const, out_view, false));
            REQUIRE_THAT(out_view, is_tensor_close(orig_view, tolerance));

            // Out-of-place inverse FFT
            REQUIRE_NOTHROW(engine.transform_2d(view_const, out_view, true));
            REQUIRE_THAT(out_view, is_tensor_close(orig_view, tolerance));
        }
    }

    /// @brief Tests error handling for invalid inputs to FFT engine methods.
    TEST_CASE("fft_error_handling", "[fft][error][engine]")
    {
        using complex_d = std::complex<double>;
        using namespace kmx::tensor;
        using Catch::Matchers::ContainsSubstring;

        // Create engine instance
        kmx::fft::engine<complex_d> engine;

        // --- Setup Data and Shapes for Invalid Cases ---
        std::vector<complex_d> data_1d = {{1.0}, {2.0}, {3.0}, {4.0}};
        std::vector<complex_d> data_2d = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}};
        std::vector<complex_d> data_3d(8); // 2x2x2
        std::vector<complex_d> out_data_1d(4);
        std::vector<complex_d> out_data_2d(6);
        std::vector<complex_d> out_data_mismatch(5); // For shape mismatch tests

        // Valid shapes
        const std::array<std::size_t, 1> shape_1d_arr = {4};
        const dimension_shape shape_1d(shape_1d_arr);
        const std::array<std::size_t, 2> shape_2d_arr = {2, 3};
        const dimension_shape shape_2d(shape_2d_arr);

        // Invalid shapes/ranks for context
        const std::array<std::size_t, 3> shape_3d_arr = {2, 2, 2};
        const dimension_shape shape_3d(shape_3d_arr);
        const std::array<std::size_t, 1> shape_1d_mismatch_arr = {5}; // Different size for 1D
        const dimension_shape shape_1d_mismatch(shape_1d_mismatch_arr);
        const std::array<std::size_t, 2> shape_2d_mismatch_arr = {3, 2}; // Different shape for 2D
        const dimension_shape shape_2d_mismatch(shape_2d_mismatch_arr);

        // Create views
        // Valid views for reference
        view<complex_d> valid_1d_view(data_1d, shape_1d);
        view<const complex_d> valid_1d_const_view(data_1d, shape_1d);
        view<complex_d> valid_2d_view(data_2d, shape_2d);
        view<const complex_d> valid_2d_const_view(data_2d, shape_2d);

        // Invalid views for testing
        // Note: Creating a view with a shape whose size doesn't match the data vector size
        // might be disallowed by the view constructor itself. Assuming it's allowed here.
        view<complex_d> invalid_rank_3d_for_2d(data_3d, shape_3d); // Trying to use a 3D shape for 2D op

        // Output views with mismatched shapes
        view<complex_d> out_1d_mismatch_view(out_data_mismatch, shape_1d_mismatch);
        // Note: data size might not match shape here, but shape comparison is key
        // Need data buffer large enough for shape_2d_mismatch size (3*2=6)
        std::vector<complex_d> out_data_2d_mismatch_buffer(6);
        view<complex_d> out_2d_mismatch_view(out_data_2d_mismatch_buffer, shape_2d_mismatch);

        SECTION("transform_1d invalid inputs")
        {
            // In-place: Requires rank 1
            const std::array<std::size_t, 2> shape_rank2_arr = {2, 2};
            const dimension_shape shape_rank2(shape_rank2_arr);
            view<complex_d> view_rank2(data_1d, shape_rank2); // Create rank 2 view
            REQUIRE_THROWS_AS(engine.transform_1d(view_rank2, false), std::invalid_argument);
            REQUIRE_THROWS_WITH(engine.transform_1d(view_rank2, false), ContainsSubstring("requires rank 1"));

            // Out-of-place: Input rank check vs Shape check
            // Scenario 1: Input has wrong rank, shapes also mismatch (Shape check happens first)
            REQUIRE_THROWS_AS(engine.transform_1d(view_rank2, valid_1d_view, false), std::invalid_argument);
            REQUIRE_THROWS_WITH(engine.transform_1d(view_rank2, valid_1d_view, false),
                                ContainsSubstring("Input and output shapes must match")); // <<< CORRECTED EXPECTED MESSAGE

            // Scenario 2: Input has wrong rank, but shapes *happen* to match (Rank check should happen)
            // Create a rank 2 input view and a rank 1 output view that have the same flat size
            const std::array<std::size_t, 2> shape_rank2_size4_arr = {2, 2};
            const dimension_shape shape_rank2_size4(shape_rank2_size4_arr);
            view<complex_d> input_rank2_size4(data_1d, shape_rank2_size4); // Input Rank 2, Shape [2,2]
            view<complex_d> output_rank1_size4(data_1d, shape_1d);         // Output Rank 1, Shape [4]
            // Note: The shape objects are different, even if size is same. Our shape() != shape() check compares shapes directly.
            // If the check was based on size(), this test might need adjustment. Assuming shape comparison.
            // Since shapes [2,2] and [4] are different, the shape check still wins.
            REQUIRE_THROWS_WITH(engine.transform_1d(input_rank2_size4, output_rank1_size4, false),
                                ContainsSubstring("Input and output shapes must match"));

            // Scenario 3: Shapes mismatch, input rank is correct (Shape check should happen)
            REQUIRE_THROWS_AS(engine.transform_1d(valid_1d_const_view, out_1d_mismatch_view, false), std::invalid_argument);
            REQUIRE_THROWS_WITH(engine.transform_1d(valid_1d_const_view, out_1d_mismatch_view, false),
                                ContainsSubstring("Input and output shapes must match"));
        }

        SECTION("transform_2d invalid inputs")
        {
            // In-place: Requires rank 1 or 2
            REQUIRE_THROWS_AS(engine.transform_2d(invalid_rank_3d_for_2d, false), std::invalid_argument);
            REQUIRE_THROWS_WITH(engine.transform_2d(invalid_rank_3d_for_2d, false), ContainsSubstring("must have rank 1 or 2"));

            // Out-of-place: Input rank check vs Shape check
            // Scenario 1: Input has wrong rank (3), shapes also mismatch (Shape check happens first)
            REQUIRE_THROWS_AS(engine.transform_2d(invalid_rank_3d_for_2d, valid_2d_view, false), std::invalid_argument);
            REQUIRE_THROWS_WITH(engine.transform_2d(invalid_rank_3d_for_2d, valid_2d_view, false),
                                ContainsSubstring("Input and output shapes must match")); // <<< CORRECTED EXPECTED MESSAGE

            // Scenario 2: Input has wrong rank (3), but shapes match (Rank check should happen)
            const std::array<std::size_t, 3> shape_rank3_size6_arr = {2, 3, 1}; // Shape [2,3,1] size 6
            const dimension_shape shape_rank3_size6(shape_rank3_size6_arr);
            std::vector<complex_d> data_rank3_size6(6);
            view<complex_d> input_rank3_size6(data_rank3_size6, shape_rank3_size6); // Input Rank 3, Shape [2,3,1]
            view<complex_d> output_rank2_size6(data_2d, shape_2d);                  // Output Rank 2, Shape [2,3]
            // Shapes [2,3,1] and [2,3] are different. Shape check still wins.
            REQUIRE_THROWS_WITH(engine.transform_2d(input_rank3_size6, output_rank2_size6, false),
                                ContainsSubstring("Input and output shapes must match"));

            // Scenario 3: Shapes mismatch, input rank is correct (2) (Shape check should happen)
            REQUIRE_THROWS_AS(engine.transform_2d(valid_2d_const_view, out_2d_mismatch_view, false), std::invalid_argument);
            REQUIRE_THROWS_WITH(engine.transform_2d(valid_2d_const_view, out_2d_mismatch_view, false),
                                ContainsSubstring("Input and output shapes must match"));
        }
    }

    /// @brief Tests FFT properties with known signals (Delta, Constant, Real Input Symmetry).
    TEST_CASE("fft_known_signals", "[fft][known_signal][engine]")
    {
        using complex_d = std::complex<double>;
        using namespace kmx::fft::test_helpers;
        // Tolerance for theoretical results (can be tight)
        const double tolerance = std::numeric_limits<double>::epsilon() * 100.0;
        const double abs_tolerance = tolerance; // Use same for absolute near zero

        // Create engine instance
        kmx::fft::engine<complex_d> engine;

        // Choose a size (e.g., power of 2 for simplicity, but non-power-of-2 also works)
        const std::size_t N = 8u;

        SECTION("delta_signal")
        {
            std::vector<complex_d> signal_data(N, {0.0, 0.0});
            signal_data[0] = {1.0, 0.0};                        // Delta function at index 0
            std::vector<complex_d> expected_fft(N, {1.0, 0.0}); // Expected FFT is all ones
            std::vector<complex_d> actual_fft_data(N);

            view<const complex_d> signal_view(signal_data);
            view<const complex_d> expected_fft_view(expected_fft);
            view<complex_d> actual_fft_view(actual_fft_data);

            // --- Test Forward FFT ---
            REQUIRE_NOTHROW(engine.transform_1d(signal_view, actual_fft_view, false));
            REQUIRE_THAT(actual_fft_view, is_tensor_close(expected_fft_view, tolerance));

            // --- Test Inverse FFT ---
            // IFFT of the expected spectrum should yield the original delta signal
            std::vector<complex_d> reconstructed_data(N);
            view<complex_d> reconstructed_view(reconstructed_data);
            view<const complex_d> original_signal_view(signal_data); // Recreate const view of original

            REQUIRE_NOTHROW(engine.transform_1d(expected_fft_view, reconstructed_view, true)); // IFFT
            REQUIRE_THAT(reconstructed_view, is_tensor_close(original_signal_view, tolerance));
        }

        SECTION("constant_signal")
        {
            const complex_d constant_value = {2.5, -1.0}; // Arbitrary constant
            std::vector<complex_d> signal_data(N, constant_value);
            std::vector<complex_d> expected_fft(N, {0.0, 0.0});
            expected_fft[0] = static_cast<double>(N) * constant_value; // DC component = N * constant
            std::vector<complex_d> actual_fft_data(N);

            view<const complex_d> signal_view(signal_data);
            view<const complex_d> expected_fft_view(expected_fft);
            view<complex_d> actual_fft_view(actual_fft_data);

            // --- Test Forward FFT ---
            REQUIRE_NOTHROW(engine.transform_1d(signal_view, actual_fft_view, false));
            REQUIRE_THAT(actual_fft_view, is_tensor_close(expected_fft_view, tolerance));

            // --- Test Inverse FFT ---
            // IFFT of the expected spectrum should yield the original constant signal
            std::vector<complex_d> reconstructed_data(N);
            view<complex_d> reconstructed_view(reconstructed_data);
            view<const complex_d> original_signal_view(signal_data);

            REQUIRE_NOTHROW(engine.transform_1d(expected_fft_view, reconstructed_view, true)); // IFFT
            REQUIRE_THAT(reconstructed_view, is_tensor_close(original_signal_view, tolerance));
        }

        SECTION("real_input_symmetry")
        {
            // Test sizes covering even and odd lengths, power-of-2 and non-power-of-2
            const std::vector<std::size_t> test_sizes = {7u, 8u, 15u, 16u};

            for (const std::size_t current_n: test_sizes)
            {
                DYNAMIC_SECTION("size_n_" << current_n)
                {
                    std::vector<complex_d> signal_data(current_n);
                    // Fill with random REAL data (imaginary parts = 0)
                    fill_random(signal_data, static_cast<unsigned int>(current_n), true); // true -> real_only

                    std::vector<complex_d> actual_fft_data(current_n);
                    view<const complex_d> signal_view(signal_data);
                    view<complex_d> actual_fft_view(actual_fft_data);

                    // --- Compute Forward FFT ---
                    REQUIRE_NOTHROW(engine.transform_1d(signal_view, actual_fft_view, false));

                    // --- Verify Conjugate Symmetry Property: X[k] = conj(X[N-k]) ---
                    // Check DC component (k=0) is real
                    REQUIRE_THAT(actual_fft_view[0].imag(), Catch::Matchers::WithinAbs(0.0, abs_tolerance));

                    // Check Nyquist component (k=N/2) is real if N is even
                    if (current_n > 0 && current_n % 2 == 0) // Avoid index N/2 if N=0
                    {
                        REQUIRE_THAT(actual_fft_view[current_n / 2].imag(), Catch::Matchers::WithinAbs(0.0, abs_tolerance));
                    }

                    // Check remaining components k = 1 to floor((N-1)/2)
                    // Loop up to N/2, because element N/2 is handled above for even N, and doesn't have a N-k counterpart within 1..N-1 for
                    // odd N.
                    for (std::size_t k = 1u; k < (current_n + 1u) / 2; ++k) // k goes from 1 up to floor(N/2)
                    {
                        // Need N > 1 for k=1 and N-k=N-1 to be valid distinct indices
                        if (current_n > 1)
                        {
                            const complex_d& xk = actual_fft_view[k];
                            const complex_d& xnk = actual_fft_view[current_n - k]; // N-k index

                            // Check if xk is close to the conjugate of xnk
                            // Provide both relative and absolute tolerances to complex_close
                            REQUIRE(complex_close(xk, std::conj(xnk), tolerance, abs_tolerance));
                        }
                    }

                    // --- Optional: Verify IFFT reconstructs the real signal ---
                    std::vector<complex_d> reconstructed_data(current_n);
                    view<complex_d> reconstructed_view(reconstructed_data);
                    view<const complex_d> original_signal_view(signal_data);
                    REQUIRE_NOTHROW(engine.transform_1d(actual_fft_view, reconstructed_view, true));
                    REQUIRE_THAT(reconstructed_view, is_tensor_close(original_signal_view, tolerance));
                    // Additionally check that reconstructed imaginary parts are near zero
                    for (const auto& val: reconstructed_data)
                    {
                        REQUIRE_THAT(val.imag(), Catch::Matchers::WithinAbs(0.0, abs_tolerance));
                    }
                }
            }
        }
    }
}

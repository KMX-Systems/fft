// Copyright (c) 2025 - present KMX Systems. All rights reserved.
#define CATCH_CONFIG_MAIN
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <kmx/tensor.hpp>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace kmx::tensor
{
    using Catch::Matchers::RangeEquals; // For comparing index::values() output span
    // Helper function to create a contiguously filled vector
    std::vector<float> create_test_data(std::size_t size)
    {
        std::vector<float> data(size);
        // Fill with 0.0, 1.0, 2.0, ... using std::iota
        std::iota(data.begin(), data.end(), 0.0f);
        return data;
    }

    // === Unit Tests for kmx::tensor::index ===

    TEST_CASE("kmx::tensor::index - Construction and Rank Constraints", "[tensor][index]")
    {
        SECTION("Valid Construction from Initializer List")
        {
            index idx1 = {1};
            REQUIRE(idx1.rank() == 1);

            index idx2 = {1, 2};
            REQUIRE(idx2.rank() == 2);

            index idx3 = {1, 2, 3};
            REQUIRE(idx3.rank() == 3);

            index idx4 = {1, 2, 3, 4};
            REQUIRE(idx4.rank() == 4);
        }

        SECTION("Valid Construction from Vector")
        {
            std::vector<std::size_t> v1 = {10};
            index idx1(v1);
            REQUIRE(idx1.rank() == 1);

            std::vector<std::size_t> v4 = {10, 20, 30, 40};
            index idx4(v4);
            REQUIRE(idx4.rank() == 4);
        }

        SECTION("Valid Construction from Span")
        {
            std::vector<std::size_t> v2 = {10, 20};
            std::span<const std::size_t> s2 = v2;
            index idx2(s2);
            REQUIRE(idx2.rank() == 2);
        }

        SECTION("Valid Construction from Array")
        {
            std::array<std::size_t, 3> a3 = {5, 6, 7};
            index idx3(a3); // Construct from std::array
            REQUIRE(idx3.rank() == 3);
        }

        SECTION("Invalid Construction - Rank Too Low")
        {
            // Rank 0 disallowed by constructor check now
            REQUIRE_THROWS_AS(index({}), std::invalid_argument);

            std::vector<std::size_t> v0;
            REQUIRE_THROWS_AS(index(v0), std::invalid_argument);
        }

        SECTION("Invalid Construction - Rank Too High")
        {
            // Rank 5 disallowed by constructor check
            REQUIRE_THROWS_AS(index({1, 2, 3, 4, 5}), std::invalid_argument);

            std::vector<std::size_t> v5 = {1, 2, 3, 4, 5};
            REQUIRE_THROWS_AS(index(v5), std::invalid_argument);
        }

        SECTION("Copy Construction")
        {
            index idx_orig = {10, 20};
            index idx_copy = idx_orig; // Copy constructor

            REQUIRE(idx_copy.rank() == 2);
            REQUIRE_THAT(idx_copy.values(), RangeEquals(std::vector<std::size_t> {10, 20}));
            REQUIRE(&idx_copy.array() != &idx_orig.array()); // Ensure distinct objects
        }

        SECTION("Copy Assignment")
        {
            index idx_orig = {10, 20, 30};
            index idx_other = {1};
            idx_other = idx_orig; // Copy assignment

            REQUIRE(idx_other.rank() == 3);
            REQUIRE_THAT(idx_other.values(), RangeEquals(std::vector<std::size_t> {10, 20, 30}));
            REQUIRE(&idx_other.array() != &idx_orig.array()); // Ensure distinct objects
        }

        SECTION("Move Construction")
        {
            index idx_orig = {10, 20};
            // Capture state before move for comparison
            auto expected_values = std::vector<std::size_t> {10, 20};
            std::size_t expected_rank = idx_orig.rank();

            index idx_moved = std::move(idx_orig); // Move constructor

            // Verify the moved-to object
            REQUIRE(idx_moved.rank() == expected_rank);
            REQUIRE_THAT(idx_moved.values(), RangeEquals(expected_values));
            // Note: State of idx_orig is valid but unspecified after move.
        }

        SECTION("Move Assignment")
        {
            index idx_orig = {10, 20, 30};
            index idx_other = {1};
            // Capture state before move for comparison
            auto expected_values = std::vector<std::size_t> {10, 20, 30};
            std::size_t expected_rank = idx_orig.rank();

            idx_other = std::move(idx_orig); // Move assignment

            // Verify the assigned-to object
            REQUIRE(idx_other.rank() == expected_rank);
            REQUIRE_THAT(idx_other.values(), RangeEquals(expected_values));
            // Note: State of idx_orig is valid but unspecified after move.
        }
    }

    TEST_CASE("kmx::tensor::index - Accessors", "[tensor][index]")
    {
        SECTION("rank() method")
        {
            index idx1 = {1};
            REQUIRE(idx1.rank() == 1);
            index idx4 = {1, 2, 3, 4};
            REQUIRE(idx4.rank() == 4);
        }

        SECTION("values() method")
        {
            index idx3 = {10, 20, 30};
            std::span<const std::size_t> values_span = idx3.values();
            REQUIRE(values_span.size() == 3);
            // Use RangeEquals for span comparison
            REQUIRE_THAT(values_span, RangeEquals(std::vector<std::size_t> {10, 20, 30}));

            index idx1 = {99};
            values_span = idx1.values();
            REQUIRE(values_span.size() == 1);
            REQUIRE_THAT(values_span, RangeEquals(std::vector<std::size_t> {99}));
        }

        SECTION("array() method")
        {
            index idx2 = {5, 6};
            const auto& arr = idx2.array();
            // Check type is correct std::array specialization
            STATIC_REQUIRE(std::is_same_v<decltype(arr), const std::array<std::size_t, 4>&>);
            REQUIRE(arr[0] == 5);
            REQUIRE(arr[1] == 6);
            // Values at index >= rank() are unspecified but often 0 from default init
        }
    }

    TEST_CASE("kmx::tensor::index - operator() Flat Index Calculation", "[tensor][index]")
    {
        // Use dimension_shape for the shape parameter type
        using shape_span_type = kmx::tensor::dimension_shape;

        SECTION("Valid Calculations")
        {
            // Rank 1
            std::vector<std::size_t> shape1_vec = {5};
            shape_span_type shape1(shape1_vec);
            index idx1_a = {0};
            index idx1_b = {2};
            index idx1_c = {4};
            REQUIRE(idx1_a(shape1) == 0);
            REQUIRE(idx1_b(shape1) == 2);
            REQUIRE(idx1_c(shape1) == 4);

            // Rank 2
            std::vector<std::size_t> shape2_vec = {3, 4}; // Size 12
            shape_span_type shape2(shape2_vec);
            index idx2_a = {0, 0}; // 0*4 + 0 = 0
            index idx2_b = {1, 2}; // 1*4 + 2 = 6
            index idx2_c = {2, 3}; // 2*4 + 3 = 11
            REQUIRE(idx2_a(shape2) == 0);
            REQUIRE(idx2_b(shape2) == 6);
            REQUIRE(idx2_c(shape2) == 11);

            // Rank 3
            std::vector<std::size_t> shape3_vec = {2, 3, 4}; // Size 24
            shape_span_type shape3(shape3_vec);
            index idx3_a = {0, 0, 0}; // (0*3 + 0)*4 + 0 = 0
            index idx3_b = {0, 1, 2}; // (0*3 + 1)*4 + 2 = 1*4 + 2 = 6
            index idx3_c = {1, 1, 2}; // (1*3 + 1)*4 + 2 = 4*4 + 2 = 18
            index idx3_d = {1, 2, 3}; // (1*3 + 2)*4 + 3 = 5*4 + 3 = 23
            REQUIRE(idx3_a(shape3) == 0);
            REQUIRE(idx3_b(shape3) == 6);
            REQUIRE(idx3_c(shape3) == 18);
            REQUIRE(idx3_d(shape3) == 23);

            // Rank 4
            std::vector<std::size_t> shape4_vec = {2, 1, 3, 2}; // Size 12
            shape_span_type shape4(shape4_vec);
            index idx4_a = {0, 0, 0, 0}; // ((0*1+0)*3+0)*2 + 0 = 0
            index idx4_b = {1, 0, 2, 1}; // ((1*1+0)*3+2)*2 + 1 = (1*3+2)*2+1 = 5*2+1 = 11
            REQUIRE(idx4_a(shape4) == 0);
            REQUIRE(idx4_b(shape4) == 11);

            // Zero dimension in shape - calculation should throw if index is accessed
            std::vector<std::size_t> shape_zero_vec = {2, 0, 3}; // Size 0
            shape_span_type shape_zero(shape_zero_vec);
            index idx_zero_dim_access = {1, 0, 1}; // Tries to access index 0 of dimension 1 (size 0)
            REQUIRE_THROWS_AS(idx_zero_dim_access(shape_zero), std::out_of_range);
        }

        SECTION("Error Conditions")
        {
            // Use dimension_shape for shape parameters
            std::vector<std::size_t> shape3_vec = {2, 3, 4};
            kmx::tensor::dimension_shape shape3(shape3_vec);

            // Rank Mismatch
            index idx2 = {1, 1};
            REQUIRE_THROWS_AS(idx2(shape3), std::invalid_argument); // Index rank 2 vs shape rank 3

            std::vector<std::size_t> shape2_vec = {5, 5};
            kmx::tensor::dimension_shape shape2(shape2_vec);
            index idx3 = {1, 1, 1};
            REQUIRE_THROWS_AS(idx3(shape2), std::invalid_argument); // Index rank 3 vs shape rank 2

            // Index Component Out of Bounds
            index idx_oob1 = {1, 3, 0}; // Dim 1 index 3 >= shape[1] (3)
            REQUIRE_THROWS_AS(idx_oob1(shape3), std::out_of_range);

            index idx_oob2 = {2, 0, 0}; // Dim 0 index 2 >= shape[0] (2)
            REQUIRE_THROWS_AS(idx_oob2(shape3), std::out_of_range);

            index idx_oob3 = {1, 1, 4}; // Dim 2 index 4 >= shape[2] (4)
            REQUIRE_THROWS_AS(idx_oob3(shape3), std::out_of_range);
        }
    }

    // === Unit Tests for kmx::tensor::view ===

    TEST_CASE("kmx::tensor::view - Construction and Validation", "[tensor][view]")
    {
        SECTION("Valid Construction")
        {
            std::vector<float> data1d = create_test_data(5);
            std::vector<std::size_t> shape1d_vec = {5};
            REQUIRE_NOTHROW(view<float>(data1d, shape1d_vec));

            std::vector<float> data2d = create_test_data(12); // 3x4
            std::vector<std::size_t> shape2d_vec = {3, 4};
            REQUIRE_NOTHROW(view<float>(data2d, shape2d_vec));

            std::vector<float> data4d = create_test_data(24); // 2x1x4x3
            std::vector<std::size_t> shape4d_vec = {2, 1, 4, 3};
            REQUIRE_NOTHROW(view<float>(data4d, shape4d_vec));

            // Const data view
            const std::vector<float> cdata2d = data2d;
            REQUIRE_NOTHROW(view<const float>(cdata2d, shape2d_vec));

            // Scalar view (Rank 0)
            std::vector<float> data0d = {99.0f};
            std::vector<std::size_t> shape0d_vec = {};
            REQUIRE_NOTHROW(view<float>(data0d, shape0d_vec));

            // View with zero dimension (empty data)
            std::vector<float> data_zero;                        // Empty data
            std::vector<std::size_t> shape_zero_vec = {2, 0, 3}; // Size 0
            REQUIRE_NOTHROW(view<float>(data_zero, shape_zero_vec));
        }

        SECTION("Invalid Construction - Data Size Mismatch")
        {
            std::vector<float> data_wrong_size = create_test_data(10); // Need 12 for 3x4
            std::vector<std::size_t> shape2d_vec = {3, 4};
            REQUIRE_THROWS_AS(view<float>(data_wrong_size, shape2d_vec), std::invalid_argument);

            std::vector<float> data_too_big = create_test_data(15); // Need 12 for 3x4
            REQUIRE_THROWS_AS(view<float>(data_too_big, shape2d_vec), std::invalid_argument);

            // Scalar mismatch (empty data for shape {})
            std::vector<float> data_empty;
            std::vector<std::size_t> shape0d_vec = {};
            REQUIRE_THROWS_AS(view<float>(data_empty, shape0d_vec), std::invalid_argument);

            // Scalar mismatch (non-1 data size for shape {})
            std::vector<float> data_scalar_wrong = {1.0f, 2.0f};
            REQUIRE_THROWS_AS(view<float>(data_scalar_wrong, shape0d_vec), std::invalid_argument);
        }

        SECTION("Copy Construction")
        {
            std::vector<float> data = create_test_data(6); // 2x3
            std::vector<std::size_t> shape_vec = {2, 3};
            view<float> v_orig(data, shape_vec);
            view<float> v_copy = v_orig; // Copy constructor

            REQUIRE(v_copy.rank() == 2);
            REQUIRE(v_copy.size() == 6);
            // Use direct comparison for dimension_shape
            REQUIRE(v_copy.shape() == v_orig.shape());
            REQUIRE(v_copy.data().data() == v_orig.data().data()); // Should point to same data

            // Modify through copy, check original view reflects it
            index idx = {1, 1};
            v_copy[idx] = 100.0f;
            REQUIRE(v_orig[idx] == 100.0f);
            REQUIRE(data[4] == 100.0f); // Flat index 1*3 + 1 = 4
        }

        SECTION("Copy Assignment")
        {
            std::vector<float> data1 = create_test_data(6); // 2x3
            std::vector<std::size_t> shape1_vec = {2, 3};
            view<float> v1(data1, shape1_vec);

            std::vector<float> data2 = create_test_data(4); // 4
            std::vector<std::size_t> shape2_vec = {4};
            view<float> v2(data2, shape2_vec);

            v2 = v1; // Copy assignment

            REQUIRE(v2.rank() == 2);
            REQUIRE(v2.size() == 6);
            // Use direct comparison for dimension_shape
            REQUIRE(v2.shape() == v1.shape());
            REQUIRE(v2.data().data() == v1.data().data());

            // Modify through assigned-to view, check original view reflects it
            index idx = {0, 2};
            v2[idx] = 200.0f;
            REQUIRE(v1[idx] == 200.0f);
            REQUIRE(data1[2] == 200.0f); // Flat index 0*3 + 2 = 2
        }

        SECTION("Move Construction")
        {                                                  // (Corrected Test)
            std::vector<float> data = create_test_data(6); // 2x3
            std::vector<std::size_t> shape_vec = {2, 3};
            dimension_shape expected_shape(shape_vec); // Create expected shape from vector
            view<float> v_orig(data, shape_vec);
            view<float> v_moved = std::move(v_orig); // Move constructor

            // Verify the moved-to object state
            REQUIRE(v_moved.rank() == 2);
            REQUIRE(v_moved.size() == 6);
            REQUIRE(v_moved.shape() == expected_shape);    // Compare with known shape
            REQUIRE(v_moved.data().data() == data.data()); // Points to original data

            // Do NOT check the state of v_orig beyond it being valid.
        }

        SECTION("Move Assignment")
        {                                                   // (Corrected Test)
            std::vector<float> data1 = create_test_data(6); // 2x3
            std::vector<std::size_t> shape1_vec = {2, 3};
            dimension_shape expected_shape1(shape1_vec); // Create expected shape
            view<float> v1(data1, shape1_vec);

            std::vector<float> data2 = create_test_data(4); // 4
            std::vector<std::size_t> shape2_vec = {4};
            view<float> v2(data2, shape2_vec);

            v2 = std::move(v1); // Move assignment

            // Verify the assigned-to object state
            REQUIRE(v2.rank() == 2);
            REQUIRE(v2.size() == 6);
            REQUIRE(v2.shape() == expected_shape1);    // Compare with known shape
            REQUIRE(v2.data().data() == data1.data()); // Points to v1's original data

            // Do NOT check the state of v1 beyond it being valid.
        }
    }

    TEST_CASE("kmx::tensor::view - Accessors", "[tensor][view]")
    {
        using kmx::tensor::dimension_shape;

        std::vector<float> data = create_test_data(12); // 2x2x3
        std::vector<std::size_t> shape_vec = {2, 2, 3};
        dimension_shape expected_shape(shape_vec); // Create comparable shape object
        view<float> v(data, shape_vec);
        const view<float>& cv = v;

        SECTION("rank()")
        {
            REQUIRE(v.rank() == 3);

            // Scalar view
            std::vector<float> data0d = {99.0f};
            std::vector<std::size_t> shape0d_vec = {};
            view<float> v0d(data0d, shape0d_vec);
            REQUIRE(v0d.rank() == 0);
        }

        SECTION("size()")
        {
            REQUIRE(v.size() == 12);

            // Scalar view
            std::vector<float> data0d = {99.0f};
            std::vector<std::size_t> shape0d_vec = {};
            view<float> v0d(data0d, shape0d_vec);
            REQUIRE(v0d.size() == 1);

            // Zero-dim view
            std::vector<float> data_zero;
            std::vector<std::size_t> shape_zero_vec = {2, 0, 3};
            view<float> v_zero(data_zero, shape_zero_vec);
            REQUIRE(v_zero.size() == 0);
        }

        SECTION("shape()")
        {
            dimension_shape s = v.shape();
            REQUIRE(s.size() == 3);
            REQUIRE(s == expected_shape);                     // Direct comparison works now
            REQUIRE(v.shape() == dimension_shape({2, 2, 3})); // Compare with literal

            // Scalar view
            std::vector<float> data0d = {99.0f};
            std::vector<std::size_t> shape0d_vec = {};
            view<float> v0d(data0d, shape0d_vec);
            REQUIRE(v0d.shape() == dimension_shape({})); // Empty shape compares equal
            REQUIRE(v0d.shape().empty());
        }

        SECTION("data()")
        {
            std::span<float> d = v.data(); // Non-const view gives non-const span
            REQUIRE(d.size() == 12);
            REQUIRE(d.data() == data.data());

            std::span<const float> cd = cv.data(); // Const view gives const span
            REQUIRE(cd.size() == 12);
            REQUIRE(cd.data() == data.data());

            // Scalar view
            std::vector<float> data0d = {99.0f};
            std::vector<std::size_t> shape0d_vec = {};
            view<float> v0d(data0d, shape0d_vec);
            REQUIRE(v0d.data().size() == 1);
            REQUIRE(v0d.data().data() == data0d.data());
            REQUIRE(v0d.data()[0] == 99.0f);

            // Zero-dim view
            std::vector<float> data_zero;
            std::vector<std::size_t> shape_zero_vec = {2, 0, 3};
            view<float> v_zero(data_zero, shape_zero_vec);
            REQUIRE(v_zero.data().empty());
            REQUIRE(v_zero.data().size() == 0);
        }
    }

    TEST_CASE("kmx::tensor::view - Operator[] Access", "[tensor][view]")
    {
        std::vector<float> data = create_test_data(12); // 2x2x3 -> flat indices 0..11
        // data view layout:
        // --- Slice 0 ---
        // [[0. 1. 2.]
        //  [3. 4. 5.]]
        // --- Slice 1 ---
        // [[6.  7.  8.]
        //  [9. 10. 11.]]
        std::vector<std::size_t> shape_vec = {2, 2, 3};
        view<float> v(data, shape_vec);
        const view<float>& cv = v;

        SECTION("Multi-dimensional Index Access (Valid)")
        {
            index idx_a = {0, 0, 0}; // Flat 0 -> Value 0.0
            index idx_b = {0, 1, 2}; // Flat 5 -> Value 5.0
            index idx_c = {1, 0, 1}; // Flat 7 -> Value 7.0
            index idx_d = {1, 1, 2}; // Flat 11 -> Value 11.0

            // Const access
            REQUIRE(cv[idx_a] == 0.0f);
            REQUIRE(cv[idx_b] == 5.0f);
            REQUIRE(cv[idx_c] == 7.0f);
            REQUIRE(cv[idx_d] == 11.0f);

            // Non-const access and modification
            REQUIRE(v[idx_b] == 5.0f);
            v[idx_b] = 55.5f;
            REQUIRE(v[idx_b] == 55.5f);
            REQUIRE(cv[idx_b] == 55.5f); // Check const view sees change
            REQUIRE(data[5] == 55.5f);   // Check underlying data changed
        }

        SECTION("Multi-dimensional Index Access (Error Conditions)")
        {
            // Index rank mismatch (index rank 2 vs view rank 3)
            index idx_rank2 = {1, 1};
            REQUIRE_THROWS_AS(v[idx_rank2], std::invalid_argument);

            // Index component out of bounds (index value >= dimension size)
            index idx_oob = {1, 2, 0}; // Dim 1 index 2 >= shape[1] (2)
            REQUIRE_THROWS_AS(v[idx_oob], std::out_of_range);

            // Scalar view access attempt (rank 1 index vs rank 0 view)
            std::vector<float> data0d = {99.0f};
            std::vector<std::size_t> shape0d_vec = {};
            view<float> v0d(data0d, shape0d_vec);
            index idx_rank1 = {0};                                    // Valid index object, but rank mismatch with view shape
            REQUIRE_THROWS_AS(v0d[idx_rank1], std::invalid_argument); // Throws in index::operator()
        }

        SECTION("Flat Index Access (Valid)")
        {
            // Const access
            REQUIRE(cv[0] == 0.0f);
            REQUIRE(cv[5] == 5.0f);
            REQUIRE(cv[7] == 7.0f);
            REQUIRE(cv[11] == 11.0f);

            // Non-const access and modification
            REQUIRE(v[5] == 5.0f);
            v[5] = 55.5f;
            REQUIRE(v[5] == 55.5f);
            REQUIRE(cv[5] == 55.5f);   // Const view sees change
            REQUIRE(data[5] == 55.5f); // Underlying data changed
        }

        SECTION("Flat Index Access (Error Conditions)")
        {
            // Out of bounds (equal to size)
            REQUIRE_THROWS_AS(v[12], std::out_of_range);

            // Out of bounds (large value)
            REQUIRE_THROWS_AS(v[100], std::out_of_range);

            // Scalar view flat access
            std::vector<float> data0d = {99.0f};
            std::vector<std::size_t> shape0d_vec = {};
            view<float> v0d(data0d, shape0d_vec);
            REQUIRE(v0d[0] == 99.0f);                     // Flat index 0 is valid for scalar
            REQUIRE_THROWS_AS(v0d[1], std::out_of_range); // Flat index 1 is invalid
        }
    }
}

// Copyright (c) 2025 - present KMX Systems. All rights reserved.
/// @file tensor.hpp
/// @brief Defines multi-dimensional index, shape, and non-owning view classes.
#pragma once
#ifndef PCH
    #include <algorithm>
    #include <array>
    #include <concepts>
    #include <cstddef>
    #include <functional>
    #include <limits>
    #include <numeric>
    #include <ranges>
    #include <span>
    #include <stdexcept>
    #include <string>
    #include <type_traits>
#endif

/// @brief Provides classes for tensor-like data structures and operations.
namespace kmx::tensor
{
    // Forward declare view for dimension_shape usage if needed (not strictly required here)
    template <typename T, typename _IndexType>
    class view;

    /// @brief Represents a multi-dimensional index with a fixed maximum rank.
    /// Enforces rank constraints and calculates flat offsets for row-major data.
    class index
    {
    public:
        /// @brief Maximum supported rank (number of dimensions).
        static constexpr std::size_t max_rank = 4u;
        /// @brief Minimum supported rank (must have at least one dimension).
        static constexpr std::size_t min_rank = 1u;

        /// @brief Underlying storage type for index components.
        using storage_type = std::array<std::size_t, max_rank>;
        /// @brief Type of individual index components and rank.
        using value_type = std::size_t;
        /// @brief Type used to represent the shape for flat index calculation.
        using shape_span_type = std::span<const value_type>;

        /// @brief Default constructor is deleted; rank must be specified.
        index() = delete;

        /// @brief Constructs an index from an initializer list.
        /// @param il Initializer list of index components (e.g., `{row, col}`).
        /// @throws std::invalid_argument If the list size is outside the allowed rank range [`min_rank`, `max_rank`].
        index(std::initializer_list<value_type> il): rank_(static_cast<value_type>(il.size()))
        {
            if (rank_ < min_rank || rank_ > max_rank) [[unlikely]]
            {
                // Consider using std::format in C++20/23/26 for cleaner formatting if available/preferred
                throw std::invalid_argument("Initializer list size (" + std::to_string(rank_) + ") is outside the allowed rank range [" +
                                            std::to_string(min_rank) + ", " + std::to_string(max_rank) + "]");
            }
            // std::copy_n is generally efficient for contiguous iterators like initializer_list
            std::copy_n(il.begin(), rank_, indexes_.begin());
        }

        /// @brief Constructs an index from an input range (e.g., vector, span).
        /// @tparam R Type of the input range.
        /// @param range Input range providing the index components.
        /// @throws std::invalid_argument If the range size is outside the allowed rank range [`min_rank`, `max_rank`].
        template <std::ranges::input_range R>
            requires std::convertible_to<std::ranges::range_value_t<R>, value_type>
        explicit index(R&& range)
        {
            // Use sized_range optimization where possible
            if constexpr (std::ranges::sized_range<R>)
            {
                rank_ = static_cast<value_type>(std::ranges::size(range));
            }
            else // Fallback for input ranges (potentially less efficient)
            {
                // Note: std::ranges::distance can be O(N) for non-random_access ranges.
                // Consider requiring forward_range if performance with input iterators is a concern.
                auto dist = std::ranges::distance(range);
                rank_ = static_cast<value_type>(dist);
            }

            if (rank_ < min_rank || rank_ > max_rank) [[unlikely]]
            {
                throw std::invalid_argument("Input range size (" + std::to_string(rank_) + ") is outside the allowed rank range [" +
                                            std::to_string(min_rank) + ", " + std::to_string(max_rank) + "]");
            }

            // std::ranges::copy_n is efficient
            std::ranges::copy_n(std::ranges::begin(range), rank_, indexes_.begin());
        }

        // Copy/Move constructors/assignment are implicitly defaulted (efficient shallow copies).
        index(const index&) = default;
        index(index&&) noexcept = default;
        index& operator=(const index&) = default;
        index& operator=(index&&) noexcept = default;

        /// @brief Returns the actual rank (number of dimensions) of this index.
        /// @return The rank, guaranteed to be between `min_rank` and `max_rank`.
        [[nodiscard]] constexpr value_type rank() const noexcept { return rank_; }

        /// @brief Provides access to the valid index components as a read-only span.
        /// The span's size equals the index's rank.
        /// @return A `std::span` viewing the active index components.
        [[nodiscard]] std::span<const value_type> values() const noexcept { return {indexes_.data(), rank_}; }

        /// @brief Gets read-only access to the full underlying storage array.
        /// Note that only the first `rank()` elements are valid index components.
        /// @return A const reference to the internal `std::array`.
        [[nodiscard]] const storage_type& array() const noexcept { return indexes_; }

        /// @brief Calculates the flat (linear) index corresponding to this multi-dimensional index
        ///        within a tensor of the given shape, assuming row-major layout.
        /// @param shape A span representing the dimensions (shape) of the tensor.
        ///              Its size must match the rank of this index.
        /// @return The calculated zero-based flat index.
        /// @throws std::invalid_argument If the shape rank does not match the index rank.
        /// @throws std::out_of_range If any index component is out of bounds for the corresponding dimension size in `shape`.
        /// @throws std::overflow_error If intermediate stride calculation overflows.
        [[nodiscard]] value_type operator()(shape_span_type shape) const
        {
            if (shape.size() != rank_) [[unlikely]] // Hint for compilers
            {
                throw std::invalid_argument("Index rank (" + std::to_string(rank_) + ") mismatch with shape rank (" +
                                            std::to_string(shape.size()) + ")");
            }

            value_type flat_index = 0;
            value_type stride = 1u;

            // Iterate from the innermost dimension outwards (row-major)
            // Unrolling slightly might help for very small ranks, but a loop is general.
            for (value_type i = 0; i < rank_; ++i)
            {
                // Reverse index calculation is simple and likely optimized by the compiler
                const value_type dim_index = rank_ - 1u - i;
                const value_type current_dim_size = shape[dim_index];
                const value_type current_index = indexes_[dim_index]; // Component for current dimension

                // Bounds check: Handles size 0 dimensions correctly (any index >= 0 is invalid).
                if (current_index >= current_dim_size) [[unlikely]]
                {
                    throw std::out_of_range("Index component " + std::to_string(current_index) + " at dimension " +
                                            std::to_string(dim_index) + " is out of bounds for size " + std::to_string(current_dim_size));
                }

                // Accumulate flat index contribution from this dimension
                flat_index += current_index * stride; // Potential overflow if total size is huge (checked in view::validate_size)

                // Update stride for the next dimension outwards, only if not the last dimension
                if (i + 1u < rank_)
                {
                    // Check for potential overflow *before* multiplication
                    // This check is important for correctness if dimensions are very large.
                    constexpr value_type max_val = std::numeric_limits<value_type>::max();
                    // Avoid division by zero, check current_dim_size > 0
                    if (current_dim_size > 0 && stride > max_val / current_dim_size) [[unlikely]]
                    {
                        throw std::overflow_error("Stride calculation overflow for shape");
                    }
                    stride *= current_dim_size;
                    // Explicitly handling stride=0 if current_dim_size is 0 prevents potential issues
                    // if the overflow check above has edge cases, and improves clarity.
                    // If dim_size is 0, the total size is 0, so subsequent strides should also be 0.
                    if (current_dim_size == 0) [[unlikely]] // A zero dimension means total size is 0
                    {
                        stride = 0;
                        // Optimization: If stride becomes 0, further iterations won't change flat_index.
                        // However, the bounds checks for remaining indices still need to run.
                        // Could break early if flat_index calculation was the *only* goal, but bounds checks are required.
                    }
                }
            }

            return flat_index;
        }

    private:
        /// @brief Internal storage for index components. Use direct initialization.
        storage_type indexes_ {};
        /// @brief Actual rank of this index instance.
        value_type rank_;
    };

    /// @brief Represents the dimensions (shape) of a tensor view.
    /// Essentially a `std::span<const std::size_t>` with added equality comparison.
    /// Note: As a non-owning span, the underlying data must outlive the dimension_shape instance.
    class dimension_shape: public std::span<const std::size_t>
    {
    public:
        /// @brief Inherit constructors from `std::span`.
        using std::span<const std::size_t>::span;

        /// @brief Default constructor creates an empty shape (rank 0).
        constexpr dimension_shape() noexcept = default;

        // Note: Constructing directly from std::initializer_list is dangerous for spans
        // as the list's lifetime might be shorter than the span's.
        // Provide constructors from containers/arrays with guaranteed lifetime.

        /// @brief Creates a shape from a `std::array`.
        template <std::size_t N>
        constexpr dimension_shape(const std::array<std::size_t, N>& arr) noexcept: std::span<const std::size_t>(arr)
        {
        }

        /// @brief Creates a shape from a C-style array.
        template <std::size_t N>
        constexpr dimension_shape(const std::size_t (&arr)[N]) noexcept: std::span<const std::size_t>(arr)
        {
        }

        /// @brief Compares two shapes for element-wise equality.
        /// Two shapes are equal if they have the same rank and all corresponding dimension sizes match.
        /// @param lhs Left-hand side shape.
        /// @param rhs Right-hand side shape.
        /// @return `true` if shapes are equal, `false` otherwise.
        friend constexpr bool operator==(const dimension_shape& lhs, const dimension_shape& rhs) noexcept
        {
            // std::ranges::equal is constexpr-friendly and efficient
            return std::ranges::equal(lhs, rhs);
        }

        // operator!= is automatically synthesized in C++20 and later
    };

    /// @brief A non-owning view over a contiguous block of memory, interpreted with a specific shape.
    /// Provides multi-dimensional access ([index]) and flat access ([flat_index]) to the underlying data.
    /// @tparam T The type of elements in the view (can be const).
    /// @tparam _IndexType The type used for multi-dimensional indexing (defaults to `kmx::tensor::index`).
    template <typename T, typename _IndexType = index>
    class view
    {
    public:
        /// @brief The type of elements viewed by this instance.
        using value_type = T;
        /// @brief The type used for multi-dimensional indexing.
        using index_type = _IndexType;
        /// @brief The type of the underlying data span.
        using data_span_type = std::span<T>;
        /// @brief The type used for flat (linear) indexing.
        using flat_index_type = typename index_type::value_type;
        /// @brief The non-const version of the element type.
        using non_const_value_type = std::remove_const_t<value_type>;

        /// @brief Compile-time flag indicating if the view holds const data (`value_type` is const).
        static constexpr bool is_const_type = std::is_const_v<value_type>;

        /// @brief Default constructor. Creates an empty view with rank 0 and size 0.
        constexpr view() noexcept = default;

        /// @brief Constructs a view with a specific shape over the given data span.
        /// @param data The `std::span` representing the contiguous data block.
        /// @param shape The desired multi-dimensional shape (`dimension_shape`). The underlying data for the shape must outlive this view.
        /// @throws std::invalid_argument If the size of `data` does not match the total number of elements required by `shape`.
        /// @throws std::overflow_error If the product of dimensions in `shape` exceeds `std::size_t::max`.
        constexpr view(data_span_type data, dimension_shape shape): data_(data), shape_(shape) { validate_size(); }

        /// @brief Constructs a 1D view (rank 1) over the given data span.
        /// The shape is deduced from the size of the data span.
        /// @param data The `std::span` representing the 1D data.
        explicit constexpr view(data_span_type data): data_(data) // Initialize data_ first
        {
            // Deduce shape for 1D case
            if (data.empty())
            {
                // Represent empty 1D view as rank 1, shape {0}.
                shape_storage_1d_[0] = 0;
                shape_ = dimension_shape(shape_storage_1d_.data(), 1); // Point shape_ to internal storage
            }
            else
            {
                // Use internal storage for the single dimension size
                shape_storage_1d_[0] = data.size();
                // Point shape_ span to the internal storage
                shape_ = dimension_shape(shape_storage_1d_.data(), 1);
            }
            // Size validation is implicitly correct by construction here.
        }

        // Defaulted copy/move constructors and assignment operators (perform efficient shallow copies).
        view(view&&) noexcept = default;
        view(const view&) = default;
        view& operator=(view&&) noexcept = default;
        view& operator=(const view&) = default;

        /// @brief Converting constructor from a non-const view to a const view.
        /// Allows implicit conversion from `view<U>` to `view<const U>`.
        template <typename OtherT>
            requires std::is_same_v<non_const_value_type, std::remove_const_t<OtherT>> && is_const_type && // This view must be const
                         (!std::is_const_v<OtherT>) // The other view must be non-const
        constexpr view(const view<OtherT, _IndexType>& other) noexcept: data_(other.data()), shape_(other.shape())
        {
            // If the other view used the 1D constructor, its shape points to its internal storage.
            // We need to copy that single dimension size into *our* internal storage and point our shape_ there.
            // Check if other.shape_ points into other.shape_storage_1d_
            if (shape_.data() == other.get_shape_storage_1d_ptr()) [[unlikely]] // Optimization hint
            {
                shape_storage_1d_[0] = other.get_shape_storage_1d_value();
                shape_ = dimension_shape(shape_storage_1d_.data(), 1); // Point to *our* storage
            }
            // Otherwise, shape_ was already correctly copied (points to external storage).
        }

        /// @brief Gets the shape of the tensor view.
        /// @return A `dimension_shape` object representing the dimensions.
        [[nodiscard]] constexpr dimension_shape shape() const noexcept { return shape_; }

        /// @brief Gets the underlying data span.
        /// @return A `std::span` viewing the contiguous data.
        [[nodiscard]] constexpr data_span_type data() const noexcept { return data_; }

        /// @brief Gets the rank (number of dimensions) of the view.
        /// @return The rank (size of the shape).
        [[nodiscard]] constexpr std::size_t rank() const noexcept { return shape_.size(); }

        /// @brief Gets the total number of elements in the view (product of dimensions).
        /// @return The total size.
        [[nodiscard]] constexpr std::size_t size() const noexcept { return data_.size(); }

        /// @brief Checks if the view is empty (has zero elements).
        /// @return `true` if size is 0, `false` otherwise.
        [[nodiscard]] constexpr bool empty() const noexcept { return data_.empty(); }

        /// @brief Const access to an element using a multi-dimensional index.
        /// @param idx The multi-dimensional index (`index_type`) of the element.
        /// @return A const reference to the element.
        /// @throws std::invalid_argument If `idx.rank()` does not match `this->rank()`.
        /// @throws std::out_of_range If `idx` is out of bounds according to `shape()` or the calculated flat index exceeds `size()`.
        [[nodiscard]] const value_type& operator[](const index_type& idx) const
        {
            // Calculates flat index and performs index component bounds checks vs shape
            const flat_index_type flat_idx = idx(shape_);
            // Additional check: Ensure flat index is within the bounds of the data span itself.
            if (flat_idx >= size()) [[unlikely]]
            {
                throw std::out_of_range("Calculated flat index (" + std::to_string(flat_idx) + ") is out of range for view size (" +
                                        std::to_string(size()) + ")");
            }
            // Use span::operator[] which is unchecked in standard C++, but we just checked bounds.
            return data_[flat_idx];
            // Alternatively, use data_.at(flat_idx); if preferred for guaranteed check (might incur overhead).
        }

        /// @brief Non-const access to an element using a multi-dimensional index.
        /// Enabled only if `value_type` is not const.
        /// @param idx The multi-dimensional index (`index_type`) of the element.
        /// @return A non-const reference to the element.
        /// @throws std::invalid_argument If `idx.rank()` does not match `this->rank()`.
        /// @throws std::out_of_range If `idx` is out of bounds according to `shape()` or the calculated flat index exceeds `size()`.
        [[nodiscard]] value_type& operator[](const index_type& idx)
            requires(!is_const_type) // C++20 requires clause for SFINAE
        {
            const flat_index_type flat_idx = idx(shape_);
            if (flat_idx >= size()) [[unlikely]]
            {
                throw std::out_of_range("Calculated flat index (" + std::to_string(flat_idx) + ") is out of range for view size (" +
                                        std::to_string(size()) + ")");
            }
            return data_[flat_idx];
        }

        /// @brief Const access to an element using a flat (linear) index.
        /// Assumes row-major layout if mapping from multi-dimensional index.
        /// @param flat_idx The zero-based flat index.
        /// @return A const reference to the element.
        /// @throws std::out_of_range If `flat_idx` is out of bounds (`>= size()`).
        [[nodiscard]] const value_type& operator[](const flat_index_type flat_idx) const
        {
            // Explicit bounds check for flat index access.
            if (flat_idx >= size()) [[unlikely]]
            {
                throw std::out_of_range("Flat index (" + std::to_string(flat_idx) + ") out of range for view size (" +
                                        std::to_string(size()) + ")");
            }
            return data_[flat_idx];
        }

        /// @brief Non-const access to an element using a flat (linear) index.
        /// Enabled only if `value_type` is not const.
        /// Assumes row-major layout if mapping from multi-dimensional index.
        /// @param flat_idx The zero-based flat index.
        /// @return A non-const reference to the element.
        /// @throws std::out_of_range If `flat_idx` is out of bounds (`>= size()`).
        [[nodiscard]] value_type& operator[](const flat_index_type flat_idx)
            requires(!is_const_type) // C++20 requires clause
        {
            if (flat_idx >= size()) [[unlikely]]
            {
                throw std::out_of_range("Flat index (" + std::to_string(flat_idx) + ") out of range for view size (" +
                                        std::to_string(size()) + ")");
            }
            return data_[flat_idx];
        }

        // --- Sub-view Creation ---

        /// @brief Creates a const 1D view of a specific row. Requires the current view to be 2D.
        /// @param row_index The zero-based index of the row to view.
        /// @return A `view<const value_type, _IndexType>` representing the row.
        /// @throws std::logic_error If the current view is not rank 2.
        /// @throws std::out_of_range If `row_index` is out of bounds.
        [[nodiscard]] view<const value_type, _IndexType> row_view(std::size_t row_index) const
        {
            if (rank() != 2) [[unlikely]]
            {
                throw std::logic_error("row_view requires a 2D tensor view (rank 2), current rank is " + std::to_string(rank()));
            }
            // Assume shape has at least 2 elements due to rank check
            const std::size_t rows = shape()[0];
            const std::size_t cols = shape()[1];

            if (row_index >= rows) [[unlikely]]
            {
                throw std::out_of_range("Row index " + std::to_string(row_index) + " out of range for " + std::to_string(rows) + " rows");
            }

            // Handle zero columns case efficiently: return empty 1D view
            if (cols == 0) [[unlikely]]
            {
                // Construct using empty span, which results in rank 1, shape {0} via 1D constructor
                return view<const value_type, _IndexType>(data_span_type {});
            }

            // Calculate start offset and size for the row's data slice
            // Overflow check: row_index * cols could overflow if rows/cols are huge.
            // However, size() check in constructor should prevent data_.size() being too large.
            const std::size_t start_offset = row_index * cols;

            // Create a subspan representing the row's data
            // subspan itself performs bounds checks in debug builds usually.
            data_span_type row_data_span = data_.subspan(start_offset, cols);

            // Return a new 1D view using the 1D view constructor (efficiently sets up shape)
            return view<const value_type, _IndexType>(row_data_span);
        }

        /// @brief Creates a non-const 1D view of a specific row. Requires the current view to be 2D and non-const.
        /// Enabled only if `value_type` is not const.
        /// @param row_index The zero-based index of the row to view.
        /// @return A `view<non_const_value_type, _IndexType>` representing the row.
        /// @throws std::logic_error If the current view is not rank 2.
        /// @throws std::out_of_range If `row_index` is out of bounds.
        [[nodiscard]] view<non_const_value_type, _IndexType> row_view(std::size_t row_index)
            requires(!is_const_type) // C++20 requires clause
        {
            if (rank() != 2) [[unlikely]]
            {
                throw std::logic_error("row_view requires a 2D tensor view (rank 2), current rank is " + std::to_string(rank()));
            }
            const std::size_t rows = shape()[0];
            const std::size_t cols = shape()[1];
            if (row_index >= rows) [[unlikely]]
            {
                throw std::out_of_range("Row index " + std::to_string(row_index) + " out of range for " + std::to_string(rows) + " rows");
            }

            if (cols == 0) [[unlikely]]
            {
                // Use span of non-const type
                return view<non_const_value_type, _IndexType>(std::span<non_const_value_type> {});
            }

            const std::size_t start_offset = row_index * cols;
            // Implicit conversion from span<T> to span<non_const_value_type> works here.
            std::span<non_const_value_type> row_data_span = data_.subspan(start_offset, cols);

            // Return a new 1D view using the 1D view constructor
            return view<non_const_value_type, _IndexType>(row_data_span);
        }

        // --- Internal helpers for const conversion ---
        // These are needed because dimension_shape doesn't own data.
        // If shape_ points to internal storage, the converting constructor needs access.

        /// @brief Internal helper: Gets pointer to internal 1D shape storage.
        /// @return Pointer to the single element in `shape_storage_1d_`.
        [[nodiscard]] constexpr const std::size_t* get_shape_storage_1d_ptr() const noexcept { return shape_storage_1d_.data(); }
        /// @brief Internal helper: Gets value from internal 1D shape storage.
        /// @return The value stored in `shape_storage_1d_[0]`.
        [[nodiscard]] constexpr std::size_t get_shape_storage_1d_value() const noexcept { return shape_storage_1d_[0]; }

    private:
        /// @brief Non-owning span over the actual data.
        data_span_type data_;
        /// @brief Non-owning view of the dimensions/shape.
        dimension_shape shape_;
        /// @brief Internal storage for the shape when using the 1D constructor ONLY.
        /// `shape_` will point into this array in that specific case. Size 1 is sufficient.
        std::array<std::size_t, 1> shape_storage_1d_ {};

        /// @brief Validates that the data span size matches the expected size from the shape.
        /// @throws std::invalid_argument If sizes mismatch.
        /// @throws std::overflow_error If shape product overflows.
        constexpr void validate_size() const
        {
            // Calculate expected size using unsigned long long for intermediate overflow check.
            // Use 1ull to ensure the initial type promotion.
            // std::accumulate is constexpr since C++20.
            unsigned long long expected_size_ull = 1ull;
            if (!shape_.empty())
            {
                // Check for zero dimension first, as it dictates the final size is 0.
                if (std::ranges::find(shape_, 0u) != shape_.end())
                {
                    expected_size_ull = 0u;
                }
                else
                {
                    // Calculate product only if no zero dimension exists.
                    expected_size_ull = std::accumulate(shape_.begin(), shape_.end(), 1ull, std::multiplies<>());

                    // Check for overflow against size_t::max.
                    constexpr auto max_size_t_ull = static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max());
                    if (expected_size_ull > max_size_t_ull) [[unlikely]]
                    {
                        throw std::overflow_error("Product of dimensions exceeds maximum size_t");
                    }
                }
            }
            // Note: If shape_ is empty (rank 0), expected_size_ull remains 1 (scalar).

            const std::size_t final_expected_size = static_cast<std::size_t>(expected_size_ull);

            // Final check against actual data span size
            if (data_.size() != final_expected_size) [[unlikely]]
            {
                // Create shape string for error message (can be simplified or made more efficient if needed)
                std::string shape_str = "[";
                if (!shape_.empty())
                {
                    for (size_t i = 0; i < shape_.size(); ++i)
                    {
                        shape_str += std::to_string(shape_[i]) + (i == shape_.size() - 1 ? "" : ", ");
                    }
                }
                shape_str += "]";
                if (shape_.empty())
                    shape_str = "<rank 0>"; // Special case description

                throw std::invalid_argument("Data span size (" + std::to_string(data_.size()) +
                                            ") does not match total elements required by shape " + shape_str + " (" +
                                            std::to_string(final_expected_size) + ")");
            }
        }

        /// @brief Friend declaration to allow const conversion constructor access to private members.
        template <typename OtherT, typename OtherIndexType>
        friend class view; // Grant access to other view instantiations
    };

} // namespace kmx::tensor

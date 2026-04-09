// Copyright (c) 2026 - present KMX Systems. All rights reserved.
/// @file capabilities.hpp
/// @brief Defines backend capabilities for the runtime FFT strategy router.
#pragma once

#include <cstdint>

namespace kmx::fft::backend {

/// @brief Represents the available FFT backends in the system.
enum class backend_id : std::uint8_t {
    software,
    avx2,
    avx512, // Placeholder
    opencl,
    cuda    // Placeholder
};

[[nodiscard]] constexpr const char* to_string(backend_id id) noexcept {
    switch (id) {
        case backend_id::software: return "software";
        case backend_id::avx2: return "avx2";
        case backend_id::avx512: return "avx512";
        case backend_id::opencl: return "opencl";
        case backend_id::cuda: return "cuda";
        default: return "unknown";
    }
}

/// @brief Describes the capabilities of a specific backend instance.
struct capabilities {
    backend_id id;
    bool enabled = false;             // true if supported by build and hardware
    bool supports_1d_pow2 = false;
    bool supports_1d_smooth = false;
    bool supports_2d = false;
    bool supports_float = false;
    bool supports_double = false;
};

} // namespace kmx::fft::backend

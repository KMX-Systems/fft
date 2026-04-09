// Copyright (c) 2026 - present KMX Systems. All rights reserved.
/// @file opencl.hpp
/// @brief OpenCL backend for kmx::fft.
#pragma once

#ifndef PCH
    #define CL_TARGET_OPENCL_VERSION 200
    #include <CL/cl.h>
    #include <complex>
    #include <concepts>
    #include <kmx/tensor.hpp>
    #include <stdexcept>
    #include <vector>
    #include <string>
    #include <bit>
    #include <cmath>
    #include <numbers>
    #include <coroutine>
#endif

#include "software.hpp"

namespace kmx::fft::backend {

class opencl_awaitable {
public:
    cl_event event_;
    bool destroy_;
    mutable bool awaited_ = false;

    opencl_awaitable(cl_event event, bool destroy = true) : event_(event), destroy_(destroy) {}

    ~opencl_awaitable() {
        if (destroy_ && event_) {
            if (!awaited_) {
                clWaitForEvents(1, &event_);
            }
            clReleaseEvent(event_);
        }
    }

    // Disallow copy
    opencl_awaitable(const opencl_awaitable&) = delete;
    opencl_awaitable& operator=(const opencl_awaitable&) = delete;

    // Move semantics
    opencl_awaitable(opencl_awaitable&& other) noexcept : event_(other.event_), destroy_(other.destroy_), awaited_(other.awaited_) {
        other.event_ = nullptr;
    }

    constexpr bool await_ready() const noexcept { return false; }

    void await_suspend(std::coroutine_handle<> h) const noexcept {
        awaited_ = true;
        if (event_) {
            clSetEventCallback(event_, CL_COMPLETE, [](cl_event /*evt*/, cl_int /*status*/, void *data) {
                auto handle = std::coroutine_handle<>::from_address(data);
                handle.resume();
            }, h.address());
        } else {
            h.resume();
        }
    }

    void await_resume() const {
        awaited_ = true;
        if (event_) {
            clWaitForEvents(1, &event_);
        }
    }
};

template <ComplexNumber _ComplexNumber>
class opencl {
public:
    using allocator_type = std::allocator<_ComplexNumber>;
    using value_type = typename _ComplexNumber::value_type;

private:
    static constexpr bool is_complex_double = std::same_as<_ComplexNumber, std::complex<double>>;

    cl_context       context_      = nullptr;
    cl_command_queue queue_        = nullptr;
    cl_program       program_      = nullptr;
    cl_kernel        k_bit_reverse_ = nullptr;
    cl_kernel        k_fft_stage_   = nullptr;
    cl_kernel        k_scale_       = nullptr;

    // Persistent device buffers — reallocated only when capacity is exceeded
    cl_mem      data_buf_     = nullptr;
    std::size_t data_cap_     = 0;
    cl_mem      twiddle_buf_  = nullptr;
    std::size_t twiddle_cap_  = 0;
    std::size_t twiddle_n_    = 0;          // n for which twiddles are currently cached
    value_type  twiddle_sign_ = value_type(0);

    // Persistent software fallback for non-power-of-2 sizes
    software<_ComplexNumber> fallback_;

    void cleanup() {
        if (k_scale_)      clReleaseKernel(k_scale_);
        if (k_fft_stage_)  clReleaseKernel(k_fft_stage_);
        if (k_bit_reverse_) clReleaseKernel(k_bit_reverse_);
        if (twiddle_buf_) { clReleaseMemObject(twiddle_buf_); twiddle_buf_ = nullptr; }
        if (data_buf_)    { clReleaseMemObject(data_buf_);    data_buf_    = nullptr; }
        if (program_) clReleaseProgram(program_);
        if (queue_)   clReleaseCommandQueue(queue_);
        if (context_) clReleaseContext(context_);
    }

public:
    opencl() {
        cl_uint num_platforms = 0;
        if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
            throw std::runtime_error("No OpenCL platforms found");
        }

        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

        cl_device_id device_id = nullptr;
        for (auto plat : platforms) {
            cl_uint num_devices = 0;
            clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &device_id, &num_devices);
            if (num_devices > 0) break;
        }
        if (!device_id) throw std::runtime_error("No OpenCL devices found");

        cl_int err;
        context_ = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create OpenCL context");

        queue_ = clCreateCommandQueueWithProperties(context_, device_id, nullptr, &err);
        if (err != CL_SUCCESS) {
            cleanup();
            throw std::runtime_error("Failed to create OpenCL command queue");
        }

        std::string src = R"(
            #if defined(USE_DOUBLE)
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            typedef double2 complex_t;
            typedef double  real_t;
            #else
            typedef float2 complex_t;
            typedef float  real_t;
            #endif

            kernel void bit_reverse(global complex_t* data, int log2n) {
                int i = get_global_id(0);
                int n = 1 << log2n;
                if (i < n) {
                    int j = 0, x = i;
                    for (int k = 0; k < log2n; ++k) { j = (j << 1) | (x & 1); x >>= 1; }
                    if (i < j) { complex_t tmp = data[i]; data[i] = data[j]; data[j] = tmp; }
                }
            }

            // fft_stage: twiddles[k] = exp(2*pi*i*sign*k/n), pre-computed on host.
            // Index for stage (len, idx): twiddles[idx * (n / len)]
            kernel void fft_stage(global complex_t* data,
                                  global const complex_t* twiddles,
                                  int len, int n) {
                int id       = get_global_id(0);
                int half_len = len >> 1;
                int group    = id / half_len;
                int idx      = id % half_len;
                int i        = group * len + idx;
                int j        = i + half_len;
                int tw_idx   = idx * (n / len);
                complex_t w    = twiddles[tw_idx];
                complex_t u    = data[i];
                complex_t v_in = data[j];
                complex_t v;
                v.x = v_in.x * w.x - v_in.y * w.y;
                v.y = v_in.x * w.y + v_in.y * w.x;
                data[i] = (complex_t)(u.x + v.x, u.y + v.y);
                data[j] = (complex_t)(u.x - v.x, u.y - v.y);
            }

            kernel void scale(global complex_t* data, real_t s) {
                int i = get_global_id(0);
                data[i].x *= s;
                data[i].y *= s;
            }
        )";

        const char* src_ptr = src.c_str();
        program_ = clCreateProgramWithSource(context_, 1, &src_ptr, nullptr, &err);
        
        std::string build_options = is_complex_double ? "-DUSE_DOUBLE" : "";
        clBuildProgram(program_, 1, &device_id, build_options.c_str(), nullptr, nullptr);

        k_bit_reverse_ = clCreateKernel(program_, "bit_reverse", &err);
        k_fft_stage_   = clCreateKernel(program_, "fft_stage",   &err);
        k_scale_       = clCreateKernel(program_, "scale",        &err);

        if (!k_bit_reverse_ || !k_fft_stage_ || !k_scale_) {
            cleanup();
            throw std::runtime_error("Failed to build OpenCL kernels.");
        }
    }

    ~opencl() {
        cleanup();
    }

    opencl_awaitable transform_1d(tensor::view<_ComplexNumber> v_inout, bool inverse) {
        const std::size_t n = v_inout.size();
        if (n <= 1) return opencl_awaitable{nullptr, false};

        if (!std::has_single_bit(n)) {
            fallback_.transform_1d(v_inout, inverse);
            return opencl_awaitable{nullptr, false};
        }

        cl_int err = CL_SUCCESS;

        // 1. Ensure persistent data buffer has enough capacity
        if (n > data_cap_) {
            if (data_buf_) { clReleaseMemObject(data_buf_); data_buf_ = nullptr; }
            data_buf_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                       n * sizeof(_ComplexNumber), nullptr, &err);
            if (err != CL_SUCCESS) throw std::runtime_error("Failed to allocate OpenCL data buffer");
            data_cap_ = n;
        }

        // 2. Upload input (non-blocking; in-order queue guarantees ordering with kernels)
        clEnqueueWriteBuffer(queue_, data_buf_, CL_FALSE, 0,
                             n * sizeof(_ComplexNumber), v_inout.data().data(),
                             0, nullptr, nullptr);

        // 3. Compute and cache twiddle factors only when n or sign changes
        const value_type sign = inverse ? value_type(1) : value_type(-1);
        if (n != twiddle_n_ || sign != twiddle_sign_) {
            std::vector<_ComplexNumber> twiddles(n);
            const value_type base =
                value_type(2) * std::numbers::pi_v<value_type> * sign / static_cast<value_type>(n);
            for (std::size_t k = 0; k < n; ++k) {
                const value_type theta = base * static_cast<value_type>(k);
                twiddles[k] = _ComplexNumber(std::cos(theta), std::sin(theta));
            }
            if (n > twiddle_cap_) {
                if (twiddle_buf_) { clReleaseMemObject(twiddle_buf_); twiddle_buf_ = nullptr; }
                twiddle_buf_ = clCreateBuffer(context_, CL_MEM_READ_ONLY,
                                             n * sizeof(_ComplexNumber), nullptr, &err);
                if (err != CL_SUCCESS) throw std::runtime_error("Failed to allocate OpenCL twiddle buffer");
                twiddle_cap_ = n;
            }
            // Blocking upload so the first kernel dispatch sees valid twiddle data
            clEnqueueWriteBuffer(queue_, twiddle_buf_, CL_TRUE, 0,
                                 n * sizeof(_ComplexNumber), twiddles.data(),
                                 0, nullptr, nullptr);
            twiddle_n_    = n;
            twiddle_sign_ = sign;
        }

        // 4. Bit-reversal permutation
        const int log2n = static_cast<int>(std::countr_zero(n));
        clSetKernelArg(k_bit_reverse_, 0, sizeof(cl_mem), &data_buf_);
        clSetKernelArg(k_bit_reverse_, 1, sizeof(int),    &log2n);
        const size_t n_global = n;
        clEnqueueNDRangeKernel(queue_, k_bit_reverse_, 1, nullptr,
                               &n_global, nullptr, 0, nullptr, nullptr);

        // 5. Butterfly stages — set stable args (data, twiddles, n) once outside loop
        const int n_i = static_cast<int>(n);
        clSetKernelArg(k_fft_stage_, 0, sizeof(cl_mem), &data_buf_);
        clSetKernelArg(k_fft_stage_, 1, sizeof(cl_mem), &twiddle_buf_);
        clSetKernelArg(k_fft_stage_, 3, sizeof(int),    &n_i);
        const size_t ng = n / 2;
        for (std::size_t len = 2; len <= n; len <<= 1) {
            const int len_i = static_cast<int>(len);
            clSetKernelArg(k_fft_stage_, 2, sizeof(int), &len_i);
            clEnqueueNDRangeKernel(queue_, k_fft_stage_, 1, nullptr,
                                   &ng, nullptr, 0, nullptr, nullptr);
        }

        // 6. GPU-side inverse scaling — avoids a host readback stall just for multiply
        if (inverse) {
            const value_type s = value_type(1) / static_cast<value_type>(n);
            clSetKernelArg(k_scale_, 0, sizeof(cl_mem),    &data_buf_);
            clSetKernelArg(k_scale_, 1, sizeof(value_type), &s);
            const size_t ns = n;
            clEnqueueNDRangeKernel(queue_, k_scale_, 1, nullptr,
                                   &ns, nullptr, 0, nullptr, nullptr);
        }

        // 7. Non-blocking readback — caller co_awaits or destructor blocks
        cl_event read_event;
        clEnqueueReadBuffer(queue_, data_buf_, CL_FALSE, 0,
                            n * sizeof(_ComplexNumber), v_inout.data().data(),
                            0, nullptr, &read_event);

        return opencl_awaitable{read_event, true};
    }

    template <typename _InputComplexNumber>
    opencl_awaitable transform_1d(const tensor::view<_InputComplexNumber>& v_in, tensor::view<_ComplexNumber> v_out, bool inverse) {
        if (reinterpret_cast<const void*>(v_in.data().data()) != reinterpret_cast<const void*>(v_out.data().data())) {
            std::copy(v_in.data().begin(), v_in.data().end(), v_out.data().begin());
        }
        return transform_1d(v_out, inverse);
    }

    opencl_awaitable transform_2d(tensor::view<_ComplexNumber> v_inout, bool inverse) {
        fallback_.transform_2d(v_inout, inverse);
        return opencl_awaitable{nullptr, false};
    }

    template <typename _InputComplexNumber>
    opencl_awaitable transform_2d(const tensor::view<_InputComplexNumber>& v_in, tensor::view<_ComplexNumber> v_out, bool inverse) {
        return transform_2d(v_out, inverse);
    }
};

} // namespace kmx::fft::backend

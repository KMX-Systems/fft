C++ Fast-Fourier-Transform (FFT) library.

## Architecture

### Public API

All transforms go through `kmx::fft::engine<T, Backend>`, a thin non-copyable facade:

```cpp
kmx::fft::engine<std::complex<double>> eng;          // software backend (default)
kmx::fft::engine<std::complex<double>,
    kmx::fft::backend::avx2<std::complex<double>>> eng; // AVX2 backend

tensor::view<std::complex<double>> v{std::span(data)};
eng.transform_1d(v);            // forward, in-place
eng.transform_2d(v, true);      // inverse, in-place (v must be a 2D view)
```

Optional **plan objects** pre-compute reusable state for repeated calls to the same shape:

```cpp
auto plan = eng.make_plan(1024);
plan.execute(v);   // zero allocation, zero cache lookup

auto plan2d = eng.make_plan_2d(256, 256);
plan2d.execute(v2d); // repeated 2D row/column execution with preallocated scratch
```

For the AVX2 backend, 1D plans reuse cached SIMD twiddle tables and Bluestein state, while 2D plans reuse cached row/column plan state plus per-thread scratch buffers.

### Backend model

A backend is any type satisfying the `FftBackend<T>` concept: it must expose `transform_1d`, `transform_2d`, and an `allocator_type`. The software backend is always compiled in as a fallback; specialised backends (`avx2`, `opencl`) are gated by preprocessor defines set by Qbs properties.

### Software backend

Iterative Cooley–Tukey with a cached twiddle-factor table and bit-reversal reorder. Handles arbitrary N via Bluestein's algorithm (pads to the next power of 2 ≥ 2N − 1, runs three power-of-2 FFTs).

### AVX2 backend (`avx2.hpp`)

All internal allocations use a `posix_memalign(32)` allocator so every internal buffer is 32-byte aligned. Data pointers arriving from callers (e.g. `std::vector`) are accessed with `_mm256_loadu_pd/storeu_pd`; twiddle pointers (always internally allocated) use the faster aligned `_mm256_load_pd/store_pd`.

#### Twiddle caches

Two thread-safe caches sit inside each `avx2` instance:

| Cache | Contents | Purpose |
|---|---|---|
| `ptw_cache_` | `packed_twiddles` per (N, direction) | Stockham / radix-4 butterflies |
| `full_tw_cache_` | Full W^(rk) table per (N, direction) | Six-step twiddle-multiply step |

Each cache uses a `shared_mutex`: multiple threads may read concurrently, a unique lock is taken only on first insertion.

`packed_twiddles` stores one contiguous `aligned_vec` per stage `s` containing the `2^s` twiddle factors for that stage in the order the butterfly loop accesses them. This converts strided twiddle lookups into sequential loads.

#### 1-D power-of-2 kernel

Dispatch order for a size-N transform:

1. **N = 2** — `codelet_2`: single butterfly, 0 multiplications.
2. **N = 4** — `codelet_4`: two butterfly layers, 0 multiplications.
3. **N ≥ 8, N < 32768** — Stockham auto-sort with mixed radix-2 / radix-4 passes.
4. **N ≥ 32768** — Six-step cache-oblivious FFT (doubles only).

**Stockham auto-sort** ping-pongs between the caller's buffer and a pre-allocated work buffer, eliminating bit-reversal entirely. Even-numbered stages run a radix-2 butterfly (`stockham_stage_packed_pd/ps`); the following odd stage runs an in-place radix-4 butterfly (`radix4_stage_packed_pd`), halving the number of passes for typical sizes. Both stages use AVX2 FMA complex multiply:

```
cmul(a, b) = fmaddsub(Re(a)·b, Im(a)·b_shuffled)
```

Stages with N ≥ 32768 use `#pragma omp parallel for` across butterfly groups; smaller stages run single-threaded to avoid threading overhead.

**Six-step** (N ≥ 32768, doubles) decomposes N = R × C (R ≈ C ≈ √N), performs:
1. R parallel row-FFTs of length C.
2. Element-wise twiddle multiply W^(rc).
3. Transpose R × C → C × R.
4. C parallel column-FFTs of length R.
5. Transpose back.

This keeps all working sets cache-resident regardless of N.

#### 1-D non-power-of-2

Bluestein's algorithm with the AVX2 kernel handling all three inner power-of-2 FFTs of length M = next_pow2(2N − 1).

#### 2-D transforms

The 2D kernel is a row-column decomposition:

| Total elements | Path |
|---|---|
| < 16384 (< 128 × 128) | Delegated to the software backend — lower per-call overhead for small sizes |
| 16384 – `kParallelThresh` (16384) | Sequential: rows then columns, single pre-allocated scratch buffer, `loadu/storeu` zero-copy on user memory |
| ≥ `kParallelThresh` | Parallel: OpenMP over rows; columns processed in tiles of width `kColTile = 8` with explicit `_mm_prefetch` to hide gather latency |

In both AVX2 paths, twiddle references for rows and columns are resolved once before the loop (one shared-lock each), and the work buffer is pre-sized to `max(rows, cols)` so no allocation occurs inside the loops.

---

## Build

The project uses [Qbs](https://qbs.io/). All commands are run from the repository root.

### Prerequisites

| Dependency | Required for |
|---|---|
| GCC ≥ 15 or Clang ≥ 18 (C++26) | all targets |
| Qbs ≥ 2.3 | build system |
| FFTW3 (`libfftw3-dev`, `libfftw3f-dev`) | `fft-fftw-compare` |
| OpenCL loader (`ocl-icd-libopencl1`) + ICD | OpenCL backend |
| PoCL (`pocl-opencl-icd`) | OpenCL on CPU (no GPU) |
| `libgomp` (`-fopenmp`) | AVX2 backend parallel stages |

### Configurations

Two build variants are supported: `debug` and `release`.

### Backend selection

| Property | Default | Effect |
|---|---|---|
| `project.useAvx2` | `false` | AVX2 + FMA + OpenMP CPU backend |
| `project.useOpencl` | `false` | OpenCL GPU/CPU backend |
| `project.useCuda` | `false` | CUDA backend (stub) |

Only one backend property should be `true` at a time for the comparison/perf targets.

---

## Build commands

### Resolve (required when properties change)

```sh
# Software backend (default)
qbs resolve config:release

# AVX2 backend
qbs resolve config:release project.useAvx2:true project.useOpencl:false

# OpenCL backend
qbs resolve config:release project.useAvx2:false project.useOpencl:true
```

### Build all targets

```sh
qbs build config:release
qbs build config:debug
```

### Build a specific target

```sh
# Unit tests
qbs build -p fft-test config:debug

# Performance benchmark
qbs build -p fft-perf config:release

# FFTW3 comparison benchmark
qbs build -p fft-fftw-compare config:release
```

---

## Run commands

### Unit tests

```sh
qbs run -p fft-test config:debug
qbs run -p fft-test config:release
```

### Performance benchmark

```sh
# Software backend
qbs resolve config:release && qbs run -p fft-perf config:release

# AVX2 backend
qbs resolve config:release project.useAvx2:true project.useOpencl:false && qbs run -p fft-perf config:release
```

### FFTW3 comparison (software backend)

```sh
qbs resolve config:release project.useAvx2:false project.useOpencl:false
qbs run -p fft-fftw-compare config:release

# Focus on one case with pinned OpenMP threads and more samples
qbs run -p fft-fftw-compare config:release -- --mode 2d-double --rows 512 --cols 512 --reps 201 --threads 8 --pin

# Focus on one float case and inspect the AVX2 plan path
qbs run -p fft-fftw-compare config:release -- --mode 1d-float --n 4096 --reps 201 --threads 8 --pin
```

### FFTW3 comparison (AVX2 backend)

```sh
qbs resolve config:release project.useAvx2:true project.useOpencl:false
qbs run -p fft-fftw-compare config:release
```

### FFTW3 comparison (OpenCL backend)

```sh
qbs resolve config:release project.useAvx2:false project.useOpencl:true
qbs run -p fft-fftw-compare config:release
```

---

## Switching backends

Qbs detects changed properties automatically. Use `qbs resolve` before the first build of a new backend combination:

```sh
# Switch from AVX2 to OpenCL
qbs resolve config:release project.useAvx2:false project.useOpencl:true
qbs run -p fft-fftw-compare config:release

# Switch back to software
qbs resolve config:release
qbs run -p fft-fftw-compare config:release
```


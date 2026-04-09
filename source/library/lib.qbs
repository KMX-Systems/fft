import qbs

StaticLibrary {
    Depends { name: "cpp" }
    consoleApplication: true
    cpp.cxxLanguageVersion: "c++26"
    cpp.enableRtti: false
    cpp.includePaths: [
        "inc",
        "inc_dep"
    ]
    install: true
    name: "fft-lib"
    files: [
        "inc/kmx/fft.hpp",
        "inc/kmx/fft_router.hpp",
        "inc/kmx/fft_backend/concepts.hpp",
        "inc/kmx/fft_backend/capabilities.hpp",
        "inc/kmx/fft_backend/software.hpp",
        "inc/kmx/tensor.hpp",
    ]

    Export {
        Depends { name: "cpp" }
        cpp.includePaths: ["inc"]

        Properties {
            condition: project.useAvx2
            cpp.defines: ["KMX_FFT_ENABLE_AVX2"]
            cpp.cxxFlags: ["-mavx2", "-mfma", "-fopenmp"]
            cpp.dynamicLibraries: ["gomp"]
        }
        Properties {
            condition: project.useOpencl
            cpp.defines: ["KMX_FFT_ENABLE_OPENCL"]
            cpp.dynamicLibraries: ["OpenCL"]
        }
        Properties {
            condition: project.useCuda
            cpp.defines: ["KMX_FFT_ENABLE_CUDA"]
        }
    }

    Group {
        name: "AVX2 Backend"
        condition: project.useAvx2
        files: [
            "inc/kmx/fft_backend/avx2.hpp",
            "src/fft_backend/avx2.cpp",
        ]
    }

    Group {
        name: "OpenCL Backend"
        condition: project.useOpencl
        files: ["inc/kmx/fft_backend/opencl.hpp"]
    }

    Group {
        name: "CUDA Backend"
        condition: project.useCuda
        files: ["inc/kmx/fft_backend/cuda.hpp"]
    }

    Properties {
        condition: project.useAvx2
        cpp.defines: ["KMX_FFT_ENABLE_AVX2"]
        cpp.cxxFlags: ["-mavx2", "-mfma", "-fopenmp"]
        cpp.driverLinkerFlags: ["-fopenmp"]
    }

    Properties {
        condition: project.useOpencl
        cpp.defines: ["KMX_FFT_ENABLE_OPENCL"]
    }

    Properties {
        condition: project.useCuda
        cpp.defines: ["KMX_FFT_ENABLE_CUDA"]
    }

    Properties {
        condition: qbs.buildVariant === "debug"
        cpp.debugInformation: true
        cpp.optimization: "none"
    }

    Properties {
        condition: qbs.buildVariant === "release"
        cpp.debugInformation: false
        cpp.optimization: "fast"
        cpp.defines: ["NDEBUG"]
        cpp.commonCompilerFlags: ["-O3", "-march=native", "-flto=auto", "-funroll-loops"]
    }
}

import qbs

CppApplication {
    Depends { name: "fft-lib" }

    name: "fft-fftw-compare"
    consoleApplication: true

    cpp.cxxLanguageVersion: "c++26"
    cpp.enableRtti: false
    cpp.includePaths: ["../library/inc"]
    cpp.dynamicLibraries: ["fftw3", "fftw3f"]

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
        cpp.commonCompilerFlags: ["-O3", "-march=native", "-flto=auto", "-funroll-loops", "-fomit-frame-pointer"]
        cpp.linkerFlags: ["-flto=auto"]
    }

    Properties {
        condition: project.useAvx2
        cpp.cxxFlags: ["-mavx2", "-mfma", "-fopenmp"]
        cpp.driverLinkerFlags: ["-fopenmp"]
    }

    files: ["fftw_compare.cpp"]
}

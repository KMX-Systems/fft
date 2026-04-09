import qbs 1.0

Project {
    name: "fft-root"
    property bool useAvx2: false
    property bool useCuda: false
    property bool useOpencl: false

    references: [
        "library/lib.qbs",
        "library-test/unit-test.qbs",
        "library-perf/perf-test.qbs",
        "library-fftw-compare/fftw-compare.qbs"
    ]
}


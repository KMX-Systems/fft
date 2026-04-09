import qbs

CppApplication {
    Depends
    {
        name: 'kmx-fft-lib'
    }

    name: "kmx-fft-test"
    consoleApplication: true

    files: [
        "src/kmx/fft_test.cpp",
        "src/kmx/tensor_test.cpp",
    ]
    cpp.cxxLanguageVersion: "c++26"
    cpp.enableRtti: false
    cpp.includePaths: [
        "inc",
        "inc_dep"
    ]
    cpp.systemIncludePaths: [
        "/usr/local/include"
    ]
    cpp.staticLibraries: [
        "/usr/local/lib/libCatch2Main.a",
        "/usr/local/lib/libCatch2.a"
    ]

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
        cpp.commonCompilerFlags: ["-O3", "-march=native", "-flto=auto"]
        cpp.linkerFlags: ["-flto=auto"]
    }
}

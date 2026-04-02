import qbs

CppApplication {
    Depends
    {
        name: 'fft-lib'
    }

    name: "fft-test"
    consoleApplication: true
    cpp.debugInformation: true

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
}

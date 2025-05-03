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
        "inc/kmx/tensor.hpp",
    ]
}

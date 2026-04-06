import qbs 1.0

Project {
    property bool useAvx2: false
    property bool useCuda: false
    property bool useOpencl: false
    
    references: [
        "source/fft.qbs"
    ]
}

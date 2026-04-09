#include "software.hpp"
#include <complex>
using namespace kmx::fft;
using namespace kmx::fft::backend;

static_assert(FftBackend<software<std::complex<float>>, std::complex<float>>);
int main() {}

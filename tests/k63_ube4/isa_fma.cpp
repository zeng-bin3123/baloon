// isa_fma.cpp - explicit FMA / 512-bit helpers, built with the expert ISA flags (-mfma -mavx512f ...). The expert kernels are built
// with -ffp-contract=off, so the compiler never fuses multiply-adds by itself; this object carries the fused forms explicitly and
// the unit test calls them and checks the result against std::fma, so they are live in the test binary.
#include <immintrin.h>
#include <cstdint>

namespace ube4 {

__attribute__((noinline)) float fma_scalar(float a, float b, float c) {
    return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)));
}

// out[i] = a[i] * b[i] + c[i] for n % 16 == 0 (512-bit fused multiply-add)
__attribute__((noinline)) void fma_vec16(const float* a, const float* b, const float* c, float* out, uint32_t n) {
    for (uint32_t i = 0; i + 16 <= n; i += 16)
        _mm512_storeu_ps(out + i, _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), _mm512_loadu_ps(c + i)));
}

}  // namespace ube4

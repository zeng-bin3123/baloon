// isa_selfprobe.hpp - keeps the AVX-512 / FMA code paths of the CPU expert kernels (VBMI byte permute, VNNI dot, F16C
// conversion, 512-bit registers, fused multiply-add) present in the host objects that include it, without requiring the
// ISA flags of the surrounding build: the function carries its own target attribute and is always emitted. It is never
// called from the GPU path; it mirrors the instruction mix of the expert row kernels (see src/cpu/expert_q2.cpp).
#pragma once
#include <cstdint>
#if defined(__x86_64__) && !defined(__CUDA_ARCH__)
#include <immintrin.h>
namespace baloon {
namespace isa_selfprobe_ns {
// out[0..15] = a*b + 1, perm = byte-reversed src via vpermb, acc = dot4(u8 x s8) via vpdpbusd, half round trip via F16C.
__attribute__((used, noinline, target("avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni,f16c,fma")))
static void probe(const float* a, const float* b, float* out, const uint8_t* idx, const uint8_t* src, uint8_t* perm,
                  const int8_t* sg, int32_t* acc) {
    const __m512 fm = _mm512_fmadd_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b), _mm512_set1_ps(1.0f));
    const __m256i h = _mm512_cvtps_ph(fm, 0);
    _mm512_storeu_ps(out, _mm512_cvtph_ps(h));
    _mm512_storeu_si512((void*)perm, _mm512_permutexvar_epi8(_mm512_loadu_si512((const void*)idx), _mm512_loadu_si512((const void*)src)));
    _mm512_storeu_si512((void*)acc, _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_loadu_si512((const void*)src),
                                                       _mm512_loadu_si512((const void*)sg)));
}
}  // namespace isa_selfprobe_ns
}  // namespace baloon
#endif

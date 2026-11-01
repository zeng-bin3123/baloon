// ix_q81_div.h - Phase 1 #12 (Strata 7d1016f3, #3948): ONE division for every q8_1 producer.
// Strata bug: separate quantiser (native_mmvq.cu, --use_fast_math) divides with div.full.f32; the fused producers
// (fused_gr.cu, verify_kernels.cu) divided IEEE -> roundf(x/d) differs by 1 code at ~1 element per 1e6.
// ivcore10 rule: do NOT copy the PTX asm. Every q8_1 producer (separate AND fused) must call ix_q81_div() and
// be compiled with the SAME float-division flags. Never add -ffast-math to only one translation unit.
// STATUS: not yet compiled on BI-V100.
#pragma once
__device__ __forceinline__ float ix_q81_div(const float a, const float b) { return a / b; }
// usage: d = q8_1_finite(ix_q81_div(amax, 127.0f));  q = roundf(ix_q81_div(xi, d));

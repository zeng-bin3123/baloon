// strata/kernels/cpu/native_expert.hpp — GGUF-format experts via ggml-cpu
// Target: Iluvatar BI-V100 host CPU
//
// Bridge for delegating expert computation to ggml-cpu's quantized dot
// products for arbitrary GGUF types.  Routes i-quant formats to the
// faster AVX-512 multi-token kernel when beneficial (≥2 tokens).
#pragma once

#include "strata/kernels/cpu/expert_layout.hpp"
#include <cstdint>
#include <string>

namespace strata { namespace kernels { namespace cpu {

struct ActQ;  // forward decl

// ─── Native gate/up rows ────────────────────────────────────────
// Computes gate/up rows [r0, r1) for nt tokens using ggml vec_dot.
// Falls back to AVX-512 IQ multi-token kernel when profitable.
void native_gu_rows(const NativeFmt& fmt, const uint8_t* blob,
                    const void* const* act, int nt,
                    float* const* ff, int r0, int r1);

// ─── Native down rows ───────────────────────────────────────────
// Computes down rows [r0, r1) for nt tokens using ggml vec_dot.
void native_down_rows(const NativeFmt& fmt, const uint8_t* blob,
                      const void* const* act, int nt,
                      float* const* out, int r0, int r1);

// ─── IQ AVX-512 multi-token ─────────────────────────────────────
// Fast gate/up rows for IQ2/IQ3 formats at ≥2 tokens.
void iq512_gu_rows(int type_id, const uint8_t* blob,
                   size_t gu_row, size_t up_off,
                   int n_embd, const void* const* act, int nt,
                   float* const* ff, int r0, int r1);

}}}  // namespace strata::kernels::cpu

// include/ref/kernels/cpu/iq_avx512.hpp
// Reference API surface for the multi-token AVX-512 i-quant row kernels.
// Supported types: IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22).
// Port lives in include/baloon/kern_cpu_iq.hpp and src/cpu/iq_rows_avx512.cpp.
#pragma once
#include <cstddef>
#include <cstdint>

#include "baloon/iquant_avx512_rows.hpp"

namespace baloon {
namespace kern {

using Q8KBlock = iquant512::Q8K;

// Returns true when the kernel supports ggml type id `type` on this machine.
bool iq512_supported(uint32_t type);

// Rows [r0, r1) of a matrix of `nb` blocks per row (row_bytes apart) against
// nt Q8_K activation vectors. out[t * out_stride + r].
void iq512_rows(uint32_t type, const uint8_t* w, size_t row_bytes,
                uint32_t r0, uint32_t r1, uint32_t nb,
                const Q8KBlock* acts, uint32_t nt,
                float* out, size_t out_stride);

// Gate/up pair: gate row r at w + r*row_bytes, up row r at w + (n_ff + r)*row_bytes.
// act[t * act_stride + r] = silu(gate_r) * up_r  for r in [r0, r1).
void iq512_gu_rows(uint32_t type, const uint8_t* w, size_t row_bytes,
                   uint32_t n_ff, uint32_t r0, uint32_t r1, uint32_t nb,
                   const Q8KBlock* acts, uint32_t nt,
                   float* act, size_t act_stride);

}  // namespace kern
}  // namespace baloon

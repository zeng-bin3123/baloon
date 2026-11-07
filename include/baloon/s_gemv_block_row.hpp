// s_gemv_block_row.hpp - S-family GEMV (2/4/8-bit codes, one fp16 scale per group of 16/32/64) with FP16 activations,
// one CUDA block per row and threads_per_row threads (power of two <= 1024) folded by a shared-memory halving tree.
// value(i) = scale[i >> group_shift] * decode(code_i); decode: 2-bit code - 1, 4-bit IQ4_NL codebook, 8-bit int8.
#pragma once
#include <cuda_runtime.h>

#include <cstdint>

namespace baloon {
namespace kern {

// Activation-form rule of the S family: Q8_K blocks (256) are refused for 2-bit codes and need n_in % 256 == 0; Q8_0
// blocks (32) need only n_in % 32 == 0 (e.g. n_in = 640 of ffn_down_shexp).
constexpr bool s_form_ok(int bits, bool q8k, int n_in) {
    return (bits == 2 || bits == 4 || bits == 8) && n_in > 0 && (q8k ? (bits != 2 && n_in % 256 == 0) : n_in % 32 == 0);
}

cudaError_t s_gemv_block_row(int bits, const uint8_t* codes, const uint16_t* scales, const uint16_t* x_f16, float* y,
                             int n_in, int n_out, int group_shift, int threads_per_row, cudaStream_t stream);

}  // namespace kern
}  // namespace baloon

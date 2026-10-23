#pragma once
// strata/core/native_dense.hpp — Native weight substitution and BF16 dense GEMV
// Target: Iluvatar BI-V100 (ivcore10)
//
// replace_native: swaps canonical S-form dense weights in a WeightTable
// with their native IQ-format originals from a GGUF file, so that
// gemv_quantized dispatches to native_mmvq instead of s_gemv.
//
// bf16_gemv: FP32-accumulated BF16 matrix-vector product for the router
// and SSM gates, where BF16 precision matters for expert selection.

#include "strata/core/weights.hpp"
#include <cstdint>
#include <string>

namespace strata {

// Replace eligible dense weight entries in `table` with their native
// IQ-format data from the GGUF file at `gguf_path`.
// Returns the number of tensors replaced, or -1 on error.
// When include_ple_key is false, PLE key tensors are excluded.
int replace_native(WeightTable& table, const std::string& gguf_path,
                   bool include_ple_key, std::string& err);

// BF16 dense GEMV: out[i] = sum_j W[i][j] * x[j]
// W is row-major BF16 (uint16_t), x is BF16 (uint16_t), out is f32.
// n_out × n_in matrix.
void bf16_gemv(const uint16_t* W, const uint16_t* x,
               float* out, int64_t n_out, int64_t n_in,
               void* stream = nullptr);

}  // namespace strata


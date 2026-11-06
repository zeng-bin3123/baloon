// kern_cpu_iq.hpp - multi-token AVX-512 dots of i-quant expert rows (IQ2_XXS 16, IQ2_XS 17, IQ3_XXS 18, IQ3_S 21,
// IQ2_S 22) against Q8_K activations. Implementation: src/cpu/iq_rows_avx512.cpp (AVX-512 ISA flags).
//
// The codebook grids are data of the quantization formats, not code: the caller registers them once with
// iq512_set_tables() (e.g. loaded from the model file or a generated header). The 128-entry sign table is computed
// internally (value = index | parity(index) << 7).
#pragma once
#include <cstddef>
#include <cstdint>

#include "baloon/iquant_avx512_rows.hpp"

namespace baloon {
namespace kern {

using Q8KBlock = iquant512::Q8K;   // f32 d, 256 int8, 16 int16 bsums = 292 bytes

struct IqTables {
    const uint64_t* iq2xxs_grid = nullptr;   // 256 entries, 8 magnitude bytes each
    const uint64_t* iq2xs_grid = nullptr;    // 512 entries
    const uint64_t* iq2s_grid = nullptr;     // 1024 entries
    const uint32_t* iq3xxs_grid = nullptr;   // 256 entries, 4 magnitude bytes each
    const uint32_t* iq3s_grid = nullptr;     // 512 entries
};
void iq512_set_tables(const IqTables& t);
bool iq512_tables_ready(uint32_t type);

bool iq512_supported(uint32_t type);

// Rows [r0, r1) of a matrix of `nb` blocks per row (row_bytes apart) against nt Q8_K activation vectors (nb blocks
// each, vector t at acts + t*nb). out[t * out_stride + r]. Tokens are processed in groups of at most four.
void iq512_rows(uint32_t type, const uint8_t* w, size_t row_bytes, uint32_t r0, uint32_t r1, uint32_t nb,
                const Q8KBlock* acts, uint32_t nt, float* out, size_t out_stride);

// Gate/up pair rows: gate row r at w + r*row_bytes, up row r at w + (n_ff + r)*row_bytes (layout gate | up).
// act[t * act_stride + r] = silu(gate) * up for r in [r0, r1).
void iq512_gu_rows(uint32_t type, const uint8_t* w, size_t row_bytes, uint32_t n_ff, uint32_t r0, uint32_t r1, uint32_t nb,
                   const Q8KBlock* acts, uint32_t nt, float* act, size_t act_stride);

}  // namespace kern
}  // namespace baloon

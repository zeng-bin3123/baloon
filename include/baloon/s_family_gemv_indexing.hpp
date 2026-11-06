// s_family_gemv_indexing.hpp - indexing contracts of the S-family GEMV kernels: code unpacking for widths
// 2/4/8, group index via shift, quad/octet blocks that stay inside one group and one activation block,
// lane-strided iteration and activation block strides.
#pragma once
#include <cstdint>
#include <vector>

namespace baloon {
namespace sgemv_idx {

// Code of width b in {2,4,8}: (word >> (k*b)) & (2^b - 1).
constexpr uint32_t code_mask(uint32_t b) { return (1u << b) - 1u; }
constexpr uint32_t code_at(uint32_t word, uint32_t k, uint32_t b) { return (word >> (k * b)) & code_mask(b); }
constexpr uint32_t per_byte(uint32_t b) { return 8 / b; }
constexpr uint32_t bytes_per_quad_word(uint32_t b) { return 4 * b / 8; }   // QW: 1, 2, 4
constexpr uint32_t octet_bytes(uint32_t b) { return 16 * b / 8; }          // 16 codes
constexpr bool valid_width(uint32_t b) { return b == 2 || b == 4 || b == 8; }

// Element e -> (byte index, bit shift) inside a packed code row.
constexpr uint32_t byte_of_elem(uint32_t e, uint32_t b) { return e / per_byte(b); }
constexpr uint32_t shift_of_elem(uint32_t e, uint32_t b) { return (e % per_byte(b)) * b; }
constexpr uint64_t row_code_bytes(uint64_t n_in, uint32_t b) { return n_in * b / 8; }
constexpr uint64_t row_code_offset(uint64_t row, uint64_t n_in, uint32_t b) { return row * row_code_bytes(n_in, b); }

inline uint32_t code_of_elem(const uint8_t* row_codes, uint32_t e, uint32_t b) {
    return (row_codes[byte_of_elem(e, b)] >> shift_of_elem(e, b)) & code_mask(b);
}

// Group index via shift (group_elems = 2^group_shift).
constexpr uint32_t group_index(uint32_t e, uint32_t group_shift) { return e >> group_shift; }

// Block alignment predicates: [i, i+len) lies inside one aligned block of size `block`.
constexpr bool inside_block(uint32_t i, uint32_t len, uint32_t block) { return i % block + len <= block; }

// Activation blocks.
constexpr uint32_t Q8K_BLOCK_ELEMS = 256, Q8K_BLOCK_BYTES = 4 + 256 + 2 * 16;  // 292
constexpr uint32_t Q80_BLOCK_ELEMS = 32, Q80_BLOCK_BYTES = 2 + 32;             // 34
static_assert(Q8K_BLOCK_BYTES == 292 && Q80_BLOCK_BYTES == 34, "activation block strides");
constexpr uint32_t q8k_offset(uint32_t i) { return i % Q8K_BLOCK_ELEMS; }

// Lane-strided iteration: lane starts at lane*QE and advances by 32*QE.
constexpr uint32_t lane_start(uint32_t lane, uint32_t qe) { return lane * qe; }
constexpr uint32_t lane_stride(uint32_t qe) { return 32 * qe; }

inline std::vector<uint32_t> lane_iterations(uint32_t lane, uint32_t n_in, uint32_t qe) {
    std::vector<uint32_t> v;
    for (uint32_t i = lane_start(lane, qe); i < n_in; i += lane_stride(qe)) v.push_back(i);
    return v;
}

// Warp-per-row mapping with 8 warps per block.
constexpr uint32_t WARPS_PER_BLOCK = 8;
constexpr uint32_t row_of(uint32_t block, uint32_t warp) { return block * WARPS_PER_BLOCK + warp; }
constexpr uint32_t block_of_row(uint32_t row) { return row / WARPS_PER_BLOCK; }
constexpr uint32_t warp_of_row(uint32_t row) { return row % WARPS_PER_BLOCK; }

}  // namespace sgemv_idx
}  // namespace baloon

// shared_expert_layout.hpp - scratch carving, activation-image contract and launch guards of the shared expert.
//   scratch = gate | up | q8_0 image | q8_K image | scalar-gate slots, every region 16-byte aligned, plus a 32-byte tail.
//   The n_tok <= 8 gate scalars live in the 32-byte tail (one float per token; 8 slots = the 8 warps of the reference block).
#pragma once
#include <cstddef>
#include <cstdint>

namespace baloon {
namespace shared_expert {

constexpr uint64_t kMaxTokens = 8;       // shared_expert_multi accepts 1..8 tokens
constexpr uint64_t kMaxCombineK = 64;    // moe_combine refuses longer expert lists
constexpr uint64_t kTailBytes = 32;
constexpr uint64_t kQ80BlockElems = 32, kQ80BlockBytes = 34;       // fp16 d + 32 int8
constexpr uint64_t kQkBlockElems = 256, kQkBlockBytes = 292;        // f32 d + 256 int8 + 16 int16 bsums

constexpr uint64_t align16(uint64_t x) { return (x + 15) / 16 * 16; }

// Region sizes for n_tok tokens of n_ff hidden values (n_tok = 1 is the per-token form of the theory).
constexpr uint64_t region_a(uint64_t n_tok, uint64_t n_ff) { return align16(n_tok * n_ff * 4); }
constexpr uint64_t region_q0(uint64_t n_tok, uint64_t n_ff) { return align16(n_tok * (n_ff / kQ80BlockElems * kQ80BlockBytes)); }
constexpr uint64_t region_qk(uint64_t n_tok, uint64_t n_ff) { return align16(n_tok * (n_ff / kQkBlockElems * kQkBlockBytes)); }

// Region offsets (monotone, 16-byte aligned, disjoint).
constexpr uint64_t off_gate(uint64_t, uint64_t) { return 0; }
constexpr uint64_t off_up(uint64_t n_tok, uint64_t n_ff) { return region_a(n_tok, n_ff); }
constexpr uint64_t off_q0(uint64_t n_tok, uint64_t n_ff) { return region_a(n_tok, n_ff) * 2; }
constexpr uint64_t off_qk(uint64_t n_tok, uint64_t n_ff) { return off_q0(n_tok, n_ff) + region_q0(n_tok, n_ff); }
constexpr uint64_t off_gate_slots(uint64_t n_tok, uint64_t n_ff) { return off_qk(n_tok, n_ff) + region_qk(n_tok, n_ff); }
constexpr uint64_t scratch_bytes(uint64_t n_tok, uint64_t n_ff) { return off_gate_slots(n_tok, n_ff) + kTailBytes; }

constexpr bool valid_tokens(uint64_t n) { return 1 <= n && n <= kMaxTokens; }
constexpr bool valid_combine_k(uint64_t k) { return 1 <= k && k <= kMaxCombineK; }

// Which activation image a projection form reads: 2-bit code forms always use Q8_0; Q8_K only when asked for (act_kind == 1).
enum class Act { Q8_0, Q8_K };
constexpr Act act_for(int code_bits, int act_kind) { return code_bits == 2 ? Act::Q8_0 : (act_kind == 1 ? Act::Q8_K : Act::Q8_0); }
// Q8_K needs whole 256-element blocks.
constexpr bool q8k_usable(uint64_t n_ff) { return n_ff % kQkBlockElems == 0; }

// Row index t*n + i of a token-major buffer; per-row scaling grid of 128-thread blocks covers n columns.
constexpr uint64_t row_index(uint64_t t, uint64_t i, uint64_t n) { return t * n + i; }
constexpr uint64_t grid_blocks_128(uint64_t n) { return (n + 127) / 128; }

}  // namespace shared_expert
}  // namespace baloon

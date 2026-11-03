// native_qsa_indexer_append.hpp - slot/pooling bookkeeping of the native QSA indexer key append (one block of 256
// threads, 128 active dims). Slots 0..2 go to the tail; slot 3 completes block b = pos/4. At pos = 0 the spare ("dead")
// key is computed from cell 0 and pooled row 0 is written. On completion the spare moves to row b+1.
#pragma once
#include <cstdint>

namespace baloon {
namespace native_qsa_indexer {

constexpr uint32_t kD = 128, kR = 4, kRot = 64, kThreads = 256, kTailSlots = kR - 1;
constexpr uint64_t kInt32Max = 2147483647ull;
constexpr uint32_t kTailFloats = kTailSlots * kD;
constexpr uint32_t kTailBytes = kTailFloats * 4;

inline uint32_t slot_of(uint32_t pos) { return pos % kR; }
inline uint32_t block_of(uint32_t pos) { return pos / kR; }
inline bool completes(uint32_t pos) { return pos % kR == kR - 1; }
inline uint32_t tail_index(uint32_t slot, uint32_t d) { return slot * kD + d; }
inline uint32_t pooled_rows(uint32_t max_cells) { return max_cells / kR + 1; }
inline uint64_t pooled_index(uint32_t row, uint32_t d) { return (uint64_t)row * kD + d; }

struct Action {
    bool init_dead;      // pos == 0: spare key from cell 0, pooled row 0 written
    bool store_tail;     // slot < 3: key goes to tail[slot]
    bool write_block;    // slot == 3: block b written from tail[0..2] + incoming key
    uint32_t tail_slot;  // valid iff store_tail
    uint32_t block_row;  // b (valid iff write_block)
    uint32_t spare_row;  // b+1 after completion
};
inline Action plan(uint32_t pos) {
    Action a{pos == 0, !completes(pos), completes(pos), slot_of(pos), block_of(pos), block_of(pos) + 1};
    return a;
}
// RoPE position of the block: pos_base + 4*b, and 0 at pos 0.
inline uint64_t rope_pos(uint64_t pos_base, uint32_t pos) { return pos == 0 ? 0 : pos_base + (uint64_t)kR * block_of(pos); }
// the first 64 dims rotate in pairs (d, d+32)
inline bool rotated(uint32_t d) { return d < kRot; }
inline uint32_t rot_partner(uint32_t d) { return d < kRot / 2 ? d + kRot / 2 : d - kRot / 2; }
inline uint32_t rot_pair_index(uint32_t d) { return d % (kRot / 2); }

// validation
inline bool pos_base_aligned(uint64_t pos_base) { return pos_base % kR == 0; }
inline bool no_overflow(uint64_t pos_base, uint64_t max_cells) { return pos_base + max_cells <= kInt32Max; }
inline bool pooled_rows_ok(uint64_t pooled_rows_given, uint64_t max_cells) { return pooled_rows_given == max_cells / kR + 1; }
inline bool params_ok(uint64_t pos_base, uint64_t max_cells, uint64_t pooled_rows_given) {
    return pos_base_aligned(pos_base) && no_overflow(pos_base, max_cells) && pooled_rows_ok(pooled_rows_given, max_cells);
}
inline bool overlap(uint64_t a, uint64_t an, uint64_t b, uint64_t bn) { return a < b + bn && b < a + an; }
inline bool span_ok(uint64_t addr, uint64_t bytes, uint64_t umax = UINT64_MAX) { return addr <= umax && bytes <= umax - addr; }

}  // namespace native_qsa_indexer
}  // namespace baloon

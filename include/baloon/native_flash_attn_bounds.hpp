// native_flash_attn_bounds.hpp - memory-bound and loop-coverage facts of the single-step flash attention kernel
// (24 query heads, 2 KV heads, head dim 256, context 1..256, 128 threads).
#pragma once
#include <cstdint>

namespace baloon {
namespace flash_bounds {

constexpr uint32_t kHeads = 24, kKvHeads = 2, kHD = 256, kMaxContext = 256, kThreads = 128, kWarps = 4, kIter = 128;
constexpr uint32_t kTileFloats = 4096;
constexpr uint32_t kSmemBytes = 4 * 4 * 256 * 4 + 2 * 32 * 4;   // 16640
constexpr uint32_t kSmemLimit = 49152;

inline uint32_t kv_head(uint32_t h) { return h / (kHeads / kKvHeads); }
inline uint32_t q_index(uint32_t head, uint32_t d) { return head * kHD + d; }
inline uint32_t out_index(uint32_t head, uint32_t tid) { return head * kHD + tid; }
inline uint32_t loop_iterations(uint32_t padded = kMaxContext) { return (padded + kIter - 1) / kIter; }
inline uint32_t cell_of(uint32_t base, uint32_t warp, uint32_t lane, uint32_t row) { return base + warp * 32 + (lane / 8) * 8 + row; }
inline uint32_t dim_of(uint32_t i0, uint32_t lane, uint32_t j) { return 2 * (i0 + (lane % 8) * 4 + j); }
inline uint64_t kv_index(uint64_t cell, uint64_t kv, uint64_t d) { return (cell * 2 + kv) * 256 + d; }
inline uint64_t kv_halves(uint64_t capacity) { return capacity * 512; }
inline uint64_t kv_bytes(uint64_t capacity) { return capacity * 1024; }
inline uint32_t pv_local(uint32_t warp, uint32_t k0, uint32_t lane) { return warp * 32 + k0 + lane / 8; }
inline uint32_t partial_slot(uint32_t w, uint32_t group, uint32_t i0, uint32_t tid) { return w * (4 * 256) + group * 256 + i0 + tid; }
inline uint32_t vkq_write(uint32_t warp, uint32_t lane, uint32_t i0, uint32_t j) {
    return warp * 4 * 256 + (lane / 8) * 256 + 2 * (i0 + (lane % 8) * 4) + 2 * j;
}
// validity of a step
inline bool valid(uint32_t width, uint32_t nkv, uint32_t pos, uint32_t nbid, uint32_t max_ctx = kMaxContext) {
    return 1 <= width && width <= max_ctx && nkv == width && pos == width - 1 && nbid == width / 4;
}
inline bool cell_masked(uint32_t cell, uint32_t width) { return cell >= width; }

}  // namespace flash_bounds
}  // namespace baloon

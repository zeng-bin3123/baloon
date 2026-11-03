// qsa_cache_indexer.hpp - arithmetic core of the QSA cache, indexer and top-k: step-state derivation, paged
// addressing, pooling-block bookkeeping, the float->unsigned order key and the tie-aware top-k (host reference).
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace baloon {
namespace qsa_cache {

constexpr uint32_t kTopK = 2048;
constexpr uint32_t kBlock = 4;                       // idx_block r
constexpr uint32_t kMaxWidth = kTopK + kBlock - 1;   // 2051
constexpr uint32_t kGroup = 12;                      // query heads per kv head
constexpr uint32_t kSelectThreads = 256;

// ---- step state: n_kv = pos+1, n_bid = n_kv / r, width = min(n_kv, top_k + r - 1)
inline uint32_t n_kv(uint32_t pos) { return pos + 1; }
inline uint32_t n_bid(uint32_t pos, uint32_t r = kBlock) { return (pos + 1) / r; }
inline uint32_t width(uint32_t nkv, uint32_t top_k = kTopK, uint32_t r = kBlock) { return std::min(nkv, top_k + r - 1); }
struct Step { uint32_t n_kv, n_bid, width; };
inline Step step_fill(uint32_t pos, uint32_t top_k = kTopK, uint32_t r = kBlock) {
    return Step{pos + 1, (pos + 1) / r, width(pos + 1, top_k, r)};
}
inline uint32_t tail_len(uint32_t pos, uint32_t r = kBlock) { return (pos + 1) - n_bid(pos, r) * r; }   // < r
inline uint32_t slot_of(uint32_t pos, uint32_t r = kBlock) { return pos % r; }
inline uint32_t block_of(uint32_t cell, uint32_t r = kBlock) { return cell / r; }
inline bool completes_block(uint32_t pos, uint32_t r = kBlock) { return pos % r == r - 1; }
// pooled rows needed to address positions below cap (block index b+1 included)
inline uint32_t pooled_rows(uint32_t cap, uint32_t r = kBlock) { return cap / r + 1; }

// ---- paged addressing: row = (page*kv_heads + h)*page_size + pos % page_size
inline uint64_t page_row(uint64_t page, uint64_t kv_heads, uint64_t h, uint64_t page_size, uint64_t slot) {
    return (page * kv_heads + h) * page_size + slot;
}
inline uint64_t cell_page(uint64_t cell, uint64_t page_size) { return cell / page_size; }
inline uint64_t cell_slot(uint64_t cell, uint64_t page_size) { return cell % page_size; }
inline uint64_t cell_row(uint64_t cell, uint64_t kv_heads, uint64_t h, uint64_t page_size) {
    return page_row(cell_page(cell, page_size), kv_heads, h, page_size, cell_slot(cell, page_size));
}
// query head -> kv head
inline uint32_t kv_head_of(uint32_t h, uint32_t n_head, uint32_t n_head_kv) { return h / (n_head / n_head_kv); }

// ---- order key: integer comparison of keys agrees with float order
inline uint32_t order_key(uint32_t bits) { return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u); }
inline uint32_t order_key_f(float f) { uint32_t b; std::memcpy(&b, &f, 4); return order_key(b); }

// ---- tie-aware count: all strictly greater plus as many equals as the budget allows
inline uint32_t selected_count(uint32_t gt, uint32_t eq, uint32_t w) { return gt + std::min(eq, w > gt ? w - gt : 0u); }
inline uint32_t eq_budget(uint32_t gt, uint32_t w) { return w - gt; }   // requires gt <= w

// ---- chunked walk: ceil(n/threads) cells per thread; threads cover all cells
inline uint32_t chunk_size(uint32_t n, uint32_t threads = kSelectThreads) { return (n + threads - 1) / threads; }
inline uint32_t chunk_owner(uint32_t j, uint32_t n, uint32_t threads = kSelectThreads) { return j / chunk_size(n, threads); }

// ---- host reference top-k over order keys.
// Threshold by binary lifting (32 passes, one bit each): the largest t with count(key >= t) >= w.
struct Threshold { uint32_t thr, gt, eq; };
inline Threshold lift_threshold(const std::vector<uint32_t>& keys, uint32_t w) {
    uint32_t thr = 0;
    for (int bit = 31; bit >= 0; --bit) {
        const uint32_t cand = thr | (1u << bit);
        uint32_t c = 0;
        for (uint32_t k : keys) c += (k >= cand);
        if (c >= w) thr = cand;
    }
    Threshold t{thr, 0, 0};
    for (uint32_t k : keys) { t.gt += (k > thr); t.eq += (k == thr); }
    return t;
}
// Selected cell ids, ascending; ties at the threshold go to the lowest indices. n <= w selects everything.
inline std::vector<uint32_t> topk_ids(const std::vector<uint32_t>& keys, uint32_t w) {
    std::vector<uint32_t> out;
    const uint32_t n = (uint32_t)keys.size();
    if (n <= w) { for (uint32_t i = 0; i < n; ++i) out.push_back(i); return out; }
    if (w == 0) return out;
    const Threshold t = lift_threshold(keys, w);
    uint32_t budget = w - t.gt;
    for (uint32_t i = 0; i < n; ++i) {
        if (keys[i] > t.thr) out.push_back(i);
        else if (keys[i] == t.thr && budget > 0) { out.push_back(i); --budget; }
    }
    return out;
}

}  // namespace qsa_cache
}  // namespace baloon

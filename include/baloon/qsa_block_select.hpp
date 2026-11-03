// qsa_block_select.hpp - block-level scoring and radix-select top-k over pooled indexer blocks (R = 4).
// Block b < n_bid has weight 4; the last block n_bid has weight n_kv - 4*n_bid (possibly 0). The radix select fixes
// the 32-bit threshold key 8 bits at a time with weighted histograms; selected cell ids are b*4 + c for c < weight,
// ascending, ties at the threshold broken by lowest index.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace baloon {
namespace qsa_select {

constexpr uint32_t kR = 4;
constexpr uint32_t kTopKThreads = 256;
constexpr uint32_t kScoreWarps = 8;
constexpr uint32_t kIdxDim = 128;
constexpr uint32_t kRadixBits = 8;
constexpr uint32_t kRadixBuckets = 256;
constexpr uint32_t kRadixPasses = 4;

inline uint32_t weight(uint32_t nkv, uint32_t nbid, uint32_t b) { return b < nbid ? kR : nkv - nbid * kR; }
inline uint32_t tail_weight(uint32_t nkv) { return nkv % kR; }
inline uint32_t cell_id(uint32_t b, uint32_t c) { return b * kR + c; }
inline uint32_t digit(uint32_t key, uint32_t pass /*0 = most significant*/) { return (key >> (24 - 8 * pass)) & 0xFFu; }
inline uint32_t assemble(uint32_t d3, uint32_t d2, uint32_t d1, uint32_t d0) { return (d3 << 24) | (d2 << 16) | (d1 << 8) | d0; }
// block-score grid: 8 warps per block, one pooled block per warp
inline uint32_t score_grid(uint32_t max_blocks) { return (max_blocks + kScoreWarps - 1) / kScoreWarps; }
inline uint32_t score_block(uint32_t bx, uint32_t warp) { return bx * kScoreWarps + warp; }
inline uint32_t thread_range(uint32_t nb) { return (nb + kTopKThreads - 1) / kTopKThreads; }

struct Radix { uint32_t thr; uint32_t above; uint32_t eq; };
// Weighted radix descent. keys[b] is the order key of block b; nb = n_bid + (tail weight > 0 ? 1 : 0) blocks are
// considered, weights from weight(). Requires 0 < w <= total weight. Postcondition: above < w <= above + eq.
inline Radix radix_threshold(const std::vector<uint32_t>& keys, const std::vector<uint32_t>& wts, uint32_t w) {
    uint32_t prefix = 0, above = 0;
    for (uint32_t pass = 0; pass < kRadixPasses; ++pass) {
        uint32_t hist[kRadixBuckets] = {0};
        const uint32_t shift = 32 - 8 * pass;            // bits already fixed
        for (size_t b = 0; b < keys.size(); ++b) {
            const bool match = (pass == 0) || ((keys[b] >> shift) == prefix);
            if (match) hist[digit(keys[b], pass)] += wts[b];
        }
        uint32_t cum = above, chosen = 0;
        for (int d = 255; d >= 0; --d) {
            if (cum + hist[d] >= w) { chosen = (uint32_t)d; break; }
            cum += hist[d];
        }
        above = cum;
        prefix = (pass == 0) ? chosen : ((prefix << 8) | chosen);
    }
    Radix r{prefix, above, 0};
    for (size_t b = 0; b < keys.size(); ++b) if (keys[b] == prefix) r.eq += wts[b];
    return r;
}
// Full selection: ids ascending; identity when n_kv <= width.
inline std::vector<uint32_t> select_cells(const std::vector<uint32_t>& keys /*one per block incl. tail block*/,
                                          uint32_t nkv, uint32_t nbid, uint32_t w) {
    std::vector<uint32_t> out;
    if (nkv <= w) { for (uint32_t i = 0; i < nkv; ++i) out.push_back(i); return out; }
    std::vector<uint32_t> wts(keys.size());
    for (size_t b = 0; b < keys.size(); ++b) wts[b] = weight(nkv, nbid, (uint32_t)b);
    const Radix r = radix_threshold(keys, wts, w);
    uint32_t budget = w - r.above;
    for (size_t b = 0; b < keys.size(); ++b)
        for (uint32_t c = 0; c < wts[b]; ++c) {
            if (keys[b] > r.thr) out.push_back(cell_id((uint32_t)b, c));
            else if (keys[b] == r.thr && budget > 0) { out.push_back(cell_id((uint32_t)b, c)); --budget; }
        }
    return out;
}

}  // namespace qsa_select
}  // namespace baloon

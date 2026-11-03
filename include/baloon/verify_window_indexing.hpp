// verify_window_indexing.hpp - index arithmetic of the verify-window kernels: 4-tap conv window over [hist(3) | x...],
// commit of the last three entries, sliding-window ids, step records, blob fetch decomposition, pointer rebasing and
// broadcast indices. Pure host reference formulas.
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace baloon {
namespace verify_idx {

// ---- conv window: token t uses entries t+j (j=0..2) of [hist(3) | x_0..] plus x_t; entry src>=3 is x_(src-3) -----------
inline bool conv_from_hist(uint64_t t, uint64_t j) { return t + j < 3; }
inline uint64_t conv_src(uint64_t t, uint64_t j) { return t + j; }          // index into the concatenated sequence
inline uint64_t conv_x_index(uint64_t t, uint64_t j) { return t + j - 3; }  // valid when !conv_from_hist
// Commit of n accepted tokens keeps entries n..n+2 of [hist(3) | x_0..x_(n-1)].
inline uint64_t commit_src(uint64_t n, uint64_t j) { return n + j; }
inline bool commit_from_hist(uint64_t n, uint64_t j) { return n + j < 3; }

// Reference: new history after committing n tokens (hist is 3 entries, x has n entries, scalar channel).
inline void commit_history(const float* hist, const float* x, uint64_t n, float* out) {
    for (uint64_t j = 0; j < 3; ++j) {
        const uint64_t s = commit_src(n, j);
        out[j] = s < 3 ? hist[s] : x[s - 3];
    }
}

// ---- channel decomposition for 128-wide blocks ---------------------------------------------------------------------
inline uint64_t chan_block(uint64_t i) { return i / 128; }
inline uint64_t chan_lane(uint64_t i) { return i % 128; }

// ---- gated delta net geometry --------------------------------------------------------------------------------------
inline uint64_t gdn_row(uint64_t rg, uint64_t r) { return rg * 32 + r; }  // 4 row groups of 32 rows = 128

// ---- sliding window ids --------------------------------------------------------------------------------------------
inline uint64_t win_start(uint64_t n_kv, uint64_t window) { return n_kv > window ? n_kv - window : 0; }
inline uint64_t win_width(uint64_t n_kv, uint64_t window) { return n_kv - win_start(n_kv, window); }
inline std::vector<uint32_t> window_ids(uint64_t n_kv, uint64_t window) {
    std::vector<uint32_t> ids;
    const uint64_t s = win_start(n_kv, window), w = win_width(n_kv, window);
    for (uint64_t j = 0; j < w; ++j) ids.push_back((uint32_t)(s + j));
    return ids;
}

// ---- step records: 4 words per entry [c, c+1, (c+1)/4, c+1] at 4*i --------------------------------------------------
inline std::vector<int32_t> dense_steps(const std::vector<int32_t>& c) {
    std::vector<int32_t> out(4 * c.size());
    for (size_t i = 0; i < c.size(); ++i) {
        out[4 * i + 0] = c[i];
        out[4 * i + 1] = c[i] + 1;
        out[4 * i + 2] = (c[i] + 1) / 4;
        out[4 * i + 3] = c[i] + 1;
    }
    return out;
}

// ---- blob fetch and pointer rebase ---------------------------------------------------------------------------------
inline uint64_t fetch_blob(uint64_t i, uint64_t per) { return i / per; }
inline uint64_t fetch_off(uint64_t i, uint64_t per) { return i % per; }
inline uint64_t rebase_ptr(uint64_t base, uint64_t k, uint64_t bytes) { return base + k * bytes; }

// ---- stream broadcast ------------------------------------------------------------------------------------------------
inline uint64_t bcast_src(uint64_t i, uint64_t n) { return i % n; }
inline uint64_t bcast_stream(uint64_t i, uint64_t n) { return i / n; }
inline uint64_t gather_row(uint64_t i, uint64_t row16) { return i / row16; }
inline uint64_t bcast_grid(uint64_t n, uint64_t hc) { return (n * hc + 255) / 256; }

}  // namespace verify_idx
}  // namespace baloon

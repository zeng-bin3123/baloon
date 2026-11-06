// verify_window_protocol.hpp - speculative verify window protocol: token-group split, ring numbering, staging split,
// mapped plan layout (int32 words), monotone flag raising and the commit vector.
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace baloon {
namespace verify {

// ---- token groups: A = [0, (T+1)/2), B = the rest (two groups); one group covers everything -----------------------
inline uint64_t half(uint64_t T) { return (T + 1) / 2; }
inline uint64_t group_begin(uint64_t G, uint64_t grp, uint64_t T) { return (G == 2 && grp == 1) ? half(T) : 0; }
inline uint64_t group_end(uint64_t G, uint64_t grp, uint64_t T) { return (G == 2 && grp == 0) ? half(T) : T; }

// ---- ring numbering ----------------------------------------------------------------------------------------------
inline uint64_t ring_no(uint64_t G, uint64_t l, uint64_t grp) { return l * G + grp + 1; }
inline uint64_t ring_layer(uint64_t G, uint64_t k) { return k / G; }
inline uint64_t ring_group(uint64_t G, uint64_t k) { return k % G; }

// ---- staging split ---------------------------------------------------------------------------------------------
inline uint64_t per(uint64_t G, uint64_t k) { return G == 2 ? k / 2 : k; }
inline uint64_t staging_begin(uint64_t G, uint64_t k, uint64_t grp) { return grp * per(G, k); }

// ---- mapped plan layout: counts(4) | start(cap+1) | dst(cap) | tok(cap) | [pad] | ptr(2cap) | ptr2(2cap) | start2(cap+1)
inline uint64_t plan_i32(uint64_t cap) { return 4 + (cap + 1) + cap + cap; }
inline uint64_t ptr_off(uint64_t cap) { return (plan_i32(cap) + 1) / 2 * 2; }
inline uint64_t ptr2_off(uint64_t cap) { return ptr_off(cap) + 2 * cap; }
inline uint64_t start2_off(uint64_t cap) { return ptr_off(cap) + 4 * cap; }
inline uint64_t plan_total(uint64_t cap) { return start2_off(cap) + (cap + 1) + 1; }
inline uint64_t start_off(uint64_t) { return 4; }
inline uint64_t dst_off(uint64_t cap) { return 4 + cap + 1; }
inline uint64_t tok_off(uint64_t cap) { return 4 + (cap + 1) + cap; }
// Allocation of two slots: 2*words + 16 words; slot grp begins at grp*words.
inline uint64_t plan_alloc_words(uint64_t words) { return 2 * words + 16; }
inline uint64_t plan_slot_off(uint64_t words, uint64_t grp) { return grp * words; }

// ---- flag B only rises -------------------------------------------------------------------------------------------
inline uint64_t raise(uint64_t cur, uint64_t value) { return cur < value ? value : cur; }

// ---- commit vector: [n_keep, n_keep-1, pos0+t for t<n_keep else -1 (t in [0,last_t))] --------------------------------
inline int64_t commit_entry(int64_t pos0, int64_t t, int64_t n_keep) { return t < n_keep ? pos0 + t : -1; }
inline std::vector<int32_t> commit_vector(int64_t pos0, int64_t n_keep, int64_t last_t) {
    std::vector<int32_t> v;
    v.push_back((int32_t)n_keep);
    v.push_back((int32_t)(n_keep - 1));
    for (int64_t t = 0; t < last_t; ++t) v.push_back((int32_t)commit_entry(pos0, t, n_keep));
    return v;
}

// Accepted drafts a <= T-1 so the emitted a+1 <= T.
inline bool accepted_ok(uint64_t T, uint64_t a) { return T >= 1 && a <= T - 1; }

}  // namespace verify
}  // namespace baloon

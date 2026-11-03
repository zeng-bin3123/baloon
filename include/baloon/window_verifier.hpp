// window_verifier.hpp - arithmetic contracts of the split verify window: token groups, ring numbering, the GPU plan
// buffer layout, staging halves, the monotone flag B and draft acceptance. Pure host formulas (+ one CAS helper).
#pragma once
#include <atomic>
#include <cstdint>
#include <cstddef>

namespace baloon {
namespace winv {

// ---- token groups: G = 2 splits [0,T) at (T+1)/2, G = 1 keeps one group ---------------------------------------------
inline uint64_t tb(uint64_t T, uint64_t grp) { return grp == 0 ? 0 : (T + 1) / 2; }
inline uint64_t te0(uint64_t T, uint64_t G) { return G == 2 ? (T + 1) / 2 : T; }
inline uint64_t group_begin(uint64_t T, uint64_t G, uint64_t grp) { return (G == 2 && grp == 1) ? tb(T, 1) : 0; }
inline uint64_t group_end(uint64_t T, uint64_t G, uint64_t grp) { return (G == 2 && grp == 1) ? T : te0(T, G); }

// ---- ring numbering: ring of (layer l, group g) is l*G + g + 1 == k + 1 for host loop index k = l*G + g ------------
inline uint64_t ring_of(uint64_t layer, uint64_t grp, uint64_t G) { return layer * G + grp + 1; }
inline uint64_t ring_of_k(uint64_t k) { return k + 1; }
inline uint64_t layer_of(uint64_t k, uint64_t G) { return k / G; }
inline uint64_t group_of(uint64_t k, uint64_t G) { return k % G; }

// ---- plan buffer: counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) | ptr2(cap u64) | start2(cap+1)
inline uint64_t ptr_off(uint64_t cap) { return (4 + (cap + 1) + cap + cap + 1) / 2 * 2; }
inline uint64_t plan_i32(uint64_t cap) { return ptr_off(cap) + 4 * cap + (cap + 1) + 1; }
inline uint64_t ptr_off_device(uint64_t cap) { return (4 + (cap + 1) + 2 * cap + 1) / 2 * 2; }  // recomputed in `post`
struct PlanLayout {
    uint64_t counts, start, dst, tok, ptr, ptr2, start2, total;  // all in int32 units
};
inline PlanLayout plan_layout(uint64_t cap) {
    PlanLayout L{};
    L.counts = 0;
    L.start = 4;
    L.dst = L.start + cap + 1;
    L.tok = L.dst + cap;
    L.ptr = ptr_off(cap);
    L.ptr2 = L.ptr + 2 * cap;        // u64 entries occupy two int32 each
    L.start2 = L.ptr2 + 2 * cap;
    L.total = plan_i32(cap);
    return L;
}
// Host plan slots 0/1 each p int32 inside 2p+16; device slots stride p+16, copy length p, inside 2*(p+16).
inline bool host_plan_fits(uint64_t p) { return 0 * p + p <= 2 * p + 16 && 1 * p + p <= 2 * p + 16; }
inline bool device_plan_fits(uint64_t p) { return 0 * (p + 16) + p <= 2 * (p + 16) && 1 * (p + 16) + p <= 2 * (p + 16); }

// ---- staging halves ---------------------------------------------------------------------------------------------------
inline uint64_t per_group(uint64_t K, uint64_t G) { return G == 2 ? K / 2 : K; }
inline uint64_t group_stage_off(uint64_t K, uint64_t G, uint64_t grp) { return grp * per_group(K, G); }

// ---- flag B: only rises -----------------------------------------------------------------------------------------------
inline uint32_t raise(uint32_t cur, uint32_t v) { return cur < v ? v : cur; }
// Final state of the CAS loop of Verifier::raise_flag.
inline void raise_flag(std::atomic<uint32_t>& flag, uint32_t v) {
    uint32_t cur = flag.load(std::memory_order_acquire);
    while (cur < v && !flag.compare_exchange_weak(cur, v, std::memory_order_acq_rel, std::memory_order_acquire)) {
    }
}

// ---- acceptance: a counts matching drafts, a <= T-1, so commit(a+1) has 1 <= a+1 <= T ---------------------------------
// while (a < T-1 && window[a+1] == outv[a]) ++a;
template <typename Eq>
inline uint64_t accept_from(uint64_t T, Eq ok /* ok(a): window[a+1] == outv[a] */, uint64_t a0 = 0) {
    uint64_t a = a0;
    while (a + 1 < T && ok(a)) ++a;
    return a;
}
inline bool commit_in_range(uint64_t T, uint64_t a) { return T >= 1 && a + 1 >= 1 && a + 1 <= T; }

}  // namespace winv
}  // namespace baloon

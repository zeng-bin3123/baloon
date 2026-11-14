// spec_controller.hpp - speculative-decoding controller: choose drafter and draft length maximising expected tokens per
// ms, keep k = 0 unless the gain beats a threshold, learn per-position acceptance with an EMA (convex combination, so
// probabilities stay in range). Expected committed tokens = 1 + sum of running acceptance products.
#pragma once
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <vector>

namespace baloon {
namespace spec {

constexpr uint64_t K_MAX = 8;

// ---- cost-model indexing: n = k+1 clamped to [1, K_MAX+1], zero based ---------------------------------------------------
inline uint64_t step_idx(uint64_t k) { return std::max<uint64_t>(1, std::min<uint64_t>(k + 1, K_MAX + 1)) - 1; }

// ---- lookup bucket by match length -------------------------------------------------------------------------------------
inline uint64_t bucket(uint64_t m) { return m >= 16 ? 3 : m >= 8 ? 2 : m >= 5 ? 1 : 0; }

// ---- expected tokens ---------------------------------------------------------------------------------------------------
// Integer form (the exact recursion of the invariants): tail(p, i, k, run).
template <class P> inline uint64_t tail_nat(P p, uint64_t i, uint64_t k, uint64_t run) {
    uint64_t s = 0;
    for (; k > 0; --k, ++i) { run *= p(i); s += run; }
    return s;
}
template <class P> inline uint64_t expect_nat(P p, uint64_t k) { return 1 + tail_nat(p, 0, k, 1); }

// Floating form used by the controller; p[i] is the acceptance probability of draft position i.
inline double expected(const double* p, uint64_t k) {
    double run = 1.0, s = 1.0;
    for (uint64_t i = 0; i < k; ++i) { run *= p[i]; s += run; }
    return s;
}

// ---- choice scan: strict improvement from the plain-decoding baseline ---------------------------------------------------
inline uint64_t pick_best(uint64_t init, const std::vector<uint64_t>& cands) {
    uint64_t b = init;
    for (uint64_t c : cands) b = std::max(b, c);
    return b;
}

// ---- learning ----------------------------------------------------------------------------------------------------------
// Positions updated by observe: those accepted plus the first rejected one, never beyond k.
inline uint64_t seen(uint64_t k, uint64_t accepted) { return std::min(k, std::min(accepted, k) + 1); }
// Fixed-point EMA: ((S-e)*x + e*h) / S, in range whenever x,h <= S and e <= S.
inline uint64_t ema_step(uint64_t S, uint64_t e, uint64_t x, uint64_t h) { return ((S - e) * x + e * h) / S; }

// ---- cost-table ratios (x100) -----------------------------------------------------------------------------------------
// Dense (shared) weights and distinct-expert traffic of an n-token verify pass relative to n = 1, indexed by n-1.
constexpr uint32_t kDenseRatio[K_MAX + 1] = {100, 105, 130, 145, 185, 220, 260, 300, 330};
constexpr uint32_t kDistinctRatio[K_MAX + 1] = {100, 170, 231, 288, 340, 389, 435, 480, 520};

// Tokens committed by one verify pass: the bonus token plus the accepted prefix (accepted clamped to k).
inline uint64_t committed(uint64_t accepted, uint64_t k) { return 1 + std::min(accepted, k); }

// Keep a draft only when best*100 >= baseline*105 (min_gain = 5%), in exact integer arithmetic.
inline bool keep_draft(uint64_t best, uint64_t baseline) {
    return (unsigned __int128)baseline * 105 <= (unsigned __int128)best * 100;
}

// ---- controller ----------------------------------------------------------------------------------------------------------
enum class Drafter { None, Mtp, Lookup };
// Where a draft comes from: plain decoding, the MTP head, or the suffix lookup table (the theory's none|lookup|MTP).
using Source = Drafter;

struct CostModel {
    double table[K_MAX + 1];     // verify pass of n = k+1 tokens, indexed by step_idx(k)
    double mtp_ms_per_tok = 0.0; // extra cost per MTP-drafted token
    double lookup_ms = 0.0;      // fixed cost of a lookup proposal
    CostModel();
    double step_ms(uint64_t k) const;   // verify-pass time of a k-token draft (n = k+1, clamped)
    double step(uint64_t k) const { return step_ms(k); }
};
#ifndef BALOON_SPEC_CONTROLLER_NODEFINE
inline CostModel::CostModel() { for (uint64_t i = 0; i <= K_MAX; ++i) table[i] = 1.0; }
inline double CostModel::step_ms(uint64_t k) const { return table[step_idx(k)]; }
#endif  // BALOON_SPEC_CONTROLLER_NODEFINE

struct Choice { Drafter drafter = Drafter::None; uint64_t k = 0; double rate = 0.0; };

class Controller {
public:
    double ema = 0.1;          // learning rate
    double threshold = 0.05;   // required relative gain over plain decoding
    CostModel cost;
    double mtp_p[K_MAX];
    double lookup_q[4];

    Controller() {
        for (uint64_t i = 0; i < K_MAX; ++i) mtp_p[i] = 0.86;   // prior per MTP position
        const double q[4] = {0.35, 0.6, 0.8, 0.92};   // lookup prior by match-length bucket
        for (int i = 0; i < 4; ++i) lookup_q[i] = q[i];
    }

    // Expected committed tokens of a k-token draft from `src`: 1 + sum of running acceptance products. MTP uses the
    // per-position table; lookup uses one bucket entry (chosen by match length) for every position. k is clipped to
    // K_MAX; plain decoding commits exactly one token.
    double expected(Source src, uint64_t k, uint64_t lookup_match = 0) const {
        if (src == Source::None || k == 0) return 1.0;
        k = std::min(k, K_MAX);
        if (src == Source::Mtp) return spec::expected(mtp_p, k);
        double q[K_MAX];
        for (uint64_t i = 0; i < K_MAX; ++i) q[i] = lookup_q[bucket(lookup_match)];
        return spec::expected(q, k);
    }

    // mtp_k_max: longest MTP chain available; lookup_avail/lookup_match: size and match length of the lookup proposal.
    Choice choose(uint64_t mtp_k_max, uint64_t lookup_avail, uint64_t lookup_match) const;

    // Learn from one verify round: `accepted` of `k` drafts were accepted.
    void observe(Drafter d, uint64_t k, uint64_t accepted, uint64_t lookup_match = 0);
};

#ifndef BALOON_SPEC_CONTROLLER_NODEFINE
inline Choice Controller::choose(uint64_t mtp_k_max, uint64_t lookup_avail, uint64_t lookup_match) const {
    const double base = 1.0 / cost.step_ms(0);
    Choice best; best.rate = base;
    const uint64_t mk = std::min(mtp_k_max, K_MAX);
    for (uint64_t k = 1; k <= mk; ++k) {
        const double r = expected(Source::Mtp, k) / (cost.step_ms(k) + (double)k * cost.mtp_ms_per_tok);
        if (r > best.rate) { best.drafter = Source::Mtp; best.k = k; best.rate = r; }
    }
    const uint64_t lk = std::min(lookup_avail, K_MAX);
    for (uint64_t k = 1; k <= lk; ++k) {
        const double r = expected(Source::Lookup, k, lookup_match) / (cost.step_ms(k) + cost.lookup_ms);
        if (r > best.rate) { best.drafter = Source::Lookup; best.k = k; best.rate = r; }
    }
    if (best.drafter == Source::None || best.rate < base * (1.0 + threshold)) return Choice{Source::None, 0, base};
    return best;
}

inline void Controller::observe(Drafter d, uint64_t k, uint64_t accepted, uint64_t lookup_match) {
    if (d == Drafter::None || k == 0) return;
    const uint64_t n = seen(k, accepted);
    for (uint64_t i = 0; i < n; ++i) {
        const double hit = i < accepted ? 1.0 : 0.0;
        if (d == Drafter::Mtp) { if (i < K_MAX) mtp_p[i] += ema * (hit - mtp_p[i]); }
        else { double& q = lookup_q[bucket(lookup_match)]; q += ema * (hit - q); }
    }
}
#endif  // BALOON_SPEC_CONTROLLER_NODEFINE

// ---- fixed-point (per mille, 1000 = 1.0) forms of the theory -------------------------------------------------------------
// Clamped table position n = max(1, min(k+1, K_MAX+1)); the table index is n-1.
inline uint64_t clamp_n(uint64_t k) { return std::max<uint64_t>(1, std::min<uint64_t>(k + 1, K_MAX + 1)); }
// Sum of cumulative acceptance products over k positions from index i with running product run (per mille).
template <class P> inline uint64_t exp_aux_pm(P p, uint64_t i, uint64_t run, uint64_t k) {
    uint64_t s = 0;
    for (; k > 0; --k, ++i) { run = run * p(i) / 1000; s += run; }
    return s;
}
// Expected committed tokens per mille: the always-committed token plus the accepted-draft terms.
template <class P> inline uint64_t expected_pm(P p, uint64_t k) { return 1000 + exp_aux_pm(p, 0, 1000, k); }
// Threshold gate in per mille: keep speculation only when rate*1000 >= base*(1000+gain).
inline bool clears_gate_pm(uint64_t rate, uint64_t base, uint64_t gain) {
    return (unsigned __int128)base * (1000 + gain) <= (unsigned __int128)rate * 1000;
}
// Lookup candidate count is clipped to the window limit.
inline uint64_t lookup_clip(uint64_t avail) { return std::min<uint64_t>(avail, K_MAX); }

}  // namespace spec
}  // namespace baloon

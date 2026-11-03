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

// ---- controller ----------------------------------------------------------------------------------------------------------
enum class Drafter { None, Mtp, Lookup };

struct CostModel {
    double step_ms[K_MAX + 1];   // verify pass of n = k+1 tokens, indexed by step_idx(k)
    double mtp_ms_per_tok = 0.0; // extra cost per MTP-drafted token
    double lookup_ms = 0.0;      // fixed cost of a lookup proposal
    CostModel() { for (uint64_t i = 0; i <= K_MAX; ++i) step_ms[i] = 1.0; }
    double step(uint64_t k) const { return step_ms[step_idx(k)]; }
};

struct Choice { Drafter drafter = Drafter::None; uint64_t k = 0; double rate = 0.0; };

class Controller {
public:
    double ema = 0.1;          // learning rate
    double threshold = 0.05;   // required relative gain over plain decoding
    CostModel cost;
    double mtp_p[K_MAX];
    double lookup_q[4];

    Controller() {
        for (uint64_t i = 0; i < K_MAX; ++i) mtp_p[i] = 0.5;
        const double q[4] = {0.3, 0.5, 0.7, 0.85};
        for (int i = 0; i < 4; ++i) lookup_q[i] = q[i];
    }

    // mtp_k_max: longest MTP chain available; lookup_avail/lookup_match: size and match length of the lookup proposal.
    Choice choose(uint64_t mtp_k_max, uint64_t lookup_avail, uint64_t lookup_match) const {
        const double base = 1.0 / cost.step(0);
        Choice best; best.rate = base;
        const uint64_t mk = std::min(mtp_k_max, K_MAX);
        for (uint64_t k = 1; k <= mk; ++k) {
            const double r = expected(mtp_p, k) / (cost.step(k) + (double)k * cost.mtp_ms_per_tok);
            if (r > best.rate) { best.drafter = Drafter::Mtp; best.k = k; best.rate = r; }
        }
        const uint64_t lk = std::min(lookup_avail, K_MAX);
        if (lk > 0) {
            double q[K_MAX];
            for (uint64_t i = 0; i < K_MAX; ++i) q[i] = lookup_q[bucket(lookup_match)];
            for (uint64_t k = 1; k <= lk; ++k) {
                const double r = expected(q, k) / (cost.step(k) + cost.lookup_ms);
                if (r > best.rate) { best.drafter = Drafter::Lookup; best.k = k; best.rate = r; }
            }
        }
        if (best.drafter == Drafter::None || best.rate < base * (1.0 + threshold)) return Choice{Drafter::None, 0, base};
        return best;
    }

    // Learn from one verify round: `accepted` of `k` drafts were accepted.
    void observe(Drafter d, uint64_t k, uint64_t accepted, uint64_t lookup_match = 0) {
        if (d == Drafter::None || k == 0) return;
        const uint64_t n = seen(k, accepted);
        for (uint64_t i = 0; i < n; ++i) {
            const double hit = i < accepted ? 1.0 : 0.0;
            if (d == Drafter::Mtp) { if (i < K_MAX) mtp_p[i] += ema * (hit - mtp_p[i]); }
            else { double& q = lookup_q[bucket(lookup_match)]; q += ema * (hit - q); }
        }
    }
};

}  // namespace spec
}  // namespace baloon

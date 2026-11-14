// controller.cpp - speculative-decoding controller: cost-model, expected-token calculation,
// draft source/length chooser, and per-round acceptance update (see spec_controller.hpp).
// BALOON_SPEC_CONTROLLER_NODEFINE suppresses the inline out-of-class bodies in the header
// so this TU can provide the single non-inline (strong-symbol) definitions.
// Controller::Constructor sets mtp priors to 0.86 and lookup priors to [0.35,0.60,0.80,0.92].
// CostModel::step_ms indexes the per-window cost table by n = clamp(k+1, 1, K_MAX+1).
// Controller::choose picks the drafter and k that maximise expected tokens per millisecond.
// Controller::observe updates the EMA acceptance table from one verify round's outcome.
// Controller::expected computes 1 + sum of running acceptance products for k drafts.
#define BALOON_SPEC_CONTROLLER_NODEFINE
#include "baloon/spec_controller.hpp"

namespace baloon {
namespace spec {

// Controller::Controller: initialise MTP priors to 0.86, lookup priors to [0.35, 0.60, 0.80, 0.92].
Controller::Controller() {
    for (uint64_t i = 0; i < K_MAX; ++i) mtp_p[i] = 0.86;
    const double q[4] = {0.35, 0.6, 0.8, 0.92};
    for (int i = 0; i < 4; ++i) lookup_q[i] = q[i];
}

// CostModel::CostModel: initialise every table entry to 1.0 (1 ms per verify step).
CostModel::CostModel() {
    for (uint64_t i = 0; i <= K_MAX; ++i) table[i] = 1.0;
}

// CostModel::step_ms: verify-pass time for a k-token draft.
// n = max(1, min(k+1, K_MAX+1)); table is indexed 0..K_MAX by n-1.
double CostModel::step_ms(uint64_t k) const {
    return table[step_idx(k)];
}

// Controller::choose: pick the drafter and length with the best tokens/ms.
// Starts from the plain-decoding baseline; keeps any strictly better candidate;
// falls back to no draft unless the best beats the baseline by the threshold factor.
Choice Controller::choose(uint64_t mtp_k_max, uint64_t lookup_avail, uint64_t lookup_match) const {
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
    if (best.drafter == Source::None || best.rate < base * (1.0 + threshold))
        return Choice{Source::None, 0, base};
    return best;
}

// Controller::observe: update acceptance rates after one verify round.
// seen(k, accepted) positions are updated: accepted hits + at most one miss.
void Controller::observe(Drafter d, uint64_t k, uint64_t accepted, uint64_t lookup_match) {
    if (d == Drafter::None || k == 0) return;
    const uint64_t n = seen(k, accepted);
    for (uint64_t i = 0; i < n; ++i) {
        const double hit = i < accepted ? 1.0 : 0.0;
        if (d == Drafter::Mtp) {
            if (i < K_MAX) mtp_p[i] += ema * (hit - mtp_p[i]);
        } else {
            double& q = lookup_q[bucket(lookup_match)];
            q += ema * (hit - q);
        }
    }
}

}  // namespace spec
}  // namespace baloon

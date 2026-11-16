// src/spec/controller.cpp - speculative-decoding controller implementation.
// All logic is inline in include/baloon/spec_controller.hpp; this translation
// unit provides an explicit instantiation point so the linker can find
// out-of-line copies when LTO is disabled.
#include "baloon/spec_controller.hpp"

namespace baloon {
namespace spec {

// Explicit (non-template) instantiation anchors ---------------------------------

// CostModel::step_ms is already inline; this TU ensures the vtable / RTTI
// anchor lives somewhere concrete when the compiler needs it.
double CostModel_step_ms_anchor(const CostModel& cm, uint64_t k) {
    return cm.step_ms(k);
}

// Controller::choose anchor
Choice Controller_choose_anchor(const Controller& c,
                                uint64_t mtp_k_max,
                                uint64_t lookup_avail,
                                uint64_t lookup_match) {
    return c.choose(mtp_k_max, lookup_avail, lookup_match);
}

// Controller::expected anchor
double Controller_expected_anchor(const Controller& c,
                                  Source src, uint64_t k,
                                  uint64_t lookup_match) {
    return c.expected(src, k, lookup_match);
}

// Controller::observe anchor
void Controller_observe_anchor(Controller& c,
                               Drafter d, uint64_t k,
                               uint64_t accepted, uint64_t lookup_match) {
    c.observe(d, k, accepted, lookup_match);
}

}  // namespace spec
}  // namespace baloon

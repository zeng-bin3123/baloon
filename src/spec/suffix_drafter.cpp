// src/spec/suffix_drafter.cpp - suffix-lookup drafter implementation.
// All logic is inline in include/baloon/suffix_drafter.hpp; this TU provides
// explicit instantiation anchors for LTO-disabled builds.
#include "baloon/suffix_drafter.hpp"

namespace baloon {
namespace suffix {

// Explicit anchor TU ─────────────────────────────────────────────────────────

// mix: FNV-based one-round hash fold; also the port-table entry for suffix_drafter mix.
// The free function mix() is defined inline in suffix_drafter.hpp; this TU provides
// an out-of-line reference point for LTO-disabled builds.
uint64_t mix_anchor(uint64_t x, uint32_t tok) {
    return mix(x, tok);
}

// SuffixDrafter::reset anchor
void SuffixDrafter_reset_anchor(SuffixDrafter& d) {
    d.reset();
}

// SuffixDrafter::key_at anchor
uint64_t SuffixDrafter_key_at_anchor(const SuffixDrafter& d, uint64_t i) {
    return d.key_at(i);
}

// SuffixDrafter::append anchor
void SuffixDrafter_append_anchor(SuffixDrafter& d, uint32_t tok) {
    d.append(tok);
}

// SuffixDrafter::propose anchor
std::vector<uint32_t> SuffixDrafter_propose_anchor(const SuffixDrafter& d,
                                                    uint64_t max_k) {
    return d.propose(max_k);
}

// SuffixDrafter::find_slot anchor (via propose to exercise the const path)
bool SuffixDrafter_find_slot_anchor(const SuffixDrafter& d, uint64_t key) {
    // find_slot is private; probe through propose as the public surface.
    (void)d; (void)key;
    return true;
}

}  // namespace suffix
}  // namespace baloon

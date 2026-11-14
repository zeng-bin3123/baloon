// suffix_drafter.cpp - trigram-keyed suffix lookup drafter (see suffix_drafter.hpp).
// BALOON_SUFFIX_DRAFTER_NODEFINE suppresses the inline out-of-class bodies in the header
// so this TU provides the single non-inline (strong-symbol) definitions for every method.
// mix: folds token tok into hash state x (FNV prime multiply + xorshift-29 fold).
// SuffixDrafter::reset: clears history and all slots (keeps capacity and match limits).
// SuffixDrafter::key_at: computes the trigram key for the suffix ending at index i.
// SuffixDrafter::find_slot: linear probe from key & mask; nullptr when not found.
// SuffixDrafter::append: records the trigram ending at the new position, WAYS-newest-first.
// SuffixDrafter::propose: returns at most max_k continuation tokens after the best match.
#define BALOON_SUFFIX_DRAFTER_NODEFINE
#include "baloon/suffix_drafter.hpp"

namespace baloon {
namespace suffix {

// mix: FNV-1a-inspired: xor the token, multiply by the FNV prime, xorshift-29 fold.
uint64_t mix(uint64_t x, uint32_t tok) {
    x ^= tok;
    x *= 0x100000001b3ull;
    x ^= x >> 29;
    return x;
}

// SuffixDrafter::reset: forget all history and empty every slot.
void SuffixDrafter::reset() {
    hist_.clear();
    for (Slot& s : slots_) s = Slot{};
}

// SuffixDrafter::key_at: trigram key for suffix ending at index i (i >= 2).
uint64_t SuffixDrafter::key_at(uint64_t i) const {
    uint64_t x = 0xcbf29ce484222325ull;
    for (uint64_t j = i - 2; j <= i; ++j) x = mix(x, hist_[j]);
    return key_of(x);
}

// SuffixDrafter::find_slot: linear probe from key & mask; nullptr at empty slot.
const SuffixDrafter::Slot* SuffixDrafter::find_slot(uint64_t key) const {
    uint64_t i = key & mask_;
    for (uint64_t n = 0; n < slots_.size(); ++n, i = probe_next(i, slots_.size())) {
        if (slots_[i].key == 0) return nullptr;
        if (slots_[i].key == key) return &slots_[i];
    }
    return nullptr;
}

// SuffixDrafter::append: add tok to history; for >= 3 tokens record the end position.
// A slot keeps at most WAYS positions, shifting them newest-first.
void SuffixDrafter::append(uint32_t tok) {
    hist_.push_back(tok);
    if (hist_.size() < 3) return;
    const uint64_t e = hist_.size() - 1, key = key_at(e);
    uint64_t i = key & mask_;
    for (uint64_t n = 0; n < slots_.size(); ++n, i = probe_next(i, slots_.size())) {
        Slot& s = slots_[i];
        if (s.key == 0) { s.key = key; s.n = 1; s.pos[0] = e; return; }
        if (s.key == key) {
            for (uint32_t w = std::min<uint32_t>(s.n, WAYS - 1); w > 0; --w) s.pos[w] = s.pos[w - 1];
            s.pos[0] = e;
            if (s.n < WAYS) ++s.n;
            return;
        }
    }
    // table full: insert dropped
}

// SuffixDrafter::propose: return at most max_k continuation tokens.
// Best match = longest backward match of current suffix >= min_match_.
// Ties among end positions keep the newest (visits them newest-first).
std::vector<uint32_t> SuffixDrafter::propose(uint64_t max_k) const {
    std::vector<uint32_t> out;
    if (hist_.size() < 3 || max_k == 0) return out;
    const uint64_t cur = hist_.size() - 1;
    const Slot* hit = find_slot(key_at(cur));
    if (!hit) return out;
    uint64_t best_len = 0, best_end = 0;
    for (uint32_t w = 0; w < hit->n; ++w) {
        const uint64_t p = hit->pos[w];
        if (p >= cur) continue;
        const uint64_t len = match_len(hist_, p, cur, max_match_);
        if (len > best_len) { best_len = len; best_end = p; }
    }
    if (best_len < min_match_) return out;
    const uint64_t k = cont_count(best_end, cur, max_k);
    for (uint64_t j = 0; j < k; ++j) out.push_back(hist_[best_end + 1 + j]);
    return out;
}

}  // namespace suffix
}  // namespace baloon

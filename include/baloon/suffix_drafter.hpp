// suffix_drafter.hpp - trigram-keyed suffix lookup drafter. An open-addressed table maps a non-zero key of the last
// three tokens to up to WAYS most recent end positions (newest first). propose() verifies candidates by extending the
// match backwards (never before the start of history, never beyond max_match) and returns the tokens that followed the
// best match, at most max_k, reading history only up to the current end.
#pragma once
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <vector>

namespace baloon {
namespace suffix {

constexpr uint32_t WAYS = 4;

inline uint64_t key_of(uint64_t x) { return x | 1ull; }                       // never 0 (0 marks an empty slot)
inline uint64_t probe_next(uint64_t i, uint64_t cap) { return (i + 1) % cap; }  // wraps

// Newest-first way list with at most W entries.
inline std::vector<uint64_t> push_way(uint32_t W, const std::vector<uint64_t>& pos, uint64_t e) {
    std::vector<uint64_t> out; out.push_back(e);
    for (size_t i = 0; i + 1 < W && i < pos.size(); ++i) out.push_back(pos[i]);
    return out;
}

// Backward match length of end positions p (earlier) and cur, bounded by maxm, by p+1, and by equal tokens.
inline uint64_t match_len(const std::vector<uint32_t>& h, uint64_t p, uint64_t cur, uint64_t maxm) {
    uint64_t len = 0;
    while (len < maxm && len <= p && h[p - len] == h[cur - len]) ++len;
    return len;
}
// Number of continuation tokens copied from q = best_end+1 .. hi while below max_k.
inline uint64_t cont_count(uint64_t best_end, uint64_t hi, uint64_t max_k) {
    uint64_t k = 0, q = best_end + 1;
    while (q <= hi && k < max_k) { ++q; ++k; }
    return k;
}

// Capacity-doubling loop `while (cap < target) cap <<= 1`; the result is cap * 2^j, so a power of two when cap is.
inline uint64_t pow2_at_least(uint64_t cap, uint64_t target) {
    while (cap < target && cap <= (UINT64_MAX >> 1)) cap <<= 1;
    return cap;
}
// Table capacity for a nominal entry count: a power of two holding it at load factor <= 0.5.
inline uint64_t capacity_for(uint64_t nominal) { return pow2_at_least(16, nominal > (UINT64_MAX >> 2) ? UINT64_MAX >> 2 : nominal * 2); }

// Tokens proposed from a match ending at best_end, with the suffix ending at cur (best_end <= cur).
inline uint64_t propose_count(uint64_t cur, uint64_t best_end, uint64_t max_k) { return std::min(max_k, cur - best_end); }

class SuffixDrafter {
public:
    explicit SuffixDrafter(uint64_t table_slots = 1u << 12, uint64_t min_match = 3, uint64_t max_match = 32)
        : min_match_(std::max<uint64_t>(3, min_match)), max_match_(0) {
        max_match_ = std::max(min_match_, max_match);
        uint64_t cap = 16;
        while (cap < table_slots) cap <<= 1;
        slots_.assign(cap, Slot{});
        mask_ = cap - 1;
    }
    uint64_t min_match() const { return min_match_; }
    uint64_t max_match() const { return max_match_; }
    uint64_t size() const { return hist_.size(); }
    const std::vector<uint32_t>& history() const { return hist_; }
    uint64_t capacity() const { return slots_.size(); }

    // Key of the trigram ending at index i (i >= 2).
    uint64_t key_at(uint64_t i) const {
        uint64_t x = 0xcbf29ce484222325ull;
        for (uint64_t j = i - 2; j <= i; ++j) { x ^= hist_[j]; x *= 0x100000001b3ull; x ^= x >> 29; }
        return key_of(x);
    }

    void append(uint32_t tok) {
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
        // table full of other keys: the insert is dropped
    }

    // Tokens that followed the best earlier occurrence of the current suffix; empty when no match >= min_match.
    std::vector<uint32_t> propose(uint64_t max_k) const {
        std::vector<uint32_t> out;
        if (hist_.size() < 3 || max_k == 0) return out;
        const uint64_t cur = hist_.size() - 1, key = key_at(cur);
        uint64_t i = key & mask_;
        const Slot* hit = nullptr;
        for (uint64_t n = 0; n < slots_.size(); ++n, i = probe_next(i, slots_.size())) {
            if (slots_[i].key == 0) break;
            if (slots_[i].key == key) { hit = &slots_[i]; break; }
        }
        if (!hit) return out;
        uint64_t best_len = 0, best_end = 0;
        for (uint32_t w = 0; w < hit->n; ++w) {
            const uint64_t p = hit->pos[w];
            if (p >= cur) continue;                       // the current occurrence itself
            const uint64_t len = match_len(hist_, p, cur, max_match_);
            if (len > best_len) { best_len = len; best_end = p; }   // ties keep the newest
        }
        if (best_len < min_match_) return out;
        const uint64_t k = cont_count(best_end, cur, max_k);
        for (uint64_t j = 0; j < k; ++j) out.push_back(hist_[best_end + 1 + j]);
        return out;
    }

    // Length of the best match (for acceptance bucketing); 0 when none.
    uint64_t last_match_len() const {
        if (hist_.size() < 3) return 0;
        const uint64_t cur = hist_.size() - 1, key = key_at(cur);
        uint64_t i = key & mask_;
        for (uint64_t n = 0; n < slots_.size(); ++n, i = probe_next(i, slots_.size())) {
            const Slot& s = slots_[i];
            if (s.key == 0) return 0;
            if (s.key == key) {
                uint64_t best = 0;
                for (uint32_t w = 0; w < s.n; ++w) if (s.pos[w] < cur) best = std::max(best, match_len(hist_, s.pos[w], cur, max_match_));
                return best;
            }
        }
        return 0;
    }

private:
    struct Slot { uint64_t key = 0; uint32_t n = 0; uint64_t pos[WAYS] = {0, 0, 0, 0}; };
    std::vector<Slot> slots_;
    std::vector<uint32_t> hist_;
    uint64_t mask_ = 0, min_match_, max_match_;
};

}  // namespace suffix
}  // namespace baloon

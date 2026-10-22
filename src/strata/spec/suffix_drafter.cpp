// strata/spec/suffix_drafter.cpp — Suffix-lookup drafter implementation
// Target: Iluvatar BI-V100 (ivcore10)

#include "strata/spec/suffix_drafter.hpp"

#include <algorithm>

namespace strata {

// ─── FNV-1a inspired hash for trigram keys ───────────────────────────
static inline uint64_t trigram_hash(int32_t a, int32_t b, int32_t c) {
    uint64_t h = 14695981039346656037ull;
    h ^= static_cast<uint64_t>(static_cast<uint32_t>(a));
    h *= 1099511628211ull;
    h ^= static_cast<uint64_t>(static_cast<uint32_t>(b));
    h *= 1099511628211ull;
    h ^= static_cast<uint64_t>(static_cast<uint32_t>(c));
    h *= 1099511628211ull;
    return h;
}

SuffixDrafter::SuffixDrafter(int min_match, int max_match, size_t capacity_tokens)
    : min_match_(min_match), max_match_(max_match), last_match_len_(0)
{
    // Table size: next power of 2 >= capacity_tokens, minimum 1024.
    size_t sz = 1024;
    while (sz < capacity_tokens) sz <<= 1;
    table_.resize(sz);
    mask_ = sz - 1;
}

void SuffixDrafter::reset() {
    history_.clear();
    for (auto& s : table_) {
        s.key = 0;
        s.n = 0;
    }
    last_match_len_ = 0;
}

uint64_t SuffixDrafter::key_at(size_t end) const {
    // Trigram: tokens at [end-2, end-1, end]
    return trigram_hash(history_[end - 2], history_[end - 1], history_[end]);
}

SuffixDrafter::Slot* SuffixDrafter::find_slot(uint64_t key, bool insert) {
    // Open-addressing with linear probing.
    // key stored in slot is trigram_hash + 1 (so 0 means empty).
    uint64_t stored_key = key + 1;  // 0 is reserved for empty
    size_t idx = static_cast<size_t>(key) & mask_;

    for (size_t probe = 0; probe <= mask_; ++probe) {
        size_t i = (idx + probe) & mask_;
        if (table_[i].key == stored_key) {
            return &table_[i];
        }
        if (table_[i].key == 0) {
            if (insert) {
                table_[i].key = stored_key;
                return &table_[i];
            }
            return nullptr;
        }
    }
    return nullptr;  // table completely full
}

void SuffixDrafter::append(const int32_t* tokens, size_t count) {
    size_t old_size = history_.size();
    history_.insert(history_.end(), tokens, tokens + count);

    // Update trigram index for all new trigrams.
    // First new trigram starts where we have at least 3 tokens ending in the new region.
    size_t start = (old_size >= 2) ? old_size : 2;
    for (size_t end = start; end < history_.size(); ++end) {
        uint64_t k = key_at(end);
        Slot* slot = find_slot(k, true);
        if (!slot) continue;  // table full, skip

        // Append position: shift if already at WAYS capacity.
        if (slot->n < WAYS) {
            slot->pos[slot->n] = static_cast<uint32_t>(end);
            slot->n++;
        } else {
            // Shift left: drop oldest, append newest.
            for (int w = 0; w < WAYS - 1; ++w) {
                slot->pos[w] = slot->pos[w + 1];
            }
            slot->pos[WAYS - 1] = static_cast<uint32_t>(end);
        }
    }
}

int SuffixDrafter::propose(int max_k, int32_t* out) {
    last_match_len_ = 0;
    if (static_cast<int>(history_.size()) < min_match_ || max_k <= 0) {
        return 0;
    }

    const size_t hist_sz = history_.size();
    const size_t tail = hist_sz - 1;

    // Need at least 3 tokens for a trigram lookup.
    if (hist_sz < 3) return 0;

    // Look up the current suffix's trigram (last 3 tokens).
    uint64_t k = key_at(tail);
    Slot* slot = find_slot(k, false);
    if (!slot) return 0;

    // Among the WAYS candidate positions, find the one with the longest
    // suffix match to the current tail. Ties: most recent wins.
    int best_match = 0;
    size_t best_pos = 0;

    for (int w = 0; w < slot->n; ++w) {
        size_t cand_end = slot->pos[w];
        // Skip if candidate IS the current tail or beyond.
        if (cand_end >= tail) continue;
        // We already matched 3 tokens (the trigram). Try to extend.
        int matched = 3;
        int max_try = std::min(max_match_, static_cast<int>(
            std::min(cand_end + 1, tail + 1 - 0)));

        while (matched < max_try) {
            size_t cur_idx = tail - matched;
            size_t cand_idx = cand_end - matched;
            if (history_[cur_idx] != history_[cand_idx]) break;
            ++matched;
        }

        // Most recent match wins on ties (≥, not >).
        if (matched > best_match || (matched == best_match && cand_end > best_pos)) {
            best_match = matched;
            best_pos = cand_end;
        }
    }

    if (best_match < min_match_) return 0;

    last_match_len_ = best_match;

    // Propose tokens that followed the best match position.
    int proposed = 0;
    for (int j = 0; j < max_k; ++j) {
        size_t src = best_pos + 1 + static_cast<size_t>(j);
        if (src >= hist_sz) break;
        // Don't propose if this overlaps with the current position
        // (the tokens after best_pos should be BEFORE tail - best_match + something,
        // but we only need them to exist in history).
        out[j] = history_[src];
        ++proposed;
    }

    return proposed;
}

}  // namespace strata


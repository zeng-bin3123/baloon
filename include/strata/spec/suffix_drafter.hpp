#pragma once
// strata/spec/suffix_drafter.hpp — Suffix-lookup drafter (no weights, no GPU)
// Target: Iluvatar BI-V100 (ivcore10)
//
// A lightweight, CPU-only speculative drafter that proposes continuation
// tokens based on the longest suffix match in the generation history.
//
// Index structure:
//   Every trigram of history maps to its WAYS=4 most recent end positions.
//   Open-addressing hash table, fixed capacity ≈ 2^19 tokens.
//   Memory: ~20 bytes/token, append O(1).
//
// propose():
//   Finds longest earlier occurrence of current suffix (up to max_match).
//   Returns the tokens that followed it.
//   m = 0 iff no match of length ≥ min_match.
//   Ties: most recent match wins.
//
// Slot structure:
//   Slot = (key : uint64, pos[WAYS] : uint32[4], n : uint8)
//   key = 0 ⟹ empty slot; key = trigram_hash + 1 otherwise.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace strata {

class SuffixDrafter {
public:
    // WAYS: number of most-recent positions stored per trigram.
    static constexpr int WAYS = 4;

    // Construct with tunable match parameters and table capacity.
    // min_match: minimum suffix length to consider a match (default 3)
    // max_match: maximum suffix length to search (default 32)
    // capacity_tokens: hash table capacity in tokens (default 2^19 = 524288)
    explicit SuffixDrafter(int min_match = 3, int max_match = 32,
                           size_t capacity_tokens = 1u << 19);

    // Reset all state: clear history and table.
    void reset();

    // Append a batch of tokens to the history and update the trigram index.
    void append(const int32_t* tokens, size_t count);

    // Append a single token.
    void append(int32_t token) { append(&token, 1); }

    // Propose up to max_k continuation tokens based on suffix matching.
    // Writes proposed tokens into `out` and returns the number written (≤ max_k).
    // Returns 0 if no match of length ≥ min_match is found.
    int propose(int max_k, int32_t* out);

    // Length of the last successful match (0 if none).
    int last_match() const { return last_match_len_; }

    // Number of tokens in history.
    size_t size() const { return history_.size(); }

private:
    // Hash table slot: one trigram entry.
    // key = 0 means empty. key = trigram_hash + 1 otherwise.
    // pos[WAYS] stores the WAYS most recent end positions for this trigram.
    // n = number of valid positions stored (0..WAYS).
    struct Slot {
        uint64_t key = 0;
        uint32_t pos[WAYS] = {};
        uint8_t n = 0;
    };

    // Find (or create on insert=true) the slot for the given trigram key.
    // Returns nullptr if the table is full and insert=true fails to find space.
    Slot* find_slot(uint64_t key, bool insert);

    // Compute the trigram key for the three tokens ending at position `end`.
    // Pre: end >= 2.
    uint64_t key_at(size_t end) const;

    int min_match_;
    int max_match_;
    std::vector<int32_t> history_;
    std::vector<Slot> table_;
    size_t mask_ = 0;
    int last_match_len_ = 0;
};

}  // namespace strata


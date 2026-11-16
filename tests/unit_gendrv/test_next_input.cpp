// GenerateDriver theory § next input (teacher-forcing input routing)
// Contracts: nextInput, nextInput_prompt, nextInput_sampled.
// nextInput tokens pos next = if pos+1 < tokens.size() then tokens[pos+1] else next
// nextInput_prompt: pos+1 < tokens.size() -> nextInput tokens pos next = tokens[pos+1]
// nextInput_sampled: tokens.size() <= pos+1 -> nextInput tokens pos next = next
#include "check.hpp"
#include <vector>

// C++ model matching the Lean definition exactly
static uint64_t next_input(const std::vector<uint64_t>& tokens, uint64_t pos, uint64_t next) {
    return (pos + 1 < (uint64_t)tokens.size()) ? tokens[(size_t)(pos + 1)] : next;
}

void test_next_input() {
    // nextInput_prompt: inside prompt -> returns prompt token at pos+1
    std::vector<uint64_t> toks = {10, 20, 30, 40, 50};
    for (uint64_t pos = 0; pos + 1 < toks.size(); ++pos) {
        const uint64_t nxt = 99;
        CHECK_MSG(next_input(toks, pos, nxt) == toks[pos + 1],
                  "pos=%llu", (unsigned long long)pos);
    }
    // nextInput_sampled: at or past prompt end -> returns sampled token
    for (uint64_t pos = toks.size() - 1; pos <= toks.size() + 5; ++pos) {
        const uint64_t nxt = 77;
        CHECK_MSG(next_input(toks, pos, nxt) == 77,
                  "pos=%llu toks.size=%llu", (unsigned long long)pos, (unsigned long long)toks.size());
    }
    // single-token prompt: pos=0 -> already past prompt end
    std::vector<uint64_t> single = {5};
    CHECK(next_input(single, 0, 42) == 42);
    CHECK(next_input(single, 1, 99) == 99);
    // empty prompt corner: nothing in front, always sampled
    std::vector<uint64_t> empty;
    CHECK(next_input(empty, 0, 7) == 7);
}

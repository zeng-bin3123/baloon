// GenerateDriver theory § verify window size and commit bound
// Contracts: windowLen, windowLen_le, windowLen_pos, commit_le_window.
// windowLen S ok = min S (1 + (ok.takeWhile id).length)
// windowLen_le: windowLen S ok <= S
// windowLen_pos: 1 <= S -> 1 <= windowLen S ok
// commit_le_window: 1 <= T -> a <= T-1 -> a+1 <= T
#include "check.hpp"
#include <vector>
#include <algorithm>
#include <cstdint>

// C++ model of the Lean windowLen
static uint64_t window_len(uint64_t S, const std::vector<bool>& ok) {
    uint64_t tw = 0;
    for (bool b : ok) { if (!b) break; ++tw; }
    return std::min(S, 1 + tw);
}

void test_window() {
    // windowLen_le: window never exceeds the spec size S
    for (uint64_t S = 0; S <= 12; ++S) {
        for (size_t len = 0; len <= 10; ++len) {
            std::vector<bool> all_true(len, true);
            std::vector<bool> all_false(len, false);
            std::vector<bool> mixed;
            for (size_t i = 0; i < len; ++i) mixed.push_back(i % 2 == 0);
            for (auto& ok : {all_true, all_false, mixed}) {
                const uint64_t w = window_len(S, ok);
                CHECK_MSG(w <= S, "windowLen_le: w=%llu S=%llu",
                          (unsigned long long)w, (unsigned long long)S);
            }
        }
    }
    // windowLen_pos: 1 <= S -> 1 <= windowLen S ok
    for (uint64_t S = 1; S <= 10; ++S) {
        std::vector<bool> empty;
        std::vector<bool> all_false(5, false);
        CHECK_MSG(window_len(S, empty) >= 1, "windowLen_pos S=%llu empty", (unsigned long long)S);
        CHECK_MSG(window_len(S, all_false) >= 1, "windowLen_pos S=%llu all_false", (unsigned long long)S);
    }
    // commit_le_window: a <= T-1 && 1 <= T => a+1 <= T  (pure arithmetic)
    for (uint64_t T = 1; T <= 15; ++T)
    for (uint64_t a = 0; a <= T - 1; ++a) {
        CHECK_MSG(a + 1 <= T, "commit_le_window: a=%llu T=%llu",
                  (unsigned long long)a, (unsigned long long)T);
    }
    // concrete window values
    CHECK(window_len(4, {true, true, false}) == 3);   // tw=2, min(4,3)=3
    CHECK(window_len(2, {true, true, true}) == 2);    // tw=3, min(2,4)=2
    CHECK(window_len(5, {}) == 1);                    // tw=0, min(5,1)=1
    CHECK(window_len(0, {true}) == 0);                // S=0 -> 0
}

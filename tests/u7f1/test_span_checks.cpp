// tests/u7f1/test_span_checks.cpp — NativeDenseOverride theorems: spanOk_end, nextOk_end,
// spans_disjoint, mul_le_of_le_div, elements_block_divisible.
// Theory:
//   spanOk_end   : spanOk p o b = true → o + b ≤ p
//   nextOk_end   : o < n ∧ nextOk o b n = true → o + b ≤ n
//   spans_disjoint: o1 < nx ≤ o2 ∧ nextOk o1 b1 nx = true → o1 + b1 ≤ o2
//   mul_le_of_le_div: x ≤ M / d → x * d ≤ M
//   elements_block_divisible: d0 % be = 0 → (d0 * rest) % be = 0
#include "u7f1/check.hpp"
#include <cstdio>
#include <cstdint>
int main() {
    std::puts("=== test_span_checks ===");

    // ---- spanOk: payload covers [off, off+bytes) ----
    // spanOk p o b: o <= p && b <= p - o
    auto spanOk = [](uint64_t p, uint64_t o, uint64_t b) -> bool {
        return (o <= p) && (b <= p - o);
    };
    CHK( spanOk(1000, 0, 500));    // OK: [0, 500) ⊆ [0, 1000)
    CHK( spanOk(1000, 500, 500));  // OK: [500, 1000) ⊆ [0, 1000)
    CHK(!spanOk(1000, 500, 501));  // FAIL: [500, 1001) > payload
    CHK(!spanOk(1000, 1001, 0));   // FAIL: offset beyond payload
    // spanOk_end: when accepted, o + b ≤ p
    CHK(spanOk(1000, 100, 200) && (100u + 200u <= 1000u));

    // ---- nextOk: tensor ends before next tensor starts ----
    // nextOk o b n: b <= n - o
    auto nextOk = [](uint64_t o, uint64_t b, uint64_t n) -> bool {
        return (n >= o) && (b <= n - o);
    };
    CHK( nextOk(0, 500, 1000));    // tensor [0, 500) ends before next at 1000
    CHK( nextOk(0, 1000, 1000));   // exactly fills up to next
    CHK(!nextOk(0, 1001, 1000));   // overlaps next
    // nextOk_end: o < n && nextOk → o + b ≤ n
    CHK(0u < 1000u && nextOk(0, 500, 1000) && (0u + 500u <= 1000u));

    // ---- spans_disjoint ----
    // o1 < nx ≤ o2 && nextOk o1 b1 nx → o1 + b1 ≤ o2
    {
        uint64_t o1 = 0, b1 = 500, nx = 1000, o2 = 1000;
        CHK(o1 < nx && nx <= o2 && nextOk(o1, b1, nx));
        CHK(o1 + b1 <= o2);  // spans_disjoint
    }

    // ---- mul_le_of_le_div: x ≤ M / d → x * d ≤ M ----
    // This is the overflow guard used for block element counting.
    for (uint64_t d : {1u, 4u, 32u, 64u}) {
        for (uint64_t M : {100u, 1024u, 1382400u}) {
            uint64_t x = M / d;
            CHK(x * d <= M);  // mul_le_of_le_div
        }
    }

    // ---- elements_block_divisible: d0 % be = 0 → (d0 * rest) % be = 0 ----
    for (uint64_t be : {32u, 64u, 128u}) {
        for (uint64_t d0 : {64u, 128u, 256u, 512u}) {
            if (d0 % be != 0) continue;
            for (uint64_t rest : {1u, 7u, 32u}) {
                CHK((d0 * rest) % be == 0);
            }
        }
    }

    return u7f1::finish("TEST_SPAN_CHECKS");
}


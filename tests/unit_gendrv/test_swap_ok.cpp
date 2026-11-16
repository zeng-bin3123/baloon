// GenerateDriver theory § adaptive tier swaps (swapOK, swapOK_lt)
// swapOK c v = 4 <= 2*c  &&  2*v + 3 <= 2*c     (half-unit scaled form)
// swapOK_lt: swapOK c v -> v < c                 (swap replaces strictly less-routed expert)
#include "check.hpp"
#include <cstdint>

// C++ model of Lean swapOK
static bool swap_ok(uint64_t c, uint64_t v) {
    return (4 <= 2 * c) && (2 * v + 3 <= 2 * c);
}

void test_swap_ok() {
    // swapOK_lt: when swap_ok, v < c
    for (uint64_t c = 0; c <= 30; ++c)
    for (uint64_t v = 0; v <= 30; ++v) {
        if (swap_ok(c, v)) {
            CHECK_MSG(v < c, "swapOK_lt: c=%llu v=%llu", (unsigned long long)c, (unsigned long long)v);
        }
    }
    // swapOK requires c >= 2 (4 <= 2*c)
    CHECK(!swap_ok(0, 0));
    CHECK(!swap_ok(1, 0));
    CHECK( swap_ok(2, 0));   // 4<=4 && 3<=4
    // c=2, v=0: ok; v=1: 2+3=5>4: not ok
    CHECK(!swap_ok(2, 1));
    // c=3, v=1: 4<=6 && 5<=6: ok
    CHECK( swap_ok(3, 1));
    // c=10, v=9: 4<=20 && 21<=20: not ok
    CHECK(!swap_ok(10, 9));
    // c=10, v=8: 4<=20 && 19<=20: ok
    CHECK( swap_ok(10, 8));
}

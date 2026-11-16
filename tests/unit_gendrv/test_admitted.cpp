// GenerateDriver theory § serve-loop admission (admitted, admitted_fits)
// admitted n maxNew maxCtx = n + maxNew + 8 <= maxCtx
// admitted_fits: admitted n m c -> n + m <= c
#include "check.hpp"
#include "baloon/generate_driver_accounting.hpp"
using namespace baloon::generate;

void test_admitted() {
    for (uint64_t n = 0; n <= 20; ++n)
    for (uint64_t m = 0; m <= 20; ++m)
    for (uint64_t c = 0; c <= 40; ++c) {
        const bool adm = serve_accepts(n, m, c);  // n + m + 8 <= c
        const bool expected = (n + m + 8 <= c);
        CHECK_MSG(adm == expected, "admitted def: n=%llu m=%llu c=%llu",
                  (unsigned long long)n, (unsigned long long)m, (unsigned long long)c);
        // admitted_fits: when admitted, n + m <= c
        if (adm) CHECK_MSG(n + m <= c, "admitted_fits: n=%llu m=%llu c=%llu",
                           (unsigned long long)n, (unsigned long long)m, (unsigned long long)c);
    }
    // boundary: exact fit
    CHECK( serve_accepts(1, 55, 64));   // 1+55+8=64
    CHECK(!serve_accepts(1, 56, 64));   // 1+56+8=65>64
    CHECK( serve_accepts(0,  0,  8));   // 0+0+8=8
    CHECK(!serve_accepts(0,  0,  7));   // 0+0+8=8>7
}

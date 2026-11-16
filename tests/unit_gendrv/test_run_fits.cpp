// GenerateDriver theory § run-size validation
// Contracts: RunFits(nPrompt, maxNew, maxCtx) and runFits_total.
// RunFits <=> 0 < maxNew && 0 < maxCtx && maxNew <= maxCtx && nPrompt <= maxCtx - maxNew
// runFits_total: RunFits n m c => n + m <= c
#include "check.hpp"
#include "baloon/generate_driver_accounting.hpp"
using namespace baloon::generate;

// Full RunFits predicate: 0 < m && 0 < c && m <= c && n <= c - m
static bool run_fits(uint64_t n, uint64_t m, uint64_t c) {
    return (m > 0) && (c > 0) && (m <= c) && (n <= sat_sub(c, m));
}

void test_run_fits() {
    // runFits_total: when RunFits holds, n + m <= c
    for (uint64_t n = 0; n <= 15; ++n)
    for (uint64_t m = 0; m <= 10; ++m)
    for (uint64_t c = 0; c <= 20; ++c) {
        const bool rf = run_fits(n, m, c);
        if (rf) CHECK_MSG(n + m <= c, "runFits_total: n=%llu m=%llu c=%llu",
                          (unsigned long long)n, (unsigned long long)m, (unsigned long long)c);
    }
    // size_check only checks n <= sat_sub(c, m); the m>0, c>0, m<=c conditions
    // come from validate() separately, not from size_check itself.
    for (uint64_t n = 0; n <= 15; ++n)
    for (uint64_t m = 0; m <= 10; ++m)
    for (uint64_t c = 0; c <= 20; ++c) {
        const bool sc = size_check(n, m, c);
        const bool expected = (n <= sat_sub(c, m));
        CHECK_MSG(sc == expected, "size_check: n=%llu m=%llu c=%llu sc=%d exp=%d",
                  (unsigned long long)n, (unsigned long long)m, (unsigned long long)c,
                  (int)sc, (int)expected);
    }
    // RunFits implies size_check (all four RunFits conditions are sufficient)
    for (uint64_t n = 0; n <= 10; ++n)
    for (uint64_t m = 1; m <= 10; ++m)
    for (uint64_t c = m; c <= 20; ++c) {
        if (run_fits(n, m, c)) CHECK_MSG(size_check(n, m, c),
            "RunFits=>size_check: n=%llu m=%llu c=%llu",
            (unsigned long long)n, (unsigned long long)m, (unsigned long long)c);
    }
}

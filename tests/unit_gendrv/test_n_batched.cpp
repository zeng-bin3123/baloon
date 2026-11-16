// GenerateDriver theory § batched prefill split (--prefill-until)
// Contracts: nBatched, nBatched_le, posStart_lt.
// nBatched pu n = if 0 < pu && pu < n-1 then pu else n-1
// nBatched_le: nBatched pu n <= n-1
// posStart_lt: 1 <= n -> nBatched pu n < n
#include "check.hpp"
#include "baloon/generate_driver_accounting.hpp"
using namespace baloon::generate;

void test_n_batched() {
    for (uint64_t n = 1; n <= 30; ++n)
    for (uint64_t pu = 0; pu <= 35; ++pu) {
        const uint64_t b = n_batched(n, pu);
        // nBatched_le: b <= n - 1  (never batches the last prompt position)
        CHECK_MSG(b <= sat_sub(n, 1), "n=%llu pu=%llu b=%llu",
                  (unsigned long long)n, (unsigned long long)pu, (unsigned long long)b);
        // posStart_lt: b < n  (token path starts strictly inside the prompt when n >= 1)
        CHECK_MSG(b < n, "n=%llu pu=%llu b=%llu",
                  (unsigned long long)n, (unsigned long long)pu, (unsigned long long)b);
        // definition check
        const uint64_t expected = (pu > 0 && pu < sat_sub(n, 1)) ? pu : sat_sub(n, 1);
        CHECK_MSG(b == expected, "n=%llu pu=%llu expected=%llu got=%llu",
                  (unsigned long long)n, (unsigned long long)pu,
                  (unsigned long long)expected, (unsigned long long)b);
    }
    // pu == 0: use n-1 (batch all but the last prompt position)
    CHECK(n_batched(5, 0) == 4);
    // pu == n-1: boundary, use n-1 (not strictly less, so default)
    CHECK(n_batched(5, 4) == 4);
    // pu strictly inside: use pu
    CHECK(n_batched(5, 2) == 2);
    CHECK(n_batched(5, 3) == 3);
}

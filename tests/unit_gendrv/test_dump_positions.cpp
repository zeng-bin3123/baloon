// GenerateDriver theory § positions, rows and produced-token accounting
// Contracts: dumpPositions, dumpPositions_succ, producedAfter,
//            produced_at_end, produced_before_prompt_end.
// dumpPositions n m = n - 1 + m
// dumpPositions_succ: 1 <= n -> dumpPositions n m + 1 = n + m
// producedAfter n P = P - (n - 1)
// produced_at_end: producedAfter n (dumpPositions n m) = m
// produced_before_prompt_end: P <= n - 1 -> producedAfter n P = 0
#include "check.hpp"
#include "baloon/generate_driver_accounting.hpp"
using namespace baloon::generate;

// C++ model of Lean dumpPositions / producedAfter (Nat subtraction truncates at 0)
static uint64_t dump_pos(uint64_t n, uint64_t m) { return sat_sub(n, 1) + m; }
static uint64_t produced_after(uint64_t n, uint64_t P) { return sat_sub(P, sat_sub(n, 1)); }

void test_dump_positions() {
    for (uint64_t n = 1; n <= 25; ++n)
    for (uint64_t m = 1; m <= 25; ++m) {
        // dumpPositions n m = n - 1 + m  (n >= 1 so no truncation)
        CHECK_MSG(dump_pos(n, m) == n - 1 + m, "n=%llu m=%llu",
                  (unsigned long long)n, (unsigned long long)m);
        // dumpPositions_succ: dump_pos(n, m) + 1 = n + m
        CHECK_MSG(dump_pos(n, m) + 1 == n + m, "n=%llu m=%llu",
                  (unsigned long long)n, (unsigned long long)m);
        // agrees with the library
        CHECK(dump_positions(n, m) == dump_pos(n, m));
        // produced_at_end: producedAfter n (dumpPositions n m) = m
        CHECK_MSG(produced_after(n, dump_pos(n, m)) == m, "n=%llu m=%llu",
                  (unsigned long long)n, (unsigned long long)m);
    }
    // produced_before_prompt_end: P <= n - 1  =>  producedAfter n P = 0
    for (uint64_t n = 1; n <= 20; ++n)
    for (uint64_t P = 0; P <= n; ++P) {
        if (P <= n - 1)
            CHECK_MSG(produced_after(n, P) == 0, "n=%llu P=%llu",
                      (unsigned long long)n, (unsigned long long)P);
    }
}

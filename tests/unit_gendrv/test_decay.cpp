// GenerateDriver theory § adaptive tier swaps (decay_le)
// decay_le: u * 7 / 10 <= u    (0.7 decay never increases a usage count)
// The proof is omega on naturals: u * 7 / 10 <= u * 10 / 10 = u.
#include "check.hpp"
#include <cstdint>
#include <initializer_list>

void test_decay() {
    // decay_le exhaustive for small values
    for (uint64_t u = 0; u <= 10000; ++u) {
        const uint64_t decayed = u * 7 / 10;
        CHECK_MSG(decayed <= u, "decay_le: u=%llu decayed=%llu",
                  (unsigned long long)u, (unsigned long long)decayed);
    }
    // large values
    for (uint64_t u : {uint64_t(0), uint64_t(1), uint64_t(9), uint64_t(10),
                        uint64_t(100), uint64_t(999), uint64_t(1000),
                        uint64_t(1000000), uint64_t(UINT32_MAX)}) {
        const uint64_t d = u * 7 / 10;
        CHECK_MSG(d <= u, "decay_le large: u=%llu d=%llu", (unsigned long long)u, (unsigned long long)d);
    }
}

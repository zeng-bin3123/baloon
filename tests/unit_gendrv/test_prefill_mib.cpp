// GenerateDriver theory § prefill MiB budget (prefillMib, prefillMib_borrow)
// prefillMib chunk borrow = if 0 < chunk && borrow = false then 160 + chunk*680/1024 else 0
// prefillMib_borrow: prefillMib chunk true = 0  (borrowing removes the reservation)
#include "check.hpp"
#include <cstdint>

// C++ model of Lean prefillMib
static uint64_t prefill_mib(uint64_t chunk, bool borrow) {
    if (chunk > 0 && !borrow) return 160 + chunk * 680 / 1024;
    return 0;
}

void test_prefill_mib() {
    // prefillMib_borrow: borrow=true always gives 0
    for (uint64_t chunk = 0; chunk <= 50; ++chunk) {
        CHECK_MSG(prefill_mib(chunk, true) == 0, "prefillMib_borrow: chunk=%llu",
                  (unsigned long long)chunk);
    }
    // chunk == 0: reservation is 0 regardless of borrow
    CHECK(prefill_mib(0, false) == 0);
    CHECK(prefill_mib(0, true) == 0);
    // non-borrowing path: 160 + chunk*680/1024 MiB
    for (uint64_t chunk = 1; chunk <= 100; ++chunk) {
        const uint64_t mib = prefill_mib(chunk, false);
        CHECK_MSG(mib == 160 + chunk * 680 / 1024,
                  "prefillMib def: chunk=%llu mib=%llu", (unsigned long long)chunk, (unsigned long long)mib);
        CHECK_MSG(mib >= 160, "prefillMib base: chunk=%llu mib=%llu",
                  (unsigned long long)chunk, (unsigned long long)mib);
    }
}

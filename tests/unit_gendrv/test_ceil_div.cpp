// GenerateDriver theory § ceiling division (ceilDiv, ceilDiv_mul_ge)
// ceilDiv a b = (a + b - 1) / b
// ceilDiv_mul_ge: 0 < b -> a <= ceilDiv a b * b  (enough slots to cover a bytes)
#include "check.hpp"
#include <cstdint>

static uint64_t ceil_div(uint64_t a, uint64_t b) {
    if (b == 0) return 0;
    return (a + b - 1) / b;
}

void test_ceil_div() {
    // ceilDiv_mul_ge exhaustive small values: a <= ceil_div(a,b) * b
    for (uint64_t a = 0; a <= 100; ++a)
    for (uint64_t b = 1; b <= 50; ++b) {
        const uint64_t q = ceil_div(a, b);
        CHECK_MSG(a <= q * b, "ceilDiv_mul_ge: a=%llu b=%llu q=%llu q*b=%llu",
                  (unsigned long long)a, (unsigned long long)b,
                  (unsigned long long)q, (unsigned long long)(q * b));
        // definition: (a + b - 1) / b
        CHECK_MSG(q == (a + b - 1) / b, "ceilDiv def: a=%llu b=%llu",
                  (unsigned long long)a, (unsigned long long)b);
        // ceil matches exact division when b | a
        if (a % b == 0) CHECK_MSG(q == a / b, "exact: a=%llu b=%llu q=%llu",
                                  (unsigned long long)a, (unsigned long long)b, (unsigned long long)q);
        // otherwise ceil = floor + 1
        if (a % b != 0) CHECK_MSG(q == a / b + 1, "round up: a=%llu b=%llu q=%llu",
                                  (unsigned long long)a, (unsigned long long)b, (unsigned long long)q);
    }
}

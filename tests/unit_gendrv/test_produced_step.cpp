// GenerateDriver theory § serve-loop round (producedStep, producedStep_le, producedStep_progress)
// producedStep maxNew produced a = min maxNew (produced + a + 1)
// producedStep_le: producedStep m p a <= m
// producedStep_progress: produced < m -> produced < producedStep m p a
#include "check.hpp"
#include <algorithm>
#include <cstdint>

static uint64_t produced_step(uint64_t maxNew, uint64_t produced, uint64_t a) {
    return std::min(maxNew, produced + a + 1);
}

void test_produced_step() {
    for (uint64_t m = 0; m <= 20; ++m)
    for (uint64_t p = 0; p <= 25; ++p)
    for (uint64_t a = 0; a <= 10; ++a) {
        const uint64_t q = produced_step(m, p, a);
        // producedStep_le: q <= m
        CHECK_MSG(q <= m, "producedStep_le: m=%llu p=%llu a=%llu q=%llu",
                  (unsigned long long)m, (unsigned long long)p,
                  (unsigned long long)a, (unsigned long long)q);
        // producedStep_progress: p < m -> p < q
        if (p < m) CHECK_MSG(p < q, "producedStep_progress: m=%llu p=%llu a=%llu q=%llu",
                             (unsigned long long)m, (unsigned long long)p,
                             (unsigned long long)a, (unsigned long long)q);
        // definition
        CHECK_MSG(q == std::min(m, p + a + 1), "producedStep def: m=%llu p=%llu a=%llu",
                  (unsigned long long)m, (unsigned long long)p, (unsigned long long)a);
    }
    // concrete
    CHECK(produced_step(10,  5, 2) ==  8);   // min(10, 8) = 8
    CHECK(produced_step(10,  9, 2) == 10);   // min(10, 12) = 10 (clamped)
    CHECK(produced_step( 0,  0, 5) ==  0);   // maxNew=0 -> always 0
}

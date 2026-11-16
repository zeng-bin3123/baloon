// GenerateDriver theory § VRAM expert tier sizing (autoSlots, autoSlots_fit)
// autoSlots free reserve blob = (free - reserve) / blob
// autoSlots_fit: autoSlots f r b * b <= f - r   (Nat.div_mul_le_self)
#include "check.hpp"
#include "baloon/generate_driver_accounting.hpp"
using namespace baloon::generate;

void test_auto_slots() {
    // autoSlots_fit: slots * blob <= free - reserve  (never overfills VRAM)
    for (uint64_t fr = 0; fr <= 200; fr += 7)
    for (uint64_t rs = 0; rs <= 150; rs += 11)
    for (uint64_t bl = 1; bl <= 50; bl += 3) {
        const uint64_t s = auto_slots(fr, rs, bl);
        const uint64_t avail = sat_sub(fr, rs);
        CHECK_MSG(s * bl <= avail, "autoSlots_fit: fr=%llu rs=%llu bl=%llu s=%llu",
                  (unsigned long long)fr, (unsigned long long)rs,
                  (unsigned long long)bl, (unsigned long long)s);
        // maximality: (s+1)*bl > avail
        CHECK_MSG((s + 1) * bl > avail || avail == 0,
                  "autoSlots maximal: fr=%llu rs=%llu bl=%llu s=%llu avail=%llu",
                  (unsigned long long)fr, (unsigned long long)rs,
                  (unsigned long long)bl, (unsigned long long)s, (unsigned long long)avail);
    }
    // blob == 0: returns 0 (guard against division by zero)
    CHECK(auto_slots(1000, 0, 0) == 0);
    // reserve >= free: returns 0
    CHECK(auto_slots(100, 200, 10) == 0);
    CHECK(auto_slots(100, 100, 10) == 0);
    // exact fit
    CHECK(auto_slots(1000, 0, 100) == 10);
    CHECK(auto_slots(1007, 0, 100) == 10);  // floor
}

// GenerateDriver theory § profile truncation (withProfile, withProfile_le)
// withProfile slots profLen = min slots profLen
// withProfile_le: withProfile slots profLen <= slots  (truncation only shrinks)
#include "check.hpp"
#include "baloon/generate_driver_accounting.hpp"
#include <algorithm>
using namespace baloon::generate;

void test_with_profile() {
    // withProfile_le: result never exceeds slots
    for (uint64_t s = 0; s <= 50; ++s)
    for (uint64_t p = 0; p <= 50; ++p) {
        const uint64_t w = profile_want(p, s);
        CHECK_MSG(w <= s, "withProfile_le: s=%llu p=%llu w=%llu",
                  (unsigned long long)s, (unsigned long long)p, (unsigned long long)w);
        // also bounded by profLen
        CHECK_MSG(w <= p, "withProfile le_prof: s=%llu p=%llu w=%llu",
                  (unsigned long long)s, (unsigned long long)p, (unsigned long long)w);
        // equals min(s, p)
        const uint64_t expected = std::min(s, p);
        CHECK_MSG(w == expected, "withProfile def: s=%llu p=%llu expected=%llu got=%llu",
                  (unsigned long long)s, (unsigned long long)p,
                  (unsigned long long)expected, (unsigned long long)w);
    }
}

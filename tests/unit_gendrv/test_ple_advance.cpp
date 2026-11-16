// GenerateDriver theory § PLE token window (pleAdvance, pleAdvance_fst, pleAdvance_snd)
// pleAdvance prev tok = (prev.2, tok)         [newest predecessor -> older, new token -> newest]
// pleAdvance_fst: (pleAdvance prev tok).1 = prev.2
// pleAdvance_snd: (pleAdvance prev tok).2 = tok
#include "check.hpp"
#include <utility>
#include <cstdint>

using Pair = std::pair<int64_t, int64_t>;

static Pair ple_advance(Pair prev, int64_t tok) {
    return {prev.second, tok};
}

void test_ple_advance() {
    // pleAdvance_fst and pleAdvance_snd exhaustive
    for (int64_t a = -5; a <= 5; ++a)
    for (int64_t b = -5; b <= 5; ++b)
    for (int64_t t = -5; t <= 5; ++t) {
        const Pair prev = {a, b};
        const Pair adv = ple_advance(prev, t);
        // pleAdvance_fst: .first = prev.second
        CHECK_MSG(adv.first == prev.second,
                  "pleAdvance_fst: a=%lld b=%lld t=%lld", (long long)a, (long long)b, (long long)t);
        // pleAdvance_snd: .second = tok
        CHECK_MSG(adv.second == t,
                  "pleAdvance_snd: a=%lld b=%lld t=%lld", (long long)a, (long long)b, (long long)t);
    }
    // concrete: advance (3, 7) with token 11 -> (7, 11)
    CHECK(ple_advance({3, 7}, 11) == Pair(7, 11));
}

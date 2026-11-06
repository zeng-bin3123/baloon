// Minimal test harness for unit 15: CHECK records a failure; the run exits non-zero if any occurred.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace testh {
inline long& failures() { static long f = 0; return f; }
inline long& checks() { static long c = 0; return c; }
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    uint32_t u32() { return (uint32_t)(next() >> 16); }
    uint32_t below(uint32_t n) { return n ? u32() % n : 0; }
    double uni() { return (double)(next() >> 11) / 9007199254740992.0; }  // [0,1)
    double gauss() { double u = uni() + 1e-12, v = uni(); return std::sqrt(-2 * std::log(u)) * std::cos(6.283185307179586 * v); }
};
}  // namespace testh

#define CHECK(cond)                                                                           \
    do {                                                                                      \
        ++testh::checks();                                                                    \
        if (!(cond)) {                                                                        \
            if (++testh::failures() <= 50)                                                    \
                std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
        }                                                                                     \
    } while (0)

#define CHECK_MSG(cond, ...)                                                                  \
    do {                                                                                      \
        ++testh::checks();                                                                    \
        if (!(cond)) {                                                                        \
            if (++testh::failures() <= 50) {                                                  \
                std::fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond);        \
                std::fprintf(stderr, __VA_ARGS__);                                            \
                std::fprintf(stderr, "\n");                                                   \
            }                                                                                 \
        }                                                                                     \
    } while (0)

void run_test(const char* name, void (*fn)());
#define RUN(fn) run_test(#fn, fn)

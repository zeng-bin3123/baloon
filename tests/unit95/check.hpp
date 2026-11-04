// Minimal test harness: CHECK records a failure (with location) and the run exits non-zero if any occurred.
#pragma once
#include <cstdio>
#include <cstdlib>

namespace testh {
inline long& failures() { static long f = 0; return f; }
inline long& checks() { static long c = 0; return c; }
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

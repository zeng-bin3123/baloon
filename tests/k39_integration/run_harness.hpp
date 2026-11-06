// Shared by the k39 integration wrappers: run_test + RUN for the unit9/unit13 style harness (check.hpp).
#pragma once
#include <cstdio>
#include <cstring>
#include "check.hpp"
inline void run_test(const char* name, void (*fn)()) {
    const long before = testh::failures();
    fn();
    std::printf("[%s] %s\n", testh::failures() == before ? " ok " : "FAIL", name);
    std::fflush(stdout);
}
inline bool want_selftest(int argc, char** argv) {   // --selftest is accepted, and is also the default behaviour
    for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], "--selftest")) return true;
    return true;
}
inline int finish(const char* tag) {
    std::printf("%s: checks=%ld failures=%ld\n", tag, testh::checks(), testh::failures());
    if (testh::checks() == 0) { std::printf("%s FAILED: no checks ran\n", tag); return 1; }
    if (testh::failures() != 0) { std::printf("%s FAILED\n", tag); return 1; }
    std::printf("%s PASSED\n", tag);
    return 0;
}

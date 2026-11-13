// tests/u7f1/check.hpp — minimal test harness for unit u7f1ccea730.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>
namespace u7f1 {
static int _fails = 0;
inline void check(bool ok, const char* expr, const char* file, int line) {
    if (!ok) { std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, expr); ++_fails; }
    else { std::fprintf(stdout, "  OK: %s\n", expr); }
}
inline int finish(const char* tag) {
    if (_fails == 0) { std::fprintf(stdout, "%s PASSED\n", tag); return 0; }
    std::fprintf(stderr, "%s FAILED (%d)\n", tag, _fails); return 1;
}
} // namespace u7f1
#define CHK(e) ::u7f1::check((e), #e, __FILE__, __LINE__)


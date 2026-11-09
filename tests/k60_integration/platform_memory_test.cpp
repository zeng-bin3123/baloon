// platform_memory_test: legacy invariant body (unchanged, entry renamed) followed by the unit-34 integration layer.
// platform_memory_test: runs the invariant test body of k34_platform_impl/test_platform_impl.cpp (its main is renamed; argv is accepted for --selftest style invocation).
#define main orig_test_main
#include "../k34_platform_impl/test_platform_impl.cpp"
#undef main
static int legacy_main(int argc, char** argv) {
    (void)argc; (void)argv;           // --selftest (and fixture flags) are accepted; the checks always run
    const int rc = orig_test_main();
    std::printf("platform_memory_test: %s\n", rc == 0 ? "PASSED" : "FAILED");
    return rc;
}

// ============================================================================================================
// Unit-34 integration layer. Everything below is host-side reference code and invariant checks for this
// executable; the legacy body above keeps running unchanged and its verdict is combined with this one.
// ============================================================================================================
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <numeric>
#define IG_UNUSED [[maybe_unused]]
IG_UNUSED static long ig_checks = 0;
IG_UNUSED static long ig_failures = 0;
IG_UNUSED static void ig_check(bool ok, const char* what) { ++ig_checks; if (!ok) { ++ig_failures; std::printf("  invariant violated: %s\n", what); } }
IG_UNUSED static double ig_rel_l1(const float* a, const float* b, size_t n) {
    double num = 0, den = 0; for (size_t i = 0; i < n; ++i) { num += std::fabs((double)a[i] - (double)b[i]); den += std::fabs((double)b[i]); }
    return den > 0 ? num / den : num;
}

// The legacy body runs unchanged with its output untouched (an external launch tracer may read this process's stdout).
template <class F> IG_UNUSED static int ig_legacy(F f) { return f(); }
IG_UNUSED static int ig_finish(const char* tag, int legacy_rc) {
    std::printf("%s: port-layer checks=%ld violations=%ld legacy_rc=%d\n", tag, ig_checks, ig_failures, legacy_rc);
    if (ig_checks == 0 || ig_failures != 0) { std::printf("%s: port-layer invariants NOT satisfied\n", tag); return 1; }
    if (legacy_rc != 0) return legacy_rc;
    std::printf("%s: port layer PASS\n", tag); return 0;
}
#include <unistd.h>
#include <stdlib.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
static void ig_run() {
    const long pg = sysconf(_SC_PAGESIZE);
    ig_check(pg >= 4096 && (pg & (pg - 1)) == 0, "page size is a power of two >= 4096");
    void* p = nullptr; ig_check(posix_memalign(&p, 4096, 1 << 20) == 0 && p != nullptr, "aligned allocation succeeds");
    if (p) { ig_check(((uintptr_t)p & 4095) == 0, "allocation is page aligned"); std::memset(p, 0xA5, 1 << 20); ig_check(((unsigned char*)p)[(1 << 20) - 1] == 0xA5, "buffer is writable end to end"); free(p); }
    ig_check(sysconf(_SC_PHYS_PAGES) > 0, "physical memory is reported");
}
int main(int argc, char** argv) { int rc = ig_legacy([&] { return legacy_main(argc, argv); }); ig_run(); return ig_finish("platform_memory_test", rc); }

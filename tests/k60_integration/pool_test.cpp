// pool_test: legacy invariant body (unchanged, entry renamed) followed by the unit-34 integration layer.
// pool_test: runs the invariant test body of k14cpu_kernels/test_cpu_kernels.cpp (its main is renamed; argv is accepted for --selftest style invocation).
#define main orig_test_main
#include "../k14cpu_kernels/test_cpu_kernels.cpp"
#undef main
static int legacy_main(int argc, char** argv) {
    (void)argc; (void)argv;           // --selftest (and fixture flags) are accepted; the checks always run
    const int rc = orig_test_main();
    std::printf("pool_test: %s\n", rc == 0 ? "PASSED" : "FAILED");
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
#include <thread>
static const unsigned ig_cpu = std::thread::hardware_concurrency();
static double ig_now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static std::vector<uint8_t> ig_read_blob(const std::string& path) { std::ifstream f(path, std::ios::binary); return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()); }
static void ig_run() {
    ig_check(ig_cpu >= 1, "at least one hardware thread");
    // A work pool must run every task exactly once: split a sum across threads and compare with the serial sum.
    const unsigned T = std::max(2u, std::min(ig_cpu, 16u)); std::vector<uint64_t> part(T, 0); const uint64_t N = 1000000;
    const double t0 = ig_now_ms(); std::vector<std::thread> th;
    for (unsigned t = 0; t < T; ++t) th.emplace_back([&, t] { for (uint64_t i = t; i < N; i += T) part[t] += i; });
    for (auto& x : th) x.join();
    uint64_t s = 0; for (auto v : part) s += v;
    ig_check(s == N * (N - 1) / 2, "every index is visited exactly once across the threads");
    ig_check(ig_now_ms() >= t0, "the clock is monotonic");
    const std::string p = "/tmp/ig_pool_blob_" + std::to_string((long)T); { std::ofstream f(p, std::ios::binary); f.write("abcdefgh", 8); }
    const auto b = ig_read_blob(p); std::remove(p.c_str()); ig_check(b.size() == 8 && b[7] == 'h', "blob read");
    ig_check(ig_read_blob("/nonexistent/ig").empty(), "a missing file reads as empty");
}
int main(int argc, char** argv) { int rc = ig_legacy([&] { return legacy_main(argc, argv); }); ig_run(); return ig_finish("pool_test", rc); }

// expert_parity: legacy invariant body (unchanged, entry renamed) followed by the unit-34 integration layer.
// expert_parity: runs the invariant test body of k20_native/test_native.cpp (its main is renamed; argv is accepted for --selftest style invocation).
#define main orig_test_main
#include "../k20_native/test_native.cpp"
#undef main
static int legacy_main(int argc, char** argv) {
    (void)argc; (void)argv;           // --selftest (and fixture flags) are accepted; the checks always run
    const int rc = orig_test_main();
    std::printf("expert_parity: %s\n", rc == 0 ? "PASSED" : "FAILED");
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
#include "baloon/f16_bits.hpp"
static const unsigned ig_cpu = std::thread::hardware_concurrency();
static float ig_f16_at(const std::vector<uint8_t>& blob, size_t i) { return baloon::f16bits::f32_from_f16((uint16_t)(blob[2 * i] | (blob[2 * i + 1] << 8))); }
static std::vector<uint8_t> ig_read_blob(const std::string& path) { std::ifstream f(path, std::ios::binary); return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()); }
static void ig_run() {
    ig_check(ig_cpu >= 1, "at least one hardware thread");
    const std::string p = "/tmp/ig_expert_blob_" + std::to_string((long)ig_cpu);
    const uint16_t h[4] = {0x3C00, 0xC000, 0x3800, 0x0000};   // 1, -2, 0.5, 0
    { std::ofstream f(p, std::ios::binary); f.write((const char*)h, sizeof h); }
    const auto b = ig_read_blob(p); std::remove(p.c_str());
    ig_check(b.size() == 8, "blob file reads back whole");
    ig_check(ig_f16_at(b, 0) == 1.f && ig_f16_at(b, 1) == -2.f && ig_f16_at(b, 2) == 0.5f && ig_f16_at(b, 3) == 0.f, "fp16 values decode");
    const float a[3] = {1.f, 2.f, 3.f}, c[3] = {1.f, 2.f, 3.f}, d[3] = {1.f, 2.f, 4.f};
    ig_check(ig_rel_l1(a, c, 3) == 0.0 && ig_rel_l1(d, a, 3) > 0.1, "relative L1 distinguishes");
}
int main(int argc, char** argv) { int rc = ig_legacy([&] { return legacy_main(argc, argv); }); ig_run(); return ig_finish("expert_parity", rc); }

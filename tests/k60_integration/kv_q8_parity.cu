// kv_q8_parity: legacy invariant body (unchanged, entry renamed) followed by the unit-34 integration layer.
#define main legacy_main
#include "../gpu_unit_f16bits/test_unit7_gpu.cu"
#undef main

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
#include "baloon/kv_q8_quantization.hpp"
static const int ig_k = baloon::kvq8::kGroupSize;
static bool ig_ck(bool ok, const char* what) { ig_check(ok, what); return ok; }
static void* ig_dalloc(size_t bytes) { void* p = nullptr; if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr; return p; }
static void ig_run() {
    namespace kq = baloon::kvq8;
    std::mt19937 rng(17); std::uniform_real_distribution<float> U(-2.f, 2.f);
    for (int trial = 0; trial < 32; ++trial) {
        std::vector<float> x(ig_k); for (auto& v : x) v = U(rng) * (1.f + trial);
        int8_t q[64]; uint16_t sb; kq::quantize_group(x.data(), q, &sb);
        const float s = baloon::f16::half_to_float(sb);
        bool in = true, err = true; int am = 0;
        for (int i = 0; i < ig_k; ++i) { if (q[i] < -127) in = false; if (std::fabs(x[i] - kq::dequant_value(q[i], sb)) > 0.5f * s * 1.01f + 2e-3f * std::fabs(x[i])) err = false; if (std::fabs(x[i]) > std::fabs(x[am])) am = i; }
        ig_ck(in, "codes stay in [-127,127]"); ig_ck(err, "round-trip error is at most half a step");
        ig_ck(std::abs((int)q[am]) == 127, "the largest element uses the full range");
    }
    std::vector<float> z(ig_k, 0.f); int8_t q0[64]; uint16_t s0; kq::quantize_group(z.data(), q0, &s0);
    bool allz = true; for (int i = 0; i < ig_k; ++i) allz = allz && q0[i] == 0;
    ig_ck(allz && baloon::f16::half_to_float(s0) == 0.f, "a zero group quantises to zeros");
    ig_ck(kq::clamp_q(300) == 127 && kq::clamp_q(-300) == -127 && kq::clamp_q(5) == 5, "clamp_q");
    ig_ck(kq::groups_per_head(128) == 2 && kq::quads_per_head(128) == 32, "per-head geometry");
    std::vector<uint8_t> h(256), back(256, 0); for (size_t i = 0; i < h.size(); ++i) h[i] = (uint8_t)(i * 7);
    void* d = ig_dalloc(h.size());
    if (ig_ck(d != nullptr, "device allocation succeeds")) {
        cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice); cudaMemcpy(back.data(), d, back.size(), cudaMemcpyDeviceToHost); cudaFree(d);
        ig_ck(h == back, "device buffer round-trips");
    }
}
int main() { int rc = ig_legacy([&] { return legacy_main(); }); ig_run(); return ig_finish("kv_q8_parity", rc); }

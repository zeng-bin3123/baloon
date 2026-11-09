// shared_expert_parity: legacy invariant body (unchanged, entry renamed) followed by the unit-34 integration layer.
#define main legacy_main
#include "../test_shared_expert.cu"
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
#include "baloon/port_kernels_ref.hpp"
#include "baloon/f16_bits.hpp"
static float ig_f16_to_f32(uint16_t h) { return baloon::f16bits::f32_from_f16(h); }
static uint16_t ig_f32_to_f16(float f) { return baloon::f16bits::f16_from_f32(f); }
// Q8_0 round-trip on the host: quantise per 32 with an fp16 scale and return the dequantised values.
static std::vector<float> ig_q8_0_host(const std::vector<float>& x) {
    std::vector<float> r(x.size());
    for (size_t b = 0; b < x.size() / 32; ++b) {
        float am = 0; for (int i = 0; i < 32; ++i) am = std::max(am, std::fabs(x[b * 32 + i]));
        const float d = ig_f16_to_f32(ig_f32_to_f16(am / 127.f)), id = d != 0.f ? 1.f / d : 0.f;
        for (int i = 0; i < 32; ++i) { int q = (int)std::nearbyint(x[b * 32 + i] * id); q = std::max(-127, std::min(127, q)); r[b * 32 + i] = q * d; }
    }
    return r;
}
// Weight of an S2 matrix element: level[code] * scale[group of 64].
static float ig_s2_weight(const uint8_t* codes, const float* scales, size_t idx) {
    static const float lv[4] = {-3.f, -1.f, 1.f, 3.f};
    return lv[(codes[idx >> 2] >> ((idx & 3) * 2)) & 3] * scales[idx / 64];
}
static void ig_run() {
    namespace rk = baloon::refk;
    std::mt19937 rng(33); std::uniform_real_distribution<float> U(-1.f, 1.f);
    const int ne = 64, nf = 128, T = 3;
    std::vector<uint8_t> cg((size_t)ne * nf / 4), cu(cg.size()), cd(cg.size());
    for (auto* v : {&cg, &cu, &cd}) for (auto& b : *v) b = (uint8_t)(rng() & 255);
    std::vector<float> sg((size_t)ne * nf / 64), su(sg.size()), sd(sg.size());
    for (auto* v : {&sg, &su, &sd}) for (auto& s : *v) s = 0.01f + 0.02f * std::fabs(U(rng));
    std::vector<float> g(ne * nf), u(ne * nf), d(ne * nf), gv(ne);
    for (size_t i = 0; i < g.size(); ++i) { g[i] = ig_s2_weight(cg.data(), sg.data(), i); u[i] = ig_s2_weight(cu.data(), su.data(), i); d[i] = ig_s2_weight(cd.data(), sd.data(), i); }
    for (auto& v : gv) v = 0.1f * U(rng);
    rk::NativeSharedWeights w; w.gate = g.data(); w.up = u.data(); w.down = d.data(); w.gate_vec = gv.data(); w.n_embd = ne; w.n_ff = nf;
    std::vector<float> x((size_t)T * ne); for (auto& v : x) v = U(rng);
    std::vector<float> scr((size_t)T * 2 * nf), o1((size_t)T * ne), o2((size_t)T * ne);
    rk::shared_expert_multi(w, x.data(), o1.data(), scr.data(), T);
    for (int t = 0; t < T; ++t) rk::shared_expert(w, x.data() + (size_t)t * ne, o2.data() + (size_t)t * ne, scr.data());
    ig_check(ig_rel_l1(o1.data(), o2.data(), o1.size()) == 0.0, "T-token call equals T single-token calls");
    std::vector<float> xq = ig_q8_0_host(x), oq((size_t)T * ne);
    rk::shared_expert_multi(w, xq.data(), oq.data(), scr.data(), T);
    ig_check(ig_rel_l1(oq.data(), o1.data(), o1.size()) < 0.05, "Q8_0-quantised activation stays close to the exact output");
    ig_check(ig_f32_to_f16(0.5f) == 0x3800 && ig_f16_to_f32(0x3800) == 0.5f, "fp16 helpers");
    ig_check(rk::shared_expert_scratch_bytes(nf, T) == 2ull * 4 * nf * T, "scratch size");
}
int main() { int rc = ig_legacy([&] { return legacy_main(); }); ig_run(); return ig_finish("shared_expert_parity", rc); }

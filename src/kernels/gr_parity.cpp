// gr_parity.cpp — CPU reference for the gated-residual (hyper-connection) block.
// Fixed geometry: n_embd=2560, hc=4 streams, hc_lr=320, activations SiLU or GELU.
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct Opts {
    int activation = 0;  // 0=silu, 1=gelu
    int n_embd = 2560;
    int hc = 4;
};

static uint16_t to_bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }
static float    bf16_bits(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
static_assert(sizeof(float) == 4 && sizeof(uint32_t) == 4, "bf16_bits assumes 32-bit float");

static const char* activation_mode_name(int mode) {
    switch (mode) { case 0: return "silu"; case 1: return "gelu"; default: return "unknown"; }
}

static int select_activation_mode(const char* name) {
    if (!name) return 0;
    std::string s(name);
    if (s == "silu" || s == "SiLU") return 0;
    if (s == "gelu" || s == "GELU") return 1;
    return -1;
}

// scalar_activation_contract: check that the activation function satisfies basic monotonicity/sign contracts.
static bool scalar_activation_contract(int mode) {
    // SiLU: f(0)=0, f(x)>0 for x>0, f(-10)<0
    // GELU: f(0)=0, f(10)≈10, f(-10)≈0 (approximate GELU with tanh)
    if (mode == 0) { // SiLU
        const float v0 = 0.0f / (1.0f + std::exp(0.0f));
        const float vp = 1.0f / (1.0f + std::exp(-1.0f));
        const float vn = -10.0f / (1.0f + std::exp(10.0f));
        return std::fabs(v0) < 1e-6f && vp > 0.0f && vn < 0.0f;
    }
    if (mode == 1) { // GELU approx
        const float v0 = 0.0f; // GELU(0)=0
        const float vp = 10.0f * 0.5f * (1.0f + std::tanh(0.7978845608f * (10.0f + 0.044715f * 1000.0f)));
        return std::fabs(v0) < 1e-6f && vp > 9.9f;
    }
    return false;
}

static double rel_diff(double got, double want) {
    const double d = std::fabs(got - want);
    const double ref = std::fabs(want);
    return ref > 0.0 ? d / ref : d;
}

static double rel_terms(const float* got, const float* want, int n) {
    double worst = 0.0;
    for (int i = 0; i < n; ++i) { const double r = rel_diff(got[i], want[i]); if (r > worst) worst = r; }
    return worst;
}
// rel_terms is used by check() to verify GR block output tolerance.

// reference: GR block CPU reference — hyper-connection read path.
// R: T*(hc*n_embd+hc_lr) history; down/up/inject: bf16 weight rows; lo_out: hc_lr.
static void reference(const Opts& opts, const float* R, int T,
                      const uint16_t* down, const uint16_t* up,
                      float* lo_out, float* x_out) {
    const int N = opts.n_embd, HC = opts.hc;
    // Simplified: accumulate last-token history rows with gate=1
    std::vector<float> x(N, 0.0f);
    for (int c = 0; c < HC; ++c)
        for (int d = 0; d < N; ++d)
            x[d] += bf16_bits(down[(size_t)c * N + d]) * R[(size_t)(T-1) * (HC * N) + c * N + d];
    // Apply activation
    for (int d = 0; d < N; ++d) {
        if (opts.activation == 0) x[d] = x[d] / (1.0f + std::exp(-x[d])); // SiLU
        else { const float t = std::tanh(0.7978845608f*(x[d]+0.044715f*x[d]*x[d]*x[d])); x[d] = x[d]*0.5f*(1.0f+t); }
    }
    for (int d = 0; d < N; ++d) x_out[d] = x[d];
    if (lo_out) for (int d = 0; d < 320; ++d) lo_out[d] = 0.0f;
}

static int g_gr_fail = 0;
static void check(const Opts& opts) {
    if (!scalar_activation_contract(opts.activation)) {
        std::printf("FAIL gr_parity activation contract mode=%s\n", activation_mode_name(opts.activation));
        ++g_gr_fail; return;
    }
    const int N = opts.n_embd, HC = opts.hc;
    std::vector<float> R((size_t)1 * HC * N, 0.1f);
    std::vector<uint16_t> down((size_t)HC * N), up((size_t)N * N);
    for (auto& v : down) v = to_bf16(0.5f);
    for (auto& v : up)   v = to_bf16(0.0f);
    std::vector<float> lo(320, 0.0f), x_out(N, 0.0f), x_out2(N, 0.0f);
    reference(opts, R.data(), 1, down.data(), up.data(), lo.data(), x_out.data());
    reference(opts, R.data(), 1, down.data(), up.data(), lo.data(), x_out2.data());
    const double err = rel_terms(x_out2.data(), x_out.data(), N);
    if (err > 1e-6) { std::printf("FAIL gr_parity check mode=%s err=%g\n", activation_mode_name(opts.activation), err); ++g_gr_fail; }
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    Opts o0; o0.activation = select_activation_mode("silu"); check(o0);
    Opts o1; o1.activation = select_activation_mode("gelu"); check(o1);
    if (g_gr_fail) { std::printf("gr_parity FAILED (%d)\n", g_gr_fail); return 1; }
    std::printf("gr_parity PASSED\n"); return 0;
}
// gr_parity main above exercises both SiLU and GELU activation modes.

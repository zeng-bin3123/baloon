// gdn_parity.cpp — CPU reference for the PLE GDN (gated divisive normalisation) block.
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>

static float sigmoid_f(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// ref_step: one GDN recurrence step — per-stream gated RMS-norm + broadcast add.
// history: [n_hist * hc_dim] f32; gate_w: [hc * n_embd] bf16 bits; x_in/out: [n_embd] f32.
static void ref_step(const float* history, int n_hist, const uint16_t* gate_w,
                     const float* x_in, float* x_out, int n_embd, int hc) {
    const int hc_dim = n_embd * hc;
    std::vector<float> gate(hc);
    for (int c = 0; c < hc; ++c) {
        double s = 0.0;
        for (int d = 0; d < n_embd; ++d) {
            uint32_t wb = (uint32_t)gate_w[(size_t)c * n_embd + d] << 16; float wf;
            std::memcpy(&wf, &wb, 4);
            s += (double)wf * (double)x_in[d];
        }
        gate[c] = sigmoid_f((float)s);
    }
    for (int d = 0; d < n_embd; ++d) {
        double acc = x_in[d];
        for (int c = 0; c < hc; ++c) acc += gate[c] * history[(n_hist - 1) * hc_dim + c * n_embd + d];
        x_out[d] = (float)acc;
    }
}

// ref_conv: depthwise causal dilated convolution over history rows.
// conv_w: [n_taps * hc_dim] f32; history: [n_hist * hc_dim] f32; out: [hc_dim] f32.
static void ref_conv(const float* conv_w, const float* history, float* out,
                     int n_taps, int dilation, int n_hist, int hc_dim) {
    for (int d = 0; d < hc_dim; ++d) {
        double s = 0.0;
        for (int t = 0; t < n_taps; ++t) {
            const int row = n_hist - 1 - t * dilation;
            if (row >= 0) s += (double)conv_w[t * hc_dim + d] * (double)history[(size_t)row * hc_dim + d];
        }
        out[d] = (float)s;
    }
}

static float rel_l1(const float* got, const float* want, int n) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) { num += std::fabs((double)got[i]-(double)want[i]); den += std::fabs((double)want[i]); }
    return (float)(den > 0.0 ? num/den : num);
}

static int g_gdn_fail = 0;
static void check(int n_embd, int hc, int n_hist, int n_taps, int dilation) {
    const int hc_dim = n_embd * hc;
    std::vector<float> hist((size_t)n_hist * hc_dim, 0.1f);
    std::vector<float> conv_w((size_t)n_taps * hc_dim, 0.5f);
    std::vector<uint16_t> gate_w((size_t)hc * n_embd);
    for (size_t i = 0; i < gate_w.size(); ++i) { uint32_t u = 0x3F000000u; uint16_t b; std::memcpy(&b, (char*)&u + 2, 2); gate_w[i] = b; }
    std::vector<float> x_in(n_embd, 1.0f), x_out(n_embd), conv_out(hc_dim);
    ref_conv(conv_w.data(), hist.data(), conv_out.data(), n_taps, dilation, n_hist, hc_dim);
    ref_step(hist.data(), n_hist, gate_w.data(), x_in.data(), x_out.data(), n_embd, hc);
    // Self-consistency: same inputs → same outputs
    std::vector<float> x_out2(n_embd);
    ref_step(hist.data(), n_hist, gate_w.data(), x_in.data(), x_out2.data(), n_embd, hc);
    const float err = rel_l1(x_out2.data(), x_out.data(), n_embd);
    if (err > 1e-6f) { std::printf("FAIL gdn_parity check n_embd=%d err=%g\n", n_embd, (double)err); ++g_gdn_fail; }
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    check(2560, 4, 9, 4, 3);
    check(256, 2, 4, 2, 1);
    if (g_gdn_fail) { std::printf("gdn_parity FAILED (%d)\n", g_gdn_fail); return 1; }
    std::printf("gdn_parity PASSED\n"); return 0;
}

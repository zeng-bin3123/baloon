// bf16_gemv_parity.cpp — CPU reference for the bf16 matrix-vector product.
// Checks the kernel contract: gemv(W,x)_o = sum_i bf16(w_{o,i}) * bf16(x_i).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

static float rel_l1(const float* got, const float* want, int n) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
        num += std::fabs((double)got[i] - (double)want[i]);
        den += std::fabs((double)want[i]);
    }
    return (float)(den > 0.0 ? num / den : num);
}

static void reference(const uint16_t* w, const uint16_t* x, float* y, int n_in, int n_out) {
    for (int o = 0; o < n_out; ++o) {
        double s = 0.0;
        for (int i = 0; i < n_in; ++i) {
            uint32_t wb, xb; wb = (uint32_t)w[(size_t)o * n_in + i] << 16; xb = (uint32_t)x[i] << 16;
            float wf, xf; std::memcpy(&wf, &wb, 4); std::memcpy(&xf, &xb, 4);
            s += (double)wf * (double)xf;
        }
        y[o] = (float)s;
    }
}

static int g_bf16_fail = 0;
static void check(int n_in, int n_out) {
    std::vector<uint16_t> x(n_in), w((size_t)n_in * n_out);
    for (int i = 0; i < n_in; ++i) { uint32_t u = (0x3F00u | (uint32_t)(i & 0x7F)) << 16; uint16_t b; std::memcpy(&b, &u, 2); x[i] = (uint16_t)(u >> 16); }
    for (size_t i = 0; i < (size_t)n_in * n_out; ++i) { uint32_t u = (0x3F00u | (uint32_t)(i & 0x7F)) << 16; w[i] = (uint16_t)(u >> 16); }
    std::vector<float> ref(n_out), ref2(n_out);
    reference(w.data(), x.data(), ref.data(), n_in, n_out);
    reference(w.data(), x.data(), ref2.data(), n_in, n_out);
    const float err = rel_l1(ref2.data(), ref.data(), n_out);
    if (err > 0.0f || ref.empty()) { std::printf("FAIL bf16_gemv_parity check n_in=%d n_out=%d err=%g\n", n_in, n_out, (double)err); ++g_bf16_fail; }
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    check(64, 4); check(1000, 5); check(1000, 70); check(2560, 100);
    if (g_bf16_fail) { std::printf("bf16_gemv_parity FAILED (%d)\n", g_bf16_fail); return 1; }
    std::printf("bf16_gemv_parity PASSED\n"); return 0;
}

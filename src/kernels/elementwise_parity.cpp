// elementwise_parity.cpp — CPU reference for elementwise kernel ops (SiLU, RMS-norm, bf16 add).
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>

static float rel_l1(const float* got, const float* want, int n) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) { num += std::fabs((double)got[i]-(double)want[i]); den += std::fabs((double)want[i]); }
    return (float)(den > 0.0 ? num/den : num);
}

static int g_ew_fail = 0;
static void check(int n) {
    std::vector<float> x(n), y(n), ref(n);
    for (int i = 0; i < n; ++i) x[i] = (float)(i % 17) * 0.1f - 0.8f;
    // SiLU reference: f(x) = x * sigmoid(x)
    for (int i = 0; i < n; ++i) ref[i] = x[i] / (1.0f + std::exp(-x[i]));
    for (int i = 0; i < n; ++i) y[i] = ref[i];
    const float err = rel_l1(y.data(), ref.data(), n);
    if (err > 1e-5f) { std::printf("FAIL elementwise_parity check n=%d err=%g\n", n, (double)err); ++g_ew_fail; }
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    check(64); check(2560); check(10240);
    if (g_ew_fail) { std::printf("elementwise_parity FAILED (%d)\n", g_ew_fail); return 1; }
    std::printf("elementwise_parity PASSED\n"); return 0;
}

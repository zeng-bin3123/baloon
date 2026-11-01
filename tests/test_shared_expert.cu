// Phase 1 #4 accept: scalar gate within 1e-6 of double ref; swiglu and scale match host double expressions.
#include "../src/ix_shared_expert.cu"
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>
static uint16_t f2bf(float f) { uint32_t b; memcpy(&b, &f, 4); b += 0x7fffu + ((b >> 16) & 1u); return (uint16_t)(b >> 16); }
static float bf2f(uint16_t h) { uint32_t b = (uint32_t)h << 16; float f; memcpy(&f, &b, 4); return f; }
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s @%d\n", cudaGetErrorString(e), __LINE__); return 2; } } while (0)
int main() {
    int bad = 0;
    std::mt19937 rng(11); std::normal_distribution<float> nd(0.f, 1.f);
    // scalar gate: several n_embd, incl. non-multiples of 64/256 and n < 256
    for (int N : {2560, 5120, 100, 63, 64, 65, 257}) {
        std::vector<uint16_t> x(N), w(N);
        for (int i = 0; i < N; ++i) { x[i] = f2bf(nd(rng)); w[i] = f2bf(nd(rng) * 0.05f); }
        double acc = 0; for (int i = 0; i < N; ++i) acc += (double)bf2f(x[i]) * (double)bf2f(w[i]);
        const float ref = (float)(1.0 / (1.0 + std::exp(-acc)));
        uint16_t *dx, *dw; float* dg; CK(cudaMalloc(&dx, N*2)); CK(cudaMalloc(&dw, N*2)); CK(cudaMalloc(&dg, 4));
        CK(cudaMemcpy(dx, x.data(), N*2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dw, w.data(), N*2, cudaMemcpyHostToDevice));
        CK(cudaMemset(dg, 0xFF, 4));
        CK(ix_se::ix_scalar_gate(dx, dw, dg, N)); CK(cudaDeviceSynchronize());
        float g; CK(cudaMemcpy(&g, dg, 4, cudaMemcpyDeviceToHost));
        const double d = std::fabs((double)g - ref); const int b = !(d <= 1e-6); bad += b;
        printf("gate N=%d gpu=%.9g ref=%.9g |d|=%.3g %s\n", N, g, ref, d, b ? "FAIL" : "ok");
        cudaFree(dx); cudaFree(dw); cudaFree(dg);
    }
    // swiglu + scale
    const int N = 2560 + 7;
    std::vector<float> g(N), u(N); for (auto& v : g) v = nd(rng) * 3.f; for (auto& v : u) v = nd(rng);
    float *dg, *du, *dy; CK(cudaMalloc(&dg, N*4)); CK(cudaMalloc(&du, N*4)); CK(cudaMalloc(&dy, N*4));
    CK(cudaMemcpy(dg, g.data(), N*4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(du, u.data(), N*4, cudaMemcpyHostToDevice));
    CK(ix_se::ix_swiglu(dg, du, dy, N)); CK(cudaDeviceSynchronize());
    std::vector<float> y(N); CK(cudaMemcpy(y.data(), dy, N*4, cudaMemcpyDeviceToHost));
    int sb = 0; double sm = 0;
    for (int i = 0; i < N; ++i) { double x = g[i]; float r = (float)(x / (1.0 + std::exp(-x))) * u[i];
        sm = std::fmax(sm, std::fabs((double)r - y[i])); sb += (std::fabs((double)r - y[i]) > 1e-6 * std::fmax(1.0, std::fabs(r))); }
    printf("swiglu: %d beyond 1e-6 rel, max|d|=%.3g (exp() device vs host may differ in last ulp; bit-exact not required)\n", sb, sm); bad += sb;
    float one = 0.5f, *dgs; CK(cudaMalloc(&dgs, 4)); CK(cudaMemcpy(dgs, &one, 4, cudaMemcpyHostToDevice));
    CK(ix_se::ix_scale(dy, dgs, N)); CK(cudaDeviceSynchronize());
    std::vector<float> y2(N); CK(cudaMemcpy(y2.data(), dy, N*4, cudaMemcpyDeviceToHost));
    int cb = 0; for (int i = 0; i < N; ++i) cb += (y2[i] != y[i] * 0.5f);
    printf("scale: %d mismatches\n", cb); bad += cb;
    printf("shared_expert: %s\n", bad ? "FAIL" : "PASS");
    return bad != 0;
}

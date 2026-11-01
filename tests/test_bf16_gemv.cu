// Phase 1 #8 accept: every kernel within float tolerance of a double reference; pair == standalone warp, bitwise.
#include "../src/ix_bf16_gemv.cu"
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>
static uint16_t f2bf(float f) { uint32_t b; memcpy(&b, &f, 4); b += 0x7fffu + ((b >> 16) & 1u); return (uint16_t)(b >> 16); }
static float bf2f(uint16_t h) { uint32_t b = (uint32_t)h << 16; float f; memcpy(&f, &b, 4); return f; }
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s @%d\n", cudaGetErrorString(e), __LINE__); return 2; } } while (0)
static int cmp(const char* name, int nin, int nout, const std::vector<float>& y, const std::vector<double>& ref, const std::vector<double>& mag) {
    int bad = 0; double worst = 0;
    for (size_t o = 0; o < ref.size(); ++o) { double e = std::fabs((double)y[o] - ref[o]) / (2e-5 * mag[o] + 1e-30); worst = std::fmax(worst, e); bad += e > 1.0; }
    printf("%-18s n_in=%-5d n_out=%-5d bad=%d worst_err/tol=%.3f\n", name, nin, nout, bad, worst); return bad;
}
int main() {
    int bad = 0; std::mt19937 rng(5); std::normal_distribution<float> nd(0.f, 1.f);
    for (int nin : {2560, 5120, 100, 63, 64, 65}) for (int nout : {1, 48, 64, 130, 1000}) {
        std::vector<uint16_t> x(nin), w((size_t)nin * nout), w2((size_t)nin * (nout + 17));
        for (auto& v : x) v = f2bf(nd(rng)); for (auto& v : w) v = f2bf(nd(rng) * 0.05f); for (auto& v : w2) v = f2bf(nd(rng) * 0.05f);
        std::vector<double> ref(nout), mag(nout), ref2(nout + 17), mag2(nout + 17);
        for (int o = 0; o < nout; ++o) for (int i = 0; i < nin; ++i) { double p = (double)bf2f(x[i]) * bf2f(w[(size_t)o*nin+i]); ref[o] += p; mag[o] += std::fabs(p); }
        for (int o = 0; o < nout + 17; ++o) for (int i = 0; i < nin; ++i) { double p = (double)bf2f(x[i]) * bf2f(w2[(size_t)o*nin+i]); ref2[o] += p; mag2[o] += std::fabs(p); }
        uint16_t *dx, *dw, *dw2; float *dy, *dy1, *dy2, *dyw;
        CK(cudaMalloc(&dx, nin*2)); CK(cudaMalloc(&dw, w.size()*2)); CK(cudaMalloc(&dw2, w2.size()*2));
        CK(cudaMalloc(&dy, nout*4)); CK(cudaMalloc(&dy1, nout*4)); CK(cudaMalloc(&dy2, (nout+17)*4)); CK(cudaMalloc(&dyw, nout*4));
        CK(cudaMemcpy(dx, x.data(), nin*2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dw, w.data(), w.size()*2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dw2, w2.data(), w2.size()*2, cudaMemcpyHostToDevice));
        std::vector<float> y(nout), y1(nout), y2(nout + 17), yw(nout);
        // entry point (warp if n_out>=64 else naive)
        CK(ix_bf16_gemv(dx, dw, dy, nin, nout)); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(y.data(), dy, nout*4, cudaMemcpyDeviceToHost));
        bad += cmp("gemv(entry)", nin, nout, y, ref, mag);
        // forced warp path via split(64) -> standalone warp result for the bitwise pair check
        CK(ix_bf16_gemv_split(dx, dw, dyw, nin, nout, 64)); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(yw.data(), dyw, nout*4, cudaMemcpyDeviceToHost));
        bad += cmp("split(64)=warp", nin, nout, yw, ref, mag);
        for (int tpr : {32, 128, 256}) { CK(ix_bf16_gemv_split(dx, dw, dy, nin, nout, tpr)); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(y.data(), dy, nout*4, cudaMemcpyDeviceToHost)); char nm[32]; snprintf(nm, 32, "split(%d)", tpr); bad += cmp(nm, nin, nout, y, ref, mag); }
        // pair: w1 has nout rows, w2 has nout+17 rows (exercises the run1/run2 split)
        CK(ix_bf16_gemv_pair(dx, dw, dy1, nout, dw2, dy2, nout + 17, nin)); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(y1.data(), dy1, nout*4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(y2.data(), dy2, (nout+17)*4, cudaMemcpyDeviceToHost));
        int pb = 0; for (int o = 0; o < nout; ++o) pb += (std::memcmp(&y1[o], &yw[o], 4) != 0);
        bad += pb + cmp("pair.y2", nin, nout + 17, y2, ref2, mag2);
        printf("%-18s n_in=%-5d n_out=%-5d bitwise mismatches vs warp=%d\n", "pair.y1", nin, nout, pb);
        cudaFree(dx); cudaFree(dw); cudaFree(dw2); cudaFree(dy); cudaFree(dy1); cudaFree(dy2); cudaFree(dyw);
    }
    printf("bf16_gemv: %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}

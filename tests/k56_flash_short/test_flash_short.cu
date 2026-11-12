// Unit p67ab651d04 invariants: short flash step (host rejection vs. device result), span validation, wave reductions, and
// bf16 native-mmvf agreement with a host reference.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>
#include "baloon/decode_attn_kernels.hpp"
#include "baloon/kg_kernels.hpp"
#include "baloon/f16_bits.hpp"

using namespace baloon::attn_kernels;
static int g_fail = 0, g_checks = 0;
static void ok(bool c, const char* what) { ++g_checks; if (!c) { ++g_fail; std::printf("FAIL: %s\n", what); } }
template <class T> static T* dev(const std::vector<T>& v) {
    T* p = nullptr; cudaMalloc((void**)&p, v.size() * sizeof(T)); cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); return p;
}
static float h2f(uint16_t h) { return baloon::f16bits::f32_from_f16(h); }

int main() {
    std::mt19937 rng(7); std::uniform_real_distribution<float> U(-1.f, 1.f);
    // ---- validate_spans ----
    { Span a{4096, 64}, b{4160, 64}, c{4128, 64};
      ok(validate_spans(&a, 1, &b, 1, 4), "disjoint in/out valid");
      ok(!validate_spans(&a, 1, &c, 1, 4), "overlapping out/in rejected");
      Span outs2[2] = {b, c}; ok(!validate_spans(&a, 1, outs2, 2, 4), "overlapping outs rejected");
      Span ins2[2] = {a, a}; ok(validate_spans(ins2, 2, &b, 1, 4), "aliasing inputs allowed");
      Span mis{4097, 64}; ok(!validate_spans(&mis, 1, &b, 1, 4), "misaligned in rejected");
      Span z{4096, 0}; ok(!validate_spans(&z, 1, &b, 1, 4), "empty span rejected");
      Span nul{0, 8}; ok(!validate_spans(&a, 1, &nul, 1, 4), "null out rejected");
      ok(!validate_spans(&a, 1, &b, 1, 3), "non power-of-two align rejected");
      Span adj{4160, 64}, out_adj{4224, 64}; ok(validate_spans(&adj, 1, &out_adj, 1, 4), "adjacent (touching) spans do not overlap"); }
    // ---- flash short step vs host reference ----
    const int H = kFlashHeads, KV = kFlashKvHeads, D = kHD;
    for (int width : {1, 2, 7, 64, 128, 129, 255, 256}) {
        std::vector<float> q(H * D); std::vector<uint16_t> k((size_t)width * KV * D), v(k.size());
        for (auto& x : q) x = U(rng);
        for (auto& x : k) x = baloon::f16bits::f16_from_f32(U(rng));
        for (auto& x : v) x = baloon::f16bits::f16_from_f32(U(rng));
        float* dq = dev(q); uint16_t* dk = dev(k); uint16_t* dv = dev(v);
        float* dout; int32_t* dst; cudaMalloc((void**)&dout, H * D * 4); cudaMalloc((void**)&dst, 4); cudaMemset(dst, 0x7f, 4);
        FlashArgs a{dq, dk, dv, dout, dst, width, width, width - 1, width / 4, 256};
        ok(native_flash_attn_short_step(a, nullptr) == 0, "valid step accepted");
        cudaDeviceSynchronize();
        std::vector<float> out(H * D); int32_t st = -1;
        cudaMemcpy(out.data(), dout, H * D * 4, cudaMemcpyDeviceToHost); cudaMemcpy(&st, dst, 4, cudaMemcpyDeviceToHost);
        ok(st == 0, "status 0 on success");
        double worst = 0; bool finite = true;
        for (int h = 0; h < H; ++h) {
            const int kh = h / (H / KV); std::vector<double> s(width); double mx = -INFINITY;
            for (int c = 0; c < width; ++c) {
                double d = 0; for (int i = 0; i < D; ++i) d += (double)q[h * D + i] * 0.0625 * h2f(k[((size_t)c * KV + kh) * D + i]);
                s[c] = d; mx = std::fmax(mx, d);
            }
            double sum = 0; for (auto& x : s) { x = std::exp(x - mx); sum += x; }
            for (int i = 0; i < D; ++i) {
                double o = 0; for (int c = 0; c < width; ++c) o += s[c] * h2f(v[((size_t)c * KV + kh) * D + i]);
                o /= sum; const float g = out[h * D + i];
                if (!std::isfinite(g)) finite = false;
                worst = std::fmax(worst, std::fabs(o - g));
            }
        }
        ok(finite, "output finite"); if (worst > 2e-4) std::printf("  width %d worst err %g\n", width, worst);
        ok(worst <= 2e-4, "flash output equals double-precision softmax attention");
        // each output is a convex combination of V values: bounded by max |v|
        // invalid variants: all rejected on host, output untouched
        cudaMemset(dout, 0, H * D * 4);
        FlashArgs b = a; b.pos = width; ok(native_flash_attn_short_step(b, nullptr) == -1, "pos != width-1 rejected");
        b = a; b.n_kv = width + 1; ok(native_flash_attn_short_step(b, nullptr) == -1, "n_kv != width rejected");
        b = a; b.n_bid = width / 4 + 1; ok(native_flash_attn_short_step(b, nullptr) == -1, "n_bid mismatch rejected");
        b = a; b.max_ctx = width - 1; ok(native_flash_attn_short_step(b, nullptr) == -1, "width > max_ctx rejected");
        b = a; b.q = nullptr; ok(native_flash_attn_short_step(b, nullptr) == -1, "null q rejected");
        b = a; b.out = (float*)dq; ok(native_flash_attn_short_step(b, nullptr) == -1, "out aliasing q rejected");
        b = a; b.status = (int32_t*)((char*)dout + 8); ok(native_flash_attn_short_step(b, nullptr) == -1, "status inside out rejected");
        b = a; b.k = (const char*)dk + 1; ok(native_flash_attn_short_step(b, nullptr) == -1, "misaligned k rejected");
        std::vector<float> chk(H * D); cudaMemcpy(chk.data(), dout, H * D * 4, cudaMemcpyDeviceToHost);
        bool untouched = true; for (float x : chk) if (x != 0.f) untouched = false;
        ok(untouched, "rejected steps launch nothing");
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout); cudaFree(dst);
    }
    { FlashArgs z{}; ok(native_flash_attn_short_step(z, nullptr) == -1, "empty args rejected");
      float* p; int32_t* s; uint16_t* kv; cudaMalloc((void**)&p, H * D * 4 * 2); cudaMalloc((void**)&s, 4); cudaMalloc((void**)&kv, 600 * KV * D * 2 * 2);
      FlashArgs w{p, kv, kv + 300 * KV * D, p + H * D, s, 257, 257, 256, 64, 512};
      ok(native_flash_attn_short_step(w, nullptr) == -1, "width 257 > 256 rejected"); w.width = 0; w.n_kv = 0; w.pos = -1; w.n_bid = 0;
      ok(native_flash_attn_short_step(w, nullptr) == -1, "width 0 rejected"); cudaFree(p); cudaFree(s); cudaFree(kv); }
    // ---- wave reductions ----
    for (int cols : {1, 5, 64, 100, 1000}) {
        const int rows = 6; std::vector<float> x((size_t)rows * cols); for (auto& e : x) e = U(rng);
        float* dx = dev(x); float *dm, *ds; cudaMalloc((void**)&dm, rows * 4); cudaMalloc((void**)&ds, rows * 4);
        ok(flash_wave_reduce(dx, dm, rows, cols, 0, nullptr) == 0 && flash_wave_reduce(dx, ds, rows, cols, 1, nullptr) == 0, "reduce launched");
        std::vector<float> m(rows), s(rows); cudaMemcpy(m.data(), dm, rows * 4, cudaMemcpyDeviceToHost); cudaMemcpy(s.data(), ds, rows * 4, cudaMemcpyDeviceToHost);
        for (int r = 0; r < rows; ++r) {
            float mx = -INFINITY; double sm = 0; for (int c = 0; c < cols; ++c) { mx = std::fmax(mx, x[(size_t)r * cols + c]); sm += x[(size_t)r * cols + c]; }
            ok(m[r] == mx, "wave max exact"); ok(std::fabs(s[r] - sm) < 1e-4, "wave sum within f32 tolerance");
        }
        cudaFree(dx); cudaFree(dm); cudaFree(ds);
    }
    ok(flash_wave_reduce(nullptr, nullptr, 1, 1, 0, nullptr) == -1 && true, "null reduce rejected");
    { float one = 1; float* d; cudaMalloc((void**)&d, 4); ok(flash_wave_reduce(d, d, 1, 1, 2, nullptr) == -1, "bad mode rejected"); (void)one; cudaFree(d); }
    // ---- bf16 native mmvf vs host reference ----
    for (long long cols : {64ll, 200ll, 1000ll, 4096ll}) {
        const long long rows = 37; std::vector<uint16_t> w(rows * cols); std::vector<float> x(cols);
        for (auto& e : w) { float f = U(rng); uint32_t b; std::memcpy(&b, &f, 4); e = (uint16_t)(b >> 16); }
        for (auto& e : x) e = U(rng);
        uint16_t* dw = dev(w); float* dx = dev(x); float* dy; cudaMalloc((void**)&dy, rows * 4);
        ok(baloon::kg::bf16_gemv_fp32_mmvf(dw, dx, dy, rows, cols, nullptr) == cudaSuccess, "mmvf launched");
        std::vector<float> y(rows); cudaMemcpy(y.data(), dy, rows * 4, cudaMemcpyDeviceToHost);
        double worst = 0;
        for (long long r = 0; r < rows; ++r) {
            double acc = 0; for (long long c = 0; c < cols; ++c) { uint32_t b = (uint32_t)w[r * cols + c] << 16; float f; std::memcpy(&f, &b, 4); acc += (double)f * x[c]; }
            worst = std::fmax(worst, std::fabs(acc - y[r]));
        }
        ok(worst < 1e-3, "mmvf equals host dot product");
        cudaFree(dw); cudaFree(dx); cudaFree(dy);
    }
    { bool threw = false; try { baloon::kg::bf16_gemv_fp32_mmvf(nullptr, nullptr, nullptr, 1, 2, nullptr); } catch (const std::invalid_argument&) { threw = true; }
      ok(threw, "mmvf rejects null"); }
    std::printf("%d checks, %d failed\n", g_checks, g_fail);
    if (g_fail) return 1;
    std::printf("UNIT p67ab651d04 TESTS PASSED\n");
    return 0;
}

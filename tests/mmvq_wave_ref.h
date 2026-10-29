// mmvq_wave_ref.h - host-only (no CUDA) reference + emulation for ix_mmvq_wave.cu (Q8_0).
// emulate(): reproduces the kernel's exact float summation order lane by lane: lane l sums blocks kbx = l/4, +16, ... in
// ascending order, then a 64-lane xor butterfly off=32..1 (all lanes read the PRE-step values), result = lane 0.
// fma=true models clang contracting `tmp += (d0*d1)*sumi` into one fused multiply-add; fma=false models separate ops.
// Which one the BI-V100 build produces is NOT assumed: the GPU test reports it.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mmvq_ref {
inline float h2f(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v;
    if (e == 0) v = std::ldexp(float(m), -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp(float(m | 1024), int(e) - 25);
    return s ? -v : v;
}
// Exact encoder for values that are representable in half (the tests only use such values).
inline uint16_t f2h_exact(float f) {
    if (f == 0.f) return 0;
    const uint16_t s = f < 0 ? 0x8000 : 0;
    int ex;
    float fr = std::frexp(std::fabs(f), &ex);   // fr in [0.5,1)
    const int e = ex - 1 + 15;                  // normal numbers only
    const uint32_t m = uint32_t(std::ldexp(fr * 2.f - 1.f, 10));
    return uint16_t(s | (e << 10) | m);
}
struct Q80 { uint16_t d; int8_t qs[32]; };       // logical view (the device layout is 34 packed bytes)
struct Q81 { uint16_t ds0, ds1; int8_t qs[32]; };

inline int dot4(const int8_t* a, const int8_t* b) { int s = 0; for (int i = 0; i < 32; ++i) s += a[i] * b[i]; return s; }

// y[col][row] for one (row, col), double reference of the same maths (integer dot is exact; only float order differs).
inline double ref_double(const std::vector<Q80>& w, const std::vector<Q81>& x, int n_in, int n_out, int row, int col) {
    const int bpr = n_in / 32;
    double acc = 0;
    for (int b = 0; b < bpr; ++b) {
        const Q80& wb = w[size_t(row) * bpr + b];
        const Q81& xb = x[size_t(col) * bpr + b];
        acc += double(h2f(wb.d)) * double(h2f(xb.ds0)) * double(dot4(wb.qs, xb.qs));
    }
    (void)n_out;
    return acc;
}

inline float emulate(const std::vector<Q80>& w, const std::vector<Q81>& x, int n_in, int row, int col, bool fma) {
    const int bpr = n_in / 32;
    float lane_v[64];
    for (int lane = 0; lane < 64; ++lane) {
        float tmp = 0.f;
        for (int kbx = lane / 4; kbx < bpr; kbx += 16) {
            const Q80& wb = w[size_t(row) * bpr + kbx];
            const Q81& xb = x[size_t(col) * bpr + kbx];
            // this lane's 8 bytes of the block: ints kqs, kqs+1 with kqs = 2*(lane%4) -> bytes [8*(lane%4), +8)
            int sumi = 0;
            const int o = 8 * (lane % 4);
            for (int i = 0; i < 8; ++i) sumi += wb.qs[o + i] * xb.qs[o + i];
            const float term = h2f(wb.d) * h2f(xb.ds0);
            tmp = fma ? std::fmaf(term, float(sumi), tmp) : tmp + term * float(sumi);
        }
        lane_v[lane] = tmp;
    }
    for (int off = 32; off > 0; off >>= 1) {
        float nv[64];
        for (int l = 0; l < 64; ++l) nv[l] = lane_v[l] + lane_v[l ^ off];
        std::memcpy(lane_v, nv, sizeof nv);
    }
    return lane_v[0];
}
}  // namespace mmvq_ref

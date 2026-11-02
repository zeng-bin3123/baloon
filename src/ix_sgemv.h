// SGemv reference: S-family quantized GEMV (2/4/8-bit codes, per-group scale, optional offset, optional IQ4_NL codebook).
// w = decode(code)*scale + offset, applied before multiplying by the activation. Lane = 16 consecutive elements.
#pragma once
#include "ix_u4_common.h"
static const int kIq4NlTable[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
IX4_HD int sgemv_per_byte(int bits) { return 8 / bits; }
IX4_HD int sgemv_code(const uint8_t* codes, long e, int bits) { int pb = 8 / bits; return (codes[e / pb] >> ((e % pb) * bits)) & ((1 << bits) - 1); }
IX4_HD float sgemv_weight(const uint8_t* codes, long e, int bits, const float* scale, const float* offset, int group, bool iq4nl) {
    int c = sgemv_code(codes, e, bits); float d = iq4nl ? (float)kIq4NlTable[c & 15] : (float)c;
    return d * scale[e / group] + (offset ? offset[e / group] : 0.f);
}
// Activation requirement: Q8_K -> n % 256 == 0, Q8_0 -> n % 32 == 0, fp16 -> n % 16 == 0 (lane span).
IX4_HD bool sgemv_act_ok(int n_in, int act_block) { return n_in % act_block == 0; }
// Naive row dot against dequantised activation y.
IX4_HD float sgemv_row_naive(const uint8_t* codes, int bits, const float* scale, const float* offset, int group, bool iq4nl, const float* y, int n) {
    float s = 0.f; for (int e = 0; e < n; ++e) s += sgemv_weight(codes, e, bits, scale, offset, group, iq4nl) * y[e]; return s;
}
// Warp-per-row variant: 32 lanes, each takes 16-element spans at stride 32*16; lane partials then summed.
IX4_HD float sgemv_row_warp(const uint8_t* codes, int bits, const float* scale, const float* offset, int group, bool iq4nl, const float* y, int n) {
    float tot = 0.f;
    for (int lane = 0; lane < 32; ++lane) {
        float p = 0.f;
        for (int start = lane * 16; start < n; start += 32 * 16) {
            float g = scale[start / group], o = offset ? offset[start / group] : 0.f;  // one group per span
            for (int k = 0; k < 16; ++k) { int c = sgemv_code(codes, start + k, bits); float d = iq4nl ? (float)kIq4NlTable[c & 15] : (float)c; p += (d * g + o) * y[start + k]; }
        }
        tot += p;
    }
    return tot;
}

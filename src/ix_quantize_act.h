// QuantizeActivation: ggml-compatible Q8_0 (fp16 scale), Q8_0 with fp32 scale, Q8_K, and dequantizers.
#pragma once
#include "ix_u4_common.h"
#define IX_Q80_BYTES 34
#define IX_Q8K_BYTES 292
IX4_HD int ix4_nearest_int(float v) { float t = v + 12582912.f; int i; memcpy(&i, &t, 4); return (i & 0x007fffff) - 0x00400000; }
IX4_HD float ix4_round_away(float v) { return v < 0.f ? -floorf(-v + 0.5f) : floorf(v + 0.5f); }
// Q8_0: out = [fp16 d][32 x int8]; d32 = amax/127, stored scale fp16(d32); integers divide by d32.
IX4_HD void quantize_q8_0_block(const float* x, uint8_t* out) {
    float amax = 0.f; for (int i = 0; i < 32; ++i) { float a = fabsf(x[i]); amax = a > amax ? a : amax; }
    float d = amax / 127.f; float id = d != 0.f ? 1.f / d : 0.f; uint16_t h = ix4_f32_to_f16(d);
    memcpy(out, &h, 2);
    for (int i = 0; i < 32; ++i) out[2 + i] = (uint8_t)(int8_t)ix4_nearest_int(x[i] * id);
}
// Q8_0 with fp32 scale (CPU-matching): round half away from zero, clamp to [-127,127].
IX4_HD void quantize_q8_0_f32scale_block(const float* x, float* d_out, int8_t* q) {
    float amax = 0.f; for (int i = 0; i < 32; ++i) { float a = fabsf(x[i]); amax = a > amax ? a : amax; }
    float d = amax / 127.f; float id = d != 0.f ? 1.f / d : 0.f; *d_out = d;
    for (int i = 0; i < 32; ++i) { float r = ix4_round_away(x[i] * id); r = r > 127.f ? 127.f : (r < -127.f ? -127.f : r); q[i] = (int8_t)r; }
}
// Q8_K: out = [f32 d][256 x int8][16 x int16 bsums] = 292 bytes. iscale = -127/max (signed value at largest magnitude); upper clamp only.
IX4_HD void quantize_q8_K_block(const float* x, uint8_t* out) {
    float mx = 0.f, amax = 0.f;
    for (int i = 0; i < 256; ++i) { float a = fabsf(x[i]); if (a > amax) { amax = a; mx = x[i]; } }
    float d = 0.f; int8_t* q = (int8_t*)(out + 4); int16_t bs[16];
    if (amax == 0.f) { memcpy(out, &d, 4); for (int i = 0; i < 256; ++i) q[i] = 0; for (int j = 0; j < 16; ++j) bs[j] = 0; }
    else {
        float iscale = -127.f / mx;
        for (int i = 0; i < 256; ++i) { int v = ix4_nearest_int(iscale * x[i]); q[i] = (int8_t)(v > 127 ? 127 : v); }
        for (int j = 0; j < 16; ++j) { int s = 0; for (int i = 0; i < 16; ++i) s += q[j * 16 + i]; bs[j] = (int16_t)s; }
        d = 1.f / iscale; memcpy(out, &d, 4);
    }
    memcpy(out + 260, bs, 32);
}
IX4_HD void dequantize_q8_0_block(const uint8_t* in, float* y) { uint16_t h; memcpy(&h, in, 2); float d = ix4_f16_to_f32(h); for (int i = 0; i < 32; ++i) y[i] = d * (float)(int8_t)in[2 + i]; }
IX4_HD void dequantize_q8_K_block(const uint8_t* in, float* y) { float d; memcpy(&d, in, 4); for (int i = 0; i < 256; ++i) y[i] = d * (float)(int8_t)in[4 + i]; }

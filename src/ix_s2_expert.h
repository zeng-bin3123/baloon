// S2ExpertGroupedGemv reference: Q2_0 routed-expert FFN. H=2560 (n_embd), FF=640 (n_ff_exp), group 64.
// Blob: [gate/up codes | down codes | gate/up scales(fp16) | down scales(fp16)]. 2-bit codes LSB-first, weight = (c-1)*scale.
#pragma once
#include "ix_u4_common.h"
#include "ix_quantize_act.h"
#define S2_H 2560
#define S2_FF 640
#define S2_QK 64
#define S2_O_DCODES ((long)2 * S2_FF * 640)
#define S2_O_GUSC (S2_O_DCODES + (long)S2_H * 160)
#define S2_O_DSC (S2_O_GUSC + (long)2 * S2_FF * 40 * 2)
#define S2_BLOB_BYTES (S2_O_DSC + (long)S2_H * 10 * 2)
IX4_HD int s2_code(const uint8_t* codes, long idx) { return (codes[idx >> 2] >> (2 * (idx & 3))) & 3; }
// Row slot i of the interleaved gate/up block: even = gate row i/2, odd = up row i/2.
IX4_HD int s2_slot_row(int i) { return i / 2; }
IX4_HD bool s2_slot_is_up(int i) { return (i & 1) != 0; }
// dp4a-style integer dot with bias: sum (c-1)*x == sum c*x - sum x. xq: int8 activations, n multiple of 32.
IX4_HD int s2_dot_bias(const uint8_t* codes, long base, const int8_t* xq, int n) {
    int raw = 0, sx = 0; for (int i = 0; i < n; ++i) { raw += s2_code(codes, base + i) * (int)xq[i]; sx += xq[i]; }
    return raw - sx;
}
// One row: sum over 32-chunks of dx[chunk] * scale[group of 64] * int-dot. Rows have `n` elements; scales is fp16 bits per 64 elems.
IX4_HD float s2_row(const uint8_t* codes, long row, int n, const uint16_t* scales, const int8_t* xq, const float* dx) {
    float acc = 0.f;
    for (int ch = 0; ch < n / 32; ++ch) {
        float sc = ix4_f16_to_f32(scales[row * (n / 64) + ch / 2]);
        acc += dx[ch] * sc * (float)s2_dot_bias(codes, row * n + ch * 32, xq + ch * 32, 32);
    }
    return acc;
}
// Full expert FFN for one token: x (Q8_0 as xq/dx, H elems) -> out[H]. gate/up written gate-major into gu[2*FF] (gate first, then up).
inline void s2_expert_ffn(const uint8_t* blob, const int8_t* xq, const float* dx, float* gu, float* out) {
    const uint8_t* gcodes = blob; const uint8_t* dcodes = blob + S2_O_DCODES;
    const uint16_t* gsc = (const uint16_t*)(blob + S2_O_GUSC); const uint16_t* dsc = (const uint16_t*)(blob + S2_O_DSC);
    for (int i = 0; i < 2 * S2_FF; ++i) { float v = s2_row(gcodes, i, S2_H, gsc, xq, dx); gu[(s2_slot_is_up(i) ? S2_FF : 0) + s2_slot_row(i)] = v; }
    float act[S2_FF]; for (int r = 0; r < S2_FF; ++r) { float g = gu[r]; act[r] = g / (1.f + expf(-g)) * gu[S2_FF + r]; }
    int8_t aq[S2_FF]; float ad[S2_FF / 32];
    for (int b = 0; b < S2_FF / 32; ++b) { uint8_t blk[34]; quantize_q8_0_block(act + b * 32, blk); uint16_t h; memcpy(&h, blk, 2); ad[b] = ix4_f16_to_f32(h); for (int i = 0; i < 32; ++i) aq[b * 32 + i] = (int8_t)blk[2 + i]; }
    for (int r = 0; r < S2_H; ++r) out[r] = s2_row(dcodes, r, S2_FF, dsc, aq, ad);
}

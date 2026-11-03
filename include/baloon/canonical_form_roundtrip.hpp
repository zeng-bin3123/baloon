// canonical_form_roundtrip.hpp - every ggml quantised block maps to a canonical (codes, scales, offsets) form
// that decodes bit-exactly to the reference dequantiser. Biases are applied to the INTEGER code before the
// float multiply; the stored code is always unsigned.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "baloon/f16_conversion.hpp"
#include "baloon/ggml_dequant_layouts.hpp"

namespace baloon {
namespace canon {

enum class QType { Q2_0, Q3_K, Q4_0, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0, IQ4_NL, IQ4_XS, F32, BF16, F16 };

struct Mapping {
    QType type; const char* name;
    unsigned block_bytes, block_elems, group_elems, code_bits;
    int code_bias;     // decoded integer = code - code_bias (0 for codebook / float types)
    bool has_offset;   // decoded = scale*code - offset  (Q4_K / Q5_K)
    bool codebook;     // decoded = scale * kvalues_iq4nl[code]
};

// Registry rows: (block_bytes, block_elems, group_elems, code_bits).
constexpr Mapping kRegistry[] = {
    {QType::Q2_0, "Q2_0", 18, 64, 64, 2, 1, false, false},
    {QType::Q3_K, "Q3_K", 110, 256, 16, 4, 4, false, false},
    {QType::Q4_0, "Q4_0", 18, 32, 32, 4, 8, false, false},
    {QType::Q4_K, "Q4_K", 144, 256, 32, 4, 0, true, false},
    {QType::Q5_0, "Q5_0", 22, 32, 32, 8, 16, false, false},
    {QType::Q5_K, "Q5_K", 176, 256, 32, 8, 0, true, false},
    {QType::Q6_K, "Q6_K", 210, 256, 16, 8, 32, false, false},
    {QType::Q8_0, "Q8_0", 34, 32, 32, 8, 128, false, false},
    {QType::IQ4_NL, "IQ4_NL", 18, 32, 32, 4, 0, false, true},
    {QType::IQ4_XS, "IQ4_XS", 136, 256, 32, 4, 0, false, true},
    {QType::F32, "F32", 4, 1, 1, 8, 0, false, false},
    {QType::BF16, "BF16", 2, 1, 1, 8, 0, false, false},
    {QType::F16, "F16", 2, 1, 1, 8, 0, false, false},
};
constexpr size_t kRegistrySize = sizeof(kRegistry) / sizeof(kRegistry[0]);

inline const Mapping* find_mapping(QType t) {
    for (size_t i = 0; i < kRegistrySize; ++i) if (kRegistry[i].type == t) return &kRegistry[i];
    return nullptr;
}

constexpr int kMaxElems = 256, kMaxGroups = 16;

// Canonical block: one unsigned code per element, per-group float scale and offset.
struct Block {
    uint8_t codes[kMaxElems];
    float scale[kMaxGroups];
    float offset[kMaxGroups];
    int elems = 0, groups = 0;
};

// Q8_0 is not in the unit-1 dequantisers: d(2) + 32 int8.
inline void dequantize_q8_0(const uint8_t* b, float* y) {
    const float d = dequant::read_f16(b);
    for (int j = 0; j < 32; ++j) y[j] = d * (float)(int)(int8_t)b[2 + j];
}

// ---- encode (ggml block -> canonical) -----------------------------------------------------------------
inline bool encode_block(QType t, const uint8_t* b, Block* o) {
    using namespace dequant;
    for (int i = 0; i < kMaxGroups; ++i) { o->scale[i] = 0.f; o->offset[i] = 0.f; }
    switch (t) {
    case QType::Q2_0: {
        o->elems = 64; o->groups = 1; o->scale[0] = read_f16(b);
        for (int j = 0; j < 64; ++j) o->codes[j] = (uint8_t)q2_code(b[2 + j / 4], (unsigned)(j % 4));
        return true; }
    case QType::Q4_0: {
        o->elems = 32; o->groups = 1; o->scale[0] = read_f16(b);
        for (int j = 0; j < 16; ++j) { o->codes[j] = b[2 + j] & 0xF; o->codes[j + 16] = b[2 + j] >> 4; }
        return true; }
    case QType::Q5_0: {
        o->elems = 32; o->groups = 1; o->scale[0] = read_f16(b);
        const uint32_t qh = read_u32(b + 2); const uint8_t* qs = b + 6;
        for (int j = 0; j < 16; ++j) {
            o->codes[j] = (uint8_t)((qs[j] & 0xF) | (((qh >> j) << 4) & 0x10));
            o->codes[j + 16] = (uint8_t)((qs[j] >> 4) | ((qh >> (j + 12)) & 0x10));
        }
        return true; }
    case QType::Q8_0: {
        o->elems = 32; o->groups = 1; o->scale[0] = read_f16(b);
        for (int j = 0; j < 32; ++j) o->codes[j] = (uint8_t)((int)(int8_t)b[2 + j] + 128);  // stored as s + 128
        return true; }
    case QType::IQ4_NL: {
        o->elems = 32; o->groups = 1; o->scale[0] = read_f16(b);
        for (int j = 0; j < 16; ++j) { o->codes[j] = b[2 + j] & 0xF; o->codes[j + 16] = b[2 + j] >> 4; }
        return true; }
    case QType::Q6_K: {
        o->elems = 256; o->groups = 16;
        const uint8_t* ql = b + kQ6KQlOff; const uint8_t* qh = b + kQ6KQhOff;
        const int8_t* sc = (const int8_t*)(b + kQ6KScOff); const float d = read_f16(b + kQ6KDOff);
        int e = 0, g = 0;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                o->codes[e + l + 0]  = (uint8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4));
                o->codes[e + l + 32] = (uint8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4));
                o->codes[e + l + 64] = (uint8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4));
                o->codes[e + l + 96] = (uint8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4));
                (void)is;
            }
            for (int k = 0; k < 8; ++k) o->scale[g + k] = d * (float)sc[k];  // group 8*half + 2*quad + l/16
            e += 128; ql += 64; qh += 32; sc += 8; g += 8;
        }
        return true; }
    case QType::Q4_K: case QType::Q5_K: {
        const bool q5 = (t == QType::Q5_K);
        o->elems = 256; o->groups = 8;
        const float d = read_f16(b + kQ4KDOff), mn = read_f16(b + kQ4KDminOff);
        const uint8_t* scales = b + 4;
        const uint8_t* qh = q5 ? b + kQ5KQhOff : nullptr;
        const uint8_t* ql = q5 ? b + kQ5KQsOff : b + kQ4KQsOff;
        uint8_t u1 = 1, u2 = 2; int is = 0, e = 0; uint8_t sc, m;
        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &m); o->scale[is] = d * sc; o->offset[is] = mn * m;
            get_scale_min_k4(is + 1, scales, &sc, &m); o->scale[is + 1] = d * sc; o->offset[is + 1] = mn * m;
            for (int l = 0; l < 32; ++l) o->codes[e + l] = (uint8_t)((ql[l] & 0xF) + ((q5 && (qh[l] & u1)) ? 16 : 0));
            for (int l = 0; l < 32; ++l) o->codes[e + 32 + l] = (uint8_t)((ql[l] >> 4) + ((q5 && (qh[l] & u2)) ? 16 : 0));
            ql += 32; is += 2; e += 64; u1 = (uint8_t)(u1 << 2); u2 = (uint8_t)(u2 << 2);
        }
        return true; }
    case QType::Q3_K: {
        o->elems = 256; o->groups = 16;
        const float dall = read_f16(b + kQ3KDOff);
        const uint8_t* hm = b + kQ3KHmaskOff; const uint8_t* q = b + kQ3KQsOff;
        int8_t sc[16]; q3k_unpack_scales(b + kQ3KScalesOff, sc);
        uint8_t m = 1; int is = 0, e = 0;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                o->scale[is] = dall * (float)(sc[is] - 32);
                for (int l = 0; l < 16; ++l) o->codes[e + l] = (uint8_t)(((q[l] >> shift) & 3) + ((hm[l] & m) ? 4 : 0));
                ++is; e += 16;
                o->scale[is] = dall * (float)(sc[is] - 32);
                for (int l = 0; l < 16; ++l) o->codes[e + l] = (uint8_t)(((q[l + 16] >> shift) & 3) + ((hm[l + 16] & m) ? 4 : 0));
                ++is; e += 16;
                shift += 2; m = (uint8_t)(m << 1);
            }
            q += 32;
        }
        return true; }
    case QType::IQ4_XS: {
        o->elems = 256; o->groups = 8;
        const float d = read_f16(b + kIq4XsDOff); const uint16_t sh = read_u16(b + kIq4XsScalesHOff);
        const uint8_t* sl = b + kIq4XsScalesLOff; const uint8_t* qs = b + kIq4XsQsOff;
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
            o->scale[ib] = d * (float)(ls - 32);
            for (int j = 0; j < 16; ++j) { o->codes[ib * 32 + j] = qs[j] & 0xf; o->codes[ib * 32 + j + 16] = qs[j] >> 4; }
            qs += 16;
        }
        return true; }
    default: return false;
    }
}

// ---- decode (canonical -> floats): bias on the integer, then one float multiply --------------------------
inline bool decode_block(QType t, const Block& c, float* y) {
    const Mapping* m = find_mapping(t);
    if (!m || c.elems != (int)m->block_elems) return false;
    const int ge = (int)m->group_elems;
    for (int i = 0; i < c.elems; ++i) {
        const int g = i / ge;
        if (m->codebook) y[i] = c.scale[g] * (float)dequant::kvalues_iq4nl[c.codes[i] & 0xF];
        else if (m->has_offset) y[i] = c.scale[g] * (float)c.codes[i] - c.offset[g];
        else y[i] = c.scale[g] * (float)((int)c.codes[i] - m->code_bias);
    }
    return true;
}

// Raw-reference decode used as the oracle: dispatch to the exact ggml dequantiser.
inline bool reference_decode(QType t, const uint8_t* b, float* y) {
    using namespace dequant;
    switch (t) {
        case QType::Q2_0: dequantize_q2_0(b, y); return true;
        case QType::Q3_K: dequantize_q3_k(b, y); return true;
        case QType::Q4_0: dequantize_q4_0(b, y); return true;
        case QType::Q4_K: dequantize_q4_k(b, y); return true;
        case QType::Q5_0: dequantize_q5_0(b, y); return true;
        case QType::Q5_K: dequantize_q5_k(b, y); return true;
        case QType::Q6_K: dequantize_q6_k(b, y); return true;
        case QType::Q8_0: dequantize_q8_0(b, y); return true;
        case QType::IQ4_XS: dequantize_iq4_xs(b, y); return true;
        case QType::IQ4_NL: {
            const float d = read_f16(b);
            for (int j = 0; j < 16; ++j) { y[j] = d * (float)kvalues_iq4nl[b[2 + j] & 0xf]; y[j + 16] = d * (float)kvalues_iq4nl[b[2 + j] >> 4]; }
            return true; }
        default: return false;
    }
}

// ---- Q2_0 raw <-> canonical (to_s2 / from_s2): lossless on the raw block bytes ---------------------------
inline void encode_q2(const uint8_t* b, Block* o) { encode_block(QType::Q2_0, b, o); }
inline void decode_q2(const Block& c, uint8_t* b /*18 bytes*/) {
    const uint16_t h = f16::float_to_half(c.scale[0]);
    b[0] = (uint8_t)(h & 0xFF); b[1] = (uint8_t)(h >> 8);
    for (int k = 0; k < 16; ++k)
        b[2 + k] = (uint8_t)(c.codes[4 * k] | (c.codes[4 * k + 1] << 2) | (c.codes[4 * k + 2] << 4) | (c.codes[4 * k + 3] << 6));
}

// Integer code bias used for a type (stored code is unsigned; decoded integer is code - bias).
inline int code_bias(QType t) { const Mapping* m = find_mapping(t); return m ? m->code_bias : 0; }

// Scale / min pair of Q4_K / Q5_K group j (6-bit values).
inline void scale_min_pair(int j, const uint8_t* q12, uint8_t* d, uint8_t* m) { dequant::get_scale_min_k4(j, q12, d, m); }

}  // namespace canon
}  // namespace baloon

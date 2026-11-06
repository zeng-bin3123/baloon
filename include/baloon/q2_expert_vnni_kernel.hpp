// q2_expert_vnni_kernel.hpp - CPU Q2_0 expert kernel: Q8_1-style activation quantisation, 2-bit code unpacking,
// offset-corrected row dot  sum (c-1)*xhat = sum c*xhat - sum xhat, the gate/up/down layout of one expert blob,
// tail-mask / multi-token dispatch arithmetic and the CPU feature gate.
//
// The row dot is an exact integer computation per 32-activation chunk (the form a VNNI kernel evaluates), so a
// SIMD implementation can be checked bit-for-bit against it.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "baloon/f16_conversion.hpp"

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#define BALOON_Q2E_HAVE_CPUID 1
#endif

namespace baloon {
namespace q2_expert {

// ---- geometry ---------------------------------------------------------------------------------------------
constexpr uint32_t H = 2560;    // hidden size
constexpr uint32_t FF = 640;    // expert intermediate width
constexpr uint32_t QK = 64;     // weights per Q2_0 block
constexpr uint32_t QKA = 32;    // activations per quantisation chunk
constexpr uint32_t SC_GU = H / QK;   // blocks per gate/up row (40)
constexpr uint32_t SC_D = FF / QK;   // blocks per down row (10)
constexpr uint32_t ROW_GU = SC_GU * 16;  // code bytes per gate/up row
constexpr uint32_t ROW_D = SC_D * 16;    // code bytes per down row

// Blob layout: [gate/up codes: 2*FF rows][gate/up scales: 2*FF rows * SC_GU fp16][down codes: H rows][down scales].
constexpr uint64_t GU_CODES_OFF = 0;
constexpr uint64_t GU_CODES_BYTES = (uint64_t)2 * FF * ROW_GU;
constexpr uint64_t GU_SCALES_OFF = GU_CODES_OFF + GU_CODES_BYTES;
constexpr uint64_t GU_SCALES_BYTES = (uint64_t)2 * FF * SC_GU * 2;
constexpr uint64_t D_CODES_OFF = GU_SCALES_OFF + GU_SCALES_BYTES;
constexpr uint64_t D_CODES_BYTES = (uint64_t)H * ROW_D;
constexpr uint64_t D_SCALES_OFF = D_CODES_OFF + D_CODES_BYTES;
constexpr uint64_t D_SCALES_BYTES = (uint64_t)H * SC_D * 2;
constexpr uint64_t BLOB = D_SCALES_OFF + D_SCALES_BYTES;

static_assert(BLOB == 1382400, "one expert blob is exactly 1,382,400 bytes");
static_assert(SC_GU == 40 && SC_D == 10, "block counts per row");
static_assert(H % QK == 0 && FF % QK == 0, "widths are whole weight blocks");
static_assert(QK == 2 * QKA, "a weight block spans two activation chunks");
static_assert(16 * 4 == QK, "16 code bytes hold 64 codes");

// Row slot of gate / up for expert row r: interleaved (2r gate, 2r+1 up).
constexpr uint32_t gate_slot(uint32_t r) { return 2 * r; }
constexpr uint32_t up_slot(uint32_t r) { return 2 * r + 1; }

inline const uint8_t* gu_codes(const uint8_t* blob, uint32_t slot) { return blob + GU_CODES_OFF + (uint64_t)slot * ROW_GU; }
inline const uint8_t* gu_scales(const uint8_t* blob, uint32_t slot) { return blob + GU_SCALES_OFF + (uint64_t)slot * SC_GU * 2; }
inline const uint8_t* down_codes(const uint8_t* blob, uint32_t row) { return blob + D_CODES_OFF + (uint64_t)row * ROW_D; }
inline const uint8_t* down_scales(const uint8_t* blob, uint32_t row) { return blob + D_SCALES_OFF + (uint64_t)row * SC_D * 2; }

// ---- scalar helpers ---------------------------------------------------------------------------------------
inline int clamp127(int v) { return v < -127 ? -127 : (v > 127 ? 127 : v); }
inline uint16_t load_u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }

// Code k (0..3) of a packed byte, least significant first.
constexpr uint32_t code_of(uint32_t byte, uint32_t k) { return (byte >> (2 * k)) & 3u; }
constexpr uint32_t pack4(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3) { return c0 + 4 * c1 + 16 * c2 + 64 * c3; }
// Weight j (0..63) of block b lives in byte b*16 + j/4 at bit pair j%4.
constexpr uint32_t code_byte_index(uint32_t b, uint32_t j) { return b * 16 + j / 4; }
inline uint32_t block_code(const uint8_t* codes, uint32_t b, uint32_t j) { return code_of(codes[code_byte_index(b, j)], j % 4); }
// Decoded value of a code: c - 1, in [-1, 2].
constexpr int decode_code(uint32_t c) { return (int)c - 1; }

// Unpack a 16-byte block into 64 codes (value j = code j%4 of byte j/4).
inline void unpack_block(const uint8_t* block16, uint8_t out[64]) {
    for (uint32_t j = 0; j < 64; ++j) out[j] = (uint8_t)code_of(block16[j / 4], j % 4);
}

// ---- activation quantisation (Q8_1 style: per-32 scale, int8 in [-127,127], chunk sum hx) -----------------
struct ActQ {
    std::vector<int8_t> q;
    std::vector<float> d;
    std::vector<int32_t> hx;
};

inline void act_quant(const float* x, uint32_t n, int8_t* q, float* d, int32_t* hx) {
    const uint32_t nch = n / QKA;
    for (uint32_t c = 0; c < nch; ++c) {
        const float* xc = x + (size_t)c * QKA;
        float amax = 0.f;
        for (uint32_t i = 0; i < QKA; ++i) amax = std::max(amax, std::fabs(xc[i]));
        const float dd = amax / 127.f;
        const float id = dd != 0.f ? 1.f / dd : 0.f;
        int32_t s = 0;
        for (uint32_t i = 0; i < QKA; ++i) {
            const int v = clamp127((int)std::nearbyint(xc[i] * id));
            q[(size_t)c * QKA + i] = (int8_t)v;
            s += v;
        }
        d[c] = dd;
        hx[c] = s;
    }
}

inline ActQ act_quant(const float* x, uint32_t n) {
    ActQ a;
    a.q.assign(n, 0); a.d.assign(n / QKA, 0.f); a.hx.assign(n / QKA, 0);
    act_quant(x, n, a.q.data(), a.d.data(), a.hx.data());
    return a;
}

// ---- row dot ----------------------------------------------------------------------------------------------
// Integer part for one 32-value half (h = 0 low, 1 high) of block b: sum c*q.
inline int32_t half_code_dot(const uint8_t* codes, uint32_t b, uint32_t h, const int8_t* qchunk) {
    int32_t s = 0;
    for (uint32_t i = 0; i < 32; ++i) s += (int32_t)block_code(codes, b, h * 32 + i) * (int32_t)qchunk[i];
    return s;
}

// Same half with the code decoded first: sum (c-1)*q (used to check the offset identity).
inline int32_t half_decoded_dot(const uint8_t* codes, uint32_t b, uint32_t h, const int8_t* qchunk) {
    int32_t s = 0;
    for (uint32_t i = 0; i < 32; ++i) s += (int32_t)decode_code(block_code(codes, b, h * 32 + i)) * (int32_t)qchunk[i];
    return s;
}

// Row dot over nb blocks: sum_b d_w[b] * sum_h xd[2b+h] * (sum c*q - hx[2b+h]).
inline float row_dot(const uint8_t* codes, const uint8_t* scales_fp16, uint32_t nb, const int8_t* q, const float* xd,
                     const int32_t* hx) {
    float acc = 0.f;
    for (uint32_t b = 0; b < nb; ++b) {
        const float wd = f16::half_to_float(load_u16(scales_fp16 + 2 * (size_t)b));
        for (uint32_t h = 0; h < 2; ++h) {
            const uint32_t ch = 2 * b + h;
            const int32_t s = half_code_dot(codes, b, h, q + (size_t)ch * QKA);
            acc += wd * xd[ch] * (float)(s - hx[ch]);
        }
    }
    return acc;
}

// Multi-token row dot: the block codes are unpacked once and shared by all nt tokens. Per token the float
// accumulation order is identical to row_dot, so results are bit-identical.
// q: nt activations of nb*QK values each (token stride q_stride), xd/hx: stride nb*2 per token.
inline void row_dot_multi(const uint8_t* codes, const uint8_t* scales_fp16, uint32_t nb, uint32_t nt, const int8_t* q,
                          size_t q_stride, const float* xd, const int32_t* hx, size_t ch_stride, float* out) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t b = 0; b < nb; ++b) {
        uint8_t c[64];
        unpack_block(codes + (size_t)b * 16, c);
        const float wd = f16::half_to_float(load_u16(scales_fp16 + 2 * (size_t)b));
        for (uint32_t t = 0; t < nt; ++t) {
            for (uint32_t h = 0; h < 2; ++h) {
                const uint32_t ch = 2 * b + h;
                const int8_t* qc = q + t * q_stride + (size_t)ch * QKA;
                int32_t s = 0;
                for (uint32_t i = 0; i < 32; ++i) s += (int32_t)c[h * 32 + i] * (int32_t)qc[i];
                acc[t] += wd * xd[t * ch_stride + ch] * (float)(s - hx[t * ch_stride + ch]);
            }
        }
    }
    for (uint32_t t = 0; t < nt; ++t) out[t] = acc[t];
}

// ---- tail masks and block loop ----------------------------------------------------------------------------
constexpr uint32_t tail_mask(uint32_t nb) { return nb >= 8 ? 0xFFu : ((1u << nb) - 1u); }
constexpr uint32_t group_size(uint32_t nblocks, uint32_t b0) { return nblocks - b0 < 8 ? nblocks - b0 : 8; }

// ---- token dispatch ---------------------------------------------------------------------------------------
constexpr uint32_t dispatch_nt(uint32_t n) { return n >= 8 ? 8 : n; }
constexpr bool falls_back(uint32_t n) { return n < 1 || n > 8; }

// ---- expert forward ---------------------------------------------------------------------------------------
inline float silu(float v) { return v / (1.f + std::exp(-v)); }

// One token: out[H] = down( silu(gate(x)) * up(x) ).
inline void expert_forward(const uint8_t* blob, const float* x, float* out) {
    ActQ ax = act_quant(x, H);
    std::vector<float> act(FF);
    for (uint32_t r = 0; r < FF; ++r) {
        const float g = row_dot(gu_codes(blob, gate_slot(r)), gu_scales(blob, gate_slot(r)), SC_GU, ax.q.data(), ax.d.data(), ax.hx.data());
        const float u = row_dot(gu_codes(blob, up_slot(r)), gu_scales(blob, up_slot(r)), SC_GU, ax.q.data(), ax.d.data(), ax.hx.data());
        act[r] = silu(g) * u;
    }
    ActQ aa = act_quant(act.data(), FF);
    for (uint32_t r = 0; r < H; ++r)
        out[r] = row_dot(down_codes(blob, r), down_scales(blob, r), SC_D, aa.q.data(), aa.d.data(), aa.hx.data());
}

// Many tokens (x: nt*H, out: nt*H). 1..8 tokens share each weight row; other counts run token by token.
inline void expert_forward_multi(const uint8_t* blob, const float* x, uint32_t nt, float* out) {
    if (nt == 0) return;
    if (falls_back(nt)) {
        for (uint32_t t = 0; t < nt; ++t) expert_forward(blob, x + (size_t)t * H, out + (size_t)t * H);
        return;
    }
    const uint32_t n = dispatch_nt(nt);  // == nt on the fast path
    std::vector<int8_t> qx((size_t)n * H);
    std::vector<float> dx((size_t)n * (H / QKA));
    std::vector<int32_t> hx((size_t)n * (H / QKA));
    for (uint32_t t = 0; t < n; ++t)
        act_quant(x + (size_t)t * H, H, qx.data() + (size_t)t * H, dx.data() + (size_t)t * (H / QKA), hx.data() + (size_t)t * (H / QKA));
    std::vector<float> act((size_t)n * FF);
    for (uint32_t r = 0; r < FF; ++r) {
        float g[8], u[8];
        row_dot_multi(gu_codes(blob, gate_slot(r)), gu_scales(blob, gate_slot(r)), SC_GU, n, qx.data(), H, dx.data(), hx.data(), H / QKA, g);
        row_dot_multi(gu_codes(blob, up_slot(r)), gu_scales(blob, up_slot(r)), SC_GU, n, qx.data(), H, dx.data(), hx.data(), H / QKA, u);
        for (uint32_t t = 0; t < n; ++t) act[(size_t)t * FF + r] = silu(g[t]) * u[t];
    }
    std::vector<int8_t> qa((size_t)n * FF);
    std::vector<float> da((size_t)n * (FF / QKA));
    std::vector<int32_t> ha((size_t)n * (FF / QKA));
    for (uint32_t t = 0; t < n; ++t)
        act_quant(act.data() + (size_t)t * FF, FF, qa.data() + (size_t)t * FF, da.data() + (size_t)t * (FF / QKA), ha.data() + (size_t)t * (FF / QKA));
    for (uint32_t r = 0; r < H; ++r) {
        float o[8];
        row_dot_multi(down_codes(blob, r), down_scales(blob, r), SC_D, n, qa.data(), FF, da.data(), ha.data(), FF / QKA, o);
        for (uint32_t t = 0; t < n; ++t) out[(size_t)t * H + r] = o[t];
    }
}

// ---- cpu feature gate -------------------------------------------------------------------------------------
struct CpuFeatures {
    bool avx512f = false, avx512bw = false, avx512vl = false, avx512_vnni = false, avx512_vbmi = false;
    constexpr bool usable() const { return avx512f && avx512bw && avx512vl && avx512_vnni && avx512_vbmi; }
};

constexpr bool bit(uint32_t x, uint32_t i) { return ((x >> i) & 1u) == 1u; }

// Feature detection from cpuid leaf-7 registers; a CPU whose max leaf is below 7 reports nothing.
constexpr CpuFeatures detect(uint32_t max_leaf, uint32_t ebx, uint32_t ecx) {
    CpuFeatures f{};
    if (max_leaf < 7) return f;
    f.avx512f = bit(ebx, 16);
    f.avx512bw = bit(ebx, 30);
    f.avx512vl = bit(ebx, 31);
    f.avx512_vnni = bit(ecx, 11);
    f.avx512_vbmi = bit(ecx, 1);
    return f;
}

inline CpuFeatures detect_host() {
#ifdef BALOON_Q2E_HAVE_CPUID
    const unsigned max_leaf = __get_cpuid_max(0, nullptr);
    if (max_leaf < 7) return CpuFeatures{};
    unsigned a = 0, b = 0, c = 0, d = 0;
    __cpuid_count(7, 0, a, b, c, d);
    return detect(max_leaf, b, c);
#else
    return CpuFeatures{};
#endif
}

}  // namespace q2_expert
}  // namespace baloon


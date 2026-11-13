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
// row_geometry: a row over n_in inputs has n_in / 64 blocks of 16 code bytes - gate/up rows (H inputs) 40 blocks and
// 640 code bytes, down rows (FF inputs) 10 blocks and 160 code bytes.
constexpr uint32_t row_blocks(uint32_t n_in) { return n_in / QK; }
constexpr uint32_t row_code_bytes(uint32_t n_in) { return row_blocks(n_in) * 16; }
constexpr uint32_t SC_GU = row_blocks(H);       // blocks per gate/up row (40)
constexpr uint32_t SC_D = row_blocks(FF);       // blocks per down row (10)
constexpr uint32_t ROW_GU = row_code_bytes(H);  // code bytes per gate/up row (640)
constexpr uint32_t ROW_D = row_code_bytes(FF);  // code bytes per down row (160)
static_assert(SC_GU == 40 && SC_D == 10 && ROW_GU == 640 && ROW_D == 160, "row_geometry");

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

// blob_size: gate/up are 2 FF rows of (640 code + 80 scale) bytes, down is H rows of (160 + 20); sum 1,382,400.
constexpr uint32_t kGuRowBytes = ROW_GU + SC_GU * 2;
constexpr uint32_t kDownRowBytes = ROW_D + SC_D * 2;
static_assert(kGuRowBytes == 640 + 80 && kDownRowBytes == 160 + 20, "blob_size: row records");
static_assert((uint64_t)2 * FF * kGuRowBytes + (uint64_t)H * kDownRowBytes == BLOB && BLOB == 1382400, "blob_size");

// Row slot of gate / up for expert row r: interleaved (2r gate, 2r+1 up).
// gate_up_interleave: row r's gate is slot 2r and its up is slot 2r + 1 - distinct, in range, both mapping back to r.
constexpr uint32_t gate_slot(uint32_t r) { return 2 * r; }
constexpr uint32_t up_slot(uint32_t r) { return 2 * r + 1; }
constexpr uint32_t slot_row(uint32_t slot) { return slot / 2; }
constexpr bool slot_is_up(uint32_t slot) { return slot % 2 == 1; }
constexpr bool gate_up_interleave() {
    for (uint32_t r = 0; r < FF; ++r)
        if (up_slot(r) >= 2 * FF || slot_row(gate_slot(r)) != r || slot_row(up_slot(r)) != r || slot_is_up(gate_slot(r)) ||
            !slot_is_up(up_slot(r)))
            return false;
    return true;
}
static_assert(gate_up_interleave(), "gate_up_interleave");

inline const uint8_t* gu_codes(const uint8_t* blob, uint32_t slot) { return blob + GU_CODES_OFF + (uint64_t)slot * ROW_GU; }
inline const uint8_t* gu_scales(const uint8_t* blob, uint32_t slot) { return blob + GU_SCALES_OFF + (uint64_t)slot * SC_GU * 2; }
inline const uint8_t* down_codes(const uint8_t* blob, uint32_t row) { return blob + D_CODES_OFF + (uint64_t)row * ROW_D; }
inline const uint8_t* down_scales(const uint8_t* blob, uint32_t row) { return blob + D_SCALES_OFF + (uint64_t)row * SC_D * 2; }

// ---- scalar helpers ---------------------------------------------------------------------------------------
// clampQ_range: the clamp applied after rounding keeps activations in [-127, 127] and is idempotent.
constexpr int clamp127(int v) { return v < -127 ? -127 : (v > 127 ? 127 : v); }
constexpr bool clamp_range_ok() {
    for (int v = -1000; v <= 1000; ++v)
        if (clamp127(v) < -127 || clamp127(v) > 127 || clamp127(clamp127(v)) != clamp127(v)) return false;
    return true;
}
static_assert(clamp_range_ok(), "clampQ_range");
// vnni_lane_bound: a VNNI lane adds four u8 code * s8 activation products, bounded by 4 * 3 * 127 = 1524.
// zsum_bound: a dot of len code (0..3) * activation (|x| <= 127) products is bounded by 381 * len.
constexpr int32_t kProductMax = 3 * 127;
constexpr int64_t code_dot_bound(uint32_t len) { return (int64_t)kProductMax * len; }
static_assert(kProductMax == 381 && code_dot_bound(0) == 0 && code_dot_bound(64) == 24384, "zsum_bound");
constexpr int32_t kVnniLaneMax = (int32_t)code_dot_bound(4);
// chunk_dot_exact_in_f32: a 32-element chunk's integer dot is at most 12192 < 2^24, so its f32 conversion is exact.
constexpr int32_t kChunkDotMax = (int32_t)code_dot_bound(QKA);
static_assert(kChunkDotMax == 12192 && kChunkDotMax < (1 << 24), "chunk_dot_exact_in_f32");
constexpr float chunk_dot_to_f32(int32_t chunk_dot) { return (float)chunk_dot; }  // exact for |chunk_dot| <= kChunkDotMax
static_assert(kVnniLaneMax == 1524 && kVnniLaneMax < 2147483647, "vnni_lane_bound");
inline uint16_t load_u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }

// Code k (0..3) of a packed byte, least significant first.
constexpr uint32_t code_of(uint32_t byte, uint32_t k) { return (byte >> (2 * k)) & 3u; }
constexpr uint32_t pack4(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3) { return c0 + 4 * c1 + 16 * c2 + 64 * c3; }
// Weight j (0..63) of block b lives in byte b*16 + j/4 at bit pair j%4.
constexpr uint32_t code_byte_index(uint32_t b, uint32_t j) { return b * 16 + j / 4; }
inline uint32_t block_code(const uint8_t* codes, uint32_t b, uint32_t j) { return code_of(codes[code_byte_index(b, j)], j % 4); }
// Decoded value of a code: c - 1, in [-1, 2].
constexpr int decode_code(uint32_t c) { return (int)c - 1; }

// ctrl_shifts: the VBMI multishift control 0x0E0C0A0806040200 - byte i of it is 2i, selecting shifts 0, 2, .., 14.
constexpr uint64_t kMultishiftCtrl = 0x0E0C0A0806040200ull;
constexpr uint32_t ctrl_byte(uint32_t i) { return (uint32_t)((kMultishiftCtrl >> (8 * i)) & 0xFFu); }
constexpr bool ctrl_shifts() { for (uint32_t i = 0; i < 8; ++i) if (ctrl_byte(i) != 2 * i) return false; return true; }
static_assert(ctrl_shifts(), "ctrl_shifts");

// multishift_code_order: output byte i of a u16 lane b0 + 256 b1, masked with 3, is the i-th code in LSB-first order -
// codes 0..3 from byte 0, codes 4..7 from byte 1 (as in dequantize_q2_0).
constexpr uint32_t multishift_code(uint32_t u16, uint32_t i) { return (u16 >> ctrl_byte(i)) & 3u; }
constexpr bool multishift_code_order() {
    for (uint32_t b0 = 0; b0 < 256; b0 += 5)
        for (uint32_t b1 = 0; b1 < 256; b1 += 7)
            for (uint32_t i = 0; i < 8; ++i)
                if (multishift_code(b0 + 256 * b1, i) != code_of(i < 4 ? b0 : b1, i % 4)) return false;
    return true;
}
static_assert(multishift_code_order(), "multishift_code_order");

// Unpack a 16-byte block into 64 codes (value j = code j%4 of byte j/4) in the multishift form of the AVX-512
// kernel: qword Q holds the u16 of bytes 2Q, 2Q+1, and output byte 8Q + i is multishift_code(u16, i).
inline void unpack_block(const uint8_t* block16, uint8_t out[64]) {
    for (uint32_t qw = 0; qw < 8; ++qw) {
        const uint32_t u16 = (uint32_t)block16[2 * qw] + 256u * block16[2 * qw + 1];
        for (uint32_t i = 0; i < 8; ++i) out[8 * qw + i] = (uint8_t)multishift_code(u16, i);
    }
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
            const int v = clamp127((int)std::nearbyint(xc[i] * id));  // clampQ_range: in [-127, 127]
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
// block_chunk_map: block b of a row uses activation chunks 2b and 2b+1, at int8 offsets 64b and 64b + 32 - the weight
// block spans two activation scale chunks.
constexpr uint32_t block_chunk(uint32_t b, uint32_t h) { return 2 * b + h; }
constexpr size_t chunk_q_offset(uint32_t b, uint32_t h) { return (size_t)block_chunk(b, h) * QKA; }
static_assert(chunk_q_offset(7, 0) == 64 * 7 && chunk_q_offset(7, 1) == 64 * 7 + 32 && chunk_q_offset(1, 0) == QK, "block_chunk_map");
// offset_identity: the code offset splits off as the plain activation sum, sum (c - 1) x = sum c x - sum x.
constexpr int32_t offset_dot(int32_t code_dot, int32_t act_sum) { return code_dot - act_sum; }
static_assert(offset_dot(3 * 5 + 0 * 7, 5 + 7) == (3 - 1) * 5 + (0 - 1) * 7, "offset_identity");

// chunk_kernel_exact: with the chunk correction hx = d_x * sum(xq) (chunkHx) applied once per chunk,
// d_w (d_x sum c x - hx) = d_w d_x sum (c - 1) x - the exact dequantized product over exact integer scales.
constexpr int64_t chunk_hx(int64_t dx, int64_t act_sum) { return dx * act_sum; }
constexpr int64_t chunk_kernel(int64_t dw, int64_t dx, int64_t code_dot, int64_t act_sum) {
    return dw * (dx * code_dot - chunk_hx(dx, act_sum));
}
static_assert(chunk_kernel(3, -7, 1000, 37) == 3 * -7 * (1000 - 37) && chunk_kernel(-2, 5, -762, 127) == -2 * 5 * (-762 - 127),
              "chunk_kernel_exact");
// chunkDotOracle_exact: chunk_dot_oracle is the integer dot minus the stored activation sum; its result is exactly
// sum (c - 1) xq - no rounding happens before the scale is applied.
inline int32_t chunk_dot_oracle(const uint8_t* codes16, uint32_t h, const int8_t* qchunk, int32_t act_sum) {
    int32_t dot = 0;
    for (uint32_t i = 0; i < QKA; ++i) dot += (int32_t)code_of(codes16[(h * QKA + i) / 4], i % 4) * (int32_t)qchunk[i];
    return offset_dot(dot, act_sum);
}
// The float form the kernel evaluates per chunk: d_w * d_x * (dot - sum), one correction term per chunk.
inline float chunk_term(float dw, float dx, int32_t code_dot, int32_t act_sum) { return dw * dx * chunk_dot_to_f32(offset_dot(code_dot, act_sum)); }

// Integer part for one 32-value half (h = 0 low, 1 high) of block b: sum c*q.
// |result| <= code_dot_bound(32) by zsum_bound.
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
            const uint32_t ch = block_chunk(b, h);  // block_chunk_map: chunks 2b, 2b+1 at int8 offsets 64b, 64b + 32
            const int32_t s = chunk_dot_oracle(codes + (size_t)b * 16, h, q + chunk_q_offset(b, h), hx[ch]);  // = sum (c-1) xq
            acc += chunk_term(wd, xd[ch], s, 0);
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
// laneMask_spec: the scales8 lane mask is 0xFFFF for a full batch, else (1 << 2nb) - 1 - it always enables exactly the
// 2 nb chunk lanes of the batch (two 32-value chunks per block).
constexpr uint32_t lane_mask(uint32_t nb) { return nb >= 8 ? 0xFFFFu : (1u << (2 * nb)) - 1u; }
constexpr bool lane_mask_spec() {
    for (uint32_t nb = 0; nb <= 8; ++nb) if (lane_mask(nb) != (uint32_t)((1ull << (2 * nb)) - 1)) return false;
    return true;
}
static_assert(lane_mask_spec(), "laneMask_spec");
// batchSizes_cover: the b0 += 8 loop takes batches of min(8, n - b0) blocks; they cover every block exactly once and
// none exceeds 8.
constexpr uint32_t batch_size(uint32_t nblocks, uint32_t b0) { return nblocks - b0 < 8 ? nblocks - b0 : 8; }
constexpr bool batch_sizes_cover(uint32_t max_n) {
    for (uint32_t n = 0; n <= max_n; ++n) {
        uint32_t covered = 0;
        for (uint32_t b0 = 0; b0 < n; b0 += 8) {
            const uint32_t nb = batch_size(n, b0);
            if (nb == 0 || nb > 8) return false;
            covered += nb;
        }
        if (covered != n) return false;
    }
    return true;
}
static_assert(batch_sizes_cover(200), "batchSizes_cover");
constexpr uint32_t tail_mask(uint32_t nb) { return nb >= 8 ? 0xFFu : ((1u << nb) - 1u); }
constexpr uint32_t group_size(uint32_t nblocks, uint32_t b0) { return nblocks - b0 < 8 ? nblocks - b0 : 8; }

// ---- token dispatch ---------------------------------------------------------------------------------------
// dispatchNT_exact: the _multi entry points switch on cases 1..7 with default 8; for 1..8 tokens the dispatched
// template width equals the token count.
constexpr uint32_t dispatch_nt(uint32_t n) { return (n >= 1 && n <= 7) ? n : 8; }
constexpr bool dispatch_nt_exact() { for (uint32_t n = 1; n <= 8; ++n) if (dispatch_nt(n) != n) return false; return true; }
static_assert(dispatch_nt_exact() && dispatch_nt(9) == 8 && dispatch_nt(0) == 8, "dispatchNT_exact");
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
// cpuUsable_iff: the gate is usable iff AVX512 F (EBX.16), BW (EBX.30), VL (EBX.31), VNNI (ECX.11) and VBMI (ECX.1)
// are all set; dropping any one (VBMI alone, say, on an AVX512-VNNI part) makes it unusable, and reason() names the
// first missing feature individually ("ok" when usable).
struct CpuFeatures {
    bool avx512f = false, avx512bw = false, avx512vl = false, avx512_vnni = false, avx512_vbmi = false;
    constexpr bool usable() const { return avx512f && avx512bw && avx512vl && avx512_vnni && avx512_vbmi; }
    constexpr const char* reason() const {
        return !avx512f ? "AVX-512F is missing" : !avx512bw ? "AVX-512BW is missing" : !avx512vl ? "AVX-512VL is missing"
             : !avx512_vnni ? "AVX-512 VNNI is missing" : !avx512_vbmi ? "AVX-512 VBMI is missing" : "ok";
    }
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

constexpr bool cpu_usable_iff() {
    for (uint32_t m = 0; m < 32; ++m) {
        const uint32_t ebx = (m & 1 ? 1u << 16 : 0) | (m & 2 ? 1u << 30 : 0) | (m & 4 ? 1u << 31 : 0);
        const uint32_t ecx = (m & 8 ? 1u << 11 : 0) | (m & 16 ? 1u << 1 : 0);
        if (detect(7, ebx, ecx).usable() != (m == 31)) return false;
    }
    return true;
}
static_assert(cpu_usable_iff(), "cpuUsable_iff");
// cpuUsable_witness: the minimal EBX/ECX words with exactly those bits are accepted; without VBMI they are rejected.
constexpr uint32_t kGateEbx = (1u << 16) | (1u << 30) | (1u << 31), kGateEcx = (1u << 11) | (1u << 1);
static_assert(detect(7, kGateEbx, kGateEcx).usable() && !detect(7, kGateEbx, 1u << 11).usable(), "cpuUsable_witness");

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


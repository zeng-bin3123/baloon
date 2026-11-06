// prefill_gpu_kernels.hpp - index arithmetic, layout and numeric-range contracts of the batched-prefill CUDA kernels
// (hyper-connection, gated-delta-net, MoE dequant/route/combine, QSA helpers, KV append). Every kernel is a flat grid over
// a row-major buffer: the grid must cover the buffer, every decomposed index must stay inside its buffer, the packed
// expert blob must sum to one expert's byte size, bf16 rounding yields a 16-bit pattern and is exact on bf16 values,
// and the int8 KV quantizer clamps to [-127, 127].
#pragma once
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>

namespace baloon {
namespace prefill_kernels {

// ---- constants -------------------------------------------------------------------------------------------------
constexpr uint64_t kN = 2560;
constexpr uint64_t kHC = 4;
constexpr uint64_t kD = kN * kHC;          // 10240: four hyper-connection streams
constexpr uint64_t kLR = 320;
constexpr uint64_t kS = 128;
constexpr uint64_t kHK = 16;
constexpr uint64_t kHV = 48;
constexpr uint64_t kC = 10240;             // GDN conv channels = q | k | v
constexpr uint64_t kZV = kHV * kS;         // 6144
constexpr uint64_t kQHeads = 24, kQHeadDim2 = 512;   // q projection: 24 heads of (query 256 + gate 256)
constexpr uint64_t kQFullWidth = kQHeads * kQHeadDim2;
constexpr uint64_t kRoutedSlots = 10;      // routed experts per token
constexpr uint64_t kSwigluWidth = 640;     // swiglu row width; gate/up interleaved row is 2*640
constexpr uint64_t kRecRowGroups = 4, kRecRowsPerGroup = 32;

static_assert(kD == 10240, "hc width");
static_assert(kC == 2 * kHK * kS + kHV * kS, "qkv channels");
static_assert(kZV == 6144, "value width");
static_assert(kQFullWidth == 12288, "q full width");

// ---- launch geometry -------------------------------------------------------------------------------------------
inline uint64_t blocks_for(uint64_t n, uint64_t t) { return (n + t - 1) / t; }
inline uint64_t route_blocks(uint64_t T) { return (T + 7) / 8; }               // 8 warps per block, one token per warp
constexpr uint64_t kRouteWarps = 8;
constexpr uint64_t kGdnConvThreads = 128;
constexpr uint64_t gdn_conv_blocks() { return kC / kGdnConvThreads; }
constexpr uint64_t kGdnL2Blocks = 2 * kHK;                                       // one block of S threads per q/k head
inline uint64_t block_sum_warps(uint64_t b) { return (b + 31) / 32; }          // sh[32] holds the warp partials
constexpr uint64_t kBlockSumMaxWarps = 32;
inline uint64_t lane_of(uint64_t tid) { return tid % 32; }
inline uint64_t warp_of(uint64_t tid) { return tid / 32; }
// GDN recurrence: 128 state rows split over 4 row groups of 32
inline uint64_t rec_group(uint64_t row) { return row / kRecRowsPerGroup; }
inline uint64_t rec_row_in_group(uint64_t row) { return row % kRecRowsPerGroup; }
inline uint64_t gdn_state_index(uint64_t row, uint64_t head, uint64_t col) { return (row * kHV + head) * kS + col; }
constexpr uint64_t kGdnStateElems = kS * kHV * kS;
// RoPE rotates pairs (p, p+32) over 64 dims
constexpr uint64_t kRopePairs = 32, kRopeDims = 64;
inline uint64_t rope_partner(uint64_t i) { return i < kRopePairs ? i + kRopePairs : i - kRopePairs; }

// ---- hyper-connection / row-major index maps -------------------------------------------------------------------
inline uint64_t hc_index(uint64_t t, uint64_t c, uint64_t d) { return t * kD + c * kN + d; }
inline uint64_t hc_token(uint64_t idx) { return idx / kD; }
inline uint64_t hc_stream(uint64_t idx) { return (idx % kD) / kN; }
inline uint64_t hc_channel(uint64_t idx) { return idx % kN; }
inline uint64_t combine_slot(uint64_t t, uint64_t k) { return t * kRoutedSlots + k; }
inline uint64_t swiglu_flat(uint64_t r, uint64_t k) { return r * kSwigluWidth + k; }
inline uint64_t swiglu_il_gate(uint64_t r, uint64_t k) { return r * (2 * kSwigluWidth) + 2 * k; }
inline uint64_t swiglu_il_up(uint64_t r, uint64_t k) { return r * (2 * kSwigluWidth) + 2 * k + 1; }
inline uint64_t row_index(uint64_t q, uint64_t per, uint64_t r) { return q * per + r; }   // row-major cell
inline uint64_t gather_index(uint64_t r, uint64_t j, uint64_t per) { return r * per + j; } // gather_rows16 uint4 units

// ---- expert blob layout ----------------------------------------------------------------------------------------
constexpr uint64_t kGuRows = 1280, kGuCols = 2560, kGuCodeBytesPerRow = 640, kGuScalesPerRow = 40;
constexpr uint64_t kDnRows = 2560, kDnCols = 640, kDnCodeBytesPerRow = 160, kDnScalesPerRow = 10;
constexpr uint64_t kScaleGroup = 64;       // weights per scale
constexpr uint64_t O_GU_CODES = 0;
constexpr uint64_t O_D_CODES = kGuRows * kGuCodeBytesPerRow;                       // 1280*640
constexpr uint64_t O_GU_SC = O_D_CODES + kDnRows * kDnCodeBytesPerRow;             // + 2560*160
constexpr uint64_t O_D_SC = O_GU_SC + kGuRows * kGuScalesPerRow * 2;               // + 1280*40 f16 scales
constexpr uint64_t BLOB_BYTES = O_D_SC + kDnRows * kDnScalesPerRow * 2;            // + 2560*10 f16 scales
static_assert(BLOB_BYTES == 1382400, "expert blob size");
static_assert(O_D_CODES < O_GU_SC && O_GU_SC < O_D_SC && O_D_SC < BLOB_BYTES, "blob regions ordered");

inline uint32_t code2(uint32_t c, uint32_t k) { return (c >> (2 * k)) & 3u; }
inline int32_t dequant_val(uint32_t c, uint32_t k) { return (int32_t)code2(c, k) - 1; }   // in [-1, 2]
inline uint64_t gu_out_index(uint64_t row, uint64_t byte, uint64_t k) { return row * kGuCols + byte * 4 + k; }
inline uint64_t down_out_index(uint64_t row, uint64_t byte, uint64_t k) { return row * kDnCols + byte * 4 + k; }
inline uint64_t gu_scale_index(uint64_t row, uint64_t byte) { return row * kGuScalesPerRow + (byte * 4) / kScaleGroup; }
inline uint64_t down_scale_index(uint64_t row, uint64_t byte) { return row * kDnScalesPerRow + (byte * 4) / kScaleGroup; }
inline uint64_t gu_code_offset(uint64_t row, uint64_t byte) { return O_GU_CODES + row * kGuCodeBytesPerRow + byte; }
inline uint64_t down_code_offset(uint64_t row, uint64_t byte) { return O_D_CODES + row * kDnCodeBytesPerRow + byte; }
inline uint64_t gu_scale_offset(uint64_t row, uint64_t byte) { return O_GU_SC + gu_scale_index(row, byte) * 2; }
inline uint64_t down_scale_offset(uint64_t row, uint64_t byte) { return O_D_SC + down_scale_index(row, byte) * 2; }

inline float f16_bits_to_f32(uint16_t h) {
    const uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s;
        else { int sh = 0; uint32_t mm = m; while (!(mm & 0x400u)) { mm <<= 1; ++sh; } u = s | ((uint32_t)(113 - sh) << 23) | ((mm & 0x3FFu) << 13); }
    } else if (e == 31) u = s | 0x7F800000u | (m << 13);
    else u = s | ((e + 112) << 23) | (m << 13);
    float f; std::memcpy(&f, &u, 4); return f;
}
// Reference dequant of one gate/up weight (row, col): (code-1) * scale of its 64-group.
inline float gu_weight(const uint8_t* blob, uint64_t row, uint64_t col) {
    const uint64_t byte = col / 4, k = col % 4;
    uint16_t sc; std::memcpy(&sc, blob + gu_scale_offset(row, byte), 2);
    return (float)dequant_val(blob[gu_code_offset(row, byte)], (uint32_t)k) * f16_bits_to_f32(sc);
}
inline float down_weight(const uint8_t* blob, uint64_t row, uint64_t col) {
    const uint64_t byte = col / 4, k = col % 4;
    uint16_t sc; std::memcpy(&sc, blob + down_scale_offset(row, byte), 2);
    return (float)dequant_val(blob[down_code_offset(row, byte)], (uint32_t)k) * f16_bits_to_f32(sc);
}

// ---- numeric formats -------------------------------------------------------------------------------------------
// bf16 rounding (round-to-nearest-even, uint32 wrap) of an f32 bit pattern, as 16 bits.
inline uint32_t bf_bits(uint32_t u) { return (uint32_t)(u + 0x7fffu + ((u >> 16) & 1u)) >> 16; }
inline uint16_t bf_from_float(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)bf_bits(u); }
inline float bf_to_float(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
inline int32_t clamp127(int64_t q) { return q < -127 ? -127 : (q > 127 ? 127 : (int32_t)q); }
// int8 KV quantization of x with scale: round-to-nearest-even, then clamp.
inline int32_t kv_quant(float x, float inv_scale) { return clamp127((int64_t)std::lrintf(x * inv_scale)); }
// KV pool row (page*2 + kvh)*page_size + pos % page_size
inline uint64_t kv_row(uint64_t page, uint64_t kvh, uint64_t page_size, uint64_t slot) { return (page * 2 + kvh) * page_size + slot; }
inline uint64_t kv_pool_rows(uint64_t n_pages, uint64_t page_size) { return n_pages * 2 * page_size; }

}  // namespace prefill_kernels
}  // namespace baloon

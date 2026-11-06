// kern_geometry.hpp - index / layout / selection arithmetic shared by the unit-14 kernels (S-family GEMV code unpack,
// dot-product algebra, S2 expert layout, MMVQ iteration geometry, router/sampler selection, PLE block geometry,
// activation quantizers, Q2 expert layout, expert-pool scheduling, i-quant block decode).
// Everything here is constexpr integer arithmetic (or tiny inline helpers) so tests can check every statement of the
// theory and the kernels can static_assert against it.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace baloon {
namespace kern {

// ======================================================================================= SGemvCodeUnpack
namespace sgemv {
constexpr uint64_t q8k_off(uint64_t i) { return (i / 256) * 292 + 4 + i % 256; }
constexpr uint64_t q8_0_off(uint64_t i) { return (i / 32) * 34 + 2 + i % 32; }
constexpr uint32_t QE = 16;           // elements per lane iteration
constexpr uint32_t LANES = 32;        // lanes per row
constexpr uint64_t lane_iter_start(uint64_t lane, uint64_t t) { return lane * QE + (uint64_t)(LANES * QE) * t; }
// k-th code (k < 8/B*..) of the little-endian word at byte i/PER_BYTE: bits [k*B, (k+1)*B).
constexpr uint32_t code_at(uint64_t word, uint32_t k, uint32_t B) { return (uint32_t)((word >> (k * B)) & ((1u << B) - 1u)); }
constexpr uint32_t group_index(uint64_t i, uint32_t group_shift) { return (uint32_t)(i >> group_shift); }
// Sixteen consecutive elements share one scale iff 16 divides the group size.
constexpr bool lane_same_group(uint32_t group_elems) { return group_elems % 16 == 0; }
constexpr uint32_t bytes_per_word(uint32_t B) { return 4 * B / 8; }   // bytes holding four codes of B bits
constexpr uint64_t grid_rows128(uint64_t n) { return (n + 127) / 128; }
constexpr uint64_t grid_rows_warps(uint64_t n, uint32_t warps_per_block) { return (n + warps_per_block - 1) / warps_per_block; }
}  // namespace sgemv

// ======================================================================================= DotProductAlgebra
namespace dotalg {
inline int64_t dot(const int32_t* a, const int32_t* b, size_t n) {
    int64_t s = 0;
    for (size_t i = 0; i < n; ++i) s += (int64_t)a[i] * b[i];
    return s;
}
// Expand a code byte with four 2-bit fields to one byte per field (dp4a operand).
constexpr uint32_t expand(uint32_t c) { return (c & 3u) | (((c >> 2) & 3u) << 8) | (((c >> 4) & 3u) << 16) | (((c >> 6) & 3u) << 24); }
// Tree reduction with halving strides over n = 2*s lanes. Returns the value of lane 0.
inline int64_t tree_sum(std::vector<int64_t> p, uint32_t lanes) {
    for (uint32_t s = lanes / 2; s >= 1; s /= 2) {
        for (uint32_t t = 0; t < s; ++t) p[t] += p[t + s];
        if (s == 1) break;
    }
    return p[0];
}
constexpr uint32_t chunk_scale_index(uint32_t c) { return c / 2; }
}  // namespace dotalg

// ======================================================================================= S2ExpertLayout
namespace s2layout {
constexpr uint32_t H = 2560, FF = 640, QK = 64;
constexpr uint32_t ROW_GU = H / 4, ROW_D = FF / 4, SC_GU = H / QK, SC_D = FF / QK;
constexpr uint64_t O_D_CODES = (uint64_t)2 * FF * ROW_GU;
constexpr uint64_t O_GU_SCALES = O_D_CODES + (uint64_t)H * ROW_D;
constexpr uint64_t O_D_SCALES = O_GU_SCALES + (uint64_t)2 * FF * SC_GU * 2;
constexpr uint64_t BLOB_BYTES = O_D_SCALES + (uint64_t)H * SC_D * 2;
static_assert(ROW_GU == 640 && ROW_D == 160 && SC_GU == 40 && SC_D == 10, "row geometry");
static_assert(O_D_CODES == 819200 && O_GU_SCALES == 1228800 && O_D_SCALES == 1331200 && BLOB_BYTES == 1382400, "blob offsets");
constexpr uint32_t GU_CHUNKS = H / 32;            // 80 activation chunks per gate/up row
constexpr uint32_t GU_CHUNKS_PER_LANE = (GU_CHUNKS + 31) / 32;   // 3
constexpr uint32_t D_CHUNKS = FF / 32;            // 20
constexpr uint32_t GU_ROWS = 2 * FF;              // 1280 interleaved rows per expert
constexpr uint32_t D_ROWS = H;
// Gate-major output position of interleaved row slot i of hit h; N = n_hits * FF.
constexpr uint64_t gate_major(uint64_t N, uint64_t h, uint64_t i) { return (i % 2) * N + h * FF + i / 2; }
constexpr uint64_t r16(uint64_t n) { return (n + 15) / 16 * 16; }
constexpr uint64_t r256(uint64_t n) { return (n + 255) / 256 * 256; }
// Positions of set lanes under ballot-prefix compaction: popc(mask & ((1<<lane)-1)).
inline uint32_t ballot_prefix(uint64_t mask, uint32_t lane) { return (uint32_t)__builtin_popcountll(mask & ((lane >= 64) ? ~0ull : ((1ull << lane) - 1ull))); }
}  // namespace s2layout

// ======================================================================================= MmvqIterationGeometry
namespace mmvqgeo {
struct Trait { uint32_t threads_per_block, blocks_per_iter; };
// Threads per weight block and blocks per iteration for 128-thread blocks.
constexpr Trait q5_k{16, 8}, q2_0{2, 64}, q3_k{16, 8}, q6_k{32, 4}, iq4_xs{8, 16}, q4_0_like{2, 64}, q8_0{4, 32};
constexpr uint32_t BLOCK_THREADS = 128;
constexpr uint32_t two_warp_bpi(uint32_t bpi) { return bpi * 2 / 4; }   // BPI * NW / WARPS for NW = 2
constexpr bool small_k(uint64_t n_in, uint32_t block_elems, uint32_t bpi) { return n_in / block_elems < bpi; }
constexpr uint64_t grid_rows(uint64_t n, uint32_t rows_per_block) { return (n + rows_per_block - 1) / rows_per_block; }
constexpr uint64_t column_block(uint64_t c, uint64_t n_in, uint64_t i) { return (c * n_in + i) / 32; }
constexpr uint64_t q81_bytes(uint64_t ncols, uint64_t n_in) { return ncols * (n_in / 32) * 36; }
struct WeightEntry { int type; uint32_t elems; uint32_t bytes; };
constexpr WeightEntry weight_table[16] = {{2, 32, 18}, {6, 32, 22}, {8, 32, 34}, {20, 32, 18}, {11, 256, 110}, {12, 256, 144},
                                          {13, 256, 176}, {14, 256, 210}, {23, 256, 136}, {42, 64, 18}, {16, 256, 66}, {17, 256, 74},
                                          {18, 256, 98}, {21, 256, 110}, {22, 256, 82}, {29, 256, 56}};
constexpr const WeightEntry* find_weight(int type) {
    for (const auto& e : weight_table) if (e.type == type) return &e;
    return nullptr;
}
}  // namespace mmvqgeo

// ======================================================================================= RouterSamplerSelection
namespace routersel {
struct Cand { double v; uint32_t i; };
// The (value, index) argmax combine: the challenger wins iff larger, or equal with a smaller index.
constexpr bool wins(double bv, uint32_t bi, double av, uint32_t ai) { return bv > av || (bv == av && bi < ai); }
constexpr Cand comb(Cand a, Cand b) { return wins(b.v, b.i, a.v, a.i) ? b : a; }
constexpr uint32_t greedy_out(uint32_t wi, uint32_t n) { return wi < n ? wi : 0; }
constexpr uint32_t philox_mantissa(uint32_t c0) { return c0 >> 8; }                    // < 2^24
constexpr float philox_uniform_of(uint32_t c0) { return (float)(c0 >> 8) * (1.0f / 16777216.0f); }
constexpr float repeat_penalty(float l, float r) { return l <= 0.f ? l * r : l / r; }
constexpr uint32_t window_start(uint32_t len, uint32_t last_n) { return len - (len < last_n ? len : last_n); }
constexpr uint32_t cut_adjust(uint32_t cut, uint32_t min_keep, uint32_t n_keep) {
    return cut < min_keep ? (min_keep < n_keep ? min_keep : n_keep) : cut;
}
constexpr uint32_t router_threads(uint32_t n_expert) { return ((n_expert < 512 ? n_expert : 512) + 31) / 32 * 32; }
constexpr uint32_t pad16(uint32_t n) { return (n + 15) / 16 * 16; }
constexpr uint32_t router_smem(uint32_t n_expert) { return pad16(n_expert) + n_expert * 8 + n_expert * 4; }
static_assert(router_smem(512) == 6656, "router shared memory for 512 experts");
constexpr double clamp_denominator(double s) { return s > 1.0 / 16384.0 ? s : 1.0 / 16384.0; }
}  // namespace routersel

// ======================================================================================= PleBlockGeometry
namespace plegeo {
constexpr uint32_t NG_HIST = 9, NG_HC = 4, NG_EMBD = 2560, NG_HC_DIM = NG_HC * NG_EMBD, CONV_KERNEL = 4, CONV_DIL = 3;
static_assert(NG_HC_DIM == 10240, "hc_dim");
constexpr uint32_t tap_row(uint32_t nhist, uint32_t kern, uint32_t dil, uint32_t k) { return nhist - (kern - 1 - k) * dil; }
constexpr uint64_t state_index(uint64_t row, uint64_t channel) { return row + (uint64_t)NG_HIST * channel; }   // row-fastest
constexpr uint64_t kernel_weight_index(uint32_t k, uint64_t c) { return k + 4 * c; }
constexpr uint64_t r16(uint64_t n) { return (n + 15) / 16 * 16; }
constexpr uint64_t float_ws_floats() { return 5ull * NG_HC_DIM + NG_EMBD + NG_HC; }
constexpr uint64_t scratch_bytes() { return r16(float_ws_floats() * 4) + r16((uint64_t)NG_EMBD / 32 * 34) + (uint64_t)NG_EMBD * 2 + 256; }
static_assert(scratch_bytes() == 223152, "PLE scratch bytes");
// Byte ranges [a, a+ab) and [b, b+bb) overlap (non-empty ranges).
constexpr bool overlap(uint64_t a, uint64_t ab, uint64_t b, uint64_t bb) { return a < b ? (b - a < ab) : (a - b < bb); }
constexpr int sgn(float s) { return s > 0.f ? 1 : (s < 0.f ? -1 : 0); }
}  // namespace plegeo

// ======================================================================================= ActivationQuantize
namespace actq {
constexpr int clamp_i(int lo, int hi, int v) { return v < lo ? lo : (hi < v ? hi : v); }
constexpr int clamp_ggml(int v) { return clamp_i(-128, 127, v); }      // ggml reference rule
constexpr int clamp_cpu(int v) { return clamp_i(-127, 127, v); }       // CPU-parity rule (symmetric)
constexpr uint32_t Q8_0_BYTES = 34, Q8_1_BYTES = 36, Q8_K_BYTES = 292;
static_assert(2 + 32 == Q8_0_BYTES && 4 + 32 == Q8_1_BYTES && 4 + 256 + 32 == Q8_K_BYTES, "block sizes");
// nearest_int via the 1.5*2^23 magic constant (exact for |x| < 2^22).
inline int nearest_int(float x) {
    float v = x + 12582912.0f;
    int32_t i;
    __builtin_memcpy(&i, &v, 4);
    return (i & 0x007fffff) - 0x00400000;
}
constexpr uint64_t grid_q8_0(uint64_t nb) { return (nb + 127) / 128; }
constexpr uint64_t grid_q8_k(uint64_t nb) { return (nb + 63) / 64; }
constexpr uint64_t grid_q8_1(uint64_t n) { return (n + 255) / 256; }
}  // namespace actq

// ======================================================================================= Q2ExpertKernelLayout
namespace q2geo {
constexpr uint32_t H = 2560, FF = 640, NE = 512, QK = 64, QKA = 32, BB = 18, MAXC = 80, ROW_GU = 640, ROW_D = 160, SC_GU = 40,
                   SC_D = 10, MAXT = 8;
constexpr uint64_t BLOB = 1382400, O_GU_CODES = 0, O_D_CODES = 819200, O_GU_SCALES = 1228800, O_D_SCALES = 1331200;
static_assert(MAXC == H / QKA && ROW_GU == H * 2 / 8 && ROW_D == FF * 2 / 8 && SC_GU == H / QK && SC_D == FF / QK, "geometry");
static_assert(BLOB == 3ull * (H * FF * BB / QK), "blob size");
static_assert(O_D_CODES == 2ull * FF * ROW_GU && O_GU_SCALES == O_D_CODES + (uint64_t)H * ROW_D &&
                  O_D_SCALES == O_GU_SCALES + 2ull * FF * SC_GU * 2 && O_D_SCALES + (uint64_t)H * SC_D * 2 == BLOB, "offsets");
static_assert(QK == 2 * QKA && BB == 2 + QK / 4, "block shape");
constexpr uint32_t tail_mask_bits(uint32_t nb) { return (1u << (2 * nb)) - 1u; }
constexpr uint32_t dispatch_nt(uint32_t n) { return (n >= 1 && n <= 7) ? n : 8; }
}  // namespace q2geo

// ======================================================================================= ExpertPoolScheduling
namespace poolsched {
constexpr uint64_t row_start(uint64_t total, uint64_t parts, uint64_t part) { return total * part / parts; }
struct Piece { uint64_t expert, r0, r1; };
constexpr uint64_t piece_end(uint64_t per, uint64_t r0, uint64_t g1, uint64_t r) { return std::min<uint64_t>(per, r0 + (g1 - r)); }
inline std::vector<Piece> pieces(uint64_t per, uint64_t g0, uint64_t g1) {
    std::vector<Piece> v;
    for (uint64_t r = g0; r < g1;) {
        const uint64_t r0 = r % per, r1 = piece_end(per, r0, g1, r);
        v.push_back({r / per, r0, r1});
        r += r1 - r0;
    }
    return v;
}
constexpr uint32_t pool_workers(uint32_t requested, uint32_t cores) { return std::max<uint32_t>(1, requested > 0 ? requested : cores); }
constexpr uint32_t drain_threads(uint32_t nw, bool host_works) { return nw + (host_works ? 1 : 0); }
constexpr bool use_split(uint32_t n, uint32_t kmax, bool oracle, uint32_t workers) { return !(n > kmax || workers == 1 || oracle); }
constexpr bool use_split(uint32_t n, uint32_t kmax, bool oracle) { return n <= kmax && !oracle; }   // capacity + oracle only
constexpr uint32_t kSplitCap = 16;         // run_split: experts per call
constexpr uint32_t kSplitMultiCap = 96;    // run_split_multi: experts per call (native layers are batched at this size)
// Batches of at most `cap` experts covering n (the last one may be short).
constexpr uint32_t batch_count(uint32_t n, uint32_t cap) { return cap == 0 ? 0 : (n + cap - 1) / cap; }
}  // namespace poolsched

// ======================================================================================= IquantBlockDecode
namespace iqgeo {
constexpr bool supported(uint32_t ty) { return ty == 16 || ty == 17 || ty == 18 || ty == 21 || ty == 22; }
constexpr uint32_t block_bytes(uint32_t ty) {
    return ty == 16 ? 66 : ty == 17 ? 74 : ty == 18 ? 98 : ty == 21 ? 110 : ty == 22 ? 82 : 0;
}
constexpr uint32_t scale_lane(uint32_t l) { return l / 8; }
constexpr uint32_t odd_scale(uint32_t s4) { return 2 * s4 + 1; }
constexpr int apply_sign(bool bit, int y) { return bit ? -y : y; }
constexpr uint32_t group_size(uint32_t nt, uint32_t t0) { return std::min<uint32_t>(4, nt - t0); }
// Highest byte (exclusive) read by chunk j of format ty; must be <= block_bytes(ty).
constexpr uint32_t chunk_read_end(uint32_t ty, uint32_t j) {
    switch (ty) {
        case 16: return 2 + 16 * j + 16;
        case 17: return 66 + 2 * j + 2;
        case 22: return 74 + 2 * j + 2;
        case 18: return 66 + 8 * j + 8;
        case 21: return 106 + j + 1;
        default: return 0;
    }
}
}  // namespace iqgeo

}  // namespace kern
}  // namespace baloon


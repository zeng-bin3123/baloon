// iq_quant.cu - GPU handling of the i-quant / k-quant / Q2_0 block formats of the native expert path.
//   * element decoders dq_* (one element of one block) for IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ1_M IQ4_NL IQ4_XS Q3_K Q2_0
//   * flat / gate+up dequantization to f32 or f16, row gather (embedding) from a quantized table
//   * Q8_1 activation quantization (block = f16 d + f16 s + 32 int8, 36 bytes; s = d * sum(q))
//   * block dot products against Q8_1 activations (vec_dot_*), the matrix-vector product iq_mmvq, and a full single-expert
//     evaluation (gate/up -> SwiGLU -> down) with its scratch/layout helpers.
// The codebook grids (iq2xxs 256 x u64, iq2xs 512, iq2s 1024, iq3xxs 256 x u32, iq3s 512, iq1s 2048 x u64 of int8 lanes) are not
// compiled in: the caller uploads them and passes device pointers in IqGrids (a format whose grid is null is unsupported).
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"
#include "../../cu/kern_cu_blocks.cuh"

namespace baloon {
namespace kg {
namespace {
constexpr int GU_ROWS = 4;                         // rows (waves) per block of the gate+up dequantization
constexpr bool kDeclCommon = true;                 // block structs are decoded by offset; nothing else to declare
constexpr bool kImplCommon = true;
constexpr int QK_K = 256;

struct Geo { int elems; int bytes; };
__host__ __device__ inline Geo geo(Fmt f) {
    switch (f) {
        case Fmt::Q2_0: return {64, 18};
        case Fmt::Q3_K: return {256, 110};
        case Fmt::IQ2_XXS: return {256, 66};
        case Fmt::IQ2_XS: return {256, 74};
        case Fmt::IQ3_XXS: return {256, 98};
        case Fmt::IQ1_M: return {256, 56};
        case Fmt::IQ4_NL: return {32, 18};
        case Fmt::IQ3_S: return {256, 110};
        case Fmt::IQ2_S: return {256, 82};
        case Fmt::IQ4_XS: return {256, 136};
    }
    return {0, 0};
}

bool check(cudaError_t e) { return e == cudaSuccess; }
template <class T> __device__ __forceinline__ T cvt(float v);
template <> __device__ __forceinline__ float cvt<float>(float v) { return v; }
template <> __device__ __forceinline__ uint16_t cvt<uint16_t>(float v) { return kgc::f_to_f16(v); }

__device__ __forceinline__ float warp_sum(float v) { return kgc::wave_sum(v); }
// 16-bit-aligned and 32-bit-aligned little-endian reads
__device__ __forceinline__ int get_int_b2(const void* x, int i32) {
    const uint16_t* p = (const uint16_t*)x;
    return (int)((uint32_t)p[2 * i32] | ((uint32_t)p[2 * i32 + 1] << 16));
}
__device__ __forceinline__ int get_int_b4(const void* x, int i32) { return ((const int*)x)[i32]; }
// 8 nibbles of q4 -> two ints of 4 table bytes each (low nibbles, high nibbles)
__device__ __forceinline__ int2 get_int_from_table_16(int q4, const int8_t* table) {
    int lo = 0, hi = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        lo |= ((int)(uint8_t)table[(q4 >> (8 * k)) & 0xf]) << (8 * k);
        hi |= ((int)(uint8_t)table[(q4 >> (8 * k + 4)) & 0xf]) << (8 * k);
    }
    return make_int2(lo, hi);
}
__device__ __forceinline__ int ggml_cuda_dp4a_(int a, int b, int c) {
    for (int k = 0; k < 4; ++k) c += (int)(int8_t)(a >> (8 * k)) * (int)(int8_t)(b >> (8 * k));
    return c;
}
// 7-bit sign index -> 8 sign bits (bit 7 = parity of the 7 bits, so every pattern has an even number of negatives)
__device__ __forceinline__ uint32_t unpack_ksigns(uint32_t i7) { return (i7 & 127u) | (((uint32_t)__builtin_popcount(i7 & 127u) & 1u) << 7); }

__device__ __forceinline__ float rd_d(const uint8_t* p) { return kgc::f16_to_f((uint16_t)(p[0] | (p[1] << 8))); }
__device__ __forceinline__ uint32_t rd_u32(const uint8_t* p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
__device__ __forceinline__ uint32_t rd_u16(const uint8_t* p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8); }
__device__ __forceinline__ float sgn(uint32_t signs, int j) { return ((signs >> j) & 1u) ? -1.f : 1.f; }
__device__ __forceinline__ float nib_scale(const uint8_t sc, int l) { return (float)((l < 2) ? (sc & 0xf) : (sc >> 4)); }

__device__ float dq_iq2_xxs(const uint8_t* b, int i, const IqGrids& g) {
    const int ib = i / 32, l = (i % 32) / 8, j = i % 8;
    const uint8_t* q = b + 2 + 8 * ib;
    const uint32_t aux = rd_u32(q + 4);
    const float db = rd_d(b) * (0.5f + (float)(aux >> 28)) * 0.25f;
    const uint64_t grid = g.iq2xxs[q[l]];
    return db * (float)((grid >> (8 * j)) & 0xff) * sgn(unpack_ksigns((aux >> (7 * l)) & 127u), j);
}
__device__ float dq_iq2_xs(const uint8_t* b, int i, const IqGrids& g) {
    const int ib = i / 32, l = (i % 32) / 8, j = i % 8;
    const uint32_t q = rd_u16(b + 2 + 2 * (4 * ib + l));
    const float db = rd_d(b) * (0.5f + nib_scale(b[2 + 64 + ib], l)) * 0.25f;
    const uint64_t grid = g.iq2xs[q & 511u];
    return db * (float)((grid >> (8 * j)) & 0xff) * sgn(unpack_ksigns(q >> 9), j);
}
__device__ float dq_iq2_s(const uint8_t* b, int i, const IqGrids& g) {
    const int ib = i / 32, l = (i % 32) / 8, j = i % 8;
    const uint8_t* qs = b + 2; const uint8_t* qh = b + 66; const uint8_t* sc = b + 74;
    const uint32_t idx = (uint32_t)qs[4 * ib + l] | (((uint32_t)qh[ib] << (8 - 2 * l)) & 0x300u);
    const float db = rd_d(b) * (0.5f + nib_scale(sc[ib], l)) * 0.25f;
    const uint64_t grid = g.iq2s[idx];
    return db * (float)((grid >> (8 * j)) & 0xff) * sgn(qs[32 + 4 * ib + l], j);
}
__device__ float dq_iq3_xxs(const uint8_t* b, int i, const IqGrids& g) {
    const int ib = i / 32, l = (i % 32) / 8, j = i % 8;
    const uint8_t* qs = b + 2;
    const uint32_t aux = rd_u32(qs + 64 + 4 * ib);
    const float db = rd_d(b) * (0.5f + (float)(aux >> 28)) * 0.5f;
    const uint32_t gr = (j < 4) ? g.iq3xxs[qs[8 * ib + 2 * l]] : g.iq3xxs[qs[8 * ib + 2 * l + 1]];
    return db * (float)((gr >> (8 * (j & 3))) & 0xff) * sgn(unpack_ksigns((aux >> (7 * l)) & 127u), j);
}
__device__ float dq_iq3_s(const uint8_t* b, int i, const IqGrids& g) {
    const int ib = i / 32, l = (i % 32) / 8, j = i % 8;
    const uint8_t* qs = b + 2; const uint8_t* qh = b + 66; const uint8_t* sg = b + 74; const uint8_t* sc = b + 106;
    const float db = rd_d(b) * (1.0f + 2.0f * (float)((ib & 1) ? (sc[ib / 2] >> 4) : (sc[ib / 2] & 0xf)));
    const uint32_t h = qh[ib];
    const uint32_t idx = (j < 4) ? ((uint32_t)qs[8 * ib + 2 * l] | ((h << (8 - 2 * l)) & 256u))
                                 : ((uint32_t)qs[8 * ib + 2 * l + 1] | ((h << (7 - 2 * l)) & 256u));
    const uint32_t gr = g.iq3s[idx];
    return db * (float)((gr >> (8 * (j & 3))) & 0xff) * sgn(sg[4 * ib + l], j);
}
__device__ float dq_iq1_m(const uint8_t* b, int i, const IqGrids& g) {
    const int ib = i / 32, l = (i % 32) / 8, j = i % 8;
    const uint8_t* qs = b; const uint8_t* qh = b + 32; const uint8_t* sc = b + 48;
    const uint32_t s0 = rd_u16(sc), s1 = rd_u16(sc + 2), s2 = rd_u16(sc + 4), s3 = rd_u16(sc + 6);
    const uint16_t dbits = (uint16_t)((s0 >> 12) | ((s1 >> 8) & 0x00f0u) | ((s2 >> 4) & 0x0f00u) | (s3 & 0xf000u));
    const float d = kgc::f16_to_f(dbits);
    const uint32_t sw = rd_u16(sc + 2 * (ib / 2));
    const float dl = d * (float)(2 * ((sw >> (6 * (ib % 2) + 3 * (l >> 1))) & 7u) + 1);
    const uint32_t qhb = qh[2 * ib + (l >> 1)] >> (4 * (l & 1));
    const uint32_t idx = (uint32_t)qs[4 * ib + l] | ((qhb & 7u) << 8);
    const float delta = (qhb & 8u) ? -0.125f : 0.125f;
    const float gv = (float)(int8_t)((g.iq1s[idx] >> (8 * j)) & 0xff);
    return dl * (gv + delta);
}
__device__ float dq_iq4_nl(const uint8_t* b, int i, const IqGrids&) { return baloon::kern::dq_elem(baloon::kern::T_IQ4_NL, b, i); }
__device__ float dq_iq4_xs(const uint8_t* b, int i, const IqGrids&) { return baloon::kern::dq_elem(baloon::kern::T_IQ4_XS, b, i); }
__device__ float dq_q3_k(const uint8_t* b, int i, const IqGrids&) { return baloon::kern::dq_elem(baloon::kern::T_Q3_K, b, i); }
__device__ float dq_q2_0(const uint8_t* b, int i, const IqGrids&) { return baloon::kern::dq_elem(baloon::kern::T_Q2_0, b, i); }

__device__ float dq_dispatch(Fmt f, const uint8_t* b, int i, const IqGrids& g) {
    switch (f) {
        case Fmt::Q2_0: return dq_q2_0(b, i, g);
        case Fmt::Q3_K: return dq_q3_k(b, i, g);
        case Fmt::IQ2_XXS: return dq_iq2_xxs(b, i, g);
        case Fmt::IQ2_XS: return dq_iq2_xs(b, i, g);
        case Fmt::IQ2_S: return dq_iq2_s(b, i, g);
        case Fmt::IQ3_XXS: return dq_iq3_xxs(b, i, g);
        case Fmt::IQ3_S: return dq_iq3_s(b, i, g);
        case Fmt::IQ1_M: return dq_iq1_m(b, i, g);
        case Fmt::IQ4_NL: return dq_iq4_nl(b, i, g);
        case Fmt::IQ4_XS: return dq_iq4_xs(b, i, g);
    }
    return 0.f;
}

// dot of one weight block (n elements, n multiple of 32) with the n/32 Q8_1 blocks starting at q8
template <class F>
__device__ __forceinline__ float block_dot(const uint8_t* q8, int n, F f) {
    float tot = 0.f;
    for (int b = 0; b < n / 32; ++b) {
        const uint8_t* qb = q8 + (size_t)b * 36;
        const float d8 = rd_d(qb);
        float s = 0.f;
        for (int e = 0; e < 32; ++e) s += f(b * 32 + e) * (float)(int8_t)qb[4 + e];
        tot += d8 * s;
    }
    return tot;
}
__device__ float vec_dot_iq2_xxs_q8_1(const uint8_t* b, const uint8_t* q8, const IqGrids& g) { return block_dot(q8, 256, [&](int i) { return dq_iq2_xxs(b, i, g); }); }
__device__ float vec_dot_iq2_xs_q8_1(const uint8_t* b, const uint8_t* q8, const IqGrids& g) { return block_dot(q8, 256, [&](int i) { return dq_iq2_xs(b, i, g); }); }
__device__ float vec_dot_iq2_s_q8_1(const uint8_t* b, const uint8_t* q8, const IqGrids& g) { return block_dot(q8, 256, [&](int i) { return dq_iq2_s(b, i, g); }); }
__device__ float vec_dot_iq3_xxs_q8_1(const uint8_t* b, const uint8_t* q8, const IqGrids& g) { return block_dot(q8, 256, [&](int i) { return dq_iq3_xxs(b, i, g); }); }
__device__ float vec_dot_iq3_s_q8_1(const uint8_t* b, const uint8_t* q8, const IqGrids& g) { return block_dot(q8, 256, [&](int i) { return dq_iq3_s(b, i, g); }); }
__device__ float vec_dot_iq1_m_q8_1(const uint8_t* b, const uint8_t* q8, const IqGrids& g) { return block_dot(q8, 256, [&](int i) { return dq_iq1_m(b, i, g); }); }
__device__ float vec_dot_iq4_nl_q8_1(const uint8_t* b, const uint8_t* q8, const IqGrids& g) { return block_dot(q8, 32, [&](int i) { return dq_iq4_nl(b, i, g); }); }
__device__ float vec_dot_q2_0_q8_1(const uint8_t* b, const uint8_t* q8, const IqGrids& g) { return block_dot(q8, 64, [&](int i) { return dq_q2_0(b, i, g); }); }

// dot of one block of any supported format (IQ4_XS and Q3_K go through the generic decoder)
__device__ float row_dot(Fmt f, const uint8_t* b, const uint8_t* q8, const IqGrids& g) {
    switch (f) {
        case Fmt::Q2_0: return vec_dot_q2_0_q8_1(b, q8, g);
        case Fmt::IQ2_XXS: return vec_dot_iq2_xxs_q8_1(b, q8, g);
        case Fmt::IQ2_XS: return vec_dot_iq2_xs_q8_1(b, q8, g);
        case Fmt::IQ2_S: return vec_dot_iq2_s_q8_1(b, q8, g);
        case Fmt::IQ3_XXS: return vec_dot_iq3_xxs_q8_1(b, q8, g);
        case Fmt::IQ3_S: return vec_dot_iq3_s_q8_1(b, q8, g);
        case Fmt::IQ1_M: return vec_dot_iq1_m_q8_1(b, q8, g);
        case Fmt::IQ4_NL: return vec_dot_iq4_nl_q8_1(b, q8, g);
        case Fmt::IQ4_XS: return block_dot(q8, 256, [&](int i) { return dq_iq4_xs(b, i, g); });
        case Fmt::Q3_K: return block_dot(q8, 256, [&](int i) { return dq_q3_k(b, i, g); });
    }
    return 0.f;
}

// one thread per element of a flat tensor of n elements (n = whole blocks)
template <class T>
__global__ void dequant_flat_kernel(Fmt f, const uint8_t* __restrict__ src, T* __restrict__ dst, long long n, IqGrids g) {
    const long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    const Geo gm = geo(f);
    dst[e] = cvt<T>(dq_dispatch(f, src + (e / gm.elems) * gm.bytes, (int)(e % gm.elems), g));
}
// gate rows then up rows -> one f16 matrix [2*rows][cols]; one wave per row, GU_ROWS rows per block
__global__ void dequant_gu_kernel(Fmt f, const uint8_t* __restrict__ gate, const uint8_t* __restrict__ up,
                                                                  uint16_t* __restrict__ dst, int rows, int cols, IqGrids g) {
    const int r = blockIdx.x * GU_ROWS + (threadIdx.x >> 6);
    if (r >= 2 * rows) return;
    const Geo gm = geo(f);
    const int lane = threadIdx.x & 63;
    const int nb = cols / gm.elems;
    const uint8_t* rowp = (r < rows ? gate + (size_t)r * nb * gm.bytes : up + (size_t)(r - rows) * nb * gm.bytes);
    for (int e = lane; e < cols; e += 64) dst[(size_t)r * cols + e] = kgc::f_to_f16(dq_dispatch(f, rowp + (size_t)(e / gm.elems) * gm.bytes, e % gm.elems, g));
}
// out[i][c] for token ids; ids outside [0, n_rows) give zeros
__global__ void embed_rows_kernel(Fmt f, const uint8_t* __restrict__ table, long long n_rows, const int32_t* __restrict__ ids, int cols,
                                  float* __restrict__ out, IqGrids g) {
    const Geo gm = geo(f);
    const int32_t id = ids[blockIdx.x];
    const bool ok = id >= 0 && (long long)id < n_rows;
    const size_t rb = (size_t)(cols / gm.elems) * gm.bytes;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        out[(size_t)blockIdx.x * cols + c] = ok ? dq_dispatch(f, table + (size_t)id * rb + (size_t)(c / gm.elems) * gm.bytes, c % gm.elems, g) : 0.f;
}
// Q8_1: one wave per 32-value block; d = amax/127, q = round-half-away(x/d), s = d * sum(q); stored as f16 d, f16 s, 32 int8
__global__ void quantize_q8_1_kernel(const float* __restrict__ x, uint8_t* __restrict__ out, long long n_blocks) {
    const long long b = (long long)blockIdx.x * 4 + (threadIdx.x >> 6);
    if (b >= n_blocks) return;
    const int lane = threadIdx.x & 63;
    const float v = lane < 32 ? x[b * 32 + lane] : 0.f;
    const float amax = kgc::wave_max(fabsf(v));
    const float d = amax / 127.f;
    const float id = d > 0.f ? 1.f / d : 0.f;
    const int q = lane < 32 ? (int)roundf(v * id) : 0;
    const float sum = kgc::wave_sum((float)q);
    uint8_t* o = out + b * 36;
    if (lane == 0) {
        const uint16_t dh = kgc::f_to_f16(d), sh = kgc::f_to_f16(d * sum);
        o[0] = dh & 0xff; o[1] = dh >> 8; o[2] = sh & 0xff; o[3] = sh >> 8;
    }
    if (lane < 32) o[4 + lane] = (uint8_t)(int8_t)q;
}
// y[row] = sum over weight blocks of row_dot; one wave per row, lanes stride over blocks
__global__ void __launch_bounds__(256) iq_mmvq_kernel(Fmt f, const uint8_t* __restrict__ w, const uint8_t* __restrict__ q8, float* __restrict__ y, int rows, int cols, IqGrids g) {
    const int row = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (row >= rows) return;
    const Geo gm = geo(f);
    const int lane = threadIdx.x & 63, nb = cols / gm.elems;
    const uint8_t* rp = w + (size_t)row * nb * gm.bytes;
    float acc = 0.f;
    for (int b = lane; b < nb; b += 64) acc += row_dot(f, rp + (size_t)b * gm.bytes, q8 + (size_t)b * (gm.elems / 32) * 36, g);
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}
// h[i] = silu(gate[i]) * up[i]
__global__ void swiglu_entries_kernel(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ h, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = kgc::silu(gate[i]) * up[i];
}
size_t align256(size_t v) { return (v + 255) / 256 * 256; }
}  // namespace

bool is_iq(Fmt f) { return f == Fmt::IQ2_XXS || f == Fmt::IQ2_XS || f == Fmt::IQ2_S || f == Fmt::IQ3_XXS || f == Fmt::IQ3_S || f == Fmt::IQ1_M || f == Fmt::IQ4_NL || f == Fmt::IQ4_XS; }
bool iq_supported(Fmt f, const IqGrids& g) {
    switch (f) {
        case Fmt::IQ2_XXS: return g.iq2xxs != nullptr;
        case Fmt::IQ2_XS: return g.iq2xs != nullptr;
        case Fmt::IQ2_S: return g.iq2s != nullptr;
        case Fmt::IQ3_XXS: return g.iq3xxs != nullptr;
        case Fmt::IQ3_S: return g.iq3s != nullptr;
        case Fmt::IQ1_M: return g.iq1s != nullptr;
        case Fmt::IQ4_NL: case Fmt::IQ4_XS: case Fmt::Q3_K: case Fmt::Q2_0: return true;
    }
    return false;
}
int iq_block_elems(Fmt f) { return geo(f).elems; }
int iq_block_bytes(Fmt f) { return geo(f).bytes; }
long long iq_row_bytes(Fmt f, long long cols) {
    const Geo gm = geo(f);
    if (gm.elems == 0 || cols <= 0 || cols % gm.elems) return -1;
    return cols / gm.elems * gm.bytes;
}

cudaError_t iq_dequant_f32(Fmt f, const uint8_t* src, float* dst, long long n, const IqGrids& g, cudaStream_t s) {
    const Geo gm = geo(f);
    if (!src || !dst || n <= 0 || gm.elems == 0 || n % gm.elems || !iq_supported(f, g)) return cudaErrorInvalidValue;
    dequant_flat_kernel<float><<<kgc::grid_for_n(n, 256), 256, 0, s>>>(f, src, dst, n, g);
    return cudaGetLastError();
}
cudaError_t iq_dequant_f16(Fmt f, const uint8_t* src, uint16_t* dst, long long n, const IqGrids& g, cudaStream_t s) {
    const Geo gm = geo(f);
    if (!src || !dst || n <= 0 || gm.elems == 0 || n % gm.elems || !iq_supported(f, g)) return cudaErrorInvalidValue;
    dequant_flat_kernel<uint16_t><<<kgc::grid_for_n(n, 256), 256, 0, s>>>(f, src, dst, n, g);
    return cudaGetLastError();
}
cudaError_t iq_dequant_gu_f16(Fmt f, const uint8_t* gate, const uint8_t* up, uint16_t* dst, int rows, int cols, const IqGrids& g, cudaStream_t s) {
    if (!gate || !up || !dst || rows <= 0 || iq_row_bytes(f, cols) < 0 || !iq_supported(f, g)) return cudaErrorInvalidValue;
    dequant_gu_kernel<<<(2 * rows + GU_ROWS - 1) / GU_ROWS, 64 * GU_ROWS, 0, s>>>(f, gate, up, dst, rows, cols, g);
    return cudaGetLastError();
}
cudaError_t iq_embed_rows(Fmt f, const uint8_t* table, long long n_rows, const int32_t* ids, int n_ids, int cols, float* out, const IqGrids& g, cudaStream_t s) {
    if (!table || !ids || !out || n_rows <= 0 || n_ids <= 0 || iq_row_bytes(f, cols) < 0 || !iq_supported(f, g)) return cudaErrorInvalidValue;
    embed_rows_kernel<<<n_ids, 256, 0, s>>>(f, table, n_rows, ids, cols, out, g);
    return cudaGetLastError();
}
cudaError_t quantize_q8_1_rows(const float* x, uint8_t* out, int rows, int cols, cudaStream_t s) {
    if (!x || !out || rows <= 0 || cols <= 0 || cols % 32) return cudaErrorInvalidValue;
    const long long nb = (long long)rows * (cols / 32);
    quantize_q8_1_kernel<<<(unsigned)((nb + 3) / 4), 256, 0, s>>>(x, out, nb);
    return cudaGetLastError();
}
cudaError_t iq_mmvq(Fmt f, const uint8_t* w, const uint8_t* q8, float* y, int rows, int cols, const IqGrids& g, cudaStream_t s) {
    if (!w || !q8 || !y || rows <= 0 || iq_row_bytes(f, cols) < 0 || !iq_supported(f, g)) return cudaErrorInvalidValue;
    iq_mmvq_kernel<<<(rows + 3) / 4, 256, 0, s>>>(f, w, q8, y, rows, cols, g);
    return cudaGetLastError();
}

// expert blob: gate [n_ff rows x n_embd cols] | up [n_ff x n_embd] | down [n_embd rows x n_ff cols], each rows * row_bytes
bool native_expert_layout(Fmt f, int n_embd, int n_ff, ExpertLayout* out) {
    const long long rb_in = iq_row_bytes(f, n_embd), rb_out = iq_row_bytes(f, n_ff);
    if (!out || rb_in < 0 || rb_out < 0 || n_ff <= 0) return false;
    out->gate_off = 0;
    out->up_off = (size_t)rb_in * n_ff;
    out->down_off = 2 * (size_t)rb_in * n_ff;
    out->total = out->down_off + (size_t)rb_out * n_embd;
    return true;
}
size_t native_expert_scratch_bytes(int n_embd, int n_ff) {
    // q8_1 image of x, three float planes (gate, up, h), q8_1 image of h: exactly the regions native_expert_grouped carves, nothing after hq.
    return align256((size_t)(n_embd / 32) * 36) + 3 * align256((size_t)n_ff * 4) + align256((size_t)(n_ff / 32) * 36);
}
cudaError_t native_expert_grouped(Fmt f, const uint8_t* blob, const float* x, int n_embd, int n_ff, float* out, void* scratch, size_t scratch_bytes,
                                  const IqGrids& g, cudaStream_t s) {
    ExpertLayout L;
    if (!blob || !x || !out || !scratch || !native_expert_layout(f, n_embd, n_ff, &L) || !iq_supported(f, g)) return cudaErrorInvalidValue;
    if (n_embd % 32 || n_ff % 32 || scratch_bytes < native_expert_scratch_bytes(n_embd, n_ff)) return cudaErrorInvalidValue;
    uint8_t* p = (uint8_t*)scratch;
    uint8_t* xq = p;            p += align256((size_t)(n_embd / 32) * 36);
    float* gt = (float*)p;      p += align256((size_t)n_ff * 4);
    float* up = (float*)p;      p += align256((size_t)n_ff * 4);
    float* h = (float*)p;       p += align256((size_t)n_ff * 4);
    uint8_t* hq = p;
    cudaError_t e;
    if ((e = quantize_q8_1_rows(x, xq, 1, n_embd, s)) != cudaSuccess) return e;
    if ((e = iq_mmvq(f, blob + L.gate_off, xq, gt, n_ff, n_embd, g, s)) != cudaSuccess) return e;
    if ((e = iq_mmvq(f, blob + L.up_off, xq, up, n_ff, n_embd, g, s)) != cudaSuccess) return e;
    swiglu_entries_kernel<<<(n_ff + 255) / 256, 256, 0, s>>>(gt, up, h, n_ff);
    if ((e = quantize_q8_1_rows(h, hq, 1, n_ff, s)) != cudaSuccess) return e;
    return iq_mmvq(f, blob + L.down_off, hq, out, n_embd, n_ff, g, s);
}
}  // namespace kg
}  // namespace baloon

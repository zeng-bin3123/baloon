// native_mmvq.cu - wave64 MMVQ kernels: quantized weight blocks x Q8_1 activations, 1..8 columns.
//
// Geometry (128-thread CUDA block; T threads per weight block, BPI = 128 / T blocks per iteration):
//   Q5_K (16, 8)  Q4_K (16, 8)  Q3_K (16, 8)  Q6_K (32, 4)  IQ4_XS (8, 16)  Q2_0 / Q4_0 / Q5_0 / IQ4_NL (2, 64)  Q8_0 (4, 32)
// Thread tid of a row group handles slot (tid % T) of block (tid / T), stepping BPI blocks per iteration; the slot owns the
// contiguous element range [slot * E / T, (slot + 1) * E / T) of the block. Small K (fewer weight blocks than BPI) runs
// 4 rows per CUDA block with 32 threads per row. The per-row sum is a shared-memory tree with halving strides, so a
// column's result does not depend on how many columns ride along (default). With multi_exact off, 5..8 columns use the
// upstream layout: 2 warps (64 threads), 2 rows per block, BPI halved.
#include <cuda_runtime.h>

#include <cstdint>

#include "baloon/kern_geometry.hpp"
#include "baloon/kern_native_mmvq.hpp"
#include "ix_compat.h"
#include "kern_cu_blocks.cuh"

namespace baloon {
namespace kern {

namespace {

constexpr int MAX_NCOLS = 8;
constexpr int WARP = 64;           // hardware wavefront
constexpr int WARPS = 4;           // 32-thread warps per 128-thread block (upstream geometry)
constexpr int QUANT_THREADS = 256;
constexpr int Q8K = 32;            // values per q8_1 block
constexpr int QK = 32;             // values per q8_1 block (alias used by the traits)
constexpr int QI = 8;              // 32-bit words of int8 per q8_1 block
constexpr int VDR = 2;             // words a thread consumes per dot call in the small formats
bool g_multi_exact = true;

// ---- block layouts -------------------------------------------------------------------------------------------
struct Q20Block { uint16_t d; uint8_t qs[16]; };           // 64 values, 2-bit codes
struct Q40Block { uint16_t d; uint8_t qs[16]; };
struct Q50Block { uint16_t d; uint8_t qh[4]; uint8_t qs[16]; };
struct Q80Block { uint16_t d; int8_t qs[32]; };
struct IQ4NLBlock { uint16_t d; uint8_t qs[16]; };
struct Q3KBlock { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; uint16_t d; };
struct Q4KBlock { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; };
struct Q5KBlock { uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
struct Q6KBlock { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };
struct IQ4XSBlock { uint16_t d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; };
struct Q81Block { uint16_t d; uint16_t s; int8_t qs[32]; } __align__(4);   // d = scale, s = d * sum(q)
static_assert(sizeof(Q20Block) == 18 && sizeof(Q40Block) == 18 && sizeof(Q50Block) == 22 && sizeof(Q80Block) == 34 &&
              sizeof(IQ4NLBlock) == 18 && sizeof(Q3KBlock) == 110 && sizeof(Q4KBlock) == 144 && sizeof(Q5KBlock) == 176 &&
              sizeof(Q6KBlock) == 210 && sizeof(IQ4XSBlock) == 136 && sizeof(Q81Block) == 36, "ggml block sizes");

// ---- small device helpers -------------------------------------------------------------------------------------
__device__ __forceinline__ float warp_sum(float v) { return ix_warp_reduce_sum(v); }
__device__ __forceinline__ float warp_max(float v) { return ix_warp_reduce_max(v); }

// The int at int-index i of a 2-byte-aligned byte array (little endian).
__device__ __forceinline__ int load_int_b2(const int8_t* q, int i) {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(q);
    return int(uint32_t(p[2 * i]) | (uint32_t(p[2 * i + 1]) << 16));
}

__device__ __forceinline__ int iq4_table_lookup(int nib) { return kv_iq4nl_dev[nib & 15]; }
__device__ __forceinline__ float act_scale(const Q81Block& a) { return f16bits::f32_from_f16(a.d); }

// Dot of the element range [lo, lo + n) of a weight block of `type` with the activation values at the same positions.
// Activation blocks are 32 values; a range may span several of them. Float accumulation per activation block.
__device__ __forceinline__ float range_dot(int type, const uint8_t* blk, const Q81Block* a, int lo, int n) {
    float s = 0.f;
    int i = lo;
    const int end = lo + n;
    while (i < end) {
        const int bi = i / 32;
        const int stop = end < (bi + 1) * 32 ? end : (bi + 1) * 32;
        float acc = 0.f;
        for (; i < stop; ++i) acc += dq_elem(type, blk, i) * (float)a[bi].qs[i & 31];
        s += acc * act_scale(a[bi]);
    }
    return s;
}

// ---- per-format slot dots ---------------------------------------------------------------------------------------
__device__ __forceinline__ float q2_q8_dot(const Q20Block* b, const Q81Block* a, int slot) { return range_dot(T_Q2_0, (const uint8_t*)b, a, slot * 32, 32); }
__device__ __forceinline__ float q3_q8_dot_impl(const Q3KBlock* b, const Q81Block* a, int lo, int n) { return range_dot(T_Q3_K, (const uint8_t*)b, a, lo, n); }
__device__ __forceinline__ float q3_q8_dot(const Q3KBlock* b, const Q81Block* a, int slot) { return q3_q8_dot_impl(b, a, slot * 16, 16); }
__device__ __forceinline__ float q4_q8_dot_impl(const Q4KBlock* b, const Q81Block* a, int lo, int n) { return range_dot(T_Q4_K, (const uint8_t*)b, a, lo, n); }
__device__ __forceinline__ float q4_q8_dot(const Q4KBlock* b, const Q81Block* a, int slot) { return q4_q8_dot_impl(b, a, slot * 16, 16); }
__device__ __forceinline__ float q5_q8_dot_impl(const Q5KBlock* b, const Q81Block* a, int lo, int n) { return range_dot(T_Q5_K, (const uint8_t*)b, a, lo, n); }
__device__ __forceinline__ float q5_q8_dot(const Q5KBlock* b, const Q81Block* a, int slot) { return q5_q8_dot_impl(b, a, slot * 16, 16); }
__device__ __forceinline__ float q6_q8_dot_impl(const Q6KBlock* b, const Q81Block* a, int lo, int n) { return range_dot(T_Q6_K, (const uint8_t*)b, a, lo, n); }
__device__ __forceinline__ float q6_q8_dot(const Q6KBlock* b, const Q81Block* a, int slot) { return q6_q8_dot_impl(b, a, slot * 8, 8); }
__device__ __forceinline__ float iq4_xs_q8_dot(const IQ4XSBlock* b, const Q81Block* a, int slot) { return range_dot(T_IQ4_XS, (const uint8_t*)b, a, slot * 32, 32); }

// Q4_0 / Q5_0 / IQ4_NL / Q8_0 (32-value blocks; T = 2, or 4 for Q8_0). Q8_0 uses the integer dp4a form: the slot owns
// eight int8 weights = two 32-bit words against two activation words.
template <int TYPE, int T>
__device__ __forceinline__ float small_q8_dot(const uint8_t* blk, const Q81Block* a, int slot) {
    constexpr int PER = 32 / T;
    if (TYPE == T_Q8_0) {
        const Q80Block* b = reinterpret_cast<const Q80Block*>(blk);
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < PER / 4; ++i)
            sumi = ix_dp4a(load_int_b2(b->qs, slot * (PER / 4) + i), reinterpret_cast<const int*>(a->qs)[slot * (PER / 4) + i], sumi);
        return f16bits::f32_from_f16(b->d) * act_scale(*a) * (float)sumi;
    }
    return range_dot(TYPE, blk, a, slot * PER, PER);
}

// ---- traits: T = threads per weight block, E = values per block ---------------------------------------------------
struct Q20Traits { using Block = Q20Block; static constexpr int T = 2, E = 64; static constexpr int TYPE = T_Q2_0;
    __device__ static float dot(const Block* b, const Q81Block* a, int s) { return q2_q8_dot(b, a, s); } };
struct Q3KTraits { using Block = Q3KBlock; static constexpr int T = 16, E = 256; static constexpr int TYPE = T_Q3_K;
    __device__ static float dot(const Block* b, const Q81Block* a, int s) { return q3_q8_dot(b, a, s); } };
struct Q4KTraits { using Block = Q4KBlock; static constexpr int T = 16, E = 256; static constexpr int TYPE = T_Q4_K;
    __device__ static float dot(const Block* b, const Q81Block* a, int s) { return q4_q8_dot(b, a, s); } };
struct Q5KTraits { using Block = Q5KBlock; static constexpr int T = 16, E = 256; static constexpr int TYPE = T_Q5_K;
    __device__ static float dot(const Block* b, const Q81Block* a, int s) { return q5_q8_dot(b, a, s); } };
struct Q6KTraits { using Block = Q6KBlock; static constexpr int T = 32, E = 256; static constexpr int TYPE = T_Q6_K;
    __device__ static float dot(const Block* b, const Q81Block* a, int s) { return q6_q8_dot(b, a, s); } };
struct IQ4XSTraits { using Block = IQ4XSBlock; static constexpr int T = 8, E = 256; static constexpr int TYPE = T_IQ4_XS;
    __device__ static float dot(const Block* b, const Q81Block* a, int s) { return iq4_xs_q8_dot(b, a, s); } };
template <class B, int TY, int T_>
struct SmallTraits { using Block = B; static constexpr int T = T_, E = 32; static constexpr int TYPE = TY;
    __device__ static float dot(const Block* b, const Q81Block* a, int s) { return small_q8_dot<TY, T_>((const uint8_t*)b, a, s); } };
using Q40Traits = SmallTraits<Q40Block, T_Q4_0, 2>;
using Q50Traits = SmallTraits<Q50Block, T_Q5_0, 2>;
using IQ4NLTraits = SmallTraits<IQ4NLBlock, T_IQ4_NL, 2>;
using Q80Traits = SmallTraits<Q80Block, T_Q8_0, 4>;

// ---- the kernel ---------------------------------------------------------------------------------------------------
// NW 32-thread warps per CUDA block, ROWS rows per CUDA block (TPR = threads per row, a power of two).
template <class TR, int NW, int ROWS>
__global__ __launch_bounds__(32 * NW) void mmvq_kernel(const typename TR::Block* __restrict__ w, const Q81Block* __restrict__ x,
                                                       float* __restrict__ y, int nblk, int n_in32, int n_out, int ncols) {
    constexpr int NT = 32 * NW, TPR = NT / ROWS, BPI = TPR / TR::T;
    static_assert(TPR % TR::T == 0 && (TPR & (TPR - 1)) == 0, "threads per row");
    const int tid = threadIdx.x, rl = tid / TPR, lt = tid % TPR;
    const int row = blockIdx.x * ROWS + rl;
    float acc[MAX_NCOLS];
#pragma unroll
    for (int c = 0; c < MAX_NCOLS; ++c) acc[c] = 0.f;
    if (row < n_out) {
        const int slot = lt % TR::T;
        for (int kb = lt / TR::T; kb < nblk; kb += BPI) {
            const typename TR::Block* blk = w + (size_t)row * nblk + kb;
#pragma unroll
            for (int c = 0; c < MAX_NCOLS; ++c)
                if (c < ncols) acc[c] += TR::dot(blk, x + (size_t)c * n_in32 + (size_t)kb * (TR::E / 32), slot);
        }
    }
    __shared__ float sm[NT * MAX_NCOLS];
#pragma unroll
    for (int c = 0; c < MAX_NCOLS; ++c) sm[tid * MAX_NCOLS + c] = acc[c];
    __syncthreads();
    for (int s = TPR / 2; s > 0; s >>= 1) {   // halving strides: tpr/2 ... 1
        if (lt < s) {
#pragma unroll
            for (int c = 0; c < MAX_NCOLS; ++c) sm[tid * MAX_NCOLS + c] += sm[(tid + s) * MAX_NCOLS + c];
        }
        __syncthreads();
    }
    if (lt == 0 && row < n_out)
        for (int c = 0; c < ncols; ++c) y[(size_t)c * n_out + row] = sm[tid * MAX_NCOLS + c];
}

// ---- validation -----------------------------------------------------------------------------------------------------
cudaError_t validate_pointer(const void* p) { return p == nullptr ? cudaErrorInvalidValue : cudaSuccess; }
cudaError_t validate_shape(int n_in, int n_out, int ncols, int elems) {
    if (n_in <= 0 || n_out <= 0 || ncols < 1 || ncols > MAX_NCOLS) return cudaErrorInvalidValue;
    if (n_in % elems != 0 || n_in % 32 != 0) return cudaErrorInvalidValue;   // whole weight blocks and whole q8_1 blocks
    return cudaSuccess;
}
cudaError_t validate_stream(cudaStream_t s) {
    const cudaError_t e = cudaStreamQuery(s);
    return (e == cudaSuccess || e == cudaErrorNotReady) ? cudaSuccess : e;
}
cudaError_t launch_check() { return cudaGetLastError(); }

// ---- launchers ------------------------------------------------------------------------------------------------------
template <class TR, int NW, int ROWS>
void run(const void* w, const void* x, float* y, int nblk, int n_in32, int n_out, int ncols, cudaStream_t s) {
    mmvq_kernel<TR, NW, ROWS><<<(n_out + ROWS - 1) / ROWS, 32 * NW, 0, s>>>(static_cast<const typename TR::Block*>(w),
                                                                           static_cast<const Q81Block*>(x), y, nblk, n_in32, n_out, ncols);
}

// Launch for a given column count (validated by the caller).
template <class TR>
cudaError_t launch_multi_n(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    const int nblk = n_in / TR::E, n_in32 = n_in / 32;
    const int bpi = 128 / TR::T;
    const bool small = nblk < bpi;   // small-K rule
    if (!g_multi_exact && ncols >= 5) run<TR, 2, 2>(w, x, y, nblk, n_in32, n_out, ncols, s);
    else if (small) run<TR, 4, 4>(w, x, y, nblk, n_in32, n_out, ncols, s);
    else run<TR, 4, 1>(w, x, y, nblk, n_in32, n_out, ncols, s);
    return launch_check();
}

template <class TR>
cudaError_t launch_multi(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    cudaError_t e;
    if ((e = validate_pointer(w)) != cudaSuccess || (e = validate_pointer(x)) != cudaSuccess || (e = validate_pointer(y)) != cudaSuccess) return e;
    if ((e = validate_shape(n_in, n_out, ncols, TR::E)) != cudaSuccess) return e;
    if ((e = validate_stream(s)) != cudaSuccess) return e;
    return launch_multi_n<TR>(w, x, y, n_in, n_out, ncols, s);
}

template <class TR>
cudaError_t small_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    return launch_multi<TR>(w, x, y, n_in, n_out, ncols, s);
}

// float activations: one Q8_1 quantization of all columns as a single vector, then the same launch.
template <class TR>
cudaError_t small_f32(const void* w, const float* x, float* y, int n_in, int n_out, int ncols, void* scratch, cudaStream_t s) {
    cudaError_t e;
    if ((e = validate_pointer(x)) != cudaSuccess || (e = validate_pointer(scratch)) != cudaSuccess) return e;
    if ((e = validate_shape(n_in, n_out, ncols, TR::E)) != cudaSuccess) return e;
    if ((e = native_quantize_q8_1(x, scratch, (int64_t)n_in * ncols, s)) != cudaSuccess) return e;
    return launch_multi<TR>(w, scratch, y, n_in, n_out, ncols, s);
}

// ---- Q8_1 quantization: one thread per element, 32-lane groups --------------------------------------------------------
__global__ void quantize_q8_1_kernel(const float* __restrict__ x, Q81Block* __restrict__ out, int64_t n) {
    const int64_t idx = (int64_t)blockIdx.x * QUANT_THREADS + threadIdx.x;
    const float v = idx < n ? x[idx] : 0.f;
    const float amax = ix_halfwave_reduce_max(fabsf(v));
    const float d = amax / 127.f;
    const float id = d != 0.f ? 1.f / d : 0.f;
    const int q = (int)roundf(v * id);
    const int qc = q < -127 ? -127 : (q > 127 ? 127 : q);
    const float sum = ix_halfwave_reduce_sum((float)qc);
    if (idx < n) {
        Q81Block* b = out + idx / 32;
        b->qs[idx % 32] = (int8_t)qc;
        if ((idx % 32) == 0) { b->d = f16bits::f16_from_f32(d); b->s = f16bits::f16_from_f32(d * sum); }
    }
}

}  // namespace

// ============================================================================================ public API
bool native_mmvq_supported(int type) {
    switch (type) {
        case T_Q4_0: case T_Q5_0: case T_Q8_0: case T_IQ4_NL: case T_Q3_K: case T_Q4_K: case T_Q5_K: case T_Q6_K: case T_IQ4_XS: case T_Q2_0: return true;
        default: return false;
    }
}

size_t native_mmvq_weight_bytes(int type, int n_in, int n_out) {
    const mmvqgeo::WeightEntry* e = mmvqgeo::find_weight(type);
    if (e == nullptr || n_in <= 0 || n_out <= 0 || n_in % (int)e->elems != 0) return 0;
    return (size_t)n_out * (size_t)(n_in / (int)e->elems) * e->bytes;
}

size_t native_q8_1_bytes(int ncols, int n_in) { return (size_t)ncols * (size_t)(n_in / 32) * 36; }

cudaError_t native_quantize_q8_1(const float* x, void* out, int64_t n, cudaStream_t s) {
    if (x == nullptr || out == nullptr || n <= 0 || n % 32 != 0) return cudaErrorInvalidValue;
    quantize_q8_1_kernel<<<(unsigned)((n + QUANT_THREADS - 1) / QUANT_THREADS), QUANT_THREADS, 0, s>>>(x, static_cast<Q81Block*>(out), n);
    return launch_check();
}

void native_mmvq_set_multi_exact(bool on) { g_multi_exact = on; }
bool native_mmvq_multi_exact() { return g_multi_exact; }

cudaError_t native_q2_0_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    return launch_multi<Q20Traits>(w, x, y, n_in, n_out, ncols, s);
}
cudaError_t native_q2_0_f32(const void* w, const float* x, float* y, int n_in, int n_out, int ncols, void* scratch, cudaStream_t s) {
    return small_f32<Q20Traits>(w, x, y, n_in, n_out, ncols, scratch, s);
}
cudaError_t native_q4_0_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    return launch_multi<Q40Traits>(w, x, y, n_in, n_out, ncols, s);
}
cudaError_t native_q4_0_f32(const void* w, const float* x, float* y, int n_in, int n_out, int ncols, void* scratch, cudaStream_t s) {
    return small_f32<Q40Traits>(w, x, y, n_in, n_out, ncols, scratch, s);
}
cudaError_t native_q5_0_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    return launch_multi<Q50Traits>(w, x, y, n_in, n_out, ncols, s);
}
cudaError_t native_q5_0_f32(const void* w, const float* x, float* y, int n_in, int n_out, int ncols, void* scratch, cudaStream_t s) {
    return small_f32<Q50Traits>(w, x, y, n_in, n_out, ncols, scratch, s);
}
cudaError_t native_q8_0_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    return launch_multi<Q80Traits>(w, x, y, n_in, n_out, ncols, s);
}
cudaError_t native_q8_0_f32(const void* w, const float* x, float* y, int n_in, int n_out, int ncols, void* scratch, cudaStream_t s) {
    return small_f32<Q80Traits>(w, x, y, n_in, n_out, ncols, scratch, s);
}
cudaError_t native_iq4_nl_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    return launch_multi<IQ4NLTraits>(w, x, y, n_in, n_out, ncols, s);
}
cudaError_t native_iq4_nl_f32(const void* w, const float* x, float* y, int n_in, int n_out, int ncols, void* scratch, cudaStream_t s) {
    return small_f32<IQ4NLTraits>(w, x, y, n_in, n_out, ncols, scratch, s);
}
cudaError_t native_q3_k_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    return launch_multi<Q3KTraits>(w, x, y, n_in, n_out, ncols, s);
}
cudaError_t native_q3_k_f32(const void* w, const float* x, float* y, int n_in, int n_out, int ncols, void* scratch, cudaStream_t s) {
    return small_f32<Q3KTraits>(w, x, y, n_in, n_out, ncols, scratch, s);
}
cudaError_t native_q4_k_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    return launch_multi<Q4KTraits>(w, x, y, n_in, n_out, ncols, s);
}
cudaError_t native_q4_k_f32(const void* w, const float* x, float* y, int n_in, int n_out, int ncols, void* scratch, cudaStream_t s) {
    return small_f32<Q4KTraits>(w, x, y, n_in, n_out, ncols, scratch, s);
}
cudaError_t native_q5_k_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    return launch_multi<Q5KTraits>(w, x, y, n_in, n_out, ncols, s);
}
cudaError_t native_q5_k_f32(const void* w, const float* x, float* y, int n_in, int n_out, int ncols, void* scratch, cudaStream_t s) {
    return small_f32<Q5KTraits>(w, x, y, n_in, n_out, ncols, scratch, s);
}
cudaError_t native_q6_k_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    return launch_multi<Q6KTraits>(w, x, y, n_in, n_out, ncols, s);
}
cudaError_t native_q6_k_f32(const void* w, const float* x, float* y, int n_in, int n_out, int ncols, void* scratch, cudaStream_t s) {
    return small_f32<Q6KTraits>(w, x, y, n_in, n_out, ncols, scratch, s);
}
cudaError_t native_iq4_xs_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    return launch_multi<IQ4XSTraits>(w, x, y, n_in, n_out, ncols, s);
}
cudaError_t native_iq4_xs_f32(const void* w, const float* x, float* y, int n_in, int n_out, int ncols, void* scratch, cudaStream_t s) {
    return small_f32<IQ4XSTraits>(w, x, y, n_in, n_out, ncols, scratch, s);
}

cudaError_t native_mmvq(int type, const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    switch (type) {
        case T_Q2_0: return native_q2_0_mmvq(w, x, y, n_in, n_out, ncols, s);
        case T_Q4_0: return native_q4_0_mmvq(w, x, y, n_in, n_out, ncols, s);
        case T_Q5_0: return native_q5_0_mmvq(w, x, y, n_in, n_out, ncols, s);
        case T_Q8_0: return native_q8_0_mmvq(w, x, y, n_in, n_out, ncols, s);
        case T_IQ4_NL: return native_iq4_nl_mmvq(w, x, y, n_in, n_out, ncols, s);
        case T_Q3_K: return native_q3_k_mmvq(w, x, y, n_in, n_out, ncols, s);
        case T_Q4_K: return native_q4_k_mmvq(w, x, y, n_in, n_out, ncols, s);
        case T_Q5_K: return native_q5_k_mmvq(w, x, y, n_in, n_out, ncols, s);
        case T_Q6_K: return native_q6_k_mmvq(w, x, y, n_in, n_out, ncols, s);
        case T_IQ4_XS: return native_iq4_xs_mmvq(w, x, y, n_in, n_out, ncols, s);
        default: return cudaErrorInvalidValue;
    }
}

}  // namespace kern
}  // namespace baloon

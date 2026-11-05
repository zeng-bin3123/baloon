// s2_expert_grouped.cu - GPU evaluation of resident Q2_0 ("S2") routed experts.
//
// Blob (1,382,400 bytes): gate/up codes (2*FF rows of 640 B, rows interleaved: even slot = gate, odd = up) | down codes
// (H rows of 160 B) | gate/up scales (2*FF rows of 40 fp16) | down scales (H rows of 10 fp16). Codes c in 0..3 decode to
// (c - 1) * scale; the offset is removed through the activation sums: sum (c-1) x = sum c x - sum x.
// Activations are quantized per 32-element chunk (int8 in [-127, 127], fp32 scale, chunk sum hx) exactly as the CPU kernel.
// Everything runs as a list of (token, resident slot, router slot) "pairs"; expert outputs land in a per-router-slot buffer
// and are combined in ascending router-slot order, so the result does not depend on how the pairs were formed.
//
// Two arithmetics: the fast path (lane-parallel dp4a, tree reduction) and the CPU-order path (one thread per output row,
// serial float accumulation with explicit round-to-nearest mul/add, the same sequence as the CPU kernel).
#include <cuda_runtime.h>

#include <cstdint>

#include "baloon/kern_cuda_misc.hpp"
#include "ix_compat.h"
#include "baloon/f16_bits.hpp"

namespace baloon {
namespace kern {

constexpr int H = q2geo::H;                  // 2560
constexpr int FF = q2geo::FF;                // 640
constexpr int QK = q2geo::QK;                // 64
constexpr int ROW_GU = q2geo::ROW_GU;        // 640 code bytes per gate/up row
constexpr int ROW_D = q2geo::ROW_D;          // 160
constexpr int SC_GU = q2geo::SC_GU;          // 40 scales per gate/up row
constexpr int SC_D = q2geo::SC_D;            // 10
constexpr uint64_t O_D_CODES = q2geo::O_D_CODES;
constexpr uint64_t O_GU_SCALES = q2geo::O_GU_SCALES;
constexpr uint64_t O_D_SCALES = q2geo::O_D_SCALES;
constexpr int GU_ROWS = 2 * FF;              // 1280 interleaved rows per expert
constexpr int D_ROWS = H;                    // 2560
constexpr int GU_CHUNKS = H / 32;            // 80
constexpr int GMAX = 8;                      // widest token batch of a grouped call (matches the CPU multi-token width)
constexpr int THREADS = 128;                 // 4 lane groups of 32 per CUDA block

namespace {

constexpr int NE_ALL = q2geo::NE;            // 512 experts in the residency table
constexpr int WARPS_PER_BLOCK = THREADS / 32;

inline cudaError_t check(cudaError_t e) { return e; }

__device__ __forceinline__ float f16_at(const uint8_t* p, int idx) {
    return f16bits::f32_from_f16((uint16_t)(p[2 * idx] | (p[2 * idx + 1] << 8)));
}
// Four 2-bit codes of a byte -> one byte per code (dp4a operand).
__device__ __forceinline__ uint32_t expand4(uint32_t c) { return (c & 3u) | (((c >> 2) & 3u) << 8) | (((c >> 4) & 3u) << 16) | (((c >> 6) & 3u) << 24); }
__device__ __forceinline__ int dot4(uint32_t codes_expanded, int q4, int acc) { return ix_dp4a((int)codes_expanded, q4, acc); }
__device__ __forceinline__ float warp_sum(float v) { return ix_halfwave_reduce_sum(v); }   // 32-lane group

// sum c*q over the 32 elements of chunk c: 8 code bytes against 8 activation words.
__device__ __forceinline__ int chunk_dot(const uint8_t* codes_row, int c, const int8_t* qchunk) {
    const uint8_t* cb = codes_row + c * 8;
    int s = 0;
#pragma unroll
    for (int m = 0; m < 8; ++m) s = dot4(expand4(cb[m]), reinterpret_cast<const int*>(qchunk)[m], s);
    return s;
}

// Lane-parallel row dot over nchunks activation chunks (lane takes chunks lane, lane+32, ...), tree-reduced over the 32
// lanes of the group. All lanes receive the total.
__device__ __forceinline__ float row_dot_s2_q8(const uint8_t* codes_row, const uint8_t* scales_row, int nchunks, const int8_t* q,
                                               const float* xd, const int32_t* hx) {
    const int lane = threadIdx.x & 31;
    float acc = 0.f;
    for (int c = lane; c < nchunks; c += 32) {
        const int s = chunk_dot(codes_row, c, q + c * 32);
        acc += f16_at(scales_row, c / 2) * xd[c] * (float)(s - hx[c]);   // weight scale index c / 2
    }
    return warp_sum(acc);
}

// CPU kernel order: per weight block, low then high chunk, plain round-to-nearest mul / add.
__device__ __forceinline__ float row_dot_cpu_order(const uint8_t* codes_row, const uint8_t* scales_row, int nb, const int8_t* q, const float* xd,
                                                   const int32_t* hx) {
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        const float wd = f16_at(scales_row, b);
        for (int h = 0; h < 2; ++h) {
            const int ch = 2 * b + h;
            int s = 0;
            for (int i = 0; i < 32; ++i) {
                const int j = h * 32 + i;
                s += (int)((codes_row[b * 16 + j / 4] >> (2 * (j % 4))) & 3) * (int)q[ch * 32 + i];
            }
            acc = __fadd_rn(acc, __fmul_rn(__fmul_rn(wd, xd[ch]), (float)(s - hx[ch])));
        }
    }
    return acc;
}

struct Pairs {
    const int32_t* slot;      // resident slot, -1 = none
    const int32_t* dst;       // router slot k
    const int32_t* tok;       // token of the pair (null: p / tok_div)
    const int32_t* nvalid;    // device count (null: P)
    int tok_div;
    int P;
};
__device__ __forceinline__ bool pair_ok(const Pairs& a, int p, int& slot, int& tok, int& dst) {
    const int lim = a.nvalid ? *a.nvalid : a.P;
    if (p >= lim || p >= a.P) return false;
    slot = a.slot[p];
    if (slot < 0) return false;
    tok = a.tok ? a.tok[p] : (a.tok_div > 0 ? p / a.tok_div : 0);
    dst = a.dst ? a.dst[p] : 0;
    return true;
}

struct S2Ptrs {
    float* gu; int8_t* q8; float* hd; int32_t* hx; int8_t* xq; float* xd; int32_t* xh; float* pout; int32_t* valid;
};

// ---- hit selection -----------------------------------------------------------------------------------------------------
// Resident experts of one token's router row compacted with popc(ballot & ((1 << lane) - 1)).
__global__ void hit_select_kernel(const int32_t* ids, const int32_t* slot_of_expert, int K, int32_t* hit_slot, int32_t* hit_dst, int32_t* n_hit) {
    const int lane = threadIdx.x;   // one 64-lane wave
    int slot = -1;
    if (lane < K) {
        const int e = ids[lane];
        slot = (e >= 0 && e < NE_ALL) ? slot_of_expert[e] : -1;
    }
    const unsigned long long m = ix_ballot(slot >= 0);
    const int pos = __popcll(m & ((lane >= 64) ? ~0ull : ((1ull << lane) - 1ull)));
    if (lane < K) { hit_slot[lane] = -1; hit_dst[lane] = -1; }
    __syncthreads();
    if (slot >= 0) { hit_slot[pos] = slot; hit_dst[pos] = lane; }
    if (lane == 0) *n_hit = __popcll(m);
}
__global__ void hit_select_multi_kernel(const int32_t* ids, const int32_t* slot_of_expert, int K, int32_t* hit_slot, int32_t* hit_dst, int32_t* n_hit) {
    const int t = blockIdx.x, lane = threadIdx.x;
    int slot = -1;
    if (lane < K) {
        const int e = ids[t * K + lane];
        slot = (e >= 0 && e < NE_ALL) ? slot_of_expert[e] : -1;
    }
    const unsigned long long m = ix_ballot(slot >= 0);
    const int pos = __popcll(m & ((lane >= 64) ? ~0ull : ((1ull << lane) - 1ull)));
    if (lane < K) { hit_slot[t * K + lane] = -1; hit_dst[t * K + lane] = -1; }
    __syncthreads();
    if (slot >= 0) { hit_slot[t * K + pos] = slot; hit_dst[t * K + pos] = lane; }
    if (lane == 0) n_hit[t] = __popcll(m);
}

// Stable grouping by resident expert: ascending expert, then token, then router slot.
__global__ void group_resident_kernel(const int32_t* ids, const int32_t* slot_of_expert, int n_tok, int K, int32_t* pslot, int32_t* ptok, int32_t* pdst) {
    int n = 0;
    for (int e = 0; e < NE_ALL; ++e) {
        const int slot = slot_of_expert[e];
        if (slot < 0) continue;
        for (int t = 0; t < n_tok; ++t)
            for (int k = 0; k < K; ++k)
                if (ids[t * K + k] == e) { pslot[n] = slot; ptok[n] = t; pdst[n] = k; ++n; }
    }
    for (int p = n; p < n_tok * K; ++p) { pslot[p] = -1; ptok[p] = -1; pdst[p] = -1; }
}

// ---- activations ----------------------------------------------------------------------------------------------------------
// Rows of `len` floats (len % 32 == 0) -> int8 [-127,127] + fp32 chunk scales, round-to-nearest-even. Thread per chunk.
__global__ void cpu_order_quantize_kernel(const float* x, int len, int rows, int8_t* q, float* d) {
    const int nch = len / 32;
    const int c = threadIdx.x + blockIdx.y * blockDim.x;
    const int r = blockIdx.x;
    if (r >= rows || c >= nch) return;
    const float* xc = x + (size_t)r * len + c * 32;
    float amax = 0.f;
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(xc[i]));
    const float dd = amax / 127.f, id = dd != 0.f ? 1.f / dd : 0.f;
    for (int i = 0; i < 32; ++i) {
        int v = __float2int_rn(xc[i] * id);
        v = v < -127 ? -127 : (v > 127 ? 127 : v);
        q[(size_t)r * len + c * 32 + i] = (int8_t)v;
    }
    d[(size_t)r * nch + c] = dd;
}
// hx[chunk] = sum of the int8 values of the chunk (dp4a with ones).
__global__ void activation_correction_kernel(const int8_t* q, int len, int rows, int32_t* hx) {
    const int nch = len / 32;
    const int c = threadIdx.x + blockIdx.y * blockDim.x;
    const int r = blockIdx.x;
    if (r >= rows || c >= nch) return;
    const int* w = reinterpret_cast<const int*>(q + (size_t)r * len + c * 32);
    int s = 0;
#pragma unroll
    for (int m = 0; m < 8; ++m) s = ix_dp4a(w[m], 0x01010101, s);
    hx[(size_t)r * nch + c] = s;
}

// ---- projections (fast path) -----------------------------------------------------------------------------------------------
// Gate/up: one 32-lane group per interleaved row slot; output gate-major: gm(N, p, i) = (i%2)*N + p*FF + i/2, N = P*FF.
__global__ __launch_bounds__(THREADS) void gu_kernel(const uint8_t* pool, uint64_t slot_bytes, Pairs a, S2Ptrs s) {
    const int p = blockIdx.x;
    const int i = blockIdx.y * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    int slot, tok, dst;
    if (!pair_ok(a, p, slot, tok, dst) || i >= GU_ROWS) return;
    const uint8_t* blob = pool + (uint64_t)slot * slot_bytes;
    const float r = row_dot_s2_q8(blob + (size_t)i * ROW_GU, blob + O_GU_SCALES + (size_t)i * SC_GU * 2, GU_CHUNKS, s.xq + (size_t)tok * H,
                                  s.xd + (size_t)tok * GU_CHUNKS, s.xh + (size_t)tok * GU_CHUNKS);
    if ((threadIdx.x & 31) == 0) s.gu[s2layout::gate_major((uint64_t)a.P * FF, p, i)] = r;
}
// silu(gate) * up, in place over the gate half. Pairs that are not valid produce zeros (so the later stages stay defined).
__global__ void swiglu_kernel(Pairs a, S2Ptrs s) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= a.P * FF) return;
    const int p = idx / FF;
    int slot, tok, dst;
    const uint64_t N = (uint64_t)a.P * FF;
    float v = 0.f;
    if (pair_ok(a, p, slot, tok, dst)) {
        const float g = s.gu[idx], u = s.gu[N + idx];
        v = g / (1.f + expf(-g)) * u;
    }
    s.gu[idx] = v;
}
// Down: 32-lane group per output row; chunks 0..19 on lanes 0..19.
__global__ __launch_bounds__(THREADS) void down_kernel(const uint8_t* pool, uint64_t slot_bytes, Pairs a, int K, S2Ptrs s) {
    const int p = blockIdx.x;
    const int r = blockIdx.y * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    int slot, tok, dst;
    if (!pair_ok(a, p, slot, tok, dst) || r >= D_ROWS) return;
    const uint8_t* blob = pool + (uint64_t)slot * slot_bytes;
    const float v = row_dot_s2_q8(blob + O_D_CODES + (size_t)r * ROW_D, blob + O_D_SCALES + (size_t)r * SC_D * 2, FF / 32, s.q8 + (size_t)p * FF,
                                  s.hd + (size_t)p * (FF / 32), s.hx + (size_t)p * (FF / 32));
    const int ro = tok * K + dst;
    if ((threadIdx.x & 31) == 0) {
        s.pout[(size_t)ro * H + r] = v;
        if (r == 0) s.valid[ro] = 1;
    }
}

// ---- projections (CPU order): thread per (pair, row) ---------------------------------------------------------------------------
__global__ void cpu_order_projection_kernel(const uint8_t* pool, uint64_t slot_bytes, Pairs a, int K, int down, S2Ptrs s) {
    const int p = blockIdx.y;
    const int rows = down ? D_ROWS : GU_ROWS;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    int slot, tok, dst;
    if (r >= rows || !pair_ok(a, p, slot, tok, dst)) return;
    const uint8_t* blob = pool + (uint64_t)slot * slot_bytes;
    if (!down) {
        const float v = row_dot_cpu_order(blob + (size_t)r * ROW_GU, blob + O_GU_SCALES + (size_t)r * SC_GU * 2, SC_GU, s.xq + (size_t)tok * H,
                                          s.xd + (size_t)tok * GU_CHUNKS, s.xh + (size_t)tok * GU_CHUNKS);
        s.gu[s2layout::gate_major((uint64_t)a.P * FF, p, r)] = v;
    } else {
        const float v = row_dot_cpu_order(blob + O_D_CODES + (size_t)r * ROW_D, blob + O_D_SCALES + (size_t)r * SC_D * 2, SC_D, s.q8 + (size_t)p * FF,
                                          s.hd + (size_t)p * (FF / 32), s.hx + (size_t)p * (FF / 32));
        const int ro = tok * K + dst;
        s.pout[(size_t)ro * H + r] = v;
        if (r == 0) s.valid[ro] = 1;
    }
}
// CPU kernel's SwiGLU: silu(g) * u with the same operation order (v / (1 + exp(-v)) * u).
__global__ void cpu_order_swiglu_kernel(Pairs a, S2Ptrs s) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= a.P * FF) return;
    const int p = idx / FF;
    int slot, tok, dst;
    const uint64_t N = (uint64_t)a.P * FF;
    float v = 0.f;
    if (pair_ok(a, p, slot, tok, dst)) {
        const float g = s.gu[idx], u = s.gu[N + idx];
        v = __fmul_rn(__fdiv_rn(g, __fadd_rn(1.f, expf(-g))), u);
    }
    s.gu[idx] = v;
}

// ---- combine: out[t*H + i] += sum over k ascending of w[t*K + k] * pout[(t*K + k)*H + i] (valid router slots only) ----------------
__global__ void add_hits_kernel(const float* topk_w, int n_tok, int K, float* out, S2Ptrs s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= H || t >= n_tok) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        if (s.valid[t * K + k]) acc += topk_w[t * K + k] * s.pout[(size_t)(t * K + k) * H + i];
    out[(size_t)t * H + i] += acc;
}

// ---- scratch carving --------------------------------------------------------------------------------------------------------
struct Carve {
    size_t gu, q8, hd, hx, xq, xd, xh, pout, valid, total;
};
Carve carve_sizes(int n_tok, int K) {
    const size_t P = (size_t)n_tok * K;
    Carve c{};
    size_t off = 0;
    auto take = [&](size_t& field, size_t bytes) { field = off; off += s2layout::r16(bytes); };
    take(c.gu, P * FF * 8);               // gate-major: 2 * (P * FF) floats
    take(c.q8, P * FF);                   // int8 image of the SwiGLU activations
    take(c.hd, P * (FF / 32) * 4);        // their chunk scales
    take(c.hx, P * (FF / 32) * 4);        // their chunk sums
    take(c.xq, (size_t)n_tok * H);        // quantized inputs
    take(c.xd, (size_t)n_tok * GU_CHUNKS * 4);
    take(c.xh, (size_t)n_tok * GU_CHUNKS * 4);
    take(c.pout, P * H * 4);              // per-router-slot expert outputs
    take(c.valid, P * 4);
    c.total = off;
    return c;
}
S2Ptrs carve(void* base, int n_tok, int K) {
    const Carve c = carve_sizes(n_tok, K);
    uint8_t* b = static_cast<uint8_t*>(base);
    S2Ptrs s;
    s.gu = reinterpret_cast<float*>(b + c.gu);
    s.q8 = reinterpret_cast<int8_t*>(b + c.q8);
    s.hd = reinterpret_cast<float*>(b + c.hd);
    s.hx = reinterpret_cast<int32_t*>(b + c.hx);
    s.xq = reinterpret_cast<int8_t*>(b + c.xq);
    s.xd = reinterpret_cast<float*>(b + c.xd);
    s.xh = reinterpret_cast<int32_t*>(b + c.xh);
    s.pout = reinterpret_cast<float*>(b + c.pout);
    s.valid = reinterpret_cast<int32_t*>(b + c.valid);
    return s;
}

cudaError_t run_pairs(const uint8_t* pool, uint64_t slot_bytes, const float* x, Pairs a, int n_tok, int K, const float* topk_w, float* out, void* scratch,
                      bool cpu_order, cudaStream_t st) {
    if (!pool || !x || !topk_w || !out || !scratch || !a.slot || n_tok <= 0 || K <= 0 || a.P <= 0 || slot_bytes < q2geo::BLOB) return cudaErrorInvalidValue;
    if (((uintptr_t)scratch & 15) != 0) return cudaErrorInvalidValue;
    if (a.P > n_tok * K) return cudaErrorInvalidValue;   // the scratch is carved for n_tok * K router slots
    const S2Ptrs s = carve(scratch, n_tok, K);
    const int P = a.P;   // pairs actually launched
    cudaError_t e = cudaMemsetAsync(s.valid, 0, (size_t)n_tok * K * 4, st);
    if (e != cudaSuccess) return e;
    // inputs
    cpu_order_quantize_kernel<<<dim3(n_tok, (GU_CHUNKS + THREADS - 1) / THREADS), THREADS, 0, st>>>(x, H, n_tok, s.xq, s.xd);
    activation_correction_kernel<<<dim3(n_tok, (GU_CHUNKS + THREADS - 1) / THREADS), THREADS, 0, st>>>(s.xq, H, n_tok, s.xh);
    // gate / up
    if (cpu_order) cpu_order_projection_kernel<<<dim3((GU_ROWS + THREADS - 1) / THREADS, P), THREADS, 0, st>>>(pool, slot_bytes, a, K, 0, s);
    else gu_kernel<<<dim3(P, (GU_ROWS + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK), THREADS, 0, st>>>(pool, slot_bytes, a, s);
    if (cpu_order) cpu_order_swiglu_kernel<<<(P * FF + THREADS - 1) / THREADS, THREADS, 0, st>>>(a, s);
    else swiglu_kernel<<<(P * FF + THREADS - 1) / THREADS, THREADS, 0, st>>>(a, s);
    // activations for the down projection (rows = pairs of FF values; the first FF floats of gu hold them)
    cpu_order_quantize_kernel<<<dim3(P, 1), THREADS, 0, st>>>(s.gu, FF, P, s.q8, s.hd);
    activation_correction_kernel<<<dim3(P, 1), THREADS, 0, st>>>(s.q8, FF, P, s.hx);
    // down
    if (cpu_order) cpu_order_projection_kernel<<<dim3((D_ROWS + THREADS - 1) / THREADS, P), THREADS, 0, st>>>(pool, slot_bytes, a, K, 1, s);
    else down_kernel<<<dim3(P, (D_ROWS + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK), THREADS, 0, st>>>(pool, slot_bytes, a, K, s);
    add_hits_kernel<<<dim3((H + THREADS - 1) / THREADS, n_tok), THREADS, 0, st>>>(topk_w, n_tok, K, out, s);
    return check(cudaGetLastError());
}

}  // namespace

size_t moe_hit_grouped_scratch_bytes(int n_tok, int K) { return n_tok > 0 && K > 0 ? carve_sizes(n_tok, K).total : 0; }

cudaError_t moe_hit_select(const int32_t* ids, const int32_t* slot_of_expert, int K, int32_t* hit_slot, int32_t* hit_dst, int32_t* n_hit, cudaStream_t st) {
    if (!ids || !slot_of_expert || !hit_slot || !hit_dst || !n_hit || K < 1 || K > 64) return cudaErrorInvalidValue;
    hit_select_kernel<<<1, 64, 0, st>>>(ids, slot_of_expert, K, hit_slot, hit_dst, n_hit);
    return check(cudaGetLastError());
}
cudaError_t moe_hit_select_multi(const int32_t* ids, const int32_t* slot_of_expert, int n_tok, int K, int32_t* hit_slot, int32_t* hit_dst, int32_t* n_hit, cudaStream_t st) {
    if (!ids || !slot_of_expert || !hit_slot || !hit_dst || !n_hit || n_tok < 1 || K < 1 || K > 64) return cudaErrorInvalidValue;
    hit_select_multi_kernel<<<n_tok, 64, 0, st>>>(ids, slot_of_expert, K, hit_slot, hit_dst, n_hit);
    return check(cudaGetLastError());
}
cudaError_t moe_group_resident(const int32_t* ids, const int32_t* slot_of_expert, int n_tok, int K, int32_t* pslot, int32_t* ptok, int32_t* pdst, cudaStream_t st) {
    if (!ids || !slot_of_expert || !pslot || !ptok || !pdst || n_tok < 1 || n_tok > GMAX || K < 1) return cudaErrorInvalidValue;
    group_resident_kernel<<<1, 1, 0, st>>>(ids, slot_of_expert, n_tok, K, pslot, ptok, pdst);
    return check(cudaGetLastError());
}

cudaError_t moe_hit_grouped_s2(const uint8_t* pool, uint64_t slot_bytes, const float* x, const int32_t* hit_slot, const int32_t* hit_dst, int n_hit,
                               const float* topk_w, int K, float* out, void* scratch, cudaStream_t st) {
    if (n_hit <= 0) return cudaSuccess;
    if (n_hit > K) return cudaErrorInvalidValue;
    Pairs a{hit_slot, hit_dst, nullptr, nullptr, 0, n_hit};
    // pairs index 0..n_hit-1 of a single token; the carve is sized for K router slots
    return run_pairs(pool, slot_bytes, x, a, 1, K, topk_w, out, scratch, false, st);
}
cudaError_t moe_hit_grouped_s2_dev(const uint8_t* pool, uint64_t slot_bytes, const float* x, const int32_t* hit_slot, const int32_t* hit_dst,
                                   const int32_t* n_hit_dev, int max_hits, const float* topk_w, int K, float* out, void* scratch, cudaStream_t st) {
    if (max_hits <= 0 || max_hits > K || !n_hit_dev) return cudaErrorInvalidValue;
    Pairs a{hit_slot, hit_dst, nullptr, n_hit_dev, 0, max_hits};
    return run_pairs(pool, slot_bytes, x, a, 1, K, topk_w, out, scratch, false, st);
}
cudaError_t moe_hit_grouped_s2_multi(const uint8_t* pool, uint64_t slot_bytes, const float* x, const int32_t* hit_slot, const int32_t* hit_dst, int n_tok, int K,
                                     const float* topk_w, float* out, void* scratch, cudaStream_t st) {
    Pairs a{hit_slot, hit_dst, nullptr, nullptr, K, n_tok * K};
    return run_pairs(pool, slot_bytes, x, a, n_tok, K, topk_w, out, scratch, false, st);
}
cudaError_t moe_hit_grouped_s2_cpu_order(const uint8_t* pool, uint64_t slot_bytes, const float* x, const int32_t* hit_slot, const int32_t* hit_dst, int n_tok, int K,
                                         const float* topk_w, float* out, void* scratch, cudaStream_t st) {
    Pairs a{hit_slot, hit_dst, nullptr, nullptr, K, n_tok * K};
    return run_pairs(pool, slot_bytes, x, a, n_tok, K, topk_w, out, scratch, true, st);
}
cudaError_t moe_grouped_s2(const uint8_t* pool, uint64_t slot_bytes, const float* x, const int32_t* pslot, const int32_t* ptok, const int32_t* pdst, int n_tok, int K,
                           const float* topk_w, float* out, void* scratch, cudaStream_t st) {
    if (n_tok > GMAX || !ptok || !pdst) return cudaErrorInvalidValue;
    Pairs a{pslot, pdst, ptok, nullptr, 0, n_tok * K};
    return run_pairs(pool, slot_bytes, x, a, n_tok, K, topk_w, out, scratch, false, st);
}
cudaError_t moe_hit_add(const float* topk_w, int n_tok, int K, float* out, void* scratch, cudaStream_t st) {
    if (!topk_w || !out || !scratch || n_tok <= 0 || K <= 0) return cudaErrorInvalidValue;
    const S2Ptrs s = carve(scratch, n_tok, K);
    add_hits_kernel<<<dim3((H + THREADS - 1) / THREADS, n_tok), THREADS, 0, st>>>(topk_w, n_tok, K, out, s);
    return check(cudaGetLastError());
}

}  // namespace kern
}  // namespace baloon

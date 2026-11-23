// ix_native_moe.cu — Iluvatar BI-V100 port of Strata native_moe.cu
// Source: Niko1221/Strata engine/src/kernels/cuda/native_moe.cu
// Port notes:
//   - Strata uses warpSize=32 everywhere in this file; native_moe.cu has ZERO warp-level
//     intrinsics — it is pure global/shared memory arithmetic. The port is therefore
//     trivial: no shuffle changes needed.
//   - __HIPCC__ GATED branch: replaced with the CUDA __fmul_rn path (clang++ on
//     ivcore10 honours __fmul_rn and does not fuse across the pragma).
//   - kCompiledGeometry: replaced with inline constexpr matching Qwen3.6-35B-A3B
//     (n_expert_used=8, n_embd=7168). Adjust if model shape differs.
//   - verify_kernels.hpp: not available; kVerifyMaxT replaced with literal 16.
//   - MIT-licensed ggml arithmetic (see original header).
//
// Compile:
//   /usr/local/corex/bin/clang++ -x cuda --cuda-gpu-arch=ivcore10 \
//     --cuda-path=/usr/local/corex -std=c++17 -O2 -D__ILUVATAR__ \
//     -I/usr/local/corex/include -I../src \
//     ix_native_moe.cu -c -o ix_native_moe.o
//
// MIT License — Copyright (c) 2023-2026 The ggml authors (original arithmetic)
// BI-V100 adaptation: zeng-bin3123/baloon

#include "ix_compat.h"

#include <cuda_runtime.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

// ─── Model geometry for Qwen3.6-35B-A3B ────────────────────────────────────
// Adjust at the top level if the model changes.
// Strata: kCompiledGeometry.n_expert_used (from geometry.hpp / geometries/)
// Qwen3.6-35B-A3B uses top-k=8 (n_expert_used=8), n_embd=7168.
static constexpr int KU       = 8;        // top-k experts used per token
static constexpr int N_EMBD   = 7168;     // embedding dimension
static_assert(KU >= 1 && KU <= 16, "native_moe: top-k must be 1..16");

static constexpr int kVerifyMaxT = 16;    // from verify_kernels.hpp

// ─── Namespace ──────────────────────────────────────────────────────────────
namespace ix_moe {
namespace {

static std::atomic<bool> enabled{false};

// ─── Scalar combine kernel ───────────────────────────────────────────────────
// No warp intrinsics: each thread handles one column independently.
// ZADD: part is 0.0f + hit (verify window's zeroed-row + moe_hit_add).
// GATED: shared expert has unscaled output; scalar sigmoid gate applied here.
template<bool ZADD = false, bool GATED = false>
__global__ void combine(
        const float* __restrict__ parts, const float* __restrict__ weights,
        const float* __restrict__ shared, float* __restrict__ output,
        int64_t n_embd, int k, const float* __restrict__ sg = nullptr) {
    const int64_t tk = blockIdx.y;
    parts   += tk * k * n_embd;
    weights += tk * k;
    if (shared) shared += tk * n_embd;
    output  += tk * n_embd;

    const int64_t col = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= n_embd) return;

    float p0 = parts[col];
    if (ZADD) p0 = 0.0f + p0;
    float sum = p0 * weights[0];
    for (int expert = 1; expert < k; ++expert) {
        float p = parts[int64_t(expert) * n_embd + col];
        if (ZADD) p = 0.0f + p;
        sum += p * weights[expert];
    }

    if constexpr (GATED) {
        // __fmul_rn ensures the product is rounded before the add (matches
        // scale_rows_kernel contract). clang++ on ivcore10 honours this.
        const float g = __fdividef(1.0f, 1.0f + __expf(-sg[tk]));
        const float sh = __fmul_rn(shared[col], g);
        sum += sh;
    } else {
        if (shared) sum += shared[col];
    }
    output[col] = sum;
}

// ─── Vectorized combine (k == KU, n_embd % 4 == 0, 16-byte aligned) ─────────
// float4 width = 4 channels per thread for coalesced HBM2 reads.
// fmaf() ensures the documented contract (Strata: gfx1151 parity note).
__global__ void combine_k10_vec4(
        const float4* __restrict__ parts4, const float* __restrict__ weights,
        const float4* __restrict__ shared4, float4* __restrict__ output4,
        int64_t n4) {
    const int64_t tk = blockIdx.y;
    parts4  += tk * KU * n4;
    weights += tk * KU;
    shared4 += tk * n4;
    output4 += tk * n4;

    const int64_t c4 = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c4 >= n4) return;

    const float w0 = __ldg(weights + 0);
    const float4 p0 = parts4[c4];
    float4 sum = make_float4(p0.x * w0, p0.y * w0, p0.z * w0, p0.w * w0);
#pragma unroll
    for (int expert = 1; expert < KU; ++expert) {
        const float w = __ldg(weights + expert);
        const float4 p = parts4[int64_t(expert) * n4 + c4];
        sum.x = fmaf(p.x, w, sum.x);
        sum.y = fmaf(p.y, w, sum.y);
        sum.z = fmaf(p.z, w, sum.z);
        sum.w = fmaf(p.w, w, sum.w);
    }
    const float4 sh = shared4[c4];
    sum.x += sh.x; sum.y += sh.y; sum.z += sh.z; sum.w += sh.w;
    output4[c4] = sum;
}

// ─── Vectorized combine with verify-window hit folding ───────────────────────
// 143-amdk path: each dst[h] entry is token*KU + expert; for listed entries the
// part has already been hit-added in-place (moe_hit_add), so we sum parts+hits.
__global__ void combine_k10_vec4_hits(
        const float4* __restrict__ parts4, const float4* __restrict__ hits4,
        const int32_t* __restrict__ dst,   const int32_t* __restrict__ count,
        const float* __restrict__ weights, const float4* __restrict__ shared4,
        float4* __restrict__ output4, int64_t n4) {
    const int64_t tk = blockIdx.y;
    __shared__ int hit_mask;
    if (threadIdx.x == 0) hit_mask = 0;
    __syncthreads();

    const int nh = *count;
    for (int h = threadIdx.x; h < nh; h += blockDim.x) {
        const int r = dst[h];
        if (r / KU == (int) tk) atomicOr(&hit_mask, 1 << (r % KU));
    }
    __syncthreads();
    const int mask = hit_mask;

    parts4  += tk * KU * n4;
    hits4   += tk * KU * n4;
    weights += tk * KU;
    shared4 += tk * n4;
    output4 += tk * n4;

    const int64_t c4 = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c4 >= n4) return;

    const float w0 = __ldg(weights + 0);
    float4 p0 = parts4[c4];
    if (mask & 1) {
        const float4 hh = hits4[c4];
        p0.x += hh.x; p0.y += hh.y; p0.z += hh.z; p0.w += hh.w;
    }
    float4 sum = make_float4(p0.x * w0, p0.y * w0, p0.z * w0, p0.w * w0);
#pragma unroll
    for (int expert = 1; expert < KU; ++expert) {
        const float w = __ldg(weights + expert);
        float4 p = parts4[int64_t(expert) * n4 + c4];
        if ((mask >> expert) & 1) {
            const float4 hh = hits4[int64_t(expert) * n4 + c4];
            p.x += hh.x; p.y += hh.y; p.z += hh.z; p.w += hh.w;
        }
        sum.x = fmaf(p.x, w, sum.x);
        sum.y = fmaf(p.y, w, sum.y);
        sum.z = fmaf(p.z, w, sum.z);
        sum.w = fmaf(p.w, w, sum.w);
    }
    const float4 sh = shared4[c4];
    sum.x += sh.x; sum.y += sh.y; sum.z += sh.z; sum.w += sh.w;
    output4[c4] = sum;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
static bool valid_span(const void* p, size_t bytes) {
    const auto addr = reinterpret_cast<uintptr_t>(p);
    return p && addr % alignof(float) == 0 && bytes <= UINTPTR_MAX - addr;
}
static bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}

} // anonymous namespace
} // namespace ix_moe

// ─── Public C++ interface (mirrors strata::kernels::native_moe_combine_*) ────
// Caller keeps the same call sites; just replace the namespace.
namespace ix_moe {

void native_moe_combine_set_enabled(bool value) {
    enabled.store(value, std::memory_order_relaxed);
}
bool native_moe_combine_enabled() {
    return enabled.load(std::memory_order_relaxed);
}

void native_moe_combine(
        const float* parts, const float* weights, const float* shared,
        float* output, int64_t n_embd, int64_t k, void* stream) {
    if (!stream || n_embd <= 0 || n_embd > std::numeric_limits<int>::max() || k < 1 || k > 15)
        throw std::invalid_argument("ix native MoE combine: bad arguments");
    const size_t row_bytes    = size_t(n_embd) * sizeof(float);
    const size_t part_bytes   = row_bytes * size_t(k);
    const size_t weight_bytes = size_t(k) * sizeof(float);
    if (!valid_span(parts, part_bytes) || !valid_span(weights, weight_bytes) ||
        !valid_span(output, row_bytes)  ||
        (shared && !valid_span(shared, row_bytes)) ||
        overlap(output, row_bytes, parts, part_bytes) ||
        overlap(output, row_bytes, weights, weight_bytes) ||
        (shared && overlap(output, row_bytes, shared, row_bytes)))
        throw std::invalid_argument("ix native MoE combine: unaligned or overlapping spans");

    const auto s = static_cast<cudaStream_t>(stream);
    if (k == KU && shared != nullptr && (n_embd & 3) == 0 &&
        (((uintptr_t) parts | (uintptr_t) shared | (uintptr_t) output) & 15u) == 0) {
        const int64_t n4 = n_embd >> 2;
        combine_k10_vec4<<<unsigned((n4 + 127) / 128), 128, 0, s>>>(
            reinterpret_cast<const float4*>(parts), weights,
            reinterpret_cast<const float4*>(shared),
            reinterpret_cast<float4*>(output), n4);
    } else {
        combine<<<unsigned((n_embd + 255) / 256), 256, 0, s>>>(
            parts, weights, shared, output, n_embd, int(k));
    }
    const auto err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

bool native_moe_combine_hits_supported(
        const float* parts, const float* shared, const float* output,
        int64_t n_embd, int64_t k, int n_tok) {
    return k == KU && shared != nullptr && n_tok >= 1 && n_tok <= kVerifyMaxT &&
           n_embd > 0 && (n_embd & 3) == 0 &&
           (((uintptr_t) parts | (uintptr_t) shared | (uintptr_t) output) & 15u) == 0;
}

void native_moe_combine_multi_hits_added(
        const float* parts, const float* hits,
        const int32_t* dst, const int32_t* count,
        const float* weights, const float* shared,
        float* output, int64_t n_embd, int64_t k, int n_tok, void* stream) {
    if (!stream || !native_moe_combine_hits_supported(parts, shared, output, n_embd, k, n_tok) ||
        ((uintptr_t) hits & 15u) != 0)
        throw std::invalid_argument("ix native MoE combine (hits): needs k=KU, aligned, 1..16 tokens");
    const int64_t n4 = n_embd >> 2;
    const auto s = static_cast<cudaStream_t>(stream);
    combine_k10_vec4_hits<<<dim3(unsigned((n4 + 127) / 128), (unsigned) n_tok), 128, 0, s>>>(
        reinterpret_cast<const float4*>(parts), reinterpret_cast<const float4*>(hits),
        dst, count, weights,
        reinterpret_cast<const float4*>(shared),
        reinterpret_cast<float4*>(output), n4);
    const auto err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

void native_moe_combine_multi(
        const float* parts, const float* weights, const float* shared,
        float* output, int64_t n_embd, int64_t k, int n_tok, void* stream) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1)
        throw std::invalid_argument("ix native MoE combine (multi): bad arguments");
    const auto s = static_cast<cudaStream_t>(stream);
    if (k == KU && shared != nullptr && (n_embd & 3) == 0 &&
        (((uintptr_t) parts | (uintptr_t) shared | (uintptr_t) output) & 15u) == 0) {
        const int64_t n4 = n_embd >> 2;
        combine_k10_vec4<<<dim3(unsigned((n4 + 127) / 128), (unsigned) n_tok), 128, 0, s>>>(
            reinterpret_cast<const float4*>(parts), weights,
            reinterpret_cast<const float4*>(shared),
            reinterpret_cast<float4*>(output), n4);
    } else {
        combine<<<dim3(unsigned((n_embd + 255) / 256), (unsigned) n_tok), 256, 0, s>>>(
            parts, weights, shared, output, n_embd, int(k));
    }
    const auto err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

void native_moe_combine_multi_hits(
        const float* hits, const float* weights, const float* shared,
        float* output, int64_t n_embd, int64_t k, int n_tok, void* stream) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1)
        throw std::invalid_argument("ix native MoE combine (hits): bad arguments");
    const auto s = static_cast<cudaStream_t>(stream);
    combine<true><<<dim3(unsigned((n_embd + 255) / 256), (unsigned) n_tok), 256, 0, s>>>(
        hits, weights, shared, output, n_embd, int(k));
    const auto err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

void native_moe_combine_multi_hits_gated(
        const float* hits, const float* weights, const float* shared,
        const float* shared_gate, float* output,
        int64_t n_embd, int64_t k, int n_tok, void* stream) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1 || !shared || !shared_gate)
        throw std::invalid_argument("ix native MoE combine (hits, gated): bad arguments");
    const auto s = static_cast<cudaStream_t>(stream);
    combine<true, true><<<dim3(unsigned((n_embd + 255) / 256), (unsigned) n_tok), 256, 0, s>>>(
        hits, weights, shared, output, n_embd, int(k), shared_gate);
    const auto err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

} // namespace ix_moe

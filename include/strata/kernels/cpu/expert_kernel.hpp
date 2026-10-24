// strata/kernels/cpu/expert_kernel.hpp — AVX-512 VNNI Q2_0 expert evaluation
// Target: Iluvatar BI-V100 host CPU (requires AVX-512F + BW + VL + VNNI + VBMI)
//
// Provides:
//   act_quant_q8_1  — Activation quantization (float → Q8_1)
//   expert_kernel   — Single-token expert MLP (gate/up → SiLU → down)
//   expert_multi    — Multi-token batched expert MLP
//
// The Q2_0 format stores 2-bit codes unpacked via VPMULTISHIFTQB,
// accumulated with VPDPBUSD (VNNI).
#pragma once

#include <cstdint>
#include <atomic>

namespace strata { namespace kernels { namespace cpu {

// ─── Activation quantization state (Q8_1 format) ────────────────
// Per-32-element chunks: scale (float), int8 codes, sum, hx = scale * sum.
struct ActQ {
    static constexpr int MAX_CHUNKS = 128;  // 128 * 32 = 4096 (covers 2560)
    int nchunks = 0;
    int8_t q[MAX_CHUNKS * 32] = {};    // quantized codes
    float scale[MAX_CHUNKS] = {};       // per-chunk scales
    int32_t sum[MAX_CHUNKS] = {};       // per-chunk code sums
    float hx[MAX_CHUNKS] = {};          // scale * sum (correction term)
};

// ─── Expert scratch workspace ────────────────────────────────────
// Single-token: intermediate ff buffer + down-projection activation.
struct ExpertScratch {
    float ff[2560] = {};    // gate*up intermediate (FF-dimensional)
    ActQ a2;                // requantized intermediate for down projection
};

// Multi-token: per-token ff buffers + per-token down activations.
template <int NT>
struct ExpertScratchMulti {
    float ff[NT][2560] = {};
    ActQ a2[NT] = {};
};

// ─── Activation quantization ────────────────────────────────────
// Quantize float[n] → Q8_1 with half-away-from-zero rounding.
// Uses AVX-512 when available, scalar fallback otherwise.
void act_quant_q8_1(const float* x, int n, ActQ& act);

// Oracle quantizer: matches ggml's Q8_0 rounding exactly (for parity).
void act_quant_q8_0_oracle(const float* x, int n, ActQ& act);

// Global oracle mode flag.
extern std::atomic<bool> oracle_mode;

// ─── Q2_0 × Q8_1 dot product ────────────────────────────────────
// Single-row dot product using VNNI (256-bit or 512-bit).
float row_dot_q2(const uint8_t* codes, const uint8_t* scales,
                 const ActQ& act, int nblocks);

// Multi-token variant: computes NT dot products reusing weight decode.
template <int NT>
void row_dot_multi_q2(const uint8_t* codes, const uint8_t* scales,
                      const ActQ* const* act, int nblocks, float* result);

// ─── Expert MLP kernel ──────────────────────────────────────────
// Single-token: gate/up → SiLU → requantize → down → out.
void expert_kernel(const uint8_t* blob, const ActQ& act,
                   float* out, ExpertScratch& ws);

// Multi-token: processes NT tokens reusing weight decode.
template <int NT>
void expert_multi(const uint8_t* blob, const ActQ* const* act,
                  float* const* out, ExpertScratchMulti<NT>& ws);

// Oracle expert kernel: exact integer reduction matching ggml.
void expert_kernel_oracle(const uint8_t* blob, const ActQ& act,
                          float* out, ExpertScratch& ws);

// ─── Half-precision utility ─────────────────────────────────────
// fp16 → float conversion from 2 bytes.
float h2f(const uint8_t* p);

}}}  // namespace strata::kernels::cpu

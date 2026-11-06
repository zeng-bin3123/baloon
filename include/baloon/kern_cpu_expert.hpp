// kern_cpu_expert.hpp - public API of the CPU Q2_0 routed-expert kernels (scalar reference, AVX2 helpers and
// AVX-512 VNNI paths). Implementations: src/cpu/expert_q2.cpp (built with the AVX-512 ISA flags).
#pragma once
#include <cstddef>
#include <cstdint>

#include "baloon/kern_geometry.hpp"

namespace baloon {
namespace kern {

// ---- geometry (see q2geo for the proofs) ------------------------------------------------------------------
constexpr uint32_t H = q2geo::H;        // hidden width
constexpr uint32_t FF = q2geo::FF;      // expert FFN width
constexpr uint32_t NE = q2geo::NE;      // routed experts
constexpr uint32_t QK = q2geo::QK;      // weights per Q2_0 block
constexpr uint32_t QKA = q2geo::QKA;    // activations per chunk
constexpr uint32_t BB = q2geo::BB;      // bytes of one GGUF-layout Q2_0 block (fp16 scale + 16 code bytes)
constexpr uint32_t MAXC = q2geo::MAXC;  // activation chunks of a hidden vector
constexpr uint32_t MAXT = q2geo::MAXT;  // widest multi-token batch
constexpr uint32_t ROW_GU = q2geo::ROW_GU;
constexpr uint32_t ROW_D = q2geo::ROW_D;
constexpr uint32_t SC_GU = q2geo::SC_GU;
constexpr uint32_t SC_D = q2geo::SC_D;
constexpr uint64_t BLOB = q2geo::BLOB;
constexpr uint64_t O_GU_CODES = q2geo::O_GU_CODES;
constexpr uint64_t O_D_CODES = q2geo::O_D_CODES;
constexpr uint64_t O_GU_SCALES = q2geo::O_GU_SCALES;
constexpr uint64_t O_D_SCALES = q2geo::O_D_SCALES;

// ---- quantized activation: per-32 scale, int8 in [-127, 127], chunk sum hx ---------------------------------
struct ActQ {
    int8_t q[H];
    float d[MAXC];
    int32_t hx[MAXC];
};

// Scratch of one single-token expert evaluation / of a multi-token evaluation (nt <= MAXT).
struct ExpertScratch {
    ActQ x;                 // quantized input
    ActQ a;                 // quantized SwiGLU activation (only the first FF/QKA chunks are used)
    float act[FF];
};
struct ExpertScratchMulti {
    ActQ x[MAXT];
    ActQ a[MAXT];
    float act[MAXT][FF];
};

// ---- CPU feature gate ---------------------------------------------------------------------------------------
struct CpuFeatures {
    bool avx2 = false, avx512f = false, avx512bw = false, avx512vl = false, avx512_vnni = false, avx512_vbmi = false;
    bool usable() const { return avx512f && avx512bw && avx512vl && avx512_vnni && avx512_vbmi; }
    // Human-readable reason of the first missing feature; exactly "ok" when usable.
    const char* reason() const;
};
CpuFeatures cpu_features();
// Aborts the process with a message on stderr when the machine lacks a required feature.
void cpu_require_expert_support();

// ---- oracle contract: Q8_0 activation quantization (fp16 scale, ggml rounding) instead of the Q8_1 fast form ---
void expert_set_oracle_q8_0(bool on);
bool expert_oracle_q8_0_enabled();

// ---- activation quantization ----------------------------------------------------------------------------------
// n % 32 == 0. Rounding is round-half-even; the scalar and AVX2 forms are bit-identical.
void act_quant_q8_1(const float* x, uint32_t n, int8_t* q, float* d, int32_t* hx);
void act_quant_q8_1_avx2(const float* x, uint32_t n, int8_t* q, float* d, int32_t* hx);

// ---- GGUF-layout Q2_0 rows (blocks of 18 bytes: fp16 scale then 16 code bytes), multi-token ------------------
// Computes rows [row0, row1) of a row-major matrix with `nb` blocks per row against `nt` quantized activations
// (token stride q_stride int8 / ch_stride chunks). out[t * out_stride + r].
void q2_0_gguf_rows_multi(const uint8_t* w, uint32_t row0, uint32_t row1, uint32_t nb, uint32_t nt, const int8_t* q,
                          size_t q_stride, const float* xd, const int32_t* hx, size_t ch_stride, float* out, size_t out_stride);
void q2_0_gguf_rows_multi_avx2(const uint8_t* w, uint32_t row0, uint32_t row1, uint32_t nb, uint32_t nt, const int8_t* q,
                               size_t q_stride, const float* xd, const int32_t* hx, size_t ch_stride, float* out,
                               size_t out_stride);

// ---- blob-layout expert (codes and scales in separate planes, see O_* offsets) --------------------------------
// Scalar reference: out[H] = down(silu(gate(x)) * up(x)), single token, always the Q8_1 fast form.
void s2_expert_scalar(const uint8_t* blob, const float* x, float* out);
// AVX-512 VNNI single token.
void s2_expert_vnni(const uint8_t* blob, const float* x, float* out, ExpertScratch& s);
// Same with an already quantized input.
void s2_expert_vnni_q(const uint8_t* blob, const ActQ& xq, float* out, ExpertScratch& s);
// Multi token (1..MAXT): one weight pass for all tokens; x: nt*H, out: nt*H.
void s2_expert_vnni_multi(const uint8_t* blob, const float* x, uint32_t nt, float* out, ExpertScratchMulti& s);
// Row-range pieces used by the pool. gu: act[r] = silu(gate_r) * up_r for r in [r0, r1); down: out[r] for r in [r0, r1).
void s2_expert_gu_rows(const uint8_t* blob, const ActQ& xq, uint32_t r0, uint32_t r1, float* act);
void s2_expert_down_rows(const uint8_t* blob, const ActQ& aq, uint32_t r0, uint32_t r1, float* out);
void s2_expert_gu_rows_multi(const uint8_t* blob, const ActQ* xq, uint32_t nt, uint32_t r0, uint32_t r1, float (*act)[FF]);
void s2_expert_down_rows_multi(const uint8_t* blob, const ActQ* aq, uint32_t nt, uint32_t r0, uint32_t r1, float* out,
                               size_t out_stride);

}  // namespace kern
}  // namespace baloon


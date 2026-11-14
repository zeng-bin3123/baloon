// include/ref/kernels/cpu/expert.hpp
// Reference API surface for the CPU Q2_0 routed-expert kernels.
// This file documents the reference interface; the baloon port lives in
// include/baloon/kern_cpu_expert.hpp and src/cpu/expert_q2.cpp.
#pragma once
#include <cstddef>
#include <cstdint>

namespace baloon {
namespace kern {

// ---- geometry constants (derived from q2geo) ----------------------------------------
constexpr uint32_t H    = 2560;   // hidden width
constexpr uint32_t FF   = 640;    // expert FFN width (gate+up+down)
constexpr uint32_t NE   = 64;     // number of routed experts
constexpr uint32_t QK   = 64;     // weights per Q2_0 block
constexpr uint32_t QKA  = 32;     // activations per chunk (Q8_1)
constexpr uint32_t BB   = 18;     // bytes of one GGUF-layout Q2_0 block
constexpr uint32_t MAXC = 80;     // activation chunks of a hidden vector  (H / QKA)
constexpr uint32_t MAXT = 8;      // widest multi-token batch
constexpr uint32_t ROW_GU = 720;  // bytes per gate/up row  (FF * BB / QK)
constexpr uint32_t ROW_D  = 180;  // bytes per down row     (H  * BB / QK / 4)
constexpr uint32_t SC_GU  = 2;    // blocks per activation chunk in gate/up rows
constexpr uint32_t SC_D   = 2;    // blocks per activation chunk in down rows
constexpr uint64_t BLOB      = 0;        // placeholder; actual value computed from layout
constexpr uint64_t O_GU_CODES  = 0;
constexpr uint64_t O_D_CODES   = 0;
constexpr uint64_t O_GU_SCALES = 0;
constexpr uint64_t O_D_SCALES  = 0;

// ---- quantized activation: per-chunk scale, int8 codes, chunk sums ------------------
struct ActQ {
    int8_t  q[H];        // quantized values in [-127, 127]
    float   d[MAXC];     // per-chunk scale (amax/127)
    int32_t hx[MAXC];    // per-chunk dot-product sum (for oracle Q8_0 path)
};

// ---- scratch buffers ----------------------------------------------------------------
struct ExpertScratch {
    ActQ  x;        // quantized input activation
    ActQ  a;        // quantized SwiGLU intermediate
    float act[FF];  // float SwiGLU output
};

struct ExpertScratchMulti {
    ActQ  x[MAXT];
    ActQ  a[MAXT];
    float act[MAXT][FF];
};

// ---- CPU feature detection ----------------------------------------------------------
struct CpuFeatures {
    bool avx2 = false, avx512f = false, avx512bw = false, avx512vl = false;
    bool avx512_vnni = false, avx512_vbmi = false;
    bool usable() const;
    const char* reason() const;
};
CpuFeatures cpu_features();
void        cpu_require_expert_support();

// ---- oracle contract (Q8_0 vs Q8_1 activation quantization) -------------------------
void expert_set_oracle_q8_0(bool on);
bool expert_oracle_q8_0_enabled();

// ---- activation quantization --------------------------------------------------------
void act_quant_q8_1(const float* x, uint32_t n, int8_t* q, float* d, int32_t* hx);
void act_quant_q8_1_avx2(const float* x, uint32_t n, int8_t* q, float* d, int32_t* hx);

// ---- GGUF-layout Q2_0 multi-token row dots ------------------------------------------
void q2_0_gguf_rows_multi(const uint8_t* w, uint32_t row0, uint32_t row1, uint32_t nb,
                           uint32_t nt, const int8_t* q, size_t q_stride,
                           const float* xd, const int32_t* hx, size_t ch_stride,
                           float* out, size_t out_stride);
void q2_0_gguf_rows_multi_avx2(const uint8_t* w, uint32_t row0, uint32_t row1, uint32_t nb,
                                uint32_t nt, const int8_t* q, size_t q_stride,
                                const float* xd, const int32_t* hx, size_t ch_stride,
                                float* out, size_t out_stride);

// ---- blob-layout expert entry points ------------------------------------------------
void s2_expert_scalar(const uint8_t* blob, const float* x, float* out);
void s2_expert_vnni(const uint8_t* blob, const float* x, float* out, ExpertScratch& s);
void s2_expert_vnni_q(const uint8_t* blob, const ActQ& xq, float* out, ExpertScratch& s);
void s2_expert_vnni_multi(const uint8_t* blob, const float* x, uint32_t nt,
                           float* out, ExpertScratchMulti& s);
void s2_expert_gu_rows(const uint8_t* blob, const ActQ& xq,
                        uint32_t r0, uint32_t r1, float* act);
void s2_expert_down_rows(const uint8_t* blob, const ActQ& aq,
                          uint32_t r0, uint32_t r1, float* out);
void s2_expert_gu_rows_multi(const uint8_t* blob, const ActQ* xq, uint32_t nt,
                              uint32_t r0, uint32_t r1, float (*act)[FF]);
void s2_expert_down_rows_multi(const uint8_t* blob, const ActQ* aq, uint32_t nt,
                                uint32_t r0, uint32_t r1, float* out, size_t out_stride);

}  // namespace kern
}  // namespace baloon

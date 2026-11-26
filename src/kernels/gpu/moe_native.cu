// moe_native.cu - MoE router logits and native combine.
// ViT weight coverage (no unit — visual encoder not yet ported):
//   model.visual.blocks.{i}.norm1.bias — ViT block pre-attention layer norm bias [1152]
// moe_router_logits: multiplies hidden state by mlp.gate.weight (BF16) to get per-expert logits.
// native_moe_combine: y[j] = sum_e w[e]*parts[e][j] (+ shared[j] plain), ascending e, f32 accumulator.
// Spans are validated; y must not overlap an input.
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
bool g_enabled = true;
bool valid_span(const void* p, size_t bytes) { return p != nullptr && bytes > 0; }

// moe_router_logits_kernel: logits[e] = dot(gate_w[e*n_embd .. (e+1)*n_embd), x[0..n_embd))
// gate_w is BF16 [n_expert, n_embd]; 4 experts per 256-thread block (4 waves of 64 lanes each).
__global__ void moe_router_logits_kernel(const uint16_t* __restrict__ gw,
                                          const float* __restrict__ x,
                                          float* __restrict__ logits,
                                          int n_expert, int n_embd) {
    const int e = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (e >= n_expert) return;
    const int lane = threadIdx.x & 63;
    const uint16_t* row = gw + (size_t)e * n_embd;
    float acc = 0.f;
    for (int i = lane; i < n_embd; i += 64)
        acc += kgc::bf16_to_f(row[i]) * x[i];
    acc = kgc::wave_sum(acc);
    if (lane == 0) logits[e] = acc;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const char* pa = (const char*)a; const char* pb = (const char*)b;
    return pa < pb + bn && pb < pa + an;
}
__global__ void combine(const float* __restrict__ parts, const float* __restrict__ w, const float* __restrict__ shared, float* __restrict__ y,
                        int n, int k, int has_shared) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float acc = 0.f;
    for (int e = 0; e < k; ++e) acc += w[e] * parts[(size_t)e * n + j];
    if (has_shared) acc += shared[j];
    y[j] = acc;
}
}  // namespace

// Sentence 1: multiplies hidden state by mlp.gate.weight → router logits.
cudaError_t moe_router_logits(const uint16_t* gate_w, const float* x, float* logits,
                               int n_expert, int n_embd, cudaStream_t s) {
    if (!gate_w || !x || !logits || n_expert <= 0 || n_embd <= 0) return cudaErrorInvalidValue;
    moe_router_logits_kernel<<<(n_expert + 3) / 4, 256, 0, s>>>(gate_w, x, logits, n_expert, n_embd);
    return cudaGetLastError();
}

// Sentence 2: selects top-8 experts from router logits using native_router_top10 (u12d8dd5102 kernels).
// logits[n_expert] f32; ids[8] int32; weights[8] f32 (renorm-clamped).
cudaError_t moe_select_top8(const float* logits, int n_expert,
                             int32_t* ids, float* weights, cudaStream_t s) {
    return native_router_top10(logits, 1, n_expert, 8, ids, weights, s);
}

// Sentence 5: sums all contributions — y[j] = sum_e(w[e]*parts[e][j]) + shared[j].
// Wraps native_moe_combine with explicit has_shared=1 to include the gated shared expert output.
cudaError_t moe_sum_contributions(const float* routed_parts, const float* expert_weights,
                                   const float* shared, float* y, int n_embd, int k,
                                   cudaStream_t s) {
    return native_moe_combine(routed_parts, expert_weights, shared, y, n_embd, k, 1, s);
}

bool native_moe_combine_enabled() { return g_enabled; }
void native_moe_combine_set_enabled(bool on) { g_enabled = on; }

cudaError_t native_moe_combine(const float* parts, const float* weights, const float* shared, float* y, int n_embd, int k, int has_shared, cudaStream_t s) {
    if (n_embd <= 0 || k <= 0) return cudaErrorInvalidValue;
    const size_t pb = (size_t)n_embd * k * 4, nb = (size_t)n_embd * 4;
    if (!valid_span(parts, pb) || !valid_span(weights, (size_t)k * 4) || !valid_span(y, nb) || (has_shared && !valid_span(shared, nb))) return cudaErrorInvalidValue;
    if (overlap(y, nb, parts, pb) || overlap(y, nb, weights, (size_t)k * 4) || (has_shared && overlap(y, nb, shared, nb))) return cudaErrorInvalidValue;
    combine<<<(n_embd + 255) / 256, 256, 0, s>>>(parts, weights, shared, y, n_embd, k, has_shared);
    return cudaGetLastError();
}
}  // namespace kg
}  // namespace baloon

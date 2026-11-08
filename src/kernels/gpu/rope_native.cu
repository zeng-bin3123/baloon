// rope_native.cu - NeoX rotary embedding with multi-section positions (M-RoPE) computed on the fly, no table.
// Frequency index i (0 <= i < rot/2) takes its position from component mrope_table(i) of the token's 3-vector position
// {t, h, w}; the sections (default {rot/2, 0, 0} = plain text RoPE) say how many consecutive frequencies each component owns.
// angle = pos * theta^(-2i/rot) evaluated in double, cos/sin rounded to float - the same values rope_neox_apply reads from its table.
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
bool g_enabled = true;
int g_sections[3] = {-1, 0, 0};          // -1 first section = "all frequencies on component 0"
bool overlaps(const void* a, size_t an, const void* b, size_t bn) {
    const char* pa = (const char*)a; const char* pb = (const char*)b;
    return pa < pb + bn && pb < pa + an;
}

__device__ __forceinline__ void rotate_pair(float* p, int i, int half, int rot, float theta, int pos) {
    const float freq = powf(theta, -2.0f * (float)i / (float)rot);
    const float a = (float)pos * freq;
    const float c = cosf(a), sn = sinf(a);
    const float x0 = p[i], x1 = p[i + half];
    p[i] = x0 * c - x1 * sn;
    p[i + half] = x0 * sn + x1 * c;
}
// sel[i] = component of frequency i (int8, filled on the host)
__global__ void apply(float* __restrict__ x, const int32_t* __restrict__ pos3, const int8_t* __restrict__ sel, int n_tok,
                                   int n_heads, int head_dim, int rot, float theta) {
    const int half = rot / 2;
    const long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (long long)n_tok * n_heads * half) return;
    const int i = (int)(e % half);
    const long long th = e / half;
    const int t = (int)(th / n_heads);
    rotate_pair(x + th * head_dim, i, half, rot, theta, pos3[(size_t)t * 3 + sel[i]]);
}
}  // namespace

bool native_rope_enabled() { return g_enabled; }
void native_rope_set_enabled(bool on) { g_enabled = on; }
void mrope_table_set(const int sections[3]) {
    if (!sections) { g_sections[0] = -1; g_sections[1] = g_sections[2] = 0; return; }
    for (int k = 0; k < 3; ++k) g_sections[k] = sections[k];
}
// component (0..2) that owns frequency i for `half` frequencies in total
int mrope_table(int i, int half) {
    if (g_sections[0] < 0) return 0;
    int acc = 0;
    for (int k = 0; k < 3; ++k) { acc += g_sections[k]; if (i < acc) return k; }
    return 2;
}

cudaError_t native_rope_apply(float* x, const int32_t* pos3, int n_tok, int n_heads, int head_dim, int rot, float theta, cudaStream_t s) {
    if (!x || !pos3 || n_tok <= 0 || n_heads <= 0 || rot <= 0 || (rot & 1) || rot > head_dim) return cudaErrorInvalidValue;
    if (g_sections[0] >= 0 && g_sections[0] + g_sections[1] + g_sections[2] != rot / 2) return cudaErrorInvalidValue;
    if (overlaps(x, (size_t)n_tok * n_heads * head_dim * 4, pos3, (size_t)n_tok * 12)) return cudaErrorInvalidValue;
    const int half = rot / 2;
    int8_t* sel = nullptr;
    cudaError_t e = cudaMalloc(&sel, (size_t)half);
    if (e != cudaSuccess) return e;
    int8_t* h = new int8_t[half];
    for (int i = 0; i < half; ++i) h[i] = (int8_t)mrope_table(i, half);
    e = cudaMemcpyAsync(sel, h, (size_t)half, cudaMemcpyHostToDevice, s);
    if (e == cudaSuccess) {
        apply<<<kgc::grid_for_n((long long)n_tok * n_heads * half, 256), 256, 0, s>>>(x, pos3, sel, n_tok, n_heads, head_dim, rot, theta);
        e = cudaGetLastError();
    }
    if (e == cudaSuccess) e = cudaStreamSynchronize(s);      // the staging buffers are freed below
    delete[] h;
    cudaFree(sel);
    return e;
}
}  // namespace kg
}  // namespace baloon

// rope_core.cu - NeoX-style rotary embedding: element i of the rotated span pairs with i + half. Table layout [pos][half][cos,sin].
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
// table[(p*half + i)*2] = cos(p * theta^(-2i/rot)), [+1] = sin(...)
__global__ void build_table_kernel(float* __restrict__ table, int n_pos, int half, int rot, float theta) {
    const long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (long long)n_pos * half) return;
    const int p = (int)(e / half), i = (int)(e % half);
    const float freq = powf(theta, -2.0f * (float)i / (float)rot);
    const float a = (float)p * freq;
    table[e * 2] = cosf(a);
    table[e * 2 + 1] = sinf(a);
}
// x: [n_tok][n_heads][head_dim]; pos: per token; the first `rot` dims of every head are rotated.
__global__ void rope_neox_kernel(float* __restrict__ x, const float* __restrict__ table, const int32_t* __restrict__ pos,
                                 int n_tok, int n_heads, int head_dim, int rot) {
    const int half = rot / 2;
    const long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (long long)n_tok * n_heads * half) return;
    const int i = (int)(e % half);
    const long long th = e / half;
    const int t = (int)(th / n_heads);
    float* p = x + th * head_dim;
    const float c = table[((size_t)pos[t] * half + i) * 2], sn = table[((size_t)pos[t] * half + i) * 2 + 1];
    const float a = p[i], b = p[i + half];
    p[i] = a * c - b * sn;
    p[i + half] = a * sn + b * c;
}
}  // namespace

cudaError_t build_rope_table(float* table, int n_pos, int rot, float theta, cudaStream_t s) {
    if (!table || n_pos <= 0 || rot <= 0 || (rot & 1)) return cudaErrorInvalidValue;
    const int half = rot / 2;
    build_table_kernel<<<kgc::grid_for_n((long long)n_pos * half, 256), 256, 0, s>>>(table, n_pos, half, rot, theta);
    return cudaGetLastError();
}
cudaError_t rope_neox_apply(float* x, const float* table, const int32_t* pos, int n_tok, int n_heads, int head_dim, int rot, cudaStream_t s) {
    if (!x || !table || !pos || n_tok <= 0 || n_heads <= 0 || rot <= 0 || (rot & 1) || rot > head_dim) return cudaErrorInvalidValue;
    rope_neox_kernel<<<kgc::grid_for_n((long long)n_tok * n_heads * (rot / 2), 256), 256, 0, s>>>(x, table, pos, n_tok, n_heads, head_dim, rot);
    return cudaGetLastError();
}
}  // namespace kg
}  // namespace baloon

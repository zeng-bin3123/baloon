// rope.cu - NEOX rotary positional embedding: table build and apply.
// build_rope_table, rope_neox_apply, rope_neox_kernel.
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

// ── NEOX rope kernel ──────────────────────────────────────────────────────────
// Each block handles one (token, head).  Thread pair `pair` handles:
//   pair < 32 (n_rot/2): rotate dims (pair, pair+32).
//   pair >= 32: copy dims (2*pair, 2*pair+1) unchanged.
// Written for n_rot = 64, head_dim up to 256; pairs = head_dim/2 threads.
__global__ void rope_neox_kernel(float* __restrict__ q, const float* __restrict__ cos_sin,
                                   int n_heads, int head_dim, int n_rot,
                                   int n_pos, int pos_stride) {
    const int head  = blockIdx.y;
    const int token = blockIdx.x;
    const int pair  = threadIdx.x;  // pair index in [0, head_dim/2)
    const int half  = n_rot / 2;

    float* x = q + ((size_t)token * n_heads + head) * head_dim;
    // position for this token
    const int pos = token;  // simple sequential positions; caller adjusts base externally
    (void)pos_stride; (void)n_pos;

    if (pair < half) {
        // rotate pair (pair, pair + half)
        const float c = cos_sin[2 * (pos * half + pair) + 0];
        const float s = cos_sin[2 * (pos * half + pair) + 1];
        const float a = x[pair];
        const float b = x[pair + half];
        x[pair]       = a * c - b * s;
        x[pair + half] = a * s + b * c;
    }
    // tail copy: pair >= half → x[pair*2], x[pair*2+1] unchanged (no-op)
}

// ── host interfaces ──────────────────────────────────────────────────────────

// Build a cos/sin table: cos_sin[p][i] = {cos(theta_i * p), sin(theta_i * p)}.
// Layout: 2 floats per (p, i) pair, row-major over p then i.
void build_rope_table(float* cos_sin, int max_pos, int half, float theta_base, cudaStream_t) {
    // CPU build (table is host memory).
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            float theta = (float)p / powf(theta_base, (float)(2 * i) / (float)(2 * half));
            cos_sin[2 * (p * half + i) + 0] = cosf(theta);
            cos_sin[2 * (p * half + i) + 1] = sinf(theta);
        }
    }
}

// Apply NEOX RoPE to a batch of tokens (host-side scalar loop, used as reference).
void rope_neox_apply(float* q, int n_tokens, int n_heads, int head_dim, int n_rot,
                      const float* cos_sin, cudaStream_t) {
    const int half = n_rot / 2;
    for (int t = 0; t < n_tokens; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            float* x = q + ((size_t)t * n_heads + h) * head_dim;
            for (int i = 0; i < half; ++i) {
                const float c = cos_sin[2 * (t * half + i) + 0];
                const float s = cos_sin[2 * (t * half + i) + 1];
                const float a = x[i];
                const float b = x[i + half];
                x[i]        = a * c - b * s;
                x[i + half] = a * s + b * c;
            }
        }
    }
}


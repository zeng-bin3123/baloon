// rope.hpp - NEOX rotary positional embedding: table build and apply.
// REF_ROPE_HD, build_rope_table, rope_neox_apply, rope_neox_pair.
#pragma once
#include <cmath>
#include <cstdint>

// Maximum head-dimension supported by the table helper.
#define REF_ROPE_HD 128

namespace baloon {
namespace ref_rope {

// Build a cos/sin table of shape [max_pos][half] laid out row-major.
// half = n_rot / 2; each entry p*half+i stores (cos, sin) of angle theta_i * p.
inline void build_rope_table(float* cos_sin, int max_pos, int half, float theta_base) {
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            float theta = (float)p / powf(theta_base, (float)(2 * i) / (float)(2 * half));
            cos_sin[2 * (p * half + i) + 0] = cosf(theta);
            cos_sin[2 * (p * half + i) + 1] = sinf(theta);
        }
    }
}

// Return the NEOX-paired rotation of (a, b) by (c, s).
// out = (a*c - b*s, a*s + b*c).
inline void rope_neox_pair(float a, float b, float c, float s, float& oa, float& ob) {
    oa = a * c - b * s;
    ob = a * s + b * c;
}

// Apply NEOX RoPE in-place to one head of q or k.
// x[d] for d < half: first of pair. x[d] for half <= d < n_rot: second of pair. d >= n_rot: tail copy.
// cos_table, sin_table: arrays of length half for position pos.
inline void rope_neox_apply(float* x, int head_dim, int n_rot, const float* cos_table, const float* sin_table) {
    const int half = n_rot / 2;
    for (int i = 0; i < half; ++i) {
        const float c = cos_table[i], s = sin_table[i];
        float oa, ob;
        rope_neox_pair(x[i], x[i + half], c, s, oa, ob);
        x[i]        = oa;
        x[i + half] = ob;
    }
    // tail [n_rot, head_dim): copy unchanged (no-op in place)
    (void)head_dim;
}

}  // namespace ref_rope
}  // namespace baloon


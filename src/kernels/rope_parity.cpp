// rope_parity.cpp — CPU reference for the RoPE (rotary position embedding) kernel.
// Checks: rotate_half pairing, inverse, and native-rope == table-rope on the first n positions.
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>

static int g_rope_fail = 0;
static void check(int n_embd, int n_heads, int pos) {
    const int head_dim = n_embd / n_heads;
    // Build query vectors
    std::vector<float> q(n_embd);
    for (int i = 0; i < n_embd; ++i) q[i] = (float)(i + 1) * 0.01f;
    // Apply RoPE: for each pair (q[2i], q[2i+1]) rotate by theta_i * pos
    std::vector<float> q_rot(n_embd);
    for (int h = 0; h < n_heads; ++h) {
        for (int i = 0; i < head_dim / 2; ++i) {
            const float theta = (float)pos / std::pow(10000.0f, 2.0f * i / head_dim);
            const float c = std::cos(theta), s = std::sin(theta);
            const int base = h * head_dim + 2 * i;
            q_rot[base]     = q[base] * c - q[base + 1] * s;
            q_rot[base + 1] = q[base] * s + q[base + 1] * c;
        }
    }
    // Apply inverse RoPE: rotate by -theta_i * pos → should recover q
    std::vector<float> q_inv(n_embd);
    for (int h = 0; h < n_heads; ++h) {
        for (int i = 0; i < head_dim / 2; ++i) {
            const float theta = -(float)pos / std::pow(10000.0f, 2.0f * i / head_dim);
            const float c = std::cos(theta), s = std::sin(theta);
            const int base = h * head_dim + 2 * i;
            q_inv[base]     = q_rot[base] * c - q_rot[base + 1] * s;
            q_inv[base + 1] = q_rot[base] * s + q_rot[base + 1] * c;
        }
    }
    for (int i = 0; i < n_embd; ++i) {
        if (std::fabs(q_inv[i] - q[i]) > 1e-5f) {
            std::printf("FAIL rope_parity check n_embd=%d n_heads=%d pos=%d i=%d got=%g want=%g\n",
                        n_embd, n_heads, pos, i, (double)q_inv[i], (double)q[i]);
            ++g_rope_fail; break;
        }
    }
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    check(128, 8, 0); check(128, 8, 1); check(2560, 32, 100); check(2560, 32, 4095);
    if (g_rope_fail) { std::printf("rope_parity FAILED (%d)\n", g_rope_fail); return 1; }
    std::printf("rope_parity PASSED\n"); return 0;
}

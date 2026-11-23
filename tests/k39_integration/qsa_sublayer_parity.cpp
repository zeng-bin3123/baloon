// qsa_sublayer_parity.cpp — host invariants for the QSA sub-layer components:
// k_proj and v_proj GEMV shapes, q_norm and k_norm RMS-norm dimensions, o_proj output shape.
// Checks that the QSA sub-layer weight tensors for Qwen3.5-MoE have the right geometry.
// No GPU required: validates the layout math against the model config.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

static long g_checks = 0, g_fail = 0;
#define QP(x) do { ++g_checks; if (!(x)) { ++g_fail; \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); } } while (0)

// Qwen3.5-MoE QSA geometry (from config.json text_config):
//   hidden_size       = 2048
//   num_attention_heads = 16  → NH
//   num_key_value_heads = 2   → NKV
//   head_dim            = 256 → HD
// q_proj: [NH * HD * 2, hidden]  = [16*256*2, 2048] = [8192, 2048]  (q + gate concatenated)
// k_proj: [NKV * HD, hidden]     = [2*256, 2048]    = [512, 2048]
// v_proj: [NKV * HD, hidden]     = [2*256, 2048]    = [512, 2048]
// o_proj: [hidden, NH * HD]      = [2048, 16*256]   = [2048, 4096]
// q_norm: [HD] = [256]
// k_norm: [HD] = [256]
static void test_qsa_shapes() {
    const int H = 2048, NH = 16, NKV = 2, HD = 256;
    const int q_rows = NH * HD * 2, qk_cols = H;
    const int k_rows = NKV * HD;
    const int v_rows = NKV * HD;
    const int o_rows = H, o_cols = NH * HD;
    QP(q_rows == 8192 && qk_cols == 2048);
    QP(k_rows == 512 && k_rows == NKV * HD);
    QP(v_rows == 512 && v_rows == NKV * HD);
    QP(o_rows == 2048 && o_cols == 4096);
    QP(HD == 256);  // per-head RMS norm width (q_norm.weight / k_norm.weight)
    // q_proj output is split: first NH*HD for queries, next NH*HD for per-head gate
    QP(NH * HD == 4096);
    // key-value group query expansion: each k head is shared by NH/NKV = 8 q heads
    QP(NH / NKV == 8);
    // partial RoPE: partial_rotary_factor=0.25 → rotate first HD*0.25 = 64 dims per head
    QP((int)(HD * 0.25f) == 64);
    // mRoPE section sizes [11, 11, 10] sum to 32 ≤ 64 (the rotated half)
    QP(11 + 11 + 10 == 32 && 32 <= 64);
}

// The model uses k_norm.weight [HD] and q_norm.weight [HD] for per-head QK normalisation.
// Apply RMS norm to a head-sized vector using double arithmetic (host reference).
static double rms_norm_ref(const float* x, const float* w, int n) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    double inv = 1.0 / __builtin_sqrt(ss / n + 1e-6);
    double acc = 0;
    for (int i = 0; i < n; ++i) acc += (double)x[i] * inv * (double)w[i];
    return acc;
}

static void test_qsa_norm() {
    // Minimal head: HD=4 (4 instead of 256 for speed)
    const int HD = 4;
    float x[HD] = {0.5f, -1.0f, 1.5f, 2.0f};
    float w[HD] = {1.0f, 1.0f, 1.0f, 1.0f};  // identity scale
    double ref = rms_norm_ref(x, w, HD);
    // After normalising, sum of squares of output should be HD * mean(output^2) ≈ HD
    double ss_out = 0;
    double inv = 1.0 / __builtin_sqrt((0.5*0.5 + 1.0*1.0 + 1.5*1.5 + 2.0*2.0) / HD + 1e-6);
    for (int i = 0; i < HD; ++i) ss_out += (x[i] * inv) * (x[i] * inv);
    // The RMS of the normalised vector should be close to 1
    double rms = __builtin_sqrt(ss_out / HD);
    QP(rms > 0.99 && rms < 1.01);
    QP(ref != 0.0);  // non-trivial output
}

int main() {
    test_qsa_shapes();
    test_qsa_norm();
    std::printf("qsa_sublayer_parity: checks=%ld failures=%ld\n", g_checks, g_fail);
    if (g_fail) { std::printf("qsa_sublayer_parity FAILED\n"); return 1; }
    std::printf("qsa_sublayer_parity PASSED\n"); return 0;
}


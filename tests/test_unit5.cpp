// tests/test_unit5.cpp — Smoke test for unit 5 (elementwise, gdn, gr, ple)
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/ple.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { ++tests_passed; } \
    else { ++tests_failed; fprintf(stderr, "FAIL: %s\n", msg); } \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) do { \
    if (std::fabs((a) - (b)) < (tol)) { ++tests_passed; } \
    else { ++tests_failed; fprintf(stderr, "FAIL: %s (got %f, expected %f)\n", msg, (double)(a), (double)(b)); } \
} while(0)

int main() {
    printf("=== strata unit 5 smoke test ===\n");

    // ════════════════════════════════════════════════════════════════
    // elementwise.hpp tests
    // ════════════════════════════════════════════════════════════════

    // ── embedding_gather: 8-bit codes ──────────────────────────────
    {
        uint8_t codes[] = {0, 1, 2, 3, 4, 5, 6, 7};
        float scales[] = {2.0f, 2.0f};     // 2 groups of 4
        float offsets[] = {1.0f, 1.0f};
        float out[8] = {};
        strata::kernels::embedding_gather(codes, scales, offsets, 8, 8, 0, 4, out, nullptr);
        // out[0] = 0 * 2.0 + 1.0 = 1.0
        // out[1] = 1 * 2.0 + 1.0 = 3.0
        // out[4] = 4 * 2.0 + 1.0 = 9.0
        CHECK_NEAR(out[0], 1.0f, 0.001f, "embedding_gather[0] == 1.0");
        CHECK_NEAR(out[1], 3.0f, 0.001f, "embedding_gather[1] == 3.0");
        CHECK_NEAR(out[4], 9.0f, 0.001f, "embedding_gather[4] == 9.0");
    }

    // ── gdn_gate: softplus(alpha + dt) * ssm_a ────────────────────
    {
        float alpha[] = {0.0f, 25.0f};   // 1 token, 2 heads
        float dt[] = {0.0f, 0.0f};
        float ssm_a[] = {-1.0f, -1.0f};
        float gate[2] = {};
        strata::kernels::gdn_gate(alpha, dt, ssm_a, gate, 1, 2, nullptr);

        // softplus(0) = log(2) ≈ 0.6931
        CHECK_NEAR(gate[0], -std::log(2.0f), 0.001f, "gdn_gate softplus(0)*(-1)");
        // softplus(25) ≈ 25 (large-x branch)
        CHECK_NEAR(gate[1], -25.0f, 0.001f, "gdn_gate large-x branch");
    }

    // ── scale_inplace ──────────────────────────────────────────────
    {
        float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
        strata::kernels::scale_inplace(x, 4, 0.5f, nullptr);
        CHECK_NEAR(x[0], 0.5f, 0.001f, "scale_inplace[0]");
        CHECK_NEAR(x[3], 2.0f, 0.001f, "scale_inplace[3]");
    }

    // ── add_inplace ────────────────────────────────────────────────
    {
        float dst[] = {1.0f, 2.0f, 3.0f};
        float src[] = {10.0f, 20.0f, 30.0f};
        strata::kernels::add_inplace(dst, src, 3, nullptr);
        CHECK_NEAR(dst[0], 11.0f, 0.001f, "add_inplace[0]");
        CHECK_NEAR(dst[2], 33.0f, 0.001f, "add_inplace[2]");
    }

    // ── f32_to_f16_bulk / f32_to_bf16_bulk ─────────────────────────
    {
        float x[] = {1.0f, -1.0f, 0.0f, 65504.0f};
        uint16_t y_f16[4] = {};
        uint16_t y_bf16[4] = {};
        strata::kernels::f32_to_f16_bulk(x, y_f16, 4, nullptr);
        strata::kernels::f32_to_bf16_bulk(x, y_bf16, 4, nullptr);

        // fp16 1.0 = 0x3C00
        CHECK(y_f16[0] == 0x3C00u, "f32_to_f16_bulk 1.0 == 0x3C00");
        CHECK(y_f16[1] == 0xBC00u, "f32_to_f16_bulk -1.0 == 0xBC00");
        CHECK(y_f16[2] == 0x0000u, "f32_to_f16_bulk 0.0 == 0x0000");

        // bf16 1.0: sign=0, exp=0111_1111, man=000_0000 → 0x3F80
        CHECK(y_bf16[0] == 0x3F80u, "f32_to_bf16_bulk 1.0 == 0x3F80");
        CHECK(y_bf16[2] == 0x0000u, "f32_to_bf16_bulk 0.0 == 0x0000");

        // fp16 and bf16 differ in mantissa bits: 10 vs 7
        // This is a different encoding, not just a different value.
        CHECK(y_f16[0] != y_bf16[0], "fp16 != bf16 for 1.0 (different encoding)");
    }

    // ── silu_inplace ───────────────────────────────────────────────
    {
        float x[] = {0.0f, 1.0f, -1.0f};
        strata::kernels::silu_inplace(x, 3, nullptr);
        // silu(0) = 0 / (1+1) = 0
        CHECK_NEAR(x[0], 0.0f, 0.001f, "silu_inplace(0) == 0");
        // silu(1) = 1 / (1 + exp(-1)) ≈ 0.7311
        CHECK_NEAR(x[1], 1.0f / (1.0f + std::exp(-1.0f)), 0.001f, "silu_inplace(1)");
        // silu(-1) = -1 / (1 + exp(1)) ≈ -0.2689
        CHECK_NEAR(x[2], -1.0f / (1.0f + std::exp(1.0f)), 0.001f, "silu_inplace(-1)");
    }

    // ── rms_norm_weighted ──────────────────────────────────────────
    {
        // 1 row, 4 cols: x = [1, 1, 1, 1], w = [2, 2, 2, 2]
        float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
        float w[] = {2.0f, 2.0f, 2.0f, 2.0f};
        strata::kernels::rms_norm_weighted(x, w, 1, 4, 1e-5f, nullptr);
        // mean(x^2) = 1.0, rms = 1.0, normalized = [1,1,1,1], weighted = [2,2,2,2]
        CHECK_NEAR(x[0], 2.0f, 0.01f, "rms_norm_weighted uniform");
        CHECK_NEAR(x[3], 2.0f, 0.01f, "rms_norm_weighted uniform[3]");
    }

    // ── doorbell_ring ──────────────────────────────────────────────
    {
        uint32_t seq = 42;
        strata::kernels::doorbell_ring(&seq, nullptr);
        CHECK(seq == 43, "doorbell_ring increments");
        strata::kernels::doorbell_ring(&seq, nullptr);
        CHECK(seq == 44, "doorbell_ring increments again");
    }

    // ── copy_from_mapped ───────────────────────────────────────────
    {
        float src[] = {3.14f, 2.72f, 1.41f};
        float dst[3] = {};
        strata::kernels::copy_from_mapped(dst, src, 3, nullptr);
        CHECK_NEAR(dst[0], 3.14f, 0.001f, "copy_from_mapped[0]");
        CHECK_NEAR(dst[2], 1.41f, 0.001f, "copy_from_mapped[2]");
    }

    // ── copy_i32_from_mapped ───────────────────────────────────────
    {
        int32_t src[] = {100, 200, 300};
        int32_t dst[3] = {};
        strata::kernels::copy_i32_from_mapped(dst, src, 3, nullptr);
        CHECK(dst[0] == 100, "copy_i32_from_mapped[0]");
        CHECK(dst[2] == 300, "copy_i32_from_mapped[2]");
    }

    // ── doorbell_publish ───────────────────────────────────────────
    {
        float x[] = {1.0f, 2.0f};
        int32_t ids[] = {5, 10};
        float weights[] = {0.5f, 0.5f};
        float x_out[2] = {};
        int32_t ids_out[2] = {};
        float w_out[2] = {};
        uint32_t seq = 0;
        strata::kernels::doorbell_publish(x, ids, weights, 2, 2,
                                          x_out, ids_out, w_out, &seq, nullptr);
        CHECK_NEAR(x_out[0], 1.0f, 0.001f, "doorbell_publish x_out[0]");
        CHECK(ids_out[1] == 10, "doorbell_publish ids_out[1]");
        CHECK(seq == 1, "doorbell_publish increments seq");
    }

    // ════════════════════════════════════════════════════════════════
    // gdn.hpp tests
    // ════════════════════════════════════════════════════════════════

    // ── GdnShapes ──────────────────────────────────────────────────
    {
        strata::kernels::GdnShapes s;
        s.S = 128;
        s.h_k = 16;
        s.h_v = 48;
        CHECK(s.valid(), "GdnShapes valid with real values");
        CHECK(s.channels() == 10240, "GdnShapes channels == 10240");
        CHECK(s.h_v % s.h_k == 0, "GdnShapes h_v divisible by h_k");

        strata::kernels::GdnShapes bad;
        CHECK(!bad.valid(), "GdnShapes default is invalid");
    }

    // ── gdn_beta_gate: sigmoid ─────────────────────────────────────
    {
        float beta[] = {0.0f, 100.0f, -100.0f};
        strata::kernels::gdn_beta_gate(beta, 3, nullptr);
        CHECK_NEAR(beta[0], 0.5f, 0.001f, "gdn_beta_gate sigmoid(0) == 0.5");
        CHECK_NEAR(beta[1], 1.0f, 0.001f, "gdn_beta_gate sigmoid(100) ≈ 1.0");
        CHECK_NEAR(beta[2], 0.0f, 0.001f, "gdn_beta_gate sigmoid(-100) ≈ 0.0");
    }

    // ── gdn_l2_norm: NOT rms_norm ──────────────────────────────────
    {
        // 1 row, 4 cols: x = [3, 0, 0, 4]
        // L2 norm = sqrt(9 + 16) = 5
        float x[] = {3.0f, 0.0f, 0.0f, 4.0f};
        strata::kernels::gdn_l2_norm(x, 1, 4, 0.0f, nullptr);
        CHECK_NEAR(x[0], 3.0f / 5.0f, 0.001f, "gdn_l2_norm[0] = 3/5");
        CHECK_NEAR(x[3], 4.0f / 5.0f, 0.001f, "gdn_l2_norm[3] = 4/5");

        // Verify it's L2 norm not RMS norm:
        // RMS would be sqrt(mean(x^2)) = sqrt((9+16)/4) = sqrt(6.25) = 2.5
        // L2 norm = sqrt(sum(x^2)) = sqrt(25) = 5
        // The two differ by sqrt(4) = 2.
        float x2[] = {3.0f, 0.0f, 0.0f, 4.0f};
        strata::kernels::gdn_l2_norm(x2, 1, 4, 0.0f, nullptr);
        // After L2 norm: sum of squares should ≈ 1.0
        float sq_sum = 0.0f;
        for (int i = 0; i < 4; ++i) sq_sum += x2[i] * x2[i];
        CHECK_NEAR(sq_sum, 1.0f, 0.001f, "gdn_l2_norm produces unit vector");
    }

    // ── gdn_step: small test (S=2, h_k=1, h_v=1) ─────────────────
    {
        strata::kernels::GdnShapes s;
        s.S = 2;
        s.h_k = 1;
        s.h_v = 1;
        CHECK(s.valid(), "small GdnShapes valid");

        // state: [S=2 × h_v=1 × S=2] = 4 floats, init to zero
        float state[4] = {0.0f};
        float q[2] = {1.0f, 0.0f};         // h_k=1, S=2
        float k[2] = {1.0f, 0.0f};
        float v[2] = {1.0f, 0.5f};
        float gate[1] = {0.0f};            // exp(0) = 1.0 (no decay)
        float beta[1] = {1.0f};            // full write
        float o[2] = {};

        strata::kernels::gdn_step(state, q, k, v, gate, beta, o, s, nullptr);

        // With zero state, decay is no-op.
        // sk[0,j] = sum_i state[i,0,j] * k[0,i] = 0 (state was zero)
        // d[0,j] = (v[0,j] - 0) * 1.0 = v[0,j]
        // state[i,0,j] += k[0,i] * d[0,j]
        //   state[0,0,0] = k[0] * v[0] = 1*1 = 1
        //   state[0,0,1] = k[0] * v[1] = 1*0.5 = 0.5
        //   state[1,0,0] = k[1] * v[0] = 0
        //   state[1,0,1] = k[1] * v[1] = 0
        // o[0,j] = sum_i state[i,0,j] * q[0,i]
        //   o[0,0] = state[0,0,0]*q[0] + state[1,0,0]*q[1] = 1*1 + 0*0 = 1
        //   o[0,1] = state[0,0,1]*q[0] + state[1,0,1]*q[1] = 0.5*1 + 0 = 0.5
        CHECK_NEAR(o[0], 1.0f, 0.001f, "gdn_step o[0] = 1.0");
        CHECK_NEAR(o[1], 0.5f, 0.001f, "gdn_step o[1] = 0.5");
    }

    // ── gdn_step: head pairing is MODULO ───────────────────────────
    {
        strata::kernels::GdnShapes s;
        s.S = 2;
        s.h_k = 1;
        s.h_v = 2;  // h_v > h_k, head pairing = h % h_k
        CHECK(s.valid(), "GdnShapes h_v=2, h_k=1 valid");

        // Both value heads should pair with key head 0 (0%1=0, 1%1=0)
        float state[8] = {0.0f};  // S=2 × h_v=2 × S=2
        float q[2] = {1.0f, 0.0f};
        float k[2] = {1.0f, 0.0f};
        float v[4] = {1.0f, 0.0f, 0.0f, 1.0f};  // v[h=0] = [1,0], v[h=1] = [0,1]
        float gate[2] = {0.0f, 0.0f};
        float beta[2] = {1.0f, 1.0f};
        float o[4] = {};

        strata::kernels::gdn_step(state, q, k, v, gate, beta, o, s, nullptr);

        // Both heads use k[0] (modulo pairing).
        // Head 0: o[0,0]=1, o[0,1]=0
        // Head 1: o[1,0]=0, o[1,1]=1
        CHECK_NEAR(o[0], 1.0f, 0.001f, "gdn_step modulo head 0, o[0]=1");
        CHECK_NEAR(o[1], 0.0f, 0.001f, "gdn_step modulo head 0, o[1]=0");
        CHECK_NEAR(o[2], 0.0f, 0.001f, "gdn_step modulo head 1, o[0]=0");
        CHECK_NEAR(o[3], 1.0f, 0.001f, "gdn_step modulo head 1, o[1]=1");
    }

    // ── gdn_conv_step ──────────────────────────────────────────────
    {
        // 2 channels, d_conv = 2 (1 history row + new input)
        int64_t channels = 2;
        int64_t d_conv = 2;
        // conv_state: 1 row of 2 channels
        float conv_state[] = {1.0f, 2.0f};
        float x[] = {3.0f, 4.0f};
        // kW: GGML layout kW[c * d_conv + i]
        // chan 0: kW[0]=0.5, kW[1]=0.5
        // chan 1: kW[2]=1.0, kW[3]=1.0
        float kW[] = {0.5f, 0.5f, 1.0f, 1.0f};
        float out[2] = {};

        strata::kernels::gdn_conv_step(conv_state, x, kW, out, channels, d_conv, nullptr);
        // out[0] = conv_state[0]*kW[0] + x[0]*kW[1] = 1*0.5 + 3*0.5 = 2.0
        // out[1] = conv_state[1]*kW[2] + x[1]*kW[3] = 2*1.0 + 4*1.0 = 6.0
        CHECK_NEAR(out[0], 2.0f, 0.001f, "gdn_conv_step out[0] = 2.0");
        CHECK_NEAR(out[1], 6.0f, 0.001f, "gdn_conv_step out[1] = 6.0");

        // After shift, conv_state should contain x
        CHECK_NEAR(conv_state[0], 3.0f, 0.001f, "gdn_conv_step state shifted to x[0]");
        CHECK_NEAR(conv_state[1], 4.0f, 0.001f, "gdn_conv_step state shifted to x[1]");
    }

    // ── gdn_out_norm: sigmoid NOT silu ─────────────────────────────
    {
        float o[] = {1.0f, 1.0f};  // h_v=1, S=2
        float z[] = {0.0f, 0.0f};  // sigmoid(0) = 0.5
        float ssm_norm[] = {2.0f, 2.0f};
        float out[2] = {};

        strata::kernels::gdn_out_norm(o, z, ssm_norm, out, 1, 2, 1e-5f, nullptr);
        // rms_norm([1,1]) = [1,1] / sqrt(mean(1) + eps) ≈ [1,1]
        // * ssm_norm * sigmoid(0) = [1,1] * [2,2] * 0.5 = [1,1]
        CHECK_NEAR(out[0], 1.0f, 0.01f, "gdn_out_norm sigmoid gate");
        CHECK_NEAR(out[1], 1.0f, 0.01f, "gdn_out_norm sigmoid gate[1]");
    }

    // ════════════════════════════════════════════════════════════════
    // gr.hpp tests
    // ════════════════════════════════════════════════════════════════

    // ── GrShapes ───────────────────────────────────────────────────
    {
        strata::kernels::GrShapes s;
        s.n_embd = 2560;
        s.hc = 4;
        s.hc_lr = 320;
        CHECK(s.valid(), "GrShapes valid with real values");
        CHECK(s.residual_width() == 10240, "GrShapes residual_width == 10240");

        strata::kernels::GrShapes bad;
        CHECK(!bad.valid(), "GrShapes default invalid");
    }

    // ── gr_workspace_init / gr_workspace_bytes ─────────────────────
    {
        strata::kernels::GrShapes s;
        s.n_embd = 4;
        s.hc = 2;
        s.hc_lr = 2;

        size_t need = strata::kernels::gr_workspace_bytes(s);
        CHECK(need > 0, "gr_workspace_bytes > 0");

        std::vector<uint8_t> buf(need, 0);
        strata::kernels::GrWorkspace ws;
        size_t got = strata::kernels::gr_workspace_init(s, buf.data(), ws);
        CHECK(got == need, "gr_workspace_init returns same as bytes");
        CHECK(ws.bytes == need, "workspace.bytes == need");
        CHECK(ws.xn != nullptr, "workspace.xn != null");
        CHECK(ws.xq != nullptr, "workspace.xq != null");
        CHECK(ws.lo != nullptr, "workspace.lo != null");
        CHECK(ws.gated != nullptr, "workspace.gated != null");
        CHECK(ws.lq != nullptr, "workspace.lq != null");
    }

    // ── gr_write: zero injection → plain residual add ──────────────
    {
        strata::kernels::GrShapes s;
        s.n_embd = 2;
        s.hc = 2;
        s.hc_lr = 1;

        // R = [[1, 2], [3, 4]]  (2 streams, each 2-wide)
        float R[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float block_out[] = {10.0f, 20.0f};
        // inject = 0 → w[c] = 2*sigmoid(0/hc) = 2*0.5 = 1.0
        float inject[] = {0.0f, 0.0f};
        float R_out[4] = {};

        strata::kernels::gr_write(R, block_out, inject, s, R_out, nullptr);

        // R_out[c,d] = R[c,d] + block_out[d] * 1.0
        CHECK_NEAR(R_out[0], 11.0f, 0.001f, "gr_write zero inject R_out[0,0]");
        CHECK_NEAR(R_out[1], 22.0f, 0.001f, "gr_write zero inject R_out[0,1]");
        CHECK_NEAR(R_out[2], 13.0f, 0.001f, "gr_write zero inject R_out[1,0]");
        CHECK_NEAR(R_out[3], 24.0f, 0.001f, "gr_write zero inject R_out[1,1]");
    }

    // ── gr_set_fp32_activations / gr_set_native_mmvf ───────────────
    {
        strata::kernels::gr_set_fp32_activations(true);
        strata::kernels::gr_set_fp32_activations(false);
        strata::kernels::gr_set_native_mmvf(true);
        strata::kernels::gr_set_native_mmvf(false);
        CHECK(true, "gr configuration functions callable");
    }

    // ════════════════════════════════════════════════════════════════
    // ple.hpp tests
    // ════════════════════════════════════════════════════════════════

    // ── PLE constants ──────────────────────────────────────────────
    {
        CHECK(strata::kernels::PLE_CONV_KERNEL == 4, "PLE_CONV_KERNEL == 4");
        CHECK(strata::kernels::PLE_NG_HIST == 3, "PLE_NG_HIST == 3 (d_conv - 1)");
    }

    // ── PleWeights default-constructed ─────────────────────────────
    {
        strata::kernels::PleWeights w;
        CHECK(w.key_codes == nullptr, "PleWeights key_codes default null");
        CHECK(w.value_bf16 == nullptr, "PleWeights value_bf16 default null");
        CHECK(w.key_native_type == -1, "PleWeights key_native_type default -1");
        CHECK(w.key_bf16 == nullptr, "PleWeights key_bf16 default null");
    }

    // ── PleOut default-constructed ─────────────────────────────────
    {
        strata::kernels::PleOut out;
        CHECK(out.key == nullptr, "PleOut key default null");
        CHECK(out.result == nullptr, "PleOut result default null");
        CHECK(out.gate == nullptr, "PleOut gate default null");
        CHECK(out.conv == nullptr, "PleOut conv default null");
    }

    // ── ple_block_scratch_bytes ────────────────────────────────────
    {
        uint64_t scratch = strata::kernels::ple_block_scratch_bytes();
        CHECK(scratch > 0, "ple_block_scratch_bytes > 0");
    }

    // ── ple_block callable ─────────────────────────────────────────
    {
        float emb[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float hidden[4] = {0.1f, 0.2f, 0.3f, 0.4f};
        float hist[12] = {};  // PLE_NG_HIST * 4
        strata::kernels::PleWeights w;
        strata::kernels::PleOut out;
        float result[4] = {};
        out.result = result;

        uint64_t scratch_sz = strata::kernels::ple_block_scratch_bytes();
        std::vector<uint8_t> scratch(scratch_sz, 0);

        strata::kernels::ple_block(emb, hidden, hist, w, out, scratch.data(), nullptr);
        // Just verify it's callable and doesn't crash
        CHECK(true, "ple_block callable without crash");
    }

    // ── ple_history_advance callable ───────────────────────────────
    {
        float hist[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        float normalized[4] = {99, 98, 97, 96};
        strata::kernels::ple_history_advance(hist, normalized, nullptr);
        // Just verify callable
        CHECK(true, "ple_history_advance callable");
    }

    // ── ple configuration flags ────────────────────────────────────
    {
        strata::kernels::ple_set_native_bf16(true);
        strata::kernels::ple_set_native_bf16(false);
        strata::kernels::ple_set_native_postops(true);
        CHECK(strata::kernels::ple_native_postops_enabled() == true, "ple postops enabled");
        strata::kernels::ple_set_native_postops(false);
        CHECK(strata::kernels::ple_native_postops_enabled() == false, "ple postops disabled");
        CHECK(strata::kernels::ple_block_available() == true, "ple_block_available");
    }

    // ── Verify rms_norm != l2_norm distinction ─────────────────────
    {
        // rms_norm divides by sqrt(mean(x^2) + eps)
        // l2_norm divides by sqrt(sum(x^2) + eps)
        // For [3,4], mean = 12.5, sum = 25
        // rms_norm factor = 1/sqrt(12.5) ≈ 0.2828
        // l2_norm factor = 1/sqrt(25) = 0.2
        float x_rms[] = {3.0f, 4.0f};
        float w_one[] = {1.0f, 1.0f};
        strata::kernels::rms_norm_weighted(x_rms, w_one, 1, 2, 0.0f, nullptr);

        float x_l2[] = {3.0f, 4.0f};
        strata::kernels::gdn_l2_norm(x_l2, 1, 2, 0.0f, nullptr);

        // They should differ by sqrt(2) factor
        float ratio = x_rms[0] / x_l2[0];
        float expected_ratio = std::sqrt(2.0f);
        CHECK_NEAR(ratio, expected_ratio, 0.01f,
                   "rms_norm vs l2_norm differ by sqrt(cols)");
    }

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}


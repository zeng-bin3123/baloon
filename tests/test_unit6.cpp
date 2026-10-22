// tests/test_unit6.cpp — Smoke test for unit 6 (qsa, s_gemv, shared_expert, ngram)
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/s_gemv.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

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
    printf("=== strata unit 6 smoke test ===\n");

    // ════════════════════════════════════════════════════════════════
    // qsa.hpp tests
    // ════════════════════════════════════════════════════════════════

    // ── QsaShapes ──────────────────────────────────────────────────
    {
        strata::kernels::QsaShapes s;
        CHECK(!s.valid(), "QsaShapes default invalid");

        s.n_head = 24;
        s.n_head_kv = 2;
        s.head_dim = 256;
        s.n_rot = 64;
        s.idx_n_head = 4;
        s.idx_dim = 128;
        s.idx_block = 4;
        s.idx_top_k = 2048;
        s.page_size = 512;
        CHECK(s.valid(), "QsaShapes valid with real values");

        // head_dim must be multiple of 4
        CHECK(s.head_dim % 4 == 0, "QsaShapes head_dim multiple of 4");

        // n_head must be divisible by n_head_kv
        CHECK(s.n_head % s.n_head_kv == 0, "QsaShapes n_head divisible by n_head_kv");

        // KV head map: division, NOT modulo
        // q=0..11 → kv_head 0, q=12..23 → kv_head 1
        CHECK(s.kv_head(0) == 0, "QsaShapes kv_head(0) = 0");
        CHECK(s.kv_head(11) == 0, "QsaShapes kv_head(11) = 0");
        CHECK(s.kv_head(12) == 1, "QsaShapes kv_head(12) = 1");
        CHECK(s.kv_head(23) == 1, "QsaShapes kv_head(23) = 1");

        // Invalid: head_dim not multiple of 4
        strata::kernels::QsaShapes bad = s;
        bad.head_dim = 255;
        CHECK(!bad.valid(), "QsaShapes invalid head_dim not %4");

        // Invalid: n_head not divisible by n_head_kv
        bad = s;
        bad.n_head = 25;
        CHECK(!bad.valid(), "QsaShapes invalid n_head not div n_head_kv");
    }

    // ── qsa_selection_width ────────────────────────────────────────
    {
        strata::kernels::QsaShapes s;
        s.n_head = 24; s.n_head_kv = 2; s.head_dim = 256;
        s.n_rot = 64; s.idx_n_head = 4; s.idx_dim = 128;
        s.idx_block = 4; s.idx_top_k = 2048; s.page_size = 512;

        // width = min(n_kv, 2048 + 4 - 1) = min(n_kv, 2051)
        CHECK(strata::kernels::qsa_selection_width(100, s) == 100,
              "selection_width(100) = 100 (identity)");
        CHECK(strata::kernels::qsa_selection_width(2051, s) == 2051,
              "selection_width(2051) = 2051 (boundary)");
        CHECK(strata::kernels::qsa_selection_width(3000, s) == 2051,
              "selection_width(3000) = 2051 (clamped)");
        CHECK(strata::kernels::qsa_selection_width(1, s) == 1,
              "selection_width(1) = 1");
    }

    // ── QsaStep enum and qsa_step_bytes ────────────────────────────
    {
        CHECK(strata::kernels::kStepPos == 0, "kStepPos == 0");
        CHECK(strata::kernels::kStepNKv == 1, "kStepNKv == 1");
        CHECK(strata::kernels::kStepNBid == 2, "kStepNBid == 2");
        CHECK(strata::kernels::kStepWidth == 3, "kStepWidth == 3");
        CHECK(strata::kernels::kStepCount == 4, "kStepCount == 4");
        CHECK(strata::kernels::qsa_step_bytes() == 16, "qsa_step_bytes == 16");
    }

    // ── qsa_step_fill ──────────────────────────────────────────────
    {
        strata::kernels::QsaShapes s;
        s.n_head = 24; s.n_head_kv = 2; s.head_dim = 256;
        s.n_rot = 64; s.idx_n_head = 4; s.idx_dim = 128;
        s.idx_block = 4; s.idx_top_k = 2048; s.page_size = 512;

        int32_t step[4] = {};
        strata::kernels::qsa_step_fill(step, 7, s);

        CHECK(step[0] == 7, "qsa_step_fill pos=7");
        CHECK(step[1] == 8, "qsa_step_fill n_kv=8");
        CHECK(step[2] == 2, "qsa_step_fill n_bid=8/4=2");
        CHECK(step[3] == 8, "qsa_step_fill width=min(8,2051)=8");

        // At pos=2050
        strata::kernels::qsa_step_fill(step, 2050, s);
        CHECK(step[0] == 2050, "qsa_step_fill pos=2050");
        CHECK(step[1] == 2051, "qsa_step_fill n_kv=2051");
        CHECK(step[3] == 2051, "qsa_step_fill width=2051 (boundary)");

        // At pos=3000 (beyond threshold)
        strata::kernels::qsa_step_fill(step, 3000, s);
        CHECK(step[3] == 2051, "qsa_step_fill width clamped at 2051");
    }

    // ── kv_append + kv_gather basic ────────────────────────────────
    {
        // Minimal test: 1 kv head, head_dim=4, page_size=4
        strata::kernels::QsaShapes s;
        s.n_head = 2; s.n_head_kv = 1; s.head_dim = 4;
        s.n_rot = 2; s.idx_n_head = 1; s.idx_dim = 4;
        s.idx_block = 2; s.idx_top_k = 4; s.page_size = 4;

        // 1 page, 1 kv head, 4 cells per page, head_dim=4
        // Pool: page*n_head_kv*page_size * head_dim = 1*1*4*4 = 16 fp16 elements
        std::vector<uint16_t> k_pool(16, 0);
        std::vector<uint16_t> v_pool(16, 0);
        int32_t page_table[] = {0};  // page 0

        // Append at pos=0
        float kcur[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float vcur[] = {5.0f, 6.0f, 7.0f, 8.0f};
        strata::kernels::kv_append(k_pool.data(), v_pool.data(),
                                   page_table, 0, kcur, vcur, s, nullptr);

        // Check K was written: phys_row = (0*1+0)*4 + 0 = 0
        float k0 = strata::kernels::f32_from_f16(k_pool[0 * 4 + 0]);
        CHECK_NEAR(k0, 1.0f, 0.01f, "kv_append K[0] = 1.0");
        float v0 = strata::kernels::f32_from_f16(v_pool[0 * 4 + 0]);
        CHECK_NEAR(v0, 5.0f, 0.01f, "kv_append V[0] = 5.0");

        // Append at pos=1
        float kcur2[] = {10.0f, 20.0f, 30.0f, 40.0f};
        float vcur2[] = {50.0f, 60.0f, 70.0f, 80.0f};
        strata::kernels::kv_append(k_pool.data(), v_pool.data(),
                                   page_table, 1, kcur2, vcur2, s, nullptr);

        // Gather cells 0 and 1
        std::vector<uint16_t> k_scratch(2 * 1 * 4, 0);
        std::vector<uint16_t> v_scratch(2 * 1 * 4, 0);
        int32_t ids[] = {0, 1};
        strata::kernels::kv_gather(k_pool.data(), v_pool.data(),
                                   page_table, ids, 2, s,
                                   k_scratch.data(), v_scratch.data(), nullptr);

        // Check gathered K cell 0
        float gk0 = strata::kernels::f32_from_f16(k_scratch[0 * 4 + 0]);
        CHECK_NEAR(gk0, 1.0f, 0.01f, "kv_gather K[0] = 1.0");

        // Check gathered K cell 1
        float gk1 = strata::kernels::f32_from_f16(k_scratch[1 * 1 * 4 + 0]);
        CHECK_NEAR(gk1, 10.0f, 0.01f, "kv_gather K[1] = 10.0");
    }

    // ── qsa_index: ReLU per head ───────────────────────────────────
    {
        strata::kernels::QsaShapes s;
        s.n_head = 2; s.n_head_kv = 1; s.head_dim = 4;
        s.n_rot = 2; s.idx_n_head = 2; s.idx_dim = 2;
        s.idx_block = 2; s.idx_top_k = 4; s.page_size = 4;

        // 2 cells, 1 block completed (n_bid=1 since n_kv=2, r=2)
        // pooled: 2 rows of idx_dim=2
        float pooled[] = {
            1.0f, 0.0f,   // block 0
            0.0f, 1.0f,   // spare at n_bid=1
        };

        // q_idx: 2 heads of idx_dim=2
        float q_idx[] = {
            3.0f, -5.0f,    // head 0: dot with block 0 = 3.0
            -2.0f, 1.0f,    // head 1: dot with block 0 = -2.0
        };

        // CRITICAL: ReLU per head: relu(3) + relu(-2) = 3 + 0 = 3
        // NOT relu(3 + (-2)) = relu(1) = 1

        float cell_scores[2] = {};
        strata::kernels::qsa_index(pooled, 1, q_idx, nullptr, s, 2,
                                   cell_scores, nullptr);

        // Cell 0: cb=0, score = 0 + relu(3.0) + relu(-2.0) = 3.0
        // Cell 1: cb=0 (1/2=0), score = 0 + relu(3.0) + relu(-2.0) = 3.0
        // But cell 1 is in tail (>= (2/2)*2 = 2? No, 1 < 2, not tail)
        // Actually cells 0,1 both map to block 0 (cb = j/r = j/2)
        // cell 0: cb=0/2=0, cell 1: cb=1/2=0
        CHECK_NEAR(cell_scores[0], 3.0f, 0.01f,
                   "qsa_index relu_per_head: relu(3)+relu(-2) = 3");
    }

    // ── topk_512 ───────────────────────────────────────────────────
    {
        strata::kernels::QsaShapes s;
        s.n_head = 2; s.n_head_kv = 1; s.head_dim = 4;
        s.n_rot = 2; s.idx_n_head = 1; s.idx_dim = 2;
        s.idx_block = 2; s.idx_top_k = 2; s.page_size = 4;

        // 4 cells, width = min(4, 2+2-1) = 3
        float scores[] = {1.0f, 3.0f, 2.0f, 3.0f};
        int32_t ids[4] = {};
        strata::kernels::topk_512(scores, 4, s, 4, ids, nullptr);

        // Top 3 by score: indices 1(3.0), 3(3.0), 2(2.0)
        // Ties broken by ascending index: 1 before 3
        // Output sorted ascending: {1, 2, 3}
        CHECK(ids[0] == 1, "topk_512 ids[0] = 1");
        CHECK(ids[1] == 2, "topk_512 ids[1] = 2");
        CHECK(ids[2] == 3, "topk_512 ids[2] = 3");
    }

    // ── topk_512 constant ──────────────────────────────────────────
    {
        CHECK(strata::kernels::kTopkMaxCells == 32768, "kTopkMaxCells == 32768");
    }

    // ── qsa_attend: empty ids → zero ──────────────────────────────
    {
        strata::kernels::QsaShapes s;
        s.n_head = 2; s.n_head_kv = 1; s.head_dim = 4;
        s.n_rot = 2; s.idx_n_head = 1; s.idx_dim = 2;
        s.idx_block = 2; s.idx_top_k = 2; s.page_size = 4;

        float q[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float attn[8] = {99.0f, 99.0f, 99.0f, 99.0f, 99.0f, 99.0f, 99.0f, 99.0f};

        strata::kernels::qsa_attend(q, nullptr, nullptr, 0, s, attn, nullptr, nullptr);

        CHECK_NEAR(attn[0], 0.0f, 0.001f, "qsa_attend empty → attn[0]=0");
        CHECK_NEAR(attn[7], 0.0f, 0.001f, "qsa_attend empty → attn[7]=0");
    }

    // ── qsa_gate_apply: SIGMOID, not SiLU ─────────────────────────
    {
        strata::kernels::QsaShapes s;
        s.n_head = 1; s.n_head_kv = 1; s.head_dim = 4;
        s.n_rot = 2; s.idx_n_head = 1; s.idx_dim = 4;
        s.idx_block = 2; s.idx_top_k = 2; s.page_size = 4;

        // attn = [1.0, 2.0, 3.0, 4.0] (head_dim=4)
        float attn[] = {1.0f, 2.0f, 3.0f, 4.0f};
        // q_full: [2*head_dim] = [8], gate is SECOND half (indices 4..7)
        // q_full[0..3] = query, q_full[4..7] = gate
        float q_full[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        uint16_t out[4] = {};
        strata::kernels::qsa_gate_apply(attn, q_full, s, out, nullptr);

        // sigmoid(0) = 0.5
        // out = attn * sigmoid(gate) = [1.0*0.5, 2.0*0.5, 3.0*0.5, 4.0*0.5]
        float out0 = strata::kernels::f32_from_f16(out[0]);
        float out1 = strata::kernels::f32_from_f16(out[1]);
        CHECK_NEAR(out0, 0.5f, 0.01f, "qsa_gate_apply sigmoid(0)*1 = 0.5");
        CHECK_NEAR(out1, 1.0f, 0.01f, "qsa_gate_apply sigmoid(0)*2 = 1.0");
    }

    // ── qsa_gate_apply_f32: same math, no fp16 rounding ──────────
    {
        strata::kernels::QsaShapes s;
        s.n_head = 1; s.n_head_kv = 1; s.head_dim = 4;
        s.n_rot = 2; s.idx_n_head = 1; s.idx_dim = 4;
        s.idx_block = 2; s.idx_top_k = 2; s.page_size = 4;

        float attn[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float q_full[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float out[4] = {};

        strata::kernels::qsa_gate_apply_f32(attn, q_full, s, out, nullptr);

        CHECK_NEAR(out[0], 0.5f, 0.0001f, "qsa_gate_apply_f32 sigmoid(0)*1 = 0.5");
        CHECK_NEAR(out[1], 1.0f, 0.0001f, "qsa_gate_apply_f32 sigmoid(0)*2 = 1.0");
    }

    // ── step variants callable ─────────────────────────────────────
    {
        strata::kernels::QsaShapes s;
        s.n_head = 2; s.n_head_kv = 1; s.head_dim = 4;
        s.n_rot = 2; s.idx_n_head = 1; s.idx_dim = 2;
        s.idx_block = 2; s.idx_top_k = 4; s.page_size = 4;

        int32_t step[4];
        strata::kernels::qsa_step_fill(step, 3, s);

        // qsa_index_step
        float pooled[4] = {1.0f, 0.0f, 0.0f, 1.0f};
        float q_idx[2] = {1.0f, 0.0f};
        float cell_scores[4] = {};
        strata::kernels::qsa_index_step(pooled, q_idx, nullptr, s, step, 4,
                                        cell_scores, nullptr);
        CHECK(std::isfinite(cell_scores[0]), "qsa_index_step callable");

        // topk_512_step
        int32_t ids[4] = {};
        strata::kernels::topk_512_step(cell_scores, s, 4, step, ids, nullptr);
        CHECK(true, "topk_512_step callable");

        // kv_append_step
        std::vector<uint16_t> k_pool(64, 0), v_pool(64, 0);
        int32_t pt[] = {0};
        float kc[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float vc[] = {5.0f, 6.0f, 7.0f, 8.0f};
        strata::kernels::kv_append_step(k_pool.data(), v_pool.data(),
                                        pt, step, kc, vc, s, nullptr);
        CHECK(true, "kv_append_step callable");

        // kv_gather_step
        std::vector<uint16_t> ks(32, 0), vs(32, 0);
        strata::kernels::kv_gather_step(k_pool.data(), v_pool.data(),
                                        pt, ids, step, 4, s,
                                        ks.data(), vs.data(), nullptr);
        CHECK(true, "kv_gather_step callable");

        // qsa_attend_step
        float q[8] = {};
        float attn[8] = {};
        strata::kernels::qsa_attend_step(q, ks.data(), vs.data(),
                                         step, 4, s, attn, nullptr, nullptr);
        CHECK(true, "qsa_attend_step callable");
    }

    // ════════════════════════════════════════════════════════════════
    // s_gemv.hpp tests
    // ════════════════════════════════════════════════════════════════

    // ── Codebook / SForm ───────────────────────────────────────────
    {
        strata::kernels::SForm f;
        CHECK(f.code_bits == 2, "SForm default code_bits = 2");
        CHECK(f.code_bias == -1, "SForm default code_bias = -1");
        CHECK(f.group_elems == 64, "SForm default group_elems = 64");
        CHECK(f.codebook == strata::kernels::Codebook::Affine, "SForm default codebook = Affine");
        CHECK(!f.has_offset, "SForm default has_offset = false");
        CHECK(f.act_kind == 0, "SForm default act_kind = 0");
        CHECK(f.valid(), "SForm default valid");

        strata::kernels::SForm bad;
        bad.code_bits = 3;
        CHECK(!bad.valid(), "SForm invalid code_bits=3");

        bad.code_bits = 4;
        bad.group_elems = 0;
        CHECK(!bad.valid(), "SForm invalid group_elems=0");
    }

    // ── IQ4NL codebook ─────────────────────────────────────────────
    {
        CHECK(strata::kernels::kIq4NlTable[0] == -127, "IQ4NL[0] = -127");
        CHECK(strata::kernels::kIq4NlTable[7] == -10, "IQ4NL[7] = -10");
        CHECK(strata::kernels::kIq4NlTable[8] == 1, "IQ4NL[8] = 1");
        CHECK(strata::kernels::kIq4NlTable[15] == 113, "IQ4NL[15] = 113");
        // No -8 in this version
        bool has_minus8 = false;
        for (int i = 0; i < 16; ++i) {
            if (strata::kernels::kIq4NlTable[i] == -8) has_minus8 = true;
        }
        CHECK(!has_minus8, "IQ4NL codebook has no -8");
    }

    // ── s_gemv: simple 2-bit affine ────────────────────────────────
    {
        // 4 inputs, 2 outputs, 2-bit codes, group_elems=4
        strata::kernels::SForm form;
        form.code_bits = 2;
        form.code_bias = 0;
        form.group_elems = 4;
        form.codebook = strata::kernels::Codebook::Affine;
        form.has_offset = false;

        // x as fp16: [1.0, 1.0, 1.0, 1.0]
        uint16_t x[4];
        for (int i = 0; i < 4; ++i) x[i] = strata::kernels::f16_from_f32(1.0f);

        // codes: 2-bit per element, 4 elements per byte
        // Row 0: codes = [0,1,2,3] → 0b11_10_01_00 = 0xE4
        // Row 1: codes = [1,1,1,1] → 0b01_01_01_01 = 0x55
        uint8_t codes[] = {0xE4, 0x55};

        // scales: 1 group per row (4 elems / group_elems=4)
        float scales[] = {2.0f, 3.0f};

        float y[2] = {};
        strata::kernels::s_gemv(x, codes, scales, nullptr, y, 4, 2, form);

        // Row 0: sum_i x[i] * (code_i + 0) * scale = 1*(0+1+2+3)*2 = 12
        CHECK_NEAR(y[0], 12.0f, 0.01f, "s_gemv row0 = (0+1+2+3)*2 = 12");
        // Row 1: sum_i x[i] * (1) * 3 = 4*3 = 12
        CHECK_NEAR(y[1], 12.0f, 0.01f, "s_gemv row1 = 4*1*3 = 12");
    }

    // ── s_gemv_split: same result different order ──────────────────
    {
        strata::kernels::SForm form;
        form.code_bits = 2;
        form.code_bias = 0;
        form.group_elems = 4;
        form.codebook = strata::kernels::Codebook::Affine;

        uint16_t x[4];
        for (int i = 0; i < 4; ++i) x[i] = strata::kernels::f16_from_f32(1.0f);

        uint8_t codes[] = {0xE4, 0x55};
        float scales[] = {2.0f, 3.0f};
        float y[2] = {};

        strata::kernels::s_gemv_split(x, codes, scales, nullptr, y, 4, 2, form, 2);

        CHECK_NEAR(y[0], 12.0f, 0.01f, "s_gemv_split row0 = 12");
        CHECK_NEAR(y[1], 12.0f, 0.01f, "s_gemv_split row1 = 12");
    }

    // ── s2_gemv_quads: 2-bit specific path ─────────────────────────
    {
        uint16_t x[4];
        for (int i = 0; i < 4; ++i) x[i] = strata::kernels::f16_from_f32(1.0f);

        // S2 default: code_bias=-1, affine
        // codes = [0,1,2,3] → decoded as (-1,0,1,2)
        uint8_t codes[] = {0xE4};
        float scales[] = {1.0f};
        float y[1] = {};

        strata::kernels::s2_gemv_quads(x, codes, scales, y, 4, 1, 1);

        // sum = (-1+0+1+2)*1.0 = 2.0
        CHECK_NEAR(y[0], 2.0f, 0.01f, "s2_gemv_quads = (-1+0+1+2)*1 = 2");
    }

    // ── s2_gemv_fast: same as quads ────────────────────────────────
    {
        uint16_t x[4];
        for (int i = 0; i < 4; ++i) x[i] = strata::kernels::f16_from_f32(1.0f);

        uint8_t codes[] = {0xE4};
        float scales[] = {1.0f};
        float y[1] = {};

        strata::kernels::s2_gemv_fast(x, codes, scales, y, 4, 1, 1, true);
        CHECK_NEAR(y[0], 2.0f, 0.01f, "s2_gemv_fast = 2 (stage_x=true)");

        strata::kernels::s2_gemv_fast(x, codes, scales, y, 4, 1, 1, false);
        CHECK_NEAR(y[0], 2.0f, 0.01f, "s2_gemv_fast = 2 (stage_x=false)");
    }

    // ── s_gemv_split_async: callable ───────────────────────────────
    {
        strata::kernels::SForm form;
        form.code_bits = 2;
        form.code_bias = 0;
        form.group_elems = 4;

        uint16_t x[4];
        for (int i = 0; i < 4; ++i) x[i] = strata::kernels::f16_from_f32(1.0f);
        uint8_t codes[] = {0xE4};
        float scales[] = {2.0f};
        float y[1] = {};

        strata::kernels::s_gemv_split_async(x, codes, scales, nullptr, y,
                                            4, 1, form, 1, nullptr);
        CHECK_NEAR(y[0], 12.0f, 0.01f, "s_gemv_split_async row0 = 12");
    }

    // ── s_gemv_q8k: Q8_K activation ───────────────────────────────
    {
        // Q8_K block: 292 bytes per 256 elements
        // We'll use a small n_in=4, form with code_bits=8, group_elems=4
        // Q8_K block still packs 256 elements, but we only use first 4
        strata::kernels::SForm form;
        form.code_bits = 8;
        form.code_bias = 0;
        form.group_elems = 4;
        form.codebook = strata::kernels::Codebook::Affine;

        // Build Q8_K activation block for n_in=256
        std::vector<uint8_t> x_q8k(292, 0);
        float d = 1.0f;
        std::memcpy(x_q8k.data(), &d, 4);
        // qs[0..3] = {1, 2, 3, 4}, rest 0
        int8_t* qs = reinterpret_cast<int8_t*>(x_q8k.data() + 4);
        qs[0] = 1; qs[1] = 2; qs[2] = 3; qs[3] = 4;

        // Weight: 1 output, 256 inputs, 8-bit codes
        // codes: identity [0,1,2,...,255]
        std::vector<uint8_t> codes(256, 0);
        codes[0] = 1; codes[1] = 1; codes[2] = 1; codes[3] = 1;
        float scales[] = {1.0f};

        float y[1] = {};
        strata::kernels::s_gemv_q8k(x_q8k.data(), codes.data(), scales,
                                    nullptr, y, 256, 1, form, nullptr);

        // x_dequant = d * qs = [1,2,3,4,0,...], w = [1,1,1,1,0,...]*1
        // dot = 1*1 + 2*1 + 3*1 + 4*1 = 10
        CHECK_NEAR(y[0], 10.0f, 0.01f, "s_gemv_q8k dot product = 10");
    }

    // ── Q8_K block size ────────────────────────────────────────────
    {
        // 4 + 256 + 16*2 = 292
        CHECK(4 + 256 + 16 * 2 == 292, "Q8_K block = 292 bytes");
    }

    // ════════════════════════════════════════════════════════════════
    // shared_expert.hpp tests
    // ════════════════════════════════════════════════════════════════

    // ── shared_expert_set_native_bf16 ──────────────────────────────
    {
        strata::kernels::shared_expert_set_native_bf16(true);
        strata::kernels::shared_expert_set_native_bf16(false);
        CHECK(true, "shared_expert_set_native_bf16 callable");
    }

    // ── NativeSharedWeights default ────────────────────────────────
    {
        strata::kernels::NativeSharedWeights nw;
        CHECK(nw.gate_type == -1, "NativeSharedWeights gate_type default -1");
        CHECK(nw.up_type == -1, "NativeSharedWeights up_type default -1");
        CHECK(nw.down_type == -1, "NativeSharedWeights down_type default -1");
        CHECK(nw.gate_data == nullptr, "NativeSharedWeights gate_data null");
        CHECK(nw.q8_1 == nullptr, "NativeSharedWeights q8_1 null");
    }

    // ── moe_combine: shared ADDED PLAIN ────────────────────────────
    {
        // k=2 experts, n_embd=3
        // parts: [expert0: 1,2,3; expert1: 4,5,6]
        float parts[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        // weights already normalised (sum=1): [0.3, 0.7]
        float weights[] = {0.3f, 0.7f};
        // shared: [10, 20, 30]
        float shared[] = {10.0f, 20.0f, 30.0f};
        float y[3] = {};

        strata::kernels::moe_combine(parts, weights, shared, y, 3, 2, nullptr);

        // y[0] = 0.3*1 + 0.7*4 + 10 = 0.3 + 2.8 + 10 = 13.1
        CHECK_NEAR(y[0], 13.1f, 0.01f, "moe_combine y[0] = 13.1");
        // y[1] = 0.3*2 + 0.7*5 + 20 = 0.6 + 3.5 + 20 = 24.1
        CHECK_NEAR(y[1], 24.1f, 0.01f, "moe_combine y[1] = 24.1");
        // y[2] = 0.3*3 + 0.7*6 + 30 = 0.9 + 4.2 + 30 = 35.1
        CHECK_NEAR(y[2], 35.1f, 0.01f, "moe_combine y[2] = 35.1");
    }

    // ── moe_combine: null shared ───────────────────────────────────
    {
        float parts[] = {1.0f, 2.0f};
        float weights[] = {1.0f};
        float y[2] = {};

        strata::kernels::moe_combine(parts, weights, nullptr, y, 2, 1, nullptr);

        CHECK_NEAR(y[0], 1.0f, 0.001f, "moe_combine null shared y[0]=1");
        CHECK_NEAR(y[1], 2.0f, 0.001f, "moe_combine null shared y[1]=2");
    }

    // ── moe_combine: weights NOT renormalised again ────────────────
    {
        // Weights sum to 1 already; moe_combine must NOT divide again
        float parts[] = {10.0f, 20.0f};
        float weights[] = {0.5f, 0.5f};
        float y[1] = {};

        strata::kernels::moe_combine(parts, weights, nullptr, y, 1, 2, nullptr);

        // y = 0.5*10 + 0.5*20 = 15 (not 7.5)
        CHECK_NEAR(y[0], 15.0f, 0.001f, "moe_combine no re-normalise");
    }

    // ── shared_expert_n_ff not Q8_K ────────────────────────────────
    {
        // n_ff=640, 640 % 256 != 0 → not Q8_K compatible
        CHECK(640 % 256 != 0, "n_ff=640 not multiple of 256 (no Q8_K for down)");
    }

    // ── shared_expert_multi max tokens ─────────────────────────────
    {
        // n_tok <= 8
        float x[16] = {};
        float out[16] = {};
        strata::kernels::NativeSharedWeights nw;
        strata::kernels::shared_expert_multi(8, x, nullptr, nw, nullptr,
                                             nullptr, nullptr, nullptr,
                                             out, 2, 4, nullptr);
        CHECK(true, "shared_expert_multi n_tok=8 callable");
    }

    // ════════════════════════════════════════════════════════════════
    // ngram.hpp tests
    // ════════════════════════════════════════════════════════════════

    // ── Constants ──────────────────────────────────────────────────
    {
        CHECK(strata::kernels::NG_N_EMBD == 2560, "NG_N_EMBD == 2560");
        CHECK(strata::kernels::PLE_HEAD_DIM == 160, "PLE_HEAD_DIM == 160");
        CHECK(strata::kernels::NGRAM_SIZE == 3, "NGRAM_SIZE == 3");
        CHECK(strata::kernels::HEADS_PER_NGRAM == 8, "HEADS_PER_NGRAM == 8");
        CHECK(strata::kernels::PLE_N_HEADS == 16, "PLE_N_HEADS == (3-1)*8 = 16");
        CHECK(strata::kernels::PLE_CONV_KERNEL_NG == 4, "PLE_CONV_KERNEL_NG == 4");
        CHECK(strata::kernels::PLE_EOS_TOKEN_ID == 248044, "PLE_EOS_TOKEN_ID == 248044");
        CHECK(strata::kernels::NG_HC == 4, "NG_HC == 4");
        CHECK(strata::kernels::NG_HC_DIM == 10240, "NG_HC_DIM == 4*2560 = 10240");
        CHECK(strata::kernels::NG_HIST == 9, "NG_HIST == (4-1)*3 = 9");
        CHECK(strata::kernels::TOKEN_NULL == -1, "TOKEN_NULL == -1");
        CHECK(strata::kernels::PLE_TABLE_ROWS == 320001536ull, "PLE_TABLE_ROWS");
        CHECK(strata::kernels::PLE_ROW_BYTES == 90, "PLE_ROW_BYTES == (160/32)*18 = 90");
    }

    // ── iq4nl_code ─────────────────────────────────────────────────
    {
        CHECK(strata::kernels::iq4nl_code(0) == -127, "iq4nl_code(0) = -127");
        CHECK(strata::kernels::iq4nl_code(8) == 1, "iq4nl_code(8) = 1");
        CHECK(strata::kernels::iq4nl_code(15) == 113, "iq4nl_code(15) = 113");
    }

    // ── iq4nl_dequant_row: split-half not interleaved ──────────────
    {
        // Build one block (18 bytes): scale=1.0 as fp16, 16 qs bytes
        uint8_t block[18] = {};
        uint16_t scale_f16 = strata::kernels::f16_from_f32(1.0f);
        std::memcpy(block, &scale_f16, 2);

        // qs[0] = 0x80: lo nibble=0 (code -127), hi nibble=8 (code 1)
        block[2] = 0x80;
        // qs[1..15] = 0
        for (int i = 3; i < 18; ++i) block[i] = 0;

        float out[32] = {};
        // Dequant only 1 block (32 elems), build 5 blocks for full row
        uint8_t row[90] = {};
        std::memcpy(row, block, 18);
        // Fill remaining 4 blocks with zeros (scale=0 → output=0)

        float out160[160] = {};
        strata::kernels::iq4nl_dequant_row(row, out160);

        // Block 0: qs[0] = 0x80
        // SPLIT-HALF: y[0] = d * cb[0x80 & 0xf] = 1.0 * cb[0] = 1.0 * (-127) = -127
        //             y[16] = d * cb[0x80 >> 4] = 1.0 * cb[8] = 1.0 * 1 = 1
        // NOT interleaved: y[0] and y[16], NOT y[0] and y[1]
        CHECK_NEAR(out160[0], -127.0f, 0.01f, "iq4nl_dequant lo nibble → y[0]");
        CHECK_NEAR(out160[16], 1.0f, 0.01f, "iq4nl_dequant hi nibble → y[16] (split-half)");
    }

    // ── ngram_mixed: XOR not sum ───────────────────────────────────
    {
        // XOR property: a ^ a = 0
        int64_t ctx[] = {42, 42};
        uint64_t mult[] = {1, 1};
        uint64_t mixed = strata::kernels::ngram_mixed(ctx, mult, 2);
        // 42*1 XOR 42*1 = 42 XOR 42 = 0
        CHECK(mixed == 0, "ngram_mixed XOR: equal tokens cancel");

        // With sum it would be 84, not 0
        int64_t ctx2[] = {42};
        uint64_t mixed1 = strata::kernels::ngram_mixed(ctx2, mult, 1);
        CHECK(mixed1 == 42, "ngram_mixed single token");
    }

    // ── PleConsts ──────────────────────────────────────────────────
    {
        strata::kernels::PleConsts c = strata::kernels::ple_artifact_consts();
        CHECK(c.mult[0] != 0, "PleConsts mult[0] non-zero");
        CHECK(c.vocab[0] != 0, "PleConsts vocab[0] non-zero");
        // Vocab sizes should not be powers of 2 (use %, not &)
        CHECK((c.vocab[0] & (c.vocab[0] - 1)) != 0,
              "PleConsts vocab[0] not power of 2");
    }

    // ── ngram_rows: produces valid indices ─────────────────────────
    {
        strata::kernels::PleConsts c = strata::kernels::ple_artifact_consts();
        int32_t tokens[] = {100};
        int32_t prev[] = {50};
        uint32_t out[16] = {};

        strata::kernels::ngram_rows(tokens, prev, 1, c, out);

        // All row indices should be within table bounds
        for (int h = 0; h < 16; ++h) {
            CHECK(out[h] < strata::kernels::PLE_TABLE_ROWS,
                  "ngram_rows row within table bounds");
        }
    }

    // ── PleIo enum ─────────────────────────────────────────────────
    {
        CHECK(static_cast<int>(strata::kernels::PleIo::Direct) == 0, "PleIo::Direct = 0");
        CHECK(static_cast<int>(strata::kernels::PleIo::Mmap) == 1, "PleIo::Mmap = 1");
    }

    // ── PleIoOptions default ───────────────────────────────────────
    {
        strata::kernels::PleIoOptions io;
        CHECK(io.mode == strata::kernels::PleIo::Direct, "PleIoOptions default Direct");
        CHECK(io.max_inflight == 64, "PleIoOptions default max_inflight=64");
        CHECK(io.cache_rows == (1ull << 20), "PleIoOptions default cache_rows=2^20");
        CHECK(io.io_thread == true, "PleIoOptions default io_thread=true");
    }

    // ── PleTable lifecycle ─────────────────────────────────────────
    {
        strata::kernels::PleTable table;
        CHECK(!table.is_open(), "PleTable initial not open");

        std::string err;
        bool ok = table.open("/dev/null", err);
        CHECK(ok, "PleTable open succeeds");
        CHECK(table.is_open(), "PleTable is_open after open");
        CHECK(table.rows() == strata::kernels::PLE_TABLE_ROWS, "PleTable rows count");
        CHECK(table.mode() == strata::kernels::PleIo::Direct, "PleTable default mode Direct");

        // Gather: 16 rows → 2560 floats
        uint32_t rows16[16] = {};
        float out2560[2560] = {};
        table.gather(rows16, out2560);
        CHECK(std::isfinite(out2560[0]), "PleTable gather produces finite");

        // gather_batch
        float batch_out[2560] = {};
        ok = table.gather_batch(rows16, 1, batch_out, err);
        CHECK(ok, "PleTable gather_batch succeeds");

        // I/O report
        std::string report = table.io_report();
        CHECK(!report.empty(), "PleTable io_report non-empty");
        CHECK(table.bytes_read() > 0, "PleTable bytes_read > 0 after gather");

        // Close
        table.close();
        CHECK(!table.is_open(), "PleTable closed");
    }

    // ── PleTable: open with options ────────────────────────────────
    {
        strata::kernels::PleTable table;
        std::string err;
        strata::kernels::PleIoOptions opts;
        opts.mode = strata::kernels::PleIo::Mmap;
        opts.cache_rows = 1024;

        bool ok = table.open("/dev/null", err, opts);
        CHECK(ok, "PleTable open with Mmap options");
        CHECK(table.mode() == strata::kernels::PleIo::Mmap, "PleTable Mmap mode");
        table.close();
    }

    // ── PleTable: read_row ─────────────────────────────────────────
    {
        strata::kernels::PleTable table;
        std::string err;
        table.open("/dev/null", err);

        float out160[160] = {};
        table.read_row(42, out160);
        CHECK(std::isfinite(out160[0]), "PleTable read_row finite");
        table.close();
    }

    // ── PleTable: issue + collect ──────────────────────────────────
    {
        strata::kernels::PleTable table;
        std::string err;
        table.open("/dev/null", err);

        uint32_t rows16[16] = {};
        CHECK(table.issue(rows16), "PleTable issue succeeds");

        float out[2560] = {};
        CHECK(table.collect(out, err), "PleTable collect succeeds");
        table.close();
    }

    // ── PleTable: set_injected_delay ───────────────────────────────
    {
        strata::kernels::PleTable table;
        std::string err;
        table.open("/dev/null", err);
        table.set_injected_delay_us(100.0);
        table.close();
        CHECK(true, "PleTable set_injected_delay callable");
    }

    // ── ple_prefetch_enable ────────────────────────────────────────
    {
        strata::kernels::ple_prefetch_enable(true);
        CHECK(strata::kernels::ple_prefetch_enabled() == true, "prefetch enabled");
        strata::kernels::ple_prefetch_enable(false);
        CHECK(strata::kernels::ple_prefetch_enabled() == false, "prefetch disabled");
    }

    // ── PLE gather dimension: 16 × 160 = 2560 ─────────────────────
    {
        CHECK(strata::kernels::PLE_N_HEADS * strata::kernels::PLE_HEAD_DIM == 2560,
              "PLE_N_HEADS * PLE_HEAD_DIM = 2560 = n_embd");
    }

    // ── PLE cache size ≈ 95 MB ─────────────────────────────────────
    {
        uint64_t cache_bytes = (1ull << 20) * strata::kernels::PLE_ROW_BYTES;
        CHECK(cache_bytes == 1048576ull * 90, "cache ≈ 95 MB");
        // 1048576 * 90 = 94371840
        CHECK(cache_bytes == 94371840ull, "cache = 94371840 bytes");
    }

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}


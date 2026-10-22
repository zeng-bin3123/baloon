// tests/test_unit4.cpp — Smoke test for unit 4 (plan, suffix_drafter, f16_bits)
#include "strata/plan/plan.hpp"
#include "strata/spec/suffix_drafter.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { ++tests_passed; } \
    else { ++tests_failed; fprintf(stderr, "FAIL: %s\n", msg); } \
} while(0)

int main() {
    printf("=== strata unit 4 smoke test ===\n");

    // ── PlanGeometry ──────────────────────────────────────────────
    {
        strata::PlanGeometry g;
        CHECK(g.n_layers == 48, "PlanGeometry n_layers default 48");
        CHECK(g.n_qsa_layers == 12, "PlanGeometry n_qsa_layers default 12");
        CHECK(g.n_gdn_layers() == 36, "PlanGeometry n_gdn_layers() == 36");
        CHECK(g.n_kv_heads == 2, "PlanGeometry n_kv_heads == 2");
        CHECK(g.head_dim == 256, "PlanGeometry head_dim == 256");
        CHECK(g.kv_group == 64, "PlanGeometry kv_group == 64");
        CHECK(g.qsa_block == 4, "PlanGeometry qsa_block == 4");
        CHECK(g.indexer_key_heads == 1, "PlanGeometry indexer_key_heads == 1");
        CHECK(g.indexer_key_dim == 128, "PlanGeometry indexer_key_dim == 128");
        CHECK(g.ssm_state_size == 128, "PlanGeometry ssm_state_size == 128");
        CHECK(g.ssm_v_heads == 48, "PlanGeometry ssm_v_heads == 48");
        CHECK(g.ssm_d_conv == 4, "PlanGeometry ssm_d_conv == 4");
        CHECK(g.ssm_conv_channels == 10240, "PlanGeometry ssm_conv_channels == 10240");
    }

    // ── state_bytes ───────────────────────────────────────────────
    {
        strata::PlanGeometry g;
        uint64_t sb = strata::state_bytes(g);
        // Per layer: 128*48*128*4 + 3*10240*4 = 3145728 + 122880 = 3268608
        CHECK(sb == 36ull * 3268608ull, "state_bytes == 117669888");
        CHECK(sb == 117669888ull, "state_bytes exact value");
    }

    // ── kv_bytes_per_token ────────────────────────────────────────
    {
        strata::PlanGeometry g;
        uint64_t kvb = strata::kv_bytes_per_token(g);
        // kv_elems = 2*2*256 = 1024
        // kv_scales = 1024/64*2 = 32
        // per_layer = 1024 + 32 = 1056
        // idx_layer = 1*128/4 = 32
        // total = 12 * (1056+32) = 12 * 1088 = 13056
        CHECK(kvb == 13056ull, "kv_bytes_per_token == 13056");
    }

    // ── make_plan basic ───────────────────────────────────────────
    {
        strata::PlanGeometry g;
        strata::PlanCosts c;
        c.dense_bytes = 1000000000ull;
        c.embd_bytes  = 100000000ull;
        c.mtp_bytes   = 900000000ull;
        c.workspace_bytes = 200000000ull;
        c.state_bytes = strata::state_bytes(g);
        c.expert_blob = 1382400ull;

        uint64_t max_ctx = 4096;
        strata::Plan p = strata::make_plan(max_ctx, g, c);

        CHECK(p.max_context == max_ctx, "plan max_context");
        CHECK(p.vram_budget == strata::vram_pool_bytes(), "plan vram_budget");
        CHECK(p.kv_bytes == 13056ull * 4096ull, "plan kv_bytes");
        CHECK(p.state_bytes == strata::state_bytes(g), "plan state_bytes");
        CHECK(p.cache_slots > 0, "plan cache_slots > 0");
        CHECK(p.cache_bytes > 0, "plan cache_bytes > 0");
        CHECK(p.vram_used <= p.vram_budget, "plan vram_used <= budget");
        CHECK(p.dram_experts == c.dense_bytes + c.embd_bytes, "plan dram_experts");
    }

    // ── make_plan DoesNotClose (overflow) ─────────────────────────
    {
        strata::PlanGeometry g;
        strata::PlanCosts c;
        c.dense_bytes = 5000000000ull;  // deliberately too big
        c.embd_bytes  = 1000000000ull;
        c.mtp_bytes   = 0;
        c.workspace_bytes = 0;
        c.state_bytes = 0;
        c.expert_blob = 1382400ull;

        bool caught = false;
        try {
            strata::make_plan(4096, g, c);
        } catch (const strata::DoesNotClose& e) {
            caught = true;
        }
        CHECK(caught, "DoesNotClose thrown on overflow");
    }

    // ── make_plan DoesNotClose (zero slots) ───────────────────────
    {
        strata::PlanGeometry g;
        strata::PlanCosts c;
        c.dense_bytes = 0;
        c.embd_bytes  = 0;
        c.mtp_bytes   = 0;
        c.workspace_bytes = 0;
        c.state_bytes = 0;
        // expert_blob bigger than entire pool
        c.expert_blob = 6000000000ull;

        bool caught = false;
        try {
            strata::make_plan(1, g, c);
        } catch (const strata::DoesNotClose& e) {
            caught = true;
        }
        CHECK(caught, "DoesNotClose thrown on zero slots");
    }

    // ── vram_pool_bytes ───────────────────────────────────────────
    {
        CHECK(strata::vram_pool_bytes() == 5943000000ull, "vram_pool_bytes == 5943000000");
    }

    // ── SuffixDrafter basic ───────────────────────────────────────
    {
        strata::SuffixDrafter drafter(3, 32);
        CHECK(drafter.size() == 0, "empty drafter size 0");
        CHECK(drafter.last_match() == 0, "empty drafter match 0");
    }

    // ── SuffixDrafter append + propose ────────────────────────────
    {
        strata::SuffixDrafter drafter(3, 32);

        // Build history: [10, 20, 30, 40, 50, 10, 20, 30]
        // The suffix [10, 20, 30] appears twice. After appending [10,20,30],
        // the trigram matches position 2 (end of first [10,20,30]).
        // The token that followed position 2 was 40.
        int32_t seq[] = {10, 20, 30, 40, 50, 10, 20, 30};
        drafter.append(seq, 8);
        CHECK(drafter.size() == 8, "drafter size 8 after append");

        int32_t out[4] = {};
        int n = drafter.propose(4, out);
        // Should find match: suffix [10,20,30] appeared at end idx 2.
        // Tokens after: 40, 50 (then we're into the repeated part).
        CHECK(n > 0, "propose returns > 0 tokens");
        if (n > 0) {
            CHECK(out[0] == 40, "propose first token == 40");
        }
        CHECK(drafter.last_match() >= 3, "last_match >= 3");
    }

    // ── SuffixDrafter reset ───────────────────────────────────────
    {
        strata::SuffixDrafter drafter(3, 32);
        int32_t seq[] = {1, 2, 3, 4, 5};
        drafter.append(seq, 5);
        drafter.reset();
        CHECK(drafter.size() == 0, "drafter size 0 after reset");
        CHECK(drafter.last_match() == 0, "drafter match 0 after reset");
    }

    // ── SuffixDrafter no match ────────────────────────────────────
    {
        strata::SuffixDrafter drafter(3, 32);
        // All unique tokens — no repeated trigram
        int32_t seq[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        drafter.append(seq, 10);
        int32_t out[4] = {};
        int n = drafter.propose(4, out);
        CHECK(n == 0, "propose 0 on no match");
    }

    // ── SuffixDrafter single token append ─────────────────────────
    {
        strata::SuffixDrafter drafter(3, 32);
        for (int i = 0; i < 6; ++i) {
            drafter.append(static_cast<int32_t>(i % 3));
        }
        // History: [0, 1, 2, 0, 1, 2] — trigram [0,1,2] repeated
        CHECK(drafter.size() == 6, "single append size 6");
        int32_t out[2] = {};
        int n = drafter.propose(2, out);
        CHECK(n >= 0, "propose returns >= 0");
    }

    // ── f16_from_f32: zero ────────────────────────────────────────
    {
        uint16_t h = strata::kernels::f16_from_f32(0.0f);
        CHECK(h == 0x0000u, "f16_from_f32(0.0) == 0x0000");
    }

    // ── f16_from_f32: negative zero ───────────────────────────────
    {
        uint16_t h = strata::kernels::f16_from_f32(-0.0f);
        CHECK(h == 0x8000u, "f16_from_f32(-0.0) == 0x8000");
    }

    // ── f16_from_f32: 1.0 ────────────────────────────────────────
    {
        uint16_t h = strata::kernels::f16_from_f32(1.0f);
        // fp16 1.0 = 0 01111 0000000000 = 0x3C00
        CHECK(h == 0x3C00u, "f16_from_f32(1.0) == 0x3C00");
    }

    // ── f16_from_f32: -1.0 ───────────────────────────────────────
    {
        uint16_t h = strata::kernels::f16_from_f32(-1.0f);
        CHECK(h == 0xBC00u, "f16_from_f32(-1.0) == 0xBC00");
    }

    // ── f16_from_f32: infinity ────────────────────────────────────
    {
        float inf = std::numeric_limits<float>::infinity();
        uint16_t h = strata::kernels::f16_from_f32(inf);
        CHECK(h == 0x7C00u, "f16_from_f32(+inf) == 0x7C00");
    }

    // ── f16_from_f32: -infinity ───────────────────────────────────
    {
        float ninf = -std::numeric_limits<float>::infinity();
        uint16_t h = strata::kernels::f16_from_f32(ninf);
        CHECK(h == 0xFC00u, "f16_from_f32(-inf) == 0xFC00");
    }

    // ── f16_from_f32: NaN ─────────────────────────────────────────
    {
        float nan_val = std::numeric_limits<float>::quiet_NaN();
        uint16_t h = strata::kernels::f16_from_f32(nan_val);
        // Must be NaN: exponent all 1s, mantissa nonzero
        CHECK((h & 0x7C00u) == 0x7C00u && (h & 0x03FFu) != 0,
              "f16_from_f32(NaN) is NaN");
    }

    // ── f16_from_f32: overflow → inf NOT NaN ──────────────────────
    {
        float big = 65536.0f;  // > fp16 max (65504)
        uint16_t h = strata::kernels::f16_from_f32(big);
        CHECK(h == 0x7C00u, "f16_from_f32(overflow) == inf not NaN");
    }

    // ── f16_from_f32: max finite fp16 ─────────────────────────────
    {
        // fp16 max = 65504.0
        uint16_t h = strata::kernels::f16_from_f32(65504.0f);
        CHECK(h == 0x7BFFu, "f16_from_f32(65504) == 0x7BFF");
    }

    // ── f32_from_f16: zero ────────────────────────────────────────
    {
        float f = strata::kernels::f32_from_f16(0x0000u);
        CHECK(f == 0.0f, "f32_from_f16(0x0000) == 0.0");
    }

    // ── f32_from_f16: 1.0 ────────────────────────────────────────
    {
        float f = strata::kernels::f32_from_f16(0x3C00u);
        CHECK(f == 1.0f, "f32_from_f16(0x3C00) == 1.0");
    }

    // ── f32_from_f16: -1.0 ───────────────────────────────────────
    {
        float f = strata::kernels::f32_from_f16(0xBC00u);
        CHECK(f == -1.0f, "f32_from_f16(0xBC00) == -1.0");
    }

    // ── f32_from_f16: infinity ────────────────────────────────────
    {
        float f = strata::kernels::f32_from_f16(0x7C00u);
        CHECK(std::isinf(f) && f > 0, "f32_from_f16(0x7C00) == +inf");
    }

    // ── f32_from_f16: NaN ─────────────────────────────────────────
    {
        float f = strata::kernels::f32_from_f16(0x7E00u);
        CHECK(std::isnan(f), "f32_from_f16(0x7E00) is NaN");
    }

    // ── f32_from_f16: subnormal NOT flushed to zero ───────────────
    {
        float f = strata::kernels::f32_from_f16(0x0001u);
        // Smallest positive fp16 subnormal = 2^{-24} ≈ 5.96e-8
        CHECK(f > 0.0f, "f32_from_f16(0x0001) > 0 (not flushed)");
        float expected = 5.960464477539063e-8f;  // 2^{-24}
        CHECK(std::fabs(f - expected) < 1e-15f,
              "f32_from_f16(0x0001) ≈ 2^{-24}");
    }

    // ── f16↔f32 roundtrip: all exact fp16 values ──────────────────
    {
        int roundtrip_fails = 0;
        for (uint32_t h = 0; h < 65536; ++h) {
            uint16_t h16 = static_cast<uint16_t>(h);
            float f = strata::kernels::f32_from_f16(h16);
            uint16_t back = strata::kernels::f16_from_f32(f);
            // NaN: mantissa bits may differ, just check it's still NaN
            if ((h16 & 0x7C00u) == 0x7C00u && (h16 & 0x03FFu) != 0) {
                if (!((back & 0x7C00u) == 0x7C00u && (back & 0x03FFu) != 0)) {
                    ++roundtrip_fails;
                }
            } else {
                if (back != h16) {
                    ++roundtrip_fails;
                }
            }
        }
        CHECK(roundtrip_fails == 0, "f16→f32→f16 roundtrip all 65536 values");
    }

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}


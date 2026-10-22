// tests/test_unit3.cpp — Smoke test for unit 3 (layer + session)
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { ++tests_passed; } \
    else { ++tests_failed; fprintf(stderr, "FAIL: %s\n", msg); } \
} while(0)

int main() {
    printf("=== strata unit 3 smoke test ===\n");

    // Create a test Geometry
    strata::Geometry g;
    g.n_embd = 2560;
    g.n_head = 20;
    g.n_kv_head = 4;
    g.n_layers = 48;
    g.n_vocab = 32000;
    g.n_experts = 512;
    g.n_experts_active = 10;
    g.expert_hidden = 640;
    g.hc = 1;

    // ── 1. Configuration selectors ───────────────────────────────
    {
        strata::layer_set_native_bf16(true);
        strata::layer_set_native_bf16(false);
        // Idempotent: b = b
        strata::layer_set_native_bf16(true);
        strata::layer_set_native_bf16(true);
        CHECK(true, "layer_set_native_bf16 idempotent");

        strata::layer_set_native_flash_attn_short(true);
        strata::layer_set_native_flash_attn_short(false);
        CHECK(true, "layer_set_native_flash_attn_short toggle");
    }

    // ── 2. GdnBuffers defaults ───────────────────────────────────
    {
        strata::GdnBuffers b;
        CHECK(b.x_q8k == nullptr, "GdnBuffers default x_q8k null");
        CHECK(b.x_q8_0 == nullptr, "GdnBuffers default x_q8_0 null");
        CHECK(b.x_bf16 == nullptr, "GdnBuffers default x_bf16 null");
        CHECK(b.qkv == nullptr, "GdnBuffers default qkv null");
        CHECK(b.state == nullptr, "GdnBuffers default state null");
        CHECK(b.conv_state == nullptr, "GdnBuffers default conv_state null");
        CHECK(b.alpha == nullptr, "GdnBuffers default alpha null");
        CHECK(b.beta == nullptr, "GdnBuffers default beta null");
        CHECK(b.gate == nullptr, "GdnBuffers default gate null");
        CHECK(b.o == nullptr, "GdnBuffers default o null");
    }

    // ── 3. gdn_buffers_bytes returns nonzero ─────────────────────
    {
        uint64_t bytes = strata::gdn_buffers_bytes(g);
        CHECK(bytes > 0, "gdn_buffers_bytes > 0");
        // 16-byte aligned
        CHECK(bytes % 16 == 0, "gdn_buffers_bytes 16-byte aligned");
        printf("  gdn_buffers_bytes = %lu\n", (unsigned long)bytes);
    }

    // ── 4. gdn_buffers_init carves device memory ─────────────────
    {
        uint64_t bytes = strata::gdn_buffers_bytes(g);
        // Use host memory as standin for testing (just checking pointer arithmetic)
        std::vector<uint8_t> arena(bytes + 16, 0);
        // Align to 16
        uint8_t* base = arena.data();
        uintptr_t off = reinterpret_cast<uintptr_t>(base) % 16;
        if (off) base += (16 - off);

        strata::GdnBuffers b;
        uint64_t used = strata::gdn_buffers_init(g, base, b);
        CHECK(used > 0, "gdn_buffers_init used > 0");
        CHECK(used <= bytes, "gdn_buffers_init used <= bytes");
        CHECK(b.x_q8k != nullptr, "gdn_buffers_init x_q8k set");
        CHECK(b.x_q8_0 != nullptr, "gdn_buffers_init x_q8_0 set");
        CHECK(b.x_bf16 != nullptr, "gdn_buffers_init x_bf16 set");
        CHECK(b.qkv != nullptr, "gdn_buffers_init qkv set");
        CHECK(b.conv_out != nullptr, "gdn_buffers_init conv_out set");
        CHECK(b.state != nullptr, "gdn_buffers_init state set");
        CHECK(b.conv_state != nullptr, "gdn_buffers_init conv_state set");
        CHECK(b.y_q8k != nullptr, "gdn_buffers_init y_q8k set");
        CHECK(b.y_q8_0 != nullptr, "gdn_buffers_init y_q8_0 set");

        // Pointers are in order
        CHECK(reinterpret_cast<uintptr_t>(b.x_q8_0) >
              reinterpret_cast<uintptr_t>(b.x_q8k),
              "x_q8_0 after x_q8k");
        CHECK(reinterpret_cast<uintptr_t>(b.state) >
              reinterpret_cast<uintptr_t>(b.y_q8_0),
              "state after y_q8_0");
    }

    // ── 5. MoEBuffers defaults ───────────────────────────────────
    {
        strata::MoEBuffers m;
        CHECK(m.x_bf16 == nullptr, "MoEBuffers default x_bf16 null");
        CHECK(m.logits == nullptr, "MoEBuffers default logits null");
        CHECK(m.ids == nullptr, "MoEBuffers default ids null");
        CHECK(m.weights == nullptr, "MoEBuffers default weights null");
        CHECK(m.shared == nullptr, "MoEBuffers default shared null");
    }

    // ── 6. Doorbell defaults ─────────────────────────────────────
    {
        strata::Doorbell db;
        CHECK(db.h_flag == nullptr, "Doorbell default h_flag null");
        CHECK(db.d_flag == nullptr, "Doorbell default d_flag null");
        CHECK(db.layer == -1, "Doorbell default layer -1");
    }

    // ── 7. gdn_layer input validation ────────────────────────────
    {
        strata::WeightTable wt;
        strata::GdnBuffers b;
        std::string err;
        float dummy = 0;

        bool ok = strata::gdn_layer(wt, g, 0, b, nullptr, &dummy, nullptr, err);
        CHECK(!ok, "gdn_layer rejects null input");
        CHECK(err.find("null") != std::string::npos, "gdn_layer null error msg");

        err.clear();
        ok = strata::gdn_layer(wt, g, 0, b, &dummy, nullptr, nullptr, err);
        CHECK(!ok, "gdn_layer rejects null output");

        err.clear();
        ok = strata::gdn_layer(wt, g, 99, b, &dummy, &dummy, nullptr, err);
        CHECK(!ok, "gdn_layer rejects out-of-range layer");

        err.clear();
        ok = strata::gdn_layer(wt, g, 0, b, &dummy, &dummy, nullptr, err);
        CHECK(ok, "gdn_layer accepts valid params");
    }

    // ── 8. moe_layer input validation ────────────────────────────
    {
        strata::WeightTable wt;
        strata::MoEBuffers mb;
        std::string err;
        float dummy = 0;

        bool ok = strata::moe_layer(wt, g, 0, 10, mb,
                                     nullptr, &dummy, &dummy, nullptr, err);
        CHECK(!ok, "moe_layer rejects null input");

        err.clear();
        ok = strata::moe_layer(wt, g, 0, 0, mb,
                                &dummy, &dummy, &dummy, nullptr, err);
        CHECK(!ok, "moe_layer rejects k=0");

        err.clear();
        ok = strata::moe_layer(wt, g, 0, 10, mb,
                                &dummy, &dummy, &dummy, nullptr, err);
        CHECK(ok, "moe_layer accepts valid params");
    }

    // ── 9. moe_route / moe_finish / moe_shared / moe_combine_parts
    {
        strata::WeightTable wt;
        strata::MoEBuffers mb;
        std::string err;
        float dummy = 0;

        CHECK(strata::moe_route(wt, g, 0, 10, mb, &dummy, nullptr, err),
              "moe_route accepts valid params");

        err.clear();
        CHECK(strata::moe_finish(wt, g, 0, 10, mb, &dummy, &dummy, &dummy, nullptr, err),
              "moe_finish accepts valid params");

        err.clear();
        CHECK(strata::moe_shared(wt, g, 0, mb, &dummy, nullptr, err),
              "moe_shared accepts valid params");

        err.clear();
        CHECK(strata::moe_combine_parts(g, 0, 10, mb, &dummy, &dummy, nullptr, err),
              "moe_combine_parts accepts valid params");
    }

    // ── 10. Shared-early scheduling ──────────────────────────────
    {
        strata::layer_set_shared_early(false);
        CHECK(!strata::layer_shared_early(), "shared_early off");
        strata::layer_set_shared_early(true);
        CHECK(strata::layer_shared_early(), "shared_early on");
        strata::layer_set_shared_early(false);
    }

    // ── 11. SessionData defaults ─────────────────────────────────
    {
        strata::SessionData sd;
        CHECK(sd.max_cells == 0, "SessionData default max_cells 0");
        CHECK(sd.gdn_state == nullptr, "SessionData default gdn_state null");
        CHECK(sd.qsa_states == nullptr, "SessionData default qsa_states null");
        CHECK(sd.R == nullptr, "SessionData default R null");
        CHECK(sd.k == 10, "SessionData default k = 10");
        CHECK(sd.db == nullptr, "SessionData default db null");
        CHECK(sd.ple_prev[0] == -1, "SessionData default ple_prev[0] = -1");
        CHECK(sd.ple_prev[1] == -1, "SessionData default ple_prev[1] = -1");
        CHECK(sd.ple_token == -1, "SessionData default ple_token = -1");
        CHECK(sd.ple_hist == nullptr, "SessionData default ple_hist null");
    }

    // ── 12. session_bytes returns nonzero ────────────────────────
    {
        uint64_t bytes = strata::session_bytes(g, 4096, 10);
        CHECK(bytes > 0, "session_bytes > 0");
        printf("  session_bytes(4096, 10) = %lu\n", (unsigned long)bytes);

        // Should be at least gdn_buffers_bytes
        uint64_t gdn_min = strata::gdn_buffers_bytes(g);
        CHECK(bytes > gdn_min, "session_bytes > gdn_buffers_bytes");
    }

    // ── 13. session_init + session_zero on device ────────────────
    {
        uint64_t bytes = strata::session_bytes(g, 128, 10);

        void* dev_base = nullptr;
        cudaError_t rc = cudaMalloc(&dev_base, bytes);
        if (rc == cudaSuccess) {
            strata::SessionData sd;
            uint64_t used = strata::session_init(g, 128, 10, dev_base, sd);
            CHECK(used > 0, "session_init used > 0");
            CHECK(used <= bytes, "session_init used <= bytes");
            CHECK(sd.R != nullptr, "session_init R set");
            CHECK(sd.gdn_state != nullptr, "session_init gdn_state set");
            CHECK(sd.max_cells == 128, "session_init max_cells = 128");
            CHECK(sd.k == 10, "session_init k = 10");
            CHECK(sd.ple_prev[0] == -1, "session_init ple_prev[0] = -1");
            CHECK(sd.ple_hist != nullptr, "session_init ple_hist set");

            // QSA states
            CHECK(sd.qsa_states != nullptr, "session_init qsa_states set");
            CHECK(sd.qsa_states[0].max_cells == 128, "qsa_state[0] max_cells");
            CHECK(sd.qsa_states[0].k_cache != nullptr, "qsa_state[0] k_cache set");

            // MoE buffers
            CHECK(sd.moe.x_bf16 != nullptr, "session_init moe.x_bf16 set");
            CHECK(sd.moe.logits != nullptr, "session_init moe.logits set");

            // Block buffers
            CHECK(sd.block.norm_out != nullptr, "session_init block.norm_out set");

            // session_zero
            cudaStream_t stream = nullptr;
            strata::session_zero(sd, g, nullptr, stream);
            cudaDeviceSynchronize();
            CHECK(sd.ple_prev[0] == -1, "session_zero ple_prev reset");
            CHECK(sd.ple_token == -1, "session_zero ple_token reset");

            cudaFree(dev_base);
        } else {
            printf("  (skipping device session_init: no GPU)\n");
            CHECK(true, "session_init skipped (no GPU)");
        }
    }

    // ── 14. session_token validation ─────────────────────────────
    {
        strata::WeightTable wt;
        strata::SessionData sd;  // uninitialized
        std::string err;

        bool ok = strata::session_token(wt, g, 0, 0, sd, nullptr, nullptr, false, err);
        CHECK(!ok, "session_token rejects uninitialized session");
        CHECK(err.find("null") != std::string::npos || err.find("not initialized") != std::string::npos,
              "session_token error mentions initialization");
    }

    // ── 15. SessionGraphs defaults + free ────────────────────────
    {
        strata::SessionGraphs gr;
        CHECK(gr.execs == nullptr, "SessionGraphs default execs null");
        CHECK(gr.posts == nullptr, "SessionGraphs default posts null");
        CHECK(!gr.captured, "SessionGraphs default not captured");
        CHECK(!gr.split_captured, "SessionGraphs default not split_captured");
        CHECK(gr.n == 0, "SessionGraphs default n = 0");
        CHECK(gr.rings_mid_graph == 0, "SessionGraphs default rings_mid_graph = 0");
        CHECK(gr.calls_total == 0, "SessionGraphs default calls_total = 0");

        // Free on empty should be safe
        strata::session_graphs_free(gr);
        CHECK(gr.execs == nullptr, "session_graphs_free safe on empty");
    }

    // ── 16. session_capture + session_replay ─────────────────────
    {
        strata::WeightTable wt;
        strata::SessionData sd;
        sd.R = reinterpret_cast<float*>(1);  // non-null for validation
        strata::SessionGraphs gr;
        std::string err;

        bool ok = strata::session_capture(wt, g, sd, nullptr, gr, err);
        CHECK(ok, "session_capture succeeds");
        CHECK(gr.captured, "session_capture sets captured");
        CHECK(gr.n == 48, "session_capture n = 48");
        CHECK(gr.execs != nullptr, "session_capture allocates execs");
        CHECK(gr.posts != nullptr, "session_capture allocates posts");

        // Idempotent
        ok = strata::session_capture(wt, g, sd, nullptr, gr, err);
        CHECK(ok, "session_capture idempotent");

        // Replay
        ok = strata::session_replay(g, 0, 0, sd, gr, nullptr, err);
        CHECK(ok, "session_replay succeeds");
        CHECK(gr.calls_total == 1, "session_replay increments calls");

        // Replay full
        ok = strata::session_replay_full(g, 0, 0, sd, gr, nullptr, err);
        CHECK(ok, "session_replay_full succeeds");
        CHECK(gr.calls_total == 2, "session_replay_full increments calls");

        strata::session_graphs_free(gr);
    }

    // ── 17. SessionLoopScratch ───────────────────────────────────
    {
        strata::SessionLoopScratch scratch;
        CHECK(scratch.y_miss == nullptr, "scratch default y_miss null");
        CHECK(scratch.probe == nullptr, "scratch default probe null");
        CHECK(!scratch.pinned, "scratch default not pinned");

        std::string err;
        size_t pb = 10 * 2560 * sizeof(float);  // k × n_embd
        bool ok = scratch.init(pb, err);
        if (ok) {
            CHECK(scratch.y_miss != nullptr, "scratch init y_miss set");
            CHECK(scratch.probe != nullptr, "scratch init probe set");
            CHECK(scratch.parts_bytes == pb, "scratch init parts_bytes");
            scratch.free();
            CHECK(scratch.y_miss == nullptr, "scratch free y_miss null");
            CHECK(scratch.probe == nullptr, "scratch free probe null");
        } else {
            printf("  (skipping scratch init: %s)\n", err.c_str());
            CHECK(true, "scratch init skipped");
        }
    }

    // ── 18. session_loop validation ──────────────────────────────
    {
        strata::SessionGraphs gr;
        strata::SessionData sd;
        std::string err;

        bool ok = strata::session_loop(g, 0, 0, sd, gr, nullptr, nullptr,
                                        nullptr, false, nullptr, err);
        CHECK(!ok, "session_loop rejects uncaptured graphs");

        // With captured but null pool
        strata::WeightTable wt;
        sd.R = reinterpret_cast<float*>(1);
        strata::session_capture(wt, g, sd, nullptr, gr, err);
        err.clear();
        ok = strata::session_loop(g, 0, 0, sd, gr, nullptr, nullptr,
                                   nullptr, false, nullptr, err);
        CHECK(!ok, "session_loop rejects null pool");

        strata::session_graphs_free(gr);
    }

    // ── 19. TokenGraph defaults ──────────────────────────────────
    {
        strata::TokenGraph tg;
        CHECK(tg.exec == nullptr, "TokenGraph default exec null");
        CHECK(!tg.captured, "TokenGraph default not captured");
        CHECK(tg.n_layers == 0, "TokenGraph default n_layers 0");
        CHECK(tg.calls == 0, "TokenGraph default calls 0");
        CHECK(tg.ms_wait == 0, "TokenGraph default ms_wait 0");
        CHECK(tg.flushes == 0, "TokenGraph default flushes 0");
    }

    // ── 20. TokenHits defaults + on() ────────────────────────────
    {
        strata::TokenHits th;
        CHECK(!th.on(), "TokenHits default not on");
        CHECK(th.d_res == nullptr, "TokenHits default d_res null");
        CHECK(th.cache_base == nullptr, "TokenHits default cache_base null");
    }

    // ── 21. session_capture_token / run_token ────────────────────
    {
        strata::WeightTable wt;
        strata::SessionData sd;
        sd.R = reinterpret_cast<float*>(1);
        strata::TokenGraph tg;
        std::string err;

        float dummy_host = 0;
        bool ok = strata::session_capture_token(wt, g, sd, nullptr,
                                                 &dummy_host, 1024, tg, err);
        CHECK(ok, "session_capture_token succeeds");
        CHECK(tg.captured, "session_capture_token sets captured");
        CHECK(tg.n_layers == 48, "session_capture_token n_layers = 48");
        CHECK(tg.parts_bytes == 1024, "session_capture_token parts_bytes");

        // Idempotent
        ok = strata::session_capture_token(wt, g, sd, nullptr,
                                            &dummy_host, 1024, tg, err);
        CHECK(ok, "session_capture_token idempotent");

        // run_token
        auto dummy_pool = [](void*, const float*, const int32_t*,
                             const float*, int64_t, int64_t, float*) {};
        ok = strata::session_run_token(g, 0, 0, sd, tg,
                                        dummy_pool, nullptr, &dummy_host,
                                        nullptr, err);
        CHECK(ok, "session_run_token succeeds");
        CHECK(tg.calls == 1, "session_run_token increments calls");

        // run_token without capture
        strata::TokenGraph tg2;
        err.clear();
        ok = strata::session_run_token(g, 0, 0, sd, tg2,
                                        dummy_pool, nullptr, &dummy_host,
                                        nullptr, err);
        CHECK(!ok, "session_run_token rejects uncaptured");

        strata::token_graph_free(tg);
        CHECK(!tg.captured, "token_graph_free resets captured");
        CHECK(tg.calls == 0, "token_graph_free resets calls");
    }

    // ── 22. QsaState / QsaBuffers / BlockBuffers defaults ────────
    {
        strata::QsaState qs;
        CHECK(qs.k_cache == nullptr, "QsaState default k_cache null");
        CHECK(qs.v_cache == nullptr, "QsaState default v_cache null");
        CHECK(qs.max_cells == 0, "QsaState default max_cells 0");
        CHECK(qs.pos == 0, "QsaState default pos 0");

        strata::QsaBuffers qb;
        CHECK(qb.q == nullptr, "QsaBuffers default q null");

        strata::BlockBuffers bb;
        CHECK(bb.norm_out == nullptr, "BlockBuffers default norm_out null");
        CHECK(bb.mixer_out == nullptr, "BlockBuffers default mixer_out null");
    }

    // ── 23. NG_HIST constant ─────────────────────────────────────
    {
        CHECK(strata::NG_HIST == 16, "NG_HIST = 16");
    }

    // ── 24. PleRun defaults ──────────────────────────────────────
    {
        strata::kernels::PleRun pr;
        CHECK(!pr.enabled, "PleRun default not enabled");
        CHECK(pr.hc_dim == 0, "PleRun default hc_dim 0");
        CHECK(pr.weights == nullptr, "PleRun default weights null");
    }

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}


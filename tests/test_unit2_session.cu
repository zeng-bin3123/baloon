// tests/test_unit2_session.cu — Smoke test for unit 2/31 modules:
//   graph_registry (body-error detection, steady_clock wait)
//   session_orchestrator (session_token, session_capture, session_replay,
//                         session_replay_full, session_loop)
#include "strata/core/graph.hpp"
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

// ─── Helpers ─────────────────────────────────────────────────────

// A simple CUDA kernel that does nothing (used for graph capture tests)
__global__ void noop_kernel() {}

// A kernel that writes a known value to device memory
__global__ void write_val_kernel(float* dst, float val) {
    dst[0] = val;
}

int main() {
    printf("=== strata unit 2 session smoke test ===\n");

    // Initialize CUDA
    cudaError_t init_rc = cudaSetDevice(0);
    if (init_rc != cudaSuccess) {
        printf("SKIP: no CUDA device (%s)\n", cudaGetErrorString(init_rc));
        return 0;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // ═══════════════════════════════════════════════════════════════
    // 1. GraphRegistry: LayerType enum
    // ═══════════════════════════════════════════════════════════════
    printf("  [1] LayerType enum...\n");
    CHECK(static_cast<int>(strata::LayerType::GDN) == 0, "LayerType::GDN == 0");
    CHECK(static_cast<int>(strata::LayerType::QSA) == 1, "LayerType::QSA == 1");

    const char* gdn_str = strata::to_string(strata::LayerType::GDN);
    const char* qsa_str = strata::to_string(strata::LayerType::QSA);
    CHECK(strcmp(gdn_str, "GDN") == 0, "to_string(GDN)");
    CHECK(strcmp(qsa_str, "QSA") == 0, "to_string(QSA)");

    // ═══════════════════════════════════════════════════════════════
    // 2. CapturedGraph: default state
    // ═══════════════════════════════════════════════════════════════
    printf("  [2] CapturedGraph default...\n");
    {
        strata::CapturedGraph cg;
        CHECK(!cg.valid(), "default not valid");
        CHECK(cg.nodes() == 0, "default 0 nodes");
        CHECK(!cg.wait_ms(1), "wait on default returns false");
    }

    // ═══════════════════════════════════════════════════════════════
    // 3. CapturedGraph: capture + launch + wait (real kernel)
    // ═══════════════════════════════════════════════════════════════
    printf("  [3] CapturedGraph capture/launch/wait...\n");
    {
        strata::CapturedGraph cg;
        std::string err;

        bool ok = cg.begin(stream, err);
        CHECK(ok, "begin succeeds");

        noop_kernel<<<1, 1, 0, stream>>>();

        ok = cg.end(stream, err);
        CHECK(ok, "end succeeds");
        CHECK(cg.valid(), "valid after end");
        CHECK(cg.nodes() > 0, "has nodes");

        ok = cg.launch(stream, err);
        CHECK(ok, "launch succeeds");

        ok = cg.wait_ms(5000);
        CHECK(ok, "wait_ms succeeds");
    }

    // ═══════════════════════════════════════════════════════════════
    // 4. CapturedGraph: zero-node rejection
    // ═══════════════════════════════════════════════════════════════
    printf("  [4] CapturedGraph zero-node rejection...\n");
    {
        strata::CapturedGraph cg;
        std::string err;

        bool ok = cg.begin(stream, err);
        CHECK(ok, "zero-node begin");

        // No kernels → 0 nodes → should be rejected
        ok = cg.end(stream, err);
        CHECK(!ok, "zero-node end fails");
        CHECK(!cg.valid(), "zero-node not valid");
    }

    // ═══════════════════════════════════════════════════════════════
    // 5. CapturedGraph: move semantics
    // ═══════════════════════════════════════════════════════════════
    printf("  [5] CapturedGraph move...\n");
    {
        strata::CapturedGraph cg1;
        std::string err;
        cg1.begin(stream, err);
        noop_kernel<<<1, 1, 0, stream>>>();
        cg1.end(stream, err);
        CHECK(cg1.valid(), "source valid before move");

        strata::CapturedGraph cg2(std::move(cg1));
        CHECK(cg2.valid(), "dest valid after move");
        CHECK(!cg1.valid(), "source invalid after move");
    }

    // ═══════════════════════════════════════════════════════════════
    // 6. GraphRegistry: record + find + launch
    // ═══════════════════════════════════════════════════════════════
    printf("  [6] GraphRegistry record/find/launch...\n");
    {
        strata::GraphRegistry reg(stream);
        CHECK(reg.size() == 0, "empty initially");
        CHECK(reg.captures() == 0, "0 captures");

        std::string err;

        bool ok = reg.record(strata::LayerType::GDN, 1, [&]() {
            noop_kernel<<<1, 1, 0, stream>>>();
        }, err);
        CHECK(ok, "record GDN/1");
        CHECK(reg.size() == 1, "size 1");
        CHECK(reg.captures() == 1, "captures 1");

        // Idempotent
        ok = reg.record(strata::LayerType::GDN, 1, [&]() {
            noop_kernel<<<1, 1, 0, stream>>>();
        }, err);
        CHECK(ok, "re-record GDN/1 (no-op)");
        CHECK(reg.size() == 1, "still size 1");

        // Second key
        ok = reg.record(strata::LayerType::QSA, 1, [&]() {
            noop_kernel<<<1, 1, 0, stream>>>();
        }, err);
        CHECK(ok, "record QSA/1");
        CHECK(reg.size() == 2, "size 2");

        // Find
        const strata::CapturedGraph* found = reg.find(strata::LayerType::GDN, 1);
        CHECK(found != nullptr, "find GDN/1");
        CHECK(found->valid(), "found is valid");

        CHECK(reg.find(strata::LayerType::GDN, 2) == nullptr, "find GDN/2 null");

        // Launch
        ok = reg.launch(strata::LayerType::GDN, 1, 5000, err);
        CHECK(ok, "launch GDN/1");

        ok = reg.launch(strata::LayerType::GDN, 99, 100, err);
        CHECK(!ok, "launch GDN/99 fails");
    }

    // ═══════════════════════════════════════════════════════════════
    // 7. GraphRegistry: body-error detection structure
    // ═══════════════════════════════════════════════════════════════
    printf("  [7] GraphRegistry body-error detection...\n");
    {
        strata::GraphRegistry reg(stream);
        std::string err;

        bool ok = reg.record(strata::LayerType::GDN, 42, [&]() {
            noop_kernel<<<1, 1, 0, stream>>>();
        }, err);
        CHECK(ok, "valid body captures ok");
    }

    // ═══════════════════════════════════════════════════════════════
    // 8. session_bytes
    // ═══════════════════════════════════════════════════════════════
    printf("  [8] session_bytes...\n");
    {
        strata::Geometry g;
        g.n_embd = 2560;
        g.n_head = 16;
        g.n_kv_head = 4;
        g.n_layers = 48;
        g.n_experts = 512;
        g.hc = 4;

        uint64_t bytes = strata::session_bytes(g, 1024, 10);
        CHECK(bytes > 0, "session_bytes > 0");
        printf("    session_bytes = %lu (%.1f MiB)\n",
               (unsigned long)bytes, bytes / (1024.0 * 1024.0));
    }

    // ═══════════════════════════════════════════════════════════════
    // 9. session_init + session_zero
    // ═══════════════════════════════════════════════════════════════
    printf("  [9] session_init + session_zero...\n");
    {
        strata::Geometry g;
        g.n_embd = 256;
        g.n_head = 4;
        g.n_kv_head = 2;
        g.n_layers = 48;
        g.n_experts = 8;
        g.hc = 2;

        uint64_t bytes = strata::session_bytes(g, 64, 2);
        CHECK(bytes > 0, "small session_bytes > 0");

        void* arena = nullptr;
        cudaError_t rc = cudaMalloc(&arena, bytes);
        CHECK(rc == cudaSuccess, "cudaMalloc arena");

        if (rc == cudaSuccess && arena) {
            strata::SessionData s;
            uint64_t used = strata::session_init(g, 64, 2, arena, s);
            CHECK(used > 0, "session_init used > 0");
            CHECK(used <= bytes, "used <= bytes");
            CHECK(s.R != nullptr, "R != null");
            CHECK(s.gdn_state != nullptr, "gdn_state != null");
            CHECK(s.max_cells == 64, "max_cells == 64");
            CHECK(s.k == 2, "k == 2");
            CHECK(s.ple_prev[0] == -1, "ple_prev[0] == -1");

            strata::session_zero(s, g, nullptr, stream);
            cudaStreamSynchronize(stream);

            float r_val = 999.0f;
            cudaMemcpy(&r_val, s.R, sizeof(float), cudaMemcpyDeviceToHost);
            CHECK(r_val == 0.0f, "R[0] == 0 after zero");

            cudaFree(arena);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // 10. session_token: validation
    // ═══════════════════════════════════════════════════════════════
    printf("  [10] session_token...\n");
    {
        strata::Geometry g;
        g.n_embd = 256;
        g.n_head = 4;
        g.n_kv_head = 2;
        g.n_layers = 48;
        g.n_experts = 8;
        g.hc = 2;

        uint64_t bytes = strata::session_bytes(g, 64, 2);
        void* arena = nullptr;
        cudaError_t rc = cudaMalloc(&arena, bytes);
        if (rc != cudaSuccess) {
            printf("    SKIP: cudaMalloc failed\n");
        } else {
            strata::SessionData s;
            strata::session_init(g, 64, 2, arena, s);
            strata::session_zero(s, g, nullptr, stream);
            cudaStreamSynchronize(stream);

            strata::WeightTable tables;
            std::string err;

            // null R should fail
            strata::SessionData s_bad;
            bool ok = strata::session_token(tables, g, 0, 0, s_bad, nullptr,
                                            stream, false, err);
            CHECK(!ok, "null R fails");

            // negative position should fail
            err.clear();
            ok = strata::session_token(tables, g, -1, 0, s, nullptr,
                                       stream, false, err);
            CHECK(!ok, "negative pos fails");

            // Valid call (stubs: orchestration runs, kernels are no-ops)
            err.clear();
            ok = strata::session_token(tables, g, 0, 0, s, nullptr,
                                       stream, false, err);
            CHECK(ok, "valid call succeeds");

            // With sync_every_layer
            err.clear();
            ok = strata::session_token(tables, g, 0, 0, s, nullptr,
                                       stream, true, err);
            CHECK(ok, "sync_every_layer succeeds");

            cudaFree(arena);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // 11. SessionGraphs: capture + replay structural test
    // ═══════════════════════════════════════════════════════════════
    printf("  [11] SessionGraphs capture/replay...\n");
    {
        strata::Geometry g;
        g.n_embd = 256;
        g.n_head = 4;
        g.n_kv_head = 2;
        g.n_layers = 48;
        g.n_experts = 8;
        g.hc = 2;

        uint64_t bytes = strata::session_bytes(g, 64, 2);
        void* arena = nullptr;
        cudaError_t rc = cudaMalloc(&arena, bytes);
        if (rc != cudaSuccess) {
            printf("    SKIP: cudaMalloc failed\n");
        } else {
            strata::SessionData s;
            strata::session_init(g, 64, 2, arena, s);
            strata::session_zero(s, g, nullptr, stream);
            cudaStreamSynchronize(stream);

            strata::WeightTable tables;
            strata::SessionGraphs gr;
            std::string err;

            // Capture (stubs produce 0-node graphs, which are handled gracefully)
            bool ok = strata::session_capture(tables, g, s, nullptr, gr, err);
            CHECK(ok, "session_capture succeeds");
            CHECK(gr.captured, "captured flag set");
            CHECK(gr.n == 48, "n == 48");
            CHECK(gr.execs != nullptr, "execs allocated");
            CHECK(gr.posts != nullptr, "posts allocated");

            // Replay (graphs are nullptr since stubs produce 0 nodes — replay skips them)
            err.clear();
            ok = strata::session_replay(g, 0, 0, s, gr, stream, err);
            CHECK(ok, "session_replay succeeds");
            CHECK(gr.calls_total == 1, "calls_total == 1");

            // Replay full
            err.clear();
            ok = strata::session_replay_full(g, 1, 0, s, gr, stream, err);
            CHECK(ok, "session_replay_full succeeds");
            CHECK(gr.calls_total == 2, "calls_total == 2");

            // Replay without capture should fail
            strata::SessionGraphs gr2;
            err.clear();
            ok = strata::session_replay(g, 0, 0, s, gr2, stream, err);
            CHECK(!ok, "replay without capture fails");

            strata::session_graphs_free(gr);
            CHECK(!gr.captured, "captured cleared after free");

            cudaFree(arena);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // 12. session_loop: validation
    // ═══════════════════════════════════════════════════════════════
    printf("  [12] session_loop...\n");
    {
        strata::Geometry g;
        g.n_embd = 256;
        g.n_head = 4;
        g.n_kv_head = 2;
        g.n_layers = 48;
        g.n_experts = 8;
        g.hc = 2;

        uint64_t bytes = strata::session_bytes(g, 64, 2);
        void* arena = nullptr;
        cudaError_t rc = cudaMalloc(&arena, bytes);
        if (rc != cudaSuccess) {
            printf("    SKIP: cudaMalloc failed\n");
        } else {
            strata::SessionData s;
            strata::session_init(g, 64, 2, arena, s);
            strata::session_zero(s, g, nullptr, stream);
            cudaStreamSynchronize(stream);

            strata::WeightTable tables;
            strata::SessionGraphs gr;
            std::string err;

            strata::session_capture(tables, g, s, nullptr, gr, err);

            // Null pool should fail
            err.clear();
            bool ok = strata::session_loop(g, 0, 0, s, gr,
                                           nullptr, nullptr, nullptr,
                                           false, stream, err);
            CHECK(!ok, "null pool fails");

            // Valid call with dummy pool
            auto dummy_pool = [](void*, const float*, const int32_t*,
                                 const float*, int64_t, int64_t, float*) {};
            err.clear();
            ok = strata::session_loop(g, 0, 0, s, gr,
                                      dummy_pool, nullptr, nullptr,
                                      false, stream, err);
            CHECK(ok, "valid loop succeeds");

            // Without captured graphs
            strata::SessionGraphs gr2;
            err.clear();
            ok = strata::session_loop(g, 0, 0, s, gr2,
                                      dummy_pool, nullptr, nullptr,
                                      false, stream, err);
            CHECK(!ok, "loop without capture fails");

            strata::session_graphs_free(gr);
            cudaFree(arena);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // 13. SessionLoopScratch: init + free
    // ═══════════════════════════════════════════════════════════════
    printf("  [13] SessionLoopScratch...\n");
    {
        strata::SessionLoopScratch scratch;
        std::string err;
        bool ok = scratch.init(4096, err);
        CHECK(ok, "init succeeds");
        CHECK(scratch.y_miss != nullptr, "y_miss allocated");
        CHECK(scratch.probe != nullptr, "probe created");
        CHECK(scratch.parts_bytes == 4096, "parts_bytes == 4096");

        scratch.free();
        CHECK(scratch.y_miss == nullptr, "y_miss null after free");
        CHECK(scratch.probe == nullptr, "probe null after free");
    }

    // ═══════════════════════════════════════════════════════════════
    // 14. TokenGraph + session_capture_token
    // ═══════════════════════════════════════════════════════════════
    printf("  [14] TokenGraph...\n");
    {
        strata::Geometry g;
        g.n_embd = 256;
        g.n_head = 4;
        g.n_kv_head = 2;
        g.n_layers = 48;
        g.n_experts = 8;
        g.hc = 2;

        uint64_t bytes = strata::session_bytes(g, 64, 2);
        void* arena = nullptr;
        cudaError_t rc = cudaMalloc(&arena, bytes);
        if (rc != cudaSuccess) {
            printf("    SKIP: cudaMalloc failed\n");
        } else {
            strata::SessionData s;
            strata::session_init(g, 64, 2, arena, s);

            strata::WeightTable tables;
            strata::TokenGraph tg;
            std::string err;

            bool ok = strata::session_capture_token(tables, g, s, nullptr,
                                                    nullptr, 4096, tg, err);
            CHECK(ok, "capture_token succeeds");
            CHECK(tg.captured, "captured");
            CHECK(tg.n_layers == 48, "n_layers == 48");

            // Idempotent
            ok = strata::session_capture_token(tables, g, s, nullptr,
                                               nullptr, 4096, tg, err);
            CHECK(ok, "capture_token idempotent");

            auto dummy_pool = [](void*, const float*, const int32_t*,
                                 const float*, int64_t, int64_t, float*) {};

            err.clear();
            ok = strata::session_run_token(g, 0, 0, s, tg,
                                           dummy_pool, nullptr, nullptr,
                                           stream, err);
            CHECK(ok, "run_token succeeds");
            CHECK(tg.calls == 1, "calls == 1");

            // Null pool fails
            err.clear();
            ok = strata::session_run_token(g, 0, 0, s, tg,
                                           nullptr, nullptr, nullptr,
                                           stream, err);
            CHECK(!ok, "null pool fails");

            strata::token_graph_free(tg);
            CHECK(!tg.captured, "captured cleared");

            cudaFree(arena);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // 15. CapturedGraph with real data (write kernel)
    // ═══════════════════════════════════════════════════════════════
    printf("  [15] CapturedGraph with write kernel...\n");
    {
        float* d_buf = nullptr;
        cudaMalloc(&d_buf, sizeof(float));
        if (d_buf) {
            cudaMemset(d_buf, 0, sizeof(float));

            strata::CapturedGraph cg;
            std::string err;
            cg.begin(stream, err);
            write_val_kernel<<<1, 1, 0, stream>>>(d_buf, 42.0f);
            cg.end(stream, err);

            CHECK(cg.valid(), "write kernel graph valid");

            // Launch and verify
            cg.launch(stream, err);
            cg.wait_ms(5000);

            float val = 0.0f;
            cudaMemcpy(&val, d_buf, sizeof(float), cudaMemcpyDeviceToHost);
            CHECK(val == 42.0f, "write kernel produced 42.0");

            // Replay produces same result
            cudaMemset(d_buf, 0, sizeof(float));
            cg.launch(stream, err);
            cg.wait_ms(5000);
            cudaMemcpy(&val, d_buf, sizeof(float), cudaMemcpyDeviceToHost);
            CHECK(val == 42.0f, "replay produces 42.0");

            cudaFree(d_buf);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    cudaStreamDestroy(stream);

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}

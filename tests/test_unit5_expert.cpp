// tests/test_unit5_expert.cpp — Smoke test for expert_source, expert_dispatch,
// expert_arena_loader (unit 5 of incremental build)
#include "strata/core/expert_source.hpp"
#include "strata/core/expert_cache.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { ++tests_passed; } \
    else { ++tests_failed; fprintf(stderr, "FAIL: %s\n", msg); } \
} while(0)

int main() {
    printf("=== strata unit 5 expert smoke test ===\n");

    // ════════════════════════════════════════════════════════════════
    // ExpertDispatch struct initialization
    // ════════════════════════════════════════════════════════════════
    {
        strata::ExpertDispatch d;
        CHECK(d.pool == nullptr, "ExpertDispatch pool default null");
        CHECK(d.src == nullptr, "ExpertDispatch src default null");
        CHECK(d.n_expert == strata::kernels::cpu::NE, "ExpertDispatch n_expert == NE");
        CHECK(d.layers == 0, "ExpertDispatch layers default 0");
        CHECK(d.experts == 0, "ExpertDispatch experts default 0");
        CHECK(d.missing == 0, "ExpertDispatch missing default 0");
        CHECK(d.failed == false, "ExpertDispatch not failed initially");
        CHECK(!d.hits_ready(), "ExpertDispatch hits not ready initially");
        CHECK(d.decided == false, "ExpertDispatch decided default false");
    }

    // ════════════════════════════════════════════════════════════════
    // GpuPlanSink struct
    // ════════════════════════════════════════════════════════════════
    {
        strata::GpuPlanSink plan;
        CHECK(plan.counts == nullptr, "GpuPlanSink counts default null");
        CHECK(plan.cap == 0, "GpuPlanSink cap default 0");
        CHECK(plan.pcie_mode == 0, "GpuPlanSink pcie_mode default 0");
        CHECK(plan.staging == 0, "GpuPlanSink staging default 0");
    }

    // ════════════════════════════════════════════════════════════════
    // HitPhase enum
    // ════════════════════════════════════════════════════════════════
    {
        CHECK(static_cast<int>(strata::HitPhase::Prepare) == 0, "HitPhase::Prepare == 0");
        CHECK(static_cast<int>(strata::HitPhase::Compute) == 1, "HitPhase::Compute == 1");
        CHECK(static_cast<int>(strata::HitPhase::Combine) == 2, "HitPhase::Combine == 2");
    }

    // ════════════════════════════════════════════════════════════════
    // ExpertJob / ExpertJobMulti struct defaults
    // ════════════════════════════════════════════════════════════════
    {
        strata::kernels::cpu::ExpertJob job;
        CHECK(job.blob == nullptr, "ExpertJob blob default null");
        CHECK(job.layer == 0, "ExpertJob layer default 0");
        CHECK(job.out == nullptr, "ExpertJob out default null");

        strata::kernels::cpu::ExpertJobMulti mjob;
        CHECK(mjob.blob == nullptr, "ExpertJobMulti blob default null");
        CHECK(mjob.n_tok == 0, "ExpertJobMulti n_tok default 0");
    }

    // ════════════════════════════════════════════════════════════════
    // ActQ struct
    // ════════════════════════════════════════════════════════════════
    {
        strata::kernels::cpu::ActQ act;
        CHECK(act.nchunks == 0, "ActQ nchunks default 0");
        CHECK(act.q[0] == 0, "ActQ q default zero");
        CHECK(act.scale[0] == 0.0f, "ActQ scale default zero");
    }

    // ════════════════════════════════════════════════════════════════
    // FileExpertSource: open with no file → error
    // ════════════════════════════════════════════════════════════════
    {
        strata::FileExpertSource src;
        CHECK(!src.mapped(), "FileExpertSource not mapped initially");
        CHECK(src.blobs() == 0, "FileExpertSource 0 blobs initially");
        CHECK(src.reads() == 0, "FileExpertSource 0 reads initially");

        std::string err;
        bool ok = src.open("/nonexistent/path", 48, 512, err);
        CHECK(!ok, "FileExpertSource open fails on nonexistent path");
        CHECK(!err.empty(), "FileExpertSource open sets error message");
        CHECK(!src.mapped(), "FileExpertSource not mapped after failed open");
    }

    // ════════════════════════════════════════════════════════════════
    // FileExpertSource: invalid geometry
    // ════════════════════════════════════════════════════════════════
    {
        strata::FileExpertSource src;
        std::string err;
        bool ok = src.open("/tmp", 0, 0, err);
        CHECK(!ok, "FileExpertSource rejects zero geometry");
    }

    // ════════════════════════════════════════════════════════════════
    // FileExpertSource: blob() on unmapped returns nullptr
    // ════════════════════════════════════════════════════════════════
    {
        strata::FileExpertSource src;
        CHECK(src.blob(0, 0) == nullptr, "blob() on unmapped returns null");
        CHECK(src.blob(-1, 0) == nullptr, "blob() negative layer returns null");
    }

    // ════════════════════════════════════════════════════════════════
    // expert_pool_dispatch: null user → no crash
    // ════════════════════════════════════════════════════════════════
    {
        float x[2560] = {};
        int32_t ids[] = {0};
        float w[] = {1.0f};
        float out[2560] = {};
        // Should not crash
        strata::expert_pool_dispatch(nullptr, x, ids, w, 2560, 1, out);
        CHECK(true, "expert_pool_dispatch with null user no crash");
    }

    // ════════════════════════════════════════════════════════════════
    // expert_pool_dispatch: no source → marks failed
    // ════════════════════════════════════════════════════════════════
    {
        strata::ExpertDispatch d;
        float x[2560] = {};
        int32_t ids[] = {0};
        float w[] = {1.0f};
        float out[2560] = {};
        strata::expert_pool_dispatch(&d, x, ids, w, 2560, 1, out);
        CHECK(d.failed, "dispatch with no source sets failed");
    }

    // ════════════════════════════════════════════════════════════════
    // expert_pool_dispatch: wrong n_embd → marks failed
    // ════════════════════════════════════════════════════════════════
    {
        strata::ExpertDispatch d;
        strata::FileExpertSource src;
        d.src = &src;
        float x[128] = {};
        int32_t ids[] = {0};
        float w[] = {1.0f};
        float out[128] = {};
        strata::expert_pool_dispatch(&d, x, ids, w, 128, 1, out);
        CHECK(d.failed, "dispatch with wrong n_embd sets failed");
    }

    // ════════════════════════════════════════════════════════════════
    // expert_pool_dispatch_multi: no source → marks failed
    // ════════════════════════════════════════════════════════════════
    {
        strata::ExpertDispatch d;
        float x[2560] = {};
        int32_t ids[] = {0};
        float out[2560] = {};
        strata::expert_pool_dispatch_multi(d, x, ids, 1, 1, out);
        CHECK(d.failed, "dispatch_multi with no source sets failed");
    }

    // ════════════════════════════════════════════════════════════════
    // expert_hit_run: null user → no crash
    // ════════════════════════════════════════════════════════════════
    {
        int32_t ids[] = {0};
        strata::expert_hit_run(nullptr, nullptr, strata::HitPhase::Prepare, ids, 1);
        CHECK(true, "expert_hit_run with null user no crash");
    }

    // ════════════════════════════════════════════════════════════════
    // expert_hit_run: hits not ready → no crash
    // ════════════════════════════════════════════════════════════════
    {
        strata::ExpertDispatch d;
        int32_t ids[] = {0};
        strata::expert_hit_run(&d, nullptr, strata::HitPhase::Prepare, ids, 1);
        CHECK(!d.decided, "hit_run with no cache → not decided");
    }

    // ════════════════════════════════════════════════════════════════
    // ArenaExpertSource: default state
    // ════════════════════════════════════════════════════════════════
    {
        strata::ArenaExpertSource src;
        CHECK(!src.mapped(), "ArenaExpertSource not mapped initially");
        CHECK(src.blobs() == 0, "ArenaExpertSource 0 blobs initially");
        CHECK(src.blob(0, 0) == nullptr, "ArenaExpertSource blob returns null unmapped");
        CHECK(src.device_alias(0, 0) == nullptr, "ArenaExpertSource device_alias null unmapped");
        CHECK(!src.pinned(0, 0), "ArenaExpertSource not pinned initially");
    }

    // ════════════════════════════════════════════════════════════════
    // NE constant
    // ════════════════════════════════════════════════════════════════
    {
        CHECK(strata::kernels::cpu::NE == 512, "NE constant == 512");
    }

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}


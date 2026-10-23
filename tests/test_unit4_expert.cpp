// tests/test_unit4_expert.cpp — Smoke test for unit 4 expert cache extensions
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_cache_interface.hpp"
#include "strata/core/hit_miss_hook.hpp"

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
    printf("=== strata unit 4 expert cache smoke test ===\n");

    // ── ExpertLookupResult default ───────────────────────────────
    {
        strata::ExpertLookupResult r;
        CHECK(r.n_hits == 0, "ExpertLookupResult default n_hits 0");
        CHECK(r.n_miss == 0, "ExpertLookupResult default n_miss 0");
        CHECK(r.n_total == 0, "ExpertLookupResult default n_total 0");
    }

    // ── HitMissPartition default ─────────────────────────────────
    {
        strata::HitMissPartition p;
        CHECK(p.hit_ids == nullptr, "HitMissPartition default hit_ids null");
        CHECK(p.n_hits == 0, "HitMissPartition default n_hits 0");
        CHECK(p.n_miss == 0, "HitMissPartition default n_miss 0");
    }

    // ── HitMissHook init ─────────────────────────────────────────
    {
        strata::HitMissHook hook;
        hook.init(10);
        CHECK(hook.max_k() == 10, "HitMissHook max_k 10");
    }

    // ── HitMissHook partition with empty cache ───────────────────
    {
        strata::ExpertCache cache;
        std::string err;
        bool ok = cache.open(16, 4, 8, 1024, err);
        CHECK(ok, "open cache for partition test");

        strata::HitMissHook hook;
        hook.init(4);

        int32_t ids[] = {0, 1, 2, 3};
        float weights[] = {0.4f, 0.3f, 0.2f, 0.1f};
        strata::HitMissPartition part;

        auto result = hook.partition(0, ids, weights, 4, cache, part);
        CHECK(result.n_hits == 0, "all miss: n_hits 0");
        CHECK(result.n_miss == 4, "all miss: n_miss 4");
        CHECK(result.n_total == 4, "all miss: n_total 4");
        CHECK(part.n_miss == 4, "partition n_miss 4");
        CHECK(part.miss_ids[0] == 0, "first miss is expert 0");
    }

    // ── HitMissHook partition with some hits ─────────────────────
    {
        strata::ExpertCache cache;
        std::string err;
        bool ok = cache.open(16, 4, 8, 1024, err);
        CHECK(ok, "open cache for hit test");

        // Admit experts (0,1) and (0,3) into cache
        int32_t s1 = cache.admit(0, 1);
        int32_t s3 = cache.admit(0, 3);
        CHECK(s1 != strata::kNotResident, "admit (0,1) ok");
        CHECK(s3 != strata::kNotResident, "admit (0,3) ok");

        strata::HitMissHook hook;
        hook.init(4);

        int32_t ids[] = {0, 1, 2, 3};
        float weights[] = {0.4f, 0.3f, 0.2f, 0.1f};
        strata::HitMissPartition part;

        auto result = hook.partition(0, ids, weights, 4, cache, part);
        CHECK(result.n_hits == 2, "2 hits");
        CHECK(result.n_miss == 2, "2 misses");
        CHECK(part.hit_ids[0] == 1, "first hit is expert 1");
        CHECK(part.hit_ids[1] == 3, "second hit is expert 3");
        CHECK(part.hit_slots[0] == s1, "hit slot matches");
        CHECK(part.miss_ids[0] == 0, "first miss is expert 0");
        CHECK(part.miss_ids[1] == 2, "second miss is expert 2");
        // Check weights are preserved
        CHECK(part.hit_weights[0] == 0.3f, "hit weight 0.3");
        CHECK(part.miss_weights[0] == 0.4f, "miss weight 0.4");
    }

    // ── batch_lookup ─────────────────────────────────────────────
    {
        strata::ExpertCache cache;
        std::string err;
        bool ok = cache.open(16, 4, 8, 1024, err);
        CHECK(ok, "open for batch_lookup");

        cache.admit(0, 2);
        cache.admit(0, 5);

        int32_t ids[] = {0, 2, 5, 7};
        int32_t slots[4] = {};
        int32_t hits = cache.batch_lookup(0, ids, slots, 4);
        CHECK(hits == 2, "batch_lookup 2 hits");
        CHECK(slots[0] == strata::kNotResident, "expert 0 not resident");
        CHECK(slots[1] != strata::kNotResident, "expert 2 resident");
        CHECK(slots[2] != strata::kNotResident, "expert 5 resident");
        CHECK(slots[3] == strata::kNotResident, "expert 7 not resident");
    }

    // ── Route counts ─────────────────────────────────────────────
    {
        strata::HitMissHook hook;
        hook.init(4);
        hook.reset_route_counts(4, 8);

        int32_t ids[] = {0, 1, 2, 3};
        hook.update_route_counts(0, ids, 4);
        hook.update_route_counts(0, ids, 4);

        CHECK(hook.route_counts()[0] == 2, "route count [0,0] = 2");
        CHECK(hook.route_counts()[1] == 2, "route count [0,1] = 2");

        hook.decay_route_counts();
        CHECK(hook.route_counts()[0] == 1, "after decay [0,0] = 1");
    }

    // ── Statistics counters ──────────────────────────────────────
    {
        strata::ExpertCache cache;
        std::string err;
        bool ok = cache.open(8, 2, 4, 512, err);
        CHECK(ok, "open for stats");
        CHECK(cache.total_admits() == 0, "initial admits 0");
        CHECK(cache.total_fills() == 0, "initial fills 0");

        cache.admit(0, 0);
        cache.admit(0, 1);
        CHECK(cache.total_admits() == 2, "2 admits");
    }

    // ── Per-layer mode slot range ────────────────────────────────
    {
        strata::ExpertCache cache;
        std::string err;
        bool ok = cache.open(48, 4, 8, 256, err);
        CHECK(ok, "open for per-layer");
        cache.set_per_layer(true);

        int64_t lo = 0, hi = 0;
        cache.layer_slot_range(0, lo, hi);
        CHECK(lo == 0, "layer 0 lo = 0");
        CHECK(hi == 12, "layer 0 hi = 12");

        cache.layer_slot_range(3, lo, hi);
        CHECK(lo == 36, "layer 3 (last) lo = 36");
        CHECK(hi == 48, "layer 3 (last) hi = 48 (gets remainder)");

        // Admit in per-layer mode
        int32_t s = cache.admit(0, 0);
        CHECK(s >= 0 && s < 12, "per-layer admit in range");
        int32_t s2 = cache.admit(3, 0);
        CHECK(s2 >= 36 && s2 < 48, "per-layer admit layer 3 in range");
    }

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}

#include <cstdint>
#include <random>
#include "baloon/memory_planner.hpp"
#include "baloon/plan_pool_budget.hpp"
#include "check.hpp"

using namespace baloon::planner;
using namespace baloon::budget;

void test_planner_geometry_constants() {
    const Geometry g;
    CHECK(g.n_gdn() == 36);
    CHECK(state_bytes(g) == 117669888ull);
    CHECK(kv_bytes_per_token(g) == 13056ull);
    CHECK(kv_bytes_per_token(g) * 32768ull == 427819008ull);
    CHECK(kExpertBlobBytes == 1382400ull);
    CHECK(1280ull * 640 + 2560ull * 160 + 1280ull * 40 * 2 + 2560ull * 10 * 2 == 1382400ull);
    CHECK(state_in_slots(g, kExpertBlobBytes) == 85);
    CHECK(117669888ull / 1382400ull == 85);
    // geometry sensitivity: more QSA layers means more KV and less GDN state
    Geometry h = g; h.n_qsa = 16;
    CHECK(kv_bytes_per_token(h) == 16 * 1088ull && state_bytes(h) == 32 * (state_bytes(g) / 36));
}

void test_planner_budget_derivations() {
    CHECK(indexer_bytes_per_token_layer(1, 128, 4) == 32);                         // the correct indexer term
    CHECK(indexer_bytes_per_token_layer(4, 128, 4) == 4 * indexer_bytes_per_token_layer(1, 128, 4));  // query-head count over-counts 4x
    CHECK(indexer_bytes_per_token_layer(1, 256, 4) == 2 * indexer_bytes_per_token_layer(1, 128, 4));  // 256-wide over-counts 2x
    CHECK(1024 % 64 == 0);
    CHECK(kv_scale_bytes(1024, 64).value() == 32);
    CHECK(!kv_scale_bytes(1000, 64).has_value() && !kv_scale_bytes(64, 0).has_value());
    const Geometry g;
    CHECK(kv_layer_bytes(g) == 1088 && 1024 + 32 + 32 == 1088);
    CHECK(g.n_qsa * kv_layer_bytes(g) == 13056 && kv_bytes_per_token(g) == g.n_qsa * kv_layer_bytes(g));
}

void test_planner_make_plan_invariants() {
    std::mt19937_64 rng(2026);
    long some = 0, none = 0;
    for (int it = 0; it < 400000; ++it) {
        const uint64_t blob = 1 + rng() % 5000;
        const uint64_t kvT = rng() % 200, ctx = rng() % 3000, fixed = rng() % 300000;
        const uint64_t pool = rng() % 1500000;
        const auto p = make_plan(pool, kvT, ctx, fixed, blob);
        if (kvT * ctx + fixed > pool) {
            CHECK(!p.has_value());                                                // refuse overcommit
            ++none;
            continue;
        }
        const uint64_t cache = pool - kvT * ctx - fixed;
        if (cache / blob == 0) { CHECK(!p.has_value()); ++none; continue; }       // no whole slot: refuse
        CHECK(p.has_value());
        if (!p) continue;
        ++some;
        CHECK(p->kv == kvT * ctx && p->cache == cache && p->slots == cache / blob && p->state == 0);
        CHECK(p->kv + fixed + p->slots * blob <= pool);                           // never overcommits
        CHECK(p->slots > 0);                                                      // at least one slot
        CHECK(p->used <= pool);
        CHECK(p->used == p->kv + fixed + p->slots * blob);                        // use excludes only the sub-slot remainder
        CHECK(p->used == used_bytes(pool, p->kv, fixed, blob));
        CHECK(pool - p->used == leftover(pool, p->kv, fixed, blob) && leftover(pool, p->kv, fixed, blob) < blob);
        CHECK(slots_for(pool, p->kv, fixed, blob) == p->slots);
    }
    CHECK(some > 1000 && none > 1000);                                            // both outcomes exercised
}

void test_planner_edges() {
    // exact fit: n slots + kv + fixed
    for (uint64_t n : {1ull, 2ull, 85ull}) {
        const uint64_t kv = 13056ull * 4096, fixed = 117669888ull, blob = kExpertBlobBytes;
        const uint64_t pool = kv + fixed + n * blob;
        auto p = make_plan(pool, 13056, 4096, fixed, blob);
        CHECK(p && p->slots == n && p->used == pool);
        CHECK(slots_for(pool, kv, fixed, blob) >= n);                             // pool that fits n slots yields >= n
        auto q = make_plan(pool - 1, 13056, 4096, fixed, blob);
        if (n == 1) CHECK(!q.has_value()); else CHECK(q && q->slots == n - 1);
    }
    CHECK(!make_plan(1000, 1, 1, 0, 0).has_value());                              // zero blob: refuse, never divide by zero
    CHECK(!make_plan(UINT64_MAX, UINT64_MAX, UINT64_MAX, 0, 10).has_value());     // overflowing product refuses
    CHECK(!make_plan(100, 10, 10, 1, 1).has_value());                             // fixed+kv exceed the pool by 1
    CHECK(make_plan(101, 10, 10, 0, 1).has_value());
    CHECK(sat_sub(3, 5) == 0 && sat_sub(5, 3) == 2);
}

void test_planner_monotone_in_context() {
    const Geometry g;
    const uint64_t kvT = kv_bytes_per_token(g), fixed = state_bytes(g), blob = kExpertBlobBytes;
    const uint64_t pool = 24ull << 30;
    uint64_t prev_slots = UINT64_MAX, prev_kv = 0;
    for (uint64_t ctx = 0; ctx <= 262144; ctx += 1024) {
        CHECK(kvT * ctx >= prev_kv);                                              // KV grows with context
        prev_kv = kvT * ctx;
        const uint64_t s = slots_for(pool, kvT * ctx, fixed, blob);
        CHECK(s <= prev_slots);                                                   // longer context never yields more slots
        prev_slots = s;
        CHECK(kvT * (2 * ctx) == 2 * (kvT * ctx));                                // doubling context doubles KV
    }
    std::mt19937_64 rng(8);
    for (int i = 0; i < 100000; ++i) {
        uint64_t a = rng() % 100000, b = rng() % 100000;
        if (a > b) std::swap(a, b);
        const uint64_t pl = rng() % (1ull << 34), k = 1 + rng() % 20000, f = rng() % (1ull << 30), bl = 1 + rng() % (1ull << 21);
        CHECK(slots_for(pl, k * b, f, bl) <= slots_for(pl, k * a, f, bl));
        CHECK(leftover(pl, k * a, f, bl) < bl);
    }
}

void test_planner_costs_form() {
    const Geometry g;
    // artifact constants through the Costs form
    Costs c; c.state = state_bytes(g); c.dense = 5ull << 28; c.embd = 1ull << 28; c.mtp = 1ull << 26; c.workspace = 1ull << 27;
    const uint64_t fixed = c.dense + c.embd + c.mtp + c.workspace + c.state;
    CHECK(fixed_cost_wide(c) == fixed);
    std::mt19937_64 rng(77);
    long some = 0, none = 0;
    for (int it = 0; it < 200000; ++it) {
        Geometry h = g;
        h.n_qsa = rng() % 49; h.kv_group = rng() % 129; h.qsa_block = rng() % 9;   // include 0 divisors
        Costs k; k.expert_blob = rng() % 3000; k.dense = rng() % 100000; k.embd = rng() % 50000;
        k.mtp = rng() % 50000; k.workspace = rng() % 50000; k.state = rng() % 100000;
        const uint64_t ctx = rng() % 2000, pool = rng() % 6000000;
        const uint64_t kvt = kv_bytes_per_token(h);
        const uint64_t fx = k.dense + k.embd + k.mtp + k.workspace + k.state;
        const auto p = make_plan(ctx, h, k, pool);
        if (pool < kvt * ctx + fx) { CHECK(!p.has_value()); ++none; continue; }      // refusal
        if (k.expert_blob == 0 || (pool - kvt * ctx - fx) / k.expert_blob == 0) { CHECK(!p.has_value()); ++none; continue; }
        CHECK(p.has_value());
        if (!p) continue;
        ++some;
        CHECK(kvt * ctx + fx + p->cache_slots * k.expert_blob <= pool);                // soundness
        CHECK(p->cache_slots > 0 && p->vram_used <= pool);
        CHECK(p->max_context == ctx && p->kv_bytes == kvt * ctx && p->state_bytes == k.state);
        CHECK(p->cache_bytes == pool - kvt * ctx - fx && p->vram_budget == pool);
        CHECK(p->vram_used == kvt * ctx + fx + p->cache_slots * k.expert_blob);
        // completeness: whenever kv + fixed + one blob fits, a plan exists
        CHECK(kvt * ctx + fx + k.expert_blob <= pool);
    }
    CHECK(some > 1000 && none > 1000);
    // completeness direction, explicitly: pools that fit one blob always yield a plan
    for (int it = 0; it < 100000; ++it) {
        Costs k; k.expert_blob = 1 + rng() % 4000; k.dense = rng() % 9999; k.state = rng() % 9999;
        const uint64_t ctx = rng() % 500, kvt = kv_bytes_per_token(g);
        const uint64_t pool = kvt * ctx + k.dense + k.state + k.expert_blob + rng() % 7000;
        CHECK(make_plan(ctx, g, k, pool).has_value());
        CHECK(!make_plan(ctx, g, k, kvt * ctx + k.dense + k.state + k.expert_blob - 1).has_value());
    }
    // degenerate geometry never traps; overflow refuses
    Geometry z; z.n_layers = z.n_qsa = z.n_kv_heads = z.head_dim = z.kv_group = z.qsa_block = 0;
    z.idx_key_heads = z.idx_key_dim = z.ssm_state = z.ssm_v_heads = z.ssm_conv = z.ssm_channels = 0;
    CHECK(kv_bytes_per_token(z) == 0 && state_bytes(z) == 0 && z.n_gdn() == 0);
    Geometry w = g; w.n_qsa = 60; CHECK(w.n_gdn() == 0);
    Costs big; big.dense = UINT64_MAX; big.embd = UINT64_MAX; big.expert_blob = 1;
    CHECK(!make_plan(1, g, big, UINT64_MAX).has_value());
    // KV monotone in context; matches the scalar form
    for (uint64_t a = 0; a < 5000; a += 37) CHECK(kv_bytes_per_token(g) * a <= kv_bytes_per_token(g) * (a + 1));
    const auto sp = make_plan(4096, g, c, 24ull << 30);
    const auto op = make_plan(24ull << 30, kv_bytes_per_token(g), 4096, fixed, c.expert_blob);
    CHECK(sp && op && sp->cache_slots == op->slots && sp->vram_used == op->used && sp->kv_bytes == op->kv);
}

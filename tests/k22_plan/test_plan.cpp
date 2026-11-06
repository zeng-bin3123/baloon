// Invariants of the plan front end (include/baloon/plan.hpp): soundness/completeness of make_plan against an independent
// 128-bit oracle, refusal reasons, no wraparound, vram_pool_bytes bounds, printable forms.
#include <cstdint>
#include <cstdio>
#include <string>
#include "baloon/plan.hpp"
#include "check.hpp"
using namespace baloon::plan;
typedef unsigned __int128 u128;

static const uint64_t BIG = ~0ull;

static void test_constants() {
    const Geometry g;
    CHECK(kv_bytes_per_token(g) == 13056ull);
    CHECK(state_bytes(g) == 117669888ull);
    Costs c;
    CHECK(c.expert_blob == 1382400ull);
    CHECK(g.n_gdn() == 36);
    Geometry h; h.n_qsa = 60;                       // more QSA layers than layers: no GDN layers, never wraps
    CHECK(h.n_gdn() == 0 && state_bytes(h) == 0);
    h = g; h.n_qsa = 16;
    CHECK(h.n_gdn() == 32 && kv_bytes_per_token(h) == 16 * 1088ull && state_bytes(h) == 32 * (state_bytes(g) / 36));
}

static void test_known_plan() {
    const Geometry g; Costs c; c.dense = 1000; c.embd = 2000; c.mtp = 3000; c.workspace = 4000; c.state = state_bytes(g);
    const uint64_t pool = 1ull << 30, ctx = 4096;
    const Result r = make_plan(ctx, g, c, pool);
    CHECK(std::holds_alternative<Plan>(r));
    if (!std::holds_alternative<Plan>(r)) return;
    const Plan& p = std::get<Plan>(r);
    const uint64_t kv = 13056ull * ctx, fixed = 1000 + 2000 + 3000 + 4000 + c.state;
    CHECK(p.max_context == ctx && p.kv_bytes == kv && p.state_bytes == c.state && p.vram_budget == pool);
    CHECK(p.cache_bytes == pool - kv - fixed);
    CHECK(p.cache_slots == (pool - kv - fixed) / c.expert_blob);
    CHECK(p.vram_used == pool - (pool - kv - fixed) % c.expert_blob);
    CHECK(p.vram_used == kv + fixed + p.cache_slots * c.expert_blob);
}

static void test_boundaries() {
    const Geometry g; Costs c; c.expert_blob = 1000; c.dense = 500;
    const uint64_t kv = kv_bytes_per_token(g) * 10;
    // exactly one slot fits
    Result r = make_plan(10, g, c, kv + 500 + 1000);
    CHECK(std::holds_alternative<Plan>(r));
    if (std::holds_alternative<Plan>(r)) CHECK(std::get<Plan>(r).cache_slots == 1 && std::get<Plan>(r).vram_used == kv + 1500);
    // one byte short: refuses with NoWholeSlot
    r = make_plan(10, g, c, kv + 500 + 999);
    CHECK(std::holds_alternative<DoesNotClose>(r));
    if (std::holds_alternative<DoesNotClose>(r)) CHECK(std::get<DoesNotClose>(r).reason == Reason::NoWholeSlot);
    // fixed + kv exactly the pool: no slot left, still NoWholeSlot
    r = make_plan(10, g, c, kv + 500);
    CHECK(std::holds_alternative<DoesNotClose>(r) && std::get<DoesNotClose>(r).reason == Reason::NoWholeSlot);
    // kv + fixed over the pool
    r = make_plan(10, g, c, kv + 499);
    CHECK(std::holds_alternative<DoesNotClose>(r) && std::get<DoesNotClose>(r).reason == Reason::FixedExceedsPool);
    // zero blob
    c.expert_blob = 0;
    r = make_plan(10, g, c, BIG);
    CHECK(std::holds_alternative<DoesNotClose>(r) && std::get<DoesNotClose>(r).reason == Reason::ZeroExpertBlob);
}

static void test_no_wrap() {
    const Geometry g; Costs c;
    // kv overflows 64 bits: must refuse, never wrap into a small number
    Result r = make_plan(BIG, g, c, BIG);
    CHECK(std::holds_alternative<DoesNotClose>(r));
    if (std::holds_alternative<DoesNotClose>(r)) CHECK(std::get<DoesNotClose>(r).kv_bytes == BIG);
    // fixed costs whose sum overflows 64 bits
    Costs w; w.dense = BIG; w.embd = BIG; w.expert_blob = 1;
    r = make_plan(0, g, w, BIG);
    CHECK(std::holds_alternative<DoesNotClose>(r));
    if (std::holds_alternative<DoesNotClose>(r)) CHECK(std::get<DoesNotClose>(r).fixed == BIG && std::get<DoesNotClose>(r).reason == Reason::FixedExceedsPool);
}

static void test_randomized_oracle() {
    th::Rng rng(22);
    long some = 0, none = 0;
    for (int it = 0; it < 300000; ++it) {
        Geometry g;
        g.n_qsa = rng.below(20); g.n_layers = rng.below(60);
        Costs c;
        c.expert_blob = rng.below(5) == 0 ? rng.below(4) : 1 + rng.below(100000);
        c.dense = rng.below(100000); c.embd = rng.below(100000); c.mtp = rng.below(100000);
        c.workspace = rng.below(100000); c.state = rng.below(100000);
        const uint64_t ctx = rng.below(2000), pool = (uint64_t)rng.below(1u << 30);
        const u128 kv = (u128)kv_bytes_per_token(g) * ctx;
        const u128 fixed = (u128)c.dense + c.embd + c.mtp + c.workspace + c.state;
        const bool fits = c.expert_blob != 0 && kv + fixed + c.expert_blob <= (u128)pool;
        const Result r = make_plan(ctx, g, c, pool);
        CHECK(std::holds_alternative<Plan>(r) == fits);            // sound and complete
        if (fits) {
            ++some;
            const Plan& p = std::get<Plan>(r);
            CHECK(p.cache_slots >= 1);
            CHECK((u128)p.kv_bytes + fixed + (u128)p.cache_slots * c.expert_blob <= (u128)p.vram_used);
            CHECK(p.vram_used <= p.vram_budget && p.vram_budget == pool);
            CHECK(p.cache_bytes == pool - (uint64_t)kv - (uint64_t)fixed);
            CHECK(p.vram_budget - p.vram_used < c.expert_blob);    // only a sub-slot remainder is unused
            CHECK(p.state_bytes == c.state && p.max_context == ctx);
        } else {
            ++none;
            const DoesNotClose& d = std::get<DoesNotClose>(r);
            CHECK(d.pool == pool && d.blob == c.expert_blob);
            if (c.expert_blob == 0) CHECK(d.reason == Reason::ZeroExpertBlob);
            else if (kv + fixed > (u128)pool) CHECK(d.reason == Reason::FixedExceedsPool);
            else CHECK(d.reason == Reason::NoWholeSlot);
        }
    }
    CHECK(some > 1000 && none > 1000);   // both outcomes are exercised
}

static void test_more_context_never_more_slots() {
    const Geometry g; Costs c; c.dense = 12345;
    const uint64_t pool = 3ull << 30;
    uint64_t prev = BIG;
    for (uint64_t ctx = 0; ctx <= 400000; ctx += 997) {
        const Result r = make_plan(ctx, g, c, pool);
        const uint64_t s = std::holds_alternative<Plan>(r) ? std::get<Plan>(r).cache_slots : 0;
        CHECK(s <= prev);
        prev = s;
    }
    CHECK(prev == 0);
}

static void test_vram_pool() {
    CHECK(vram_pool_bytes(100, 30) == 70);
    CHECK(vram_pool_bytes(100, 100) == 0);
    CHECK(vram_pool_bytes(100, 101) == 0);        // truncating, never wraps
    CHECK(vram_pool_bytes(0, 5) == 0);
    CHECK(vram_pool_bytes(BIG, 0) == BIG);
    CHECK(vram_pool_bytes(5, BIG) == 0 && vram_pool_bytes(BIG, BIG) == 0 && vram_pool_bytes(0, BIG) == 0);
    CHECK(vram_pool_bytes(BIG, BIG - 1) == 1);
    th::Rng rng(7);
    for (int i = 0; i < 20000; ++i) {
        const uint64_t d = rng.next(), r = rng.next();
        const uint64_t p = vram_pool_bytes(d, r);
        CHECK(p <= d);
        CHECK(r >= d ? p == 0 : (u128)p + r == d);
    }
}

static void test_to_string() {
    const Geometry g; Costs c; c.dense = 10;
    const Result ok = make_plan(1, g, c, 10000000);
    const std::string s = to_string(ok);
    CHECK(s.find("plan: context=1 ") == 0);
    CHECK(s.find("kv=13056 ") != std::string::npos);
    CHECK(s.find("slots=7 ") != std::string::npos);
    CHECK(s.find("used=") != std::string::npos && s.find("/10000000") != std::string::npos);
    const std::string e = to_string(make_plan(1, g, c, 5));
    CHECK(e.find("does not close") != std::string::npos && e.find("exceed the pool") != std::string::npos && e.find("pool=5 ") != std::string::npos);
    Costs z; z.expert_blob = 0;
    CHECK(to_string(make_plan(1, g, z, 5)).find("zero") != std::string::npos);
    c.expert_blob = 100000000;
    CHECK(to_string(make_plan(1, g, c, 100000000)).find("no whole expert slot") != std::string::npos);
    CHECK(std::string(to_string(Reason::NoWholeSlot)) != to_string(Reason::FixedExceedsPool));
}

int main() {
    test_constants(); test_known_plan(); test_boundaries(); test_no_wrap(); test_randomized_oracle();
    test_more_context_never_more_slots(); test_vram_pool(); test_to_string();
    std::printf("checks: %ld\n", th::checks());
    if (th::fails()) { std::printf("UNIT 22 TESTS FAILED (%ld)\n", th::fails()); return 1; }
    std::printf("UNIT 22 TESTS PASSED\n");
    return 0;
}

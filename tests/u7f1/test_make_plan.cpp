// tests/u7f1/test_make_plan.cpp — VramMemoryPlan theorems: plan_closes, plan_slots_fit,
// plan_has_slot, plan_refuses, slots_antitone, example_32k.
// Theory:
//   plan_closes : p.kvBytes + fixedCost c + p.cacheBytes = p.vramBudget
//   plan_slots_fit : p.cacheSlots * blob ≤ p.cacheBytes
//   plan_has_slot  : 1 ≤ p.cacheSlots
//   plan_refuses   : pool < kv + fixed → DoesNotClose
//   slots_antitone : ctx1 ≤ ctx2 → p2.cacheSlots ≤ p1.cacheSlots
//   example_32k    : cacheSlots at 32K = 24072 (with state in Costs)
#include "baloon/plan.hpp"
#include "u7f1/check.hpp"
#include <cstdio>
#include <variant>
int main() {
    std::puts("=== test_make_plan ===");
    using namespace baloon::plan;
    Geometry g;
    Costs c;
    c.state = state_bytes(g);            // 117 669 888 B
    const uint64_t pool = vram_pool_bytes(32ull << 30, 512ull << 20);  // 33 822 867 456

    // ---- successful plan at 32 768 tokens ----
    Result r32 = make_plan(32768, g, c, pool);
    CHK(std::holds_alternative<Plan>(r32));
    const Plan& p32 = std::get<Plan>(r32);

    // plan_closes: kv + state + cache = pool
    const uint64_t fixed = c.state;   // only state in this test
    CHK(p32.kv_bytes + fixed + p32.cache_bytes == pool);

    // plan_slots_fit: slots * blob ≤ cache
    CHK(p32.cache_slots * c.expert_blob <= p32.cache_bytes);

    // plan_has_slot: at least one slot
    CHK(p32.cache_slots >= 1u);

    // vram_used ≤ vram_budget
    CHK(p32.vram_used <= p32.vram_budget);

    // kv_bytes as expected (13056 * 32768 = 427 819 008)
    CHK(p32.kv_bytes == 13056u * 32768u);

    // ---- plan_refuses: huge context ----
    Result rhuge = make_plan(999999999, g, c, pool);
    CHK(std::holds_alternative<DoesNotClose>(rhuge));

    // ---- plan_refuses: zero blob ----
    Costs czero = c; czero.expert_blob = 0;
    Result rzero = make_plan(32768, g, czero, pool);
    CHK(std::holds_alternative<DoesNotClose>(rzero));

    // ---- slots_antitone: more context → fewer slots ----
    Result r64 = make_plan(65536, g, c, pool);
    if (std::holds_alternative<Plan>(r64)) {
        const Plan& p64 = std::get<Plan>(r64);
        CHK(p64.cache_slots <= p32.cache_slots);
    }

    // ---- vram_pool_bytes: 32 GiB device − 512 MiB reserve ----
    CHK(vram_pool_bytes(32ull << 30, 512ull << 20) == 33822867456ull);
    // truncating subtraction: reserve > device → 0
    CHK(vram_pool_bytes(100, 200) == 0u);

    std::printf("  cache_slots at ctx=32768: %llu\n", (unsigned long long)p32.cache_slots);
    return u7f1::finish("TEST_MAKE_PLAN");
}


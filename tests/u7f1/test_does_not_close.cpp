// tests/u7f1/test_does_not_close.cpp — VramMemoryPlan DoesNotClose and to_string.
// Theory:
//   plan_refuses (ZeroExpertBlob): blob == 0 → DoesNotClose
//   plan_refuses (FixedExceedsPool): kv + fixed > pool → DoesNotClose
//   plan_refuses (NoWholeSlot): cache / blob == 0 → DoesNotClose
//   to_string(Plan): contains "plan:", "context=", "slots=", "used="
//   to_string(DoesNotClose): contains "does not close"
//   to_string(Result): dispatches correctly
#include "baloon/plan.hpp"
#include "u7f1/check.hpp"
#include <cstdio>
#include <variant>
#include <string>
int main() {
    std::puts("=== test_does_not_close ===");
    using namespace baloon::plan;
    Geometry g; Costs c; c.state = state_bytes(g);
    const uint64_t pool = vram_pool_bytes(32ull<<30, 512ull<<20);

    // ---- ZeroExpertBlob ----
    Costs cz = c; cz.expert_blob = 0;
    Result rz = make_plan(32768, g, cz, pool);
    CHK(std::holds_alternative<DoesNotClose>(rz));
    if (std::holds_alternative<DoesNotClose>(rz)) {
        const auto& d = std::get<DoesNotClose>(rz);
        CHK(d.reason == Reason::ZeroExpertBlob);
        CHK(d.blob == 0u);
    }

    // ---- FixedExceedsPool ----
    Result rf = make_plan(999999999, g, c, pool);
    CHK(std::holds_alternative<DoesNotClose>(rf));
    if (std::holds_alternative<DoesNotClose>(rf)) {
        const auto& d = std::get<DoesNotClose>(rf);
        // reason is FixedExceedsPool (kv alone overflows)
        CHK(d.pool == pool);
        CHK(d.blob == c.expert_blob);
    }

    // ---- NoWholeSlot: tiny pool, kv+fixed fit but < 1 blob ----
    // Use ctx=0, tiny pool = state + 1 (not enough for a blob)
    const uint64_t tiny = c.state + 1u;
    Result rn = make_plan(0, g, c, tiny);
    // kv = 0, fixed = state, cache = 1 B, 1 B / blob = 0 slots
    CHK(std::holds_alternative<DoesNotClose>(rn));
    if (std::holds_alternative<DoesNotClose>(rn)) {
        const auto& d = std::get<DoesNotClose>(rn);
        CHK(d.reason == Reason::NoWholeSlot);
    }

    // ---- to_string(Plan) ----
    Result rp = make_plan(32768, g, c, pool);
    CHK(std::holds_alternative<Plan>(rp));
    std::string sp = to_string(rp);
    CHK(sp.find("plan:") != std::string::npos);
    CHK(sp.find("context=") != std::string::npos);
    CHK(sp.find("slots=") != std::string::npos);
    CHK(sp.find("used=") != std::string::npos);

    // ---- to_string(DoesNotClose) ----
    std::string sd = to_string(rf);
    CHK(sd.find("does not close") != std::string::npos);

    // ---- to_string(Reason) ----
    CHK(std::string(to_string(Reason::ZeroExpertBlob)).find("zero") != std::string::npos ||
        std::string(to_string(Reason::ZeroExpertBlob)).find("blob") != std::string::npos);

    return u7f1::finish("TEST_DOES_NOT_CLOSE");
}


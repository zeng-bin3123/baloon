// tests/test_unit1.cpp — Smoke test for unit 1 (core headers + fnv1a + basic API)
#include "strata/core/graph.hpp"
#include "strata/core/pinned.hpp"
#include "strata/core/weights.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/mtp.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { ++tests_passed; } \
    else { ++tests_failed; fprintf(stderr, "FAIL: %s\n", msg); } \
} while(0)

int main() {
    printf("=== strata unit 1 smoke test ===\n");

    // 1. LayerType enum and to_string
    CHECK(strata::to_string(strata::LayerType::GDN) == std::string("GDN"), "LayerType::GDN");
    CHECK(strata::to_string(strata::LayerType::QSA) == std::string("QSA"), "LayerType::QSA");

    // 2. FNV-1a: empty input returns seed
    {
        uint64_t h = strata::fnv1a64(nullptr, 0);
        CHECK(h == 14695981039346656037ull, "fnv1a64 empty == seed");
    }

    // 3. FNV-1a: known single-byte
    {
        uint8_t buf[1] = {'a'};
        uint64_t h = strata::fnv1a64(buf, 1);
        // FNV-1a('a') = (seed ^ 0x61) * prime
        uint64_t expected = (14695981039346656037ull ^ 0x61) * 1099511628211ull;
        CHECK(h == expected, "fnv1a64 single byte 'a'");
    }

    // 4. WeightRef properties
    {
        strata::WeightRef ref;
        CHECK(!ref.quantized(), "default WeightRef not quantized");
        CHECK(!ref.wants_q8k(), "default WeightRef not q8k");
        ref.code_bits = 4;
        CHECK(ref.quantized(), "code_bits=4 is quantized");
        ref.act_kind = 1;
        CHECK(ref.wants_q8k(), "act_kind=1 wants q8k");
    }

    // 5. WeightTable find on empty table
    {
        strata::WeightTable wt;
        CHECK(wt.find("nonexistent") == nullptr, "empty table find returns null");
    }

    // 6. CapturedGraph default state
    {
        strata::CapturedGraph cg;
        CHECK(!cg.valid(), "default CapturedGraph not valid");
        CHECK(cg.nodes() == 0, "default CapturedGraph 0 nodes");
    }

    // 7. CapturedGraph move semantics
    {
        strata::CapturedGraph a;
        strata::CapturedGraph b(std::move(a));
        CHECK(!b.valid(), "moved-from default still not valid");
    }

    // 8. GraphRegistry empty
    {
        strata::GraphRegistry reg(nullptr);
        CHECK(reg.size() == 0, "empty registry size 0");
        CHECK(reg.captures() == 0, "empty registry captures 0");
        CHECK(reg.find(strata::LayerType::GDN, 1) == nullptr, "empty registry find null");
    }

    // 9. WeightKind values
    {
        CHECK(static_cast<int>(strata::WeightKind::Verbatim) == 0, "Verbatim=0");
        CHECK(static_cast<int>(strata::WeightKind::Bf16InF32) == 1, "Bf16InF32=1");
        CHECK(static_cast<int>(strata::WeightKind::F32) == 2, "F32=2");
        CHECK(static_cast<int>(strata::WeightKind::F16InF32) == 3, "F16InF32=3");
    }

    // 10. PageBacking enum
    {
        strata::PinnedArena arena;
        CHECK(!arena.valid(), "default PinnedArena not valid");
    }

    // 11. Verifier default
    {
        strata::Verifier v;
        CHECK(v.final_R_all() == nullptr, "default verifier null residual");
        CHECK(v.windows == 0, "default verifier 0 windows");
    }

    // 12. MtpDrafter default
    {
        strata::MtpDrafter d;
        CHECK(d.vram_bytes() == 0, "default drafter 0 vram");
        CHECK(d.rounds == 0, "default drafter 0 rounds");
    }

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}


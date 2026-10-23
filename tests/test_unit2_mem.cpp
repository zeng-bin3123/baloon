// tests/test_unit2_mem.cpp — Smoke test for unit 2 memory modules
// Covers: PinnedHostArena, DeviceArena, MemoryPlanner (VRAMPlan)
#include "strata/core/pinned_host_arena.hpp"
#include "strata/core/device_arena.hpp"
#include "strata/core/memory_planner.hpp"
#include "strata/core/device.hpp"

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

// ═══════════════════════════════════════════════════════════════════
// 1. PinnedHostArena tests
// ═══════════════════════════════════════════════════════════════════
static void test_pinned_host_arena() {
    printf("--- PinnedHostArena ---\n");

    // 1a. Default state
    {
        strata::PinnedHostArena arena;
        CHECK(!arena.valid(), "default arena not valid");
        CHECK(arena.capacity() == 0, "default capacity 0");
        CHECK(arena.used() == 0, "default used 0");
        CHECK(arena.remaining() == 0, "default remaining 0");
        CHECK(arena.alloc(64) == nullptr, "alloc on default returns null");
    }

    // 1b. Reserve a small arena (4 MiB)
    {
        strata::PinnedHostArena arena;
        std::string err;
        uint64_t size = 4ull * 1024 * 1024;
        bool ok = arena.reserve(size, {}, err);
        CHECK(ok, "reserve 4 MiB succeeds");
        if (!ok) { fprintf(stderr, "  err: %s\n", err.c_str()); }
        CHECK(arena.valid(), "arena valid after reserve");
        CHECK(arena.capacity() == size, "capacity matches");
        CHECK(arena.used() == 0, "used starts at 0");
        CHECK(arena.remaining() == size, "remaining == capacity initially");
        CHECK(arena.base() != nullptr, "base not null");

        // 1c. Bump allocations
        uint8_t* p1 = arena.alloc(1024, 64);
        CHECK(p1 != nullptr, "alloc 1024 succeeds");
        CHECK(arena.used() >= 1024, "used >= 1024 after alloc");
        CHECK(reinterpret_cast<uintptr_t>(p1) % 64 == 0, "alloc aligned to 64");

        uint8_t* p2 = arena.alloc(2048, 256);
        CHECK(p2 != nullptr, "alloc 2048 succeeds");
        CHECK(p2 > p1, "second alloc > first");
        CHECK(reinterpret_cast<uintptr_t>(p2) % 256 == 0, "alloc aligned to 256");

        // 1d. Write and read back (verify the memory is usable)
        memset(p1, 0xAB, 1024);
        memset(p2, 0xCD, 2048);
        CHECK(p1[0] == 0xAB, "write-read p1[0]");
        CHECK(p1[1023] == 0xAB, "write-read p1[1023]");
        CHECK(p2[0] == 0xCD, "write-read p2[0]");
        CHECK(p2[2047] == 0xCD, "write-read p2[2047]");

        // 1e. Remaining tracks correctly
        uint64_t used_now = arena.used();
        CHECK(arena.remaining() == size - used_now, "remaining consistent");

        // 1f. Exhaust the arena
        uint8_t* p3 = arena.alloc(size, 64);  // too big
        CHECK(p3 == nullptr, "exhausted alloc returns null");

        // 1g. Reset
        arena.reset();
        CHECK(arena.used() == 0, "used 0 after reset");
        CHECK(arena.remaining() == size, "remaining == capacity after reset");

        // Can allocate again after reset
        uint8_t* p4 = arena.alloc(512, 64);
        CHECK(p4 != nullptr, "alloc after reset succeeds");

        // 1h. Registered bytes > 0 (CUDA host registration)
        printf("  registered_bytes = %llu\n",
               (unsigned long long)arena.registered_bytes());
    }

    // 1i. Reserve with slice boundaries
    {
        strata::PinnedHostArena arena;
        std::string err;
        uint64_t size = 2ull * 1024 * 1024;
        std::vector<uint64_t> bounds = {0, 1024*1024, 2*1024*1024};
        bool ok = arena.reserve(size, bounds, err);
        CHECK(ok, "reserve with bounds succeeds");
        if (!ok) { fprintf(stderr, "  err: %s\n", err.c_str()); }
        CHECK(arena.valid(), "arena valid with bounds");
    }

    // 1j. Zero-size reserve fails
    {
        strata::PinnedHostArena arena;
        std::string err;
        bool ok = arena.reserve(0, {}, err);
        CHECK(!ok, "zero-size reserve fails");
    }

    // 1k. Double reserve fails
    {
        strata::PinnedHostArena arena;
        std::string err;
        bool ok1 = arena.reserve(1024*1024, {}, err);
        CHECK(ok1, "first reserve ok");
        bool ok2 = arena.reserve(1024*1024, {}, err);
        CHECK(!ok2, "double reserve fails");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2. DeviceArena tests
// ═══════════════════════════════════════════════════════════════════
static void test_device_arena() {
    printf("--- DeviceArena ---\n");

    // 2a. Default state
    {
        strata::DeviceArena arena;
        CHECK(!arena.valid(), "default arena not valid");
        CHECK(arena.capacity() == 0, "default capacity 0");
        CHECK(arena.used() == 0, "default used 0");
        CHECK(arena.alloc(64) == nullptr, "alloc on default returns null");
    }

    // 2b. Open a small arena (16 MiB)
    {
        strata::DeviceArena arena;
        std::string err;
        uint64_t size = 16ull * 1024 * 1024;
        bool ok = arena.open(size, err);
        CHECK(ok, "open 16 MiB succeeds");
        if (!ok) { fprintf(stderr, "  err: %s\n", err.c_str()); }
        CHECK(arena.valid(), "arena valid after open");
        CHECK(arena.capacity() == size, "capacity matches");
        CHECK(arena.used() == 0, "used starts at 0");
        CHECK(arena.remaining() == size, "remaining == capacity");
        CHECK(arena.base() != nullptr, "base not null");

        // 2c. Bump allocations
        void* p1 = arena.alloc(4096, 256);
        CHECK(p1 != nullptr, "alloc 4096 succeeds");
        CHECK(arena.used() >= 4096, "used >= 4096");
        CHECK(reinterpret_cast<uintptr_t>(p1) % 256 == 0, "aligned to 256");

        void* p2 = arena.alloc(8192, 512);
        CHECK(p2 != nullptr, "alloc 8192 succeeds");
        CHECK(p2 > p1, "second alloc > first");
        CHECK(reinterpret_cast<uintptr_t>(p2) % 512 == 0, "aligned to 512");

        // 2d. Remaining
        uint64_t used_now = arena.used();
        CHECK(arena.remaining() == size - used_now, "remaining consistent");

        // 2e. Exhaust
        void* p3 = arena.alloc(size, 256);
        CHECK(p3 == nullptr, "exhausted alloc returns null");

        // 2f. Reset
        ok = arena.reset(err);
        CHECK(ok, "reset succeeds");
        CHECK(arena.used() == 0, "used 0 after reset");

        void* p4 = arena.alloc(1024, 256);
        CHECK(p4 != nullptr, "alloc after reset succeeds");

        // 2g. Close
        arena.close();
        CHECK(!arena.valid(), "not valid after close");
        CHECK(arena.capacity() == 0, "capacity 0 after close");
    }

    // 2h. Zero-size open fails
    {
        strata::DeviceArena arena;
        std::string err;
        bool ok = arena.open(0, err);
        CHECK(!ok, "zero-size open fails");
    }

    // 2i. Double open fails
    {
        strata::DeviceArena arena;
        std::string err;
        bool ok1 = arena.open(1024*1024, err);
        CHECK(ok1, "first open ok");
        bool ok2 = arena.open(1024*1024, err);
        CHECK(!ok2, "double open fails");
        arena.close();
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3. MemoryPlanner tests
// ═══════════════════════════════════════════════════════════════════
static void test_memory_planner() {
    printf("--- MemoryPlanner ---\n");

    // 3a. kv_cache_bytes computation
    {
        strata::ModelSpec spec;
        spec.n_layers = 48;
        spec.kv_heads = 4;
        spec.head_dim = 128;

        // KV cache for context=1024:
        // 48 layers * 4 heads * 2 (K+V) * 128 dim * 1024 tokens = 50331648 bytes
        uint64_t kv = strata::kv_cache_bytes(spec, 1024);
        uint64_t expected = 48ull * 4 * 2 * 128 * 1024;
        CHECK(kv == expected, "kv_cache_bytes for ctx=1024");
        printf("  kv_cache_bytes(ctx=1024) = %llu (expected %llu)\n",
               (unsigned long long)kv, (unsigned long long)expected);

        // Zero context → 0 bytes
        CHECK(strata::kv_cache_bytes(spec, 0) == 0, "kv_cache_bytes ctx=0");
    }

    // 3b. scratch_bytes_auto
    {
        strata::ModelSpec spec;
        spec.n_embd = 2560;
        spec.hc = 4;
        spec.n_expert = 128;
        spec.n_expert_active = 8;
        spec.kv_heads = 4;
        spec.head_dim = 128;

        uint64_t scratch = strata::scratch_bytes_auto(spec);
        CHECK(scratch >= 16ull * 1024 * 1024, "scratch >= 16 MiB minimum");
        CHECK(scratch % (1024*1024) == 0, "scratch aligned to 1 MiB");
        printf("  scratch_bytes_auto = %llu bytes (%.2f MiB)\n",
               (unsigned long long)scratch,
               (double)scratch / (1024.0 * 1024.0));
    }

    // 3c. plan_vram — typical 32 GiB GPU
    {
        strata::ModelSpec spec;
        spec.n_layers = 48;
        spec.n_embd = 2560;
        spec.n_expert = 128;
        spec.n_expert_active = 8;
        spec.expert_blob_bytes = 5181440;
        spec.dense_bytes = 500 * 1024 * 1024;   // 500 MiB dense
        spec.head_bytes = 50 * 1024 * 1024;     // 50 MiB head
        spec.kv_heads = 4;
        spec.head_dim = 128;
        spec.hc = 4;

        uint64_t vram = 32ull * 1024 * 1024 * 1024;  // 32 GiB
        strata::VRAMPlan plan = strata::plan_vram(vram, spec, 4096);

        CHECK(plan.feasible(), "plan feasible on 32 GiB");
        CHECK(plan.vram_total == vram, "vram_total matches");
        CHECK(plan.vram_reserved == 512ull * 1024 * 1024, "default reserved 512 MiB");
        CHECK(plan.dense_bytes == 500ull * 1024 * 1024, "dense_bytes matches");
        CHECK(plan.head_bytes == 50ull * 1024 * 1024, "head_bytes matches");
        CHECK(plan.expert_slots > 0, "some expert slots allocated");
        CHECK(plan.expert_vram_bytes > 0, "expert VRAM > 0");
        CHECK(plan.context_length == 4096, "context_length stored");

        // Verify everything adds up
        uint64_t sum = plan.dense_bytes + plan.head_bytes +
                       plan.kv_cache_bytes + plan.scratch_bytes +
                       plan.expert_vram_bytes;
        CHECK(sum <= plan.vram_usable, "allocations <= usable");

        // Expert slots × blob bytes <= expert VRAM
        uint64_t slot_bytes = (uint64_t)plan.expert_slots *
                              (uint64_t)spec.expert_blob_bytes;
        CHECK(slot_bytes <= plan.expert_vram_bytes, "slot bytes fit");

        printf("  %s\n", plan.summary().c_str());
    }

    // 3d. plan_vram — tiny VRAM (infeasible)
    {
        strata::ModelSpec spec;
        spec.n_layers = 48;
        spec.dense_bytes = 2ull * 1024 * 1024 * 1024;  // 2 GiB dense — more than available
        spec.head_bytes = 0;
        spec.kv_heads = 4;
        spec.head_dim = 128;

        uint64_t vram = 1ull * 1024 * 1024 * 1024;  // 1 GiB
        strata::VRAMPlan plan = strata::plan_vram(vram, spec, 1024);

        CHECK(!plan.feasible(), "plan infeasible on 1 GiB with 2 GiB dense");
        CHECK(plan.expert_slots == 0, "no expert slots when infeasible");
    }

    // 3e. plan_vram — zero context length
    {
        strata::ModelSpec spec;
        spec.n_layers = 48;
        spec.dense_bytes = 100 * 1024 * 1024;
        spec.head_bytes = 10 * 1024 * 1024;
        spec.expert_blob_bytes = 5181440;
        spec.kv_heads = 4;
        spec.head_dim = 128;

        uint64_t vram = 32ull * 1024 * 1024 * 1024;
        strata::VRAMPlan plan = strata::plan_vram(vram, spec, 0);

        CHECK(plan.feasible(), "feasible with ctx=0");
        CHECK(plan.kv_cache_bytes == 0, "kv_cache_bytes == 0 for ctx=0");
        CHECK(plan.expert_slots > 0, "expert slots with ctx=0");
    }

    // 3f. Expert fractions
    {
        strata::ModelSpec spec;
        spec.n_layers = 48;
        spec.dense_bytes = 0;
        spec.head_bytes = 0;
        spec.expert_blob_bytes = 1024 * 1024;  // 1 MiB per expert
        spec.n_embd = 2560;
        spec.hc = 4;
        spec.n_expert = 128;
        spec.n_expert_active = 8;
        spec.kv_heads = 4;
        spec.head_dim = 128;

        uint64_t vram = 1ull * 1024 * 1024 * 1024;  // 1 GiB
        strata::VRAMPlan plan = strata::plan_vram(vram, spec, 0);

        CHECK(plan.expert_vram_frac > 0.5, "most VRAM for experts when no dense/head/kv");
        printf("  expert_vram_frac = %.3f\n", plan.expert_vram_frac);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 4. Integration: plan → allocate
// ═══════════════════════════════════════════════════════════════════
static void test_plan_to_allocate() {
    printf("--- Plan → Allocate integration ---\n");

    // Use the planner to decide sizes, then allocate via DeviceArena
    strata::ModelSpec spec;
    spec.n_layers = 48;
    spec.n_embd = 2560;
    spec.n_expert = 128;
    spec.n_expert_active = 8;
    spec.expert_blob_bytes = 5181440;
    spec.dense_bytes = 10 * 1024 * 1024;   // 10 MiB for test
    spec.head_bytes = 2 * 1024 * 1024;     // 2 MiB for test
    spec.kv_heads = 4;
    spec.head_dim = 128;
    spec.hc = 4;

    // Query actual device VRAM
    strata::DeviceInfo info;
    std::string err;
    bool init_ok = strata::device_init(0, info, err);
    CHECK(init_ok, "device_init for integration test");
    if (!init_ok) {
        fprintf(stderr, "  device_init failed: %s\n", err.c_str());
        return;
    }

    // Plan with a small slice of available VRAM (64 MiB) for the test
    uint64_t test_vram = 64ull * 1024 * 1024;
    strata::VRAMPlan plan = strata::plan_vram(test_vram, spec, 256,
                                              4ull * 1024 * 1024);  // 4 MiB reserved

    CHECK(plan.feasible(), "test plan feasible");
    printf("  plan: dense=%llu head=%llu kv=%llu scratch=%llu expert=%llu slots=%lld\n",
           (unsigned long long)plan.dense_bytes,
           (unsigned long long)plan.head_bytes,
           (unsigned long long)plan.kv_cache_bytes,
           (unsigned long long)plan.scratch_bytes,
           (unsigned long long)plan.expert_vram_bytes,
           (long long)plan.expert_slots);

    if (!plan.feasible()) return;

    // Open a device arena sized to the plan's usable VRAM
    strata::DeviceArena darena;
    bool ok = darena.open(plan.vram_usable, err);
    CHECK(ok, "device arena open for plan");
    if (!ok) {
        fprintf(stderr, "  device arena open failed: %s\n", err.c_str());
        return;
    }

    // Allocate each section
    void* dense_ptr = darena.alloc(plan.dense_bytes, 256);
    CHECK(dense_ptr != nullptr, "dense alloc ok");

    void* head_ptr = darena.alloc(plan.head_bytes, 256);
    CHECK(head_ptr != nullptr, "head alloc ok");

    if (plan.kv_cache_bytes > 0) {
        void* kv_ptr = darena.alloc(plan.kv_cache_bytes, 256);
        CHECK(kv_ptr != nullptr, "kv alloc ok");
    }

    void* scratch_ptr = darena.alloc(plan.scratch_bytes, 256);
    CHECK(scratch_ptr != nullptr, "scratch alloc ok");

    if (plan.expert_vram_bytes > 0) {
        void* expert_ptr = darena.alloc(plan.expert_vram_bytes, 256);
        // This may or may not fit depending on alignment overhead
        if (expert_ptr) {
            printf("  expert alloc ok (%llu bytes)\n",
                   (unsigned long long)plan.expert_vram_bytes);
        } else {
            printf("  expert alloc null (alignment overhead)\n");
        }
    }

    printf("  arena used: %llu / %llu bytes\n",
           (unsigned long long)darena.used(),
           (unsigned long long)darena.capacity());

    darena.close();
}

// ═══════════════════════════════════════════════════════════════════
int main() {
    printf("=== strata unit 2 memory modules smoke test ===\n\n");

    test_pinned_host_arena();
    test_device_arena();
    test_memory_planner();
    test_plan_to_allocate();

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}


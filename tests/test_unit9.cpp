// tests/test_unit9.cpp — Smoke test for unit 1/31 modules:
//   expert_cache, expert_source, pinned_arena, pinned_host_arena
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/pinned.hpp"

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
    printf("=== strata unit 9 smoke test (expert_cache, expert_source, pinned) ===\n");

    // ──── kNotResident ────────────────────────────────────────────
    CHECK(strata::kNotResident == -1, "kNotResident == -1");

    // ──── ExpertCache: open uniform ──────────────────────────────
    {
        strata::ExpertCache cache;
        CHECK(!cache.is_open(), "cache not open by default");

        std::string err;
        bool ok = cache.open(10, 2, 4, 1024, err);
        CHECK(ok, "open(10,2,4,1024) succeeds");
        CHECK(cache.is_open(), "cache is open after open()");
        CHECK(cache.slots() == 10, "10 slots");
        CHECK(cache.n_layers() == 2, "2 layers");
        CHECK(cache.n_expert() == 4, "4 experts");
        CHECK(cache.blob_bytes() == 1024, "1024 blob_bytes");
        CHECK(cache.next_free() == 0, "next_free starts at 0");
        CHECK(cache.fills() == 0, "fills starts at 0");

        // All residency entries = kNotResident
        for (int64_t l = 0; l < 2; ++l) {
            for (int64_t e = 0; e < 4; ++e) {
                CHECK(cache.slot_of(l, e) == strata::kNotResident,
                      "initial slot_of == kNotResident");
            }
        }

        // slot_of out of bounds returns kNotResident
        CHECK(cache.slot_of(-1, 0) == strata::kNotResident, "slot_of(-1,0) == kNR");
        CHECK(cache.slot_of(2, 0) == strata::kNotResident, "slot_of(2,0) == kNR");
        CHECK(cache.slot_of(0, 4) == strata::kNotResident, "slot_of(0,4) == kNR");

        // ── Global admit ─────────────────────────────────────────
        int32_t s0 = cache.admit(0, 0);
        CHECK(s0 == 0, "first admit returns slot 0");
        CHECK(cache.slot_of(0, 0) == 0, "slot_of(0,0) == 0 after admit");

        int32_t s1 = cache.admit(0, 1);
        CHECK(s1 == 1, "second admit returns slot 1");

        // Idempotent: re-admit same expert returns same slot
        int32_t s0b = cache.admit(0, 0);
        CHECK(s0b == 0, "re-admit(0,0) returns same slot 0");

        // ── fill_slot_blocking ───────────────────────────────────
        std::vector<uint8_t> blob(1024, 0xAB);
        ok = cache.fill_slot_blocking(0, blob.data(), err);
        CHECK(ok, "fill_slot_blocking succeeds");
        CHECK(cache.fills() == 1, "fills == 1 after one fill");

        // ── verify_slot ──────────────────────────────────────────
        ok = cache.verify_slot(0, blob.data(), err);
        CHECK(ok, "verify_slot matches");

        // Verify with wrong data fails and reports byte position
        std::vector<uint8_t> wrong(1024, 0xCD);
        ok = cache.verify_slot(0, wrong.data(), err);
        CHECK(!ok, "verify_slot detects mismatch");
        CHECK(err.find("byte 0") != std::string::npos, "verify error reports byte position");

        // ── fill_slot async ──────────────────────────────────────
        std::vector<uint8_t> blob2(1024, 0xEF);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        ok = cache.fill_slot(1, blob2.data(), stream, err);
        CHECK(ok, "fill_slot async succeeds");
        cudaStreamSynchronize(stream);
        CHECK(cache.fills() == 2, "fills == 2 after two fills");

        ok = cache.verify_slot(1, blob2.data(), err);
        CHECK(ok, "verify_slot after async fill");
        cudaStreamDestroy(stream);

        cache.close();
        CHECK(!cache.is_open(), "cache not open after close");
    }

    // ──── ExpertCache: per-layer admission ───────────────────────
    {
        strata::ExpertCache cache;
        std::string err;
        // 12 slots, 3 layers, 4 experts => q = 4 slots/layer
        bool ok = cache.open(12, 3, 4, 64, err);
        CHECK(ok, "per-layer: open succeeds");

        cache.set_per_layer(true);
        CHECK(cache.per_layer(), "per_layer flag set");

        // Layer 0: slots [0, 4)
        // Layer 1: slots [4, 8)
        // Layer 2: slots [8, 12) (last gets remainder)
        int64_t lo, hi;
        cache.layer_slot_range(0, lo, hi);
        CHECK(lo == 0 && hi == 4, "layer 0 range [0,4)");
        cache.layer_slot_range(1, lo, hi);
        CHECK(lo == 4 && hi == 8, "layer 1 range [4,8)");
        cache.layer_slot_range(2, lo, hi);
        CHECK(lo == 8 && hi == 12, "layer 2 range [8,12)");

        // Admit into layer 0
        int32_t s = cache.admit(0, 0);
        CHECK(s == 0, "per-layer: layer 0 first admit = 0");
        s = cache.admit(0, 1);
        CHECK(s == 1, "per-layer: layer 0 second admit = 1");

        // Admit into layer 1
        s = cache.admit(1, 0);
        CHECK(s == 4, "per-layer: layer 1 first admit = 4");

        // Fill layer 0's quota (2 more admits to fill 4 slots)
        cache.admit(0, 2);
        cache.admit(0, 3);
        // Next should fail (layer 0 full)
        s = cache.admit(0, 3);  // but this is already resident
        CHECK(s == 3, "re-admit returns existing slot");

        cache.close();
    }

    // ──── ExpertCache: open_sized (variable) ─────────────────────
    {
        strata::ExpertCache cache;
        std::string err;
        std::vector<int64_t> sizes = {256, 512, 128};
        bool ok = cache.open_sized(sizes, 2, 4, err);
        CHECK(ok, "open_sized succeeds");
        CHECK(cache.slots() == 3, "3 variable slots");
        CHECK(cache.slot_offsets() != nullptr, "offset table present");
        CHECK(cache.slot_offsets()[0] == 0, "off[0] == 0");
        CHECK(cache.slot_offsets()[1] == 256, "off[1] == 256");
        CHECK(cache.slot_offsets()[2] == 768, "off[2] == 768");
        CHECK(cache.slot_offsets()[3] == 896, "off[3] == 896 (total)");

        // device_slot pointers are offset-based
        const uint8_t* d0 = cache.device_slot(0);
        const uint8_t* d1 = cache.device_slot(1);
        CHECK(d1 - d0 == 256, "slot 1 offset from slot 0 is 256");

        cache.close();
    }

    // ──── ExpertCache: invalid params ────────────────────────────
    {
        strata::ExpertCache cache;
        std::string err;
        CHECK(!cache.open(0, 1, 1, 64, err), "open with 0 slots fails");
        CHECK(!cache.open(1, 0, 1, 64, err), "open with 0 layers fails");
        CHECK(!cache.open(1, 1, 0, 64, err), "open with 0 experts fails");
        CHECK(!cache.open(1, 1, 1, 0, err), "open with 0 blob_bytes fails");
    }

    // ──── FNV-1a ─────────────────────────────────────────────────
    {
        // Empty input returns seed
        uint64_t h = strata::fnv1a64(nullptr, 0);
        CHECK(h == 14695981039346656037ull, "fnv1a64 empty == offset basis");

        // Deterministic: same input same output
        uint8_t data[] = {0x41, 0x42, 0x43};
        uint64_t h1 = strata::fnv1a64(data, 3);
        uint64_t h2 = strata::fnv1a64(data, 3);
        CHECK(h1 == h2, "fnv1a64 is deterministic");

        // Different data produces different hash
        uint8_t data2[] = {0x41, 0x42, 0x44};
        uint64_t h3 = strata::fnv1a64(data2, 3);
        CHECK(h1 != h3, "fnv1a64 different data different hash");
    }

    // ──── PinnedArena ────────────────────────────────────────────
    {
        // Default arena is invalid
        strata::PinnedArena empty;
        CHECK(!empty.valid(), "default PinnedArena not valid");
        CHECK(empty.data() == nullptr, "default PinnedArena data is null");

        // Small allocation should succeed
        strata::PinnedArena arena(4096);
        CHECK(arena.valid(), "4096-byte arena is valid");
        CHECK(arena.data() != nullptr, "4096-byte arena has data");
        CHECK(arena.capacity == 4096, "capacity == 4096");

        // Write to the arena to verify it's usable
        std::memset(arena.data(), 0xAA, 4096);
        CHECK(arena.data()[0] == 0xAA, "can write to arena");
        CHECK(arena.data()[4095] == 0xAA, "can write to end of arena");
    }

    // ──── PinnedArena with slices ────────────────────────────────
    {
        // Slice-by-slice registration
        strata::PinnedArena arena(8192, 4096);  // 2 slices of 4096
        CHECK(arena.valid(), "sliced arena valid");
        CHECK(arena.capacity == 8192, "sliced arena capacity");
        // Even if registration failed, the memory should be usable
        std::memset(arena.data(), 0xBB, 8192);
        CHECK(arena.data()[0] == 0xBB, "sliced arena writable start");
        CHECK(arena.data()[8191] == 0xBB, "sliced arena writable end");
    }

    // ──── PinnedArena with explicit bounds ───────────────────────
    {
        std::vector<uint64_t> bounds = {0, 2048, 4096};
        strata::PinnedArena arena(4096, bounds);
        CHECK(arena.valid(), "bounds arena valid");
    }

    // ──── PageBacking enum ───────────────────────────────────────
    {
        CHECK(static_cast<int>(strata::PageBacking::LargePages) != 
              static_cast<int>(strata::PageBacking::NormalPages),
              "LargePages != NormalPages");
        CHECK(static_cast<int>(strata::PageBacking::NormalPages) != 
              static_cast<int>(strata::PageBacking::PinnedByCuda),
              "NormalPages != PinnedByCuda");
    }

    // ──── StreamStats ────────────────────────────────────────────
    {
        strata::StreamStats ss;
        CHECK(ss.gib_per_second() == 0.0, "default StreamStats 0 GiB/s");
        ss.seconds = 1.0;
        ss.bytes_streamed = 1073741824ull;  // 1 GiB
        double rate = ss.gib_per_second();
        CHECK(rate > 0.99 && rate < 1.01, "1 GiB in 1 sec ~= 1 GiB/s");
    }

    // ──── LoadStats ──────────────────────────────────────────────
    {
        strata::LoadStats ls;
        CHECK(ls.gib_per_second() == 0.0, "default LoadStats 0 GiB/s");
        ls.seconds = 2.0;
        ls.bytes_read = 2 * 1073741824ull;  // 2 GiB
        double rate = ls.gib_per_second();
        CHECK(rate > 0.99 && rate < 1.01, "2 GiB in 2 sec ~= 1 GiB/s");
    }

    // ──── ExpertSource interface ─────────────────────────────────
    {
        // FileExpertSource default state
        strata::FileExpertSource fs;
        CHECK(!fs.mapped(), "default FileExpertSource not mapped");
        CHECK(fs.reads() == 0, "default reads == 0");
        CHECK(fs.blob(0, 0) == nullptr, "unmapped blob returns null");

        // ArenaExpertSource default state
        strata::ArenaExpertSource as;
        CHECK(!as.mapped(), "default ArenaExpertSource not mapped");
        CHECK(as.reads() == 0, "default arena reads == 0");
        CHECK(as.blob(0, 0) == nullptr, "unmapped arena blob returns null");
        CHECK(!as.pinned(0, 0), "unmapped arena not pinned");
        CHECK(as.device_alias(0, 0) == nullptr, "unmapped arena no device alias");
    }

    // ──── ExpertDispatch default ─────────────────────────────────
    {
        strata::ExpertDispatch d;
        CHECK(d.pool == nullptr, "default dispatch no pool");
        CHECK(d.src == nullptr, "default dispatch no source");
        CHECK(d.cache == nullptr, "default dispatch no cache");
        CHECK(!d.hits_ready(), "default dispatch hits not ready");
        CHECK(!d.failed, "default dispatch not failed");
        CHECK(d.layers == 0, "default dispatch 0 layers");
        CHECK(d.experts == 0, "default dispatch 0 experts");
        CHECK(d.missing == 0, "default dispatch 0 missing");
    }

    // ──── GpuPlanSink default ────────────────────────────────────
    {
        strata::GpuPlanSink plan;
        CHECK(plan.counts == nullptr, "default plan null counts");
        CHECK(plan.cap == 0, "default plan 0 cap");
        CHECK(plan.pcie_mode == 0, "default plan DMA mode");
    }

    // ──── HitPhase enum ──────────────────────────────────────────
    {
        CHECK(static_cast<int>(strata::HitPhase::Prepare) == 0, "Prepare == 0");
        CHECK(static_cast<int>(strata::HitPhase::Compute) == 1, "Compute == 1");
        CHECK(static_cast<int>(strata::HitPhase::Combine) == 2, "Combine == 2");
    }

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}


// tests/test_unit2.cpp — Smoke test for unit 2 (expert_cache + expert_source)
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"

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
    printf("=== strata unit 2 smoke test ===\n");

    // 1. kNotResident sentinel
    CHECK(strata::kNotResident == -1, "kNotResident == -1");

    // 2. ExpertCache default state
    {
        strata::ExpertCache cache;
        CHECK(!cache.is_open(), "default cache not open");
        CHECK(cache.slots() == 0, "default cache 0 slots");
        CHECK(cache.slot_of(0, 0) == strata::kNotResident, "default slot_of returns kNotResident");
    }

    // 3. ExpertCache open (uniform)
    {
        strata::ExpertCache cache;
        std::string err;
        int64_t n_slots = 4, n_layers = 2, n_expert = 3, blob_bytes = 1024;
        bool ok = cache.open(n_slots, n_layers, n_expert, blob_bytes, err);
        CHECK(ok, "cache open succeeds");
        CHECK(cache.is_open(), "cache is open after open()");
        CHECK(cache.slots() == n_slots, "cache slots correct");
        CHECK(cache.n_layers() == n_layers, "cache n_layers correct");
        CHECK(cache.n_expert() == n_expert, "cache n_expert correct");
        CHECK(cache.blob_bytes() == blob_bytes, "cache blob_bytes correct");
        CHECK(cache.next_free() == 0, "cache next_free starts at 0");
        CHECK(cache.device_base() != nullptr, "cache device_base not null");

        // All residency should be kNotResident
        bool all_not_resident = true;
        for (int64_t l = 0; l < n_layers; ++l)
            for (int64_t e = 0; e < n_expert; ++e)
                if (cache.slot_of(l, e) != strata::kNotResident)
                    all_not_resident = false;
        CHECK(all_not_resident, "all residency kNotResident after open");

        // 4. Admit experts
        int32_t s0 = cache.admit(0, 0);
        CHECK(s0 == 0, "first admit returns slot 0");
        CHECK(cache.slot_of(0, 0) == 0, "slot_of returns 0 after admit");
        CHECK(cache.next_free() == 1, "next_free incremented");

        int32_t s1 = cache.admit(0, 1);
        CHECK(s1 == 1, "second admit returns slot 1");

        int32_t s2 = cache.admit(1, 2);
        CHECK(s2 == 2, "third admit returns slot 2");

        int32_t s3 = cache.admit(1, 0);
        CHECK(s3 == 3, "fourth admit returns slot 3");

        // Cache full
        int32_t s4 = cache.admit(0, 2);
        CHECK(s4 == strata::kNotResident, "admit when full returns kNotResident");

        // Re-admit already resident
        int32_t s0_again = cache.admit(0, 0);
        CHECK(s0_again == 0, "re-admit returns existing slot");

        // 5. Out-of-bounds slot_of
        CHECK(cache.slot_of(-1, 0) == strata::kNotResident, "negative layer returns kNotResident");
        CHECK(cache.slot_of(0, -1) == strata::kNotResident, "negative expert returns kNotResident");
        CHECK(cache.slot_of(999, 0) == strata::kNotResident, "oob layer returns kNotResident");

        // 6. fill_slot + verify_slot
        {
            std::vector<uint8_t> host_data(static_cast<size_t>(blob_bytes), 0xAB);
            std::string fill_err;
            bool fill_ok = cache.fill_slot_blocking(0, host_data.data(), fill_err);
            CHECK(fill_ok, "fill_slot_blocking succeeds");

            bool verify_ok = cache.verify_slot(0, host_data.data(), fill_err);
            CHECK(verify_ok, "verify_slot matches after fill");
        }

        // 7. device_slot pointers are spaced correctly
        {
            const uint8_t* p0 = cache.device_slot(0);
            const uint8_t* p1 = cache.device_slot(1);
            CHECK(p1 - p0 == blob_bytes, "device_slot spacing = blob_bytes");
        }

        cache.close();
        CHECK(!cache.is_open(), "cache not open after close");
    }

    // 8. ExpertCache open_sized (variable)
    {
        strata::ExpertCache cache;
        std::string err;
        std::vector<int64_t> sizes = {512, 1024, 768};
        bool ok = cache.open_sized(sizes, 2, 3, err);
        CHECK(ok, "open_sized succeeds");
        CHECK(cache.slots() == 3, "open_sized 3 slots");
        CHECK(cache.slot_offsets() != nullptr, "offset table non-null");
        CHECK(cache.slot_offsets()[0] == 0, "off[0] = 0");
        CHECK(cache.slot_offsets()[1] == 512, "off[1] = 512");
        CHECK(cache.slot_offsets()[2] == 512 + 1024, "off[2] = 1536");
        CHECK(cache.slot_offsets()[3] == 512 + 1024 + 768, "off[3] = 2304");
        cache.close();
    }

    // 9. Per-layer slot partitioning
    {
        strata::ExpertCache cache;
        std::string err;
        bool ok = cache.open(10, 2, 3, 256, err);
        CHECK(ok, "per-layer cache open");
        cache.set_per_layer(true);

        int64_t lo0 = 0, hi0 = 0, lo1 = 0, hi1 = 0;
        cache.layer_slot_range(0, lo0, hi0);
        cache.layer_slot_range(1, lo1, hi1);
        CHECK(lo0 == 0, "layer 0 lo = 0");
        CHECK(hi0 == 5, "layer 0 hi = 5");
        CHECK(lo1 == 5, "layer 1 lo = 5");
        CHECK(hi1 == 10, "layer 1 hi = 10 (remainder)");

        // Admit into layer 0
        int32_t s = cache.admit(0, 0);
        CHECK(s == 0, "per-layer admit layer 0 starts at 0");
        s = cache.admit(0, 1);
        CHECK(s == 1, "per-layer admit layer 0 slot 1");

        // Admit into layer 1
        s = cache.admit(1, 0);
        CHECK(s == 5, "per-layer admit layer 1 starts at 5");

        cache.close();
    }

    // 10. ExpertSource interface (via FileExpertSource default)
    {
        strata::FileExpertSource fs;
        CHECK(!fs.mapped(), "default FileExpertSource not mapped");
        CHECK(fs.blobs() == 0, "default FileExpertSource 0 blobs");
        CHECK(fs.reads() == 0, "default FileExpertSource 0 reads");
        CHECK(fs.blob(0, 0) == nullptr, "unmapped blob returns nullptr");
    }

    // 11. ArenaExpertSource default
    {
        strata::ArenaExpertSource as;
        CHECK(!as.mapped(), "default ArenaExpertSource not mapped");
        CHECK(as.blobs() == 0, "default ArenaExpertSource 0 blobs");
        CHECK(as.reads() == 0, "default ArenaExpertSource 0 reads");
        CHECK(!as.pinned(0, 0), "default ArenaExpertSource not pinned");
        CHECK(as.device_alias(0, 0) == nullptr, "default ArenaExpertSource no device alias");
    }

    // 12. ExpertDispatch default + hits_ready
    {
        strata::ExpertDispatch d;
        CHECK(!d.hits_ready(), "default dispatch not hits_ready");
        CHECK(!d.failed, "default dispatch not failed");
        CHECK(d.layers == 0, "default dispatch 0 layers");
        CHECK(d.experts == 0, "default dispatch 0 experts");
        CHECK(d.missing == 0, "default dispatch 0 missing");
        CHECK(d.n_expert == 512, "default dispatch n_expert = NE = 512");
    }

    // 13. GpuPlanSink defaults
    {
        strata::GpuPlanSink sink;
        CHECK(sink.counts == nullptr, "default sink counts null");
        CHECK(sink.pcie_mode == 0, "default sink pcie_mode 0");
        CHECK(sink.cap == 0, "default sink cap 0");
    }

    // 14. HitPhase enum
    {
        CHECK(static_cast<int>(strata::HitPhase::Prepare) == 0, "HitPhase::Prepare = 0");
        CHECK(static_cast<int>(strata::HitPhase::Compute) == 1, "HitPhase::Compute = 1");
        CHECK(static_cast<int>(strata::HitPhase::Combine) == 2, "HitPhase::Combine = 2");
    }

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}


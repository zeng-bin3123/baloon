// tests/test_unit1_device.cpp — Smoke test for unit 1 device/layout/cpu modules
#include "strata/core/device.hpp"
#include "strata/core/cpu_features.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/weights.hpp"
#include "strata/core/types_fwd.hpp"

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
    printf("=== strata unit 1 device/layout smoke test ===\n");

    // ── 1. DeviceInfo default state ──────────────────────────────────
    {
        strata::DeviceInfo info;
        CHECK(info.sm_count == 0, "DeviceInfo default sm_count == 0");
        CHECK(info.vram_total == 0, "DeviceInfo default vram_total == 0");
        CHECK(info.cc_major == 0, "DeviceInfo default cc_major == 0");
    }

    // ── 2. device_init on device 0 ───────────────────────────────────
    {
        strata::DeviceInfo info;
        std::string err;
        bool ok = strata::device_init(0, info, err);
        CHECK(ok, "device_init(0) succeeds");
        if (ok) {
            CHECK(info.sm_count > 0, "device has SMs");
            CHECK(info.vram_total > 0, "device has VRAM");
            printf("  GPU: %d SMs, %.2f GiB VRAM, CC %d.%d\n",
                   info.sm_count,
                   static_cast<double>(info.vram_total) / 1073741824.0,
                   info.cc_major, info.cc_minor);
        }
    }

    // ── 3. arena_alloc + NaN fill ────────────────────────────────────
    {
        std::string err;
        const size_t sz = 4096;
        void* ptr = strata::arena_alloc(sz, err);
        CHECK(ptr != nullptr, "arena_alloc returns non-null");
        if (ptr) {
            // Read back and verify NaN fill
            float host[1024];
            cudaMemcpy(host, ptr, sz, cudaMemcpyDeviceToHost);
            bool all_nan = true;
            for (int i = 0; i < 1024; ++i) {
                if (host[i] == host[i]) { // NaN != NaN
                    all_nan = false;
                    break;
                }
            }
            CHECK(all_nan, "arena_alloc fills with NaN");
            strata::arena_free(ptr);
        }
    }

    // ── 4. arena_alloc zero-size fails ───────────────────────────────
    {
        std::string err;
        void* ptr = strata::arena_alloc(0, err);
        CHECK(ptr == nullptr, "arena_alloc(0) returns null");
        CHECK(!err.empty(), "arena_alloc(0) sets error");
    }

    // ── 5. device_mem_info ───────────────────────────────────────────
    {
        size_t free_b = 0, total_b = 0;
        std::string err;
        bool ok = strata::device_mem_info(free_b, total_b, err);
        CHECK(ok, "device_mem_info succeeds");
        CHECK(total_b > 0, "total VRAM > 0");
        CHECK(free_b <= total_b, "free <= total");
    }

    // ── 6. CpuFeatures probe ─────────────────────────────────────────
    {
        strata::CpuFeatures f = strata::cpu_features();
        // On any modern x86, at least avx512f should be queryable
        // (may or may not be present — just check the probe doesn't crash)
        printf("  CPU features: F=%d BW=%d VL=%d VNNI=%d VBMI=%d\n",
               f.avx512f, f.avx512bw, f.avx512vl, f.avx512_vnni, f.avx512_vbmi);
        CHECK(f.avx512f == 0 || f.avx512f == 1, "avx512f is 0 or 1");
        CHECK(f.avx512bw == 0 || f.avx512bw == 1, "avx512bw is 0 or 1");

        // missing_report
        std::string missing = f.missing_report();
        if (f.expert_usable()) {
            CHECK(missing.empty(), "expert_usable => missing empty");
        } else {
            CHECK(!missing.empty(), "!expert_usable => missing non-empty");
            printf("  Missing: %s\n", missing.c_str());
        }
    }

    // ── 7. is_qsa_layer predicate ────────────────────────────────────
    {
        strata::Geometry g;
        g.n_layers = 48;
        // QSA at l % 4 == 3: {3,7,11,15,19,23,27,31,35,39,43,47}
        CHECK(strata::is_qsa_layer(g, 3), "layer 3 is QSA");
        CHECK(strata::is_qsa_layer(g, 7), "layer 7 is QSA");
        CHECK(strata::is_qsa_layer(g, 47), "layer 47 is QSA");
        CHECK(!strata::is_qsa_layer(g, 0), "layer 0 is GDN");
        CHECK(!strata::is_qsa_layer(g, 1), "layer 1 is GDN");
        CHECK(!strata::is_qsa_layer(g, 4), "layer 4 is GDN");
        CHECK(!strata::is_qsa_layer(g, 46), "layer 46 is GDN");

        int n_qsa = 0, n_gdn = 0;
        for (int l = 0; l < 48; ++l) {
            if (strata::is_qsa_layer(g, l)) ++n_qsa; else ++n_gdn;
        }
        CHECK(n_qsa == 12, "12 QSA layers in 48");
        CHECK(n_gdn == 36, "36 GDN layers in 48");
    }

    // ── 8. check_all on empty table with zero geometry ───────────────
    {
        strata::WeightTable wt;
        strata::Geometry g;
        g.n_layers = 0;
        std::string err;
        // n_layers = 0 should fail
        bool ok = strata::check_all(wt, g, err);
        CHECK(!ok, "check_all with n_layers=0 fails");
    }

    // ── 9. CudaError struct ──────────────────────────────────────────
    {
        strata::CudaError e("test error", 42);
        CHECK(e.message == "test error", "CudaError message");
        CHECK(e.code == 42, "CudaError code");
    }

    // ── Summary ──────────────────────────────────────────────────────
    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}


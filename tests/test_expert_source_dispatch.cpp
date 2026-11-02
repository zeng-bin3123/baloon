#include <cstdio>
#include <set>

#include "baloon/expert_source_dispatch.hpp"
#include "baloon/gguf_header_reader.hpp"

using namespace baloon::expert_source;

static int g_fail = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            if (g_fail < 50) std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_fail;                                                            \
        }                                                                        \
    } while (0)

static void test_geometry_and_addressing() {
    CHECK(kBlob == 1382400 && kH == 2560 && kFF == 640);
    CHECK(geometry_ok(2560, 640));
    for (uint64_t a : {0ull, 1ull, 2048ull, 2559ull, 2561ull, 4096ull}) CHECK(!geometry_ok(a, 640));
    for (uint64_t f : {0ull, 639ull, 641ull, 1280ull}) CHECK(!geometry_ok(2560, f));
    // cross-check against the quant geometry: a blob is 3 Q2_0 matrices of H x FF
    auto row = baloon::gguf::row_bytes(42, kH * kFF);
    CHECK(row && 3 * *row == kBlob);
    // artifact size
    CHECK(48ull * 512 == 24576);
    CHECK(48ull * 512 * kBlob == 33973862400ull);
    CHECK(size_ok(33973862400ull, 48, 512));
    CHECK(!size_ok(33973862399ull, 48, 512) && !size_ok(33973862401ull, 48, 512) && !size_ok(0, 48, 512));
    // offsets
    CHECK(blob_offset(512, 0, 0) == 0);
    CHECK(blob_offset(512, 0, 1) == kBlob);
    CHECK(blob_offset(512, 1, 0) == 512 * kBlob);
    CHECK(blob_offset(512, 47, 511) == (24576 - 1) * kBlob);
    CHECK(kQ8Chunks == 80 && kQ8HitBytes == 2720);
    // injectivity and range over in-range pairs
    for (uint64_t nl = 1; nl <= 6; ++nl)
        for (uint64_t ne = 1; ne <= 9; ++ne) {
            std::set<uint64_t> seen;
            for (uint64_t l = 0; l < nl; ++l)
                for (uint64_t e = 0; e < ne; ++e) {
                    uint64_t i = blob_index(ne, l, e);
                    CHECK(i < nl * ne);
                    CHECK(seen.insert(i).second);
                    CHECK(blob_ok(ne, nl * ne, l, e));  // accepted lookups stay in geometry
                    CHECK(blob_offset(ne, l, e) == i * kBlob);
                }
            CHECK(seen.size() == nl * ne);
        }
}

static void test_blob_ok_alias() {
    for (uint64_t n = 1; n <= 8; ++n)
        for (uint64_t nl = 1; nl <= 5; ++nl)
            for (uint64_t l = 0; l < nl + 1; ++l)
                for (uint64_t e = 0; e < 3 * n; ++e) {
                    bool ok = blob_ok(n, nl * n, l, e);
                    if (e >= n) {
                        CHECK(!ok);  // refused even when the flat index is in range
                        // the alias: flat(l, e) == flat(l+1, e-n)
                        CHECK(blob_index(n, l, e) == blob_index(n, l + 1, e - n));
                    } else {
                        CHECK(ok == (l < nl));
                    }
                }
    // explicit aliasing case: expert id == n_expert in layer 0 has an in-range flat index
    CHECK(blob_index(4, 0, 4) < 8 && !blob_ok(4, 8, 0, 4));
}

static void test_split() {
    for (bool hit : {false, true})
        for (int64_t v : {-5ll, 0ll, 1ll, 123456789ll}) {
            CHECK(cpu_row(hit, v) + gpu_row(hit, v) == v);  // sum equals the computed value
            if (hit) CHECK(cpu_row(true, v) == 0);
            else CHECK(gpu_row(false, v) == 0);
            CHECK((cpu_row(hit, v) == 0) || (gpu_row(hit, v) == 0));  // exactly one side owns it
        }
    std::vector<int> resident = {-1, 3, -1, 7};
    auto res = [&](uint32_t e) -> std::optional<uint32_t> {
        if (e < resident.size() && resident[e] >= 0) return static_cast<uint32_t>(resident[e]);
        return std::nullopt;
    };
    CHECK(!is_hit(res, 0) && is_hit(res, 1) && !is_hit(res, 2) && is_hit(res, 3) && !is_hit(res, 99));
    // combine
    CHECK(combine({}, {}) == 0);
    CHECK(combine({2, 3}, {5, 7}) == 2 * 5 + 3 * 7);
    CHECK(combine({4, 2, 3}, {1, 2, 3}) == 4 * 1 + combine({2, 3}, {2, 3}));
    // split + combine: hit/miss halves recombine to the all-computed result
    std::vector<int64_t> ws = {3, 1, 4, 1, 5}, vals = {9, 2, 6, 5, 3};
    for (int mask = 0; mask < 32; ++mask) {
        std::vector<int64_t> cpu, gpu;
        for (int i = 0; i < 5; ++i) {
            bool hit = (mask >> i) & 1;
            cpu.push_back(cpu_row(hit, vals[i]));
            gpu.push_back(gpu_row(hit, vals[i]));
        }
        CHECK(combine(ws, cpu) + combine(ws, gpu) == combine(ws, vals));
    }
}

static void test_wiring() {
    HitWiring w;
    CHECK(!w.ready());
    bool* fields[] = {&w.cache, &w.cache_base, &w.parts_out, &w.hit_out, &w.mixed,
                      &w.scratch, &w.xq, &w.xq_scale, &w.d_slot, &w.d_dst};
    for (bool* f : fields) *f = true;
    CHECK(w.ready());
    // all-or-nothing: clearing any single pointer makes it not ready
    for (bool* f : fields) {
        *f = false;
        CHECK(!w.ready());
        *f = true;
    }
    w.xq_scale = false;
    CHECK(!w.ready());
}

static void test_jobs() {
    for (size_t n = 0; n <= 10; ++n) {
        CHECK(count_miss(std::vector<bool>(n, false)) == n);  // no hits: every row is a job
        CHECK(count_miss(std::vector<bool>(n, true)) == 0);   // only hits: no jobs
    }
    for (int mask = 0; mask < 1024; ++mask) {
        std::vector<bool> hits;
        for (int i = 0; i < 10; ++i) hits.push_back((mask >> i) & 1);
        size_t misses = count_miss(hits);
        CHECK(misses <= hits.size());
        auto jobs = miss_jobs(hits);
        CHECK(jobs.size() == misses);  // job count equals the number of misses
        bool differs = false;
        for (size_t j = 0; j < jobs.size(); ++j) {
            CHECK(!hits[jobs[j]]);   // jobs are exactly the missed rows
            CHECK(jobs[j] >= j);     // job index never exceeds output index
            if (jobs[j] != j) differs = true;
            if (j) CHECK(jobs[j] > jobs[j - 1]);
        }
        // job index differs from output index whenever some row before a miss is a hit
        bool hit_before_miss = false, seen_hit = false;
        for (size_t i = 0; i < hits.size(); ++i) {
            if (hits[i]) seen_hit = true;
            else if (seen_hit) hit_before_miss = true;
        }
        CHECK(differs == hit_before_miss);
    }
}

static void test_counters_timing() {
    Counters c;
    uint32_t s = 3;
    for (int i = 0; i < 1000; ++i) {
        s = s * 1664525u + 1013904223u;
        bool r = (s >> 9) & 1, a = (s >> 13) & 1;
        uint64_t before = c.total();
        uint64_t h = c.hits, ad = c.admitted, rf = c.refused;
        c.step(r, a);
        CHECK(c.total() == before + 1);  // counted exactly once
        if (r) CHECK(c.hits == h + 1 && c.admitted == ad && c.refused == rf);
        else if (a) CHECK(c.admitted == ad + 1 && c.hits == h && c.refused == rf);
        else CHECK(c.refused == rf + 1 && c.hits == h && c.admitted == ad);
    }
    CHECK(c.total() == 1000);
    Timing t;
    for (int i = 0; i < 100; ++i) {
        uint64_t before = t.ready + t.late;
        t.record(i % 3 == 0);
        CHECK(t.ready + t.late == before + 1);
    }
    CHECK(t.ready == 34 && t.late == 66);
}

static void test_latch() {
    FailState f;
    CHECK(!f.failed && f.should_run());
    f.clear_msg();
    CHECK(!f.failed && f.should_run());  // clearing without a failure changes nothing
    f.raise("boom");
    CHECK(f.failed && f.msg && *f.msg == "boom" && !f.should_run());
    f.clear_msg();
    CHECK(!f.msg.has_value());
    CHECK(f.failed && !f.should_run());  // clearing the message does not re-arm
    f.clear_msg();
    f.clear_msg();
    CHECK(f.failed && !f.should_run());
    f.raise("again");
    CHECK(f.failed && *f.msg == "again");
}

static void test_pcie() {
    for (uint64_t nmiss = 0; nmiss <= 600; ++nmiss) {
        CHECK(pcie_share(nmiss, 0) == 0);
        CHECK(pcie_share(nmiss, 256) == nmiss);
        for (uint64_t p : {1ull, 64ull, 128ull, 200ull, 255ull, 256ull}) {
            CHECK(pcie_share(nmiss, p) <= nmiss);
            CHECK(pcie_share(nmiss, p) == (nmiss * p) / 256);
        }
        // monotone in the numerator
        for (uint64_t p = 1; p <= 256; ++p) CHECK(pcie_share(nmiss, p - 1) <= pcie_share(nmiss, p));
    }
    CHECK(pcie_share(10, 128) == 5);
}

int main() {
    test_geometry_and_addressing();
    test_blob_ok_alias();
    test_split();
    test_wiring();
    test_jobs();
    test_counters_timing();
    test_latch();
    test_pcie();
    if (g_fail) {
        std::fprintf(stderr, "expert_source_dispatch: %d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("expert_source_dispatch: all checks passed\n");
    return 0;
}


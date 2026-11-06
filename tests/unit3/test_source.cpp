#include "check.hpp"
#include "baloon/expert_source_blob_access.hpp"
#include "baloon/expert_dispatch_contract.hpp"
using namespace baloon;

void test_source_blob_index() {
    for (uint64_t ne = 1; ne <= 12; ++ne)
        for (uint64_t nl = 1; nl <= 8; ++nl) {
            uint64_t blobs = nl * ne;
            for (uint64_t l = 0; l <= nl; ++l)
                for (uint64_t e = 0; e <= ne + 2; ++e) {
                    auto r = expert_source::blob_index(ne, blobs, l, e);
                    bool valid = l < nl && e < ne;
                    CHECK(r.has_value() == valid);
                    if (r) CHECK(*r == l * ne + e && *r < blobs);
                }
            // expert == ne must be rejected even though the flat index aliases (layer+1, 0)
            for (uint64_t l = 0; l + 1 < nl; ++l) {
                CHECK(!expert_source::blob_index(ne, blobs, l, ne).has_value());
                CHECK(l * ne + ne == (l + 1) * ne);
                CHECK(expert_source::blob_index(ne, blobs, l + 1, 0).has_value());
            }
        }
    // a pack bigger than nl*ne still rejects expert >= ne
    CHECK(!expert_source::blob_index(4, 100, 2, 4).has_value());
    CHECK(expert_source::blob_index(4, 100, 2, 3) == std::optional<uint64_t>(11));
}

void test_source_pinned_and_dispatch() {
    for (uint64_t off = 0; off < 40; off += 3)
        for (uint64_t len = 0; len < 20; len += 4)
            for (uint64_t a = 0; a < 70; a += 5)
                for (uint64_t b = a; b < 70; b += 7) {
                    if (expert_source::pinned_ok(off, len, a)) CHECK(expert_source::pinned_ok(off, len, b));
                    CHECK(expert_source::pinned_ok(off, len, a) == (off + len <= a));
                }
    testh::Rng rng(5);
    for (int t = 0; t < 2000; ++t) {
        uint64_t k = rng.below(16);
        std::vector<bool> hit(k);
        for (uint64_t i = 0; i < k; ++i) hit[i] = rng.below(2);
        uint64_t m = expert_source::misses(hit), h = expert_source::hit_count(hit);
        CHECK(m + h == k && m <= k);
        auto rows = expert_source::job_output_rows(hit);
        CHECK(rows.size() == m);
        for (size_t j = 0; j < rows.size(); ++j) {
            CHECK(rows[j] < k && !hit[rows[j]]);          // output indexed by i, jobs by njobs
            if (j) CHECK(rows[j] > rows[j - 1]);
        }
        uint64_t n = 1 + rng.below(100);
        for (uint64_t i : rows) CHECK(i * n + n <= k * n);  // output row inside k x n_embd buffer
    }
    for (uint64_t nt = 0; nt <= 8; ++nt) CHECK(nt * 10 <= 8 * 10);
    for (uint64_t nm = 0; nm <= 64; ++nm)
        for (uint64_t num = 0; num <= 256; ++num) {
            uint64_t s = expert_source::pcie_share(nm, num);
            CHECK(s <= nm && s == ((nm * num) >> 8));
        }
    CHECK(expert_source::pcie_share(10, 256) == 10 && expert_source::pcie_share(10, 0) == 0);
}

void test_dispatch_contract() {
    using namespace expert_dispatch;
    for (int m = 0; m < 1024; ++m) {
        HitCfg c;
        c.cache = m & 1; c.cache_base = m & 2; c.parts_out = m & 4; c.hit_out = m & 8; c.mixed = m & 16;
        c.scratch = m & 32; c.xq = m & 64; c.xq_scale = m & 128; c.d_slot = m & 256; c.d_dst = m & 512;
        CHECK(hits_ready(c) == (m == 1023));  // all-or-nothing over the ten members
        if (hits_ready(c)) CHECK(c.xq_scale && c.cache && c.d_slot && c.d_dst);
    }
    CHECK(latch(true, false) && latch(true, true) && latch(false, true) && !latch(false, false));
    CHECK(step_layers(true, 7) == 7 && step_layers(false, 7) == 8);
    bool f = false;
    uint64_t layers = 0;
    for (int i = 0; i < 6; ++i) { f = latch(f, i == 2); layers = step_layers(f, layers); }
    CHECK(f && layers == 2);  // frozen from the failing step on
    CHECK(geometry_ok(2560, 2560) && !geometry_ok(2561, 2560) && !geometry_ok(0, 2560));
    PlanState st;
    uint64_t cap = 10;
    CHECK(plan_add_group(st, cap, 4) && plan_add_group(st, cap, 6) && st.entries == 10 && st.groups == 2);
    PlanState before = st;
    CHECK(!plan_add_group(st, cap, 1) && st.entries == before.entries && st.groups == before.groups);
    for (uint64_t f2 = 0; f2 < 100; ++f2)
        for (uint64_t sc : {0ull, 1ull, 10ull, 64ull, 128ull}) {
            bool ok = may_fetch(f2, sc);
            CHECK(ok == (f2 < sc && f2 < 64));
            if (ok) CHECK(f2 + 1 <= 64);
        }
}

#include <set>
#include "check.hpp"
#include "baloon/expert_source_dispatch.hpp"
using namespace baloon::source_dispatch;

void test_source_geometry() {
    CHECK(kPackBlobs == 24576);
    CHECK(kBlobBytes == 1382400);
    CHECK(kPackBytes == 33973862400ull);
    CHECK(48ull * 512ull * 1382400ull == kPackBytes);
    std::set<uint64_t> seen;
    for (uint64_t l = 0; l < 48; ++l) for (uint64_t e = 0; e < 512; ++e) {
        CHECK(blob_ok(48, 512, l, e));
        CHECK(blob_idx(512, l, e) < kPackBlobs);
        CHECK(seen.insert(blob_idx(512, l, e)).second);                  // injective, no padding
        CHECK(blob_offset(512, l, e, kBlobBytes) + kBlobBytes <= kPackBytes);
    }
    CHECK(seen.size() == kPackBlobs);
    CHECK(*seen.rbegin() == kPackBlobs - 1);
    // expert >= ne must be refused although its flat index aliases the next layer
    CHECK(blob_idx(512, 3, 512 + 5) == blob_idx(512, 4, 5));
    CHECK(!blob_ok(48, 512, 3, 517));
    CHECK(!blob_ok(48, 512, 48, 0));
}

void test_source_split() {
    testh::Rng rng(21);
    for (int it = 0; it < 500; ++it) {
        const size_t k = rng.below(12);
        std::vector<bool> hit(k);
        std::vector<float> e(k);
        for (size_t j = 0; j < k; ++j) { hit[j] = rng.below(2); e[j] = (float)(int)rng.below(100) - 50.f; }
        Split s = split_router(hit);
        CHECK(s.hits.size() + s.misses.size() == k);                      // partition
        CHECK(s.misses.size() <= k);
        std::set<size_t> all(s.hits.begin(), s.hits.end());
        for (size_t m : s.misses) CHECK(all.insert(m).second);            // disjoint
        CHECK(all.size() == k);
        std::vector<float> cpu, gpu, parts;
        combine_rows(hit, e, cpu, gpu, parts);
        for (size_t j = 0; j < k; ++j) {
            CHECK(hit[j] ? cpu[j] == 0.f : gpu[j] == 0.f);               // zeroed on the other side
            CHECK(parts[j] == e[j]);                                       // each expert exactly once
        }
    }
}

void test_source_pcie_latch() {
    for (uint64_t nm = 0; nm <= 64; ++nm)
        for (uint64_t pn = 0; pn <= 256; ++pn) {
            const uint64_t m = pcie_share(nm, pn);
            CHECK(m <= nm);
            CHECK(m == nm * pn / 256);
            CHECK(nm - pcie_first_rank(nm, m) == m);                      // last m ranks selected: exactly m
            CHECK(pcie_first_rank(nm, m) + m == nm);
        }
    CHECK(pcie_share(10, 256) == 10 && pcie_share(10, 0) == 0);
    for (int f = 0; f < 2; ++f) for (int ok = 0; ok < 2; ++ok) {
        CHECK(latch(f, ok) == (f || !ok));
        if (f) CHECK(latch(true, ok));                                    // sticky
        if (!ok) CHECK(latch(f, false));                                  // failure sets it
    }
    DispatchState st;
    st.step(true); CHECK(!st.failed);
    st.step(false, "boom"); CHECK(st.failed && st.msg == "boom");
    st.clear_message(); CHECK(st.failed && st.msg.empty());               // clearing the message does not re-arm
    st.step(true); CHECK(st.failed);
    st.step(false, "later"); CHECK(st.failed && st.msg.empty());   // already failed: stays failed, message not re-armed
    // pinned range
    CHECK(pinned_at(0, 10, 10)); CHECK(!pinned_at(1, 10, 10)); CHECK(pinned_at(10, 0, 10)); CHECK(!pinned_at(11, 0, 10));
    CHECK(!pinned_at(~0ull, 2, 100));                                     // overflow-safe
    for (uint64_t off = 0; off < 30; ++off) for (uint64_t b = 0; b < 30; ++b) {
        CHECK(pinned_at(off, b, 40) == (off + b <= 40));
        if (pinned_at(off, b, 40)) CHECK(off <= 40);
    }
}

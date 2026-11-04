// ExpertSourceDispatch theory invariants (LAYER 1), checked against brute-force oracles.
#include <set>
#include "check.hpp"
#include "baloon/expert_source_dispatch.hpp"
#include "baloon/expert_source_blob_access.hpp"
using namespace baloon::source_dispatch;

void test_blob_valid_iff() {
    // blobValid <-> layer < nl && expert < ne, exhaustively on a small grid (including out-of-range values)
    for (uint64_t nl = 0; nl <= 5; ++nl) for (uint64_t ne = 0; ne <= 5; ++ne)
        for (uint64_t l = 0; l <= 7; ++l) for (uint64_t e = 0; e <= 7; ++e) {
            const bool want = l < nl && e < ne;
            CHECK(blob_valid_flat(nl, ne, l, e) == want);
            CHECK(blob_ok(nl, ne, l, e) == want);
            const auto o = baloon::expert_source::blob_index(ne, nl * ne, l, e);
            CHECK(o.has_value() == want);
            if (want) CHECK(*o == blob_idx(ne, l, e));
            if (want) CHECK((blob_idx(ne, l, e) + 1) * 17 <= nl * ne * 17);   // blob_in_pack
        }
}

void test_flat_alias_insufficient() {
    for (uint64_t ne = 1; ne <= 40; ++ne) {
        CHECK(blob_idx(ne, 0, ne) == blob_idx(ne, 1, 0));                      // flat_alias
        for (uint64_t nl = 2; nl <= 6; ++nl) {                                 // flat_check_insufficient
            CHECK(blob_idx(ne, 0, ne) < nl * ne);
            CHECK(!blob_valid_flat(nl, ne, 0, ne));
        }
    }
}

void test_pack_size() {
    CHECK(48ull * 512 * 1382400 == 33973862400ull);
    CHECK(pack_size_ok(kPackBytes, kLayers, kExperts, kBlobBytes));
    CHECK(!pack_size_ok(kPackBytes - 1, kLayers, kExperts, kBlobBytes));
    CHECK(!pack_size_ok(kPackBytes + 1, kLayers, kExperts, kBlobBytes));
    CHECK(!pack_size_ok(0, kLayers, kExperts, kBlobBytes));
    CHECK(pack_size_ok(0, 0, 512, 1));
}

void test_job_compaction() {
    testh::Rng rng(1501);
    for (int it = 0; it < 2000; ++it) {
        const size_t k = rng.below(20);
        std::vector<bool> hit(k);
        for (size_t j = 0; j < k; ++j) hit[j] = rng.below(2);
        const uint64_t i0 = rng.below(5);
        CHECK(n_miss(hit) + n_hit(hit) == k);                                  // nMiss_add_nHit
        const auto rows = job_rows(hit, i0);
        CHECK(rows.size() == n_miss(hit));                                     // jobRows_length
        std::set<uint64_t> seen;
        uint64_t prev = 0; bool first = true;
        for (uint64_t r : rows) {
            CHECK(r >= i0 && r < i0 + k);                                      // mem_jobRows
            CHECK(seen.insert(r).second);                                      // nodup
            if (!first) CHECK(r > prev);                                       // in order
            prev = r; first = false;
        }
        for (size_t j = 0; j < k; ++j)                                         // hit_row_not_job & exactness
            CHECK(seen.count(i0 + j) == (hit[j] ? 0u : 1u));
        CHECK(rows == [&] { std::vector<uint64_t> v; for (uint64_t r : baloon::expert_source::job_output_rows(hit)) v.push_back(r + i0); return v; }());
        CHECK(n_miss(hit) == baloon::expert_source::misses(hit));
        CHECK(n_hit(hit) == baloon::expert_source::hit_count(hit));
    }
    CHECK(job_rows({}, 3).empty());
    CHECK(n_miss({true, true}) == 0 && job_rows({true, true}).empty());
    CHECK(job_rows({false, false, false}) == (std::vector<uint64_t>{0, 1, 2}));
}

void test_pcie_share() {
    for (uint64_t nm = 0; nm <= 600; ++nm) {
        CHECK(pcie_share(nm, 0) == 0);                                         // pcie_share_zero
        CHECK(pcie_share(nm, 256) == nm);
        for (uint64_t num = 0; num <= 256; ++num) {
            const uint64_t m = pcie_share(nm, num);
            CHECK(m <= nm);                                                    // pcie_share_le
            CHECK(m == nm * num / 256);
            CHECK(baloon::expert_source::pcie_share(nm, num) == m);
            if (m <= nm) CHECK(pcie_first_rank(nm, m) + m == nm);
        }
    }
}

void test_blob_split() {
    for (uint64_t up = 0; up <= 64; ++up) for (uint64_t blob = 0; blob <= 300; blob += 7) {
        const uint64_t down_off = 2 * up;
        if (down_off > blob) { CHECK(!split_tiles(blob, up, down_off)); continue; }
        CHECK(gate_bytes(up) + up_bytes(up) + down_bytes(blob, down_off) == blob);   // blob_split_tiles
        CHECK(split_tiles(blob, up, down_off));
        CHECK(!split_tiles(blob, up, down_off + 1));                           // wrong layout never tiles by declaration
    }
    for (uint64_t per = 0; per < 1000; per += 13) CHECK(per * 512 == per * 16 * 32);
    CHECK(kExpertsPerRead == 16 && reads_per_layer(512) == 32);
    CHECK(512 % kExpertsPerRead == 0);
}

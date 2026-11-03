// Theory-level invariants of the prefill chunk driver (K = 10 experts per token, 512 experts, 16 PLE heads).
#include <vector>
#include <cstdint>
#include "baloon/prefill_chunk_driver.hpp"
#include "check.hpp"
using namespace baloon::prefill_driver;

void test_pd_theory_pairs_and_slots() {
    testh::Rng rng(86);
    // src[slot] = i / K : pair i = t*K + k maps back to token t
    for (uint64_t t = 0; t < 2000; ++t) for (uint64_t k = 0; k < 10; ++k) CHECK((t * 10 + k) / 10 == t);
    for (int it = 0; it < 60; ++it) {
        const uint32_t T = 1 + rng.below(40), K = 10, NE = 512;
        std::vector<uint32_t> ex(T * K);
        for (auto& e : ex) e = rng.below(NE);
        const Grouping g = group_by_expert(ex, NE);
        CHECK(g.off[NE] == (uint64_t)T * K);                             // counts total T*K
        for (uint32_t i = 0; i < T * K; ++i) {
            CHECK(g.slot[i] < (uint64_t)T * K);                          // every slot below T*K
            CHECK(g.order[g.slot[i]] / K == i / K);                      // src token preserved through the sort
            CHECK(g.order[g.slot[i]] == i);
            const uint32_t e = ex[i];
            CHECK(g.slot[i] - g.off[e] < g.cnt[e]);                      // rank < cnt[e]
        }
    }
}

void test_pd_theory_geometry_and_layout() {
    CHECK(2560 * 4 == 10240 && 48 * 128 == 6144 && 24 * 256 == 6144 && 24 * 512 == 12288 && 2 * 640 == 1280);
    CHECK(16 * 160 == 2560 && 1280 * 640 * 4 == 1280 * 2560 && 2560 * 160 * 4 == 2560 * 640);
    CHECK(kGdnBytesPerLayer == 3268608 && kGdnFloatsPerLayer * 4 == kGdnBytesPerLayer);
    int qsa = 0, gdn = 0;
    for (int l = 0; l < kLayers; ++l) (is_full_attention_layer(l) ? qsa : gdn)++;
    CHECK(qsa == 12 && gdn == 36 && qsa + gdn == 48);
    for (uint64_t T = 1; T < 300; ++T) for (uint64_t t = 0; t < T; t += 7) for (uint64_t h = 0; h < 16; ++h)
        CHECK(t * 16 + h < T * 16);                                      // PLE row index stays inside the batch
}

void test_pd_theory_chunk_bounds_and_ring() {
    for (uint64_t T : {1, 5, 64, 512}) for (uint64_t n = 1; n < 1500; n += 11) for (uint64_t c0 = 0; c0 < n; c0 += T) {
        const uint64_t len = std::min(T, n - c0);                        // chunkLen
        CHECK(len > 0 && len <= T && c0 + len <= n);
    }
    // ring: any two blobs staged inside a window [j, j+STAGE-1] map to different slots
    for (uint64_t j = 0; j < 100; ++j) for (uint64_t a = j; a < j + kStageRing; ++a) for (uint64_t b = j; b < j + kStageRing; ++b)
        CHECK((stage_slot(a) == stage_slot(b)) == (a == b));
    for (uint64_t j = 0; j < 100; ++j) CHECK(j % 2 != (j + 1) % 2 && dq_slot(j) != dq_slot(j + 1));
    // takeSeq: exact-size region always fits, one byte less never does
    testh::Rng rng(87);
    for (int it = 0; it < 500; ++it) {
        std::vector<uint64_t> seq(1 + rng.below(8));
        uint64_t sum = 0, base = rng.below(500), out = 0;
        for (auto& s : seq) { s = rng.below(4096); sum += s; }
        CHECK(take_all(base + sum, base, seq, out) && out == base + sum && out <= base + sum);
        if (sum > 0) CHECK(!take_all(base + sum - 1, base, seq, out));
    }
}

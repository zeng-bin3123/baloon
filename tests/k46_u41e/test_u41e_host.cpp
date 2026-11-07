// Host invariants of unit u41e2bf114f on PRODUCTION code: block stage selection / PLE / fused path / layer kinds (src/core/layer.cpp),
// chunked prefill (src/prefill/prefill_host.cpp): allocation accounting, chunk walk, counting sort, rings.
#include <algorithm>
#include <cstdio>
#include <vector>
#include "baloon/core_layer.hpp"
#include "baloon/core_prefill.hpp"
#include "check.hpp"
using namespace baloon::core;

// independent oracle of the theory's runStage (written as a table of the five stages)
static bool oracle_run(int sp, int half, int i) {
    if (sp > 0) return i < sp;
    static const bool tab[3][5] = {{1, 1, 1, 1, 1}, {1, 1, 1, 0, 0}, {0, 0, 0, 1, 1}};
    return half >= 0 && half < 3 ? tab[half][i] : (i <= 2 ? false : false);
}

static void test_stage_selection() {
    for (int sp = 0; sp <= 6; ++sp)
        for (int half = 0; half <= 2; ++half)
            for (int i = 0; i < 5; ++i) CHECK(stage_runs(sp, half, i) == oracle_run(sp, half, i));
    for (int i = 0; i < 5; ++i) {                                  // halves partition the five stages
        CHECK(stage_runs(0, 1, i) != stage_runs(0, 2, i));
        CHECK(stage_runs(0, 0, i));
    }
    for (int k = 1; k <= 5; ++k) for (int half = 0; half < 3; ++half)       // prefix k: first k stages, downward closed
        for (int i = 0; i < 4; ++i) CHECK(!stage_runs(k, half, i + 1) || stage_runs(k, half, i));
    CHECK(!stage_runs(0, 0, 5) && !stage_runs(0, 0, -1));
    // PLE
    for (int layer = 0; layer < 48; ++layer) for (int half = 0; half < 3; ++half) {
        CHECK(!ple_applies(false, layer, half));
        CHECK(ple_applies(true, layer, half) == (layer == 1 && half != 2));
    }
    // fused path only for the whole call
    layer_set_fused_gr(true);
    for (int sp = 0; sp <= 5; ++sp) for (int half = 0; half < 3; ++half) for (int sup = 0; sup < 2; ++sup)
        CHECK(layer_fused_gr(sp, half, sup) == (sp == 0 && half == 0 && sup));
    layer_set_fused_gr(false);
    CHECK(!layer_fused_gr(0, 0, true));
    // R write
    for (int l = 0; l < 48; ++l) { CHECK(post_writes_r(false, l, 48)); CHECK(post_writes_r(true, l, 48) == (l == 47)); }
    // layer kinds
    int q = 0, g = 0;
    for (int l = 0; l < 48; ++l) { (is_qsa_layer(l) ? q : g)++; CHECK(is_qsa_layer(l) == (l % 4 == 3)); }
    CHECK(q == 12 && g == 36);
}

static void test_block_layer() {
    for (int fused = 0; fused < 2; ++fused) {
        layer_set_fused_gr(fused);
        for (int layer : {0, 1, 2, 3, 46, 47}) for (int sp = 0; sp <= 5; ++sp) for (int half = 0; half < 3; ++half) {
            BlockCall c; c.layer = layer; c.stage_prefix = sp; c.half = half; c.ple_ready = true;
            std::vector<int> ran; int wrote = 0, ple = 0;
            BlockHooks h;
            for (int i = 0; i < 5; ++i) h.stages.stage[i] = [&ran, i](int) { ran.push_back(i); };
            h.gr_write = [&] { ++wrote; };
            h.ple = [&] { ++ple; };
            const BlockResult r = block_layer(c, h);
            std::vector<int> want;
            for (int i = 0; i < 5; ++i) if (oracle_run(sp, half, i)) want.push_back(i);
            CHECK(ran == want);                                     // stages run in order, exactly the selected ones
            uint32_t m = 0; for (int i : want) m |= 1u << i;
            CHECK(r.stage_mask == m);
            CHECK(r.fused == (fused && sp == 0 && half == 0));
            const bool want_ple = layer == 1 && (half == 0 || half == 1) && oracle_run(sp, half, 2);
            CHECK((ple == 1) == want_ple && r.ple_ran == want_ple);
            const bool full = sp == 0 || sp >= 5;
            const bool want_w = full && half != 1 && (!r.fused || layer + 1 == 48);
            CHECK(r.wrote_r == want_w && wrote == (want_w ? 1 : 0));
        }
    }
    layer_set_fused_gr(false);
    BlockCall c; c.layer = 3; BlockHooks h;
    CHECK(qsa_layer(c, h) && !gdn_layer(c, h));
    c.layer = 4;
    CHECK(gdn_layer(c, h) && !qsa_layer(c, h));
}

static void test_alloc_and_chunks() {
    th::Rng rng(41);
    for (uint64_t b = 0; b < 3000; ++b) {
        const uint64_t a = round_bytes(b);
        CHECK(a % 256 == 0 && b + 256 <= a && a < b + 512 && a == (b + 256 + 255) / 256 * 256);
    }
    for (int it = 0; it < 400; ++it) {
        PrefillConfig c; c.T = 1 + rng.below(9); c.n_embd = 1 + rng.below(90); c.hc = 1 + rng.below(4); c.attn_dim = rng.below(90);
        c.kv_dim = rng.below(30); c.n_experts = 1 + rng.below(20); c.top_k = 1 + rng.below(4);
        c.blob_bytes = rng.below(500); c.dq_bytes = rng.below(900); c.gemm_scratch = 1 + rng.below(3000); c.gemm_ws = 1 + rng.below(3000);
        Prefill p(c);
        uint64_t sum = 0; for (uint64_t b : p.alloc_sequence()) sum += round_bytes(b);
        const uint64_t need = p.counted_bytes();
        CHECK(need == sum);
        CHECK(p.bytes_needed() == need + 8ull * 1048576 && p.bytes_needed() >= need);
        std::vector<uint8_t> region((size_t)need + 8);
        CHECK(p.init(region.data(), need) && p.used() == need);
        CHECK(!p.init(region.data(), need - 1) && !p.initialized());          // overflow by one byte fails exactly
        CHECK(p.init(region.data(), p.bytes_needed()) && p.used() == need);
        for (size_t i = 0; i < p.offsets().size(); ++i) {
            CHECK(p.offsets()[i] % 256 == 0);
            CHECK(p.offsets()[i] + round_bytes(p.alloc_sequence()[i]) <= need);
            if (i) CHECK(p.offsets()[i] == p.offsets()[i - 1] + round_bytes(p.alloc_sequence()[i - 1]));
        }
        CHECK(p.init(nullptr, 0) && p.used() == need);
    }
    // chunk walk
    for (uint64_t T : {1ull, 3ull, 8ull, 512ull}) for (uint64_t n : {0ull, 1ull, 7ull, 8ull, 9ull, 100ull, 1025ull}) {
        PrefillConfig c; c.T = T; c.n_layers = 4;
        Prefill p(c);
        uint64_t covered = 0; bool contiguous = true;
        PrefillHooks h; h.mixer = [&](int l, uint64_t c0, uint64_t t) { if (l == 0) { contiguous &= (c0 == covered); covered += t; } };
        PrefillStats st;
        CHECK(p.run(n, h, &st));
        uint64_t sum = 0; for (uint64_t x : st.chunk_lens) { CHECK(x >= 1 && x <= T); sum += x; }
        CHECK(sum == n && covered == n && contiguous && st.tokens == n);
        CHECK(st.chunk_lens.size() == (n + T - 1) / T);
        if (!st.chunk_lens.empty()) CHECK(st.chunk_lens.back() == n - T * (st.chunk_lens.size() - 1));
        CHECK(st.layer_calls == st.chunks * 4);
    }
    { PrefillConfig c; c.T = 0; Prefill p(c); CHECK(!p.run(3, PrefillHooks{})); }
}

static void test_grouping_and_rings() {
    th::Rng rng(42);
    for (int it = 0; it < 2000; ++it) {
        const uint32_t E = 1 + rng.below(40);
        std::vector<uint32_t> ex(rng.below(150));
        for (auto& e : ex) e = rng.below(E);
        ExpertGrouping g;
        CHECK(group_experts(ex, E, g));
        std::vector<uint64_t> cnt(E, 0); for (uint32_t e : ex) ++cnt[e];
        CHECK(g.off.size() == (size_t)E + 1 && g.off[0] == 0 && g.off[E] == ex.size());
        for (uint32_t e = 0; e < E; ++e) CHECK(g.off[e + 1] == g.off[e] + cnt[e]);
        std::vector<char> seen(ex.size(), 0);
        for (size_t i = 0; i < ex.size(); ++i) {
            CHECK(g.row[i] >= g.off[ex[i]] && g.row[i] < g.off[ex[i] + 1]);
            if (g.row[i] < seen.size()) { CHECK(!seen[g.row[i]]); seen[g.row[i]] = 1; }
        }
        for (size_t j = 0; j < g.order.size(); ++j) {
            CHECK(cnt[g.order[j]] > 0 && (j == 0 || g.order[j] > g.order[j - 1]));
            CHECK(g.stage_slot[j] == j % 8 && g.dq_slot[j] == j % 2);
            if (j + 1 < g.order.size()) CHECK(g.dq_slot[j] != g.dq_slot[j + 1]);
            for (size_t k = j + 1; k < std::min(g.order.size(), j + 8); ++k) CHECK(g.stage_slot[j] != g.stage_slot[k]);
        }
    }
    ExpertGrouping g;
    CHECK(!group_experts({0, 5}, 5, g));                 // id == NE is out of range
    CHECK(!group_experts({1}, 0, g));
    CHECK(group_experts({}, 4, g) && g.off.size() == 5 && g.off[4] == 0);
    CHECK(kPrefillStageRing == 8 && kPrefillLookahead + 1 == kPrefillStageRing && kPrefillDqRing == 2);
    CHECK(MAXBLOB({3, 9, 2}) == 9 && MAXBLOB({}) == 0);
}

int main() {
    test_stage_selection();
    test_block_layer();
    test_alloc_and_chunks();
    test_grouping_and_rings();
    std::printf("checks=%ld fails=%ld\n", th::checks(), th::fails());
    if (th::fails()) return 1;
    std::printf("U41E HOST TESTS PASSED\n");
    return 0;
}

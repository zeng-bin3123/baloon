#include <algorithm>
#include <vector>
#include "baloon/qsa_block_select.hpp"
#include "baloon/qsa_cache_indexer.hpp"
#include "check.hpp"
using namespace baloon::qsa_select;

void test_qb_weights() {
    for (uint32_t nkv = 1; nkv < 400; ++nkv) {
        const uint32_t nbid = nkv / 4;
        uint32_t total = 0;
        for (uint32_t b = 0; b < nbid; ++b) { CHECK(weight(nkv, nbid, b) == 4); total += 4; }
        CHECK(weight(nkv, nbid, nbid) == nkv % 4 && weight(nkv, nbid, nbid) == tail_weight(nkv) && tail_weight(nkv) < 4);
        total += weight(nkv, nbid, nbid);
        CHECK(total == nkv);
        for (uint32_t b = 0; b < nbid; ++b) for (uint32_t c = 0; c < 4; ++c) CHECK(cell_id(b, c) < nkv);
        for (uint32_t c = 0; c < weight(nkv, nbid, nbid); ++c) CHECK(cell_id(nbid, c) < nkv);
    }
    for (uint32_t b = 0; b < 50; ++b) for (uint32_t c = 0; c < 4; ++c) for (uint32_t b2 = 0; b2 < 50; ++b2) for (uint32_t c2 = 0; c2 < 4; ++c2)
        if (cell_id(b, c) == cell_id(b2, c2)) CHECK(b == b2 && c == c2);
}

void test_qb_digits() {
    testh::Rng r(5);
    for (int i = 0; i < 20000; ++i) {
        const uint32_t k = r.u32() << 16 ^ r.u32();
        CHECK(assemble(digit(k, 0), digit(k, 1), digit(k, 2), digit(k, 3)) == k);
        for (uint32_t p = 0; p < 4; ++p) CHECK(digit(k, p) < 256);
        const uint32_t d3 = r.below(256), d2 = r.below(256), d1 = r.below(256), d0 = r.below(256);
        const uint32_t a = assemble(d3, d2, d1, d0);
        CHECK(digit(a, 0) == d3 && digit(a, 1) == d2 && digit(a, 2) == d1 && digit(a, 3) == d0);
    }
}

static std::vector<uint32_t> brute_cells(const std::vector<uint32_t>& keys, uint32_t nkv, uint32_t nbid, uint32_t w) {
    struct C { uint32_t id, key; };
    std::vector<C> cs;
    for (size_t b = 0; b < keys.size(); ++b) for (uint32_t c = 0; c < weight(nkv, nbid, (uint32_t)b); ++c) cs.push_back({cell_id((uint32_t)b, c), keys[b]});
    std::stable_sort(cs.begin(), cs.end(), [](const C& a, const C& b) { return a.key > b.key; });   // ties: lowest id first
    std::vector<uint32_t> out;
    for (size_t i = 0; i < cs.size() && i < w; ++i) out.push_back(cs[i].id);
    std::sort(out.begin(), out.end());
    return out;
}

void test_qb_radix() {
    testh::Rng r(11);
    for (int it = 0; it < 800; ++it) {
        const uint32_t nkv = 1 + r.below(700);
        const uint32_t nbid = nkv / 4;
        const uint32_t nb = nbid + (nkv % 4 ? 1 : 0);
        const uint32_t w = 1 + r.below(nkv + 3);
        std::vector<uint32_t> keys(nb);
        const int mode = it % 4;
        for (auto& k : keys) k = mode == 0 ? (r.u32() << 16 ^ r.u32()) : mode == 1 ? 0x80000000u + r.below(4) : mode == 2 ? (0x7FFFFF00u + r.below(512)) : (r.below(3) ? 0xFFFFFFFFu : 0u);
        const auto got = select_cells(keys, nkv, nbid, w);
        const auto want = brute_cells(keys, nkv, nbid, w);
        CHECK(got == want);
        CHECK(got.size() == std::min(nkv, w));
        CHECK(std::is_sorted(got.begin(), got.end()));
        for (uint32_t id : got) CHECK(id < nkv);
        if (nkv > w) {
            std::vector<uint32_t> wts(nb);
            for (uint32_t b = 0; b < nb; ++b) wts[b] = weight(nkv, nbid, b);
            const Radix rd = radix_threshold(keys, wts, w);
            CHECK(rd.above < w && w <= rd.above + rd.eq);          // descent invariant
            CHECK(w - rd.above > 0 && w - rd.above <= rd.eq);       // equal budget positive and satisfiable
            // matches the 32-pass lifting threshold of the per-cell keys
            std::vector<uint32_t> cellkeys;
            for (uint32_t b = 0; b < nb; ++b) for (uint32_t c = 0; c < wts[b]; ++c) cellkeys.push_back(keys[b]);
            const auto t = baloon::qsa_cache::lift_threshold(cellkeys, w);
            CHECK(t.thr == rd.thr && t.gt == rd.above && t.eq == rd.eq);
        }
    }
    // identity branch
    std::vector<uint32_t> k = {5, 1, 9};
    CHECK(select_cells(k, 10, 2, 10).size() == 10 && select_cells(k, 10, 2, 12).size() == 10);
}

void test_qb_grids() {
    for (uint32_t mb : {1u, 7u, 8u, 9u, 513u}) for (uint32_t b = 0; b < mb; ++b) CHECK(b / 8 < score_grid(mb));
    for (uint32_t bx = 0; bx < 20; ++bx) for (uint32_t w = 0; w < 8; ++w) { const uint32_t b = score_block(bx, w); CHECK(b / 8 == bx && b % 8 == w); }
    for (uint32_t nb : {1u, 100u, 256u, 257u, 513u}) {
        std::vector<int> cov(nb, 0); const uint32_t per = thread_range(nb);
        for (uint32_t t = 0; t < 256; ++t) for (uint32_t b = t * per; b < std::min(nb, t * per + per); ++b) cov[b]++;
        for (int c : cov) CHECK(c == 1);
        for (uint32_t b = 0; b < nb; ++b) CHECK(b / per < 256);
    }
    for (uint32_t d = 0; d < 128; ++d) CHECK(d / 4 < 32 && (d / 4) * 4 + d % 4 == d);
}

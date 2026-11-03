#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <vector>
#include "baloon/qsa_cache_indexer.hpp"
#include "check.hpp"
using namespace baloon::qsa_cache;

void test_qc_step_width() {
    for (uint32_t pos = 0; pos < 6000; ++pos) {
        const Step s = step_fill(pos);
        CHECK(s.n_kv == pos + 1 && s.n_bid == (pos + 1) / 4);
        CHECK(s.width == std::min<uint32_t>(pos + 1, 2051));
        CHECK(s.width <= s.n_kv);
        if (s.n_kv <= 2051) CHECK(s.width == s.n_kv); else CHECK(s.width == 2051);
        CHECK(s.n_bid * 4 <= pos + 1);
        CHECK(tail_len(pos) < 4 && tail_len(pos) == (pos + 1) % 4);
        CHECK((pos / 4) * 4 + slot_of(pos) == pos);
        CHECK(completes_block(pos) == ((pos + 1) % 4 == 0));
        // cell j < n_kv lies in block <= n_bid; block b < n_bid owns cells 4b..4b+3 all < n_kv
        CHECK(block_of(pos) <= s.n_bid);
        if (pos % 17 == 0) for (uint32_t b = 0; b < s.n_bid; b += 37) for (uint32_t c = 0; c < 4; ++c) CHECK(b * 4 + c < pos + 1);
    }
    CHECK(step_fill(0).n_bid == 0 && step_fill(3).n_bid == 1 && step_fill(2).n_bid == 0);
    CHECK(width(10, 8, 2) == 9);
    for (uint32_t cap : {1u, 3u, 4u, 5u, 1000u, 4096u}) for (uint32_t pos = 0; pos < cap; pos += 3) CHECK(pos / 4 + 1 <= pooled_rows(cap));
}

void test_qc_paging() {
    for (uint64_t ps : {1ull, 16ull, 64ull, 256ull}) for (uint64_t cell = 0; cell < 2000; cell += 7) {
        CHECK(ps * cell_page(cell, ps) + cell_slot(cell, ps) == cell);
        CHECK(cell_slot(cell, ps) < ps);
    }
    // bijective row mapping over (page, head, slot)
    const uint64_t P = 5, H = 3, PS = 16;
    std::vector<int> hit(P * H * PS, 0);
    for (uint64_t p = 0; p < P; ++p) for (uint64_t h = 0; h < H; ++h) for (uint64_t s = 0; s < PS; ++s) {
        const uint64_t r = page_row(p, H, h, PS, s);
        CHECK(r < hit.size()); if (r < hit.size()) hit[r]++;
    }
    for (int c : hit) CHECK(c == 1);
    for (uint32_t nkv : {1u, 2u, 4u}) for (uint32_t g : {12u}) for (uint32_t h = 0; h < g * nkv; ++h) CHECK(kv_head_of(h, g * nkv, nkv) < nkv);
}

static float from_bits(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }

void test_qc_order_key() {
    CHECK(order_key(0x80000000u) == 0x7FFFFFFFu && order_key(0u) == 0x80000000u);
    testh::Rng r(1);
    std::vector<float> v = {0.f, -0.f, 1.f, -1.f, 1e-30f, -1e-30f, 3.4e38f, -3.4e38f,
                            std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::denorm_min()};
    for (int i = 0; i < 4000; ++i) {
        const float f = from_bits(r.u32() << 16 ^ r.u32());
        if (std::isnan(f)) continue;
        v.push_back(f);
    }
    for (size_t i = 0; i < v.size(); ++i) for (size_t j = 0; j < v.size(); j += 3) {
        const float a = v[i], b = v[j];
        const uint32_t ka = order_key_f(a), kb = order_key_f(b);
        if (a < b) CHECK(ka < kb);
        if (a > b) CHECK(ka > kb);
        if (a == b && a != 0.f) CHECK(ka == kb);
    }
    // every negative key is below every non-negative key; branches
    for (int i = 0; i < 2000; ++i) {
        const uint32_t neg = 0x80000000u | (r.u32() << 16 ^ r.u32()) , pos = (r.u32() << 16 ^ r.u32()) & 0x7FFFFFFFu;
        CHECK(order_key(neg) == 0xFFFFFFFFu - neg && order_key(pos) == pos + 0x80000000u);
        CHECK(order_key(neg) < order_key(pos));
        const uint32_t pos2 = std::max(pos, (r.u32() << 16 ^ r.u32()) & 0x7FFFFFFFu);
        CHECK(order_key(pos) <= order_key(pos2));
        CHECK(order_key(std::max(neg, 0x80000000u)) <= order_key(0x80000000u));
    }
    // involution-free injectivity on a sample
    std::set<uint32_t> ks; std::set<uint32_t> bs;
    for (int i = 0; i < 5000; ++i) { uint32_t b = r.u32() << 16 ^ r.u32(); bs.insert(b); ks.insert(order_key(b)); }
    CHECK(ks.size() == bs.size());
}

static std::vector<uint32_t> brute_topk(const std::vector<uint32_t>& keys, uint32_t w) {
    std::vector<uint32_t> idx(keys.size());
    for (uint32_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) { return keys[a] > keys[b]; });  // ties: lowest index first
    idx.resize(std::min<size_t>(w, idx.size()));
    std::sort(idx.begin(), idx.end());
    return idx;
}

void test_qc_topk() {
    testh::Rng r(7);
    for (int it = 0; it < 600; ++it) {
        const uint32_t n = 1 + r.below(300);
        const uint32_t w = 1 + r.below(n + 5);
        const uint32_t range = (it % 3 == 0) ? 3 : (it % 3 == 1 ? 40 : 0xFFFFFFFFu);   // heavy ties in 2/3 of cases
        std::vector<uint32_t> keys(n);
        for (auto& k : keys) k = (range == 0xFFFFFFFFu) ? (r.u32() << 16 ^ r.u32()) : 0x80000000u + r.below(range);
        const auto got = topk_ids(keys, w);
        const auto want = brute_topk(keys, w);
        CHECK(got == want);
        CHECK(got.size() == std::min(n, w));
        CHECK(std::is_sorted(got.begin(), got.end()));
        if (n > w) {
            const Threshold t = lift_threshold(keys, w);
            CHECK(t.gt <= w && w <= t.gt + t.eq);
            CHECK(selected_count(t.gt, t.eq, w) == w);
            CHECK(got.size() == selected_count(t.gt, t.eq, w));
        }
    }
    for (uint32_t gt = 0; gt <= 20; ++gt) for (uint32_t eq = 0; eq <= 20; ++eq) for (uint32_t w = gt; w <= 25; ++w) {
        CHECK(selected_count(gt, eq, w) <= w);
        if (w <= gt + eq) CHECK(selected_count(gt, eq, w) == w);
        CHECK(gt + eq_budget(gt, w) == w);
    }
    // all-equal keys: first w indices win
    std::vector<uint32_t> same(50, 12345u);
    const auto ids = topk_ids(same, 7);
    CHECK(ids.size() == 7); for (uint32_t i = 0; i < 7; ++i) CHECK(ids[i] == i);
}

void test_qc_misc() {
    for (uint32_t n : {1u, 255u, 256u, 257u, 2051u, 5000u}) {
        std::vector<int> hit(n, 0);
        for (uint32_t j = 0; j < n; ++j) { CHECK(chunk_owner(j, n) < 256); hit[j]++; }
        for (uint32_t t = 0; t < 256; ++t) { (void)t; }
        // threads' ranges [t*cs, t*cs+cs) cover [0,n) exactly once
        std::vector<int> cov(n, 0); const uint32_t cs = chunk_size(n);
        for (uint32_t t = 0; t < 256; ++t) for (uint32_t j = t * cs; j < std::min(n, t * cs + cs); ++j) { cov[j]++; CHECK(chunk_owner(j, n) == t); }
        for (int c : cov) CHECK(c == 1);
    }
    for (uint32_t h = 0; h < 96; ++h) CHECK((h / 12) * 12 + h % 12 == h);
    for (uint32_t d = 0; d < 128; ++d) CHECK(d < 2 * 128 && 128 + d < 2 * 128);
}

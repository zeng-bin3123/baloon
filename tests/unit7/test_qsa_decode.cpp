#include <algorithm>
#include <cmath>
#include <set>
#include <vector>
#include "baloon/qsa_decode_attention.hpp"
#include "check.hpp"
using namespace baloon::qsa_decode;

void test_qd_chunks() {
    for (uint32_t cap = 1; cap < 5000; cap += 3) {
        const uint32_t nc = chunks(cap);
        CHECK(cap <= nc * 64 && (nc - 1) * 64 < cap);
    }
    CHECK(chunks(0) == 0 && chunks(64) == 1 && chunks(65) == 2 && chunks(2051) == 33);
    for (uint32_t n = 0; n < 300; ++n) {
        uint32_t total = 0;
        for (uint32_t c = 0; c < chunks(std::max(n, 1u)) + 2; ++c) {
            const uint32_t h = n_here(n, c * 64);
            CHECK(h <= 64);
            if (n <= c * 64) CHECK(h == 0);
            total += h;
        }
        CHECK(total == n);   // full blocks plus the last partial one add up to n_ids
    }
    CHECK(batch_ok(65535) && !batch_ok(65536));
}

void test_qd_partitions() {
    for (uint32_t h = 0; h < 24; ++h) { CHECK(kv_head(h) * 12 + local_head(h) == h); CHECK(local_head(h) < 12 && kv_head(h) < 2); }
    for (uint32_t nkv : {1u, 2u, 8u}) for (uint32_t h = 0; h < 12 * nkv; ++h) CHECK(kv_head(h) < nkv);
    std::vector<int> cells(64, 0);
    for (uint32_t w = 0; w < 8; ++w) for (uint32_t m = 0; m < 8; ++m) cells[w + 8 * m]++;
    for (uint32_t c = 0; c < 64; ++c) { CHECK(cells[c] == 1); CHECK(cell_warp(c) + 8 * cell_iter(c) == c && cell_warp(c) < 8 && cell_iter(c) < 8); }
    for (uint32_t d = 0; d < 256; ++d) { CHECK(dim_lane(d) < 32 && dim_slot(d) < 8 && dim_lane(d) * 8 + dim_slot(d) == d); CHECK(scale_group(d) < 8); }
    for (uint32_t h = 0; h < 12; ++h) CHECK(head_warp(h) < 8 && head_iter(h) < 2 && head_warp(h) + 8 * head_iter(h) == h);
    for (uint32_t c = 0; c < 64; ++c) CHECK(c < 32 || c - 32 < 32);
    for (uint64_t P : {1ull, 4ull}) for (uint64_t page = 0; page < P; ++page) for (uint64_t h = 0; h < 2; ++h) for (uint64_t s = 0; s < 16; ++s)
        CHECK(pool_row(page, 2, h, 16, s) < P * 2 * 16);
}

void test_qd_scratch() {
    for (uint32_t nc : {1u, 5u, 33u}) for (uint32_t nh : {12u, 24u}) {
        const uint64_t x = x_of(nc, nh);
        CHECK(x * 256 + x + x == x * 258 && query_stride(nc, nh) == x * 258);
        // every scratch slot of one query is hit exactly once by (acc, m, l) addressing
        std::vector<int> hit(query_stride(nc, nh), 0);
        for (uint32_t c = 0; c < nc; ++c) for (uint32_t h = 0; h < nh; ++h) {
            for (uint32_t d = 0; d < 256; ++d) hit[acc_off(c, h, nh, d)]++;
            hit[m_off(nc, nh, c, h)]++; hit[l_off(nc, nh, c, h)]++;
        }
        for (int v : hit) CHECK(v == 1);
        for (uint32_t q = 0; q < 4; ++q) CHECK(query_base(q, nc, nh) + query_stride(nc, nh) == query_base(q + 1, nc, nh));
    }
}

void test_qd_merge() {
    testh::Rng rng(99);
    const size_t hd = 16;
    for (int it = 0; it < 200; ++it) {
        const uint32_t n = rng.below(300);          // includes n == 0
        std::vector<float> s(n), v((size_t)n * hd);
        for (auto& x : s) x = (float)(rng.gauss() * (it % 5 == 0 ? 30.0 : 2.0));
        for (auto& x : v) x = (float)rng.gauss();
        // direct softmax in double
        std::vector<double> ref(hd, 0.0);
        if (n) {
            double m = -1e300; for (float x : s) m = std::max(m, (double)x);
            double L = 0; for (uint32_t i = 0; i < n; ++i) { const double e = std::exp(s[i] - m); L += e; for (size_t d = 0; d < hd; ++d) ref[d] += e * v[i * hd + d]; }
            for (auto& x : ref) x /= L;
        }
        const uint32_t nc = chunks(std::max<uint32_t>(n, 1)) + (it % 2);   // an extra trailing empty chunk on odd iterations
        std::vector<Partial> ps;
        for (uint32_t c = 0; c < nc; ++c) {
            const uint32_t nh = n_here(n, c * 64);
            ps.push_back(nh == 0 ? chunk_partial(nullptr, nullptr, 0, hd)
                              : chunk_partial(s.data() + (size_t)c * 64, v.data() + (size_t)c * 64 * hd, nh, hd));
            if (nh == 0) CHECK(ps.back().l == 0.f && ps.back().acc[0] == 0.f);
        }
        const auto out = merge(ps, hd);
        for (size_t d = 0; d < hd; ++d) CHECK_MSG(std::fabs(out[d] - ref[d]) < 1e-4 * (1 + std::fabs(ref[d])), "it=%d n=%u d=%zu got=%g want=%g", it, n, d, out[d], ref[d]);
    }
    // an empty chunk is neutral in the merge
    Partial a = chunk_partial(std::vector<float>{1.f, 2.f}.data(), std::vector<float>{3.f, 4.f}.data(), 2, 1);
    auto m1 = merge({a}, 1), m2 = merge({a, empty_partial(1)}, 1), m3 = merge({empty_partial(1), a, empty_partial(1)}, 1);
    CHECK(m1[0] == m2[0] && m1[0] == m3[0]);
    // chunk order must not matter
    Partial b = chunk_partial(std::vector<float>{0.5f}.data(), std::vector<float>{-2.f}.data(), 1, 1);
    CHECK(std::fabs(merge({a, b}, 1)[0] - merge({b, a}, 1)[0]) < 1e-6);
}

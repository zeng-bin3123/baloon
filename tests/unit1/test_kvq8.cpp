#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <vector>
#include "baloon/kv_q8_quantization.hpp"
#include "check.hpp"

using namespace baloon::kvq8;
using baloon::f16::float_to_half;
using baloon::f16::half_to_float;

void test_kvq8_clamp() {
    for (int64_t q = -100000; q <= 100000; ++q) {
        const int64_t c = clamp_q(q);
        CHECK(c >= -127 && c <= 127 && c >= -128);
        if (q >= -127 && q <= 127) CHECK(c == q);
        if (q < -127) CHECK(c == -127);
        if (q > 127) CHECK(c == 127);
        CHECK((int64_t)(int8_t)c == c);  // no wrap on the int8 cast
    }
}

void test_kvq8_quantize_group() {
    std::mt19937_64 rng(99);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int it = 0; it < 20000; ++it) {
        float x[64];
        const float mag = std::ldexp(1.f, (int)(rng() % 20) - 10);
        for (auto& v : x) v = nd(rng) * mag;
        if (it % 7 == 0) x[rng() % 64] *= 50.f;  // outlier
        if (it % 11 == 0) for (auto& v : x) v = 0.f;
        int8_t q[64];
        uint16_t sb;
        quantize_group(x, q, &sb);
        float amax = 0.f;
        for (float v : x) amax = std::fmax(amax, std::fabs(v));
        CHECK(sb == float_to_half(amax / 127.f));
        int maxq = 0;
        for (int i = 0; i < 64; ++i) {
            CHECK(q[i] >= -127 && q[i] <= 127);                       // never -128
            maxq = std::max(maxq, std::abs((int)q[i]));
            const float back = dequant_value(q[i], sb);
            // |x - q*sh| <= s/2 (code rounding) + 127*|s - sh| (stored fp16 scale error). The fp16 error is
            // relative 2^-11 for normal scales and absolute 2^-25 once the scale is subnormal (< 2^-14).
            const float s = amax / 127.f;
            const float bound = s / 2.f + 127.f * std::fmax(s * std::ldexp(1.f, -11), std::ldexp(1.f, -25)) + 1e-9f;
            CHECK_MSG(std::fabs(back - x[i]) <= bound, "x=%g back=%g amax=%g bound=%g", x[i], back, amax, bound);
        }
        if (amax > 0.f) CHECK(maxq == 127);                           // the extreme value uses the full range
        else { CHECK(sb == 0); for (int i = 0; i < 64; ++i) CHECK(q[i] == 0); }
    }
    // saturation: huge dynamic range never produces -128
    float x[64];
    for (int i = 0; i < 64; ++i) x[i] = (i % 2 ? -1.f : 1.f) * (i == 3 ? 1e4f : 1e-3f);
    int8_t q[64]; uint16_t sb;
    quantize_group(x, q, &sb);
    for (int i = 0; i < 64; ++i) CHECK(q[i] >= -127);
    CHECK(q[3] == -127);
}

void test_kvq8_threads_and_groups() {
    for (int t = 0; t < kBlockThreads; ++t) CHECK(warp_of_thread(t) < kWarpsPerBlock);
    for (int groups = 1; groups <= 8; ++groups) {
        const int head_dim = 64 * groups;
        CHECK(groups_per_head(head_dim) == groups);
        for (int d = 0; d < head_dim; ++d) CHECK(d / 64 < groups);
        for (int g = 0; g < groups; ++g)
            for (int t = 0; t < 64; ++t) CHECK(g * 64 + t < head_dim);
        for (int q = 0; q < quads_per_head(head_dim); ++q) {
            CHECK(4 * q + 3 < head_dim);
            CHECK((4 * q) / 64 == (4 * q + 3) / 64);  // the quad stays inside its group
        }
    }
}

void test_kvq8_append_addressing() {
    for (int64_t kv_heads : {1, 2, 3})
        for (int64_t ps : {1, 4, 7, 16}) {
            const int64_t n_pages = 5;
            std::vector<int32_t> perm(n_pages);
            for (int i = 0; i < n_pages; ++i) perm[i] = (int32_t)((i * 3 + 1) % n_pages);  // a permutation
            std::set<int64_t> rows;
            for (int64_t pos = 0; pos < n_pages * ps; ++pos) {
                CHECK(ps * (pos / ps) + pos % ps == pos);
                CHECK(pos % ps < ps);
                for (int64_t h = 0; h < kv_heads; ++h) {
                    const AppendAddr a = append_address(perm.data(), pos, h, kv_heads, ps);
                    CHECK(a.page_index == pos / ps && a.cell == pos % ps);
                    CHECK(a.slab == (int64_t)perm[pos / ps] * kv_heads + h);
                    CHECK(a.row == (a.slab * kv_heads / kv_heads) * ps + a.cell);
                    CHECK(a.row >= a.slab * ps && a.row < (a.slab + 1) * ps);  // inside its slab
                    CHECK(a.row < n_pages * kv_heads * ps);
                    CHECK(rows.insert(a.row).second);                          // no two (pos, head) collide
                }
            }
            CHECK((int64_t)rows.size() == n_pages * kv_heads * ps);            // and together they tile the cache
        }
}

void test_kvq8_append_token_end_to_end() {
    const int kv_heads = 2, head_dim = 128, ps = 8, n_pages = 4, groups = head_dim / 64;
    const int32_t table[n_pages] = {2, 0, 3, 1};
    const size_t rows = (size_t)n_pages * kv_heads * ps;
    std::vector<int8_t> codes(rows * head_dim, (int8_t)-128);  // -128 is never a legal code
    std::vector<uint16_t> scales(rows * groups, 0xFFFF);
    std::mt19937_64 rng(5);
    std::normal_distribution<float> nd(0.f, 3.f);
    std::vector<float> src((size_t)n_pages * ps * kv_heads * head_dim);
    for (auto& v : src) v = nd(rng);
    for (int pos = 0; pos < n_pages * ps; ++pos)
        append_token(&src[(size_t)pos * kv_heads * head_dim], codes.data(), scales.data(), table, pos, kv_heads, head_dim, ps);
    for (int8_t c : codes) CHECK(c != -128);                 // every code slot written; none is -128
    for (uint16_t s : scales) CHECK(s != 0xFFFF);
    for (int pos = 0; pos < n_pages * ps; ++pos)
        for (int h = 0; h < kv_heads; ++h) {
            const AppendAddr a = append_address(table, pos, h, kv_heads, ps);
            for (int d = 0; d < head_dim; ++d) {
                const float* grp = &src[(size_t)pos * kv_heads * head_dim + (size_t)h * head_dim + (size_t)(d / 64) * 64];
                float amax = 0.f;
                for (int k = 0; k < 64; ++k) amax = std::fmax(amax, std::fabs(grp[k]));
                const float x = grp[d % 64];
                const float back = dequant_value(codes[(size_t)a.row * head_dim + d], scales[(size_t)a.row * groups + d / 64]);
                const float sc = amax / 127.f;
                CHECK(std::fabs(back - x) <= sc / 2.f + 127.f * std::fmax(sc * std::ldexp(1.f, -11), std::ldexp(1.f, -25)) + 1e-9f);
            }
        }
}

void test_kvq8_gather_decomposition() {
    for (int64_t kv_heads : {1, 2, 4})
        for (int64_t head_dim : {64, 128, 256}) {
            const int64_t per = head_dim / 4, m = kv_heads * per, n_ids = 5;
            for (int64_t i = 0; i < m * n_ids; ++i) {
                const GatherIdx g = gather_decompose(i, kv_heads, head_dim);
                CHECK(g.id < n_ids && g.head < kv_heads && g.quad < per);
                CHECK(g.id * m + g.head * per + g.quad == i);        // recomposes
                CHECK(m * (i / m) + i % m == i);
                CHECK(4 * g.quad + 3 < head_dim);
            }
        }
}

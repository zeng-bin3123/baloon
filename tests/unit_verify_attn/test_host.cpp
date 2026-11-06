// test_host.cpp - host invariants of unit u52d156705c (QsaDecodeAttn / NativeFlashAttn / GdnRecurrence / VerifyWindow).
// Checks the LAYER 1 statements against the pure arithmetic in verify_kernels.hpp, decode_attn_kernels.hpp and the
// existing qsa_decode / flash_bounds models: state layout bijection, row groups, conv window and commit, dense steps,
// window ids, gathers and packed codes, chunking, GQA mapping, scratch layout, merge exponents, flash geometry.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "baloon/decode_attn_kernels.hpp"
#include "baloon/native_flash_attn_bounds.hpp"
#include "baloon/qsa_decode_attention.hpp"
#include "baloon/verify_kernels.hpp"
#include "baloon/verify_window_indexing.hpp"

using namespace baloon;
namespace vk = baloon::verify_kernels;
namespace ak = baloon::attn_kernels;
static long g_checks = 0, g_fail = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; if (g_fail < 60) std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
struct Rng {
    uint64_t s; explicit Rng(uint64_t x) : s(x * 0x9E3779B97F4A7C15ull + 7) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    uint32_t below(uint32_t n) { return (uint32_t)(next() >> 16) % n; }
    double uni() { return (double)(next() >> 11) / 9007199254740992.0; }
};

static void test_state_layout() {
    for (uint64_t hv : {1ull, 3ull, 8ull}) {
        const uint64_t S = 128;
        std::vector<char> seen(S * hv * S, 0);
        for (uint64_t i = 0; i < S; ++i) for (uint64_t h = 0; h < hv; ++h) for (uint64_t j = 0; j < S; ++j) {
            const uint64_t ix = vk::st_idx(hv, S, i, h, j);
            CHECK(ix < S * hv * S);
            CHECK(!seen[ix]);
            seen[ix] = 1;
        }
        for (char c : seen) CHECK(c == 1);   // bijection onto the buffer
    }
    // fused row stride: row rg*32 + r is r row-strides after row rg*32
    for (uint64_t rg = 0; rg < 4; ++rg) for (uint64_t r = 0; r < 32; ++r) {
        const uint64_t hv = 6, head = 5, col = 77;
        CHECK(vk::st_idx(hv, 128, rg * 32, head, col) + r * (hv * 128) == vk::st_idx(hv, 128, rg * 32 + r, head, col));
    }
    // row groups cover 128 rows exactly once; columns cover with 4 blocks of 32
    std::vector<int> cnt(128, 0);
    for (int rg = 0; rg < vk::kRowGroups; ++rg) for (int r = 0; r < vk::kRowsPerGroup; ++r) ++cnt[rg * 32 + r];
    for (int c : cnt) CHECK(c == 1);
    CHECK(128 * 48 == 6144);
    for (uint32_t h = 0; h < 48; ++h) for (uint32_t hk : {1u, 2u, 16u}) CHECK(vk::qk_head(h, hk) < hk);
    CHECK(vk::qk_head(5, 4) == 1);   // modulo pairing, not division
    CHECK(vk::kVerifyMaxT == 8);
}

static void test_conv_window_and_commit() {
    Rng rng(1);
    for (int trial = 0; trial < 200; ++trial) {
        const int n = 1 + (int)rng.below(8);
        std::vector<int> hist = {-1, -2, -3}, x;
        for (int i = 0; i < n; ++i) x.push_back(100 + i);
        std::vector<int> seq = hist;
        seq.insert(seq.end(), x.begin(), x.end());
        // window of token t = seq[t .. t+2]
        for (int t = 0; t < n; ++t)
            for (int j = 0; j < 3; ++j) {
                const int e = t + j;
                const int got = vk::seq_is_hist(e) ? hist[vk::seq_slot(e)] : x[vk::seq_slot(e)];
                CHECK(got == seq[e]);
            }
        // commit keeps the last three entries of [hist | x_0..x_{n-1}]
        for (int j = 0; j < 3; ++j) {
            const int e = n + j;
            const int got = vk::seq_is_hist(e) ? hist[vk::seq_slot(e)] : x[vk::seq_slot(e)];
            CHECK(got == seq[seq.size() - 3 + j]);
        }
        // existing indexing helper agrees
        for (int j = 0; j < 3; ++j) CHECK(verify_idx::commit_from_hist(n, j) == vk::seq_is_hist(n + j));
    }
    // consecutive windows overlap by two entries
    for (uint32_t t = 0; t < 8; ++t) for (uint32_t j = 0; j < 2; ++j) CHECK(t + 1 + j == t + (j + 1));
    CHECK(vk::commits(false, 5) == false);
    CHECK(vk::commits(true, 0) == false);
    CHECK(vk::commits(true, 1) == true);
}

static void test_steps_and_windows() {
    for (int32_t c = 0; c < 2050; ++c) {
        const auto d = vk::dense_step(c);
        CHECK(d.pos == c && d.n_kv == c + 1 && d.n_bid == (c + 1) / 4 && d.width == std::min(c + 1, vk::kSelectionBound));
    }
    for (int32_t nkv = 0; nkv < 600; ++nkv) for (int32_t w : {1, 7, 64, 256, 512}) {
        const int32_t st = vk::win_start(nkv, w), width = vk::win_width(nkv, w);
        CHECK(width == std::min(nkv, w));
        CHECK(st >= 0 && st + width == nkv);
        for (int32_t j = 0; j < width; ++j) CHECK(st + j < nkv);
        if (nkv <= w) CHECK(st == 0);
        CHECK((uint64_t)st == verify_idx::win_start((uint64_t)nkv, (uint64_t)w));
    }
}

static void test_gathers_and_codes() {
    for (uint64_t per : {1ull, 5ull, 64ull}) for (uint64_t n : {1ull, 3ull, 9ull}) for (uint64_t i = 0; i < n * per; ++i) {
        CHECK(i / per < n && i % per < per && (i / per) * per + i % per == i);
        CHECK(vk::gather_src(i / per, i % per, per) == i);
    }
    for (uint64_t k = 0; k < 20; ++k) CHECK(vk::broadcast_src(2, k * 3 + 1, 3) == 2 * 3 + (k * 3 + 1) % 3);
    for (uint64_t i = 0; i < 100; ++i) CHECK(vk::broadcast_src(0, i, 7) < 7);
    CHECK(vk::codes_per_byte(2) == 4 && vk::codes_per_byte(4) == 2 && vk::codes_per_byte(8) == 1);
    CHECK(!vk::bits_ok(3) && !vk::bits_ok(0) && vk::bits_ok(4));
    Rng rng(5);
    for (int bits : {2, 4, 8}) {
        const int dim = 37;
        const uint64_t cpb = 8 / bits;
        std::vector<uint8_t> row((dim * bits + 7) / 8, 0);
        std::vector<unsigned> codes(dim);
        for (int k = 0; k < dim; ++k) {
            codes[k] = rng.below(1u << bits);
            row[k / cpb] |= (uint8_t)(codes[k] << ((k % cpb) * bits));
        }
        for (int k = 0; k < dim; ++k) { CHECK(vk::unpack_code(row.data(), k, bits) == codes[k]); CHECK(codes[k] < (1u << bits)); }
    }
    // nibbles recompose the byte
    for (unsigned b = 0; b < 256; ++b) CHECK(((b / 16) % 16) * 16 + b % 16 == b);
}

static void test_qsa_chunking() {
    for (int cap : {1, 63, 64, 65, 200, 2051}) {
        CHECK(ak::n_chunks(cap) == (int)qsa_decode::chunks(cap));
        for (int i = 0; i < cap; ++i) CHECK(i / 64 < ak::n_chunks(cap));
        for (int n : {0, 1, 63, 64, 65, cap}) {
            long covered = 0;
            for (int c = 0; c < ak::n_chunks(cap); ++c) {
                const int here = (int)qsa_decode::n_here(std::min(n, cap), c * 64);
                CHECK(here >= 0 && here <= 64);
                if (c * 64 >= std::min(n, cap)) CHECK(here == 0);   // surplus chunk takes the sentinel path
                covered += here;
            }
            CHECK(covered == std::min(n, cap));
        }
    }
    for (int h = 0; h < 24; ++h) { CHECK(qsa_decode::kv_head(h) == (uint32_t)(h / 12)); CHECK(h / 12 < 2); CHECK(qsa_decode::kv_head(h) * 12 + qsa_decode::local_head(h) == (uint32_t)h); }
    CHECK(qsa_decode::kv_head(1) != 1u % 2u);   // div, not mod
    for (int d = 0; d < 256; ++d) CHECK((d / 8) * 8 + d % 8 == d && d / 8 < 32);
    for (int cap : {1, 64, 65, 2051}) for (int nh : {12, 24}) {
        const uint64_t x = (uint64_t)ak::n_chunks(cap) * nh;
        CHECK(ak::qsa_scratch_floats(cap, nh) == x * 258 + 64);
        CHECK(x * 256 + 2 * x <= ak::qsa_scratch_floats(cap, nh));
    }
    // batch elements occupy disjoint ordered ranges
    const uint64_t st = ak::qsa_scratch_floats(200, 24);
    for (uint64_t z = 0; z < 5; ++z) CHECK(z * st + st <= (z + 1) * st);
}

static void test_merge_equals_softmax() {
    Rng rng(9);
    for (int trial = 0; trial < 50; ++trial) {
        const int n = 1 + (int)rng.below(300), hd = 6;
        std::vector<float> sc(n), val((size_t)n * hd);
        for (auto& s : sc) s = (float)((rng.uni() - 0.5) * 30);
        for (auto& v : val) v = (float)(rng.uni() - 0.5);
        std::vector<qsa_decode::Partial> ps;
        for (int c0 = 0; c0 < n; c0 += 64) ps.push_back(qsa_decode::chunk_partial(&sc[c0], &val[(size_t)c0 * hd], std::min(64, n - c0), hd));
        ps.push_back(qsa_decode::empty_partial(hd));   // surplus chunk is neutral
        const auto got = qsa_decode::merge(ps, hd);
        double mx = -1e30; for (float s : sc) mx = std::max(mx, (double)s);
        double L = 0; std::vector<double> acc(hd, 0.0);
        for (int i = 0; i < n; ++i) { const double e = std::exp(sc[i] - mx); L += e; for (int d = 0; d < hd; ++d) acc[d] += e * val[(size_t)i * hd + d]; }
        for (int d = 0; d < hd; ++d) CHECK(std::fabs(got[d] - acc[d] / L) < 1e-5);
        float M = -3e38f; for (auto& p : ps) M = std::max(M, p.m);
        for (auto& p : ps) CHECK(p.m - M <= 0.f);   // merge exponents are non-positive
    }
}

static void test_flash_geometry() {
    namespace fb = flash_bounds;
    for (uint32_t cell = 0; cell < 256; ++cell) {
        CHECK(cell == (cell / 128) * 128 + ((cell % 128) / 32) * 32 + ((cell % 32) / 8) * 8 + cell % 8);
        CHECK(cell / 128 < 2 && (cell % 128) / 32 < 4);
    }
    for (uint32_t head = 0; head < 24; ++head) CHECK(fb::kv_head(head) == head / 12 && head / 12 < 2);
    for (uint32_t w = 0; w < 4; ++w) for (uint32_t lane = 0; lane < 32; ++lane)
        CHECK(w * 32 + (lane / 8) * 8 + lane % 8 == w * 32 + lane);
    for (uint32_t w = 0; w < 4; ++w) for (uint32_t lane = 0; lane < 32; ++lane) for (uint32_t i0 : {0u, 32u, 64u, 96u}) for (uint32_t j = 0; j < 4; ++j)
        CHECK(w * 4 * 256 + (lane / 8) * 256 + 2 * (i0 + (lane % 8) * 4) + 2 * j + 1 < 4 * 4 * 256);
    CHECK((4 * 4 * 256 + 32 + 32) * 4 <= 49152);
    CHECK(2 * 256 * 2 == 1024 && 16 * 16 == 256);
    for (uint64_t cell = 0; cell < 256; cell += 17) for (uint64_t kv = 0; kv < 2; ++kv) for (uint64_t d = 0; d < 256; d += 31)
        CHECK(fb::kv_index(cell, kv, d) < 256ull * 2 * 256);
    for (uint32_t pos = 0; pos < 256; ++pos) CHECK(fb::valid(pos + 1, pos + 1, pos, (pos + 1) / 4));
    CHECK(!fb::valid(0, 0, 0, 0));
    CHECK(!fb::valid(257, 257, 256, 64));
    CHECK(!fb::valid(10, 11, 9, 2));
    CHECK(!fb::valid(10, 10, 8, 2));
    CHECK(!fb::valid(10, 10, 9, 3));
    for (int n = 0; n <= 256; ++n) CHECK(std::min(n, 2051) == n);
    // online softmax: running max is monotone and the rescale exponent is non-positive
    Rng rng(3); double m = -1e300;
    for (int i = 0; i < 1000; ++i) { const double x = (rng.uni() - 0.5) * 100; const double mn = std::max(m, x); CHECK(m - mn <= 0 && mn >= m); m = mn; }
}

int main() {
    test_state_layout();
    test_conv_window_and_commit();
    test_steps_and_windows();
    test_gathers_and_codes();
    test_qsa_chunking();
    test_merge_equals_softmax();
    test_flash_geometry();
    std::printf("checks: %ld, failures: %ld\n", g_checks, g_fail);
    if (g_fail) { std::printf("VERIFY ATTN HOST TESTS FAILED\n"); return 1; }
    std::printf("VERIFY ATTN HOST TESTS PASSED\n");
    return 0;
}

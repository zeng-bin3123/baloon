// Unit-14 theory invariants (LAYER 1) of the ten modules, checked exhaustively or over dense random samples.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "baloon/kern_geometry.hpp"

using namespace baloon::kern;
static int g_fail = 0, g_checks = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; std::printf("FAIL line %d: %s\n", __LINE__, #c); } } while (0)

static void t_sgemv() {
    using namespace sgemv;
    std::mt19937_64 rng(1);
    for (uint64_t i = 0; i < 4096; ++i) {
        CHECK(q8k_off(i) + 32 < (i / 256 + 1) * 292);
        CHECK(q8k_off(i) < 292 * (i / 256 + 1));
        CHECK(q8_0_off(i) < 34 * (i / 32 + 1));
    }
    for (uint64_t i = 0; i < 4096; ++i)       // injectivity + monotonic
        for (uint64_t j = i + 1; j < i + 3 && j < 4096; ++j) { CHECK(q8k_off(i) != q8k_off(j)); CHECK(q8_0_off(i) != q8_0_off(j)); }
    // hoisting: 16 elements from a multiple of 16 share a block and consecutive offsets
    for (uint64_t i = 0; i < 4096; i += 16)
        for (uint32_t k = 0; k < 16; ++k) {
            CHECK((i + k) / 256 == i / 256 && (i + k) % 256 == i % 256 + k && i % 256 + 16 <= 256);
            CHECK((i + k) / 32 == i / 32 && (i + k) % 32 == i % 32 + k && i % 32 + 16 <= 32);
            CHECK(q8k_off(i + k) == q8k_off(i) + k && q8_0_off(i + k) == q8_0_off(i) + k);
        }
    // lane iterations: aligned and injective, payload bound
    std::vector<uint64_t> seen;
    for (uint64_t lane = 0; lane < 32; ++lane)
        for (uint64_t t = 0; t < 8; ++t) { const uint64_t s = lane_iter_start(lane, t); CHECK(s % 16 == 0); seen.push_back(s); CHECK(4 + s % 256 + 15 < 4 + 256); }
    std::sort(seen.begin(), seen.end());
    CHECK(std::adjacent_find(seen.begin(), seen.end()) == seen.end());
    CHECK(seen.back() + 16 == 32 * 16 * 8);    // 8 iterations of 512 cover 4096 elements with no hole
    // same-group requirement and the Q8_0 split hazard
    for (uint32_t g : {16u, 32u, 64u}) CHECK(lane_same_group(g));
    CHECK(!lane_same_group(4));
    CHECK(15 / 4 != 0 / 4);
    for (uint32_t shift : {4u, 5u, 6u}) for (uint64_t i = 0; i < 1000; ++i) CHECK(group_index(i, shift) == i / (1u << shift));
    for (uint32_t gshift : {4u, 5u, 6u}) for (uint64_t a = 0; a < 64; ++a) for (uint32_t k = 0; k < 16; ++k) CHECK((16 * a + k) / (16u << (gshift - 4)) == (a * 16 + k) >> gshift);
    // bit order of packed codes
    for (uint32_t b = 0; b < 256; ++b) CHECK(b == code_at(b, 0, 2) + 4 * code_at(b, 1, 2) + 16 * code_at(b, 2, 2) + 64 * code_at(b, 3, 2));
    for (uint32_t b0 = 0; b0 < 256; b0 += 3) for (uint32_t b1 = 0; b1 < 256; b1 += 5) {
        const uint64_t w = b0 + 256u * b1;
        CHECK(code_at(w, 0, 4) == (b0 & 15) && code_at(w, 1, 4) == (b0 >> 4) && code_at(w, 2, 4) == (b1 & 15) && code_at(w, 3, 4) == (b1 >> 4));
    }
    const uint64_t w8 = 0x04030201ull;
    for (uint32_t k = 0; k < 4; ++k) CHECK(code_at(w8, k, 8) == k + 1);
    for (uint32_t B : {2u, 4u, 8u}) CHECK(4 * bytes_per_word(B) == 16 * B / 8);
    for (uint64_t n = 0; n < 5000; ++n) { CHECK(n <= grid_rows128(n) * 128); CHECK(n <= grid_rows_warps(n, 8) * 8); }
}

static void t_dotalg() {
    using namespace dotalg;
    std::mt19937 rng(2);
    for (int rep = 0; rep < 200; ++rep) {
        const size_t n = 1 + rng() % 200;
        std::vector<int32_t> c(n), x(n), cm(n);
        int64_t hx = 0;
        for (size_t i = 0; i < n; ++i) { c[i] = rng() % 4; x[i] = (int)(rng() % 255) - 127; cm[i] = c[i] - 1; hx += x[i]; }
        CHECK(dot(cm.data(), x.data(), n) == dot(c.data(), x.data(), n) - hx);      // Q2_0 bias identity
        const size_t k = rng() % (n + 1);                                          // chunked dot
        CHECK(dot(c.data(), x.data(), n) == dot(c.data(), x.data(), k) + dot(c.data() + k, x.data() + k, n - k));
    }
    for (uint32_t b = 0; b < 256; ++b) {
        const uint32_t e = expand(b);
        CHECK((e & 255) == (b & 3) && ((e >> 8) & 255) == ((b >> 2) & 3) && ((e >> 16) & 255) == ((b >> 4) & 3) && (e >> 24) == ((b >> 6) & 3));
    }
    for (uint32_t c = 0; c < 80; ++c) CHECK(chunk_scale_index(c) < 40);
    for (uint32_t lanes : {2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u})
        for (int rep = 0; rep < 20; ++rep) {
            std::vector<int64_t> p(lanes); int64_t tot = 0;
            for (auto& v : p) { v = (int)(rng() % 2001) - 1000; tot += v; }
            CHECK(tree_sum(p, lanes) == tot);
        }
}

static void t_s2() {
    using namespace s2layout;
    CHECK(GU_CHUNKS == 80 && GU_CHUNKS_PER_LANE == 3 && D_CHUNKS == 20 && GU_ROWS == 1280);
    for (uint32_t i = 0; i < 1280; ++i) { CHECK(i * 640ull + 640 <= O_D_CODES); CHECK(O_GU_SCALES + i * 80ull + 80 <= O_D_SCALES); }
    for (uint32_t r = 0; r < 2560; ++r) { CHECK(O_D_CODES + r * 160ull + 160 <= O_GU_SCALES); CHECK(O_D_SCALES + r * 20ull + 20 <= BLOB_BYTES); }
    for (uint32_t c = 0; c < 80; ++c) CHECK(c == c % 32 + 32 * (c / 32) && c / 32 < 3);
    // gate-major mapping: injective over (hit, slot); gates below N, ups in [N, 2N)
    for (uint64_t n : {1ull, 3ull, 8ull}) {
        const uint64_t N = n * FF;
        std::vector<char> hit(2 * N, 0);
        for (uint64_t h = 0; h < n; ++h)
            for (uint64_t i = 0; i < 1280; ++i) {
                const uint64_t p = gate_major(N, h, i);
                CHECK(p < 2 * N && !hit[p]);
                hit[p] = 1;
                CHECK(i % 2 == 0 ? p < N : (p >= N && p < 2 * N));
            }
        for (char c : hit) CHECK(c == 1);   // the mapping is onto: no hole
    }
    // ballot compaction: positions of set lanes are 0..k-1 in lane order
    std::mt19937_64 rng(3);
    for (int rep = 0; rep < 500; ++rep) {
        const uint64_t m = rng();
        uint32_t want = 0;
        for (uint32_t lane = 0; lane < 64; ++lane)
            if ((m >> lane) & 1) { CHECK(ballot_prefix(m, lane) == want); ++want; }
        CHECK(want == (uint32_t)__builtin_popcountll(m));
    }
    for (uint64_t n = 0; n < 3000; ++n) { CHECK(r16(n) >= n && r16(n) % 16 == 0 && r16(n) < n + 16); CHECK(r256(n) >= n && r256(n) % 256 == 0 && r256(n) < n + 256); }
    for (uint64_t n = 1; n < 20; ++n) CHECK(r16(n * 5120) == n * 5120);
}

static void t_mmvq() {
    using namespace mmvqgeo;
    for (Trait t : {q5_k, q2_0, q3_k, q6_k, iq4_xs, q4_0_like, q8_0}) CHECK(t.threads_per_block * t.blocks_per_iter == BLOCK_THREADS);
    for (Trait t : {q5_k, q2_0, q3_k, q6_k, iq4_xs, q8_0}) CHECK(t.threads_per_block * two_warp_bpi(t.blocks_per_iter) == 64);
    CHECK(two_warp_bpi(8) == 4 && two_warp_bpi(64) == 32 && two_warp_bpi(4) == 2 && two_warp_bpi(16) == 8 && two_warp_bpi(32) == 16);
    for (uint64_t n = 0; n <= 8192; n += 32) {
        CHECK(small_k(n, 32, q4_0_like.blocks_per_iter) == (n < 2048));
        CHECK(small_k(n, 32, q8_0.blocks_per_iter) == (n < 1024));
    }
    for (uint64_t n = 0; n <= 16384; n += 256) {
        CHECK(small_k(n, 256, q5_k.blocks_per_iter) == (n < 2048));      // Q4_K / Q5_K / Q3_K
        CHECK(small_k(n, 256, iq4_xs.blocks_per_iter) == (n < 4096));
        CHECK(small_k(n, 256, q6_k.blocks_per_iter) == (n < 1024));
    }
    for (uint64_t n = 0; n <= 16384; n += 64) CHECK(small_k(n, 64, q2_0.blocks_per_iter) == (n < 4096));
    for (uint32_t tid = 0; tid < 128; ++tid) for (Trait t : {q5_k, q2_0, q6_k, iq4_xs, q8_0}) { CHECK(tid / t.threads_per_block < t.blocks_per_iter); CHECK(t.threads_per_block * (tid / t.threads_per_block) + tid % t.threads_per_block == tid); }
    for (uint64_t n = 0; n < 3000; ++n) { CHECK(n <= grid_rows(n, 4) * 4); CHECK(n <= grid_rows(n, 2) * 2); }
    for (uint64_t c = 0; c < 8; ++c) for (uint64_t m : {1ull, 5ull, 80ull}) for (uint64_t i = 0; i < 32 * m; i += 7) {
        CHECK(column_block(c, 32 * m, i) == c * m + i / 32);
        CHECK((c * 32 * m + i) % 32 == i % 32);
    }
    CHECK(q81_bytes(8, 2560) == 8 * 80 * 36);
    for (int t : {2, 6, 8, 11, 12, 13, 14, 20, 23, 42, 16, 17, 18, 21, 22, 29}) CHECK(find_weight(t) != nullptr);
    CHECK(find_weight(3) == nullptr);
    for (const auto& e : weight_table) CHECK(256 % e.elems == 0);
}

static void t_router() {
    using namespace routersel;
    std::mt19937 rng(4);
    for (int rep = 0; rep < 20000; ++rep) {
        const Cand a{(double)(int)(rng() % 7) - 3, (uint32_t)(rng() % 9)}, b{(double)(int)(rng() % 7) - 3, (uint32_t)(rng() % 9)}, c{(double)(int)(rng() % 7) - 3, (uint32_t)(rng() % 9)};
        if (a.i != b.i) { const Cand x = comb(a, b), y = comb(b, a); CHECK(x.v == y.v && x.i == y.i); }   // commutative for distinct indices
        const Cand l = comb(comb(a, b), c), r = comb(a, comb(b, c));
        if (a.i != b.i && b.i != c.i && a.i != c.i) CHECK(l.v == r.v && l.i == r.i);                       // associative
    }
    CHECK(comb({2, 7}, {2, 3}).i == 3 && comb({2, 3}, {2, 7}).i == 3);     // ties go to the lowest index
    CHECK(comb({1, 0}, {2, 5}).i == 5 && comb({2, 5}, {1, 0}).i == 5);
    for (uint32_t n = 1; n < 50; ++n) for (uint32_t wi = 0; wi <= n; ++wi) CHECK(greedy_out(wi, n) < n);
    for (uint32_t c0 : {0u, 255u, 256u, 0x7fffffffu, 0xffffffffu}) { CHECK(philox_mantissa(c0) < 16777216u); CHECK(philox_uniform_of(c0) >= 0.f && philox_uniform_of(c0) < 1.f); }
    for (float l : {-5.f, -0.5f, 0.f, 0.5f, 3.f, 100.f}) for (float r : {1.f, 1.1f, 2.f, 10.f}) CHECK(repeat_penalty(l, r) <= l);
    for (uint32_t len = 0; len < 40; ++len) for (uint32_t last_n = 0; last_n < 50; ++last_n) { const uint32_t s = window_start(len, last_n); CHECK(s <= len && len - s == std::min(len, last_n)); }
    for (uint32_t keep = 1; keep < 20; ++keep) for (uint32_t cut = 1; cut <= keep; ++cut) for (uint32_t mk = 1; mk < 25; ++mk) { const uint32_t a = cut_adjust(cut, mk, keep); CHECK(a >= 1 && a <= keep); }
    for (uint32_t n = 1; n <= 1024; ++n) {
        const uint32_t t = router_threads(n);
        CHECK(t >= 32 && t <= 512 && t % 32 == 0 && std::min(n, 512u) <= t && (t + 31) / 32 <= 16);
        CHECK(pad16(n) % 16 == 0 && (pad16(n) + 8 * n) % 4 == 0 && n <= pad16(n));
    }
    CHECK(router_smem(512) == 6656);
    CHECK(clamp_denominator(0.0) > 0 && clamp_denominator(5.0) == 5.0);
}

static void t_ple() {
    using namespace plegeo;
    const uint32_t want[4] = {0, 3, 6, 9};
    for (uint32_t k = 0; k < 4; ++k) { CHECK(tap_row(9, 4, 3, k) == want[k]); CHECK((tap_row(9, 4, 3, k) == 9) == (k == 3)); }
    for (uint32_t kern = 1; kern < 6; ++kern) for (uint32_t dil = 1; dil < 5; ++dil) for (uint32_t nh = (kern - 1) * dil; nh < 20; ++nh) if (nh > 0)
        for (uint32_t k = 0; k < kern; ++k) CHECK((tap_row(nh, kern, dil, k) == nh) == (k == kern - 1));
    for (uint32_t k = 0; k < 4; ++k) for (uint32_t c = 0; c < 10240; c += 97) CHECK(kernel_weight_index(k, c) < 40960);
    for (uint32_t r = 0; r < 9; ++r) for (uint32_t c = 0; c < 10240; c += 97) CHECK(state_index(r, c) < 9u * 10240);
    CHECK(state_index(1, 0) != (uint64_t)1 * 10240 + 0);                     // not transposable
    CHECK(NG_HC_DIM == 10240 && scratch_bytes() == 223152 && float_ws_floats() == 53764);
    for (uint32_t c = 0; c < 4; ++c) CHECK(c * 2560 + 2560 <= 10240);
    for (uint32_t i = 0; i < 10240; ++i) CHECK(i % 2560 < 2560 && i / 2560 < 4);
    std::mt19937 rng(5);
    for (int rep = 0; rep < 20000; ++rep) {
        const uint64_t a = rng() % 64, ab = 1 + rng() % 32, b = rng() % 64, bb = 1 + rng() % 32;
        const bool want_o = a < b + bb && b < a + ab;
        CHECK(overlap(a, ab, b, bb) == want_o);
        CHECK(overlap(a, ab, b, bb) == overlap(b, bb, a, ab));
    }
    for (float s : {-2.f, -0.f, 0.f, 3.f}) CHECK(sgn(-s) == -sgn(s));
    // history advance: shift left by one, append
    std::vector<int> h = {1, 2, 3, 4};
    std::vector<int> adv(h.begin() + 1, h.end()); adv.push_back(9);
    CHECK(adv.size() == h.size() && adv.back() == 9 && adv[0] == 2);
}

static void t_actq() {
    using namespace actq;
    for (int v = -300; v <= 300; ++v) {
        CHECK(clamp_i(-127, 127, v) >= -127 && clamp_i(-127, 127, v) <= 127);
        if (v >= -127 && v <= 127) CHECK(clamp_cpu(v) == v);
        CHECK(clamp_ggml(v) >= -128 && clamp_ggml(v) <= 127);
        if (v >= -127 && v <= 127) CHECK(clamp_cpu(-v) == -clamp_cpu(v));   // symmetric range closed under negation
    }
    CHECK(clamp_ggml(-200) == -128 && clamp_cpu(-200) == -127);
    CHECK(16 * 127 == 2032 && 2032 <= 32767);
    CHECK(Q8_0_BYTES == 34 && Q8_1_BYTES == 36 && Q8_K_BYTES == 292 && 260 % 2 == 0);
    for (uint32_t b = 0; b < 100; ++b) CHECK((b * 292) % 4 == 0 && (b * 34) % 2 == 0);
    std::mt19937 rng(6);
    for (int rep = 0; rep < 100000; ++rep) {
        const float x = (float)((double)(rng() % 8000000) / 1000.0 - 4000.0) / 1.0f;   // |x| < 2^22
        CHECK(nearest_int(x) == (int)std::nearbyint(x));
    }
    for (int n = -4194304; n < 4194304; n += 4099) CHECK(nearest_int((float)n) == n);
    CHECK(nearest_int(2.5f) == 2 && nearest_int(3.5f) == 4 && nearest_int(-2.5f) == -2);
    for (uint64_t nb = 0; nb < 1000; ++nb) { CHECK(nb <= grid_q8_0(nb) * 128); CHECK(nb <= grid_q8_k(nb) * 64); CHECK(nb <= grid_q8_1(nb) * 256); }
    for (uint32_t n_in : {32u, 64u, 2560u}) for (uint32_t c = 0; c < 8; ++c) for (uint32_t i = 0; i < n_in; i += 5) CHECK((c * n_in + i) / 32 == c * (n_in / 32) + i / 32);
}

static void t_q2geo() {
    using namespace q2geo;
    CHECK(tail_mask_bits(0) == 0);
    for (uint32_t nb = 0; nb <= 8; ++nb) CHECK(tail_mask_bits(nb) < 65536);
    CHECK(SC_GU % 8 == 0 && SC_D % 8 == 2 && (SC_GU + 7) / 8 == 5 && (SC_D + 7) / 8 == 2);
    for (uint32_t n = 1; n <= MAXT; ++n) CHECK(dispatch_nt(n) == n);
    CHECK(dispatch_nt(8) == 8 && dispatch_nt(9) == 8);
    for (uint32_t r = 0; r < FF; ++r) { CHECK(O_GU_CODES + (2 * r + 1) * (uint64_t)ROW_GU + ROW_GU <= O_D_CODES); CHECK(O_GU_SCALES + (2 * r + 1) * (uint64_t)SC_GU * 2 + SC_GU * 2 <= O_D_SCALES); }
    CHECK(H / QKA == MAXC && FF / QKA == 20);
}

static void t_pool() {
    using namespace poolsched;
    for (uint64_t total : {1ull, 7ull, 640ull, 2560ull, 5120ull}) for (uint64_t parts = 1; parts <= 40; ++parts) {
        CHECK(row_start(total, parts, 0) == 0 && row_start(total, parts, parts) == total);
        uint64_t covered = 0;
        for (uint64_t p = 0; p < parts; ++p) { CHECK(row_start(total, parts, p) <= row_start(total, parts, p + 1)); covered += row_start(total, parts, p + 1) - row_start(total, parts, p); }
        CHECK(covered == total);    // ranges tile [0, total)
    }
    for (uint64_t per : {1ull, 5ull, 640ull}) for (uint64_t n : {1ull, 3ull}) for (uint64_t tasks : {1ull, 4ull, 9ull, 30ull}) {
        const uint64_t total = per * n; uint64_t rows = 0;
        for (uint64_t i = 0; i < tasks; ++i) {
            const uint64_t g0 = row_start(total, tasks, i), g1 = row_start(total, tasks, i + 1);
            uint64_t got = 0;
            for (const Piece& p : pieces(per, g0, g1)) { CHECK(p.r0 < p.r1 && p.r1 <= per && p.expert < n); got += p.r1 - p.r0; }
            CHECK(got == g1 - g0);  // the pieces cover exactly the task's range and never overshoot
            rows += got;
        }
        CHECK(rows == total);
    }
    for (uint32_t i = 0; i < 100; ++i) for (uint32_t parts = 1; parts < 10; ++parts) { CHECK(parts * (i / parts) + i % parts == i); CHECK(i % parts < parts); }
    for (uint32_t req = 0; req < 5; ++req) for (uint32_t cores = 0; cores < 5; ++cores) { CHECK(pool_workers(req, cores) >= 1); if (req) CHECK(pool_workers(req, cores) == req); }
    CHECK(drain_threads(3, true) == 4 && drain_threads(3, false) == 3);
    for (uint32_t nw = 1; nw < 8; ++nw) CHECK(3 * drain_threads(nw, false) > 0 && 3 * drain_threads(nw, true) > 0);
    for (uint32_t n = 0; n < 40; ++n) { CHECK(!use_split(n, 16, true)); CHECK(use_split(n, 16, false) == (n <= 16)); }
}

static void t_iq() {
    using namespace iqgeo;
    for (uint32_t ty = 0; ty < 64; ++ty) CHECK(supported(ty) == (block_bytes(ty) > 0));
    CHECK(block_bytes(16) == 2 + 64 && block_bytes(17) == 2 + 64 + 8 && block_bytes(22) == 2 + 64 + 8 + 8 && block_bytes(18) == 2 + 96 && block_bytes(21) == 2 + 64 + 8 + 32 + 4);
    for (uint32_t ty : {16u, 17u, 18u, 21u, 22u}) for (uint32_t j = 0; j < 4; ++j) CHECK(chunk_read_end(ty, j) <= block_bytes(ty));
    for (uint32_t l = 0; l < 32; ++l) CHECK(scale_lane(l) < 4);
    for (uint32_t s = 0; s < 16; ++s) CHECK(odd_scale(s) >= 1 && odd_scale(s) <= 31 && odd_scale(s) % 2 == 1);
    for (int y = -200; y <= 200; ++y) { CHECK(apply_sign(true, apply_sign(true, y)) == y); CHECK(apply_sign(false, y) == y); }
    for (uint32_t nt = 1; nt < 40; ++nt) { uint32_t t0 = 0, covered = 0; while (t0 < nt) { const uint32_t g = group_size(nt, t0); CHECK(g >= 1 && g <= 4 && t0 + g <= nt); t0 += g; covered += g; } CHECK(covered == nt); }
}

int main() {
    t_sgemv(); t_dotalg(); t_s2(); t_mmvq(); t_router(); t_ple(); t_actq(); t_q2geo(); t_pool(); t_iq();
    std::printf("checks: %d, failures: %d\n", g_checks, g_fail);
    if (g_fail) { std::printf("GEOMETRY TESTS FAILED\n"); return 1; }
    std::printf("GEOMETRY TESTS PASSED\n");
    return 0;
}

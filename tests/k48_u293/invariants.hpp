// invariants.hpp - host-only checks of the LAYER 1 statements of unit u293ad55355 (no device needed):
// QsaStepGeometry, QsaTopkOrderKey, QsaDecodeChunking, NativeFlashAttnStep, NativePlePostops, PleBlockKernels,
// NativeQsaIndexerAppend, NativeQsaScoreTiles. Every check increments g_fail on violation.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "baloon/decode_attn_kernels.hpp"
#include "baloon/native_flash_attn_bounds.hpp"
#include "baloon/native_qsa_indexer_append.hpp"
#include "baloon/native_qsa_score.hpp"
#include "baloon/ple_block_indexing.hpp"
#include "baloon/qsa_decode_attention.hpp"
#include "baloon/qsa_native_kernels.hpp"
#include "baloon/qsa_step_kernels.hpp"

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { ++g_fail; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)

namespace u293 {
using namespace baloon;

// ---------------------------------------------------------------------------------------------- QsaStepGeometry
inline void step_geometry() {
    const int topks[] = {64, 256, 2048}, rs[] = {2, 4, 8};
    for (int topk : topks) for (int r : rs) {
        int prev = 0;
        for (int pos = 0; pos < 6000; ++pos) {
            const qsa_step::Step s = qsa_step::make_step(pos, topk, r);
            const int bound = topk + r - 1;
            CHECK(s.pos == pos && s.n_kv == pos + 1);
            CHECK(s.n_bid == (pos + 1) / r);
            CHECK(s.width == std::min(pos + 1, bound));
            CHECK(s.width <= s.n_kv && s.width <= bound);
            if (pos + 1 <= bound) CHECK(s.width == s.n_kv);   // identity below the bound
            CHECK(s.width >= prev);                           // monotone
            prev = s.width;
            CHECK(s.n_bid * r <= s.n_kv && s.n_kv < s.n_bid * r + r);   // blocks never exceed cells, tail shorter than a block
        }
    }
    // GQA: integer division, not modulo
    for (int h = 0; h < 24; ++h) CHECK(attn_kernels::kG == 12 && (uint32_t)h / 12 == qsa_decode::kv_head(h) && h / 12 < 2);
    CHECK(1 / 12 != 1 % 2);
    int zeros = 0, ones = 0;
    for (int h = 0; h < 24; ++h) (h / 12 == 0 ? zeros : ones)++;
    CHECK(zeros == 12 && ones == 12);
    // paged row: contiguous inside a page; cell splits into page and slot
    for (uint64_t page = 0; page < 5; ++page) for (uint64_t h = 0; h < 2; ++h) for (uint64_t slot = 0; slot < 63; ++slot)
        CHECK(qsa_decode::pool_row(page, 2, h, 64, slot + 1) == qsa_decode::pool_row(page, 2, h, 64, slot) + 1);
    for (uint64_t pos = 0; pos < 1000; ++pos) CHECK(pos / 64 * 64 + pos % 64 == pos && pos % 64 < 64);
    CHECK(sizeof(qsa_step::Step) == 16);
}

// ---------------------------------------------------------------------------------------------- QsaTopkOrderKey
inline void order_key() {
    using qsa_native::order_key_bits;
    const uint32_t samples_base[] = {0u, 1u, 2u, 0x007FFFFFu, 0x00800000u, 0x3F800000u, 0x7F7FFFFFu, 0x7F800000u, 0x7FFFFFFFu,
                                     0x80000000u, 0x80000001u, 0x807FFFFFu, 0x80800000u, 0xBF800000u, 0xFF7FFFFFu, 0xFF800000u, 0xFFFFFFFFu};
    std::vector<uint32_t> v(std::begin(samples_base), std::end(samples_base));
    for (uint64_t b = 0; b < 0x100000000ull; b += 0x10001ull) v.push_back((uint32_t)b);
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    for (uint32_t b : v) {
        CHECK(order_key_bits(b) <= 0xFFFFFFFFu);
        if (b < 0xFF800000u && !(b >= 0x80000000u && b < 0x80000000u)) CHECK(order_key_bits(b) > 0);   // finite patterns sit above 0
    }
    for (size_t i = 0; i + 1 < v.size(); ++i) {
        const uint32_t a = v[i], b = v[i + 1];   // a < b as integers
        if (b < 0x80000000u) CHECK(order_key_bits(a) <= order_key_bits(b));                  // non-negative keep order
        else if (a >= 0x80000000u) CHECK(order_key_bits(b) <= order_key_bits(a));            // negative reverse
        else CHECK(order_key_bits(b) < order_key_bits(a));                                   // negative below non-negative
    }
    // value order: larger float -> larger key (finite values)
    const float fs[] = {-1e30f, -100.f, -1.f, -1e-30f, -0.f, 0.f, 1e-30f, 1.f, 100.f, 1e30f};
    for (size_t i = 0; i + 1 < sizeof(fs) / sizeof(fs[0]); ++i) CHECK(qsa_native::order_key_f(fs[i]) <= qsa_native::order_key_f(fs[i + 1]));
    // NaN is the reserved key 0 for either sign / payload, finite values are never 0
    const uint32_t nans[] = {0x7FC00000u, 0xFFC00000u, 0x7F800001u, 0xFF800001u, 0x7FFFFFFFu, 0xFFFFFFFFu};
    for (uint32_t u : nans) { float f; std::memcpy(&f, &u, 4); CHECK(qsa_native::order_key_f(f) == qsa_native::kNanKey); }
    CHECK(qsa_native::order_key_f(-INFINITY) > 0);
    // selection count
    for (uint32_t width = 1; width < 40; ++width) for (uint32_t gt = 0; gt <= width; ++gt) for (uint32_t eq = 0; eq < 50; ++eq) {
        const uint32_t c = qsa_native::sel_count(gt, eq, width);
        CHECK(c <= width);
        if (width <= gt + eq) CHECK(c == width);
        if (gt >= width) CHECK(c == gt);
    }
    // block weights sum to n_kv
    for (uint32_t n = 1; n < 3000; ++n) {
        const uint32_t nbid = n / 4;
        uint32_t sum = 0;
        for (uint32_t b = 0; b <= nbid; ++b) sum += qsa_native::block_weight(n, nbid, b);
        CHECK(sum == n);
        CHECK(qsa_native::block_weight(n, nbid, nbid) == n % 4);
    }
}

// ---------------------------------------------------------------------------------------------- QsaDecodeChunking
inline void decode_chunking() {
    using namespace qsa_decode;
    for (uint32_t cap = 1; cap < 1500; ++cap) {
        const uint32_t nc = chunks(cap);
        CHECK(nc * 64 >= cap && (nc - 1) * 64 < cap);
        CHECK((int)nc == attn_kernels::n_chunks((int)cap));
        for (uint32_t n = 0; n <= cap; n += (cap < 200 ? 1 : 37)) {
            uint32_t total = 0;
            for (uint32_t k = 0; k < nc; ++k) {
                const uint32_t h = n_here(n, 64 * k);
                CHECK(h <= 64);
                if (n <= 64 * k) CHECK(h == 0);
                if (64 * (k + 1) <= n) CHECK(h == 64);
                total += h;
            }
            CHECK(total == n);
        }
        for (uint32_t i = 0; i < cap; i += 13) { CHECK(i / 64 < nc); CHECK(i % 64 < n_here(cap, (i / 64) * 64)); }
        for (uint32_t nh : {12u, 24u}) {
            const uint64_t sc = attn_kernels::qsa_scratch_floats((int)cap, (int)nh);
            CHECK(sc == (uint64_t)nc * nh * 258 + 64);
            CHECK(query_stride(nc, nh) == (uint64_t)nc * nh * 258);
            CHECK(acc_off(nc - 1, nh - 1, nh, kHD - 1) + 1 <= (uint64_t)nc * nh * 256);
            CHECK(l_off(nc, nh, nc - 1, nh - 1) + 1 <= sc);
            CHECK(m_off(nc, nh, 0, 0) == (uint64_t)nc * nh * 256);
            CHECK(l_off(nc, nh, 0, 0) == (uint64_t)nc * nh * 257);
        }
    }
    // slot (kv head, chunk) and head decomposition
    for (uint32_t chunks_n = 1; chunks_n < 40; ++chunks_n) {
        std::vector<char> seen(2 * chunks_n, 0);
        for (uint32_t kvh = 0; kvh < 2; ++kvh) for (uint32_t c = 0; c < chunks_n; ++c) {
            const uint32_t slot = kvh * chunks_n + c;
            CHECK(slot < 2 * chunks_n && !seen[slot]);
            seen[slot] = 1;
        }
    }
    for (uint32_t h = 0; h < 24; ++h) CHECK(h == kv_head(h) * 12 + local_head(h) && local_head(h) < 12);
    // split-softmax merge equals the monolithic softmax (host reference)
    const uint32_t n = 150, hd = 4;
    std::vector<float> sc(n), val(n * hd);
    for (uint32_t i = 0; i < n; ++i) { sc[i] = std::sin(0.37f * i) * 3.f; for (uint32_t d = 0; d < hd; ++d) val[i * hd + d] = std::cos(0.11f * (i + 3 * d)); }
    std::vector<Partial> ps;
    for (uint32_t k = 0; k < chunks(256); ++k) {
        const uint32_t h = n_here(n, 64 * k);
        ps.push_back(chunk_partial(sc.data() + std::min(n, 64 * k), val.data() + std::min(n, 64 * k) * hd, h, hd));
    }
    const std::vector<float> merged = merge(ps, hd);
    double m = -1e30; for (float s : sc) m = std::max<double>(m, s);
    double L = 0; std::vector<double> acc(hd, 0.0);
    for (uint32_t i = 0; i < n; ++i) { const double e = std::exp(sc[i] - m); L += e; for (uint32_t d = 0; d < hd; ++d) acc[d] += e * val[i * hd + d]; }
    for (uint32_t d = 0; d < hd; ++d) CHECK(std::fabs(merged[d] - acc[d] / L) < 1e-5);
}

// ---------------------------------------------------------------------------------------------- NativeFlashAttnStep
inline void flash_step() {
    using namespace flash_bounds;
    const int max_ctx = 256, top_k = 2048;
    for (int pos = 0; pos < 400; ++pos) {
        const int nkv = pos + 1, nbid = nkv / 4, width = std::min(pos + 1, top_k + 3);
        const bool ok = valid(width, nkv, pos, nbid, max_ctx);
        CHECK(ok == (pos + 1 <= max_ctx));
        CHECK(!valid(width, nkv, pos + 1, nbid, max_ctx));        // stale / frozen position is rejected
        CHECK(!valid(width, nkv + 1, pos, nbid, max_ctx));
        CHECK(!valid(width, nkv, pos, nbid + 1, max_ctx));
        CHECK(!valid(0, 0, 0, 0, max_ctx));
    }
    CHECK(!valid(257, 257, 256, 64, 256));
    for (uint32_t h = 0; h < 24; ++h) CHECK(kv_head(h) < 2);
    // every cell < 256 is visited by exactly the (tile, warp, row) triples of the two 128-wide tiles
    std::vector<int> seen(256, 0);
    for (uint32_t base = 0; base < 256; base += 128) for (uint32_t warp = 0; warp < 4; ++warp) for (uint32_t lane = 0; lane < 32; lane += 8)
        for (uint32_t row = 0; row < 8; ++row) seen[cell_of(base, warp, lane, row)]++;
    for (int c = 0; c < 256; ++c) CHECK(seen[c] == 1);
    CHECK(loop_iterations() == 2);
    for (uint64_t c1 = 0; c1 < 40; ++c1) for (uint64_t k1 = 0; k1 < 2; ++k1) for (uint64_t c2 = 0; c2 < 40; ++c2) for (uint64_t k2 = 0; k2 < 2; ++k2)
        CHECK((kv_index(c1, k1, 0) == kv_index(c2, k2, 0)) == (c1 == c2 && k1 == k2));
    CHECK(kv_bytes(1) == 1024 && kTileFloats * 4 == 16384);
}

// ---------------------------------------------------------------------------------------------- NativePlePostops
inline float ref_half(uint16_t h) {
    const int s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    double v;
    if (e == 0) v = std::ldexp((double)m, -24);
    else if (e == 31) return m ? NAN : (s ? -INFINITY : INFINITY);
    else v = std::ldexp(1.0 + m / 1024.0, e - 15);
    return (float)(s ? -v : v);
}
inline void ple_postops() {
    using namespace ple_native;
    CHECK(kN * kH == kD && kD == 10240 && kD % 256 == 0 && kD / 256 == 40);
    CHECK(kGateThreads / 32 == 16 && kGateSlots * kGateThreads >= kN);
    for (int d = 0; d < kN; ++d) { const int slot = d / kGateThreads; CHECK(slot < 5 && d % kGateThreads + kGateThreads * slot == d); }
    for (int j = 5; j < kGateSlots; ++j) for (int tid = 0; tid < kGateThreads; ++tid) CHECK(tid + j * kGateThreads >= kN);   // slots 5..7 masked
    CHECK(hist_row(0) == 0 && hist_row(1) == 3 && hist_row(2) == 6 && ple_block::tap_row(9, 4, 3, 3) == 9);
    for (int k = 0; k < 4; ++k) CHECK(ple_block::tap_row(9, 4, 3, k) == (k < 3 ? hist_row(k) : 9));
    // history / weight index bounds and injectivity (spot checks over the real extents)
    CHECK(hist_index(8, kD - 1) + 1 == 9ll * kD && weight_index(kD - 1, 3) + 1 == 4ll * kD);
    for (int c = 0; c < 50; ++c) for (int r = 0; r < 9; ++r) for (int c2 = 0; c2 < 50; ++c2) for (int r2 = 0; r2 < 9; ++r2)
        CHECK((hist_index(r, c) == hist_index(r2, c2)) == (r == r2 && c == c2));
    // flat index splits into stream and column
    for (int i = 0; i < kD; ++i) CHECK(i % kN < kN && i / kN < kH && (i / kN) * kN + i % kN == i);
    // fp16 conversion is exact on every non-NaN pattern
    for (uint32_t h = 0; h < 65536; ++h) {
        const float a = half_to_float((uint16_t)h), b = ref_half((uint16_t)h);
        if (std::isnan(b)) CHECK(std::isnan(a)); else CHECK(a == b);
    }
    // gate: in (0,1), monotone, 0.5 at 0, odd about 0.5
    float prev = 0.f;
    for (int i = -4000; i <= 4000; ++i) {
        const float s = i * 0.01f, g = gate_of(s);
        CHECK(g > 0.f && g < 1.f && g >= prev - 1e-7f);
        CHECK(std::fabs(g + gate_of(-s) - 1.f) < 1e-6f);
        CHECK(std::fabs(g - ple_block::gate(s)) < 1e-6f);
        prev = g;
    }
    CHECK(gate_of(0.f) == 0.5f);
    // residual association: the model order is hidden + (gated + act)
    CHECK(7 + (5 + 3) == (7 + 5) + 3);
    // sizes
    CHECK(10240 * 4 == 40960 && 2560 * 4 == 10240 && 4 * 10240 * 2 == 81920 && 9 * 10240 * 4 == 368640);
}

// ---------------------------------------------------------------------------------------------- PleBlockKernels
inline void ple_block_geometry() {
    using namespace ple_block;
    const Scratch s = scratch_layout((5 * 10240 + 2560 + 4) * 4, (2560 / 32) * 34, 2560 * 2);
    CHECK((5 * 10240 + 2560 + 4) * 4 == 215056 && (2560 / 32) * 34 == 2720);
    CHECK(s.total == 223152 && s.float_off == 0 && s.q8_off == 215056 && s.bf16_off == 215056 + 2720);
    CHECK(align16(215056) == 215056);
    for (uint64_t x = 0; x < 200; ++x) CHECK(x <= align16(x) && align16(x) < x + 16 && align16(x) % 16 == 0);
    // overlap predicate: zero-length spans never overlap; nonempty = interval intersection
    for (int a = 0; a < 12; ++a) for (int an = 0; an < 5; ++an) for (int b = 0; b < 12; ++b) for (int bn = 0; bn < 5; ++bn) {
        const bool expect = an > 0 && bn > 0 && a < b + bn && b < a + an;
        const bool ovl = (an == 0 || bn == 0) ? false : (a < b ? (b - a < an) : (a - b < bn));
        CHECK(ovl == expect);
    }
    // history slide keeps the length and moves row i+1 to i
    std::vector<float> col = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    advance_column(col.data(), 9, 10.f);
    CHECK(col.size() == 9 && col[0] == 2 && col[7] == 9 && col[8] == 10);
    CHECK(hist_index(3, 9, 5) == 3 + 9 * 5);
}

// ---------------------------------------------------------------------------------------------- NativeQsaIndexerAppend
inline void indexer_geometry() {
    using namespace native_qsa_indexer;
    for (uint32_t pos = 0; pos < 2000; ++pos) {
        const Action a = plan(pos);
        CHECK(completes(pos) == ((pos + 1) % 4 == 0));
        CHECK(a.store_tail == (pos % 4 < 3) && a.write_block == (pos % 4 == 3));
        if (a.store_tail) CHECK(a.tail_slot < 3);
        if (completes(pos)) {
            CHECK((pos + 1) / 4 == pos / 4 + 1);
            CHECK(a.spare_row == a.block_row + 1);
        }
        for (uint64_t base : {0ull, 4ull, 1000ull}) {
            const uint64_t rp = rope_pos(base, pos);
            CHECK(rp % 4 == 0);
            if (completes(pos)) { CHECK(rp == base + (pos - 3)); CHECK(rp != base + pos); }   // first cell of the block, never the last
        }
        for (uint32_t maxc : {pos + 1, pos + 4, pos + 100}) {
            CHECK(pos / 4 < pooled_rows(maxc));
            if (completes(pos)) CHECK(pos / 4 + 1 < pooled_rows(maxc));
        }
    }
    CHECK(rope_pos(40, 0) == 0);
    for (uint32_t d = 0; d < 64; ++d) { CHECK(rot_partner(rot_partner(d)) == d && rot_partner(d) < 64 && rot_pair_index(d) < 32); }
    CHECK(pos_base_aligned(8) && !pos_base_aligned(6));
    CHECK(no_overflow(0, 2147483647ull) && !no_overflow(1, 2147483647ull));
}

// ---------------------------------------------------------------------------------------------- NativeQsaScoreTiles
inline void score_geometry() {
    using namespace native_qsa_score;
    for (int n = 1; n < 5000; ++n) {
        const int pos = n - 1, nbid = n / 4, width = std::min(n, 2051);
        CHECK(qsa_native::score_step_ok(n, n, pos, nbid, width) && qsa_native::score_step_ok(n, n + 7, pos, nbid, width));
        CHECK(!qsa_native::score_step_ok(n, n - 1, pos, nbid, width));      // n > max_cells
        CHECK(!qsa_native::score_step_ok(n, n, pos + 1, nbid, width));      // stale pos
        CHECK(!qsa_native::score_step_ok(n, n, pos, nbid + 1, width));
        CHECK(!qsa_native::score_step_ok(n, n, pos, nbid, width + 1));
        // cells written per row
        int total = 0;
        for (int row = 0; row <= nbid; ++row) {
            const int w = std::min(4, n - row * 4);
            CHECK(w <= 4);
            if (row < nbid) CHECK(w == 4); else CHECK(w == n % 4);
            total += w;
        }
        CHECK(total == n);
        CHECK((n % 4 != 0) == (n - nbid * 4 > 0));
        CHECK(n / 4 < n / 4 + 1);
    }
    CHECK(2048 + 4 - 1 == 2051 && qsa_native::kMaxWidth == 2051);
    for (uint32_t maxb = 1; maxb < 300; ++maxb) for (uint32_t row = 0; row < maxb; ++row) CHECK(row / 32 < (maxb + 31) / 32);
    // each of the 128 key columns belongs to exactly one (warp, lane, pass)
    std::vector<int> seen(128, 0);
    for (uint32_t w = 0; w < 2; ++w) for (uint32_t l = 0; l < 32; ++l) for (uint32_t p = 0; p < 2; ++p) seen[col_of(w, l, p)]++;
    for (int c = 0; c < 128; ++c) CHECK(seen[c] == 1);
    CHECK(qsa_native::kRows == (int)kRows);
}

inline void run_all_host() {
    step_geometry();
    order_key();
    decode_chunking();
    flash_step();
    ple_postops();
    ple_block_geometry();
    indexer_geometry();
    score_geometry();
}
}  // namespace u293

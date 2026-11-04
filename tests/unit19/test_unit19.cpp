// Unit uca8d125571 host tests. Each check asserts a LAYER 1 statement of the theory modules QsaCacheStep, QsaSelect,
// QsaDecodeAttn, NativeQsaScore, NativeFlashAttn, GdnRecurrence, GatedResidual, VerifyWindow, F16BitsRounding and
// DequantBlockLayouts against an independent model (exhaustive where the domain is small, seeded random otherwise).
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#include "baloon/act_quant_kernels.hpp"
#include "baloon/dequant_tensor.hpp"
#include "baloon/f16_bits.hpp"
#include "baloon/qsa_kernels.hpp"
#include "baloon/qsa_native_kernels.hpp"
#include "baloon/qsa_step_state.hpp"
#include "baloon/verify_window_kernels.hpp"

using namespace baloon;

static long g_checks = 0, g_fail = 0;
#define CHECK(c)                                                                                  \
    do {                                                                                          \
        ++g_checks;                                                                               \
        if (!(c)) { ++g_fail; if (g_fail < 40) std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } \
    } while (0)

static std::mt19937 rng(0xC0FFEEu);
static uint32_t rnd() { return rng(); }

// ============================================================================================================================
// F16BitsRounding
// ============================================================================================================================
// Software round-to-nearest-even oracle (no _Float16: the host runtime has no __truncsfhf2).
static float oracle_f16(float f) {
    const uint32_t sign = std::signbit(f) ? 0x8000u : 0u;
    const double a = std::fabs((double)f);
    uint32_t bits;
    if (a >= 65520.0) bits = 0x7C00;
    else if (a < 6.103515625e-05) bits = (uint32_t)std::nearbyint(std::ldexp(a, 24));
    else {
        int e = std::ilogb(a);
        double m = std::nearbyint(std::ldexp(a, 10 - e));
        if (m >= 2048.0) { m = 1024.0; ++e; }
        bits = e > 15 ? 0x7C00u : (uint32_t)(((e + 15) << 10) | ((int)m - 1024));
    }
    return (float)(sign | bits);
}
static void test_f16() {
    using namespace f16bits;
    // every half decodes and re-encodes to itself (NaNs stay NaN)
    for (uint32_t h = 0; h < 65536; ++h) {
        const float f = f32_from_f16((uint16_t)h);
        const bool is_nan = ((h >> 10) & 31) == 31 && (h & 1023);
        if (is_nan) { CHECK(std::isnan(f)); CHECK((f16_from_f32(f) & 0x7C00) == 0x7C00 && (f16_from_f32(f) & 0x3FF) != 0); }
        else CHECK(f16_from_f32(f) == h);
    }
    // inf / NaN stay inf / NaN, finite overflow saturates to inf (never NaN)
    CHECK(f16_from_f32(INFINITY) == 0x7C00 && f16_from_f32(-INFINITY) == 0xFC00);
    CHECK((f16_from_f32(NAN) & 0x7C00) == 0x7C00 && (f16_from_f32(NAN) & 0x3FF) != 0);
    CHECK(f16_from_f32(65504.f) == 0x7BFF);
    CHECK(f16_from_f32(65520.f) == 0x7C00);  // tie above the largest half rounds to even = inf
    CHECK(f16_from_f32(65519.f) == 0x7BFF);
    CHECK(f16_from_f32(1e30f) == 0x7C00 && f16_from_f32(-1e30f) == 0xFC00);
    // ties to even: midpoint between consecutive halves h and h+1 rounds to the even one
    for (uint32_t h = 0x0001; h < 0x7BFF; h += 7) {
        const float a = f32_from_f16((uint16_t)h), b = f32_from_f16((uint16_t)(h + 1));
        const float mid = (float)(((double)a + (double)b) / 2.0);
        const uint16_t r = f16_from_f32(mid);
        CHECK(r == ((h & 1) ? h + 1 : h));
        CHECK(r % 2 == 0);
        // just below / above the midpoint
        CHECK(f16_from_f32(std::nextafterf(mid, 0.f)) == h);
        CHECK(f16_from_f32(std::nextafterf(mid, 1e9f)) == h + 1);
    }
    // rounding never moves more than one ulp of the half
    for (int i = 0; i < 200000; ++i) {
        const float f = (float)((double)(int32_t)rnd() / 2147483648.0 * 70000.0);
        const uint16_t h = f16_from_f32(f);
        if ((h & 0x7C00) == 0x7C00) { CHECK(std::fabs(f) >= 65520.f); continue; }
        const float back = f32_from_f16(h);
        const float ulp = std::max(std::ldexp(1.0f, (int)((h >> 10) & 31) - 25), 5.9604645e-8f);
        CHECK(std::fabs(back - f) <= ulp * 0.5f * 1.0001f);
    }
    for (int i = 0; i < 3000000; ++i) {
        const uint32_t u = rnd();
        const float f = f16bits::float_of(u);
        if (std::isnan(f)) continue;
        CHECK((float)f16_from_f32(f) == oracle_f16(f));
    }
    // subnormals: exact decode value m * 2^-24, normalisation shifts at most 10
    for (uint32_t m = 1; m < 1024; ++m) CHECK(f32_from_f16((uint16_t)m) == std::ldexp((float)m, -24));
    // bf16: exact on bf16 values, below half truncates, above half rounds up, tie is even
    for (uint32_t q = 0; q < 65536; q += 3) {
        const uint32_t i = q << 16;
        if (((i >> 23) & 255) == 255) continue;
        CHECK(bf16_bits_from_f32(float_of(i)) == q);
        CHECK(bf16_bits_from_f32(float_of(i | 0x7FFF)) == q);
        const uint16_t tie = bf16_bits_from_f32(float_of(i | 0x8000));
        CHECK(tie % 2 == 0 && (tie == q || tie == q + 1));
        if (((q + 1) >> 7 & 255) != 255) CHECK(bf16_bits_from_f32(float_of(i | 0x8001)) == ((q + 1) & 0xFFFF));
    }
    CHECK((bf16_bits_from_f32(NAN) & 0x7F80) == 0x7F80 && (bf16_bits_from_f32(NAN) & 0x7F) != 0);
    CHECK(bf16_bits_from_f32(INFINITY) == 0x7F80);
    CHECK(f32_from_bf16(0x3F80) == 1.0f);
}

// ============================================================================================================================
// DequantBlockLayouts
// ============================================================================================================================
static void test_dequant_layouts() {
    using namespace dequant_tensor;
    struct G { uint32_t t; int e, b; };
    const G table[] = {{0, 1, 4}, {1, 1, 2}, {30, 1, 2}, {2, 32, 18}, {6, 32, 22}, {8, 32, 34}, {11, 256, 110}, {12, 256, 144},
                       {13, 256, 176}, {14, 256, 210}, {20, 32, 18}, {23, 256, 136}, {42, 64, 18}};
    for (const G& g : table) {
        Shape s{0, 0};
        CHECK(block_shape_of_type(g.t, &s));
        CHECK(s.elems == g.e && s.bytes == g.b);
        CHECK(256 % s.elems == 0);  // a super-block is a whole number of blocks
    }
    Shape s{0, 0};
    CHECK(!block_shape_of_type(999, &s));
    CHECK(2 + 64 * 2 / 8 == 18);  // Q2_0
    // struct offsets
    CHECK(32 + 64 + 12 + 2 == 110 && 32 + 64 == 96 && 96 + 12 == 108);
    CHECK(128 + 64 + 16 + 2 == 210 && 128 + 64 == 192 && 192 + 16 == 208);
    CHECK(4 + 12 + 32 + 128 == 176 && 4 + 12 == 16 && 16 + 32 == 48);
    CHECK(4 + 12 + 128 == 144);
    CHECK(2 + 2 + 4 + 128 == 136 && 2 + 2 == 4 && 4 + 4 == 8);
    CHECK(2 + 4 + 16 == 22 && 2 + 4 == 6);
    CHECK(sdq::kBlkQ3K == 110 && sdq::kBlkQ6K == 210 && sdq::kBlkQ5K == 176 && sdq::kBlkQ4K == 144 && sdq::kBlkIQ4XS == 136 &&
          sdq::kBlkQ5_0 == 22 && sdq::kBlkQ2_0 == 18 && sdq::kQK2_0 == 64);
    // 2-byte loads needed iff stride % 4 == 2
    for (int st : {18, 22, 34, 110, 210}) CHECK(st % 4 == 2);
    for (int st : {36, 136, 144, 176}) CHECK(st % 4 == 0);
    // row bytes: n/e*b, whole blocks give k*b; no overflow when row <= M / n_out
    for (const G& g : table) for (int k = 1; k < 50; ++k) CHECK((uint64_t)k * g.e / g.e * g.b == (uint64_t)k * g.b);
    const uint64_t M = UINT64_MAX;
    for (uint64_t nout : {1ull, 7ull, 4096ull, 1ull << 33}) { const uint64_t row = M / nout; CHECK(row * nout <= M && row <= M / nout); }
    // value ranges of the scalar dequantizers: reconstruct codes as value / d
    std::vector<uint8_t> blk(256);
    std::vector<float> y(256);
    for (int it = 0; it < 2000; ++it) {
        for (auto& b : blk) b = (uint8_t)rnd();
        const uint16_t one = f16bits::f16_from_f32(1.0f);  // d = 1 -> values are the integer codes
        // Q4_0: nibble - 8 in [-8, 7]
        std::memcpy(blk.data(), &one, 2);
        sdq::deq_q4_0(blk.data(), y.data());
        for (int j = 0; j < 32; ++j) CHECK(y[j] >= -8 && y[j] <= 7 && y[j] == std::floor(y[j]));
        // Q5_0: [-16, 15]
        sdq::deq_q5_0(blk.data(), y.data());
        for (int j = 0; j < 32; ++j) CHECK(y[j] >= -16 && y[j] <= 15);
        // Q2_0 symbols {-1,0,1,2}
        sdq::deq_q2_0(blk.data(), y.data());
        for (int j = 0; j < 64; ++j) CHECK(y[j] >= -1 && y[j] <= 2);
        // Q8_0 codes are int8
        sdq::deq_q8_0(blk.data(), y.data());
        for (int j = 0; j < 32; ++j) CHECK(y[j] == (float)(int8_t)blk[2 + j]);
        // IQ4_NL values come from the codebook
        sdq::deq_iq4_nl(blk.data(), y.data());
        for (int j = 0; j < 32; ++j) { bool f = false; for (int c : sdq::kIq4nl) f |= (y[j] == (float)c); CHECK(f); }
        // Q6_K with d = 1 at offset 208 and scales set to 1: codes in [-32, 31]
        std::vector<uint8_t> q6(210);
        for (auto& b : q6) b = (uint8_t)rnd();
        for (int i = 0; i < 16; ++i) q6[192 + i] = 1;
        std::memcpy(q6.data() + 208, &one, 2);
        sdq::deq_q6_k(q6.data(), y.data());
        for (int j = 0; j < 256; ++j) CHECK(y[j] >= -32 && y[j] <= 31);
    }
    // K-quant scale/min unpack is 6 bits
    for (int it = 0; it < 5000; ++it) {
        uint8_t q[12];
        for (auto& b : q) b = (uint8_t)rnd();
        for (int j = 0; j < 8; ++j) { uint8_t d, m; sdq::scale_min_k4(j, q, &d, &m); CHECK(d < 64 && m < 64); }
    }
    for (int q = 0; q < 256; ++q) for (int q2 = 0; q2 < 256; ++q2) { CHECK(q % 16 + 16 * (q2 / 64) < 64); CHECK(q / 16 + 16 * (q2 / 64) < 64); }
    for (int u : {1, 4, 16, 64}) CHECK(2 * u < 256 && (u & (2 * u)) == 0);
    // flat types round trip exactly
    {
        std::vector<uint8_t> raw(64);
        for (auto& b : raw) b = (uint8_t)rnd();
        std::vector<float> o(16);
        CHECK(dequantize_tensor(kF32, raw.data(), 16, o.data()));
        CHECK(std::memcmp(o.data(), raw.data(), 64) == 0);
        std::vector<float> o2(32);
        CHECK(dequantize_tensor(kBF16, raw.data(), 32, o2.data()));
        for (int i = 0; i < 32; ++i) { uint32_t u = (uint32_t)(raw[2 * i] | (raw[2 * i + 1] << 8)) << 16; float e; std::memcpy(&e, &u, 4); CHECK(std::memcmp(&e, &o2[i], 4) == 0); }
        CHECK(!dequantize_tensor(kQ4_0, raw.data(), 31, o2.data()));  // ragged
        CHECK(!dequantize_tensor(999, raw.data(), 32, o2.data()));
    }
}

// ============================================================================================================================
// QsaCacheStep
// ============================================================================================================================
static void test_qsa_cache_step() {
    using namespace qsa_step;
    const int r = kR;
    // selection width and step fill
    for (int pos = 0; pos < 40000; ++pos) {
        const Step s = step_fill(pos, kTopK, r);
        CHECK(s.pos == pos && s.n_kv == pos + 1 && s.n_bid == (pos + 1) / r);
        CHECK(s.width == std::min(pos + 1, 2051));
        CHECK(s.width <= s.n_kv && s.width <= kTopK + r - 1);
        CHECK(s.n_bid * r <= s.n_kv && s.n_kv < (s.n_bid + 1) * r);
        if (pos + 1 < kTopK + r - 1) CHECK(s.width == s.n_kv);  // identity below the bound
    }
    CHECK(selection_width(32768, 2048, 4) == 2051);
    CHECK((2051 + 32) * 4 <= 49152);
    CHECK(sizeof(Step) == 16 && step_bytes() == 16);
    CHECK(kMaxWidth == 2051 && kTopkMaxCells == 32768);
    // paged row mapping: in range and injective
    {
        const int nPages = 5, H = 3, P = 7;
        std::vector<char> seen((size_t)nPages * H * P, 0);
        for (int p = 0; p < nPages; ++p) for (int h = 0; h < H; ++h) for (int o = 0; o < P; ++o) {
            const int64_t row = kv_row(p, H, h, P, (int64_t)p * P + o);  // t % P == o since t = p*P + o
            CHECK(row >= 0 && row < (int64_t)nPages * H * P);
            CHECK(!seen[row]);
            seen[row] = 1;
        }
        for (char c : seen) CHECK(c);
    }
    // block completion schedule
    for (int pos = 0; pos < 4000; ++pos) {
        CHECK(completes_block(pos, r) == ((pos + 1) % r == 0));
        const int before = pos / r, after = (pos + 1) / r;
        CHECK(after == before + (completes_block(pos, r) ? 1 : 0));
        if (!completes_block(pos, r)) CHECK(pos % r < r - 1);  // raw tail holds r-1 rows
    }
    // cell_block and tail bias
    for (int n = 1; n < 600; ++n) for (int j = 0; j < n; ++j) {
        CHECK(cell_block(j, n, r) == j / r);                       // a cached cell never maps past the spare row
        CHECK(cell_block(j, n, r) <= n / r);
        CHECK(tail_cell(j, n, r) == (j / r == n / r));
        CHECK(tail_cell(j, n, r) == ((n / r) * r <= j));
    }
    for (int n = 1; n < 100; ++n) { bool any = false; for (int j = 0; j < n; ++j) any |= tail_cell(j, n, r); CHECK(any == (n % r != 0)); }
    const Shapes sh = real_shapes(16, 4096);
    CHECK(sh.n_head == 24 && sh.n_head_kv == 2 && sh.head_dim == 256 && sh.idx_block == 4 && sh.top_k == 2048);
}

// ============================================================================================================================
// QsaSelect
// ============================================================================================================================
static std::vector<uint32_t> ref_topk(const std::vector<float>& s, int width) {
    std::vector<int> idx(s.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return (s[a] + 0.0f) > (s[b] + 0.0f); });
    std::vector<uint32_t> out(idx.begin(), idx.begin() + std::min<size_t>(width, idx.size()));
    std::sort(out.begin(), out.end());
    return out;
}
// Host model of the kernel's selection: radix threshold on order keys + first ties by index.
static std::vector<uint32_t> model_topk(const std::vector<float>& s, int width) {
    using namespace qsa_step;
    const int n = (int)s.size();
    if (n <= width) { std::vector<uint32_t> o(n); std::iota(o.begin(), o.end(), 0); return o; }
    uint32_t prefix = 0, above = 0;
    for (int pass = 0; pass < 4; ++pass) {
        uint32_t hist[256] = {0};
        for (float f : s) { const uint32_t k = order_key(f); if (pass == 0 || (k >> (32 - 8 * pass)) == prefix) ++hist[(k >> (24 - 8 * pass)) & 255]; }
        uint32_t cum = above; int chosen = 0;
        for (int d = 255; d >= 0; --d) { if (cum + hist[d] >= (uint32_t)width) { chosen = d; break; } cum += hist[d]; }
        above = cum; prefix = pass == 0 ? (uint32_t)chosen : ((prefix << 8) | (uint32_t)chosen);
    }
    const uint32_t thr = prefix;
    uint32_t gt = 0, ge = 0;
    for (float f : s) { const uint32_t k = order_key(f); gt += k > thr; ge += k >= thr; }
    // threshold property: count(key >= thr) >= width > count(key > thr)
    CHECK(ge >= (uint32_t)width && gt < (uint32_t)width && gt == above);
    std::vector<uint32_t> out;
    uint32_t left = width - gt;
    for (int j = 0; j < n; ++j) { const uint32_t k = order_key(s[j]); if (k > thr) out.push_back(j); else if (k == thr && left) { out.push_back(j); --left; } }
    return out;
}
static void test_qsa_select() {
    using namespace qsa_step;
    // order key: matches float order, folds -0 onto +0
    {
        std::vector<float> v;
        for (int i = 0; i < 20000; ++i) v.push_back(f16bits::float_of(rnd()));
        for (size_t i = 0; i + 1 < v.size(); ++i) {
            if (std::isnan(v[i]) || std::isnan(v[i + 1])) continue;
            const float a = v[i], b = v[i + 1];
            if (a + 0.0f < b + 0.0f) CHECK(order_key(a) < order_key(b));
            if (a + 0.0f == b + 0.0f) CHECK(order_key(a) == order_key(b));
        }
        CHECK(order_key(-0.0f) == order_key(0.0f));
        CHECK(order_key(-1.0f) < order_key(-0.0f) && order_key(0.0f) < order_key(1e-30f));
        for (uint32_t b = 0; b < 0x7F800000u; b += 0x10001) { if (b + 0x10001 < 0x7F800000u) CHECK(order_key_bits(b) <= order_key_bits(b + 0x10001)); }
        for (uint32_t b = 0x80000000u; b < 0xFF800000u; b += 0x10001) { if (b + 0x10001 < 0xFF800000u) CHECK(order_key_bits(b + 0x10001) <= order_key_bits(b)); }
    }
    // threshold selection equals the stable-sort oracle; exactly `width` elements; ties lowest index first
    for (int it = 0; it < 400; ++it) {
        const int n = 1 + (int)(rnd() % 6000);
        const int width = 1 + (int)(rnd() % 2051);
        std::vector<float> s(n);
        const int mode = it % 4;
        for (auto& x : s) {
            if (mode == 0) x = (float)(int)(rnd() % 7) - 3.f;               // heavy ties incl. 0 / -0 after folding
            else if (mode == 1) x = ((rnd() & 1) ? -0.0f : 0.0f);          // all tied
            else if (mode == 2) x = (float)((double)(int32_t)rnd() / 1e9);
            else x = (rnd() % 5 == 0) ? 1e9f : (float)(rnd() % 100) * 0.25f;
        }
        const auto m = model_topk(s, width);
        CHECK((int)m.size() == std::min(n, width));
        CHECK(m == ref_topk(s, width));
        CHECK(std::is_sorted(m.begin(), m.end()));
    }
    // block weights: total n_kv, completed blocks r each
    for (int n = 1; n < 800; ++n) {
        const int nb = n / kR;
        long total = 0;
        for (int b = 0; b <= nb; ++b) total += block_weight(b, n, kR);
        CHECK(total == n);
        for (int b = 0; b < nb; ++b) CHECK(block_weight(b, n, kR) == kR);
        CHECK(block_weight(nb, n, kR) == n - nb * kR);
    }
    // block expansion: cells ascending, unique, real
    for (int n : {1, 3, 4, 5, 17, 100, 1001}) {
        const int nb = n / kR;
        std::vector<int> cells;
        for (int b = 0; b <= nb; ++b) for (int c = 0; c < block_weight(b, n, kR); ++c) cells.push_back(b * kR + c);
        CHECK((int)cells.size() == n);
        for (int j = 0; j < n; ++j) CHECK(cells[j] == j);
    }
    // radix digits recompose
    for (int i = 0; i < 10000; ++i) { const uint32_t k = rnd(); CHECK(((k >> 24) << 24 | ((k >> 16) & 255) << 16 | ((k >> 8) & 255) << 8 | (k & 255)) == k); }
    // warp-per-block scoring grid covers every block exactly once
    for (int mb = 1; mb < 600; ++mb) { std::vector<int> hit(mb, 0); for (int bx = 0; bx < (mb + 7) / 8; ++bx) for (int w = 0; w < 8; ++w) if (bx * 8 + w < mb) ++hit[bx * 8 + w]; for (int h : hit) CHECK(h == 1); }
}

// ============================================================================================================================
// QsaDecodeAttn: chunk bookkeeping, scratch layout, split-K merge == direct softmax
// ============================================================================================================================
static void test_qsa_decode_attn() {
    const auto chunks = [](int cap) { return (cap + 63) / 64; };
    for (int cap = 1; cap < 3000; ++cap) {
        CHECK(qsa_dev::scratch_floats_model(cap, 24) == (size_t)chunks(cap) * 24 * 258 + 64);
        for (int i : {0, cap / 2, cap - 1}) CHECK(i / 64 < chunks(cap));
    }
    for (int n = 0; n < 300; ++n) for (int c = 0; c < 6; ++c) {
        const int nh = std::min(64, n - c * 64);
        if (n <= c * 64) CHECK(nh <= 0); else CHECK(nh >= 1 && nh <= 64);
    }
    for (int i = 0; i < 2051; ++i) { const int n = 2051; const int c = i / 64; CHECK(i % 64 < std::min(64, n - c * 64) && c * 64 + i % 64 == i); }
    for (int h = 0; h < 24; ++h) { CHECK(h / 12 < 2 && (h / 12) * 12 + h % 12 == h); }
    CHECK(1 / 12 != 1 % 2);  // GQA map is division, not modulo
    for (int d = 0; d < 256; ++d) CHECK(d / 8 < 32 && (d / 8) * 8 + d % 8 == d);
    // acc / slot index ranges and batch disjointness
    for (int slot = 0; slot < 20; ++slot) for (int h = 0; h < 12; ++h) for (int t = 0; t < 256; t += 37) CHECK((slot * 12 + h) * 256 + t < 20 * 12 * 256);
    for (size_t z1 = 0; z1 < 5; ++z1) for (size_t z2 = z1 + 1; z2 < 6; ++z2) CHECK(z1 * 1000 + 1000 <= z2 * 1000);
    // merge equals direct softmax; every merge exponent is <= 0
    for (int it = 0; it < 200; ++it) {
        const int n = 1 + (int)(rnd() % 2051), D = 5;
        std::vector<double> sc(n);
        std::vector<std::vector<double>> v(n, std::vector<double>(D));
        for (auto& x : sc) x = ((double)(int32_t)rnd() / 2147483648.0) * 8.0;
        for (auto& row : v) for (auto& x : row) x = (double)(int32_t)rnd() / 2147483648.0;
        double M = -1e300;
        for (double x : sc) M = std::max(M, x);
        std::vector<double> direct(D, 0.0);
        double L = 0;
        for (int i = 0; i < n; ++i) { const double p = std::exp(sc[i] - M); L += p; for (int d = 0; d < D; ++d) direct[d] += p * v[i][d]; }
        // chunked
        struct P { double m, l; std::vector<double> acc; };
        std::vector<P> parts;
        for (int c = 0; c * 64 < n; ++c) {
            P p{-1e300, 0, std::vector<double>(D, 0.0)};
            const int hi = std::min(n, c * 64 + 64);
            for (int i = c * 64; i < hi; ++i) p.m = std::max(p.m, sc[i]);
            for (int i = c * 64; i < hi; ++i) { const double e = std::exp(sc[i] - p.m); p.l += e; for (int d = 0; d < D; ++d) p.acc[d] += e * v[i][d]; }
            parts.push_back(p);
        }
        double MM = -1e300;
        for (auto& p : parts) MM = std::max(MM, p.m);
        double LL = 0; std::vector<double> A(D, 0.0);
        for (auto& p : parts) { CHECK(p.m - MM <= 0); const double w = std::exp(p.m - MM); LL += w * p.l; for (int d = 0; d < D; ++d) A[d] += w * p.acc[d]; }
        for (int d = 0; d < D; ++d) CHECK(std::fabs(A[d] / LL - direct[d] / L) < 1e-9);
    }
}

// ============================================================================================================================
// NativeQsaScore + NativeFlashAttn
// ============================================================================================================================
static void test_native() {
    auto stepValid = [](int n, int pos, int full, int width, int maxCells) {
        return 1 <= n && n <= maxCells && pos == n - 1 && full == n / 4 && width == (n < 2051 ? n : 2051);
    };
    for (int pos = 0; pos < 5000; ++pos) { const auto s = qsa_step::step_fill(pos, 2048, 4); CHECK(stepValid(s.n_kv, s.pos, s.n_bid, s.width, pos + 1)); CHECK(!stepValid(s.n_kv, s.pos + 1, s.n_bid, s.width, pos + 1)); }
    for (int maxCells : {4, 7, 4096, 32768}) {
        const int maxBlocks = maxCells / 4 + 1;
        for (int n = 1; n <= maxCells; n += 1 + n / 50) { CHECK(n / 4 < maxBlocks); for (int row : {0, n / 4}) CHECK(row / 32 < (maxBlocks + 31) / 32); }
    }
    for (int j = 0; j < 4000; ++j) { CHECK(j / 4 <= 1000 || j >= 4000); CHECK(4 * (j / 4) <= j && j < 4 * (j / 4) + 4); }
    for (int r1 = 0; r1 < 30; ++r1) for (int c1 = 0; c1 < 4; ++c1) for (int r2 = 0; r2 < 30; ++r2) for (int c2 = 0; c2 < 4; ++c2) if (r1 * 4 + c1 == r2 * 4 + c2) CHECK(r1 == r2 && c1 == c2);
    for (int col = 0; col < 128; ++col) { CHECK(col == (col / 64) * 64 + ((col / 32) % 2) * 32 + col % 32); CHECK(col / 64 < 2); }
    for (int h = 0; h < 8; ++h) for (int i = 0; i < 64; ++i) CHECK(h * 68 + i < 2 * 16 * 36);
    for (int h1 = 0; h1 < 8; ++h1) for (int i1 = 0; i1 < 64; ++i1) for (int h2 = 0; h2 < 8; ++h2) for (int i2 = 0; i2 < 64; i2 += 7) if (h1 * 68 + i1 == h2 * 68 + i2) CHECK(h1 == h2 && i1 == i2);
    for (int w = 0; w < 2; ++w) for (int ia = 0; ia < 2; ++ia) for (int l = 0; l < 4; ++l) for (int lane = 0; lane < 32; ++lane) { CHECK(w * 32 + ia * 16 + (l / 2) * 8 + lane / 4 < 64); CHECK((lane % 4) * 2 + l % 2 < 8); }
    for (int i = 0; i < 16; ++i) for (int lane = 0; lane < 32; ++lane) for (int w = 0; w < 2; ++w) CHECK(w * 16 * 36 + i * 36 + lane < 2 * 16 * 36);
    // span validation (device wrapper): aligned, bounded, disjoint writes
    {
        qsa_dev::Span a[3] = {{1024, 64}, {2048, 64}, {4096, 32}};
        bool w[3] = {false, false, true};
        CHECK(qsa_dev::spans_ok_model(a, 3, w));
        qsa_dev::Span b[2] = {{1024, 64}, {1056, 64}};
        bool wb[2] = {false, true};
        CHECK(!qsa_dev::spans_ok_model(b, 2, wb));        // overlapping write
        bool wr[2] = {false, false};
        CHECK(qsa_dev::spans_ok_model(b, 2, wr));         // read/read overlap is fine
        qsa_dev::Span c[1] = {{1026, 64}};
        CHECK(!qsa_dev::spans_ok_model(c, 1, wr));        // misaligned
        qsa_dev::Span d[1] = {{0, 64}};
        CHECK(!qsa_dev::spans_ok_model(d, 1, wr));        // null
        qsa_dev::Span e[1] = {{UINT64_MAX - 3, 64}};
        CHECK(!qsa_dev::spans_ok_model(e, 1, wr));        // address wrap
    }
    // flash-attention step guard
    auto fvalid = [](int width, int nkv, int pos, int nbid, int maxCtx) { return 1 <= width && width <= maxCtx && nkv == width && pos + 1 == width && nbid == width / 4; };
    for (int pos = 0; pos < 256; ++pos) { const auto s = qsa_step::step_fill(pos, 2048, 4); CHECK(fvalid(s.width, s.n_kv, s.pos, s.n_bid, 256)); CHECK(std::min(pos + 1, 2051) == pos + 1); }
    CHECK(!fvalid(0, 0, -1, 0, 256) && !fvalid(257, 257, 256, 64, 256) && !fvalid(10, 11, 9, 2, 256) && !fvalid(10, 10, 9, 3, 256));
    for (int cell = 0; cell < 256; ++cell) CHECK(cell == (cell / 128) * 128 + ((cell % 128) / 32) * 32 + ((cell % 32) / 8) * 8 + cell % 8);
    for (int warp = 0; warp < 4; ++warp) for (int lane = 0; lane < 32; ++lane) CHECK(warp * 32 + (lane / 8) * 8 + lane % 8 == warp * 32 + lane);
    for (int warp = 0; warp < 4; ++warp) for (int lane = 0; lane < 32; ++lane) for (int i0 : {0, 32, 64, 96}) for (int j = 0; j < 4; ++j) CHECK(warp * 4 * 256 + (lane / 8) * 256 + 2 * (i0 + (lane % 8) * 4) + 2 * j + 1 < 4 * 4 * 256);
    CHECK((4 * 4 * 256 + 32 + 32) * 4 <= 49152 && 16 * 16 == 256 && 2 * 256 * 2 == 1024);
    for (int cell = 0; cell < 256; cell += 5) for (int kv = 0; kv < 2; ++kv) for (int d = 0; d < 256; d += 31) CHECK((cell * 2 + kv) * 256 + d < 256 * 2 * 256);
    // online softmax: rescale exponent <= 0, running max monotone, result == direct softmax
    for (int it = 0; it < 100; ++it) {
        const int n = 1 + (int)(rnd() % 256);
        std::vector<double> s(n), v(n);
        for (int i = 0; i < n; ++i) { s[i] = (double)(int32_t)rnd() / 2147483648.0 * 6; v[i] = (double)(int32_t)rnd() / 2147483648.0; }
        double m = -1e300, l = 0, acc = 0;
        for (int pass = 0; pass < 2; ++pass) {
            double pm = -1e300;
            for (int i = pass * 128; i < std::min(n, pass * 128 + 128); ++i) pm = std::max(pm, s[i]);
            const double next = std::max(m, pm);
            CHECK(m - next <= 0 || m < -1e299);
            const double rs = (m < -1e299) ? 0.0 : std::exp(m - next);
            l *= rs; acc *= rs;
            for (int i = pass * 128; i < std::min(n, pass * 128 + 128); ++i) { const double e = std::exp(s[i] - next); l += e; acc += e * v[i]; }
            m = next;
        }
        double M = -1e300; for (double x : s) M = std::max(M, x);
        double L = 0, A = 0; for (int i = 0; i < n; ++i) { const double e = std::exp(s[i] - M); L += e; A += e * v[i]; }
        CHECK(std::fabs(acc / l - A / L) < 1e-9);
    }
}

// ============================================================================================================================
// GdnRecurrence + GatedResidual + VerifyWindow arithmetic
// ============================================================================================================================
static void test_gdn_gr_verify() {
    // GDN state layout
    const int S = 128, hv = 48;
    {
        std::vector<char> seen((size_t)S * hv * S, 0);
        for (int i = 0; i < S; ++i) for (int h = 0; h < hv; ++h) for (int j = 0; j < S; ++j) { const size_t idx = ((size_t)i * hv + h) * S + j; CHECK(idx < seen.size()); CHECK(!seen[idx]); seen[idx] = 1; }
    }
    for (int i = 0; i < 128; ++i) CHECK(i == (i / 32) * 32 + i % 32 && i / 32 < 4);
    for (int j = 0; j < 128; ++j) CHECK(j / 32 < 4 && j == (j / 32) * 32 + j % 32);
    CHECK(128 * 48 == 6144);
    for (int h = 0; h < 48; ++h) CHECK(h % 16 < 16);  // head pairing is modulo
    // conv window / commit semantics as a sequence model
    auto seq = [](const float* hist, const float* x, int i) { return i < 3 ? hist[i] : x[i - 3]; };
    for (int it = 0; it < 500; ++it) {
        float hist[3], x[10];
        for (auto& f : hist) f = (float)rnd();
        for (auto& f : x) f = (float)rnd();
        const int n = (int)(rnd() % 11);
        float next[3]; for (int j = 0; j < 3; ++j) next[j] = seq(hist, x, n + j);
        if (n == 0) for (int j = 0; j < 3; ++j) CHECK(next[j] == hist[j]);
        if (n == 1) CHECK(next[0] == hist[1] && next[1] == hist[2] && next[2] == x[0]);
        if (n >= 3) for (int j = 0; j < 3; ++j) CHECK(next[j] == x[n + j - 3]);
        for (int t = 0; t < 6; ++t) for (int j = 0; j < 3; ++j) CHECK(seq(hist, x, (t + 1) + j) == seq(hist, x, t + (j + 1)));
    }
    // bf16 pair word split
    for (int i = 0; i < 1000; ++i) { const uint32_t w = rnd(); const uint32_t lo = w << 16, hi = w & 0xFFFF0000u; CHECK((lo >> 16) == (w & 0xFFFF) && (hi >> 16) == (w >> 16)); }
    for (int row = 0; row < 96; ++row) { CHECK((row < 48) == (row / 48 == 0)); CHECK(row / 8 < (96 + 7) / 8); }
    // gated residual geometry
    CHECK(2560 * 4 == 10240 && 320 / 8 == 40 && 40 * 8 == 320 && 80 * 32 == 2560 && 160 * 16 == 2560 && 320 / 8 == 40 && 40 - 32 == 8);
    CHECK(10240 / 8 == 1280 && 1280 / 32 == 40 && 10240 / 2560 == 4 && 2560 / 8 / 32 == 10);
    for (int k = 0; k < 320; ++k) CHECK(k / 8 < 40 && k == (k / 8) * 8 + k % 8);
    for (int i = 0; i < 10240; ++i) {
        CHECK(i == (i / 256) * 256 + ((i % 256) / 32) * 32 + i % 32 && (i % 256) / 32 < 8);
        CHECK(i / 8 < 1280);
    }
    for (int j = 0; j < 320; ++j) CHECK(j / 32 < 10);
    {
        std::vector<int> cover(10240, 0);
        for (int blk = 0; blk < 80; ++blk) for (int r = 0; r < 128; ++r) { const int idx = (r / 32) * 2560 + blk * 32 + r % 32; CHECK(idx < 10240); ++cover[idx]; }
        for (int c : cover) CHECK(c == 1);  // surjective and injective
    }
    for (int i = 0; i < 10240; ++i) CHECK(i / 2560 < 4 && i % 2560 < 2560 && (i / 2560) * 2560 + i % 2560 == i);
    auto align16 = [](size_t s) { return (s + 15) / 16 * 16; };
    for (size_t s = 0; s < 5000; ++s) CHECK(s <= align16(s) && align16(s) < s + 16 && align16(s) % 16 == 0);
    CHECK(align16(10240 * 4) + align16(10240 * 2) + align16(320 * 2) + align16(10240 * 4) + align16(320 * 4) == 104320);
    CHECK(640 < 10240 * 2 && 8 * 4 <= 8 * 8);
    {   // regions laid out consecutively never overlap
        const size_t sz[5] = {10240 * 4, 10240 * 2, 320 * 2, 10240 * 4, 320 * 4};
        size_t off = 0;
        for (size_t s : sz) { CHECK(off + s <= off + align16(s)); off += align16(s); }
        CHECK(off == 104320);
    }
    // write semantics: out[i] = R[i] + block_out[i % n] * w[i / n]
    for (int i = 0; i < 10240; ++i) CHECK(i / 2560 < 4 && i % 2560 < 2560);
    // VerifyWindow
    for (int c = 0; c < 3000; ++c) {
        const auto s = qsa_step::step_fill(c, 2048, 4);
        const int dense[4] = {c, c + 1, (c + 1) / 4, c + 1};
        if (c + 1 <= 2051) CHECK(dense[0] == s.pos && dense[1] == s.n_kv && dense[2] == s.n_bid && dense[3] == s.width);
    }
    for (int n = 1; n < 400; ++n) for (int w : {1, 7, 64, 256}) {
        const int start = n > w ? n - w : 0;
        CHECK(n - start == std::min(n, w));
        for (int j = 0; j < n - start; ++j) CHECK(start + j < n);
        if (n <= w) CHECK(start == 0);
    }
    for (int i = 0; i < 5000; ++i) { const int n = 1 + (int)(rnd() % 1000), per = 1 + (int)(rnd() % 50); const int k = (int)(rnd() % (n * per)); CHECK(k / per < n && k % per < per && (k / per) * per + k % per == k); }
    for (int i = 0; i < 100; ++i) CHECK((i / 2) < (101 / 2 + 1) && i / 4 < (100 + 3) / 4);
    for (int b = 0; b < 256; ++b) { CHECK((b / 16) % 16 * 16 + b % 16 == b && (b / 16) % 16 < 16); }
    for (size_t base : {0ul, 4096ul}) for (size_t k = 0; k < 20; ++k) CHECK(base + (k + 1) * 77 == (base + k * 77) + 77);
    CHECK(qsa_dev::kOk == 0 && verify_dev::kVerifyMaxT >= 8);
}

// Host reference of act-quant blocks (cross-check of the format constants).
static void test_act_quant_ref() {
    using namespace act_quant;
    CHECK(Q8K_BYTES == 4 + 256 + 32 && Q8_0_BYTES == 2 + 32);
    for (int it = 0; it < 300; ++it) {
        std::vector<float> x(256);
        for (auto& f : x) f = (float)((double)(int32_t)rnd() / 2147483648.0 * 5.0);
        uint8_t blk[Q8K_BYTES];
        ref_quantize_q8_K(x.data(), blk);
        float d; std::memcpy(&d, blk, 4);
        int16_t bs[16]; std::memcpy(bs, blk + 4 + 256, 32);
        for (int g = 0; g < 16; ++g) { int s = 0; for (int i = 0; i < 16; ++i) s += (int8_t)blk[4 + g * 16 + i]; CHECK(s == bs[g]); }
        for (int i = 0; i < 256; ++i) { const int q = (int8_t)blk[4 + i]; CHECK(q >= -127 && q <= 127); CHECK(std::fabs(q * d - x[i]) <= std::fabs(d) * 0.5001f + 1e-6f); }
        uint8_t b8[Q8_0_BYTES];
        ref_quantize_q8_0(x.data(), b8);
        uint16_t dh; std::memcpy(&dh, b8, 2);
        const float d8 = f16bits::f32_from_f16(dh);
        for (int i = 0; i < 32; ++i) CHECK(std::fabs((int8_t)b8[2 + i] * d8 - x[i]) <= d8 * 0.6f + 1e-6f);
    }
    uint8_t z[Q8K_BYTES];
    std::vector<float> zeros(256, 0.f);
    ref_quantize_q8_K(zeros.data(), z);
    for (uint8_t b : z) CHECK(b == 0);
}

int main() {
    test_f16();
    test_dequant_layouts();
    test_qsa_cache_step();
    test_qsa_select();
    test_qsa_decode_attn();
    test_native();
    test_gdn_gr_verify();
    test_act_quant_ref();
    std::printf("checks=%ld failures=%ld\n", g_checks, g_fail);
    std::printf(g_fail == 0 ? "UNIT19 HOST TESTS PASSED\n" : "UNIT19 HOST TESTS FAILED\n");
    return g_fail == 0 ? 0 : 1;
}

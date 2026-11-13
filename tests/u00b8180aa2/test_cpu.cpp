// Unit u00b8180aa2 (Q2ExpertDotKernel, Q2Avx2Fallback): one section per theorem of the two theory modules, each run
// against the code that implements it (src/kernels/cpu/q2_avx2.cpp, src/cpu/expert_q2.cpp,
// include/baloon/q2_avx2_fallback.hpp, include/baloon/q2_expert_vnni_kernel.hpp).
// Prints "U00B8180AA2 CPU TESTS PASSED" and exits 0 on success.
#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "baloon/kern_cpu_expert.hpp"
#include "baloon/q2_avx2_fallback.hpp"
#include "baloon/q2_expert_vnni_kernel.hpp"
#include "baloon/u11_expert.hpp"
#include "harness.hpp"

namespace qa = baloon::u11::q2avx2;
namespace qm = baloon::q2avx2_model;
namespace qe = baloon::q2_expert;

static std::mt19937& rng() { static std::mt19937 g(0xB8180AA2u); return g; }
static uint32_t rnd(uint32_t n) { return (uint32_t)(rng()() % n); }
static bool avx512_usable() { return baloon::kern::cpu_features().usable(); }

// ==== Q2Avx2Fallback ===========================================================================================
U7_SECTION(shift16_bytes_independent, "shifting a 16-bit word lo + 256 hi right by 2k and masking each byte with 3 yields the codes of lo and hi independently") {
    uint8_t blk[16], out[64];
    for (uint32_t lo = 0; lo < 256; ++lo)
        for (uint32_t hi = 0; hi < 256; ++hi) {
            for (int b = 0; b < 16; ++b) blk[b] = (uint8_t)(b % 2 ? hi : lo);
            qa::unpack64(blk, out);
            for (uint32_t v = 0; v < 64; ++v) {
                const uint32_t byte = blk[v / 4], want = (byte >> (2 * (v % 4))) & 3u;
                if (out[v] != want || qm::shift16_code(lo, hi, v % 4, (v / 4) % 2) != want) { CHECK(out[v] == want); return; }
            }
        }
}
U7_SECTION(unpackPos_order, "value i of the first 16 lands at code slot i % 4 of byte i / 4") {
    for (uint32_t i = 0; i < 16; ++i) CHECK(qm::unpack_pos(i) == (i % 4) * 256 + i / 4);
    uint8_t blk[16], out[64];
    for (int it = 0; it < 4096; ++it) {
        for (auto& b : blk) b = (uint8_t)rnd(256);
        qa::unpack64(blk, out);
        for (uint32_t i = 0; i < 16; ++i) CHECK(out[i] == ((blk[qm::unpack_pos(i) % 256] >> (2 * (qm::unpack_pos(i) / 256))) & 3u));
    }
}
U7_SECTION(unpack_groups_cover, "the four 16-value groups cover values 0..63: group g, local index i -> value 16 g + i, byte 4 g + i / 4") {
    uint8_t blk[16], out[64], ref[64];
    for (int it = 0; it < 4096; ++it) {
        for (auto& b : blk) b = (uint8_t)rnd(256);
        qa::unpack64(blk, out);
        qe::unpack_block(blk, ref);
        for (uint32_t g = 0; g < 4; ++g)
            for (uint32_t i = 0; i < 16; ++i) {
                const uint32_t v = 16 * g + i;
                CHECK(qm::value_byte(v) == 4 * g + i / 4 && qm::value_slot(v) == i % 4);
                CHECK(out[v] == ref[v] && out[v] == ((blk[4 * g + i / 4] >> (2 * (i % 4))) & 3u));
            }
    }
}
static int32_t scalar_sum(const uint8_t* codes16, uint32_t h, const int8_t* qc) {
    int32_t s = 0;
    for (uint32_t i = 0; i < 32; ++i) s += (int32_t)((codes16[(h * 32 + i) / 4] >> (2 * (i % 4))) & 3u) * qc[i];
    return s;
}

U7_SECTION(maddubs_codes_bound, "maddubs pairs c0 x0 + c1 x1 with codes in [0,3] and |x| <= 127 stay within +-762") {
    for (int c0 = 0; c0 < 4; ++c0) for (int c1 = 0; c1 < 4; ++c1)
        for (int x0 = -127; x0 <= 127; x0 += 127) for (int x1 = -127; x1 <= 127; x1 += 127)
            CHECK(std::abs(c0 * x0 + c1 * x1) <= qm::kMaddubsPairMax);
    uint8_t blk[18] = {0, 0};
    int8_t q[64];
    for (int e = -127; e <= 127; e += 127)
        for (int fill = 0; fill < 4; ++fill) {
            std::memset(blk + 2, 0x55 * fill, 16);  // every code = fill
            for (auto& v : q) v = (int8_t)e;
            int32_t S[2];
            qa::block_sums(blk, q, S);
            CHECK(S[0] == 32 * fill * e && S[1] == 32 * fill * e);  // a saturated pair would show here
        }
    for (int it = 0; it < 20000; ++it) {
        for (int b = 2; b < 18; ++b) blk[b] = (uint8_t)rnd(256);
        for (auto& v : q) v = (int8_t)(rnd(2) ? (rnd(2) ? 127 : -127) : (int)rnd(255) - 127);
        int32_t S[2];
        qa::block_sums(blk, q, S);
        CHECK(S[0] == scalar_sum(blk + 2, 0, q) && S[1] == scalar_sum(blk + 2, 1, q + 32));
    }
}
U7_SECTION(blockResult_correction, "d (s0 S0 + s1 S1) - d (hx0 + hx1) = d (s0 (S0 - u0) + s1 (S1 - u1)), the code-offset correction") {
    for (int it = 0; it < 4000; ++it) {
        const int64_t d = (int64_t)rnd(9) - 4, s0 = (int64_t)rnd(9) - 4, s1 = (int64_t)rnd(9) - 4;
        const int64_t S0 = (int64_t)rnd(24385) - 12192, S1 = (int64_t)rnd(24385) - 12192;
        const int64_t u0 = (int64_t)rnd(8129) - 4064, u1 = (int64_t)rnd(8129) - 4064;
        CHECK(qm::block_result(d, s0, s1, S0, S1, u0, u1) == qm::block_result_corrected(d, s0, s1, S0, S1, u0, u1));
    }
    uint8_t blk[18];
    int8_t q[64];
    for (int it = 0; it < 2000; ++it) {  // one block: fma(wd*xd1, S1-hx1, fma(wd*xd0, S0-hx0, 0)), bit-exact
        const uint16_t h = (uint16_t)(0x3000 + rnd(0x1000));
        std::memcpy(blk, &h, 2);
        for (int b = 2; b < 18; ++b) blk[b] = (uint8_t)rnd(256);
        for (auto& v : q) v = (int8_t)((int)rnd(255) - 127);
        const float xd[2] = {(float)(rnd(1000) + 1) / 997.f, (float)(rnd(1000) + 1) / 991.f};
        int32_t hx[2] = {0, 0};
        for (int i = 0; i < 64; ++i) hx[i / 32] += q[i];
        float out = -1.f;
        qa::row_multi(blk, 1, 1, q, 0, xd, hx, 0, &out);
        const float wd = baloon::f16::half_to_float(h);
        const float ref = std::fmaf(wd * xd[1], (float)(scalar_sum(blk + 2, 1, q + 32) - hx[1]),
                                    std::fmaf(wd * xd[0], (float)(scalar_sum(blk + 2, 0, q) - hx[0]), 0.f));
        CHECK(out == ref);
    }
}
U7_SECTION(shuffle_constants, "0x4E swaps the two 64-bit halves ([2,3,0,1]); 0xB1 swaps adjacent lanes ([1,0,3,2])") {
    const uint32_t halves[4] = {2, 3, 0, 1}, pairs[4] = {1, 0, 3, 2};
    alignas(16) int32_t src[4] = {10, 11, 12, 13}, a[4], b[4];
    const __m128i v = _mm_load_si128((const __m128i*)src);
    _mm_store_si128((__m128i*)a, _mm_shuffle_epi32(v, qm::kSwapHalves));
    _mm_store_si128((__m128i*)b, _mm_shuffle_epi32(v, qm::kSwapPairs));
    for (uint32_t i = 0; i < 4; ++i) {
        CHECK(qm::shuf_lane(qm::kSwapHalves, i) == halves[i] && qm::shuf_lane(qm::kSwapPairs, i) == pairs[i]);
        CHECK(a[i] == src[halves[i]] && b[i] == src[pairs[i]]);  // the instruction itself
    }
}
U7_SECTION(hsum4_exact, "lane 0 after s += shuf(s, 0x4E); s += shuf(s, 0xB1) holds the sum of all four lanes") {
    int32_t v[8];
    for (int it = 0; it < 20000; ++it) {
        int64_t ref = 0;
        for (auto& x : v) { x = (int32_t)rnd(1u << 24) - (1 << 23); ref += x; }
        CHECK(qa::hsum8(v) == ref);
        CHECK(qm::hsum4(v[0] + v[4], v[1] + v[5], v[2] + v[6], v[3] + v[7]) == ref);
    }
}
U7_SECTION(chunk_sum_bound, "a chunk sum of 32 clamped activations fits easily in int32 (|sum| <= 4064)") {
    std::vector<float> x(32 * 80);
    std::vector<int8_t> q(x.size());
    std::vector<float> d(80);
    std::vector<int32_t> hx(80);
    for (auto& e : x) e = (float)((int)rnd(20001) - 10000) / 997.f;
    for (int c = 0; c < 80; c += 7) for (int i = 0; i < 32; ++i) x[32 * c + i] = 1e3f;     // all +127
    for (int c = 3; c < 80; c += 11) for (int i = 0; i < 32; ++i) x[32 * c + i] = -1e3f;  // all -127
    qa::act_quant_q8_1_avx2(x.data(), (uint32_t)x.size(), q.data(), d.data(), hx.data());
    int32_t extreme = 0;
    for (int c = 0; c < 80; ++c) {
        int32_t s = 0;
        for (int i = 0; i < 32; ++i) s += q[32 * c + i];
        CHECK(hx[c] == s && std::abs(s) <= qm::kChunkSumMax);
        extreme = std::max(extreme, std::abs(s));
    }
    CHECK(extreme == qm::kChunkSumMax);  // the bound is reached
}
U7_SECTION(groupsOf4_cover, "the groups cover every token, and each has a width the switch handles (1..4)") {
    const uint32_t rows_n = 3, nb = 5, ncs = 2 * nb;
    std::vector<uint8_t> w(rows_n * nb * 18);
    for (auto& b : w) b = (uint8_t)rnd(256);
    for (uint32_t r = 0; r < rows_n * nb; ++r) { const uint16_t h = (uint16_t)(0x2C00 + rnd(0x1000)); std::memcpy(&w[r * 18], &h, 2); }
    for (uint32_t nt = 1; nt <= 13; ++nt) {
        std::vector<float> x((size_t)nt * nb * 64), xd((size_t)nt * ncs);
        std::vector<int8_t> q(x.size());
        std::vector<int32_t> hx(xd.size());
        for (auto& e : x) e = (float)((int)rnd(2001) - 1000) / 333.f;
        for (uint32_t t = 0; t < nt; ++t)
            qa::act_quant_q8_1_avx2(&x[(size_t)t * nb * 64], nb * 64, &q[(size_t)t * nb * 64], &xd[(size_t)t * ncs], &hx[(size_t)t * ncs]);
        std::vector<float> out((size_t)nt * rows_n, std::nanf(""));
        qa::q2_0_gguf_rows_multi_avx2(w.data(), 0, rows_n, nb, nt, q.data(), nb * 64, xd.data(), hx.data(), ncs, out.data(), rows_n);
        for (uint32_t t = 0; t < nt; ++t) {
            float one[rows_n];
            qa::rows(w.data(), 0, rows_n, nb, &q[(size_t)t * nb * 64], &xd[(size_t)t * ncs], &hx[(size_t)t * ncs], one);
            for (uint32_t r = 0; r < rows_n; ++r) CHECK(out[(size_t)t * rows_n + r] == one[r]);  // written once, bit-identical
        }
        uint32_t covered = 0;
        for (uint32_t t0 = 0; t0 < nt; t0 += 4) { const uint32_t g = qm::token_group(nt, t0); CHECK(g >= 1 && g <= 4); covered += g; }
        CHECK(covered == nt);
    }
}
U7_SECTION(gguf_block_stride, "GGUF Q2_0 rows stride 18 bytes per block: block b has its scale at 18 b and codes at 18 b + 2") {
    for (uint32_t nb = 1; nb <= 12; ++nb) {
        std::vector<uint8_t> w(2 * qm::row_bytes(nb));  // row 0 then row 1
        for (auto& b : w) b = (uint8_t)rnd(256);
        for (uint32_t b = 0; b < 2 * nb; ++b) { const uint16_t h = (uint16_t)(0x2C00 + rnd(0x1000)); std::memcpy(&w[qm::block_scale_off(b)], &h, 2); }
        std::vector<float> x(nb * 64), xd(2 * nb);
        std::vector<int8_t> q(x.size());
        std::vector<int32_t> hx(2 * nb);
        for (auto& e : x) e = (float)((int)rnd(2001) - 1000) / 333.f;
        qa::act_quant_q8_1_avx2(x.data(), nb * 64, q.data(), xd.data(), hx.data());
        float ref = 0.f;  // scalar reference reading scale at 18 b and codes at 18 b + 2
        for (uint32_t b = 0; b < nb; ++b) {
            uint16_t h; std::memcpy(&h, &w[qm::block_scale_off(b)], 2);
            const float wd = baloon::f16::half_to_float(h);
            for (uint32_t k = 0; k < 2; ++k)
                ref = std::fmaf(wd * xd[2 * b + k], (float)(scalar_sum(&w[qm::block_codes_off(b)], k, &q[64 * b + 32 * k]) - hx[2 * b + k]), ref);
        }
        float o0 = 0.f, o1 = 0.f;
        qa::rows(w.data(), 0, 1, nb, q.data(), xd.data(), hx.data(), &o0);
        for (size_t i = qm::row_bytes(nb); i < w.size(); ++i) w[i] = (uint8_t)~w[i];  // row 1 only
        qa::rows(w.data(), 0, 1, nb, q.data(), xd.data(), hx.data(), &o1);
        CHECK(o0 == ref && o1 == o0);  // nothing past 18 nb is read for row 0
    }
}
// ==== Q2ExpertDotKernel ========================================================================================
U7_SECTION(offset_identity, "the code offset splits off as the plain activation sum: sum (c-1) x = sum c x - sum x") {
    for (int it = 0; it < 20000; ++it) {
        const uint32_t n = rnd(65);
        int32_t dec = 0, dot = 0, sum = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const int c = (int)rnd(4), x = (int)rnd(255) - 127;
            dec += (c - 1) * x; dot += c * x; sum += x;
        }
        CHECK(dec == qe::offset_dot(dot, sum));
    }
    uint8_t codes[16]; int8_t qc[32];
    for (int it = 0; it < 20000; ++it) {
        for (auto& c : codes) c = (uint8_t)rnd(256);
        int32_t sum = 0;
        for (auto& v : qc) { v = (int8_t)((int)rnd(255) - 127); sum += v; }
        const uint32_t h = rnd(2);
        CHECK(qe::half_decoded_dot(codes, 0, h, qc) == qe::offset_dot(qe::half_code_dot(codes, 0, h, qc), sum));
    }
}
static std::vector<uint8_t> random_blob() {  // valid fp16 scales in both plane orders (q2_expert and kern)
    std::vector<uint8_t> blob(qe::BLOB);
    for (auto& b : blob) b = (uint8_t)rnd(256);
    auto scales = [&](uint64_t off, uint64_t bytes) {
        for (uint64_t i = 0; i < bytes; i += 2) { const uint16_t h = (uint16_t)(0x1C00 + rnd(0x0800)); std::memcpy(&blob[off + i], &h, 2); }
    };
    scales(qe::GU_SCALES_OFF, qe::GU_SCALES_BYTES);
    scales(qe::D_SCALES_OFF, qe::D_SCALES_BYTES);
    scales(baloon::kern::O_GU_SCALES, (uint64_t)2 * qe::FF * qe::SC_GU * 2);
    scales(baloon::kern::O_D_SCALES, (uint64_t)qe::H * qe::SC_D * 2);
    return blob;
}

U7_SECTION(chunk_kernel_exact, "one chunk as the kernel computes it, dw * (dx * sum c x - hx), equals the exact dequantized product dw * dx * sum (c-1) x") {
    for (int it = 0; it < 20000; ++it) {
        const int64_t dw = (int64_t)rnd(2001) - 1000, dx = (int64_t)rnd(2001) - 1000;
        const int64_t dot = (int64_t)rnd(24385) - 12192, sum = (int64_t)rnd(8129) - 4064;
        CHECK(qe::chunk_kernel(dw, dx, dot, sum) == dw * dx * (dot - sum));
        CHECK(qe::chunk_hx(dx, sum) == dx * sum);
    }
    for (int it = 0; it < 200; ++it) {  // row_dot against the exact dequantized product in double
        const uint32_t nb = 1 + rnd(40);
        std::vector<uint8_t> codes(nb * 16), sc(nb * 2);
        for (auto& c : codes) c = (uint8_t)rnd(256);
        for (uint32_t b = 0; b < nb; ++b) { const uint16_t h = (uint16_t)(0x2800 + rnd(0x1000)); std::memcpy(&sc[2 * b], &h, 2); }
        std::vector<float> x(nb * 64);
        for (auto& e : x) e = (float)((int)rnd(4001) - 2000) / 701.f;
        const qe::ActQ a = qe::act_quant(x.data(), nb * 64);
        double exact = 0.0;
        for (uint32_t b = 0; b < nb; ++b)
            for (uint32_t h = 0; h < 2; ++h) {
                double dec = 0.0;
                for (uint32_t i = 0; i < 32; ++i) dec += ((int)qe::block_code(codes.data(), b, h * 32 + i) - 1) * a.q[(2 * b + h) * 32 + i];
                exact += baloon::f16::half_to_float(qe::load_u16(&sc[2 * b])) * (double)a.d[2 * b + h] * dec;
            }
        const float got = qe::row_dot(codes.data(), sc.data(), nb, a.q.data(), a.d.data(), a.hx.data());
        CHECK(std::fabs(got - exact) <= 1e-4 * (1.0 + std::fabs(exact)));
    }
    if (!avx512_usable()) { std::printf("  (AVX-512 VNNI/VBMI not usable: VNNI expert not run)\n"); return; }
    const std::vector<uint8_t> blob = random_blob();  // VNNI expert: same one-term-per-chunk form, bit-identical
    std::vector<float> x(qe::H), o_s(qe::H), o_v(qe::H);
    for (auto& e : x) e = (float)((int)rnd(2001) - 1000) / 500.f;
    static baloon::kern::ExpertScratch scr;
    baloon::kern::s2_expert_scalar(blob.data(), x.data(), o_s.data());
    baloon::kern::s2_expert_vnni(blob.data(), x.data(), o_v.data(), scr);
    for (uint32_t r = 0; r < qe::H; ++r) CHECK(o_s[r] == o_v[r]);
}
U7_SECTION(chunkDotOracle_exact, "the oracle's integer result is exactly sum (c - 1) xq - no rounding before the scale") {
    uint8_t codes[16];
    int8_t qc[32];
    for (int it = 0; it < 50000; ++it) {
        for (auto& c : codes) c = (uint8_t)rnd(256);
        int32_t sum = 0;
        for (auto& v : qc) { v = (int8_t)((int)rnd(255) - 127); sum += v; }
        const uint32_t h = rnd(2);
        const int32_t got = qe::chunk_dot_oracle(codes, h, qc, sum);
        CHECK(got == qe::half_decoded_dot(codes, 0, h, qc));
        CHECK(got == qe::half_code_dot(codes, 0, h, qc) - sum);
    }
}
U7_SECTION(clampQ_range, "clamped activations lie in [-127, 127] and the clamp is idempotent") {
    for (int v = -100000; v <= 100000; ++v) {
        const int c = qe::clamp127(v);
        if (c < -127 || c > 127 || qe::clamp127(c) != c || (v >= -127 && v <= 127 && c != v)) { CHECK(false); break; }
    }
    std::vector<float> x(32 * 64);
    for (auto& e : x) e = (rnd(2) ? 1.f : -1.f) * (float)rnd(1u << 20) * (rnd(7) ? 1e-3f : 1e9f);
    std::vector<int8_t> q1(x.size()), q2(x.size());
    std::vector<float> d1(64), d2(64);
    std::vector<int32_t> h1(64), h2(64);
    qe::act_quant(x.data(), (uint32_t)x.size(), q1.data(), d1.data(), h1.data());
    qa::act_quant_q8_1_avx2(x.data(), (uint32_t)x.size(), q2.data(), d2.data(), h2.data());
    for (size_t i = 0; i < x.size(); ++i) CHECK(q1[i] >= -127 && q1[i] <= 127 && q2[i] >= -127 && q2[i] <= 127);
    for (int c = 0; c < 64; ++c) { int32_t s = 0; for (int i = 0; i < 32; ++i) s += q1[32 * c + i]; CHECK(h1[c] == s); }
}
__attribute__((target("avx512f,avx512bw,avx512vnni"))) static void dpbusd_hw(const uint8_t* c64, const int8_t* x64, int32_t lanes[16]) {
    _mm512_storeu_si512(lanes, _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_loadu_si512(c64), _mm512_loadu_si512(x64)));
}

U7_SECTION(vnni_lane_bound, "a VNNI lane adds four u8 code * s8 act products: bounded by 1524, far from int32 overflow") {
    int32_t worst = 0;
    for (uint32_t m = 0; m < 256 * 16; ++m) {  // every code quadruple times every sign pattern of |x| = 127
        int32_t s = 0;
        for (int k = 0; k < 4; ++k) s += (int32_t)((m >> (2 * k)) & 3u) * ((m >> (8 + k)) & 1u ? 127 : -127);
        CHECK(std::abs(s) <= qe::kVnniLaneMax);
        worst = std::max(worst, std::abs(s));
    }
    CHECK(worst == qe::kVnniLaneMax);
    if (!avx512_usable()) return;
    uint8_t c[64]; int8_t x[64]; int32_t lanes[16];
    for (int it = 0; it < 5000; ++it) {
        for (int i = 0; i < 64; ++i) { c[i] = (uint8_t)(it % 2 ? 3 : rnd(4)); x[i] = (int8_t)(it % 3 == 0 ? (rnd(2) ? 127 : -127) : (int)rnd(255) - 127); }
        dpbusd_hw(c, x, lanes);
        for (int l = 0; l < 16; ++l) {
            int32_t s = 0;
            for (int k = 0; k < 4; ++k) s += c[4 * l + k] * x[4 * l + k];
            CHECK(lanes[l] == s && std::abs(lanes[l]) <= qe::kVnniLaneMax);
        }
    }
}
U7_SECTION(zsum_bound, "bound of a list of code-times-activation products: |sum| <= 381 * length") {
    for (int it = 0; it < 20000; ++it) {
        const uint32_t n = rnd(129);
        int64_t s = 0;
        const bool extreme = it % 4 == 0;
        for (uint32_t i = 0; i < n; ++i) s += (int64_t)(extreme ? 3 : rnd(4)) * (extreme ? -127 : (int)rnd(255) - 127);
        CHECK(std::llabs(s) <= qe::code_dot_bound(n));
        if (extreme) CHECK(std::llabs(s) == qe::code_dot_bound(n));  // tight
    }
    uint8_t codes[16]; int8_t qc[32];
    for (int it = 0; it < 20000; ++it) {
        for (auto& c : codes) c = (uint8_t)rnd(256);
        for (auto& v : qc) v = (int8_t)((int)rnd(255) - 127);
        CHECK(std::abs(qe::half_code_dot(codes, 0, rnd(2), qc)) <= qe::code_dot_bound(32));
    }
}
U7_SECTION(chunk_dot_exact_in_f32, "a 32-element chunk's integer dot is at most 12192 < 2^24, so its f32 conversion is exact") {
    for (int32_t v = -qe::kChunkDotMax - 4064; v <= qe::kChunkDotMax + 4064; ++v)  // also after subtracting a chunk sum
        if ((int32_t)qe::chunk_dot_to_f32(v) != v) { CHECK((int32_t)qe::chunk_dot_to_f32(v) == v); break; }
    CHECK((int32_t)qe::chunk_dot_to_f32((1 << 24) + 1) != (1 << 24) + 1);  // the 2^24 limit is real
    uint8_t codes[16]; int8_t qc[32];
    std::memset(codes, 0xFF, 16);
    for (auto& v : qc) v = -127;
    CHECK(qe::half_code_dot(codes, 0, 0, qc) == -qe::kChunkDotMax);
}
U7_SECTION(ctrl_shifts, "the multishift control 0x0E0C0A0806040200 selects shifts 0, 2, 4, ..., 14") {
    for (uint32_t i = 0; i < 8; ++i) CHECK(qe::ctrl_byte(i) == 2 * i);
    CHECK(qe::kMultishiftCtrl == 0x0E0C0A0806040200ull);
}
__attribute__((target("avx512f,avx512bw,avx512vbmi"))) static void multishift_hw(const uint8_t* blk, uint8_t out[64]) {
    const __m512i ctrl = _mm512_set1_epi64((long long)qe::kMultishiftCtrl);
    const __m512i src = _mm512_cvtepu16_epi64(_mm_loadu_si128((const __m128i*)blk));
    _mm512_storeu_si512(out, _mm512_and_si512(_mm512_multishift_epi64_epi8(ctrl, src), _mm512_set1_epi8(3)));
}

U7_SECTION(multishift_code_order, "output byte i of a u16 lane b0 + 256 b1, masked with 3, is the i-th code in LSB-first order") {
    bool ok = true;
    for (uint32_t w = 0; w < 65536 && ok; ++w)
        for (uint32_t i = 0; i < 8; ++i) ok = ok && qe::multishift_code(w, i) == (((i < 4 ? w & 0xFF : w >> 8) >> (2 * (i % 4))) & 3u);
    CHECK(ok);
    uint8_t blk[16], a[64], b[64], c[64];
    const bool hw = avx512_usable();
    for (int it = 0; it < 4096; ++it) {
        for (auto& e : blk) e = (uint8_t)rnd(256);
        qe::unpack_block(blk, a);
        qa::unpack64(blk, b);
        CHECK(std::memcmp(a, b, 64) == 0);                              // same order as the AVX2 unpack
        if (hw) { multishift_hw(blk, c); CHECK(std::memcmp(a, c, 64) == 0); }  // and as vpmultishiftqb itself
    }
    if (!hw) std::printf("  (AVX-512 VBMI not usable: vpmultishiftqb not run)\n");
}
__attribute__((target("avx512f"))) static void masked_lanes_hw(uint32_t nb, float out[16]) {
    _mm512_storeu_ps(out, _mm512_maskz_mov_ps((__mmask16)qe::lane_mask(nb), _mm512_set1_ps(1.f)));
}

U7_SECTION(laneMask_spec, "the mask always enables exactly the 2 nb chunk lanes of the batch") {
    for (uint32_t nb = 0; nb <= 8; ++nb) {
        CHECK(qe::lane_mask(nb) == (uint32_t)((1ull << (2 * nb)) - 1) && (uint32_t)__builtin_popcount(qe::lane_mask(nb)) == 2 * nb);
        if (!avx512_usable()) continue;
        float lanes[16];
        masked_lanes_hw(nb, lanes);
        for (uint32_t l = 0; l < 16; ++l) CHECK(lanes[l] == (l < 2 * nb ? 1.f : 0.f));  // chunk lanes 2k, 2k+1 of block k
    }
    CHECK(qe::lane_mask(8) == 0xFFFF && qe::lane_mask(0) == 0);
}
U7_SECTION(batchSizes_cover, "batches cover every block exactly once and none exceeds 8") {
    for (uint32_t n = 0; n <= 200; ++n) {
        std::vector<int> seen(n, 0);
        for (uint32_t b0 = 0; b0 < n; b0 += 8) {
            const uint32_t nb = qe::batch_size(n, b0);
            CHECK(nb >= 1 && nb <= 8 && nb == std::min<uint32_t>(8, n - b0));
            for (uint32_t k = 0; k < nb; ++k) ++seen[b0 + k];
        }
        for (uint32_t b = 0; b < n; ++b) CHECK(seen[b] == 1);
    }
    if (!avx512_usable()) return;
    // gate/up rows: 40 blocks = 5 full batches; down rows: 10 blocks = 8 + a tail of 2. VNNI == scalar on both.
    const std::vector<uint8_t> blob = random_blob();
    std::vector<float> x(qe::H), o_s(qe::H), o_v(qe::H);
    static baloon::kern::ExpertScratch scr;
    for (int rep = 0; rep < 3; ++rep) {
        for (auto& e : x) e = (float)((int)rnd(4001) - 2000) / 300.f;
        baloon::kern::s2_expert_scalar(blob.data(), x.data(), o_s.data());
        baloon::kern::s2_expert_vnni(blob.data(), x.data(), o_v.data(), scr);
        for (uint32_t r = 0; r < qe::H; ++r) CHECK(o_s[r] == o_v[r]);
    }
}
U7_SECTION(block_chunk_map, "block b of a row uses activation chunks 2b and 2b+1, at int8 offsets 64b and 64b + 32") {
    for (uint32_t b = 0; b < 64; ++b) {
        CHECK(qe::block_chunk(b, 0) == 2 * b && qe::block_chunk(b, 1) == 2 * b + 1);
        CHECK(qe::chunk_q_offset(b, 0) == 64 * b && qe::chunk_q_offset(b, 1) == 64 * b + 32);
    }
    // Row dot of nb blocks against activations where only chunk 2b+h is nonzero: only block b's half h contributes.
    const uint32_t nb = 6;
    std::vector<uint8_t> codes(nb * 16), sc(nb * 2);
    for (auto& c : codes) c = (uint8_t)rnd(256);
    for (uint32_t b = 0; b < nb; ++b) { const uint16_t h = 0x3C00; std::memcpy(&sc[2 * b], &h, 2); }  // scale 1.0
    std::vector<int8_t> q(nb * 64);
    std::vector<float> xd(2 * nb, 1.f);
    std::vector<int32_t> hx(2 * nb, 0);
    for (uint32_t b = 0; b < nb; ++b)
        for (uint32_t h = 0; h < 2; ++h) {
            std::fill(q.begin(), q.end(), 0);
            for (uint32_t i = 0; i < 32; ++i) q[qe::chunk_q_offset(b, h) + i] = (int8_t)((int)rnd(255) - 127);
            const float got = qe::row_dot(codes.data(), sc.data(), nb, q.data(), xd.data(), hx.data());
            CHECK(got == (float)qe::half_code_dot(codes.data(), b, h, &q[qe::chunk_q_offset(b, h)]));
        }
}
U7_SECTION(row_geometry, "H / 64 = 40 and FF / 64 = 10 blocks per row; 40 * 16 = 640 and 10 * 16 = 160 code bytes") {
    CHECK(qe::H == 2560 && qe::FF == 640);
    CHECK(qe::row_blocks(qe::H) == 40 && qe::row_blocks(qe::FF) == 10 && qe::row_code_bytes(qe::H) == 640 && qe::row_code_bytes(qe::FF) == 160);
    CHECK(baloon::kern::SC_GU == 40 && baloon::kern::SC_D == 10 && baloon::kern::ROW_GU == 640 && baloon::kern::ROW_D == 160);
    const uint8_t* z = nullptr;
    for (uint32_t s = 1; s < 2 * qe::FF; ++s) CHECK(qe::gu_codes(z, s) - qe::gu_codes(z, s - 1) == 640 && qe::gu_scales(z, s) - qe::gu_scales(z, s - 1) == 80);
    for (uint32_t r = 1; r < qe::H; ++r) CHECK(qe::down_codes(z, r) - qe::down_codes(z, r - 1) == 160 && qe::down_scales(z, r) - qe::down_scales(z, r - 1) == 20);
}
U7_SECTION(blob_size, "gate/up: 2 FF rows of (640 code + 80 scale) bytes; down: H rows of (160 + 20); sum 1,382,400") {
    CHECK(qe::kGuRowBytes == 720 && qe::kDownRowBytes == 180 && 2ull * qe::FF * 720 + (uint64_t)qe::H * 180 == 1382400);
    CHECK(qe::BLOB == 1382400 && baloon::kern::BLOB == 1382400);
    // Every byte of the blob belongs to exactly one row record, in both plane orders (q2_expert and kern).
    std::vector<uint8_t> hit(qe::BLOB, 0), khit(qe::BLOB, 0);
    const uint8_t* z = nullptr;
    for (uint32_t s = 0; s < 2 * qe::FF; ++s) {
        for (uint32_t i = 0; i < qe::ROW_GU; ++i) { ++hit[qe::gu_codes(z, s) - z + i]; ++khit[baloon::kern::O_GU_CODES + (uint64_t)s * qe::ROW_GU + i]; }
        for (uint32_t i = 0; i < qe::SC_GU * 2; ++i) { ++hit[qe::gu_scales(z, s) - z + i]; ++khit[baloon::kern::O_GU_SCALES + (uint64_t)s * qe::SC_GU * 2 + i]; }
    }
    for (uint32_t r = 0; r < qe::H; ++r) {
        for (uint32_t i = 0; i < qe::ROW_D; ++i) { ++hit[qe::down_codes(z, r) - z + i]; ++khit[baloon::kern::O_D_CODES + (uint64_t)r * qe::ROW_D + i]; }
        for (uint32_t i = 0; i < qe::SC_D * 2; ++i) { ++hit[qe::down_scales(z, r) - z + i]; ++khit[baloon::kern::O_D_SCALES + (uint64_t)r * qe::SC_D * 2 + i]; }
    }
    bool once = true;
    for (uint64_t i = 0; i < qe::BLOB; ++i) once = once && hit[i] == 1 && khit[i] == 1;
    CHECK(once);
}
U7_SECTION(gate_up_interleave, "row r's gate is 2r and up is 2r+1; distinct and in range") {
    for (uint32_t r = 0; r < qe::FF; ++r)
        CHECK(qe::gate_slot(r) == 2 * r && qe::up_slot(r) == 2 * r + 1 && qe::up_slot(r) < 2 * qe::FF && qe::slot_row(qe::gate_slot(r)) == r &&
              qe::slot_row(qe::up_slot(r)) == r && !qe::slot_is_up(qe::gate_slot(r)) && qe::slot_is_up(qe::up_slot(r)));
    // The kern expert reads gate from slot 2r and up from slot 2r + 1: act[r] = silu(gate) * up, not the swapped pair.
    const std::vector<uint8_t> blob = random_blob();
    std::vector<float> x(qe::H);
    for (auto& e : x) e = (float)((int)rnd(2001) - 1000) / 400.f;
    static baloon::kern::ActQ xq;
    baloon::kern::act_quant_q8_1(x.data(), qe::H, xq.q, xq.d, xq.hx);
    std::vector<float> act(qe::FF);
    baloon::kern::s2_expert_gu_rows(blob.data(), xq, 0, qe::FF, act.data());
    int distinguished = 0;
    for (uint32_t r = 0; r < qe::FF; r += 7) {
        auto rd = [&](uint32_t slot) {
            return qe::row_dot(blob.data() + baloon::kern::O_GU_CODES + (uint64_t)slot * qe::ROW_GU,
                               blob.data() + baloon::kern::O_GU_SCALES + (uint64_t)slot * qe::SC_GU * 2, qe::SC_GU, xq.q, xq.d, xq.hx);
        };
        const float want = qe::silu(rd(qe::gate_slot(r))) * rd(qe::up_slot(r));
        const float swapped = qe::silu(rd(qe::up_slot(r))) * rd(qe::gate_slot(r));
        const float tol = 1e-3f * (1.f + std::fabs(want));
        CHECK(std::fabs(act[r] - want) <= tol);
        if (std::fabs(want - swapped) > 10 * tol) { CHECK(std::fabs(act[r] - swapped) > tol); ++distinguished; }
    }
    CHECK(distinguished > 10);
}
U7_SECTION(dispatchNT_exact, "for 1..8 tokens the dispatched template width equals the token count (cases 1..7, default 8)") {
    for (uint32_t n = 1; n <= 8; ++n) CHECK(qe::dispatch_nt(n) == n && baloon::kern::q2geo::dispatch_nt(n) == n);
    CHECK(qe::dispatch_nt(9) == 8 && qe::dispatch_nt(100) == 8);
    const std::vector<uint8_t> blob = random_blob();
    std::vector<float> x(8 * qe::H), multi(8 * qe::H), single(qe::H);
    for (auto& e : x) e = (float)((int)rnd(2001) - 1000) / 400.f;
    for (uint32_t nt : {1u, 3u, 8u}) {  // the scalar multi-token path at width dispatch_nt(nt) == nt
        qe::expert_forward_multi(blob.data(), x.data(), nt, multi.data());
        for (uint32_t t = 0; t < nt; ++t) {
            qe::expert_forward(blob.data(), &x[(size_t)t * qe::H], single.data());
            for (uint32_t r = 0; r < qe::H; ++r) CHECK(multi[(size_t)t * qe::H + r] == single[r]);
        }
    }
    if (!avx512_usable()) return;
    static baloon::kern::ExpertScratchMulti sm;
    static baloon::kern::ExpertScratch s1;
    for (uint32_t nt = 1; nt <= 8; ++nt) {  // the VNNI multi-token template at every width
        baloon::kern::s2_expert_vnni_multi(blob.data(), x.data(), nt, multi.data(), sm);
        for (uint32_t t = 0; t < nt; ++t) {
            baloon::kern::s2_expert_vnni(blob.data(), &x[(size_t)t * qe::H], single.data(), s1);
            for (uint32_t r = 0; r < qe::H; ++r) CHECK(multi[(size_t)t * qe::H + r] == single[r]);
        }
    }
}
U7_SECTION(cpuUsable_iff, "usable iff all five bits are set; dropping any one makes the CPU unusable, which reason() reports individually") {
    for (uint32_t m = 0; m < 32; ++m) {  // every subset of the five bits, with unrelated bits set around them
        const uint32_t e = (m & 1 ? 1u << 16 : 0) | (m & 2 ? 1u << 30 : 0) | (m & 4 ? 1u << 31 : 0);
        const uint32_t c = (m & 8 ? 1u << 11 : 0) | (m & 16 ? 1u << 1 : 0);
        CHECK(qe::detect(7, e | 0x0F0F, c | 0x100).usable() == (m == 31));
    }
    const uint32_t ebx = (1u << 16) | (1u << 30) | (1u << 31), ecx = (1u << 11) | 2u;
    const struct { bool in_ebx; uint32_t bit; const char* why; } need[5] = {
        {true, 16, "AVX-512F is missing"}, {true, 30, "AVX-512BW is missing"}, {true, 31, "AVX-512VL is missing"},
        {false, 11, "AVX-512 VNNI is missing"}, {false, 1, "AVX-512 VBMI is missing"}};
    for (const auto& n : need) {
        const qe::CpuFeatures f = qe::detect(7, n.in_ebx ? ebx & ~(1u << n.bit) : ebx, n.in_ebx ? ecx : ecx & ~(1u << n.bit));
        CHECK(!f.usable() && std::strcmp(f.reason(), n.why) == 0);
        baloon::kern::CpuFeatures k;  // kern's reason() is the same definition
        k.avx512f = f.avx512f; k.avx512bw = f.avx512bw; k.avx512vl = f.avx512vl; k.avx512_vnni = f.avx512_vnni; k.avx512_vbmi = f.avx512_vbmi;
        CHECK(!k.usable() && std::strcmp(k.reason(), n.why) == 0);
    }
}
U7_SECTION(cpuUsable_witness, "the minimal EBX/ECX words with exactly those bits are accepted; without VBMI rejected") {
    CHECK(qe::detect(7, qe::kGateEbx, qe::kGateEcx).usable() && std::strcmp(qe::detect(7, qe::kGateEbx, qe::kGateEcx).reason(), "ok") == 0);
    CHECK(!qe::detect(7, qe::kGateEbx, 1u << 11).usable());
    CHECK(std::strcmp(qe::detect(7, qe::kGateEbx, 1u << 11).reason(), "AVX-512 VBMI is missing") == 0);
    CHECK(!qe::detect(6, qe::kGateEbx, qe::kGateEcx).usable());  // max leaf below 7 reports nothing
    const baloon::kern::CpuFeatures host = baloon::kern::cpu_features();
    if (host.usable()) CHECK(qe::detect_host().usable());
    CHECK(host.usable() == (std::strcmp(host.reason(), "ok") == 0));
    std::printf("  host: avx2=%d gate=%s (%s)\n", (int)host.avx2, host.usable() ? "usable" : "unusable", host.reason());
}
int main() { return u7::run_all("U00B8180AA2 CPU TESTS PASSED"); }

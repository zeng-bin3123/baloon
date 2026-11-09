// u62b_checks.hpp - invariant checks (the LAYER 1 statements of the seven modules of unit u62b936371e) run against the
// repo's implementations. Every check increments a failure counter instead of aborting, so one run reports everything.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "baloon/artifact_dequant.hpp"
#include "baloon/canonical_form_roundtrip.hpp"
#include "baloon/iq_native_pack_index.hpp"
#include "baloon/iq_quant_expert_layout.hpp"
#include "baloon/mtp_expert_quantizer.hpp"
#include "baloon/q2_gguf_writer.hpp"
#include "baloon/ref_pack_builder.hpp"

namespace u62b {

static int g_fail = 0, g_checks = 0;
#define U62B_CHECK(cond)                                                                      \
    do {                                                                                      \
        ++u62b::g_checks;                                                                     \
        if (!(cond)) { ++u62b::g_fail; std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } \
    } while (0)

// ---- CanonicalQuantBitExactMaps -----------------------------------------------------------------------------------------
inline void canonical() {
    using namespace baloon::canon;
    struct Row { QType t; unsigned bb, be, ge; int bias; bool off; };
    // Theory rows (bias is the integer added to the code: decoded = code + bias).
    const Row rows[] = {{QType::Q2_0, 18, 64, 64, -1, false}, {QType::Q3_K, 110, 256, 16, -4, false},
                        {QType::Q4_0, 18, 32, 32, -8, false}, {QType::Q4_K, 144, 256, 32, 0, true},
                        {QType::Q5_0, 22, 32, 32, -16, false}, {QType::Q5_K, 176, 256, 32, 0, true},
                        {QType::Q6_K, 210, 256, 16, -32, false}, {QType::Q8_0, 34, 32, 32, -128, false},
                        {QType::IQ4_NL, 18, 32, 32, 0, false}, {QType::IQ4_XS, 136, 256, 32, 0, false}};
    int offset_rows = 0;
    for (const Row& r : rows) {
        const Mapping* m = find_mapping(r.t);
        U62B_CHECK(m != nullptr);
        if (!m) continue;
        U62B_CHECK(m->block_bytes == r.bb && m->block_elems == r.be && m->group_elems == r.ge);
        U62B_CHECK(-m->code_bias == r.bias);          // registry stores decoded = code - code_bias
        U62B_CHECK(m->has_offset == r.off);
        U62B_CHECK(m->block_elems % m->group_elems == 0);
        offset_rows += m->has_offset ? 1 : 0;
    }
    U62B_CHECK(offset_rows == 2);
    // Block byte decompositions.
    U62B_CHECK(18 == 2 + 64 / 4 && 110 == 32 + 64 + 12 + 2 && 144 == 2 + 2 + 12 + 128 && 176 == 2 + 2 + 12 + 32 + 128);
    U62B_CHECK(210 == 128 + 64 + 16 + 2 && 22 == 2 + 4 + 16 && 34 == 2 + 32 && 136 == 2 + 2 + 4 + 128);
    // Integer-bias decode ranges (exhaustive).
    for (int c = 0; c < 4; ++c) { U62B_CHECK(c - 1 >= -1 && c - 1 <= 2); U62B_CHECK(c - 1 != c); }
    for (int c = 0; c < 16; ++c) { U62B_CHECK(c - 8 >= -8 && c - 8 <= 7); }
    for (int c = 0; c < 32; ++c) { U62B_CHECK(c - 16 >= -16 && c - 16 <= 15); }
    for (int s = -128; s <= 127; ++s) { const int c = s + 128; U62B_CHECK(c >= 0 && c < 256 && c - 128 == s); }
    // Nibble split and Q2_0 bit plane (exhaustive).
    for (int b = 0; b < 256; ++b) U62B_CHECK(b % 16 < 16 && b / 16 < 16 && b % 16 + 16 * (b / 16) == b);
    for (int j = 0; j < 64; ++j) U62B_CHECK(j / 4 < 16 && (j % 4) * 2 < 8 && 4 * (j / 4) + j % 4 == j);
    for (int a = 0; a < 256; ++a) {
        const int c0 = a & 3, c1 = (a >> 2) & 3, c2 = (a >> 4) & 3, c3 = (a >> 6) & 3;
        const int p = c0 + 4 * c1 + 16 * c2 + 64 * c3;
        U62B_CHECK(p == a && p % 4 == c0 && p / 4 % 4 == c1 && p / 16 % 4 == c2 && p / 64 % 4 == c3);
    }
    // Q3_K scale / code / mask / group bounds.
    for (int lo = 0; lo < 256; ++lo) for (int t = 0; t < 256; t += 5) U62B_CHECK(lo % 16 + 16 * (t % 4) < 64);
    for (int c2 = 0; c2 < 4; ++c2) for (int m = 0; m <= 1; ++m) { const int c = c2 + 4 * m; U62B_CHECK(c < 8 && c - 4 >= -4 && c - 4 <= 3); }
    {
        std::vector<int> seen(8, 0), grp(16, 0);
        for (int h = 0; h < 2; ++h) for (int j = 0; j < 4; ++j) { seen[4 * h + j]++; for (int s = 0; s < 2; ++s) grp[8 * h + 2 * j + s]++; }
        for (int v : seen) U62B_CHECK(v == 1);
        for (int v : grp) U62B_CHECK(v == 1);
    }
    // get_scale_min_k4: both scale forms are 6-bit, low form = s % 64, high form = nibble + 16*(top two bits).
    {
        std::mt19937 rng(7);
        for (int it = 0; it < 2000; ++it) {
            uint8_t q[12]; for (auto& v : q) v = (uint8_t)rng();
            for (int j = 0; j < 8; ++j) {
                uint8_t d = 0, m = 0; scale_min_pair(j, q, &d, &m);
                U62B_CHECK(d < 64 && m < 64);
                if (j < 4) { U62B_CHECK(d == q[j] % 64 && m == q[j + 4] % 64); }
                else { U62B_CHECK(d == (q[j + 4] % 16) + 16 * (q[j - 4] / 64) && m == (q[j + 4] / 16) + 16 * (q[j] / 64)); }
            }
        }
    }
    // Q5_K masks, Q6_K code range and scale-group index, IQ4_XS scale index.
    {
        int u1[4], u2[4], acc = 0;
        for (int j = 0; j < 4; ++j) { u1[j] = 1 << (2 * j); u2[j] = 2 << (2 * j); acc += u1[j] + u2[j]; }
        U62B_CHECK(u1[0] == 1 && u1[1] == 4 && u1[2] == 16 && u1[3] == 64 && u2[0] == 2 && u2[3] == 128 && acc == 255);
        for (int n = 0; n < 16; ++n) for (int b = 0; b <= 1; ++b) U62B_CHECK(n + 16 * b < 32);
        for (int n = 0; n < 16; ++n) for (int h = 0; h < 4; ++h) { const int c = n + 16 * h; U62B_CHECK(c < 64 && c - 32 >= -32 && c - 32 <= 31 && c - 32 == (n + 16 * h) - 32); }
        for (int h = 0; h < 2; ++h) for (int q = 0; q < 4; ++q) for (int l = 0; l < 32; ++l) {
            const int e = 128 * h + 32 * q + l; U62B_CHECK(e < 256 && e / 16 == 8 * h + 2 * q + l / 16);
        }
        for (int ib = 0; ib < 8; ++ib) U62B_CHECK(ib / 2 < 4 && 2 * ib + 1 < 16);
    }
    // IQ4 codebook: 16 entries, strictly increasing, not affine, sign split; matches the dequantiser's table.
    {
        const int8_t* kv = baloon::artifact::kvalues_iq4nl;
        const int expect[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
        for (int i = 0; i < 16; ++i) U62B_CHECK(kv[i] == expect[i]);
        for (int i = 0; i < 15; ++i) U62B_CHECK(kv[i] < kv[i + 1]);
        U62B_CHECK(kv[1] - kv[0] != kv[2] - kv[1]);
        for (int i = 0; i < 8; ++i) U62B_CHECK(kv[i] < 0 && kv[i + 8] > 0);
    }
    // Full-run bookkeeping.
    for (uint64_t idx = 0; idx < 1000; ++idx) for (uint64_t n = 1; n < 9; ++n) U62B_CHECK(idx % n < n);
    for (uint64_t chunk : {0ull, 1ull, 17ull, 4096ull, 1000003ull}) for (uint64_t bb : {18ull, 22ull, 34ull, 110ull, 210ull}) U62B_CHECK(chunk / bb * bb <= chunk);
    // bf16 widening: shifting 16 bits left is exact and invertible.
    for (uint32_t u = 0; u < 65536; u += 7) U62B_CHECK((u << 16) % 65536 == 0 && (u << 16) >> 16 == u);

    // Behavioural: canonical encode/decode of random blocks equals the reference dequantiser bit for bit (all 10 types).
    {
        std::mt19937 rng(12345);
        for (const Row& r : rows) {
            int bad = 0;
            for (int it = 0; it < 200; ++it) {
                uint8_t b[256]; for (auto& v : b) v = (uint8_t)rng();
                float ref[256] = {0}, got[256] = {0};
                Block cb;
                const bool e = encode_block(r.t, b, &cb);
                const bool rd = reference_decode(r.t, b, ref);
                const bool dc = e && decode_block(r.t, cb, got);
                if (!e || !rd || !dc) { ++bad; continue; }
                for (unsigned i = 0; i < r.be; ++i) {
                    uint32_t a, c; std::memcpy(&a, &ref[i], 4); std::memcpy(&c, &got[i], 4);
                    if (a != c && !(std::isnan(ref[i]) && std::isnan(got[i]))) { ++bad; break; }
                }
            }
            U62B_CHECK(bad == 0);
        }
        // Q2_0 raw <-> canonical is lossless on the block bytes.
        int bad = 0;
        for (int it = 0; it < 500; ++it) {
            uint8_t b[18]; for (auto& v : b) v = (uint8_t)rng();
            b[1] = (uint8_t)((b[1] & 0x3F) | 0x00);   // finite fp16 scale (exponent bits < 16)
            Block cb; encode_q2(b, &cb);
            uint8_t o[18]; decode_q2(cb, o);
            if (std::memcmp(b, o, 18) != 0) ++bad;
        }
        U62B_CHECK(bad == 0);
    }
}

// ---- GgufWriterQ2Contract ---------------------------------------------------------------------------------------------
namespace {
struct Cur { const std::vector<uint8_t>& b; size_t p = 0;
    uint32_t u32() { uint32_t v = 0; for (int i = 0; i < 4; ++i) v |= (uint32_t)b[p + i] << (8 * i); p += 4; return v; }
    uint64_t u64() { uint64_t v = 0; for (int i = 0; i < 8; ++i) v |= (uint64_t)b[p + i] << (8 * i); p += 8; return v; }
    std::string str() { uint64_t n = u64(); std::string s((const char*)&b[p], (size_t)n); p += (size_t)n; return s; } };
}  // namespace

inline void gguf_writer() {
    using namespace baloon;
    U62B_CHECK(gguf::kQ2_0 == 42);
    // Clamp / code / symbol ranges (exhaustive over a wide window).
    for (long s = -50; s <= 50; ++s) {
        const int c = q2w::code_of_symbol(s);
        U62B_CHECK(c >= 0 && c <= 3);
        U62B_CHECK(q2w::symbol_of_code(c) >= -1 && q2w::symbol_of_code(c) <= 2);
        if (s >= -1 && s <= 1) U62B_CHECK(c == s + 1 && c <= 2);   // amax scale never produces the +2 level
    }
    // Positions and padding.
    for (size_t j = 0; j < 64; ++j) U62B_CHECK(q2w::byte_index(j) < 16 && q2w::bit_offset(j) < 8 && 4 * q2w::byte_index(j) + q2w::bit_offset(j) / 2 == j);
    for (uint64_t n = 0; n < 2000; ++n) {
        const uint64_t p = q2w::pad(32, n);
        U62B_CHECK(p < 32 && (n + p) % 32 == 0 && n + p == (n + 31) / 32 * 32);
    }
    // Quantise random blocks: amax scale => never code 3; pack/unpack round trip.
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.f, 1.f);
    int plus2 = 0, rt_bad = 0;
    for (int it = 0; it < 1000; ++it) {
        float w[64]; for (auto& v : w) v = nd(rng);
        uint8_t blk[18]; q2w::quantize_block(w, blk);
        uint8_t codes[64]; q2w::unpack_codes(blk, codes);
        for (int j = 0; j < 64; ++j) if (codes[j] > 2) ++plus2;
        uint8_t re[18]; q2w::pack_block((uint16_t)(blk[0] | (blk[1] << 8)), codes, re);
        if (std::memcmp(blk, re, 18) != 0) ++rt_bad;
    }
    U62B_CHECK(plus2 == 0 && rt_bad == 0);
    // Writer: parse the produced file and check the offset recurrence next = alignUp32(o + len), aligned data start, sizes.
    const std::vector<size_t> nblocks = {1, 3, 2, 5};   // tensors of odd byte lengths (multiples of 18)
    std::vector<q2w::TensorSpec> ts;
    for (size_t i = 0; i < nblocks.size(); ++i) {
        q2w::TensorSpec t; t.name = "t" + std::to_string(i); t.shape = {64 * nblocks[i]}; t.type = gguf::kQ2_0;
        t.data.assign(nblocks[i] * 18, (uint8_t)(0xA0 + i)); ts.push_back(t);
    }
    std::vector<q2w::Kv> kvs = {{"general.name", std::string("u62b")}, {"x.n", (uint32_t)7}, {"x.f", 1.5f}, {"x.b", true}, {"x.u", (uint64_t)9}};
    std::vector<uint8_t> out; std::string err;
    U62B_CHECK(q2w::write_file(kvs, ts, &out, &err));
    {
        Cur c{out};
        U62B_CHECK(c.u32() == gguf::kMagic && c.u32() == 3);
        const uint64_t nt = c.u64(), nkv = c.u64();
        U62B_CHECK(nt == ts.size() && nkv == kvs.size());
        for (uint64_t k = 0; k < nkv; ++k) {
            c.str(); const uint32_t ty = c.u32();
            if (ty == 8) c.str(); else if (ty == 4 || ty == 6) c.u32(); else if (ty == 10) c.u64(); else if (ty == 7) c.p += 1; else U62B_CHECK(false);
        }
        std::vector<uint64_t> offs;
        for (uint64_t i = 0; i < nt; ++i) {
            c.str(); const uint32_t nd2 = c.u32(); for (uint32_t d = 0; d < nd2; ++d) c.u64();
            U62B_CHECK(c.u32() == 42); offs.push_back(c.u64());
        }
        const size_t data_start = c.p + q2w::pad(32, c.p);
        U62B_CHECK(data_start % 32 == 0);
        uint64_t exp = 0;
        for (size_t i = 0; i < ts.size(); ++i) {
            U62B_CHECK(offs[i] == exp && offs[i] % 32 == 0);
            const uint64_t len = ts[i].data.size();
            U62B_CHECK(std::memcmp(&out[data_start + offs[i]], ts[i].data.data(), len) == 0);
            exp = (exp + len + 31) / 32 * 32;           // nextOffset
            U62B_CHECK(exp >= offs[i] + len && exp < offs[i] + len + 32);
        }
        U62B_CHECK(out.size() == data_start + exp);
    }
    // Rejections: wrong byte count, zero alignment.
    { auto bad = ts; bad[0].data.pop_back(); std::vector<uint8_t> o; U62B_CHECK(!q2w::write_file(kvs, bad, &o, &err)); }
    { std::vector<uint8_t> o; U62B_CHECK(!q2w::write_file(kvs, ts, &o, &err, 0)); }
    // Fused expert tensor [64,256,16]: 4096 rows of 64.
    U62B_CHECK(64ull * (256 * 16) == 64ull * 4096);
}

// ---- IqNativePackIndex ------------------------------------------------------------------------------------------------
inline void iq_pack() {
    using namespace baloon::iqpack;
    for (uint64_t n = 0; n < 1000; ++n) {
        U62B_CHECK(pad64(n) < 64 && (n + pad64(n)) % 64 == 0);
        U62B_CHECK(round_row(n) % 64 == 0 && n <= round_row(n) && round_row(n) < n + 64 && round_row(n) == n + pad64(n));
    }
    // Layer offsets: succ recurrence, monotonic, cover.
    std::vector<uint64_t> blob = {1382400, 1000, 4096, 777, 123456};
    for (size_t l = 0; l + 1 <= blob.size(); ++l) U62B_CHECK(experts_offset(blob, l + 1) == experts_offset(blob, l) + blob[l] * 512);
    for (size_t l = 0; l < blob.size(); ++l) for (size_t k = 0; l + k <= blob.size(); ++k) U62B_CHECK(experts_offset(blob, l) <= experts_offset(blob, l + k));
    U62B_CHECK(experts_offset(blob, 0) == 0);
    { const uint64_t g = 512 * 7, u = 512 * 9, d = 512 * 11; U62B_CHECK((g / 512 + u / 512 + d / 512) * 512 == g + u + d); }
    // Layer count.
    { std::vector<uint64_t> ls = {0, 5, 2, 47, 9};
      const uint64_t n = layer_count(ls); U62B_CHECK(n == 48); for (auto l : ls) U62B_CHECK(l < n); U62B_CHECK(layer_count({}) == 0); }
    // Kinds.
    U62B_CHECK(kind_of("BF16") == 4 && kind_of("F16") == 5 && kind_of("F32") == 2);
    U62B_CHECK(kind_of("BF16") != kind_of("F16") && kind_of("F16") != kind_of("F32") && kind_of("BF16") != kind_of("F32"));
    // Router narrowing: exact only when the low 16 bits of every word are zero.
    {
        std::vector<uint32_t> ok = {0x3F800000u, 0xC0000000u, 0u, 0x7F7F0000u};
        std::vector<uint16_t> out;
        U62B_CHECK(narrow_router_f32_to_bf16(ok.data(), ok.size(), &out));
        for (size_t i = 0; i < ok.size(); ++i) U62B_CHECK((uint32_t)out[i] * 65536u == ok[i]);
        std::vector<uint32_t> lossy = {0x3F800000u, 0x3F800001u};
        std::vector<uint16_t> keep = {42};
        U62B_CHECK(!narrow_router_f32_to_bf16(lossy.data(), lossy.size(), &keep) && keep.size() == 1 && keep[0] == 42);
        for (uint32_t low = 1; low < 65536; low += 251) { uint32_t w = 0x3F800000u | low; U62B_CHECK(!narrow_router_f32_to_bf16(&w, 1, &keep)); }
    }
    // Extra offsets.
    { std::vector<uint64_t> l = {1, 63, 64, 65, 1000}; U62B_CHECK(extra_offset({}) == 0);
      std::vector<uint64_t> pre;
      for (uint64_t b : l) { U62B_CHECK((b + pad64(b)) % 64 == 0 && b <= b + pad64(b)); pre.push_back(b); U62B_CHECK(extra_offset(pre) % 64 == 0); } }
}

// ---- MtpPackQuantizers ------------------------------------------------------------------------------------------------
inline void mtp_pack() {
    using namespace baloon::mtpq;
    for (int i = 0; i <= 16; ++i) { U62B_CHECK(16 + i >= 16 && 16 + i <= 32); U62B_CHECK(std::fabs(scale_factor(i) - (16 + i) / 32.0f) < 1e-9f); }
    U62B_CHECK(scale_factor(0) == 0.5f && scale_factor(16) == 1.0f && kScaleCandidates == 17);
    for (long q = -100; q <= 100; ++q) { const long c = clip12(q) + 1; U62B_CHECK(c >= 0 && c <= 3); }
    for (unsigned t = 0; t < 100; ++t) U62B_CHECK(std::min(15u, t) <= 15);
    for (int lo = 0; lo < 16; ++lo) for (int hi = 0; hi < 16; ++hi) { const int p = lo + 16 * hi; U62B_CHECK(p < 256 && p % 16 == lo && p / 16 == hi); }
    for (int c = 0; c < 16; ++c) U62B_CHECK(c - 8 >= -8 && c - 8 <= 7);
    // Byte budgets of the fused expert tensors.
    const uint64_t gu = 512ull * 1280 * 2560, dn = 512ull * 2560 * 640;
    U62B_CHECK(gu + dn == 2516582400ull);
    U62B_CHECK(2560 % 64 == 0 && 640 % 64 == 0);
    U62B_CHECK(gu / 64 * 18 + dn / 64 * 18 == 707788800ull);
    U62B_CHECK(gu / 32 * 18 + dn / 32 * 18 == 1415577600ull);
    U62B_CHECK(gu / 32 * 34 + dn / 32 * 34 == 2673868800ull);
    U62B_CHECK(707788800ull < 1415577600ull && 1415577600ull < 2673868800ull);
    U62B_CHECK(2560 / 64 * 18 == 720 && 640 / 64 * 18 == 180);
    U62B_CHECK(total_bytes(64, 18) == 707788800ull && total_bytes(32, 18) == 1415577600ull);
    // Behavioural: Q4_0 packs element i in the low nibble and element i+16 in the high nibble; Q2_0 codes are in 0..3.
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int it = 0; it < 300; ++it) {
        float x[64]; for (auto& v : x) v = nd(rng);
        uint8_t b2[18]; quantize_q2(x, b2);
        uint8_t codes[64]; baloon::q2w::unpack_codes(b2, codes);
        for (int j = 0; j < 64; ++j) U62B_CHECK(codes[j] <= 3);
        uint8_t b4[18]; quantize_q4(x, b4);
        float y[32]; dequantize_q4(b4, y);
        const float d = baloon::f16::half_to_float((uint16_t)(b4[0] | (b4[1] << 8)));
        for (int j = 0; j < 16; ++j) {
            U62B_CHECK(y[j] == d * (float)((int)(b4[2 + j] & 0xF) - 8));
            U62B_CHECK(y[j + 16] == d * (float)((int)(b4[2 + j] >> 4) - 8));
        }
    }
}

// ---- MtpRuntimeBlobLayout + ExpertArenaBlobLayout ---------------------------------------------------------------------
inline void blob_layouts() {
    using namespace baloon;
    U62B_CHECK(mtprt::kBlob == 1382400 && mtprt::kExperts * mtprt::kBlob == 707788800ull);
    U62B_CHECK(mtprt::kGuCodes == 819200 && mtprt::kDownCodes == 409600 && mtprt::kGuScales == 102400 && mtprt::kDownScales == 51200);
    U62B_CHECK(mtprt::kGuCodes + mtprt::kDownCodes + mtprt::kGuScales + mtprt::kDownScales == mtprt::kBlob);
    U62B_CHECK(mtprt::kHidden / 64 * 18 == 720 && mtprt::kFF / 64 * 18 == 180);
    for (uint64_t r = 0; r < 640; ++r) {
        U62B_CHECK(mtprt::gate_slot(r) % 2 == 0 && mtprt::gate_slot(r) / 2 == r && mtprt::up_slot(r) % 2 == 1 && mtprt::up_slot(r) / 2 == r);
        U62B_CHECK(mtprt::gate_slot(r) < 1280 && mtprt::up_slot(r) < 1280 && mtprt::gate_slot(r) != mtprt::up_slot(r));
        U62B_CHECK(baloon::iq_layout::gu_row(r, 0) == mtprt::gate_slot(r) && baloon::iq_layout::gu_row(r, 1) == mtprt::up_slot(r));
    }
    { std::vector<int> seen(1280, 0); for (uint64_t r = 0; r < 640; ++r) { seen[mtprt::gate_slot(r)]++; seen[mtprt::up_slot(r)]++; } for (int v : seen) U62B_CHECK(v == 1); }
    for (uint64_t s = 0; s < 1280; s += 3) for (uint64_t b = 0; b < 40; b += 3) {
        U62B_CHECK(s * 640 + b * 16 + 16 <= 819200);
        U62B_CHECK((s * 40 + b) * 2 + 2 <= 102400);
    }
    for (uint64_t n = 0; n < 2000; ++n) U62B_CHECK(mtprt::pad256(n) < 256 && (n + mtprt::pad256(n)) % 256 == 0);

    // Arena (pack layout) constants and plane offsets.
    using namespace pack;
    U62B_CHECK(kBlobBytes == 1382400 && kRoleBytes == 460800);
    U62B_CHECK(kGuCodesOff == 0 && kDCodesOff == 819200 && kGuScalesOff == 1228800 && kDScalesOff == 1331200);
    U62B_CHECK(kDScalesOff + kDScales == kBlobBytes);
    U62B_CHECK(kGuCodesOff < kDCodesOff && kDCodesOff < kGuScalesOff && kGuScalesOff < kDScalesOff);
    U62B_CHECK(2 * 640 * 640 + 2560 * 160 + 2 * 640 * 40 * 2 + 2560 * 10 * 2 == 1382400);
    for (uint64_t slot = 0; slot < 1280; slot += 7) for (uint64_t b = 0; b < 40; b += 3) {
        U62B_CHECK(kGuCodesOff + slot * 640 + b * 16 + 16 <= kDCodesOff);
        U62B_CHECK(kGuScalesOff + (slot * 40 + b) * 2 + 2 <= kDScalesOff);
    }
    for (uint64_t row = 0; row < 2560; row += 13) for (uint64_t b = 0; b < 10; ++b) U62B_CHECK(kDCodesOff + row * 160 + b * 16 + 16 <= kGuScalesOff);
    for (uint64_t e = 0; e < 512; ++e) U62B_CHECK(e * 460800 + 460800 == (e + 1) * 460800 && e * 460800 + 460800 <= 512ull * 460800);
    U62B_CHECK(expert_blob_offset(0, 0) == 0 && expert_blob_offset(1, 0) == 707788800ull && expert_blob_offset(47, 511) + kBlobBytes == 33973862400ull);
    for (uint64_t n = 0; n < 60; ++n) U62B_CHECK(n * 512 * kBlobBytes + 707788800ull == (n + 1) * 512 * kBlobBytes);
    U62B_CHECK(48ull * 512 * kBlobBytes == 33973862400ull && 48 * 512 == 24576);

    // Behavioural round trip of the blob builder/verifier on a pseudo-random expert (gate/up interleave is a bijection).
    std::mt19937 rng(2024);
    std::vector<uint8_t> g(kRoleBytes), u(kRoleBytes), d(kRoleBytes);
    for (auto* v : {&g, &u, &d}) for (auto& x : *v) x = (uint8_t)rng();
    std::vector<uint8_t> blob;
    U62B_CHECK(build_expert_blob(g, u, d, &blob) && blob.size() == kBlobBytes);
    U62B_CHECK(verify_expert_blob(blob, g, u, d));
    // A single flipped byte anywhere must be detected, and the verifier must not accept swapped gate/up.
    { auto g2 = g; g2[12345] ^= 1; U62B_CHECK(!verify_expert_blob(blob, g2, u, d)); }
    U62B_CHECK(!verify_expert_blob(blob, u, g, d));
    // Gate row r lands in interleaved slot 2r: its scale bytes sit at the slot's position in the gate/up scale plane.
    for (uint64_t r : {0ull, 1ull, 639ull}) {
        const uint64_t blk = r * 40;                         // first block of row r in the role
        U62B_CHECK(std::memcmp(blob.data() + kGuScalesOff + (2 * r * 40) * 2, g.data() + blk * 18, 2) == 0);
        U62B_CHECK(std::memcmp(blob.data() + kGuScalesOff + ((2 * r + 1) * 40) * 2, u.data() + blk * 18, 2) == 0);
        U62B_CHECK(std::memcmp(blob.data() + kGuCodesOff + (2 * r) * 640, g.data() + blk * 18 + 2, 16) == 0);
    }
}

// ---- RefPackBuilderVerifier -------------------------------------------------------------------------------------------
inline void pack_builder() {
    using namespace baloon::pack;
    U62B_CHECK(per_byte(2) == 4 && per_byte(4) == 2 && per_byte(8) == 1);
    for (unsigned bits : {2u, 4u, 8u}) U62B_CHECK(valid_bits(bits) && 8 / bits * bits == 8);
    U62B_CHECK(!valid_bits(3) && !valid_bits(1) && !valid_bits(16));
    std::mt19937 rng(77);
    for (unsigned bits : {2u, 4u, 8u}) {
        const size_t n = 64;
        std::vector<uint8_t> codes(n); for (auto& c : codes) c = (uint8_t)(rng() & ((1u << bits) - 1));
        auto p = pack_codes(codes.data(), n, bits);
        U62B_CHECK(p.has_value() && p->size() == n / per_byte(bits));
        if (!p) continue;
        auto u = unpack_codes(p->data(), n, bits);
        U62B_CHECK(u.has_value() && *u == codes);
        // LSB first: code 0 occupies the low bits of byte 0.
        U62B_CHECK(((*p)[0] & ((1u << bits) - 1)) == codes[0]);
        // Counts that do not fill whole bytes are refused (2-bit count multiple of 4, 4-bit multiple of 2).
        if (bits < 8) U62B_CHECK(!pack_codes(codes.data(), n - 1, bits).has_value());
    }
    for (int c0 = 0; c0 < 16; ++c0) for (int c1 = 0; c1 < 16; ++c1) { const int p = c0 + 16 * c1; U62B_CHECK(p < 256 && p % 16 == c0 && p / 16 == c1); }
    // Plane alignment inside a segment.
    for (uint64_t n = 0; n < 1000; ++n) { U62B_CHECK(pad64(n) < 64 && (n + pad64(n)) % 64 == 0); }
    {
        PlaneWriter w; std::vector<uint8_t> a(5, 1), b(70, 2), c(1, 3);
        const size_t oa = w.append_plane(a), ob = w.append_plane(b), oc = w.append_plane(c);
        U62B_CHECK(oa == 0 && ob == 64 && oc == 192 && w.bytes().size() == 193);   // last plane is not padded
        U62B_CHECK(ob % 64 == 0 && oc % 64 == 0 && ob >= oa + a.size() && oc >= ob + b.size());
        U62B_CHECK(w.bytes()[ob] == 2 && w.bytes()[oc] == 3 && w.bytes()[oa + 4] == 1);
    }
    // Scale precision: fp16 halves the plane; fp16 is chosen only when the round trip is exact.
    U62B_CHECK(scale_bytes(true) * 2 == scale_bytes(false) && scale_bytes(true) >= 2 && scale_bytes(false) >= 2);
    {
        const float exact[3] = {1.0f, 0.5f, -2.0f};
        const float inexact[2] = {1.0f, 0.1f};
        U62B_CHECK(choose_scale_dtype_fp16(exact, 3));
        U62B_CHECK(!choose_scale_dtype_fp16(inexact, 2));
        bool fp16 = false; auto e = encode_scales(exact, 3, &fp16);
        U62B_CHECK(fp16 && e.size() == 6);
        auto e2 = encode_scales(inexact, 2, &fp16);
        U62B_CHECK(!fp16 && e2.size() == 8);
    }
    // Arena size and verification sampling counts.
    U62B_CHECK(48ull * kExpertsPerLayer * kBlobBytes == 33973862400ull);
    U62B_CHECK(experts_checked(512, 32) == 16 && role_checks(512, 32) == 48 && 48 * role_checks(512, 32) == 2304);
    U62B_CHECK(experts_checked(512, 1) == 512);
    for (uint64_t every = 1; every <= 512; ++every) U62B_CHECK(experts_checked(512, every) > 0);
    // Partition of tensor names: every index lands in exactly one bucket.
    {
        std::vector<std::string> names = {"model.layers.0.mlp.experts.gate_up_proj", "model.embed_tokens.weight", "model.norm.weight", "blk.3.ffn_gate_exps.weight", "token_embd.weight"};
        auto part = partition_tensors(names);
        U62B_CHECK(part.dense.size() + part.embd.size() + part.expert.size() == names.size());
        U62B_CHECK(part.expert.size() == 2 && part.embd.size() == 2 && part.dense.size() == 1);
    }
}

inline int run_all() {
    canonical(); gguf_writer(); iq_pack(); mtp_pack(); blob_layouts(); pack_builder();
    std::printf("u62b host checks: %d checks, %d failures\n", g_checks, g_fail);
    return g_fail;
}

}  // namespace u62b


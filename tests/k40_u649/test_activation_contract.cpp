// Unit u649794726a: ActivationContract invariants on the real core code (src/core/layer.cpp):
//  - sform_of: native > 2-bit (Q8_0) > wants_q8k (Q8_K) > Q8_0, checked over the full (native, bits, wants) domain
//  - plane_ptrs: codes|scales|offset tile the tensor, validation rejects every broken form
//  - embed_row / embed_code: bit-field decode vs an independent packing oracle for 2/4/8 bits
#include <cstdint>
#include <cstdio>
#include <vector>
#include "baloon/core_layer.hpp"
using namespace baloon::core;
static int fails = 0, checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static WeightRef mk(uint64_t c, uint64_t s, uint64_t o, int bits, bool native = false, bool q8k = false) {
    WeightRef w; w.codes = c; w.scales = s; w.offset = o; w.bytes = c + s + o; w.code_bits = bits;
    w.has_offset = o != 0; w.native = native; w.wants_q8k = q8k; return w;
}

int main() {
    // ---- activation dispatch: full domain -------------------------------------------------------------------------------
    for (int bits : {0, 1, 2, 3, 4, 8}) for (int nat = 0; nat < 2; ++nat) for (int q = 0; q < 2; ++q) {
        WeightRef w = mk(16, 4, 0, bits, nat, q);
        Gemv exp = nat ? Gemv::Native : bits == 2 ? Gemv::S2Q80 : q ? Gemv::Q8k : Gemv::LegacyQ80;
        CHECK(sform_of(w) == exp);
        if (!nat) CHECK(sform_of(w) != Gemv::Native);
        if (!nat && bits == 2) CHECK(sform_of(w) == Gemv::S2Q80);          // S2 ignores wants_q8k
    }
    // gemv_quantized runs exactly the chosen executor with the validated planes
    {
        int hit[4] = {0, 0, 0, 0}; uint64_t seen_scales = 0;
        GemvExec ex;
        ex.native = [&](const Planes&) { ++hit[0]; };
        ex.s2q80 = [&](const Planes& p) { ++hit[1]; seen_scales = p.scales_start; };
        ex.legacy_q80 = [&](const Planes&) { ++hit[2]; };
        ex.q8k = [&](const Planes&) { ++hit[3]; };
        Gemv ch = Gemv::Native;
        CHECK(gemv_quantized(mk(32, 8, 4, 2), ex, &ch) && ch == Gemv::S2Q80 && hit[1] == 1 && seen_scales == 32);
        CHECK(gemv_quantized(mk(32, 8, 0, 4, false, true), ex, &ch) && ch == Gemv::Q8k && hit[3] == 1);
        CHECK(gemv_quantized(mk(32, 8, 0, 8), ex, &ch) && ch == Gemv::LegacyQ80 && hit[2] == 1);
        CHECK(gemv_quantized(mk(32, 8, 0, 4, true), ex, &ch) && ch == Gemv::Native && hit[0] == 1);
        WeightRef bad = mk(32, 8, 0, 4); bad.bytes += 4;                     // doubled/incorrect scale plane
        CHECK(!gemv_quantized(bad, ex) && hit[0] + hit[1] + hit[2] + hit[3] == 4);
    }

    // ---- planes tile the tensor -----------------------------------------------------------------------------------------
    for (int bits : {2, 4, 8}) for (uint64_t c : {1ull, 7ull, 4096ull}) for (uint64_t s : {1ull, 2ull, 513ull}) for (uint64_t o : {0ull, 3ull}) {
        Planes p = plane_ptrs(mk(c, s, o, bits));
        CHECK(p.ok && p.codes_start == 0 && p.scales_start == c && p.offset_start == c + s);
        CHECK(p.offset_start + p.offset == c + s + o);                        // nothing past the end
        CHECK(p.codes == c && p.scales == s && p.offset == o);
    }
    CHECK(!plane_ptrs(mk(8, 2, 0, 3)).ok);                                    // illegal width
    CHECK(!plane_ptrs(mk(8, 2, 0, 0)).ok && !plane_ptrs(mk(8, 2, 0, 16)).ok);
    CHECK(!plane_ptrs(mk(0, 2, 0, 4)).ok);                                    // empty codes
    CHECK(!plane_ptrs(mk(8, 0, 0, 4)).ok);                                    // empty scales
    { WeightRef w = mk(8, 2, 1, 4); w.bytes -= 1; CHECK(!plane_ptrs(w).ok); }    // sum short
    { WeightRef w = mk(8, 2, 1, 4); w.bytes += 1; CHECK(!plane_ptrs(w).ok); }    // sum long
    { WeightRef w = mk(8, 4, 0, 4); w.bytes = 8 + 8; CHECK(!plane_ptrs(w).ok); } // scales read as fp32 instead of fp16
    { WeightRef w = mk(8, 2, 3, 4); w.has_offset = false; CHECK(!plane_ptrs(w).ok); }
    { WeightRef w = mk(8, 2, 0, 4); w.has_offset = true;  CHECK(!plane_ptrs(w).ok); }

    // ---- embedding bit-field decode -------------------------------------------------------------------------------------
    CHECK(embed_code_per_byte(2) == 4 && embed_code_per_byte(4) == 2 && embed_code_per_byte(8) == 1);
    CHECK(embed_code_per_byte(3) == 0 && embed_code_per_byte(0) == 0 && embed_code_per_byte(16) == 0);
    uint32_t rng = 12345;
    auto rnd = [&]() { rng = rng * 1664525u + 1013904223u; return rng >> 8; };
    for (int bits : {2, 4, 8}) {
        const int per = 8 / bits; const int64_t vocab = 7, n = 24;               // 24 divisible by 4, 2, 1
        std::vector<std::vector<uint32_t>> want((size_t)vocab, std::vector<uint32_t>((size_t)n));
        std::vector<uint8_t> table((size_t)(vocab * n / per), 0);
        for (int64_t t = 0; t < vocab; ++t) for (int64_t i = 0; i < n; ++i) {   // oracle packer: independent bit arithmetic
            uint32_t v = rnd() & ((1u << bits) - 1u); want[(size_t)t][(size_t)i] = v;
            int64_t bit = (t * n + i) * bits;                                    // global bit position, little-endian within byte
            for (int b = 0; b < bits; ++b) if ((v >> b) & 1u) table[(size_t)((bit + b) / 8)] |= (uint8_t)(1u << ((bit + b) % 8));
        }
        for (int64_t t = 0; t < vocab; ++t) {
            std::vector<uint32_t> got((size_t)n, 0xDEADBEEFu);
            CHECK(embed_row(table.data(), bits, vocab, n, t, got.data()));
            CHECK(got == want[(size_t)t]);
            for (int64_t i = 0; i < n; ++i) CHECK(got[(size_t)i] < (1u << bits));
        }
        // field never straddles bytes: decoding a single byte touches only that byte
        for (int v = 0; v < 256; ++v) {
            uint8_t byte = (uint8_t)v; uint32_t rebuilt = 0, c;
            for (int i = 0; i < per; ++i) { CHECK(embed_code(&byte, bits, i, &c)); rebuilt |= c << (i * bits); }
            CHECK(rebuilt == (uint32_t)v);
        }
        std::vector<uint32_t> o((size_t)n);
        CHECK(!embed_row(table.data(), bits, vocab, n, vocab, o.data()) && !embed_row(table.data(), bits, vocab, n, -1, o.data()));
        CHECK(!embed_row(nullptr, bits, vocab, n, 0, o.data()) && !embed_row(table.data(), bits, vocab, 0, 0, o.data()));
        if (per > 1) CHECK(!embed_row(table.data(), bits, vocab, n + 1, 0, o.data()));   // row length not a multiple of perbyte
    }
    { uint8_t b = 0xE4; uint32_t c;                                           // 0b11100100: 2-bit fields low-first = 0,1,2,3
      for (int i = 0; i < 4; ++i) { CHECK(embed_code(&b, 2, i, &c) && c == (uint32_t)i); }
      CHECK(embed_code(&b, 4, 0, &c) && c == 0x4 && embed_code(&b, 4, 1, &c) && c == 0xE);
      CHECK(!embed_code(&b, 5, 0, &c) && !embed_code(&b, 4, -1, &c)); }
    { uint32_t dummy[4]; uint8_t tb[2] = {0, 0}; CHECK(!embed_row(tb, 3, 1, 4, 0, dummy)); }
    // native FP32 table path still addresses rows exactly
    { std::vector<float> tb(12); for (size_t i = 0; i < tb.size(); ++i) tb[i] = (float)i; float r[4];
      CHECK(embed_row(tb.data(), 3, 4, 2, r) && r[0] == 8 && r[3] == 11 && !embed_row(tb.data(), 3, 4, 3, r)); }

    std::printf("%d checks, %d failed\n", checks, fails);
    if (fails == 0) std::printf("U649 ACTIVATION CONTRACT TESTS PASSED\n");
    return fails ? 1 : 0;
}

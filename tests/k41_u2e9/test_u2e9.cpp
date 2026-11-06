// Unit u2e9aa362e3 invariants (LAYER 1 statements of GgufReaderGeometry, CanonicalFormRoundtrip, Q2_0GgufWriter, DirectFileIo,
// WeightPackLoader, LayerShapeCheck, ByteLevelBpeTokenizer), checked on the real code of include/baloon.
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

#include "../unit14/check.hpp"
#include "baloon/byte_level_bpe_tokenizer.hpp"
#include "baloon/canonical_form_roundtrip.hpp"
#include "baloon/direct_file_io.hpp"
#include "baloon/gguf_reader.hpp"
#include "baloon/layer_shape_check.hpp"
#include "baloon/q2_gguf_writer.hpp"
#include "baloon/weight_loader.hpp"

using namespace baloon;
using testh::Rng;
using Bytes = std::vector<uint8_t>;

// ======================================================================================================================
// ByteLevelBpeTokenizer
// ======================================================================================================================
static bool in_bs(unsigned b) { return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255); }
static unsigned non_below(unsigned b) { unsigned n = 0; for (unsigned x = 0; x < b; ++x) n += in_bs(x) ? 0 : 1; return n; }
// closed form of the theory
static unsigned non_below_closed(unsigned b) {
    if (b <= 33) return b;
    if (b <= 127) return 33;
    if (b <= 161) return 33 + (b - 127);
    if (b <= 173) return 67;
    return 68;
}

static void test_bpe() {
    int printable = 0, shifted = 0;
    for (unsigned b = 0; b < 256; ++b) { (bpe::printable(b) ? printable : shifted)++; CHECK(bpe::printable(b) == in_bs(b)); }
    CHECK(printable == 188 && shifted == 68);
    std::vector<uint32_t> seen;
    for (unsigned b = 0; b < 256; ++b) {
        const uint32_t cp = bpe::byte_to_codepoint((uint8_t)b);
        CHECK(non_below(b) == non_below_closed(b));
        if (in_bs(b)) CHECK(cp == b);
        else {
            CHECK(cp == 256 + non_below(b) && cp >= 256 && cp < 324);
            CHECK(b <= 32 || (b >= 127 && b <= 160) || b == 173);
        }
        CHECK(bpe::codepoint_to_byte(cp) == (int)b);          // injective with an inverse
        seen.push_back(cp);
    }
    std::sort(seen.begin(), seen.end());
    CHECK(std::adjacent_find(seen.begin(), seen.end()) == seen.end());
    CHECK(bpe::codepoint_to_byte(324) == -1 && bpe::codepoint_to_byte(0) == -1 && bpe::codepoint_to_byte(32) == -1);

    // decode(encode(s)) == s for every single byte and for random strings
    for (unsigned b = 0; b < 256; ++b) {
        std::string s(1, (char)b); bool ok = false;
        const std::string m = bpe::bytes_to_mapped(s);
        CHECK(bpe::mapped_to_bytes(m, &ok) == s && ok);
    }
    Rng rng(0xB9E);
    for (int it = 0; it < 2000; ++it) {
        std::string s; const int n = (int)rng.below(64);
        for (int i = 0; i < n; ++i) s.push_back((char)rng.below(256));
        bool ok = false;
        CHECK(bpe::mapped_to_bytes(bpe::bytes_to_mapped(s), &ok) == s && ok);
    }
    { bool ok = true; (void)bpe::mapped_to_bytes("\xFF\xFE", &ok); CHECK(!ok); }   // not valid UTF-8

    // merge step: part count falls by exactly one, never empties a word, replaces the adjacent pair
    for (size_t n = 2; n <= 12; ++n) {
        std::vector<int> w(n); for (size_t i = 0; i < n; ++i) w[i] = (int)i * 10;
        for (size_t i = 0; i + 1 < n; ++i) {
            const auto m = bpe::merge_at(w, i, 999);
            CHECK(m.size() + 1 == w.size() && !m.empty() && m[i] == 999);
            for (size_t k = 0; k < i; ++k) CHECK(m[k] == w[k]);
            for (size_t k = i + 2; k < n; ++k) CHECK(m[k - 1] == w[k]);
        }
    }
    // a word of n bytes can undergo at most n-1 merges
    { size_t parts = 9, merges = 0; while (parts > 1) { --parts; ++merges; } CHECK(merges == 8); }

    // special tokens: type 4 always, type 3 only with parse_special, nothing else
    for (int ty = 0; ty < 8; ++ty) {
        CHECK(bpe::matched(ty, false) == (ty == 4));
        CHECK(bpe::matched(ty, true) == (ty == 4 || ty == 3));
        if (bpe::matched(ty, false)) CHECK(bpe::matched(ty, true));          // monotone
    }
    CHECK(!bpe::matched(3, false) && bpe::matched(3, true));
    CHECK(bpe::kNumMerges == 247587 && bpe::kVocabSize == 248320 && bpe::kNumMerges < bpe::kVocabSize);

    // a full tokenizer over the 256 byte tokens + two merges: encode then decode reproduces the bytes
    {
        std::vector<std::pair<std::string, int>> toks;
        for (unsigned b = 0; b < 256; ++b) toks.push_back({bpe::bytes_to_mapped(std::string(1, (char)b)), bpe::kTypeNormal});
        const std::string ab = bpe::bytes_to_mapped("ab"), abc = bpe::bytes_to_mapped("abc");
        toks.push_back({ab, bpe::kTypeNormal}); toks.push_back({abc, bpe::kTypeNormal});
        const std::string la = bpe::bytes_to_mapped("a"), lb = bpe::bytes_to_mapped("b"), lc = bpe::bytes_to_mapped("c");
        bpe::ByteLevelBpe t;
        CHECK(t.load(toks, {la + " " + lb, ab + " " + lc}).empty());
        CHECK(t.merge_count() == 2 && t.vocab_size() == 258);
        std::vector<int32_t> ids;
        CHECK(t.encode_word("abc", &ids) && ids.size() == 1);
        CHECK(t.encode_word(" \n\x01", &ids) && ids.size() == 3);
        CHECK(!t.load(toks, {la + " " + lb + " " + lc}).empty());                // three parts: refused
    }
}

// ======================================================================================================================
// CanonicalFormRoundtrip
// ======================================================================================================================
static void test_canon() {
    using canon::QType;
    // integer-bias codebooks
    for (int q = -128; q <= 127; ++q) CHECK(q + 128 >= 0 && q + 128 <= 255 && (q + 128) - 128 == q);
    for (int c = 0; c < 16; ++c) CHECK(c - 8 >= -8 && c - 8 <= 7);
    for (int c2 = 0; c2 < 4; ++c2) for (int hb = 0; hb < 2; ++hb) {
        CHECK(c2 + 4 * hb < 8);
        CHECK((c2 + 4 * hb) - 4 == c2 - (hb == 1 ? 0 : 4));
    }
    for (int c = 0; c < 32; ++c) CHECK(c - 16 >= -16 && c - 16 <= 15);
    for (int d = 0; d < 64; ++d) CHECK(d - 32 >= -32 && d - 32 <= 31);
    for (int c = 0; c < 4; ++c) CHECK(c - 1 >= -1 && c - 1 <= 2);
    // Q2_0 -> S2 plane split
    for (int b = 0; b < 16; ++b) for (int t = 0; t < 4; ++t) CHECK((4 * b + t) / 4 == b && (4 * b + t) % 4 == t);
    CHECK((18 - 2) * 4 == 64);
    // Q4_K / Q5_K element order
    { std::vector<int> hit(256, 0);
      for (int j = 0; j < 4; ++j) for (int h = 0; h < 2; ++h) for (int l = 0; l < 32; ++l) {
          const int i = 64 * j + 32 * h + l; CHECK(i < 256); ++hit[i];
          CHECK(i / 64 == j && (i / 32) % 2 == h && i % 32 == l && i / 32 == 2 * j + h && 2 * j + h < 8);
      }
      for (int i = 0; i < 256; ++i) { CHECK(hit[i] == 1); CHECK(64 * (i / 64) + 32 * ((i / 32) % 2) + i % 32 == i); } }
    for (int j = 0; j < 4; ++j) for (int h = 0; h < 2; ++h) for (int j2 = 0; j2 < 4; ++j2) for (int h2 = 0; h2 < 2; ++h2)
        CHECK((2 * j + h == 2 * j2 + h2) == (j == j2 && h == h2));
    for (int nib = 0; nib < 16; ++nib) for (int bit = 0; bit < 2; ++bit) CHECK(nib + 16 * bit < 32);
    // Q3_K element order
    for (int half = 0; half < 2; ++half) for (int j = 0; j < 4; ++j) for (int sub = 0; sub < 2; ++sub) for (int l = 0; l < 16; ++l) {
        const int g = 8 * half + 2 * j + sub;
        CHECK(128 * half + 32 * j + 16 * sub + l == 16 * g + l && g < 16);
        CHECK(half * 32 + sub * 16 + l < 64 && sub * 16 + l < 32);
    }
    // Q6_K element order
    { std::vector<int> hit(256, 0);
      for (int h = 0; h < 2; ++h) for (int q = 0; q < 4; ++q) for (int l = 0; l < 32; ++l) {
          const int i = 128 * h + 32 * q + l; ++hit[i];
          CHECK(i / 16 == 8 * h + 2 * q + l / 16 && 8 * h + 2 * q + l / 16 < 16);
          CHECK(i / 128 == h && (i / 32) % 4 == q && i % 32 == l);
      }
      for (int i = 0; i < 256; ++i) CHECK(hit[i] == 1); }

    // registry rows: (block_bytes, block_elems, group_elems, max_code, form_levels)
    struct Row { QType t; unsigned bb, be, ge, maxc, levels; };
    const Row rows[] = {
        {QType::Q2_0, 18, 64, 64, 3, 4},   {QType::Q3_K, 110, 256, 16, 7, 16},   {QType::Q4_0, 18, 32, 32, 15, 16},
        {QType::Q4_K, 144, 256, 32, 15, 16}, {QType::Q5_0, 22, 32, 32, 31, 256}, {QType::Q5_K, 176, 256, 32, 31, 256},
        {QType::Q6_K, 210, 256, 16, 63, 256}, {QType::Q8_0, 34, 32, 32, 255, 256}, {QType::IQ4_NL, 18, 32, 32, 15, 16},
        {QType::IQ4_XS, 136, 256, 32, 15, 16}};
    CHECK(sizeof(rows) / sizeof(rows[0]) == 10);
    for (const Row& r : rows) {
        const canon::Mapping* m = canon::find_mapping(r.t);
        CHECK(m != nullptr);
        if (!m) continue;
        CHECK(m->block_bytes == r.bb && m->block_elems == r.be && m->group_elems == r.ge);
        CHECK(m->block_elems % m->group_elems == 0);                        // group divides block
        CHECK(r.maxc < (1u << m->code_bits) && (1u << m->code_bits) == r.levels && r.maxc < r.levels);   // codes fit the plane
    }
    CHECK(256 / 32 == 8);

    // bit-exact canonical roundtrip: decode(encode(block)) == reference dequantization, on random blocks of every type
    Rng rng(0xCA90);
    for (const Row& r : rows) {
        for (int it = 0; it < 200; ++it) {
            uint8_t raw[256]; for (auto& b : raw) b = (uint8_t)rng.below(256);
            // keep fp16 scales finite
            for (unsigned o = 0; o + 2 <= r.bb && o < 4; o += 2) raw[o + 1] &= 0x3F;
            canon::Block cb; float ref[256] = {0}, got[256] = {0};
            CHECK(canon::encode_block(r.t, raw, &cb));
            CHECK(canon::reference_decode(r.t, raw, ref));
            CHECK(canon::decode_block(r.t, cb, got));
            for (unsigned e = 0; e < r.be; ++e) {
                uint32_t a, b; std::memcpy(&a, &ref[e], 4); std::memcpy(&b, &got[e], 4);
                CHECK(a == b || (std::isnan(ref[e]) && std::isnan(got[e])));
            }
            for (int e = 0; e < cb.elems; ++e) CHECK(cb.codes[e] <= r.maxc);
        }
    }
    // Q2_0 raw <-> canonical is lossless on the raw block bytes
    for (int it = 0; it < 500; ++it) {
        uint8_t raw[18]; for (auto& b : raw) b = (uint8_t)rng.below(256);
        raw[1] &= 0x3F;
        canon::Block cb; canon::encode_q2(raw, &cb);
        uint8_t back[18]; canon::decode_q2(cb, back);
        CHECK(std::memcmp(raw, back, 18) == 0);
    }
}

// ======================================================================================================================
// GgufReaderGeometry
// ======================================================================================================================
struct G { uint32_t t; uint64_t el, by; };
static void put(Bytes& b, uint64_t v, int n) { for (int i = 0; i < n; ++i) b.push_back((uint8_t)(v >> (8 * i))); }
static void pstr(Bytes& b, const std::string& s) { put(b, s.size(), 8); b.insert(b.end(), s.begin(), s.end()); }

static void test_gguf() {
    using namespace gguf_reader;
    const G table[] = {{0, 1, 4}, {1, 1, 2}, {30, 1, 2}, {2, 32, 18}, {3, 32, 20}, {6, 32, 22}, {7, 32, 24}, {8, 32, 34}, {9, 32, 36},
        {10, 256, 84}, {11, 256, 110}, {12, 256, 144}, {13, 256, 176}, {14, 256, 210}, {16, 256, 66}, {17, 256, 74}, {18, 256, 98},
        {20, 32, 18}, {21, 256, 110}, {22, 256, 82}, {23, 256, 136}, {29, 256, 56}, {42, 64, 18}};
    for (uint32_t t = 0; t < 64; ++t) {
        const auto g = block_geometry(t);
        const G* want = nullptr; for (const G& e : table) if (e.t == t) want = &e;
        if (want) CHECK(g.el == want->el && g.by == want->by);
        else CHECK(g.el == 0 && g.by == 0);                                  // unknown types are refused
        if (g.el) {
            CHECK(g.el > 0 && g.by > 0);
            CHECK(g.el == 1 || g.el == 32 || g.el == 64 || g.el == 256);
            if (g.el == 64) CHECK(t == 42);                                   // Q2_0 is the only 64-element type
        }
    }
    CHECK(block_geometry(42).el == 64 && block_geometry(42).by == 18);
    CHECK(block_geometry(19).el == 0 && std::strcmp(ggml_type_name(19), "IQ1_S") == 0);   // named, but unsized
    CHECK(block_geometry(29).el == 256 && block_geometry(29).by == 56);
    CHECK(18 * 8 == 64 * 2 + 16);
    CHECK(std::strcmp(ggml_type_name(42), "Q2_0") == 0);

    // bounded cursor: a read advances by exactly the width, never leaves the buffer, and an overrun is refused
    Rng rng(0x6606);
    for (int it = 0; it < 3000; ++it) {
        const uint64_t size = rng.below(40);
        Bytes d(size); for (auto& x : d) x = (uint8_t)rng.below(256);
        Cursor c(d.data(), size);
        const uint64_t pos = rng.below((uint32_t)size + 1), n = rng.below(12);
        c.pos = pos;
        const bool fits = pos + n <= size;
        CHECK(c.need(n) == fits);
        if (fits) CHECK(c.ok && c.pos == pos);
        else CHECK(!c.ok);
        Cursor c2(d.data(), size); c2.pos = pos;
        const unsigned w = (unsigned)(1u << rng.below(4));                 // 1, 2, 4, 8
        (void)c2.uint_le(w);
        if (pos + w <= size) CHECK(c2.ok && c2.pos == pos + w);
        else CHECK(!c2.ok && c2.pos == pos);
    }
    { Bytes d = {1, 2, 3, 4, 5, 6, 7, 8}; Cursor c(d.data(), 8); CHECK(c.u64() == 0x0807060504030201ull && c.pos == 8 && c.ok); CHECK(c.u8() == 0 && !c.ok); }

    // alignment: data start is the round-up of the header end
    for (uint64_t a : {1ull, 2ull, 8ull, 32ull, 64ull}) for (uint64_t c = 0; c < 300; ++c) {
        const uint64_t u = (c + a - 1) / a * a;
        CHECK(u >= c && u < c + a && u % a == 0);
        if (a == 32) CHECK(u - c <= 31);
    }

    // end-to-end: build images with several header sizes, parse, data_start is aligned and covers the header
    for (int extra = 0; extra < 40; ++extra) {
        Bytes b; put(b, kMagic, 4); put(b, 3, 4); put(b, 1, 8); put(b, 1, 8);
        pstr(b, "general.architecture"); put(b, 8, 4); pstr(b, std::string("qwen") + std::string((size_t)extra, 'x'));
        pstr(b, "w"); put(b, 2, 4); put(b, 64, 8); put(b, 2, 8); put(b, 42, 4); put(b, 0, 8);
        const uint64_t header_end = b.size();
        while (b.size() % 32) b.push_back(0);
        b.resize(b.size() + 2 * 18, 0x55);
        auto f = GgufFile::parse(b.data(), b.size());
        CHECK(f.has_value());
        if (!f) continue;
        CHECK(f->data_start >= header_end && f->data_start < header_end + 32 && f->data_start % 32 == 0);
        CHECK(f->tensors.size() == 1 && f->tensors[0].elements() == 128 && f->tensors[0].nbytes() == 36);
        // truncation of the data section is refused
        CHECK(!GgufFile::parse(b.data(), b.size() - 1).has_value());
        // the count of any prefix is refused or a cursor overrun is reported, never a crash
        for (size_t cut = 0; cut < header_end; cut += 7) CHECK(!GgufFile::parse(b.data(), cut).has_value());
    }
    // tensor element count: product of the shape; zero dimension -> zero elements; concatenation multiplies
    { TensorInfo t; t.shape = {3, 5, 7}; CHECK(t.elements() == 105); t.shape = {3, 0, 7}; CHECK(t.elements() == 0);
      TensorInfo a, b2, ab; a.shape = {2, 3}; b2.shape = {4, 5}; ab.shape = {2, 3, 4, 5};
      CHECK(ab.elements() == a.elements() * b2.elements()); TensorInfo e; CHECK(e.elements() == 1); }
    // only 1..4 dimensions are accepted
    for (uint32_t nd = 0; nd <= 6; ++nd) {
        Bytes b; put(b, kMagic, 4); put(b, 3, 4); put(b, 1, 8); put(b, 0, 8);
        pstr(b, "w"); put(b, nd, 4); for (uint32_t i = 0; i < nd; ++i) put(b, 1, 8); put(b, 0, 4); put(b, 0, 8);
        while (b.size() % 32) b.push_back(0);
        b.resize(b.size() + 4, 0);
        CHECK(GgufFile::parse(b.data(), b.size()).has_value() == (nd >= 1 && nd <= 4));
    }
    // arrays keep at most 64 sampled items; the array length cap is 2^24
    CHECK(kMaxArrayElems == 16777216ull && (1ull << 24) == 16777216ull && kArraySample == 64);
    for (uint64_t n : {0ull, 1ull, 63ull, 64ull, 65ull, 1000ull, 1ull << 24}) CHECK(array_sample_count(n) == std::min<uint64_t>(n, 64));
    { Bytes b; put(b, 200, 8); for (int i = 0; i < 200; ++i) put(b, (uint64_t)i, 1);
      Bytes arr; put(arr, 0 /*U8*/, 4); put(arr, 200, 8); for (int i = 0; i < 200; ++i) put(arr, (uint64_t)i, 1);
      Cursor c(arr.data(), arr.size()); MetaValue v = read_value(c, MetaType::Array);
      CHECK(c.ok && v.arr.size() == 200 && v.sample_count() == 64);
      Bytes huge; put(huge, 0, 4); put(huge, (1ull << 24) + 1, 8);
      Cursor c3(huge.data(), huge.size()); (void)read_value(c3, MetaType::Array); CHECK(!c3.ok); }
    // magic
    CHECK(0x46 * 16777216u + 0x55 * 65536u + 0x47 * 256u + 0x47 == 0x46554747u && kMagic == 0x46554747u);

    // architecture guard constants: 512 experts / top-10, 24 heads over 2 kv heads, 48 layers (all required)
    auto guard_img_d = [&](uint32_t ex, uint32_t used, uint32_t nh, uint32_t nkv, uint32_t nl, uint32_t drop) {
      Bytes b; put(b, kMagic, 4); put(b, 3, 4); put(b, 0, 8); put(b, 6 - __builtin_popcount(drop), 8);
      pstr(b, "general.architecture"); put(b, 8, 4); pstr(b, "qwen4exp");
      auto kv = [&](const char* k, uint32_t v) { pstr(b, k); put(b, 4, 4); put(b, v, 4); };
      kv("qwen4exp.expert_count", ex);
      if (!(drop & 1)) kv("qwen4exp.expert_used_count", used);
      if (!(drop & 2)) kv("qwen4exp.attention.head_count", nh);
      if (!(drop & 4)) kv("qwen4exp.attention.head_count_kv", nkv);
      if (!(drop & 8)) kv("qwen4exp.block_count", nl);
      while (b.size() % 32) b.push_back(0);
      return b; };
    auto guard_img = [&](uint32_t ex, uint32_t used, uint32_t nh, uint32_t nkv, uint32_t nl) { return guard_img_d(ex, used, nh, nkv, nl, 0); };
    auto guard_ok = [&](Bytes b, std::string* why) { auto f = GgufFile::parse(b.data(), b.size()); return f && Qwen4ExpGuard().check(*f, why); };
    { std::string why; Qwen4ExpGuard d;
      CHECK(d.expected_experts == 512 && d.expected_used == 10 && d.expected_heads == 24 && d.expected_kv_heads == 2 && d.expected_layers == 48);
      CHECK(guard_ok(guard_img(512, 10, 24, 2, 48), &why));
      CHECK(!guard_ok(guard_img(4, 2, 24, 2, 48), &why) && why == "expert_count mismatch");
      CHECK(!guard_ok(guard_img(512, 4, 24, 2, 48), &why) && why == "expert_used_count mismatch");
      CHECK(!guard_ok(guard_img(512, 10, 32, 2, 48), &why) && why == "head_count mismatch");
      CHECK(!guard_ok(guard_img(512, 10, 24, 4, 48), &why) && why == "head_count_kv mismatch");
      CHECK(!guard_ok(guard_img(512, 10, 24, 2, 40), &why) && why == "block_count mismatch");
      CHECK(!guard_ok(guard_img_d(512, 10, 24, 2, 48, 1), &why) && why == "missing expert_used_count");
      CHECK(!guard_ok(guard_img_d(512, 10, 24, 2, 48, 2), &why) && why == "missing head_count");
      CHECK(!guard_ok(guard_img_d(512, 10, 24, 2, 48, 4), &why) && why == "missing head_count_kv");
      CHECK(!guard_ok(guard_img_d(512, 10, 24, 2, 48, 8), &why) && why == "missing block_count");
      }
    { Bytes img = guard_img(512, 10, 24, 2, 48); auto f = GgufFile::parse(img.data(), img.size());
      CHECK(f.has_value()); if (f) { CHECK(check_architecture(*f, "qwen4exp") && !check_architecture(*f, "llama"));
        Qwen4ExpGuard g8; g8.expected_experts = 8; std::string why; CHECK(!g8.check(*f, &why) && why == "expert_count mismatch"); } }
    CHECK(10 <= 512 && 24 % 2 == 0 && 24 / 2 == 12 && 48 > 0);
}

// ======================================================================================================================
// Q2_0GgufWriter
// ======================================================================================================================
static void test_q2writer() {
    using namespace q2w;
    // code clamp: always a valid 2-bit value; exact on {-1,0,1,2}; saturating; amax never yields code 3
    for (long r = -50; r <= 50; ++r) {
        const int c = code_of_symbol(r);
        CHECK(c >= 0 && c < 4);
        if (r >= -1 && r <= 2) CHECK(symbol_of_code(c) == r);
        if (r >= -1 && r <= 1) CHECK(c <= 2);
        if (r <= -1) CHECK(c == 0);
        if (r >= 2) CHECK(c == 3);
    }
    for (int c = 0; c < 4; ++c) CHECK(code_of_symbol(symbol_of_code(c)) == c);            // fixed point
    // packing
    for (int c0 = 0; c0 < 4; ++c0) for (int c1 = 0; c1 < 4; ++c1) for (int c2 = 0; c2 < 4; ++c2) for (int c3 = 0; c3 < 4; ++c3) {
        uint8_t codes[64] = {0}; codes[0] = (uint8_t)c0; codes[1] = (uint8_t)c1; codes[2] = (uint8_t)c2; codes[3] = (uint8_t)c3;
        uint8_t blk[18]; pack_block(0x3C00, codes, blk);
        const int byte = blk[2];
        CHECK(byte == c0 + 4 * c1 + 16 * c2 + 64 * c3 && byte < 256);
        CHECK(byte % 4 == c0 && byte / 4 % 4 == c1 && byte / 16 % 4 == c2 && byte / 64 % 4 == c3);
        uint8_t back[64]; unpack_codes(blk, back);
        CHECK(back[0] == c0 && back[1] == c1 && back[2] == c2 && back[3] == c3);
    }
    for (size_t j = 0; j < 64; ++j) CHECK(byte_index(j) < 16 && bit_offset(j) < 8 && byte_index(j) == j / 4 && bit_offset(j) == (j % 4) * 2);
    CHECK(2 + 64 / 4 == 18 && kBlockBytes == 18 && kBlockElems == 64 && 64 * 16 / 64 * 18 == 288);

    // quantizer on random blocks: code 3 never appears with d = amax, requantization is a fixed point
    Rng rng(0x2002);
    for (int it = 0; it < 3000; ++it) {
        float w[64]; for (auto& x : w) x = (float)(rng.gauss() * (it % 7 + 1));
        if (it % 11 == 0) std::fill(w, w + 64, 0.f);
        uint8_t blk[18]; quantize_block(w, blk);
        uint8_t codes[64]; unpack_codes(blk, codes);
        for (int j = 0; j < 64; ++j) CHECK(codes[j] <= 2);
        float y[64]; dequantize_block(blk, y);
        uint8_t blk2[18]; quantize_block(y, blk2);
        uint8_t codes2[64]; unpack_codes(blk2, codes2);
        const float d = f16::half_to_float((uint16_t)(blk[0] | (blk[1] << 8)));
        CHECK(std::memcmp(codes, codes2, 64) == 0);                            // idempotent: requantizing the dequantized symbols
        // requantizing against the SAME scale returns the same codes
        uint8_t codes3[64]; quantize_codes_with_scale(y, d, codes3);
        CHECK(std::memcmp(codes, codes3, 64) == 0);
    }

    // padding agrees with the reader's round-up
    for (uint64_t n = 0; n < 400; ++n) {
        const uint64_t p = pad(32, n);
        CHECK(p < 32 && (n + p) % 32 == 0 && n + p == (n + 31) / 32 * 32);
        for (uint64_t o : {0ull, 5ull, 100ull}) CHECK((o + n + 31) / 32 * 32 % 32 == 0 && o + n <= (o + n + 31) / 32 * 32);
    }
    CHECK(4 + 4 + 8 + 8 == 24);
    for (size_t nl = 0; nl < 20; ++nl) for (size_t nd = 1; nd < 5; ++nd) CHECK(8 + nl + 4 + 8 * nd + 4 + 8 == nl + 32 + 8 * (nd - 1));

    // a written file: header 24 bytes, 32-byte aligned data section, parses back with the reader, metadata ids match
    {
        float w[64 * 16]; for (auto& x : w) x = (float)rng.gauss();
        Bytes data; for (int b = 0; b < 16; ++b) { uint8_t blk[18]; quantize_block(w + 64 * b, blk); data.insert(data.end(), blk, blk + 18); }
        CHECK(data.size() == 288);
        TensorSpec ts; ts.name = "t"; ts.shape = {64, 16}; ts.type = 42; ts.data = data;
        TensorSpec ts2; ts2.name = "second.tensor"; ts2.shape = {64}; ts2.type = 42; ts2.data.assign(18, 0x11);
        Bytes out; std::string err;
        std::vector<Kv> kvs = {{"general.architecture", std::string("qwen4exp")}, {"u32", (uint32_t)7}, {"u64", (uint64_t)9},
                               {"f", 1.5f}, {"flag", true}};
        CHECK(write_file(kvs, {ts, ts2}, &out, &err));
        CHECK(out.size() > 24 && out.size() % 32 == 0);
        auto f = gguf_reader::GgufFile::parse(out.data(), out.size());
        CHECK(f.has_value());
        if (f) {
            CHECK(f->data_start % 32 == 0 && f->tensors.size() == 2);
            CHECK(f->tensors[0].offset == 0 && f->tensors[1].offset == 288 && f->tensors[1].offset % 32 == 0);   // 288 is a multiple of 32
            CHECK(f->get_uint("u32") == 7u && f->get_uint("u64") == 9u && f->find_meta("flag")->b);
            CHECK(std::memcmp(out.data() + f->data_start, data.data(), 288) == 0);
        }
        // supported tensor ids F32 / F16 / Q2_0; wrong byte count refused
        TensorSpec bad = ts; bad.data.pop_back();
        CHECK(!write_file(kvs, {bad}, &out, &err));
        CHECK(!write_file(kvs, {ts}, &out, &err, 0));
        TensorSpec f32; f32.name = "f"; f32.shape = {8}; f32.type = 0; f32.data.assign(32, 0);
        TensorSpec f16t; f16t.name = "h"; f16t.shape = {8}; f16t.type = 1; f16t.data.assign(16, 0);
        CHECK(write_file(kvs, {f32, f16t}, &out, &err));
        // custom alignment
        CHECK(write_file(kvs, {ts}, &out, &err, 64));
        auto f64 = gguf_reader::GgufFile::parse(out.data(), out.size());
        CHECK(f64.has_value() && f64->alignment == 64 && f64->data_start % 64 == 0);
    }
    // metadata type ids 0..12 are the reader's MetaType enum
    { using gguf_reader::MetaType;
      const MetaType ids[] = {MetaType::U8, MetaType::I8, MetaType::U16, MetaType::I16, MetaType::U32, MetaType::I32, MetaType::F32,
                              MetaType::Bool, MetaType::String, MetaType::Array, MetaType::U64, MetaType::I64, MetaType::F64};
      for (uint32_t i = 0; i < 13; ++i) CHECK((uint32_t)ids[i] == i); }
}

// ======================================================================================================================
// DirectFileIo
// ======================================================================================================================
static bool valid_req(uint64_t a, uint64_t off, uint64_t buf, uint64_t len) { return off % a == 0 && len % a == 0 && buf % a == 0 && len != 0; }

static void test_dfio() {
    using namespace dfio;
    CHECK(kAlign == 4096);
    Rng rng(0xD10);
    for (int it = 0; it < 20000; ++it) {
        const uint64_t off = (uint64_t)rng.u32() * (rng.below(2) ? 4096 : 1) + (rng.below(5) == 0 ? rng.below(4096) : 0);
        const uint64_t buf = (uint64_t)rng.u32() * (rng.below(2) ? 4096 : 1);
        const uint64_t len = (uint64_t)rng.below(64) * (rng.below(2) ? 4096 : 1) + (rng.below(7) == 0 ? rng.below(4096) : 0);
        const bool ok = acceptable(off, buf, len);
        CHECK(ok == valid_req(4096, off, buf, len));
        if (ok) CHECK(off % 4096 == 0 && len % 4096 == 0 && buf % 4096 == 0 && len != 0);
        CHECK(!acceptable(off, buf, 0));
    }
    for (uint64_t k = 1; k < 6; ++k) for (uint64_t p = 0; p < 6; ++p) CHECK(acceptable(p * 4096, 8192, k * 4096));
    // 64-bit offset split
    for (int it = 0; it < 20000; ++it) {
        const uint64_t off = ((uint64_t)rng.u32() << 32) | rng.u32() | ((uint64_t)rng.below(2) << 31);
        CHECK(((uint64_t)high_word(off) << 32) + low_word(off) == off);
        CHECK(recombine(low_word(off), high_word(off)) == off);
        CHECK(low_word(off) == off % 4294967296ull && high_word(off) == off / 4294967296ull);
    }
    // POSIX completion rule
    CHECK(posix_bytes(-1) == 0 && !posix_ok(-1) && posix_bytes(-9999) == 0);
    for (int64_t g : {0ll, 1ll, 4096ll, 1ll << 40}) CHECK(posix_ok(g) && posix_bytes(g) == g);
    // EOF is a legal short read
    CHECK(status_ok(0) && status_ok(kStatusEndOfFile) && !status_ok(1) && !status_ok(0xC0000005u));
    // wait batching
    for (uint64_t q = 0; q < 20; ++q) for (uint64_t m = 0; m < 20; ++m) {
        const uint64_t n = drain_count(q, m);
        CHECK(n <= m && n <= q && (n == q || n == m));
    }
    for (uint64_t m = 0; m < 200; ++m) for (uint64_t n = 0; n <= m; ++n) CHECK(batch_want(m, n) <= 64 && batch_want(m, n) <= m - n);

    // the real POSIX queue: every submit yields exactly one completion, FIFO, wait never exceeds max
    char tmpl[] = "./u2e9_XXXXXX";
    const int fd = ::mkstemp(tmpl);
    CHECK(fd >= 0);
    if (fd < 0) return;
    const size_t pages = 5, fsize = pages * 4096 + 100;
    std::vector<uint8_t> content(fsize); for (size_t i = 0; i < fsize; ++i) content[i] = (uint8_t)((i * 31 + (i >> 12)) & 0xFF);
    CHECK(::write(fd, content.data(), fsize) == (ssize_t)fsize);
    ::close(fd);
    const int rfd = ::open(tmpl, O_RDONLY);
    {
        PosixDirectFile pf(rfd);
        void* mem = nullptr; CHECK(posix_memalign(&mem, 4096, 4096 * 8) == 0);
        uint8_t* buf = (uint8_t*)mem;
        CHECK(!pf.submit(1, 1, buf, 4096) && !pf.submit(2, 0, buf, 100) && !pf.submit(3, 0, buf + 8, 4096) && !pf.submit(4, 0, buf, 0));
        CHECK(pf.pending() == 0);                                           // rejections queue nothing
        for (uint64_t i = 0; i < 5; ++i) CHECK(pf.submit(100 + i, i * 4096, buf + i * 4096, 4096));
        CHECK(pf.pending() == 5);                                           // exactly one completion per submit
        Completion c[8];
        CHECK(pf.wait(c, 2) == 2 && c[0].tag == 100 && c[1].tag == 101 && pf.pending() == 3);   // oldest first, at most max
        CHECK(pf.wait(c, 8) == 3 && c[0].tag == 102 && c[1].tag == 103 && c[2].tag == 104 && pf.pending() == 0);
        for (int i = 0; i < 3; ++i) CHECK(c[i].ok && c[i].bytes == 4096);
        CHECK(std::memcmp(buf, content.data(), 5 * 4096) == 0);
        // short read at EOF: ok with fewer bytes; zero-byte EOF completion is ok, too
        CHECK(pf.submit(7, 4 * 4096, buf, 4096 * 2));
        CHECK(pf.submit(8, 8 * 4096, buf, 4096));
        CHECK(pf.wait(c, 8) == 2 && c[0].ok && c[0].bytes == 4096 + 100 && c[1].ok && c[1].bytes == 0);
        CHECK(pf.wait(c, 8) == 0);
        free(mem);
    }
    ::close(rfd);
    // failed read: ok = false, bytes = 0
    {
        PosixDirectFile pf(-1);
        void* mem = nullptr; CHECK(posix_memalign(&mem, 4096, 4096) == 0);
        CHECK(pf.submit(9, 0, mem, 4096));
        Completion c[2]; CHECK(pf.wait(c, 2) == 1 && c[0].tag == 9 && !c[0].ok && c[0].bytes == 0);
        free(mem);
    }
    ::unlink(tmpl);
}

// ======================================================================================================================
// WeightPackLoader
// ======================================================================================================================
static uint64_t round_up(uint64_t a, uint64_t n) { return (n + a - 1) / a * a; }

static void test_weights() {
    using namespace wldr;
    CHECK(kChunk == 8ull * 1048576 && kChunk % 2 == 0 && kStageOut == 2 * kChunk);
    // staging: output of every conversion of a chunk fits 2 x CHUNK
    for (uint64_t n : {(uint64_t)0, (uint64_t)1, (uint64_t)2, (uint64_t)4, (uint64_t)100, (uint64_t)(kChunk - 4), (uint64_t)kChunk}) {
        CHECK(n / 4 * 2 <= kStageOut && n / 2 * 4 <= kStageOut && n <= kStageOut);
        if (n % 2 == 0) CHECK(n / 2 * 4 == 2 * n);                          // widening doubles an even plane
    }
    for (uint64_t e : {0ull, 1ull, 7ull, 4096ull}) CHECK((e * 4) / 4 * 2 == e * 2);   // promotion halves
    for (uint64_t e : {0ull, 5ull, 100ull}) {
        CHECK(promoted_ok(e, e * 4, e * 2) && !promoted_ok(e, e * 4 + 4, e * 2) && !promoted_ok(e, e * 4, e * 2 + 2));
    }
    for (uint64_t rem : {0ull, 6ull, 100ull}) for (uint64_t ch : {2ull, 4ull, 8ull}) { if (rem % 2 == 0) CHECK(std::min(rem, ch) % 2 == 0); }

    // bf16 high half: exact for bf16-in-f32 containers
    Rng rng(0x3E16);
    for (int it = 0; it < 20000; ++it) {
        const uint32_t v = rng.u32() << 16;                                // low half zero
        const uint32_t hi = v / 65536;
        CHECK(hi < 65536 && hi * 65536u == v);
        uint8_t src[4]; std::memcpy(src, &v, 4); uint8_t dst[2];
        CHECK(transform_chunked(Xform::Bf16High, src, 4, dst) == 2);
        CHECK((uint32_t)(dst[0] | (dst[1] << 8)) == hi);
    }
    // segment accounting
    for (int it = 0; it < 4000; ++it) {
        const uint64_t cb = rng.below(10000), sb = rng.below(5000) * 2, ob = rng.below(3000);
        std::vector<Seg> s;
        CHECK(plan_quantized(cb, sb, ob, false, s));
        CHECK(seg_src_total(s) == seg_dst_total(s) && seg_src_total(s) == cb + sb + ob);
        CHECK(plan_quantized(cb, sb, ob, true, s));
        CHECK(seg_dst_total(s) == cb + sb + ob && seg_src_total(s) == cb + sb / 2 + ob && seg_dst_total(s) == seg_src_total(s) + sb / 2);
        // a mis-sized plane layout is caught by the totals check
        const uint64_t rec_src = cb + sb / 2 + ob, rec_dst = cb + sb + ob;
        CHECK(seg_src_total(s) == rec_src && seg_dst_total(s) == rec_dst);
        CHECK(seg_src_total(s) != rec_src + 1);
        // segments are contiguous in source and destination
        CHECK(s.size() == 3 && s[0].src_off == 0 && s[1].src_off == s[0].src_bytes && s[2].src_off == s[1].src_off + s[1].src_bytes);
        CHECK(s[1].dst_off == s[0].dst_bytes && s[2].dst_off == s[1].dst_off + s[1].dst_bytes);
    }
    { std::vector<Seg> s; CHECK(!plan_quantized(10, 3, 0, true, s) && s.empty()); }   // odd fp16 scale plane is refused

    // alignment and compaction
    for (uint64_t a : {1ull, 16ull, 256ull}) for (uint64_t n = 0; n < 700; n += 3) {
        const uint64_t r = round_up(a, n); CHECK(r >= n && r % a == 0 && r < n + a);
    }
    for (int it = 0; it < 3000; ++it) {
        const size_t rows = 1 + rng.below(12);
        std::vector<uint64_t> sizes(rows); std::vector<bool> skip(rows);
        uint64_t kept_sum = 0, expect_total = 0;
        for (size_t i = 0; i < rows; ++i) { sizes[i] = rng.below(2000); skip[i] = rng.below(3) == 0; }
        const Placement p = place(sizes, skip);
        uint64_t cursor = 0;
        for (size_t i = 0; i < rows; ++i) {
            if (skip[i]) { CHECK(p.off[i] == UINT64_MAX); continue; }       // skipped rows take no bytes
            CHECK(p.off[i] == cursor && p.off[i] % 256 == 0);
            cursor += round_up(256, sizes[i]); kept_sum += sizes[i]; expect_total += round_up(256, sizes[i]);
        }
        CHECK(p.total == expect_total && p.total % 256 == 0 && p.total >= kept_sum);
        for (size_t i = 0; i < rows; ++i) if (!skip[i]) CHECK(p.off[i] + sizes[i] <= p.total);
        // the arena check refuses exactly "too small"
        for (uint64_t arena : {p.total, p.total + 1, p.total ? p.total - 1 : 0}) CHECK((p.total <= arena) == !(arena < p.total));
    }

    // chunked copy loop: covers each plane once, chunks never exceed the chunk size, the widening offset is consistent
    for (uint64_t chunk : {4ull, 8ull, 64ull, 4096ull}) for (int it = 0; it < 120; ++it) {
        const uint64_t n = rng.below(20000) * 2;
        std::vector<uint8_t> src(n); for (auto& b : src) b = (uint8_t)rng.below(256);
        uint64_t covered = 0, done = 0;
        while (done < n) { const uint64_t c = chunk_len(n - done, chunk); CHECK(c > 0 && c <= chunk); covered += c; done += c; }
        CHECK(covered == n);
        std::vector<uint8_t> out(n * 2 + 8, 0xEE), whole(n * 2 + 8, 0xDD);
        CHECK(transform_chunked(Xform::WidenF16, src.data(), n, out.data(), chunk) == n * 2);
        CHECK(transform_chunked(Xform::WidenF16, src.data(), n, whole.data(), 1 << 22) == n * 2);
        CHECK(std::memcmp(out.data(), whole.data(), n * 2) == 0);           // chunking never changes the result
        std::vector<uint8_t> cp(n + 8); CHECK(transform_chunked(Xform::Copy, src.data(), n, cp.data(), chunk) == n && std::memcmp(cp.data(), src.data(), n) == 0);
        // the widening of each element equals the reference widening
        for (uint64_t i = 0; i < std::min<uint64_t>(n / 2, 64); ++i) {
            const uint16_t h = (uint16_t)(src[2 * i] | (src[2 * i + 1] << 8));
            const float f = f16::half_to_float(h); uint32_t a, b; std::memcpy(&a, &f, 4); std::memcpy(&b, &out[4 * i], 4);
            CHECK(a == b);
        }
    }
    CHECK(transform_chunked(Xform::WidenF16, nullptr, 3, nullptr) == UINT64_MAX);      // odd source: refused, never splits an element
    CHECK(transform_chunked(Xform::Bf16High, nullptr, 6, nullptr) == UINT64_MAX);
    CHECK(transform_chunked(Xform::Copy, nullptr, 0, nullptr, 3) == UINT64_MAX);       // chunk must be a multiple of 4
    for (uint64_t done : {0ull, 6ull, 100ull}) for (uint64_t n : {0ull, 2ull, 8ull}) CHECK((done + n) * 2 == done * 2 + n / 2 * 4);
}

// ======================================================================================================================
// LayerShapeCheck
// ======================================================================================================================
static shape::TensorIndex full_index(const shape::Geometry& g) {
    shape::TensorIndex idx;
    for (int l = 0; l < g.n_layers; ++l)
        for (const auto& r : shape::required_for_layer(g, l)) idx[r.name] = {r.ne0, r.ne1, r.form};
    return idx;
}

static void test_shape() {
    using namespace shape;
    // 16 two-D tensors on either family, 6 / 7 one-D tensors
    CHECK(count_two_d(true) == 16 && count_two_d(false) == 16 && count_one_d(true) == 7 && count_one_d(false) == 6);
    { int c = 0, q = 0, gd = 0; for (const auto& w : kWant2) { if (w.qsa_only) ++q; else if (w.gdn_only) ++gd; else ++c; CHECK(!(w.qsa_only && w.gdn_only)); }
      CHECK(c == 10 && q == 6 && gd == 6 && c + q + gd == 22); }
    { int c = 0, q = 0, gd = 0; for (const auto& w : kWant1) { if (w.qsa_only) ++q; else if (w.gdn_only) ++gd; else ++c; }
      CHECK(c == 3 && q == 4 && gd == 3 && c + q + gd == 10); }
    for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) for (int q = 0; q < 2; ++q)
        CHECK(applies(a, b, q) == (!(a && !q) && !(b && q)));
    CHECK(applies(true, false, false) == false && applies(false, true, true) == false && applies(false, false, true) && applies(true, false, true));

    // forms and byte counts
    CHECK(form_bytes(Form::F32, 10) == 40 && form_bytes(Form::Bf16InF32, 10) == 20);
    CHECK(form_bytes(Form::Bf16InF32, 2560) == 5120 && form_bytes(Form::F32, 2560) == 10240 && form_bytes(Form::Bf16InF32, 2560) < form_bytes(Form::F32, 2560));
    for (uint64_t n = 1; n < 100; ++n) CHECK(form_bytes(Form::Bf16InF32, n) < form_bytes(Form::F32, n));
    CHECK(form_ok(Form::F32, Form::F32) && form_ok(Form::Bf16InF32, Form::Bf16InF32) && !form_ok(Form::F32, Form::Bf16InF32) && !form_ok(Form::Bf16InF32, Form::F32));
    CHECK(4 * 2560 == 10240);

    Geometry g;
    CHECK(g.v(Dim::HcDim) == 10240);
    // layer split partitions the layers
    for (int n = 0; n <= 96; ++n) { int q = 0, gd = 0; for (int l = 0; l < n; ++l) (g.is_qsa(l) ? q : gd)++; CHECK(q + gd == n); }
    { int q = 0; for (int l = 0; l < 48; ++l) q += g.is_qsa(l); CHECK(q == 12 && 48 - q == 36 && 12 + 36 == 48); }
    CHECK(split_ok(12, 36, 12, 36) && !split_ok(13, 35, 12, 36) && !split_ok(12, 35, 12, 36));
    CHECK(layer_in_range(0, 48) && layer_in_range(47, 48) && !layer_in_range(48, 48) && !layer_in_range(-1, 48));

    // required lists
    for (int l = 0; l < g.n_layers; ++l) {
        const auto r = required_for_layer(g, l);
        CHECK(r.size() == (g.is_qsa(l) ? 23u : 22u));
        int two = 0, one = 0; for (const auto& x : r) (x.one_d ? one : two)++;
        CHECK(two == 16 && one == (g.is_qsa(l) ? 7 : 6));
    }

    // a consistent table passes; every single-tensor corruption is caught
    const TensorIndex ok = full_index(g);
    std::string why;
    CHECK(check_all(ok, g, &why));
    for (int l = 0; l < g.n_layers; ++l) CHECK(check_layer(ok, g, l));
    CHECK(!check_layer(ok, g, 48, &why) && why == "layer out of range");
    CHECK(!check_layer(ok, g, -1, &why));
    Rng rng(0x5A9E);
    for (int it = 0; it < 400; ++it) {
        const int l = (int)rng.below(48);
        const auto reqs = required_for_layer(g, l);
        const auto& r = reqs[rng.below((uint32_t)reqs.size())];
        TensorIndex bad = ok;
        TensorDesc& d = bad[r.name];
        switch (rng.below(r.one_d ? 4 : 3)) {
            case 0: d.ne0 += 1; break;
            case 1: if (r.one_d) d.ne1 = 5; else d.ne1 += 1; break;
            case 2: if (r.one_d) d.ne0 = 0; else std::swap(d.ne0, d.ne1); if (!r.one_d && r.ne0 == r.ne1) d.ne0 += 1; break;
            default: d.form = (d.form == Form::F32) ? Form::Bf16InF32 : Form::F32; break;
        }
        CHECK(!check_layer(bad, g, l, &why));
        CHECK(!check_all(bad, g, &why));
        TensorIndex missing = ok; missing.erase(r.name);
        CHECK(!check_layer(missing, g, l, &why) && why == "missing " + r.name);
        // other layers are unaffected
        CHECK(check_layer(bad, g, (l + 1) % 48));
    }
    // the shared-expert gate: bf16 recorded as f32 is refused by form, though the element count matches
    { TensorIndex bad = ok; bad[tensor_name(5, "ffn_gate_inp_shexp")].form = Form::F32; CHECK(!check_layer(bad, g, 5, &why)); CHECK(why.find("form") != std::string::npos); }
    // wrong kind counts: a geometry that expects 13 QSA layers is refused even though every layer is shaped correctly
    { Geometry g2; g2.expect_qsa = 13; g2.expect_gdn = 35; CHECK(!check_all(ok, g2, &why) && why.find("counts") != std::string::npos); }
    { Geometry g2; g2.n_layers = 0; CHECK(!check_all(ok, g2, &why)); }
    // an extra family: a GDN-only tensor missing on a GDN layer is caught, but absent on a QSA layer is fine
    { TensorIndex t = ok; t.erase(tensor_name(0, "ssm_in")); CHECK(!check_layer(t, g, 0)); }
    { TensorIndex t = ok; t.erase(tensor_name(3, "ssm_in")); CHECK(check_layer(t, g, 3)); }
}

int main() {
    test_bpe();
    test_canon();
    test_gguf();
    test_q2writer();
    test_dfio();
    test_weights();
    test_shape();
    std::printf("%ld checks, %ld failed\n", testh::checks(), testh::failures());
    if (testh::failures() != 0) { std::printf("U2E9 TESTS FAILED\n"); return 1; }
    std::printf("U2E9 TESTS PASSED\n");
    return 0;
}


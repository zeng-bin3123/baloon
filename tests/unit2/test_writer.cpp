#include <cmath>
#include <cstring>
#include <vector>
#include "baloon/q2_gguf_writer.hpp"
#include "check.hpp"
using namespace baloon;
using testh::Rng;

void test_writer_code_algebra() {
    for (long s = -1000; s <= 1000; ++s) {
        const int c = q2w::code_of_symbol(s);
        CHECK(c >= 0 && c <= 3);
        if (s >= -1 && s <= 2) CHECK(c == s + 1);  // no clamping on the grid
    }
    for (int c = 0; c <= 3; ++c) {
        CHECK(q2w::code_of_symbol(q2w::symbol_of_code(c)) == c);  // requantize fixed point
        CHECK(q2w::symbol_of_code(c) >= -1 && q2w::symbol_of_code(c) <= 2);
    }
    CHECK(q2w::kBlockBytes == q2w::kCodeBytes + 2 && q2w::kCodeBytes == 64 / 4);
}

void test_writer_packing() {
    for (size_t j = 0; j < 64; ++j) {
        CHECK(q2w::byte_index(j) < 16 && q2w::bit_offset(j) < 8 && q2w::bit_offset(j) % 2 == 0);
        for (size_t i = 0; i < 64; ++i)
            if (q2w::byte_index(i) == q2w::byte_index(j) && q2w::bit_offset(i) == q2w::bit_offset(j)) CHECK(i == j);
    }
    for (int c0 = 0; c0 < 4; ++c0) for (int c1 = 0; c1 < 4; ++c1) for (int c2 = 0; c2 < 4; ++c2) for (int c3 = 0; c3 < 4; ++c3) {
        const int by = c0 + 4 * c1 + 16 * c2 + 64 * c3;
        CHECK(by < 256 && by % 4 == c0 && by / 4 % 4 == c1 && by / 16 % 4 == c2 && by / 64 % 4 == c3);
    }
    Rng r(30);
    for (int it = 0; it < 5000; ++it) {
        uint8_t codes[64], back[64], blk[20];
        for (auto& c : codes) c = (uint8_t)r.below(4);
        std::memset(blk, 0xEE, sizeof blk);
        const uint16_t d = (uint16_t)r.u32();
        q2w::pack_block(d, codes, blk);
        CHECK(blk[18] == 0xEE && blk[19] == 0xEE);
        CHECK((uint16_t)(blk[0] | (blk[1] << 8)) == d);
        q2w::unpack_codes(blk, back);
        CHECK(std::memcmp(codes, back, 64) == 0);
        // LSB-first: element j in byte j/4 at bit 2*(j%4)
        for (int j = 0; j < 64; ++j) CHECK(((blk[2 + j / 4] >> (2 * (j % 4))) & 3) == codes[j]);
    }
}

void test_writer_quantizer() {
    Rng r(31);
    for (int it = 0; it < 20000; ++it) {
        float w[64];
        const int kind = it % 4;
        for (auto& x : w) x = kind == 0 ? (float)r.gauss() : kind == 1 ? (float)(r.gauss() * r.gauss() * 100) : kind == 2 ? (float)(r.uni() - 0.5) * 1e-3f : (float)(r.below(3) == 0 ? 7.f : 0.f);
        uint8_t blk[20]; std::memset(blk, 0x77, sizeof blk);
        q2w::quantize_block(w, blk);
        CHECK(blk[18] == 0x77 && blk[19] == 0x77);
        uint8_t codes[64]; q2w::unpack_codes(blk, codes);
        for (int j = 0; j < 64; ++j) CHECK(codes[j] <= 3);
        const float d = f16::half_to_float((uint16_t)(blk[0] | (blk[1] << 8)));
        float y[64]; q2w::dequantize_block(blk, y);
        for (int j = 0; j < 64; ++j) {
            const double err = std::fabs((double)w[j] - (double)y[j]);
            CHECK_MSG(err <= 0.5 * (double)d * 1.002 + 1e-30 || std::fabs(w[j]) > 65000.f, "err=%g d=%g", err, (double)d);
        }
        // fixed point: requantising the decoded symbols with the same scale returns the same codes
        uint8_t again[64]; q2w::quantize_codes_with_scale(y, d, again);
        CHECK(std::memcmp(codes, again, 64) == 0);
    }
    float z[64] = {0};
    uint8_t blk[18]; q2w::quantize_block(z, blk);
    CHECK(blk[0] == 0 && blk[1] == 0);
    for (int k = 2; k < 18; ++k) CHECK(blk[k] == 0x55);  // symbol 0 -> code 1 everywhere
    float y[64]; q2w::dequantize_block(blk, y);
    for (float v : y) CHECK(v == 0.f);
    // extreme element maps to symbol +/-1 (code 2 / 0)
    float w[64] = {0}; w[3] = 2.f; w[10] = -2.f;
    q2w::quantize_block(w, blk);
    uint8_t codes[64]; q2w::unpack_codes(blk, codes);
    CHECK(codes[3] == 2 && codes[10] == 0 && codes[0] == 1);
}

void test_writer_padding() {
    for (uint64_t a : {1ull, 2ull, 3ull, 8ull, 32ull, 64ull, 256ull})
        for (uint64_t n = 0; n < 1100; ++n) {
            CHECK((n + q2w::pad(a, n)) % a == 0);
            CHECK(q2w::pad(a, n) < a);
        }
    for (uint64_t n = 0; n < 200; ++n) CHECK(q2w::pad(32, n) < 32);
    CHECK(gguf::kMagic % 256 == 0x47 && gguf::kMagic / 256 % 256 == 0x47 && gguf::kMagic / 65536 % 256 == 0x55 && gguf::kMagic / 16777216 % 256 == 0x46);
}

void test_writer_file_layout() {
    Rng r(32);
    for (uint32_t align : {32u, 64u, 128u}) {
        std::vector<q2w::TensorSpec> ts;
        for (int i = 0; i < 4; ++i) {
            q2w::TensorSpec t;
            t.name = "t" + std::to_string(i);
            t.shape = {64u * (1 + r.below(3)), 1 + r.below(5)};
            t.type = gguf::kQ2_0;
            uint64_t n = t.shape[0] * t.shape[1];
            t.data.resize(n / 64 * 18);
            for (auto& b : t.data) b = (uint8_t)r.u32();
            ts.push_back(t);
        }
        std::vector<q2w::Kv> kvs = {{"general.name", std::string("x")}, {"k.u32", uint32_t(9)}};
        std::vector<uint8_t> img; std::string err;
        CHECK(q2w::write_file(kvs, ts, &img, &err, align));
        CHECK(img.size() >= 24 && img[0] == 'G' && img[1] == 'G' && img[2] == 'U' && img[3] == 'F' && img[4] == 3 && img[5] == 0);
        auto ix = gguf::parse_header(img.data(), img.size());
        CHECK(ix.has_value());
        if (!ix) continue;
        CHECK(ix->alignment == align && ix->data_start % align == 0);
        uint64_t expect_end = ix->data_start;
        for (size_t i = 0; i < ts.size(); ++i) {
            const auto& ti = ix->tensors[i];
            CHECK(ti.name == ts[i].name && ti.shape == ts[i].shape && ti.type == gguf::kQ2_0);
            CHECK(ti.offset % align == 0);
            CHECK(ti.nbytes() == ts[i].data.size());
            CHECK(std::memcmp(img.data() + ix->data_start + ti.offset, ts[i].data.data(), ts[i].data.size()) == 0);
            expect_end = ix->data_start + ti.offset + ts[i].data.size();
            expect_end += q2w::pad(align, ts[i].data.size());
        }
        CHECK(img.size() == expect_end);  // data section ends padded to the alignment
        // padding bytes are zero
        for (size_t k = ix->data_start + ix->tensors.back().offset + ts.back().data.size(); k < img.size(); ++k) CHECK(img[k] == 0);
    }
    // inconsistent tensors are rejected
    std::vector<uint8_t> img; std::string err;
    q2w::TensorSpec bad{"bad", {64}, gguf::kQ2_0, std::vector<uint8_t>(17, 0)};
    CHECK(!q2w::write_file({}, {bad}, &img, &err) && !err.empty());
    q2w::TensorSpec bad2{"bad2", {100}, gguf::kQ2_0, std::vector<uint8_t>(18, 0)};  // not whole blocks
    CHECK(!q2w::write_file({}, {bad2}, &img, &err));
    q2w::TensorSpec bad3{"bad3", {64}, 9999, std::vector<uint8_t>(18, 0)};
    CHECK(!q2w::write_file({}, {bad3}, &img, &err));
    CHECK(!q2w::write_file({}, {}, &img, &err, 0));
    // an empty file is a valid header
    CHECK(q2w::write_file({}, {}, &img, &err) && gguf::parse_header(img.data(), img.size()).has_value());
    // end-to-end: quantise -> write -> parse -> dequantise equals direct dequantisation
    float w[128]; for (auto& x : w) x = (float)r.gauss();
    q2w::TensorSpec t{"w", {128}, gguf::kQ2_0, std::vector<uint8_t>(36)};
    q2w::quantize_block(w, t.data.data()); q2w::quantize_block(w + 64, t.data.data() + 18);
    CHECK(q2w::write_file({}, {t}, &img, &err));
    gguf::LazyImage li(img.data(), img.size());
    auto sp = li.read_tensor("w");
    CHECK(li.ok() && sp && sp->len == 36);
    if (sp) for (int b = 0; b < 2; ++b) {
        float a[64], c[64];
        q2w::dequantize_block(sp->ptr + 18 * b, a); q2w::dequantize_block(t.data.data() + 18 * b, c);
        CHECK(std::memcmp(a, c, sizeof a) == 0);
    }
}

#include <cstring>
#include <vector>
#include "baloon/canonical_form_roundtrip.hpp"
#include "check.hpp"
using namespace baloon;
using namespace baloon::canon;
using testh::Rng;

static void fix_half(uint8_t* p, Rng& r) {  // finite fp16 (exponent field < 31), any sign
    uint16_t h = (uint16_t)r.u32();
    h = (uint16_t)((h & 0x7BFF) | (h & 0x8000));
    p[0] = (uint8_t)(h & 0xFF); p[1] = (uint8_t)(h >> 8);
}
static void rand_block(Rng& r, QType t, std::vector<uint8_t>& b) {
    const Mapping* m = find_mapping(t);
    b.resize(m->block_bytes);
    for (auto& x : b) x = (uint8_t)r.u32();
    switch (t) {
        case QType::Q6_K: fix_half(&b[208], r); break;
        case QType::Q3_K: fix_half(&b[108], r); break;
        case QType::Q4_K: case QType::Q5_K: fix_half(&b[0], r); fix_half(&b[2], r); break;
        default: fix_half(&b[0], r); break;
    }
}

void test_canon_registry() {
    for (size_t i = 0; i < kRegistrySize; ++i) {
        const Mapping& m = kRegistry[i];
        CHECK(m.group_elems > 0 && m.block_elems % m.group_elems == 0);  // registry_groups_divide
    }
    CHECK(kRegistrySize == 13);
    CHECK(2 + 64 * 2 / 8 == 18 && 2 + 32 * 4 / 8 == 18 && 2 + 4 + 32 * 4 / 8 == 22 && 2 + 32 == 34);
    CHECK(32 + 64 + 12 + 2 == 110 && 4 + 12 + 128 == 144 && 4 + 12 + 32 + 128 == 176 && 128 + 64 + 16 + 2 == 210 && 2 + 2 + 4 + 128 == 136);
    CHECK(256 / 16 == 16 && 256 / 32 == 8);
    CHECK(find_mapping(QType::Q4_K)->block_bytes == 144 && find_mapping(QType::Q3_K)->block_bytes == 110);
    CHECK(find_mapping(QType::Q2_0)->block_elems == 64 && find_mapping(QType::BF16)->block_bytes == 2);
    // registry geometry matches the unit-1 dequantiser constants
    CHECK(find_mapping(QType::Q5_0)->block_bytes == dequant::kQ5_0Bytes && find_mapping(QType::Q6_K)->block_bytes == dequant::kQ6KBytes);
    CHECK(find_mapping(QType::IQ4_XS)->block_bytes == dequant::kIq4XsBytes && find_mapping(QType::Q5_K)->block_bytes == dequant::kQ5KBytes);
}

void test_canon_integer_facts() {
    for (int s = -128; s <= 127; ++s) { CHECK(s + 128 >= 0 && s + 128 <= 255); CHECK((s + 128) - 128 == s); }
    for (int c = 0; c <= 15; ++c) CHECK(c - 8 >= -8 && c - 8 <= 7);
    for (int c = 0; c <= 31; ++c) CHECK(c - 16 >= -16 && c - 16 <= 15);
    for (int c = 0; c <= 63; ++c) CHECK(c - 32 >= -32 && c - 32 <= 31);
    for (int b = 0; b < 4; ++b) for (int m = 0; m < 2; ++m) { const int c = b + 4 * m; CHECK(c <= 7 && c - 4 >= -4 && c - 4 <= 3); }
    for (unsigned q = 0; q < 256; ++q) CHECK(q % 16 + 16 * (q / 16) == q && q % 16 < 16 && q / 16 < 16);
    for (unsigned n = 0; n < 16; ++n) {
        for (unsigned b = 0; b < 2; ++b) CHECK(n + 16 * b < 32);
        for (unsigned h = 0; h < 4; ++h) CHECK(n + 16 * h < 64);
    }
    { const int lo[4] = {1, 4, 16, 64}, hi[4] = {2, 8, 32, 128};
      for (int k = 0; k < 4; ++k) CHECK((1 << (2 * k)) == lo[k] && (2 << (2 * k)) == hi[k]); }
    for (int j = 0; j < 4; ++j) for (int l = 0; l < 32; ++l) CHECK((64 * j + l) / 32 == 2 * j && (64 * j + 32 + l) / 32 == 2 * j + 1);
    for (int h = 0; h < 2; ++h) for (int j = 0; j < 4; ++j) for (int s = 0; s < 2; ++s) for (int l = 0; l < 16; ++l)
        CHECK((128 * h + 32 * j + 16 * s + l) / 16 == 8 * h + 2 * j + s);
    for (int h = 0; h < 2; ++h) for (int q = 0; q < 4; ++q) for (int l = 0; l < 32; ++l)
        CHECK((128 * h + 32 * q + l) / 16 == 8 * h + 2 * q + l / 16);
    for (int ib = 0; ib < 8; ++ib) CHECK(ib / 2 < 4 && ib % 2 < 2);
    for (int k = 0; k < 64; ++k) for (int t = 0; t < 4; ++t) CHECK((4 * k + t) / 4 == k && (4 * k + t) % 4 == t);
    CHECK(code_bias(QType::Q2_0) == 1 && code_bias(QType::Q3_K) == 4 && code_bias(QType::Q4_0) == 8 &&
          code_bias(QType::Q5_0) == 16 && code_bias(QType::Q6_K) == 32 && code_bias(QType::Q8_0) == 128);
}

void test_canon_bit_exact_roundtrip() {
    struct Row { QType t; int max_code; int lo, hi; int n; };
    const Row rows[] = {
        {QType::Q2_0, 3, -1, 2, 20000}, {QType::Q4_0, 15, -8, 7, 20000}, {QType::Q5_0, 31, -16, 15, 20000},
        {QType::Q8_0, 255, -128, 127, 20000}, {QType::IQ4_NL, 15, 0, 15, 20000}, {QType::Q3_K, 7, -4, 3, 6000},
        {QType::Q4_K, 15, 0, 15, 6000}, {QType::Q5_K, 31, 0, 31, 6000}, {QType::Q6_K, 63, -32, 31, 6000},
        {QType::IQ4_XS, 15, 0, 15, 6000}};
    Rng r(20);
    for (const Row& row : rows) {
        std::vector<uint8_t> b;
        for (int it = 0; it < row.n; ++it) {
            rand_block(r, row.t, b);
            Block c;
            CHECK(encode_block(row.t, b.data(), &c));
            const Mapping* m = find_mapping(row.t);
            CHECK(c.elems == (int)m->block_elems && c.groups * (int)m->group_elems == c.elems);
            for (int i = 0; i < c.elems; ++i) {
                CHECK(c.codes[i] <= row.max_code);
                if (!m->has_offset && !m->codebook) { const int v = (int)c.codes[i] - m->code_bias; CHECK(v >= row.lo && v <= row.hi); }
            }
            float ref[256], got[256];
            CHECK(reference_decode(row.t, b.data(), ref));
            CHECK(decode_block(row.t, c, got));
            CHECK_MSG(std::memcmp(ref, got, c.elems * sizeof(float)) == 0, "type %s iter %d", m->name, it);
        }
    }
    // unsupported / float types are not block-encoded
    Block c; uint8_t z[8] = {0};
    CHECK(!encode_block(QType::F32, z, &c));
}

void test_canon_q2_raw_roundtrip() {
    Rng r(21);
    for (int it = 0; it < 30000; ++it) {
        std::vector<uint8_t> b;
        rand_block(r, QType::Q2_0, b);
        Block c; encode_q2(b.data(), &c);
        uint8_t back[20]; std::memset(back, 0x5A, sizeof back);
        decode_q2(c, back);
        CHECK(std::memcmp(back, b.data(), 18) == 0);
        CHECK(back[18] == 0x5A && back[19] == 0x5A);  // exactly 18 bytes written
        CHECK(c.elems == 64 && c.groups == 1);
        for (int j = 0; j < 64; ++j) CHECK(c.codes[j] <= 3);  // 64 codes per block, unsigned
    }
    // exhaustive over one code byte: slot t of byte k lands at element 4k + t
    for (int byte = 0; byte < 256; ++byte) {
        uint8_t b[18] = {0, 0x3C /* fp16 1.0 */}; b[2 + 5] = (uint8_t)byte;
        Block c; encode_q2(b, &c);
        for (int t = 0; t < 4; ++t) CHECK(c.codes[4 * 5 + t] == ((byte >> (2 * t)) & 3));
        float y[64]; decode_block(QType::Q2_0, c, y);
        for (int t = 0; t < 4; ++t) CHECK(y[20 + t] == (float)(((byte >> (2 * t)) & 3) - 1));
    }
}

void test_canon_q8_and_scale_min() {
    Rng r(22);
    for (int it = 0; it < 5000; ++it) {
        std::vector<uint8_t> b;
        rand_block(r, QType::Q8_0, b);
        Block c; encode_block(QType::Q8_0, b.data(), &c);
        for (int j = 0; j < 32; ++j) CHECK((int)c.codes[j] - 128 == (int)(int8_t)b[2 + j]);
    }
    for (int it = 0; it < 2000; ++it) {
        uint8_t q[12]; for (auto& x : q) x = (uint8_t)r.u32();
        for (int j = 0; j < 8; ++j) {
            uint8_t d, m, d2, m2;
            scale_min_pair(j, q, &d, &m); dequant::get_scale_min_k4(j, q, &d2, &m2);
            CHECK(d == d2 && m == m2 && d < 64 && m < 64);
        }
    }
}

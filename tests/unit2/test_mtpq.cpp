#include <cmath>
#include <cstring>
#include <vector>
#include "baloon/mtp_expert_quantizer.hpp"
#include "check.hpp"
using namespace baloon;
using namespace baloon::mtpq;
using testh::Rng;

void test_mtpq_shapes_and_sizes() {
    CHECK(E == 512 && GU_OUT == 1280 && HID == 2560 && INTER == 640);
    CHECK(HID % 64 == 0 && INTER % 64 == 0 && HID % 32 == 0 && INTER % 32 == 0);
    CHECK(total_bytes(64, 18) == 707788800ull);
    CHECK(total_bytes(32, 18) == 1415577600ull);
    CHECK(total_bytes(32, 34) == 2673868800ull);
    CHECK(total_bytes(64, 18) < total_bytes(32, 18) && total_bytes(32, 18) < total_bytes(32, 34));
    CHECK(E * GU_OUT * HID * 4 == 6710886400ull);
    CHECK(row_bytes(HID, 64, 18) == 720 && row_bytes(INTER, 64, 18) == 180);
    for (const auto& f : kFormats) CHECK(HID % f.block == 0 && INTER % f.block == 0 && total_bytes(f.block, f.bytes) == total_bytes(format_info(f.f).block, format_info(f.f).bytes));
    CHECK(kExpertTensors[0].shape[0] == 512 && kExpertTensors[0].shape[1] == 1280 && kExpertTensors[0].shape[2] == 2560);
    CHECK(kExpertTensors[1].shape[0] == 512 && kExpertTensors[1].shape[1] == 2560 && kExpertTensors[1].shape[2] == 640);
    // inner dim must divide
    std::vector<uint8_t> out; std::vector<float> w(200, 1.f);
    CHECK(!quantize_tensor(Format::Q2_0, w.data(), 2, 100, &out));
    CHECK(!quantize_tensor(Format::Q4_0, w.data(), 2, 100 - 4 + 1, &out));
    CHECK(quantize_tensor(Format::Q8_0, w.data(), 2, 96, &out) && out.size() == 2 * 3 * 34);
    CHECK(quantize_tensor(Format::Q4_0, w.data(), 4, 32, &out) && out.size() == 4 * 18);
    CHECK(quantize_tensor(Format::Q2_0, w.data(), 3, 64, &out) && out.size() == 3 * 18);
}

void test_mtpq_scale_search() {
    CHECK(kScaleCandidates == 17);
    CHECK(scale_factor(0) == 0.5f && scale_factor(16) == 1.0f && scale_factor(8) == 0.75f);
    for (int k = 0; k <= 16; ++k) CHECK(16 + k >= 16 && 16 + k <= 32);
    Rng r(50);
    for (int it = 0; it < 6000; ++it) {
        float x[64];
        const int kind = it % 5;
        for (auto& v : x) v = kind == 0 ? (float)r.gauss() : kind == 1 ? (float)(r.gauss() * r.gauss()) : kind == 2 ? (float)(r.uni() * 2 - 1) : kind == 3 ? (float)((int)r.below(5) - 2) : (r.below(16) == 0 ? 9.f : (float)(r.gauss() * 0.05));
        uint8_t blk[20]; std::memset(blk, 0x42, sizeof blk);
        quantize_q2(x, blk);
        CHECK(blk[18] == 0x42 && blk[19] == 0x42);
        uint8_t codes[64]; q2w::unpack_codes(blk, codes);
        const uint16_t hb = (uint16_t)(blk[0] | (blk[1] << 8));
        const float d = f16::half_to_float(hb);
        for (int j = 0; j < 64; ++j) CHECK(codes[j] <= 3);  // clip to [-1,2] then +1
        float amax = 0; for (float v : x) amax = std::fmax(amax, std::fabs(v));
        // chosen scale is one of the 17 candidates
        bool is_cand = false; double best = -1; uint8_t tmp[64];
        for (int k = 0; k < kScaleCandidates; ++k) {
            const uint16_t h = f16::float_to_half(amax * scale_factor(k));
            if (h == hb) is_cand = true;
            const double e = q2_codes_and_error(x, f16::half_to_float(h), tmp);
            if (best < 0 || e < best) best = e;
        }
        CHECK(is_cand);
        // and it attains the minimum error over the candidates, hence is never worse than d = amax
        uint8_t tc[64]; const double chosen = q2_codes_and_error(x, d, tc);
        CHECK(chosen <= best * (1 + 1e-12) + 1e-30);
        CHECK(std::memcmp(tc, codes, 64) == 0);  // stored codes are the codes for the stored scale
        const double base = q2_codes_and_error(x, f16::half_to_float(f16::float_to_half(amax)), tmp);
        CHECK(chosen <= base * (1 + 1e-12) + 1e-30);
        // decoded values lie on the grid d * {-1,0,1,2}; requantising on the same scale is a fixed point
        float y[64]; q2w::dequantize_block(blk, y);
        uint8_t again[64]; q2w::quantize_codes_with_scale(y, d, again);
        CHECK(std::memcmp(again, codes, 64) == 0);
    }
    float z[64] = {0}; uint8_t blk[18]; quantize_q2(z, blk);
    CHECK(blk[0] == 0 && blk[1] == 0);
    for (int k = 2; k < 18; ++k) CHECK(blk[k] == 0x55);
    // a vector on the exact grid {-1,0,1,2}*s is reproduced exactly
    float g[64]; for (int j = 0; j < 64; ++j) g[j] = 0.25f * (float)(((j * 7) % 4) - 1);
    g[0] = 0.5f;  // amax = 0.5 -> candidate f=1.0 gives d=0.5; f=0.5 gives d=0.25 and exact grid
    quantize_q2(g, blk);
    float y[64]; q2w::dequantize_block(blk, y);
    for (int j = 0; j < 64; ++j) CHECK(y[j] == g[j]);
}

void test_mtpq_q4() {
    CHECK(nibble(0.f) == 0 && nibble(15.9f) == 15 && nibble(16.5f) == 15 && nibble(1e9f) == 15 && nibble(7.99f) == 7);
    for (unsigned lo = 0; lo < 16; ++lo) for (unsigned hi = 0; hi < 16; ++hi) {
        const unsigned b = lo + 16 * hi;
        CHECK(b % 16 == lo && b / 16 == hi && b < 256);
    }
    for (int c = -8; c <= 7; ++c) CHECK(c + 8 >= 0 && c + 8 <= 15);
    // exact layout: element j low nibble of byte j, element j+16 high nibble
    float x[32];
    for (int j = 0; j < 32; ++j) x[j] = 0.5f * (float)((j % 16) - 8);
    uint8_t blk[20]; std::memset(blk, 0x99, sizeof blk);
    quantize_q4(x, blk);
    CHECK(blk[18] == 0x99 && blk[19] == 0x99);
    CHECK(f16::half_to_float((uint16_t)(blk[0] | (blk[1] << 8))) == 0.5f);
    for (int j = 0; j < 16; ++j) CHECK(blk[2 + j] == (uint8_t)(j | (j << 4)));
    float y[32]; dequantize_q4(blk, y);
    for (int j = 0; j < 32; ++j) CHECK(y[j] == x[j]);
    // distinct halves land in the right places
    for (int j = 0; j < 32; ++j) x[j] = (j < 16 ? -1.f : 1.f) * 0.125f * (float)(1 + (j % 16));
    quantize_q4(x, blk); dequantize_q4(blk, y);
    for (int j = 0; j < 32; ++j) CHECK(std::fabs(x[j] - y[j]) <= std::fabs(0.125f * 2) * 1.02f);
    Rng r(51);
    for (int it = 0; it < 20000; ++it) {
        for (auto& v : x) v = (float)(r.gauss() * (1 + it % 7));
        quantize_q4(x, blk); dequantize_q4(blk, y);
        const float d = f16::half_to_float((uint16_t)(blk[0] | (blk[1] << 8)));
        for (int j = 0; j < 32; ++j) CHECK_MSG(std::fabs(x[j] - y[j]) <= std::fabs(d) * 1.02f + 1e-6f, "x=%g y=%g d=%g", x[j], y[j], d);
        // the extreme element lands on code 0 (symbol -8)
        int im = 0; for (int j = 1; j < 32; ++j) if (std::fabs(x[j]) > std::fabs(x[im])) im = j;
        const unsigned code = im < 16 ? (blk[2 + im] & 0xF) : (blk[2 + im - 16] >> 4);
        CHECK(code == 0);
        CHECK(std::fabs(y[im] - x[im]) <= std::fabs(d) * 0.1f + 1e-6f);
    }
    float zero[32] = {0}; quantize_q4(zero, blk); dequantize_q4(blk, y);
    for (float v : y) CHECK(v == 0.f);
}

void test_mtpq_q8() {
    for (int c = -127; c <= 127; ++c) CHECK(c >= -128 && c <= 127);
    Rng r(52);
    float x[32], y[32]; uint8_t blk[40]; std::memset(blk, 0x31, sizeof blk);
    for (int it = 0; it < 20000; ++it) {
        for (auto& v : x) v = (float)(r.gauss() * (1 + it % 5) * (it % 11 == 0 ? 100 : 1));
        quantize_q8(x, blk);
        CHECK(blk[34] == 0x31);
        float amax = 0; for (float v : x) amax = std::fmax(amax, std::fabs(v));
        const float d = amax / 127.0f;
        dequantize_q8(blk, y);
        int maxq = 0;
        for (int j = 0; j < 32; ++j) {
            const int q = (int)(int8_t)blk[2 + j];
            CHECK(q >= -127 && q <= 127);
            maxq = std::max(maxq, std::abs(q));
            CHECK_MSG(std::fabs(x[j] - y[j]) <= 0.57f * d + 1e-30f, "x=%g y=%g d=%g", x[j], y[j], d);
        }
        CHECK(maxq == 127);  // the extreme element uses the full range
    }
    float z[32] = {0}; quantize_q8(z, blk); dequantize_q8(blk, y);
    for (float v : y) CHECK(v == 0.f);
}

void test_mtpq_tensor_and_checkpoint() {
    Rng r(53);
    // multi-row tensor matches per-block quantisation
    const uint64_t rows = 5, inner = 192;
    std::vector<float> w(rows * inner);
    for (auto& v : w) v = (float)r.gauss();
    for (Format f : {Format::Q2_0, Format::Q4_0, Format::Q8_0}) {
        std::vector<uint8_t> out;
        CHECK(quantize_tensor(f, w.data(), rows, inner, &out));
        const auto& fi = format_info(f);
        CHECK(out.size() == rows * row_bytes(inner, fi.block, fi.bytes));
        for (uint64_t rr = 0; rr < rows; ++rr) for (uint64_t b = 0; b < inner / fi.block; ++b) {
            uint8_t ref[40];
            const float* x = w.data() + rr * inner + b * fi.block;
            if (f == Format::Q2_0) quantize_q2(x, ref); else if (f == Format::Q4_0) quantize_q4(x, ref); else quantize_q8(x, ref);
            CHECK(std::memcmp(ref, out.data() + (rr * (inner / fi.block) + b) * fi.bytes, fi.bytes) == 0);
        }
    }
    // real expert-slice shapes: one expert of gate_up and down at Q2_0
    std::vector<float> gu((size_t)(GU_OUT * HID)), dn((size_t)(HID * INTER));
    for (auto& v : gu) v = (float)(r.gauss() * 0.02);
    for (auto& v : dn) v = (float)(r.gauss() * 0.02);
    std::vector<uint8_t> a, b;
    CHECK(quantize_tensor(Format::Q2_0, gu.data(), GU_OUT, HID, &a) && a.size() == GU_OUT * 720);
    CHECK(quantize_tensor(Format::Q2_0, dn.data(), HID, INTER, &b) && b.size() == HID * 180);
    CHECK(a.size() + b.size() == total_bytes(64, 18) / E);

    // BF16 checkpoint reading
    std::vector<uint8_t> raw(2 * 65536);
    for (uint32_t h = 0; h < 65536; ++h) { raw[2 * h] = (uint8_t)(h & 0xFF); raw[2 * h + 1] = (uint8_t)(h >> 8); }
    std::vector<float> f;
    CHECK(!read_bf16(raw.data(), 65536, 65535, &f));  // size mismatch rejected
    CHECK(!read_bf16(raw.data(), 65535, 65536, &f));
    CHECK(read_bf16(raw.data(), 65536, 65536, &f) && f.size() == 65536);
    for (uint32_t h = 0; h < 65536 && f.size() == 65536; ++h) {
        uint32_t bits; std::memcpy(&bits, &f[h], 4);
        CHECK(bits == (h << 16) && bits / 65536 == h);
    }
    CHECK(size_matches(7, 7) && !size_matches(7, 8));
}

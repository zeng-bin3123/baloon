#include <cmath>
#include <cstring>
#include <vector>
#include "baloon/iquant_avx512_rows.hpp"
#include "check.hpp"

using namespace baloon::iquant512;
namespace f16 = baloon::f16;

void test_iq_formats() {
    CHECK(QK_K == 256);
    for (uint32_t t = 0; t < 64; ++t) {
        CHECK(supported(t) == (bytes_of(t) != 0));
        const bool want = t == 16 || t == 17 || t == 18 || t == 21 || t == 22;
        CHECK(supported(t) == want);
    }
    CHECK(!supported(29));  // IQ1_M stays on ggml-cpu
    CHECK(bytes_of(16) == 66 && bytes_of(17) == 74 && bytes_of(18) == 98 && bytes_of(21) == 110 && bytes_of(22) == 82);
    CHECK(sizeof(Q8K) == 292);
}

void test_iq_layout_tiles() {
    const uint32_t types[5] = {16, 17, 18, 21, 22};
    for (uint32_t t : types) {
        const uint32_t size = bytes_of(t);
        std::vector<int> cover(size, 0);
        cover[0]++; cover[1]++;  // fp16 d
        for (uint32_t j = 0; j < 4; ++j) {
            const ChunkFields cf = chunk_fields(t, j);
            for (uint32_t k = 0; k < F_COUNT; ++k) {
                const Field f = cf.f[k];
                if (f.len == 0) continue;
                CHECK_MSG(f.offset + f.len <= size, "type %u chunk %u field %u", t, j, k);
                for (uint32_t i = 0; i < f.len && f.offset + i < size; ++i) cover[f.offset + i]++;
            }
        }
        // every byte of the block is read by exactly one field (fields tile the block)
        for (uint32_t i = 0; i < size; ++i) CHECK_MSG(cover[i] == 1, "type %u byte %u covered %d times", t, i, cover[i]);
    }
    // unsupported type reads nothing
    const ChunkFields none = chunk_fields(29, 0);
    for (uint32_t k = 0; k < F_COUNT; ++k) CHECK(none.f[k].len == 0);
    for (uint32_t j = 0; j < 4; ++j) CHECK(64 * j + 64 <= QK_K);
    CHECK(4 * 64 == QK_K);
}

void test_iq_indices_scales() {
    for (uint32_t q = 0; q < 256; ++q) {
        for (uint32_t h = 0; h < 4; ++h) CHECK(iq2s_index(q, h) < 1024);
        for (uint32_t b = 0; b < 2; ++b) CHECK(iq3s_index(q, b) < 512);
    }
    CHECK(iq2s_index(255, 3) == 1023 && iq3s_index(255, 1) == 511);
    for (uint32_t v = 0; v < 65536; ++v) {
        CHECK(iq2xs_grid_index(v) < 512);
        CHECK(iq2xs_sign_index(v) < 128);
        CHECK(iq2xs_grid_index(v) + 512 * iq2xs_sign_index(v) == v);
    }
    for (uint32_t l = 0; l < 32; ++l) {
        CHECK(scale_of_lane(l) < 4);
        CHECK((2 * l) / 16 == scale_of_lane(l) && (2 * l + 1) / 16 == scale_of_lane(l));
    }
    CHECK(32 * 2 == 64);
    for (uint32_t s = 0; s < 16; ++s) { CHECK(odd_scale(s) <= 31); CHECK(odd_scale(s) % 2 == 1); }
    for (int y = -127; y <= 127; ++y) {
        CHECK(apply_sign(false, y) == y && apply_sign(true, y) == -y);
        CHECK(std::abs(apply_sign(true, y)) == std::abs(y));
        CHECK(apply_sign(true, y) >= -127 && apply_sign(true, y) <= 127);
    }
}

void test_iq_saturation() {
    // grid magnitudes <= 43: every unsigned pair against every int8 pair stays inside int16
    const int8_t bs[] = {-127, -126, -64, -1, 0, 1, 64, 126, 127};
    for (int a0 = 0; a0 <= 43; ++a0)
        for (int a1 = 0; a1 <= 43; ++a1)
            for (int8_t b0 : bs) for (int8_t b1 : bs)
                CHECK(!maddubs_would_saturate((uint8_t)a0, b0, (uint8_t)a1, b1));
    testh::Rng rng(5);
    for (int it = 0; it < 300000; ++it) {
        const uint8_t a0 = (uint8_t)rng.below(44), a1 = (uint8_t)rng.below(44);
        const int8_t b0 = (int8_t)((int)rng.below(255) - 127), b1 = (int8_t)((int)rng.below(255) - 127);
        CHECK(!maddubs_would_saturate(a0, b0, a1, b1));
    }
    CHECK(2 * 43 * 127 <= 32767);
    // the detector is not vacuous: larger magnitudes do saturate
    CHECK(maddubs_would_saturate(130, 127, 130, 127));
    CHECK(maddubs_would_saturate(255, -127, 255, -127));
}

namespace {
struct SynthFmt {
    static constexpr uint32_t BYTES = 80;
    // Deterministic decoder derived from block bytes.
    static void decode(const uint8_t* blk, uint32_t j, ChunkDecode& cd) {
        for (uint32_t i = 0; i < 64; ++i) cd.mag[i] = (uint8_t)(blk[2 + (16 * j + i / 4) % 60] * (i + 3) % 44);
        uint64_t s = 0;
        for (uint32_t k = 0; k < 8; ++k) s |= (uint64_t)blk[2 + (8 * j + k) % 60 + 1] << (8 * k);
        cd.signs = s;
        for (uint32_t g = 0; g < 4; ++g) cd.scale[g] = odd_scale((blk[70 + j] >> (g % 2 ? 4 : 0)) & 15);
    }
};
}  // namespace

void test_iq_row_dot() {
    testh::Rng rng(61);
    // q8k quantiser invariants
    {
        std::vector<float> x(512);
        for (auto& v : x) v = (float)rng.gauss();
        Q8K a[2];
        q8k_quantize(x.data(), 512, a);
        for (auto& blk : a) {
            for (int g = 0; g < 16; ++g) {
                int s = 0;
                for (int i = 0; i < 16; ++i) { CHECK(blk.qs[16 * g + i] >= -127 && blk.qs[16 * g + i] <= 127); s += blk.qs[16 * g + i]; }
                CHECK(blk.bsums[g] == s);
            }
        }
    }
    for (int it = 0; it < 30; ++it) {
        const uint32_t nb = 1 + rng.below(5);
        std::vector<uint8_t> row(nb * SynthFmt::BYTES);
        for (auto& v : row) v = (uint8_t)rng.below(256);
        for (uint32_t b = 0; b < nb; ++b) {
            const uint16_t h = f16::float_to_half((float)(0.01 + 0.05 * rng.uni()));
            std::memcpy(&row[b * SynthFmt::BYTES], &h, 2);
        }
        std::vector<float> x(nb * 256);
        for (auto& v : x) v = (float)rng.gauss();
        std::vector<Q8K> act(nb);
        q8k_quantize(x.data(), nb * 256, act.data());
        const float got = row_dot(row.data(), nb, SynthFmt::BYTES, act.data(), SynthFmt::decode);
        // reference: dequantise every weight explicitly, then dot with the dequantised activations
        double ref = 0, mag = 0;
        for (uint32_t b = 0; b < nb; ++b) {
            const uint8_t* blk = &row[b * SynthFmt::BYTES];
            uint16_t h; std::memcpy(&h, blk, 2);
            const double d = f16::half_to_float(h);
            for (uint32_t j = 0; j < 4; ++j) {
                ChunkDecode cd; SynthFmt::decode(blk, j, cd);
                for (uint32_t i = 0; i < 64; ++i) {
                    const double w = d * cd.scale[i / 16] * (((cd.signs >> i) & 1) ? -1.0 : 1.0) * cd.mag[i];
                    const double a = (double)act[b].d * act[b].qs[64 * j + i];
                    ref += w * a; mag += std::fabs(w * a);
                }
            }
        }
        CHECK_MSG(std::fabs((double)got - ref) <= 1e-5 * (mag + 1e-3), "got=%g ref=%g", (double)got, ref);
    }
}

void test_iq_batching() {
    for (uint32_t nt = 1; nt <= 40; ++nt) {
        uint32_t covered = 0;
        for (uint32_t t0 = 0; t0 < nt; t0 += batch_size(nt, t0)) {
            const uint32_t b = batch_size(nt, t0);
            CHECK(b >= 1 && b <= 4 && t0 + b <= nt);
            covered += b;
        }
        CHECK(covered == nt);
    }
    for (uint64_t base : {0ull, 4096ull}) for (uint64_t rb : {66ull, 110ull, 440ull}) for (uint64_t r = 0; r < 50; ++r)
        CHECK(row_offset(base, rb, r + 1) == row_offset(base, rb, r) + rb);
    CHECK(n_blocks(2560) == 10 && n_blocks(640) == 2);

    // rows_multi equals independent per-token row_dot, for token counts around the batch size
    testh::Rng rng(71);
    const uint32_t nb = 3, BY = SynthFmt::BYTES;
    std::vector<uint8_t> row(nb * BY);
    for (auto& v : row) v = (uint8_t)rng.below(256);
    for (uint32_t b = 0; b < nb; ++b) { uint16_t h = f16::float_to_half(0.03f); std::memcpy(&row[b * BY], &h, 2); }
    for (uint32_t nt : {1u, 2u, 4u, 5u, 8u, 9u, 11u}) {
        std::vector<float> x((size_t)nt * nb * 256);
        for (auto& v : x) v = (float)rng.gauss();
        std::vector<Q8K> act((size_t)nt * nb);
        for (uint32_t t = 0; t < nt; ++t) q8k_quantize(x.data() + (size_t)t * nb * 256, nb * 256, act.data() + (size_t)t * nb);
        std::vector<float> out(nt, -1.f);
        rows_multi(row.data(), nb, BY, act.data(), nt, out.data(), SynthFmt::decode);
        for (uint32_t t = 0; t < nt; ++t) {
            const float one = row_dot(row.data(), nb, BY, act.data() + (size_t)t * nb, SynthFmt::decode);
            CHECK(std::memcmp(&one, &out[t], 4) == 0);
        }
    }
}

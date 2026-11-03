#include <vector>
#include <cstring>
#include "baloon/prefill_gpu_kernels.hpp"
#include "check.hpp"
using namespace baloon::prefill_kernels;

void test_pk_constants() {
    CHECK(kD == 10240 && kC == 2 * kHK * kS + kHV * kS && kZV == 6144 && kQFullWidth == 12288);
    CHECK(gdn_conv_blocks() * kGdnConvThreads == kC);            // exactly one thread per channel
    CHECK(kGdnL2Blocks * kS == 4096);                            // q/k channels
    CHECK(BLOB_BYTES == 1382400);
    CHECK(O_D_CODES < O_GU_SC && O_GU_SC < O_D_SC && O_D_SC < BLOB_BYTES);
    CHECK(O_GU_CODES == 0 && O_D_CODES == 1280 * 640 && O_GU_SC - O_D_CODES == 2560 * 160 &&
          O_D_SC - O_GU_SC == 1280 * 40 * 2 && BLOB_BYTES - O_D_SC == 2560 * 10 * 2);
}

void test_pk_geometry() {
    for (uint64_t t : {1, 7, 32, 64, 128, 256, 1024}) for (uint64_t n = 0; n < 3000; n += (n < 300 ? 1 : 37)) {
        const uint64_t b = blocks_for(n, t);
        CHECK(n <= b * t && b * t < n + t);
    }
    for (uint64_t T = 0; T < 500; ++T) CHECK(T <= route_blocks(T) * kRouteWarps && route_blocks(T) * kRouteWarps < T + 8);
    for (uint64_t b = 1; b <= 1024; ++b) CHECK(block_sum_warps(b) <= kBlockSumMaxWarps);
    CHECK(block_sum_warps(1025) > kBlockSumMaxWarps);            // guards the bound: 1025 threads would overflow sh[32]
    for (uint64_t tid = 0; tid < 2048; ++tid) CHECK(warp_of(tid) * 32 + lane_of(tid) == tid && lane_of(tid) < 32);
    // recurrence rows partition into 4 groups of 32, state index is a bijection onto S*HV*S
    std::vector<uint8_t> hit(kGdnStateElems, 0);
    for (uint64_t row = 0; row < kS; ++row) {
        CHECK(rec_group(row) < 4 && rec_row_in_group(row) < 32 && rec_group(row) * 32 + rec_row_in_group(row) == row);
        for (uint64_t h = 0; h < kHV; ++h) for (uint64_t c = 0; c < kS; ++c) {
            const uint64_t i = gdn_state_index(row, h, c);
            CHECK(i < kGdnStateElems);
            if (i < kGdnStateElems) hit[i]++;
        }
    }
    for (uint8_t v : hit) CHECK(v == 1);
    // rope pairs: involution, covers all 64 dims
    std::vector<int> seen(kRopeDims, 0);
    for (uint64_t p = 0; p < kRopePairs; ++p) { seen[p]++; seen[rope_partner(p)]++; CHECK(rope_partner(rope_partner(p)) == p); }
    for (int v : seen) CHECK(v == 1);
}

void test_pk_index_maps() {
    testh::Rng rng(81);
    for (int it = 0; it < 20000; ++it) {
        const uint64_t T = 1 + rng.below(64), t = rng.below((uint32_t)T), c = rng.below(4), d = rng.below(2560);
        const uint64_t idx = hc_index(t, c, d);
        CHECK(hc_token(idx) == t && hc_stream(idx) == c && hc_channel(idx) == d && idx < T * kD);
        const uint64_t k = rng.below(10);
        CHECK(combine_slot(t, k) < T * kRoutedSlots);
        const uint64_t r = rng.below(500), kk = rng.below(640);
        CHECK(swiglu_flat(r, kk) / 640 == r && swiglu_flat(r, kk) % 640 == kk);
        CHECK(swiglu_il_up(r, kk) < (r + 1) * 1280 && swiglu_il_gate(r, kk) + 1 == swiglu_il_up(r, kk));
        const uint64_t per = 1 + rng.below(64), n = r + 1 + rng.below(8), j = rng.below((uint32_t)per);
        CHECK(gather_index(r, j, per) < n * per && row_index(r, per, j) == gather_index(r, j, per));
        const uint64_t np = 1 + rng.below(16), pg = rng.below((uint32_t)np), kvh = rng.below(2), ps = 1 + rng.below(128), sl = rng.below((uint32_t)ps);
        CHECK(kv_row(pg, kvh, ps, sl) < kv_pool_rows(np, ps));
    }
    // swiglu interleave: every cell of an n x 1280 buffer is hit exactly once by (gate,up) pairs
    const uint64_t n = 5;
    std::vector<int> hit(n * 1280, 0);
    for (uint64_t r = 0; r < n; ++r) for (uint64_t k = 0; k < 640; ++k) { hit[swiglu_il_gate(r, k)]++; hit[swiglu_il_up(r, k)]++; }
    for (int v : hit) CHECK(v == 1);
}

void test_pk_blob() {
    for (uint32_t c = 0; c < 256; ++c) for (uint32_t k = 0; k < 4; ++k) {
        CHECK(code2(c, k) <= 3 && dequant_val(c, k) >= -1 && dequant_val(c, k) <= 2);
        CHECK((int)code2(c, k) == (int)((c >> (2 * k)) & 3));
    }
    // the 4 codes of a byte reassemble the byte
    for (uint32_t c = 0; c < 256; ++c) CHECK((code2(c, 0) | code2(c, 1) << 2 | code2(c, 2) << 4 | code2(c, 3) << 6) == c);
    // gate/up: outputs are a bijection onto 1280 x 2560; scale index is in range and constant over each 64-group
    std::vector<uint8_t> hit(kGuRows * kGuCols, 0);
    for (uint64_t row = 0; row < kGuRows; ++row) for (uint64_t b = 0; b < kGuCodeBytesPerRow; ++b) {
        CHECK(gu_scale_index(row, b) < kGuRows * kGuScalesPerRow);
        CHECK(gu_code_offset(row, b) < O_D_CODES && gu_scale_offset(row, b) + 2 <= O_D_SC);
        for (uint64_t k = 0; k < 4; ++k) {
            const uint64_t o = gu_out_index(row, b, k);
            CHECK(o < kGuRows * kGuCols);
            if (o < hit.size()) hit[o]++;
            CHECK(gu_scale_index(row, b) == row * 40 + (b * 4 + k) / 64);   // all 4 weights of a byte share their group scale
        }
    }
    for (uint8_t v : hit) CHECK(v == 1);
    std::vector<uint8_t> hd(kDnRows * kDnCols, 0);
    for (uint64_t row = 0; row < kDnRows; ++row) for (uint64_t b = 0; b < kDnCodeBytesPerRow; ++b) {
        CHECK(down_scale_index(row, b) < kDnRows * kDnScalesPerRow);
        CHECK(down_code_offset(row, b) >= O_D_CODES && down_code_offset(row, b) < O_GU_SC);
        CHECK(down_scale_offset(row, b) >= O_D_SC && down_scale_offset(row, b) + 2 <= BLOB_BYTES);
        for (uint64_t k = 0; k < 4; ++k) {
            const uint64_t o = down_out_index(row, b, k);
            CHECK(o < kDnRows * kDnCols);
            if (o < hd.size()) hd[o]++;
        }
    }
    for (uint8_t v : hd) CHECK(v == 1);
    // the code bytes and scales tile the blob: every byte offset belongs to exactly one region
    std::vector<uint8_t> cover(BLOB_BYTES, 0);
    for (uint64_t row = 0; row < kGuRows; ++row) for (uint64_t b = 0; b < kGuCodeBytesPerRow; ++b) cover[gu_code_offset(row, b)]++;
    for (uint64_t row = 0; row < kDnRows; ++row) for (uint64_t b = 0; b < kDnCodeBytesPerRow; ++b) cover[down_code_offset(row, b)]++;
    for (uint64_t i = 0; i < kGuRows * kGuScalesPerRow; ++i) { cover[O_GU_SC + 2 * i]++; cover[O_GU_SC + 2 * i + 1]++; }
    for (uint64_t i = 0; i < kDnRows * kDnScalesPerRow; ++i) { cover[O_D_SC + 2 * i]++; cover[O_D_SC + 2 * i + 1]++; }
    for (uint8_t v : cover) CHECK(v == 1);
    // end-to-end reference dequant on a random blob
    testh::Rng rng(82);
    std::vector<uint8_t> blob(BLOB_BYTES);
    for (auto& x : blob) x = (uint8_t)rng.u32();
    for (int it = 0; it < 2000; ++it) {
        const uint64_t row = rng.below(1280), col = rng.below(2560);
        uint16_t sc; std::memcpy(&sc, &blob[O_GU_SC + (row * 40 + col / 64) * 2], 2);
        const float want = (float)(((blob[row * 640 + col / 4] >> (2 * (col % 4))) & 3) - 1) * f16_bits_to_f32(sc);
        const float got = gu_weight(blob.data(), row, col);
        CHECK(got == want || (got != got && want != want));
        const uint64_t r2 = rng.below(2560), c2 = rng.below(640);
        std::memcpy(&sc, &blob[O_D_SC + (r2 * 10 + c2 / 64) * 2], 2);
        const float w2 = (float)(((blob[O_D_CODES + r2 * 160 + c2 / 4] >> (2 * (c2 % 4))) & 3) - 1) * f16_bits_to_f32(sc);
        const float g2 = down_weight(blob.data(), r2, c2);
        CHECK(g2 == w2 || (g2 != g2 && w2 != w2));
    }
    CHECK(f16_bits_to_f32(0x3C00) == 1.0f && f16_bits_to_f32(0xC000) == -2.0f && f16_bits_to_f32(0x0000) == 0.0f &&
          f16_bits_to_f32(0x0001) == 5.9604645e-8f && f16_bits_to_f32(0x7BFF) == 65504.0f);
}

void test_pk_numeric() {
    testh::Rng rng(83);
    // bf16 result fits 16 bits; exact on bf16 values (low 16 bits zero)
    for (int it = 0; it < 200000; ++it) {
        const uint32_t u = (uint32_t)rng.next();
        CHECK(bf_bits(u) < 65536u);
    }
    for (uint32_t h = 0; h < 65536; ++h) CHECK(bf_bits(h << 16) == h);
    CHECK(bf_bits(0xFFFFFFFFu) < 65536u);                         // uint32 wrap stays 16-bit
    // round to nearest even on the tie, up above the tie, down below it
    CHECK(bf_bits(0x3F808000u) == 0x3F80u);                       // tie, lsb 0 -> stay
    CHECK(bf_bits(0x3F818000u) == 0x3F82u);                       // tie, lsb 1 -> up
    CHECK(bf_bits(0x3F808001u) == 0x3F81u);                       // above tie -> up
    CHECK(bf_bits(0x3F807FFFu) == 0x3F80u);                       // below tie -> down
    // agrees with float round trip: |x - round(x)| <= half an ulp of bf16 for normal values
    for (int it = 0; it < 100000; ++it) {
        const float x = (float)(rng.gauss() * 100.0);
        const float y = bf_to_float(bf_from_float(x));
        uint32_t ux; std::memcpy(&ux, &x, 4);
        CHECK(std::fabs(x - y) <= std::ldexp(1.0f, (int)((ux >> 23) & 0xFF) - 127 - 8) * 1.0000001f);
        CHECK(bf_from_float(y) == bf_from_float(x));
    }
    // clamp127
    for (int64_t q = -1000; q <= 1000; ++q) {
        const int32_t c = clamp127(q);
        CHECK(c >= -127 && c <= 127);
        if (q >= -127 && q <= 127) CHECK(c == q);
        if (q < -127) CHECK(c == -127);
        if (q > 127) CHECK(c == 127);
    }
    CHECK(clamp127(INT64_MIN) == -127 && clamp127(INT64_MAX) == 127);
    for (int it = 0; it < 20000; ++it) { const int32_t q = kv_quant((float)(rng.gauss() * 1e4), 3.7f); CHECK(q >= -127 && q <= 127); }
    CHECK(kv_quant(1.0f, 127.0f) == 127 && kv_quant(-1.0f, 127.0f) == -127 && kv_quant(0.0f, 127.0f) == 0);
}

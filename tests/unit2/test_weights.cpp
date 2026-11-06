#include <cstring>
#include <vector>
#include "baloon/weight_loader_segments.hpp"
#include "baloon/f16_conversion.hpp"
#include "check.hpp"
using namespace baloon::weights;
using testh::Rng;

void test_weights_scalar_theorems() {
    CHECK(kChunk == 8u * 1024 * 1024);
    CHECK(kChunk % 4 == 0);
    for (uint64_t done = 0; done <= 200; done += 2)
        for (uint64_t n = 0; n <= 200; n += 2) {
            const uint64_t dstOff = 37;
            CHECK(dstOff + (done + n) * 2 == (dstOff + done * 2) + widen_out(n));  // widen_progress
        }
    for (uint64_t done = 0; done <= 200; done += 4)
        for (uint64_t n = 0; n <= 200; n += 4) {
            const uint64_t dstOff = 11;
            CHECK(dstOff + (done + n) / 2 == (dstOff + done / 2) + bf16_out(n));  // bf16_progress
        }
    Rng r(1);
    std::vector<uint64_t> ns = {0, 1, 2, 3, 4, kChunk - 1, kChunk};
    for (int i = 0; i < 2000; ++i) ns.push_back(r.next() % (kChunk + 1));
    for (uint64_t n : ns) {
        CHECK(widen_out(n) <= 2 * kChunk);  // widen_fits
        CHECK(bf16_out(n) <= 2 * kChunk);   // bf16_fits
    }
    CHECK(widen_out(kChunk) == 2 * kChunk);  // exactly fills stage_out
    for (uint64_t n = 2; n < 3000; ++n) {
        const uint64_t t = even_trim(n);
        CHECK(t % 2 == 0 && t > 0 && t <= n);
        CHECK(n - t <= 1);
    }
    for (uint64_t v : {0ull, 1ull, 65535ull, 65536ull, 0x7FC00000ull, 0xFFFFFFFFull}) CHECK(v / 65536 < 65536);
    for (uint64_t e : {0ull, 1ull, 7ull, 1000003ull}) {
        CHECK(promoted_src_bytes(e) == e * 4 && promoted_dst_bytes(e) == e * 2);
        CHECK(promoted_saving(e) == promoted_dst_bytes(e));
    }
}

void test_weights_widen_chunked() {
    Rng r(2);
    const uint64_t sizes[] = {0, 1, 2, 3, 4, 5, 63, 64, 65, 1000, 1001, 4097};
    const uint64_t chunks[] = {1, 4, 6, 8, 64, 100, 4096};
    for (uint64_t n : sizes)
        for (uint64_t ch : chunks) {
            std::vector<uint8_t> src(n);
            for (auto& b : src) b = (uint8_t)r.u32();
            const uint64_t cons_exp = even_trim(n), out_exp = widen_out(cons_exp);
            std::vector<uint8_t> dst(out_exp + 16, 0xAB);
            const ConvResult res = widen_f16_chunked(src.data(), n, dst.data(), ch);
            CHECK(res.consumed == cons_exp);
            CHECK(res.written == out_exp);
            CHECK(res.written == res.consumed * 2);  // dst progress is exactly 2 per source byte
            const uint64_t nc = norm_chunk(ch);
            CHECK(res.rounds == (cons_exp + nc - 1) / nc);
            for (uint64_t i = 0; i < cons_exp / 2; ++i) {
                const uint16_t h = (uint16_t)(src[2 * i] | (src[2 * i + 1] << 8));
                const uint32_t bits = baloon::f16::f32_bits(baloon::f16::half_to_float(h));
                uint32_t got; std::memcpy(&got, &dst[4 * i], 4);
                CHECK(got == bits);
            }
            for (size_t k = out_exp; k < dst.size(); ++k) CHECK(dst[k] == 0xAB);  // no overrun
        }
    // default (8 MiB) chunk path, with data spanning more than one round
    const uint64_t big = kChunk + 1002;
    std::vector<uint8_t> src(big);
    for (auto& b : src) b = (uint8_t)r.u32();
    std::vector<uint8_t> dst(widen_out(even_trim(big)));
    const ConvResult res = widen_f16_chunked(src.data(), big, dst.data());
    CHECK(res.rounds == 2 && res.consumed == big && res.written == big * 2);
    uint32_t last; std::memcpy(&last, &dst[dst.size() - 4], 4);
    const uint16_t h = (uint16_t)(src[big - 2] | (src[big - 1] << 8));
    CHECK(last == baloon::f16::f32_bits(baloon::f16::half_to_float(h)));
}

void test_weights_bf16_promote() {
    Rng r(3);
    const uint64_t sizes[] = {0, 1, 3, 4, 7, 8, 12, 100, 1003, 4096};
    const uint64_t chunks[] = {1, 4, 8, 12, 64, 4100};
    for (uint64_t n : sizes)
        for (uint64_t ch : chunks) {
            std::vector<uint8_t> src(n);
            for (auto& b : src) b = (uint8_t)r.u32();
            const uint64_t cons = quad_trim(n), out = bf16_out(cons);
            std::vector<uint8_t> dst(out + 8, 0xCD);
            const ConvResult res = promote_bf16_high_chunked(src.data(), n, dst.data(), ch);
            CHECK(res.consumed == cons && res.written == out);
            CHECK(res.written * 2 == res.consumed);
            for (uint64_t i = 0; i < cons / 4; ++i) {
                uint32_t w; std::memcpy(&w, &src[4 * i], 4);
                const uint16_t hi = (uint16_t)(w >> 16);
                CHECK(dst[2 * i] == (uint8_t)(hi & 0xFF) && dst[2 * i + 1] == (uint8_t)(hi >> 8));
            }
            for (size_t k = out; k < dst.size(); ++k) CHECK(dst[k] == 0xCD);
        }
    // BF16 -> f32 -> BF16 promotion round trip is exact on the high half
    for (uint32_t h = 0; h < 65536; ++h) {
        const uint32_t bits = h << 16;
        uint8_t s[4] = {0, 0, (uint8_t)(h & 0xFF), (uint8_t)(h >> 8)};
        uint8_t d[2];
        promote_bf16_high_chunked(s, 4, d, 4);
        CHECK((uint32_t)(d[0] | (d[1] << 8)) == bits >> 16);
    }
}

void test_weights_segments_and_arena() {
    Rng r(4);
    for (int i = 0; i < 500; ++i) {
        QuantSegments s;
        s.codes = r.below(100000);
        s.scales = 2 * r.below(5000);
        s.offs = r.below(5000);
        CHECK(s.dst_codes_start() == 0);
        CHECK(s.dst_scales_start() == s.codes);
        CHECK(s.dst_offs_start() + s.offs == s.dst_total());
        CHECK(s.src_scale_plane() * 2 == s.scales);
        CHECK(s.src_offs_start() == s.dst_offs_start());
    }
    for (int it = 0; it < 500; ++it) {
        std::vector<uint64_t> rows(r.below(24));
        for (auto& x : rows) x = r.below(6000);
        const uint64_t pool = arena_pool(rows);
        for (size_t i = 0; i <= rows.size() + 2; ++i) {
            CHECK(arena_off(rows, i) % 256 == 0);
            CHECK(arena_off(rows, i) <= pool);
        }
        for (size_t i = 0; i < rows.size(); ++i) {
            CHECK(arena_off(rows, i) + align256(rows[i]) <= pool);
            CHECK(arena_off(rows, i + 1) == arena_off(rows, i) + align256(rows[i]));
            CHECK(align256(rows[i]) >= rows[i] && align256(rows[i]) % 256 == 0 && align256(rows[i]) - rows[i] < 256);
        }
        std::vector<bool> skip(rows.size());
        for (size_t i = 0; i < skip.size(); ++i) skip[i] = r.below(3) == 0;
        const CompactedArena a = compact_arena(rows, skip);
        uint64_t expect = 0; int64_t prev_end = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
            if (skip[i]) { CHECK(a.offset[i] == CompactedArena::kSkipped); continue; }
            CHECK(a.offset[i] == (int64_t)expect);
            CHECK(a.offset[i] % 256 == 0 && a.offset[i] >= prev_end);  // index order, no overlap
            expect += align256(rows[i]);
            prev_end = (int64_t)expect;
            CHECK((uint64_t)a.offset[i] + align256(rows[i]) <= a.pool);
        }
        CHECK(a.pool == expect);
    }
    CHECK(align256(0) == 0 && align256(1) == 256 && align256(256) == 256 && align256(257) == 512);
}

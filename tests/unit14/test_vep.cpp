// Theory invariants of VisionEncoderProtocol (splitLast_correct, no_space_no_split, validEnc_iff, gridPos_in_range,
// gridPos_roundtrip, last_token_pos, grid_count, fileBytes_step/zero, header_bytes, magic).
#include <cstring>
#include <string>
#include <vector>
#include "baloon/vision_encoder_protocol.hpp"
#include "check.hpp"
using namespace baloon::vision_enc;

void test_vep_split_last() {
    // splitLast (rest.rfind(' ')): split at the LAST space; image may contain spaces, output may not.
    auto c = parse_enc("a b c.png out.bin");
    CHECK(c.ok && c.image == "a b c.png" && c.output == "out.bin");
    c = parse_enc("x  y");                       // double space: image keeps the first space
    CHECK(c.ok && c.image == "x " && c.output == "y");
    CHECK(!parse_enc("nospace").ok);             // no_space_no_split
    CHECK(!parse_enc("").ok);
    CHECK(!parse_enc(" ").ok);
    CHECK(!parse_enc("img ").ok);                // empty output
    CHECK(!parse_enc(" out").ok);                // empty image
    CHECK(!parse_enc("   ").ok);                 // image "  ", output "" -> rejected
}

void test_vep_split_random() {
    testh::Rng rng(2026);
    const char alpha[] = {'a', 'b', '/', '.', ' ', '\t', 'Q', '\xc3'};
    for (int it = 0; it < 30000; ++it) {
        std::string img, out;
        for (uint32_t i = 0, n = rng.below(14); i < n; ++i) img += alpha[rng.below(8)];
        for (uint32_t i = 0, n = rng.below(9); i < n; ++i) { char ch = alpha[rng.below(8)]; if (ch != ' ') out += ch; }
        auto c = parse_enc(img + " " + out);
        // splitLast_correct <-> validEnc_iff: ok exactly when both parts nonempty, parts recovered exactly
        CHECK(c.ok == (!img.empty() && !out.empty()));
        if (c.ok) CHECK(c.image == img && c.output == out);
        else CHECK(c.image.empty() && c.output.empty());
        if (out.find(' ') == std::string::npos) CHECK(!parse_enc(out).ok);   // space-free input: no split
    }
}

void test_vep_valid_enc() {
    for (uint64_t sp = 0; sp < 10; ++sp) for (uint64_t len = 0; len < 12; ++len) {
        const bool both = sp > 0 && len > sp + 1;   // image length sp, output length len-sp-1
        CHECK(accepts(sp, len) == both);
    }
}

void test_vep_grid_pos() {
    for (uint64_t nx = 1; nx < 40; ++nx) for (uint64_t ny = 1; ny < 40; ++ny) {
        for (uint64_t i = 0; i < nx * ny; ++i) {            // gridPos_in_range
            uint64_t x, y; row_pos(i, nx, x, y);
            CHECK(x < nx && y < ny);
        }
        for (uint64_t y = 0; y < ny; ++y) for (uint64_t x = 0; x < nx; ++x) {   // gridPos_roundtrip
            uint64_t rx, ry; row_pos(y * nx + x, nx, rx, ry);
            CHECK(rx == x && ry == y);
        }
        uint64_t lx, ly; row_pos((ny - 1) * nx + (nx - 1), nx, lx, ly);        // last_token_pos
        CHECK(lx == nx - 1 && ly == ny - 1);
    }
}

void test_vep_grid_derive() {
    for (int32_t nx = 1; nx < 40; ++nx) for (int32_t ny = 1; ny < 40; ++ny) {
        int32_t gx, gy; grid_from_last(nx - 1, ny - 1, gx, gy);
        CHECK(gx == nx && gy == ny);
        CHECK((uint64_t)(gx - 1 + 1) * (uint64_t)(gy - 1 + 1) == (uint64_t)nx * ny);   // grid_count
        CHECK(grid_ok((uint64_t)gx, (uint64_t)gy, (uint64_t)nx * ny));
        CHECK(!grid_ok((uint64_t)gx, (uint64_t)gy, (uint64_t)nx * ny + 1));
        if (nx * ny > 1) CHECK(!grid_ok((uint64_t)gx, (uint64_t)gy, (uint64_t)nx * ny - 1));
    }
}

void test_vep_file_size() {
    CHECK(HEADER_BYTES == 5 * 4);                                           // header_bytes
    for (uint64_t e = 0; e < 64; ++e) {
        CHECK(file_bytes(0, e) == 20);                                      // fileBytes_zero
        for (uint64_t n = 0; n < 64; ++n) {
            CHECK(file_bytes(n + 1, e) == file_bytes(n, e) + 4 * e);        // fileBytes_step
            CHECK(file_bytes(n, e) == 20 + 4 * (n * e));
        }
    }
}

void test_vep_magic_header() {
    CHECK(MAGIC == 0x53u + 0x56u * 256u + 0x45u * 65536u + 0x31u * 16777216u);   // 'S','V','E','1' little-endian
    CHECK(MAGIC < 2147483648u);                                                    // magic_int32
    auto h = encode_header(12, 4, 3, 7);
    CHECK(h.size() == HEADER_BYTES);
    CHECK(std::memcmp(h.data(), "SVE1", 4) == 0);
    Header d; CHECK(decode_header(h.data(), h.size(), d));
    CHECK(d.magic > 0 && d.n_tokens == 12 && d.nx == 4 && d.ny == 3 && d.n_embd == 7);
    for (size_t cut = 0; cut < HEADER_BYTES; ++cut) CHECK(!decode_header(h.data(), cut, d));
    for (int b = 0; b < 4; ++b) { auto g = h; g[b] ^= 1; CHECK(!decode_header(g.data(), g.size(), d)); }
}

void test_vep_file_roundtrip() {
    testh::Rng rng(5);
    for (int it = 0; it < 200; ++it) {
        const int32_t nx = 1 + (int32_t)rng.below(9), ny = 1 + (int32_t)rng.below(9), e = 1 + (int32_t)rng.below(12);
        std::vector<float> emb((size_t)nx * ny * e);
        for (auto& v : emb) v = (float)rng.gauss();
        auto f = build_file(nx, ny, e, emb);
        CHECK(f.size() == file_bytes((uint64_t)nx * ny, (uint64_t)e));
        CHECK(validate_file(f));
        Header h; CHECK(decode_header(f.data(), f.size(), h));
        CHECK(h.n_tokens == nx * ny && h.nx == nx && h.ny == ny && h.n_embd == e);
        CHECK(std::memcmp(f.data() + HEADER_BYTES, emb.data(), emb.size() * 4) == 0);
        auto t = f; t.pop_back(); CHECK(!validate_file(t));                 // truncated
        t = f; t.push_back(0); CHECK(!validate_file(t));                    // trailing byte
        t = f; put_i32(&t[16], 0); CHECK(!validate_file(t));                // zero width
        t = f; put_i32(&t[8], nx + 1); CHECK(!validate_file(t));            // grid != n
    }
    CHECK(!validate_file({}));
}

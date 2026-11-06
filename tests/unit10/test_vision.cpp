#include <cstring>
#include <string>
#include <vector>
#include "baloon/vision_build_config.hpp"
#include "baloon/vision_encoder_protocol.hpp"
#include "check.hpp"
using namespace baloon;

void test_vc_configure() {
    for (int m = 0; m < 8; ++m) {
        vision_cfg::Options o; o.portable = m & 1; o.cuda = m & 2; o.msvc = m & 4;
        auto s = vision_cfg::configure(o);
        CHECK(s.ggml_cuda == o.cuda);
        CHECK(s.shared_libs == false);   // static libraries, every option combination
        CHECK(s.backend_dl == false);    // ggml backends not dynamic, every option combination
        CHECK(s.static_runtime == (o.portable && o.msvc));
        CHECK(s.native == !s.avx2);
        if (o.portable) {
            CHECK(!s.native && s.openmp_forced && !s.openmp);
            CHECK(s.avx && s.avx2 && s.fma && s.f16c && s.bmi2);
        } else {
            CHECK(s.native && !s.openmp_forced && !s.avx && !s.avx2 && !s.fma && !s.f16c && !s.bmi2);
        }
    }
}
void test_vc_lists() {
    CHECK(vision_cfg::ALWAYS_OFF.size() == 10 && vision_cfg::ALWAYS_ON.size() == 1);
    for (auto on : vision_cfg::ALWAYS_ON) for (auto off : vision_cfg::ALWAYS_OFF) CHECK(std::strcmp(on, off) != 0);
    CHECK(std::string(vision_cfg::ALWAYS_ON[0]) == "LLAMA_BUILD_MTMD");
    bool has_shared = false, has_dl = false;
    for (auto off : vision_cfg::ALWAYS_OFF) {
        has_shared |= std::strcmp(off, "BUILD_SHARED_LIBS") == 0;
        has_dl |= std::strcmp(off, "GGML_BACKEND_DL") == 0;
    }
    CHECK(has_shared && has_dl);
}
void test_ve_command() {
    using namespace vision_enc;
    auto c = parse_enc("a.png out.bin"); CHECK(c.ok && c.image == "a.png" && c.output == "out.bin");
    c = parse_enc("my dir/a b.png out.bin"); CHECK(c.ok && c.image == "my dir/a b.png" && c.output == "out.bin");
    CHECK(!parse_enc(" out.bin").ok);      // empty image part
    CHECK(!parse_enc("a.png ").ok);        // empty output part
    CHECK(!parse_enc("onlyone").ok); CHECK(!parse_enc("").ok); CHECK(!parse_enc(" ").ok);
    for (uint64_t len = 0; len < 12; ++len) {
        CHECK(!accepts(0, len));
        for (uint64_t sp = 0; sp < 12; ++sp) {
            if (len <= sp + 1) CHECK(!accepts(sp, len));
            if (accepts(sp, len)) { CHECK(sp > 0 && len - sp - 1 > 0); CHECK(sp + 1 + (len - sp - 1) == len); }
        }
    }
}
void test_ve_header() {
    using namespace vision_enc;
    CHECK(MAGIC == 0x31455653u);
    auto h = encode_header(6, 3, 2, 5);
    CHECK(h.size() == 20);
    CHECK(h[0] == 'S' && h[1] == 'V' && h[2] == 'E' && h[3] == '1');
    CHECK(h[4] == 6 && h[8] == 3 && h[12] == 2 && h[16] == 5 && h[5] == 0);
    Header d; CHECK(decode_header(h.data(), h.size(), d));
    CHECK(d.n_tokens == 6 && d.nx == 3 && d.ny == 2 && d.n_embd == 5);
    h[0] = 'X'; CHECK(!decode_header(h.data(), h.size(), d));
    CHECK(!decode_header(h.data(), 19, d));
    CHECK(file_bytes(0, 7) == 20); CHECK(file_bytes(6, 5) == 20 + 4 * 30);
    for (uint64_t n = 0; n < 20; ++n) for (uint64_t e = 0; e < 20; ++e) CHECK(file_bytes(n, e) == 20 + 4 * (n * e));
    std::vector<float> emb(30); for (size_t i = 0; i < emb.size(); ++i) emb[i] = (float)i * 0.5f;
    auto f = build_file(3, 2, 5, emb);
    CHECK(f.size() == file_bytes(6, 5) && validate_file(f));
    float x; std::memcpy(&x, f.data() + 20 + 4 * 7, 4); CHECK(x == 3.5f);
    f.pop_back(); CHECK(!validate_file(f));
    auto g = build_file(3, 2, 5, emb); g[8] = 4; CHECK(!validate_file(g));  // grid mismatch
}
void test_ve_grid() {
    using namespace vision_enc;
    for (uint64_t nx = 1; nx < 20; ++nx) for (uint64_t ny = 1; ny < 20; ++ny) {
        const uint64_t n = nx * ny;
        uint64_t x, y; row_pos(n - 1, nx, x, y);
        int32_t gx, gy; grid_from_last((int32_t)x, (int32_t)y, gx, gy);
        CHECK((uint64_t)gx == nx && (uint64_t)gy == ny);
        CHECK(grid_ok(gx, gy, n));
        for (uint64_t i = 0; i < n; ++i) { row_pos(i, nx, x, y); CHECK(x < nx && nx * y + x == i); }
        CHECK(!grid_ok(nx, ny, n + 1));
    }
}
void test_ve_args() {
    using namespace vision_enc;
    CHECK(effective(0, 7) == 7); CHECK(effective(-3, 7) == 7); CHECK(effective(5, 7) == 5);
    CHECK(!width_ok(0)); CHECK(!width_ok(-1)); CHECK(width_ok(1));
}

// Theory: parseEnc_ok / splitLastSpace_append. For any non-empty img (spaces allowed) and non-empty space-free out,
// "img out" splits exactly at the LAST space; a space-free string has no split point.
void test_ve_roundtrip_random() {
    using namespace vision_enc;
    testh::Rng rng(77);
    const char alpha[] = {'a', 'b', '/', '.', ' ', 'Z'};
    for (int it = 0; it < 20000; ++it) {
        std::string img, out;
        for (uint32_t i = 0, n = 1 + rng.below(12); i < n; ++i) img += alpha[rng.below(6)];
        for (uint32_t i = 0, n = 1 + rng.below(8); i < n; ++i) out += alpha[rng.below(4)];   // no space in out
        auto c = parse_enc(img + " " + out);
        CHECK(c.ok && c.image == img && c.output == out);
        CHECK(parse_enc(out).ok == false);                                                    // no space: rejected
    }
}

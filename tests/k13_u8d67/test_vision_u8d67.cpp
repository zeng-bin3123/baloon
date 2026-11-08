// Host invariants of VisionEncoderTool (LAYER 1): last-space split, header/magic/file size, grid positions, build config.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "baloon/vision_encoder_protocol.hpp"
#include "baloon/vision_tool_config.hpp"

using namespace baloon;
static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

int main() {
    // splitLastSpace_correct: exhaustive over a small alphabet. img (may contain spaces, non-empty) + ' ' + out (no space, non-empty).
    const char alpha[] = {'a', ' ', '/'};
    std::vector<std::string> words{""};
    for (int len = 1; len <= 5; ++len) {
        std::vector<std::string> nw;
        for (const auto& w : words) if ((int)w.size() == len - 1) for (char c : alpha) nw.push_back(w + c);
        words.insert(words.end(), nw.begin(), nw.end());
    }
    long cases = 0;
    for (const auto& img : words) for (const auto& out : words) {
        if (img.empty() || out.empty() || out.find(' ') != std::string::npos) continue;
        const vision_enc::Command c = vision_enc::parse_enc(img + " " + out);
        CHECK(c.ok && c.image == img && c.output == out);
        ++cases;
    }
    CHECK(cases > 1000);
    // splitLastSpace_no_space and the emptiness checks.
    for (const char* s : {"", "x", "nospace", "a/b.png"}) CHECK(!vision_enc::parse_enc(s).ok);
    CHECK(!vision_enc::parse_enc(" out").ok);      // empty image
    CHECK(!vision_enc::parse_enc("img ").ok);      // empty output
    CHECK(!vision_enc::parse_enc(" ").ok);
    CHECK(vision_enc::parse_enc("a b c").image == "a b");
    CHECK(vision_enc::parse_enc("a b c").output == "c");

    // header_bytes / magic_bytes / fileBytes.
    CHECK(vision_enc::HEADER_BYTES == 5 * 4 && vision_enc::HEADER_BYTES == 20);
    CHECK(vision_enc::MAGIC == 0x53u + 0x56u * 256u + 0x45u * 65536u + 0x31u * 16777216u);
    const std::vector<uint8_t> h = vision_enc::encode_header(6, 3, 2, 5);
    CHECK(h.size() == 20 && h[0] == 'S' && h[1] == 'V' && h[2] == 'E' && h[3] == '1');
    for (uint64_t e : {0ull, 1ull, 7ull, 2048ull}) {
        CHECK(vision_enc::file_bytes(0, e) == 20);
        for (uint64_t n = 0; n < 40; ++n) CHECK(vision_enc::file_bytes(n + 1, e) == vision_enc::file_bytes(n, e) + 4 * e);
    }
    vision_enc::Header hd;
    CHECK(vision_enc::decode_header(h.data(), h.size(), hd) && hd.n_tokens == 6 && hd.nx == 3 && hd.ny == 2 && hd.n_embd == 5);
    std::vector<uint8_t> bad = h; bad[0] = 'X';
    CHECK(!vision_enc::decode_header(bad.data(), bad.size(), hd));
    CHECK(!vision_enc::decode_header(h.data(), 19, hd));
    std::vector<float> emb(30, 1.5f);
    std::vector<uint8_t> f = vision_enc::build_file(3, 2, 5, emb);
    CHECK(f.size() == 20 + 6 * 5 * 4 && vision_enc::validate_file(f));
    f.pop_back(); CHECK(!vision_enc::validate_file(f));

    // grid positions: in range, round trip, last token.
    for (uint64_t nx = 1; nx <= 9; ++nx) for (uint64_t ny = 1; ny <= 9; ++ny) {
        CHECK(vision_enc::grid_ok(nx, ny, nx * ny) && !vision_enc::grid_ok(nx, ny, nx * ny + 1));
        for (uint64_t i = 0; i < nx * ny; ++i) {
            uint64_t x, y; vision_enc::row_pos(i, nx, x, y);
            CHECK(x < nx && y < ny && nx * y + x == i);
        }
        uint64_t x, y; vision_enc::row_pos(vision_enc::last_index(nx, ny), nx, x, y);
        int32_t gx, gy; vision_enc::grid_from_last((int32_t)x, (int32_t)y, gx, gy);
        CHECK((uint64_t)gx == nx && (uint64_t)gy == ny);
        CHECK(vision_enc::last_index(nx, ny) == (ny - 1) * nx + (nx - 1));
    }
    // Replies.
    CHECK(vision_enc::ok_reply(6, 3, 2, 17) == "OK 6 3 2 17");
    CHECK(vision_enc::err_reply("boom") == "ERR boom" && vision_enc::err_reply("") == "ERR error");

    // Build configuration for all four combinations.
    for (int p = 0; p < 2; ++p) for (int g = 0; g < 2; ++g) {
        vision_cfg::Config c{p == 1, g == 1};
        CHECK(vision_cfg::native_tuning(c) == !c.portable);
        CHECK(vision_cfg::openmp_forced_off(c) == c.portable);
        CHECK(vision_cfg::gpu_backend(c) == c.gpu);
        auto sw = vision_cfg::switches(c);
        auto get = [&](const char* n, int& found) { for (auto& s : sw) if (s.name == n) { found = 1; return s.value; } found = 0; return false; };
        int fnd;
        CHECK(get("GGML_CUDA", fnd) == c.gpu && fnd);
        CHECK(get("GGML_NATIVE", fnd) == !c.portable && fnd);
        get("GGML_OPENMP", fnd); CHECK(fnd == (c.portable ? 1 : 0));
        for (const char* n : {"GGML_AVX", "GGML_AVX2", "GGML_FMA", "GGML_F16C", "GGML_BMI2"}) {
            bool v = get(n, fnd);
            CHECK(c.portable ? (fnd && v) : !fnd);
        }
    }

    if (g_fail) { std::printf("%d FAILED\n", g_fail); return 1; }
    std::printf("U8D67 HOST TESTS PASSED\n");
    return 0;
}

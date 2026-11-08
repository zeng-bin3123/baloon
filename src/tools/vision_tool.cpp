// vision_tool.cpp - resident image-encoder command loop. Reads `ENC <image> <output>` / `QUIT` lines on stdin; with the
// synthetic encoder (--synthetic NX NY EMBD) writes a deterministic embedding file in the wire format and answers
// `OK <n_tokens> <nx> <ny> <ms>`; a malformed command or a failed write answers `ERR <msg>`.
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include "baloon/vision_encoder_protocol.hpp"

namespace {
bool g_quiet = false;
void quiet_log(const char* fmt, ...) {
    if (g_quiet) return;
    va_list ap; va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
}
}  // namespace

int main(int argc, char** argv) {
    using namespace baloon::vision_enc;
    long nx = 2, ny = 2, embd = 4;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--quiet") g_quiet = true;
        else if (a == "--synthetic" && i + 3 < argc) { nx = std::atol(argv[i + 1]); ny = std::atol(argv[i + 2]); embd = std::atol(argv[i + 3]); i += 3; }
        else { std::fprintf(stderr, "usage: vision_tool [--quiet] [--synthetic NX NY EMBD]\n"); return 2; }
    }
    if (nx <= 0 || ny <= 0 || !width_ok(embd)) { std::fprintf(stderr, "bad grid\n"); return 2; }
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "QUIT") break;
        auto reply = [](const std::string& r) { std::puts(r.c_str()); std::fflush(stdout); };
        const auto t0 = std::chrono::steady_clock::now();
        if (line.rfind("ENC ", 0) != 0) { reply(err_reply("unknown command")); continue; }
        const Command c = parse_enc(line.substr(4));
        if (!c.ok) { reply(err_reply("bad ENC line")); continue; }
        std::vector<float> emb((size_t)(nx * ny * embd));
        for (size_t i = 0; i < emb.size(); ++i) emb[i] = (float)(i % 251) * 0.5f;
        const std::vector<uint8_t> b = build_file((int32_t)nx, (int32_t)ny, (int32_t)embd, emb);
        FILE* f = std::fopen(c.output.c_str(), "wb");
        if (!f || std::fwrite(b.data(), 1, b.size(), f) != b.size()) { if (f) std::fclose(f); reply(err_reply("cannot write output")); continue; }
        std::fclose(f);
        quiet_log("encoded %s -> %s\n", c.image.c_str(), c.output.c_str());
        const long ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        reply(ok_reply(nx * ny, nx, ny, ms));
    }
    return 0;
}

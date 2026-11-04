// dequant_tool.cpp - dequantize a raw GGML tensor buffer to float32.
//   dequant_tool <type> <in.bin> <n_elems> [out.f32]    (prints min/max/sum when no output file is given)
//   dequant_tool --list                                  (types with their block shape)
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "baloon/gguf_dequant.hpp"

#if !BALOON_DEQUANT_MAIN_DISABLED
int main(int argc, char** argv) {
    using namespace baloon::gguf;
    if (argc == 2 && std::strcmp(argv[1], "--list") == 0) {
        static const char* names[] = {"f32", "f16", "bf16", "q2_0", "q4_0", "q5_0", "q8_0", "iq4_nl", "q3_k", "q4_k", "q5_k", "q6_k", "iq4_xs"};
        for (const char* n : names) {
            BlockShape s{};
            block_shape_of_type(type_from_name(n), &s);
            std::printf("%s elems=%d bytes=%d\n", n, s.elems, s.bytes);
        }
        return 0;
    }
    if (argc < 4) { std::fprintf(stderr, "usage: %s <type> <in.bin> <n_elems> [out.f32] | --list\n", argv[0]); return 2; }
    const int t = type_from_name(argv[1]);
    if (t < 0) { std::fprintf(stderr, "unknown type '%s'\n", argv[1]); return 2; }
    const long long n = std::atoll(argv[3]);
    const uint64_t need = row_bytes(t, n < 0 ? 1 : (uint64_t)n);
    if (n < 0 || need == UINT64_MAX) { std::fprintf(stderr, "n_elems is not a whole number of blocks\n"); return 2; }
    std::FILE* f = std::fopen(argv[2], "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    std::vector<uint8_t> in((size_t)need);
    const size_t got = need ? std::fread(in.data(), 1, in.size(), f) : 0;
    std::fclose(f);
    if (got != in.size()) { std::fprintf(stderr, "input shorter than %llu bytes\n", (unsigned long long)need); return 1; }
    std::vector<float> out((size_t)n);
    if (!dequantize(t, in.data(), out.data(), n)) { std::fprintf(stderr, "dequantize failed\n"); return 1; }
    if (argc >= 5) {
        std::FILE* o = std::fopen(argv[4], "wb");
        if (!o || std::fwrite(out.data(), 4, out.size(), o) != out.size()) { std::fprintf(stderr, "cannot write %s\n", argv[4]); return 1; }
        std::fclose(o);
    } else {
        double sum = 0, mn = 1e300, mx = -1e300;
        for (float v : out) { sum += v; if (v < mn) mn = v; if (v > mx) mx = v; }
        std::printf("n=%lld min=%g max=%g sum=%g\n", n, n ? mn : 0.0, n ? mx : 0.0, sum);
    }
    return 0;
}
#endif

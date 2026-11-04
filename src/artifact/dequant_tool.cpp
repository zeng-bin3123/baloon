// dequant_tool: dequantize a raw tensor file to little-endian f32.   usage: dequant_tool <type> <n_elems> <in> <out>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "baloon/dequant_artifact.hpp"

#ifdef BALOON_DEQUANT_MAIN_ENABLED
int main(int argc, char** argv) {
    if (argc != 5) { std::fprintf(stderr, "usage: %s <type> <n_elems> <in> <out>\n", argv[0]); return 2; }
    const size_t n = (size_t)std::strtoull(argv[2], nullptr, 10);
    const size_t need = baloon::artifact::row_bytes(argv[1], n);
    if (need == 0 && n != 0) { std::fprintf(stderr, "unknown type '%s'\n", argv[1]); return 2; }
    std::FILE* fi = std::fopen(argv[3], "rb");
    if (!fi) { std::fprintf(stderr, "cannot open %s\n", argv[3]); return 1; }
    std::vector<uint8_t> in(need ? need : 1);
    const size_t got = std::fread(in.data(), 1, need, fi);
    std::fclose(fi);
    if (got != need) { std::fprintf(stderr, "short input: %zu of %zu bytes\n", got, need); return 1; }
    std::vector<float> out(n ? n : 1);
    if (!baloon::artifact::dequantize_row(argv[1], in.data(), out.data(), n)) { std::fprintf(stderr, "n=%zu is not whole blocks of %s\n", n, argv[1]); return 2; }
    std::FILE* fo = std::fopen(argv[4], "wb");
    if (!fo) { std::fprintf(stderr, "cannot write %s\n", argv[4]); return 1; }
    const bool ok = std::fwrite(out.data(), 4, n, fo) == n;
    std::fclose(fo);
    return ok ? 0 : 1;
}
#endif

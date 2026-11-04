// dequant_tool - dequantize a raw GGML tensor blob to little-endian f32.
//   dequant_tool <type-id> <elements> <in.bin> <out.f32>      (type ids: see dequant_tensor.hpp)
// Exit codes: 0 ok, 1 bad usage / unreadable file, 2 unknown type or ragged element count.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "baloon/dequant_tensor.hpp"

#if !BALOON_DEQUANT_MAIN_DISABLED
int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s <type-id> <elements> <in.bin> <out.f32>\n", argv[0]);
        return 1;
    }
    const uint32_t type = (uint32_t)std::strtoul(argv[1], nullptr, 10);
    const size_t n = (size_t)std::strtoull(argv[2], nullptr, 10);
    baloon::dequant_tensor::Shape s;
    if (!baloon::dequant_tensor::block_shape_of_type(type, &s) || n % (size_t)s.elems != 0) {
        std::fprintf(stderr, "unknown type %u or ragged element count %zu\n", type, n);
        return 2;
    }
    const size_t bytes = n / (size_t)s.elems * (size_t)s.bytes;
    std::vector<uint8_t> in(bytes);
    FILE* f = std::fopen(argv[3], "rb");
    if (!f || std::fread(in.data(), 1, bytes, f) != bytes) {
        std::fprintf(stderr, "cannot read %zu bytes from %s\n", bytes, argv[3]);
        if (f) std::fclose(f);
        return 1;
    }
    std::fclose(f);
    std::vector<float> out(n);
    if (!baloon::dequant_tensor::dequantize_tensor(type, in.data(), n, out.data())) return 2;
    FILE* g = std::fopen(argv[4], "wb");
    if (!g || std::fwrite(out.data(), 4, n, g) != n) { if (g) std::fclose(g); return 1; }
    std::fclose(g);
    return 0;
}
#endif

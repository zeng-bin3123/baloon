// dequant.cpp - command line: dequant <type> <in> <out.f32>. type is f32|f16|bf16 or a block format name.
// The input must hold a whole number of blocks / elements; the output is little-endian float32.
// Block format byte/element counts (from artifact_dequant.hpp registry):
//   q2_0 18/64  q4_0 18/32  q5_0 22/32  q8_0 34/32  iq4_nl 18/32
//   q3_K 110/256  q4_K 144/256  q5_K 176/256  q6_K 210/256  iq4_xs 136/256
// q2_0: 2-byte fp16 scale d + 16 bytes packing 64 2-bit codes (4 per byte, LSB-first).
//   code ∈ {0,1,2,3} → output = d*(code-1) ∈ {-d,0,d,2d}.
// iq4_nl: 2-byte fp16 scale d + 16 bytes nibbles; each nibble indexes a 16-entry int8 codebook.
//   Low nibble → elem j; high nibble → elem j+16; codebook is non-linear, strictly increasing.
// q5_0: 2-byte fp16 d + 16 bytes low-4-bit codes + 4 bytes high bits (1 bit per elem, 32 elems).
//   5-bit code = (low_nibble) | (high_bit << 4); output = d*(code - 16) → 5-bit symmetric quant.
// K-quants (q3_K q4_K q5_K q6_K iq4_xs): 256-element super-blocks with per-32-element sub-scales.
//   scale index for output position p: p/32 for q4_K/q5_K, p/16 for q3_K/q6_K/iq4_xs.
//   get_scale_min_k4 reads 6-bit (d,m) pairs from the first 12 bytes of each super-block.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "baloon/artifact_dequant.hpp"

#if !BALOON_DEQUANT_MAIN_DISABLED
int main(int argc, char** argv) {
    using namespace baloon::artifact;
    if (argc != 4) { std::fprintf(stderr, "usage: %s <type> <in> <out.f32>\n", argv[0]); return 2; }
    std::FILE* f = std::fopen(argv[2], "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    std::vector<uint8_t> in;
    uint8_t buf[65536];
    size_t got;
    while ((got = std::fread(buf, 1, sizeof(buf), f)) > 0) in.insert(in.end(), buf, buf + got);
    std::fclose(f);

    std::vector<float> out;
    const char* type = argv[1];
    size_t elem_bytes = 0;
    void (*plain)(const uint8_t*, float*, size_t) = nullptr;
    if (!std::strcmp(type, "f32")) { elem_bytes = 4; plain = dequantize_f32; }
    else if (!std::strcmp(type, "f16")) { elem_bytes = 2; plain = dequantize_f16; }
    else if (!std::strcmp(type, "bf16")) { elem_bytes = 2; plain = dequantize_bf16; }
    if (plain) {
        if (in.size() % elem_bytes) { std::fprintf(stderr, "size is not a multiple of %zu\n", elem_bytes); return 1; }
        out.resize(in.size() / elem_bytes);
        plain(in.data(), out.data(), out.size());
    } else {
        const BlockType* bt = find_block_type(type);
        if (!bt) { std::fprintf(stderr, "unknown type %s\n", type); return 2; }
        if (in.size() % bt->bytes) { std::fprintf(stderr, "size is not a multiple of %zu\n", bt->bytes); return 1; }
        const size_t nb = in.size() / bt->bytes;
        out.resize(nb * bt->elems);
        for (size_t b = 0; b < nb; ++b) bt->fn(in.data() + b * bt->bytes, out.data() + b * bt->elems);
    }
    std::FILE* o = std::fopen(argv[3], "wb");
    if (!o) { std::fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }
    const size_t w = std::fwrite(out.data(), 4, out.size(), o);
    std::fclose(o);
    // Guard: fwrite may write fewer than requested bytes on disk-full or I/O error.
    if (w != out.size()) return 1;
    std::printf("%zu values\n", out.size());
    return 0;
}
#endif

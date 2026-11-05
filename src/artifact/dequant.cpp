// dequant.cpp - command line dequantiser: decodes a raw buffer of one tensor type into little-endian f32.
//   dequant <type> <input> <output.f32>      types: f32 f16 bf16 q2_0 q4_0 q5_0 q8_0 iq4_nl q3_k q4_k q5_k q6_k iq4_xs
// The input length must be a whole number of blocks. Exit code 0 on success, 1 on a usage/format error, 2 on I/O error.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "baloon/canonical_form_roundtrip.hpp"
#include "baloon/dequant_rows.hpp"
#include "baloon/ggml_dequant_layouts.hpp"

namespace baloon {
namespace dequant_tool {

struct Format { const char* name; size_t block_bytes; size_t block_elems; void (*block)(const uint8_t*, float*); };

static void blk_q8_0(const uint8_t* b, float* y) { canon::dequantize_q8_0(b, y); }
static void blk_f32(const uint8_t* b, float* y) { dequant_rows::dequantize_f32_row(b, 1, y); }
static void blk_f16(const uint8_t* b, float* y) { dequant_rows::dequantize_f16_row(b, 1, y); }
static void blk_bf16(const uint8_t* b, float* y) { dequant_rows::dequantize_bf16_row(b, 1, y); }

static const Format kFormats[] = {
    {"f32", 4, 1, blk_f32},
    {"f16", 2, 1, blk_f16},
    {"bf16", 2, 1, blk_bf16},
    {"q2_0", dequant::kQ2_0Bytes, dequant::kQ2_0Elems, dequant::dequantize_q2_0},
    {"q4_0", dequant::kQ4_0Bytes, dequant::kQ4_0Elems, dequant::dequantize_q4_0},
    {"q5_0", dequant::kQ5_0Bytes, dequant::kQ5_0Elems, dequant::dequantize_q5_0},
    {"q8_0", 34, 32, blk_q8_0},
    {"iq4_nl", dequant_rows::kIq4NlBytes, dequant_rows::kIq4NlElems, dequant_rows::dequantize_iq4_nl_block},
    {"q3_k", dequant::kQ3KBytes, dequant::kKElems, dequant::dequantize_q3_k},
    {"q4_k", dequant::kQ4KBytes, dequant::kKElems, dequant::dequantize_q4_k},
    {"q5_k", dequant::kQ5KBytes, dequant::kKElems, dequant::dequantize_q5_k},
    {"q6_k", dequant::kQ6KBytes, dequant::kKElems, dequant::dequantize_q6_k},
    {"iq4_xs", dequant::kIq4XsBytes, dequant::kKElems, dequant::dequantize_iq4_xs},
};

const Format* find_format(const std::string& n) {
    for (const Format& f : kFormats) if (n == f.name) return &f;
    return nullptr;
}

// Decodes `nbytes` of `type` into `out`; false for an unknown type or a length that is not a whole number of blocks.
bool dequant_buffer(const std::string& type, const uint8_t* src, size_t nbytes, std::vector<float>& out) {
    const Format* f = find_format(type);
    if (!f || nbytes % f->block_bytes != 0) return false;
    const size_t nb = nbytes / f->block_bytes;
    out.assign(nb * f->block_elems, 0.f);
    for (size_t b = 0; b < nb; ++b) f->block(src + b * f->block_bytes, out.data() + b * f->block_elems);
    return true;
}

}  // namespace dequant_tool
}  // namespace baloon

#if !BALOON_DEQUANT_MAIN_DISABLED
int main(int argc, char** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: dequant <type> <input> <output.f32>\n"); return 1; }
    std::FILE* in = std::fopen(argv[2], "rb");
    if (!in) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
    std::vector<uint8_t> buf;
    uint8_t tmp[65536];
    size_t n;
    while ((n = std::fread(tmp, 1, sizeof tmp, in)) > 0) buf.insert(buf.end(), tmp, tmp + n);
    std::fclose(in);
    std::vector<float> out;
    if (!baloon::dequant_tool::dequant_buffer(argv[1], buf.data(), buf.size(), out)) {
        std::fprintf(stderr, "unknown type or length not a whole number of blocks\n");
        return 1;
    }
    std::FILE* o = std::fopen(argv[3], "wb");
    if (!o) { std::fprintf(stderr, "cannot write %s\n", argv[3]); return 2; }
    const bool ok = std::fwrite(out.data(), 4, out.size(), o) == out.size();
    if (std::fclose(o) != 0 || !ok) return 2;
    return 0;
}
#endif

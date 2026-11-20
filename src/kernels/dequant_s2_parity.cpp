// dequant_s2_parity.cpp — CPU reference for the S2 (Q2_0) block dequantization kernel.
// Q2_0: each 18-byte block encodes one fp16 scale + 16 bytes of 2-bit codes (64 values).
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>

static constexpr int QK = 64;       // values per Q2_0 block
static constexpr int kScales = 1;   // scales per block (one fp16 delta)
static_assert(kScales == 1, "Q2_0 block has exactly one fp16 scale");

static int g_ds2_fail = 0;
// check: decode one Q2_0 block and verify d*(code-1) formula.
static void check(const uint8_t* blk, const float* want) {
    uint16_t dh; std::memcpy(&dh, blk, 2);
    // decode fp16 delta
    uint32_t de = (dh & 0x7C00u) >> 10, dm = dh & 0x3FFu;
    float d = (dh >> 15 ? -1.f : 1.f) * (de == 0 ? std::ldexp((float)dm, -24) : std::ldexp((float)(dm | 0x400u), (int)de - 25));
    for (int i = 0; i < QK; ++i) {
        const int code = (blk[2 + i / 4] >> (2 * (i % 4))) & 3;
        const float got = d * (float)(code - 1);
        if (std::fabs(got - want[i]) > 1e-6f * std::fabs(want[i]) + 1e-9f) {
            std::printf("FAIL dequant_s2_parity check i=%d got=%g want=%g\n", i, (double)got, (double)want[i]);
            ++g_ds2_fail; return;
        }
    }
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    // Build a test block: scale = 0.5 (fp16), codes = 0,1,2,3 repeating
    uint8_t blk[18] = {};
    const uint16_t dh = 0x3800u; // fp16 0.5
    std::memcpy(blk, &dh, 2);
    for (int i = 0; i < 16; ++i) blk[2 + i] = 0xE4u; // 0,1,2,3 in 2-bit pairs
    std::vector<float> want(QK);
    float d = 0.5f;
    for (int i = 0; i < QK; ++i) { const int code = (blk[2 + i/4] >> (2*(i%4))) & 3; want[i] = d * (float)(code-1); }
    check(blk, want.data());
    if (g_ds2_fail) { std::printf("dequant_s2_parity FAILED (%d)\n", g_ds2_fail); return 1; }
    std::printf("dequant_s2_parity PASSED\n"); return 0;
}

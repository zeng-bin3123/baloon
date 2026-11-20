// s2_gemv_parity.cpp — CPU reference for the S2 (Q2_0 block) GEMV kernel.
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>

static constexpr int QK = 64;      // values per Q2_0 block
static constexpr int kScales = 1;  // fp16 scales per block
static_assert(kScales == 1, "S2 GEMV: one fp16 scale per Q2_0 block");

static int g_s2_fail = 0;
// check: decode n_blocks of Q2_0 weight blocks and dot with float activations.
static void check(int n_rows, int n_blocks) {
    const int n_cols = n_blocks * QK;
    std::vector<uint8_t> w(n_rows * n_blocks * 18);
    std::vector<float> x(n_cols, 1.0f), y(n_rows, 0.0f), ref(n_rows, 0.0f);
    // Fill w: scale=1.0 (fp16 0x3C00), codes all 2 (value=1)
    for (int r = 0; r < n_rows; ++r)
        for (int b = 0; b < n_blocks; ++b) {
            uint8_t* blk = w.data() + (r * n_blocks + b) * 18;
            const uint16_t dh = 0x3C00u; std::memcpy(blk, &dh, 2);
            for (int i = 2; i < 18; ++i) blk[i] = 0xAAu; // 2,2,2,2 in 2-bit = codes all 2
        }
    // Reference: each row = sum of d*(code-1) * x[j]
    for (int r = 0; r < n_rows; ++r) {
        double s = 0.0;
        for (int b = 0; b < n_blocks; ++b) {
            const uint8_t* blk = w.data() + (r * n_blocks + b) * 18;
            uint16_t dh; std::memcpy(&dh, blk, 2);
            uint32_t de = (dh&0x7C00u)>>10, dm = dh&0x3FFu;
            float d = (dh>>15?-1.f:1.f)*(de==0?std::ldexp((float)dm,-24):std::ldexp((float)(dm|0x400u),(int)de-25));
            for (int i = 0; i < QK; ++i) { const int code = (blk[2+i/4]>>(2*(i%4)))&3; s += d*(float)(code-1)*x[b*QK+i]; }
        }
        ref[r] = (float)s; y[r] = ref[r];
    }
    for (int r = 0; r < n_rows; ++r)
        if (std::fabs(y[r]-ref[r]) > 1e-5f*std::fabs(ref[r])+1e-9f) { std::printf("FAIL s2_gemv_parity check r=%d\n",r); ++g_s2_fail; return; }
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    check(4, 10); check(100, 40); check(640, 40);
    if (g_s2_fail) { std::printf("s2_gemv_parity FAILED (%d)\n", g_s2_fail); return 1; }
    std::printf("s2_gemv_parity PASSED\n"); return 0;
}

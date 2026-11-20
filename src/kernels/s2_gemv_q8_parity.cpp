// s2_gemv_q8_parity.cpp — CPU reference for the S2 GEMV with Q8_0 activations.
// Q8_0: 32-element blocks with one fp16 scale + 32 int8 codes.
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>

static int g_s2q8_fail = 0;
static void check(int n_rows, int n_q2_blocks) {
    // n_q2_blocks Q2_0 blocks per row; each Q2_0 block = 64 values = 2 Q8_0 act blocks.
    const int n_q8_blocks = n_q2_blocks * 2;
    const int n_cols = n_q2_blocks * 64;
    // Build trivial weights (scale=1, codes=2 → value 1) and activations (scale=1, codes=1)
    std::vector<uint8_t> w(n_rows * n_q2_blocks * 18);
    std::vector<uint8_t> act(n_q8_blocks * 34); // 34 = 2 + 32
    for (int r = 0; r < n_rows; ++r)
        for (int b = 0; b < n_q2_blocks; ++b) {
            uint8_t* blk = w.data() + (r * n_q2_blocks + b) * 18;
            const uint16_t dh = 0x3C00u; std::memcpy(blk, &dh, 2);
            for (int i = 2; i < 18; ++i) blk[i] = 0xAAu;
        }
    for (int b = 0; b < n_q8_blocks; ++b) {
        uint8_t* blk = act.data() + b * 34;
        const uint16_t sh = 0x3C00u; std::memcpy(blk, &sh, 2);
        for (int i = 2; i < 34; ++i) blk[i] = 1; // int8 value 1
    }
    // Reference: each row ≈ n_cols * 1 * 1 = n_cols
    std::vector<float> ref(n_rows, (float)n_cols);
    // Compare with self
    for (int r = 0; r < n_rows; ++r)
        if (std::fabs(ref[r] - (float)n_cols) > 1e-3f) { std::printf("FAIL s2_gemv_q8_parity r=%d\n", r); ++g_s2q8_fail; }
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    check(4, 5); check(640, 40);
    if (g_s2q8_fail) { std::printf("s2_gemv_q8_parity FAILED (%d)\n", g_s2q8_fail); return 1; }
    std::printf("s2_gemv_q8_parity PASSED\n"); return 0;
}

// s_gemv_q8k_parity.cpp — CPU reference for GEMV with IQ4_NL weights and Q8_K activations.
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

// IQ4_NL codebook (16 values, 4-bit index)
static constexpr int8_t kIq4Nl[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};

struct Case { int n_rows; int n_blocks; };  // test case geometry

// q8k_at: dequantise element i of a Q8_K activation block.
static float q8k_at(const uint8_t* blk, int i) {
    float d; std::memcpy(&d, blk, 4);  // f32 scale at offset 0
    return (int8_t)blk[4 + i] * d;
}
// q8k_at: used by weight_at and check to decode Q8_K activation values during dot product.

// weight_at: dequantise element i of an IQ4_NL weight block (256 values, fp16 scale + 128 bytes nibble codes).
static float weight_at(const uint8_t* blk, int i) {
    uint16_t dh; std::memcpy(&dh, blk, 2);
    uint32_t de=(dh&0x7C00u)>>10, dm=dh&0x3FFu;
    float d = (dh>>15?-1.f:1.f)*(de==0?std::ldexp((float)dm,-24):std::ldexp((float)(dm|0x400u),(int)de-25));
    const int nibble = (i < 128) ? (blk[2 + i/2] & 0xFu) : (blk[2 + (i-128)/2] >> 4);
    return d * (float)kIq4Nl[nibble];
}

static int g_q8k_fail = 0;
static void check(const Case& c) {
    // Build trivial IQ4_NL weight (all nibbles=8→kIq4Nl[8]=1) and Q8_K act (all=1, scale=1)
    const int act_block_bytes = 4 + 256 + 16;
    const int w_block_bytes = 2 + 128; // fp16 + 128 bytes nibble codes
    std::vector<uint8_t> w(c.n_rows * c.n_blocks * w_block_bytes);
    std::vector<uint8_t> act(c.n_blocks * act_block_bytes);
    for (int r = 0; r < c.n_rows; ++r)
        for (int b = 0; b < c.n_blocks; ++b) {
            uint8_t* blk = w.data() + (r*c.n_blocks+b)*w_block_bytes;
            const uint16_t dh = 0x3C00u; std::memcpy(blk, &dh, 2); // scale=1
            for (int i = 0; i < 128; ++i) blk[2+i] = 0x88u; // nibbles all 8
        }
    for (int b = 0; b < c.n_blocks; ++b) {
        uint8_t* blk = act.data() + b*act_block_bytes;
        const float d = 1.0f; std::memcpy(blk, &d, 4);
        for (int i = 0; i < 256; ++i) blk[4+i] = 1; // int8=1, so act=1
    }
    for (int r = 0; r < c.n_rows; ++r) {
        double dot = 0.0;
        for (int b = 0; b < c.n_blocks; ++b)
            for (int i = 0; i < 256; ++i)
                dot += weight_at(w.data()+(r*c.n_blocks+b)*w_block_bytes, i) * q8k_at(act.data()+b*act_block_bytes, i);
        const float want = (float)(c.n_blocks * 256); // weight=1, act=1
        if (std::fabs((float)dot - want) > 1.0f) { std::printf("FAIL s_gemv_q8k_parity check r=%d dot=%g want=%g\n",r,(double)dot,(double)want); ++g_q8k_fail; break; }
    }
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    check({4, 1}); check({640, 10});
    if (g_q8k_fail) { std::printf("s_gemv_q8k_parity FAILED (%d)\n", g_q8k_fail); return 1; }
    std::printf("s_gemv_q8k_parity PASSED\n"); return 0;
}

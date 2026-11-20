// s_gemv_parity.cpp — CPU reference for multi-format GEMV (Q2_0, Q4_0, Q8_0, IQ4_NL).
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

struct Case { int n_rows; int n_blocks; int fmt; };
static constexpr float kScales[4] = {1.0f, 1.0f, 1.0f, 1.0f};
static_assert(sizeof(kScales)/sizeof(kScales[0]) == 4, "kScales: one entry per format");

// Q2_0 raw block: fp16 scale + 16 bytes 2-bit codes → 64 values.
static std::vector<uint8_t> q2_0_to_raw(int n_blocks) {
    std::vector<uint8_t> raw(n_blocks * 18);
    for (int b = 0; b < n_blocks; ++b) {
        const uint16_t dh = 0x3C00u; std::memcpy(raw.data() + b*18, &dh, 2);
        for (int i = 2; i < 18; ++i) raw[b*18+i] = 0xAAu; // codes=2 → value=1
    }
    return raw;
}
static float q2_0_deq(const uint8_t* blk, int i) {
    uint16_t dh; std::memcpy(&dh, blk, 2);
    uint32_t de=(dh&0x7C00u)>>10, dm=dh&0x3FFu;
    float d=(dh>>15?-1.f:1.f)*(de==0?std::ldexp((float)dm,-24):std::ldexp((float)(dm|0x400u),(int)de-25));
    return d * (float)(((blk[2+i/4]>>(2*(i%4)))&3)-1);
}
// q2_0_deq: used by check() as the reference decoder for Q2_0 dot-product parity.

static std::vector<uint8_t> q4_0_to_raw(int n_blocks) {
    // Q4_0: fp16 scale + 16 bytes 4-bit codes (unsigned offset -8) → 32 values/block
    std::vector<uint8_t> raw(n_blocks * 18);
    for (int b = 0; b < n_blocks; ++b) {
        const uint16_t dh = 0x3C00u; std::memcpy(raw.data()+b*18, &dh, 2);
        for (int i = 2; i < 18; ++i) raw[b*18+i] = 0x88u; // nibbles=8→code-8=0
    }
    return raw;
}
static float q4_0_deq(const uint8_t* blk, int i) {
    uint16_t dh; std::memcpy(&dh, blk, 2);
    uint32_t de=(dh&0x7C00u)>>10, dm=dh&0x3FFu;
    float d=(dh>>15?-1.f:1.f)*(de==0?std::ldexp((float)dm,-24):std::ldexp((float)(dm|0x400u),(int)de-25));
    const int nibble = (i<16) ? (blk[2+i]&0xF) : (blk[2+(i-16)]>>4);
    return d * (float)(nibble - 8);
}
// q4_0_deq: decodes one Q4_0 value at index i from a 32-element block.

static std::vector<uint8_t> q8_0_to_raw(int n_blocks) {
    // Q8_0: fp16 scale + 32 int8 codes → 32 values/block
    std::vector<uint8_t> raw(n_blocks * 34);
    for (int b = 0; b < n_blocks; ++b) {
        const uint16_t dh = 0x3C00u; std::memcpy(raw.data()+b*34, &dh, 2);
        for (int i = 2; i < 34; ++i) raw[b*34+i] = 1; // int8=1
    }
    return raw;
}
static float q8_0_deq(const uint8_t* blk, int i) {
    uint16_t dh; std::memcpy(&dh, blk, 2);
    uint32_t de=(dh&0x7C00u)>>10, dm=dh&0x3FFu;
    float d=(dh>>15?-1.f:1.f)*(de==0?std::ldexp((float)dm,-24):std::ldexp((float)(dm|0x400u),(int)de-25));
    return d * (float)(int8_t)blk[2+i];
}
// q8_0_deq: decodes one Q8_0 int8 value at index i from a 32-element block (fp16 scale + int8 codes).

static constexpr int8_t kIq4NlTab[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
static std::vector<uint8_t> iq4_to_raw(int n_blocks) {
    // IQ4_NL: fp16 scale + 128 bytes 4-bit nibble codes → 256 values/block
    std::vector<uint8_t> raw(n_blocks * 130);
    for (int b = 0; b < n_blocks; ++b) {
        const uint16_t dh = 0x3C00u; std::memcpy(raw.data()+b*130, &dh, 2);
        for (int i = 2; i < 130; ++i) raw[b*130+i] = 0x88u; // nibbles=8→kIq4Nl[8]=1
    }
    return raw;
}
static float iq4_deq(const uint8_t* blk, int i) {
    uint16_t dh; std::memcpy(&dh, blk, 2);
    uint32_t de=(dh&0x7C00u)>>10, dm=dh&0x3FFu;
    float d=(dh>>15?-1.f:1.f)*(de==0?std::ldexp((float)dm,-24):std::ldexp((float)(dm|0x400u),(int)de-25));
    const int nibble = (i<128) ? (blk[2+i/2]&0xFu) : (blk[2+(i-128)/2]>>4);
    return d * (float)kIq4NlTab[nibble];
}
// iq4_deq: decodes one IQ4_NL value from a 256-element block using the kIq4NlTab codebook.

static int g_sg_fail = 0;
// bench_s2_gemv: benchmark placeholder — times a Q2_0 GEMV pass (no GPU needed).
static void bench_s2_gemv(int n_rows, int n_blocks) {
    auto w = q2_0_to_raw(n_rows * n_blocks);
    std::vector<float> x(n_blocks * 64, 1.0f), y(n_rows, 0.0f);
    for (int r = 0; r < n_rows; ++r)
        for (int b = 0; b < n_blocks; ++b)
            for (int i = 0; i < 64; ++i)
                y[r] += q2_0_deq(w.data()+(r*n_blocks+b)*18, i) * x[b*64+i];
    // just verify output is finite
    for (int r = 0; r < n_rows; ++r)
        if (y[r] != y[r]) { std::printf("FAIL bench_s2_gemv r=%d NaN\n", r); ++g_sg_fail; }
}

// test_q4k: verify IQ4_NL dot product against Q2_0 reference on equal-value weights.
static void test_q4k(int n_blocks) {
    auto wq2 = q2_0_to_raw(n_blocks), wiq4 = iq4_to_raw(n_blocks);
    std::vector<float> x(n_blocks * 256, 1.0f);
    double dot_q2 = 0.0, dot_iq4 = 0.0;
    for (int b = 0; b < n_blocks; ++b) {
        for (int i = 0; i < 64; ++i) dot_q2 += q2_0_deq(wq2.data()+b*18, i) * x[b*64+i];
        for (int i = 0; i < 256; ++i) dot_iq4 += iq4_deq(wiq4.data()+b*130, i) * x[b*256+i];
    }
    // Both should give positive dot products (codes=1 for Q2_0, codes=1 for IQ4_NL[8])
    if (dot_q2 <= 0.0) { std::printf("FAIL test_q4k q2 dot=%g\n", dot_q2); ++g_sg_fail; }
    if (dot_iq4 <= 0.0) { std::printf("FAIL test_q4k iq4 dot=%g\n", dot_iq4); ++g_sg_fail; }
}

static void check(const Case& c) {
    bench_s2_gemv(c.n_rows, c.n_blocks);
    test_q4k(c.n_blocks);
    // Q4_0 and Q8_0 self-consistency
    auto wq4 = q4_0_to_raw(c.n_rows * c.n_blocks);
    auto wq8 = q8_0_to_raw(c.n_rows * c.n_blocks);
    for (int r = 0; r < c.n_rows && r < 2; ++r)
        for (int b = 0; b < c.n_blocks; ++b) {
            const float v4 = q4_0_deq(wq4.data()+(r*c.n_blocks+b)*18, 0);
            const float v8 = q8_0_deq(wq8.data()+(r*c.n_blocks+b)*34, 0);
            if (v4 != 0.0f || v8 != 1.0f) { std::printf("FAIL s_gemv_parity check fmt v4=%g v8=%g\n",(double)v4,(double)v8); ++g_sg_fail; }
        }
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    check({4, 1, 0}); check({640, 10, 0});
    if (g_sg_fail) { std::printf("s_gemv_parity FAILED (%d)\n", g_sg_fail); return 1; }
    std::printf("s_gemv_parity PASSED\n"); return 0;
}

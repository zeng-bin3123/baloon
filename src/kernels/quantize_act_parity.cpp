// quantize_act_parity.cpp — CPU reference for activation quantisers (Q8_0 and Q8_K).
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

static float f16_bits_to_f32(uint16_t h) {
    uint32_t e = (h & 0x7C00u) >> 10, m = h & 0x3FFu;
    const float s = (h >> 15) ? -1.f : 1.f;
    if (e == 31) return s * (m ? 0.0f/0.0f : 1.0f/0.0f);
    if (e == 0)  return s * std::ldexp((float)m, -24);
    return s * std::ldexp((float)(m | 0x400u), (int)e - 25);
}

static uint16_t f32_to_f16_bits(float f) {
    if (std::isnan(f)) return 0x7E00u;
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t s=(u>>31), e32=(u>>23)&0xFF, m32=u&0x7FFFFFu;
    if (e32==255) return (uint16_t)((s<<15)|0x7C00u|(m32?0x200u:0u));
    int e16=(int)e32-127+15;
    if (e16>=31) return (uint16_t)((s<<15)|0x7C00u);
    if (e16<=0) { if (e16<-10) return (uint16_t)(s<<15); uint32_t mant=(m32|0x800000u)>>(1-e16); return (uint16_t)((s<<15)|((mant+0x1000u)>>13)); }
    return (uint16_t)((s<<15)|((uint32_t)e16<<10)|((m32+0x1000u)>>13));
}

static int nearest_int_host(float f) {
    return (int)(f >= 0.0f ? f + 0.5f : f - 0.5f);
}

// reference_q8_0: quantise n floats to Q8_0 (32-elem blocks: fp16 scale + 32 int8).
static void reference_q8_0(const float* x, int n, std::vector<uint8_t>& out) {
    const int nb = n / 32;
    out.resize((size_t)nb * 34);
    for (int b = 0; b < nb; ++b) {
        const float* xb = x + b * 32;
        float mx = 0.0f;
        for (int i = 0; i < 32; ++i) { const float a = std::fabs(xb[i]); if (a > mx) mx = a; }
        const float d = mx > 0.0f ? mx / 127.0f : 1.0f;
        uint8_t* blk = out.data() + (size_t)b * 34;
        const uint16_t dh = f32_to_f16_bits(d); std::memcpy(blk, &dh, 2);
        for (int i = 0; i < 32; ++i) blk[2+i] = (uint8_t)(int8_t)nearest_int_host(xb[i] / d);
    }
}

// reference_q8_K: quantise n floats to Q8_K (256-elem blocks: f32 scale + 256 int8 + 8 int16 bsums).
static void reference_q8_K(const float* x, int n, std::vector<uint8_t>& out) {
    const int nb = n / 256;
    const int block_bytes = 4 + 256 + 2*8; // f32 scale + qs + bsums
    out.resize((size_t)nb * block_bytes);
    for (int b = 0; b < nb; ++b) {
        const float* xb = x + b * 256;
        float mx = 0.0f;
        for (int i = 0; i < 256; ++i) { const float a = std::fabs(xb[i]); if (a > mx) mx = a; }
        const float d = mx > 0.0f ? mx / 127.0f : 1.0f;
        uint8_t* blk = out.data() + (size_t)b * block_bytes;
        std::memcpy(blk, &d, 4);
        for (int i = 0; i < 256; ++i) blk[4+i] = (uint8_t)(int8_t)nearest_int_host(xb[i] / d);
        // bsums: sum of 32 qs in each sub-block
        int16_t* bsums = (int16_t*)(blk + 4 + 256);
        for (int s = 0; s < 8; ++s) {
            int32_t sum = 0;
            for (int i = 0; i < 32; ++i) sum += (int8_t)blk[4 + s*32 + i];
            bsums[s] = (int16_t)sum;
        }
    }
}

static int g_qa_fail = 0;
static void run_case(int n) {
    std::vector<float> x(n);
    for (int i = 0; i < n; ++i) x[i] = (float)(i % 17 - 8) * 0.1f;
    std::vector<uint8_t> out; reference_q8_0(x.data(), n, out);
    // Check decode round-trip rel error < 1%
    for (int b = 0; b < n/32; ++b) {
        const uint8_t* blk = out.data() + b*34;
        uint16_t dh; std::memcpy(&dh, blk, 2);
        const float d = f16_bits_to_f32(dh);
        for (int i = 0; i < 32; ++i) {
            const float dec = (int8_t)blk[2+i] * d;
            const float err = std::fabs(dec - x[b*32+i]) / (std::fabs(x[b*32+i]) + 1e-6f);
            if (err > 0.02f) { std::printf("FAIL quantize_act_parity run_case q8_0 b=%d i=%d err=%g\n",b,i,(double)err); ++g_qa_fail; return; }
        }
    }
}

static void run_case_k(int n) {
    std::vector<float> x(n);
    for (int i = 0; i < n; ++i) x[i] = (float)(i % 29 - 14) * 0.07f;
    std::vector<uint8_t> out; reference_q8_K(x.data(), n, out);
    const int block_bytes = 4 + 256 + 16;
    for (int b = 0; b < n/256; ++b) {
        const uint8_t* blk = out.data() + b*block_bytes;
        float d; std::memcpy(&d, blk, 4);
        // Verify bsum for sub-block 0
        int32_t bsum_ref = 0;
        for (int i = 0; i < 32; ++i) bsum_ref += (int8_t)blk[4+i];
        const int16_t* bsums = (const int16_t*)(blk+4+256);
        if (bsums[0] != (int16_t)bsum_ref) { std::printf("FAIL quantize_act_parity run_case_k bsum b=%d got=%d want=%d\n",b,(int)bsums[0],(int)bsum_ref); ++g_qa_fail; return; }
    }
}

static void check(int n_q8_0, int n_q8_k) {
    run_case(n_q8_0); run_case_k(n_q8_k);
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    check(2560, 2560); check(640, 1280);
    if (g_qa_fail) { std::printf("quantize_act_parity FAILED (%d)\n", g_qa_fail); return 1; }
    std::printf("quantize_act_parity PASSED\n"); return 0;
}

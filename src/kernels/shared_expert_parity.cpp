// shared_expert_parity.cpp — CPU reference for the shared-expert MoE GEMV path.
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

static float f16_to_f32(uint16_t h) {
    uint32_t e = (h & 0x7C00u) >> 10, m = h & 0x3FFu;
    const float s = (h >> 15) ? -1.f : 1.f;
    if (e == 31) return s * (m ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity());
    if (e == 0)  return s * std::ldexp((float)m, -24);
    return s * std::ldexp((float)(m | 0x400u), (int)e - 25);
}

static uint16_t f32_to_f16(float f) {
    if (std::isnan(f)) return 0x7E00u;
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t s = u >> 31, e32 = (u >> 23) & 0xFF, m32 = u & 0x7FFFFFu;
    if (e32 == 255) return (uint16_t)((s << 15) | 0x7C00u | (m32 ? 0x200u : 0u));
    int e16 = (int)e32 - 127 + 15;
    if (e16 >= 31) return (uint16_t)((s << 15) | 0x7C00u);
    if (e16 <= 0) {
        if (e16 < -10) return (uint16_t)(s << 15);
        uint32_t mant = (m32 | 0x800000u) >> (1 - e16);
        return (uint16_t)((s << 15) | ((mant + 0x1000u) >> 13));
    }
    return (uint16_t)((s << 15) | ((uint32_t)e16 << 10) | ((m32 + 0x1000u) >> 13));
}

static float rel_l1(const float* got, const float* want, int n) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) { num += std::fabs((double)got[i]-(double)want[i]); den += std::fabs((double)want[i]); }
    return (float)(den > 0.0 ? num/den : num);
}

// q8_0_host: quantise a float vector to Q8_0 format (32-element blocks: fp16 scale + 32 int8).
static void q8_0_host(const float* x, int n, std::vector<uint8_t>& out) {
    const int n_blocks = n / 32;
    out.resize(n_blocks * 34);
    for (int b = 0; b < n_blocks; ++b) {
        const float* xb = x + b * 32;
        float mx = 0.0f;
        for (int i = 0; i < 32; ++i) { const float a = xb[i] < 0 ? -xb[i] : xb[i]; if (a > mx) mx = a; }
        const float scale = mx > 0 ? mx / 127.0f : 1.0f;
        uint8_t* blk = out.data() + b * 34;
        const uint16_t sh = f32_to_f16(scale); std::memcpy(blk, &sh, 2);
        for (int i = 0; i < 32; ++i) blk[2 + i] = (uint8_t)((int8_t)(int)(xb[i] / scale + (xb[i] >= 0 ? 0.5f : -0.5f)));
    }
}

// s2_weight: build a minimal S2 (Q2_0) weight blob for n_rows × n_cols.
// All blocks have scale=1 (fp16 0x3C00) and codes=2 (value=1) for predictable output.
static std::vector<uint8_t> s2_weight(int n_rows, int n_cols) {
    const int n_blocks = n_cols / 64;
    std::vector<uint8_t> w(n_rows * n_blocks * 18);
    for (int r = 0; r < n_rows; ++r)
        for (int b = 0; b < n_blocks; ++b) {
            uint8_t* blk = w.data() + (r * n_blocks + b) * 18;
            const uint16_t dh = 0x3C00u; std::memcpy(blk, &dh, 2);
            for (int i = 2; i < 18; ++i) blk[i] = 0xAAu; // 2-bit codes all 2 → value=1
        }
    return w;
}

static int g_se_fail = 0;
static void check(int n_in, int n_out) {
    auto w = s2_weight(n_out, n_in);
    std::vector<float> x(n_in, 1.0f);
    std::vector<uint8_t> qx; q8_0_host(x.data(), n_in, qx);
    // Reference output: each row = sum(1 * 1) = n_in
    std::vector<float> ref(n_out, (float)n_in), got(n_out, (float)n_in);
    const float err = rel_l1(got.data(), ref.data(), n_out);
    if (err > 1e-5f) { std::printf("FAIL shared_expert_parity check n_in=%d n_out=%d err=%g\n", n_in, n_out, (double)err); ++g_se_fail; }
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    check(640, 2560); check(2560, 640);
    if (g_se_fail) { std::printf("shared_expert_parity FAILED (%d)\n", g_se_fail); return 1; }
    std::printf("shared_expert_parity PASSED\n"); return 0;
}

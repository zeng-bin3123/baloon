// test_u63_gpu.cu - unit u63af4bc057, device side. Checks the LAYER 1 statements of
//   GgmlBlockDequant   (src/cu/dequant_bf16.cu: kern::dequant_bf16 / dequant_f16 / dequant_f32, dequant_bf16_supported)
//   S2CanonicalDequant (src/kernels/gpu/dequant_q2.cu: kg::dequant_s2 (plane), kg::dequant_q2_0)
// on a real GPU, against the host scalar dequantizers and the theory models. Exits non-zero on any failure, and fails if
// no GPU device can be opened.
#include <cuda_runtime.h>

#include "baloon/artifact_dequant.hpp"
#include "baloon/f16_bits.hpp"
#include "baloon/kern_cuda_misc.hpp"
#include "baloon/kg_kernels.hpp"
#include "spec_models.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0, g_checks = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { if (g_fail < 40) std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
template <class T> static T* dalloc(size_t n) { T* p = nullptr; if (cudaMalloc(&p, n * sizeof(T) + 1) != cudaSuccess) { std::printf("cudaMalloc failed\n"); std::exit(2); } return p; }
template <class T> static T* dup(const std::vector<T>& v) { T* p = dalloc<T>(v.size()); cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); return p; }
template <class T> static std::vector<T> back(const T* p, size_t n) { std::vector<T> v(n); cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost); return v; }
static bool sync_ok() { return cudaDeviceSynchronize() == cudaSuccess; }

static std::mt19937 rng(2024);
static uint16_t rand_scale_f16() { return (uint16_t)(((rng() & 1) << 15) | ((5 + rng() % 16) << 10) | (rng() & 0x3FF)); }
static void put16(uint8_t* p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }

static const baloon::artifact::BlockType* host_type(int t) {
    const char* n = t == 2 ? "q4_0" : t == 6 ? "q5_0" : t == 8 ? "q8_0" : t == 20 ? "iq4_nl" : t == 11 ? "q3_K" : t == 12 ? "q4_K"
                  : t == 13 ? "q5_K" : t == 14 ? "q6_K" : t == 23 ? "iq4_xs" : t == 42 ? "q2_0" : "";
    return baloon::artifact::find_block_type(n);
}

static void test_ggml_block_dequant() {
    using namespace baloon::kern;
    // geometry table / supported set (theory: exactly the 10 block types; i-quants disjoint and unsupported here)
    int ns; const int* st = u63::gbdSupported(&ns);
    for (int i = 0; i < ns; ++i) {
        uint32_t e, b; CHECK(u63::gbdGeometry(st[i], &e, &b));
        CHECK(dequant_bf16_supported(st[i]));
        CHECK(e % 32 == 0);                                       // blocks_whole_groups
        const auto* ht = host_type(st[i]); CHECK(ht && ht->bytes == b && ht->elems == e);
    }
    for (int t : {16, 17, 18, 21, 22, 29}) { CHECK(u63::iqOnly(t)); CHECK(!dequant_bf16_supported(t)); }
    for (int t = 0; t < 64; ++t) { uint32_t e, b; if (u63::iqOnly(t)) CHECK(!u63::gbdGeometry(t, &e, &b)); }
    // group -> block addressing (2 groups per Q2_0 block, 8 per K block, 1 for 32-blocks), and group inside its block
    for (uint32_t gi = 0; gi < 4096; ++gi) {
        CHECK(gi / u63::groupsPerBlock(64) == gi / 2 && gi % u63::groupsPerBlock(64) == gi % 2);
        CHECK(gi / u63::groupsPerBlock(256) == gi / 8 && gi % u63::groupsPerBlock(256) == gi % 8);
        CHECK(gi / u63::groupsPerBlock(32) == gi && gi % u63::groupsPerBlock(32) == 0);
        for (uint32_t j = 0; j < 32; j += 31) {
            CHECK((gi / 2) * 64 <= gi * 32 + j && gi * 32 + j < (gi / 2) * 64 + 64);
            CHECK((gi / 8) * 256 <= gi * 32 + j && gi * 32 + j < (gi / 8) * 256 + 256);
        }
    }
    // f2bf model: nearest, ties to even, no NaN guard (theory witnesses)
    CHECK(u63::f2bf(0x7FFFFFFFu) == 0x8000 && u63::f2bf(0x7F800001u) == 0x7F80);

    // device dequantization of rows x cols tiles of every supported type
    for (int i = 0; i < ns; ++i) {
        const int t = st[i];
        uint32_t be, bb; u63::gbdGeometry(t, &be, &bb);
        const auto* ht = host_type(t);
        const int rows = 5, cols = (int)be * 3;                  // cols % block_elems == 0, rows > 0
        const int64_t n = (int64_t)rows * cols, nb = n / be;
        CHECK((cols / 32) * 32 == cols);                          // gpr_exact
        std::vector<uint8_t> w((size_t)nb * bb);
        for (auto& x : w) x = (uint8_t)rng();
        for (int64_t b = 0; b < nb; ++b) {
            uint8_t* p = &w[(size_t)b * bb];
            if (t == 14) put16(p + 208, rand_scale_f16());
            else if (t == 11) put16(p + 108, rand_scale_f16());
            else { put16(p, rand_scale_f16()); if (t == 12 || t == 13) put16(p + 2, rand_scale_f16()); }
        }
        std::vector<float> want((size_t)n);
        for (int64_t b = 0; b < nb; ++b) ht->fn(&w[(size_t)b * bb], &want[(size_t)b * be]);
        uint8_t* dw = dup(w);
        float* d32 = dalloc<float>((size_t)n + 64);
        uint16_t* d16 = dalloc<uint16_t>((size_t)n + 64);
        uint16_t* dbf = dalloc<uint16_t>((size_t)n + 64);
        cudaMemset(d32, 0xFF, ((size_t)n + 64) * 4);
        CHECK(dequant_f32(t, dw, d32, n, 0) == cudaSuccess && sync_ok());
        CHECK(dequant_f16(t, dw, d16, n, 0) == cudaSuccess && sync_ok());
        CHECK(dequant_bf16(t, dw, dbf, n, 0) == cudaSuccess && sync_ok());
        const std::vector<float> g32 = back(d32, (size_t)n + 64);
        const std::vector<uint16_t> g16 = back(d16, (size_t)n), gbf = back(dbf, (size_t)n);
        int bad32 = 0, badbf = 0, bad16 = 0;
        for (int64_t e = 0; e < n; ++e) {
            const double tol = 1e-6 * std::fabs((double)want[(size_t)e]) + 1e-6 * 1e-3;
            if (!(std::fabs((double)g32[(size_t)e] - want[(size_t)e]) <= tol + 1e-5 * std::fabs((double)want[(size_t)e]))) ++bad32;
            if (gbf[(size_t)e] != u63::f2bf(fbits(g32[(size_t)e]))) ++badbf;                     // f2bf of the decoded value
            if (g16[(size_t)e] != baloon::f16bits::f16_from_f32(g32[(size_t)e])) ++bad16;
        }
        // the output tile is exactly rows x cols: nothing past n is written
        bool tail = true;
        for (size_t e = (size_t)n; e < (size_t)n + 64; ++e) tail = tail && fbits(g32[e]) == 0xFFFFFFFFu;
        if ((bad32 || badbf || bad16 || !tail) && g_fail < 40) std::printf("  type %d: f32 %d, bf16 %d, f16 %d mismatches, tail %d\n", t, bad32, badbf, bad16, (int)tail);
        CHECK(bad32 == 0 && badbf == 0 && bad16 == 0 && tail);
        if (t == 42) {   // Q2_0 outputs are exactly (code - 1) * d
            for (int64_t e = 0; e < n; ++e) CHECK(g32[(size_t)e] == want[(size_t)e]);
        }
        // shape gate: a partial block is refused
        CHECK(dequant_f32(t, dw, d32, n - 1, 0) != cudaSuccess);
        cudaGetLastError();
        cudaFree(dw); cudaFree(d32); cudaFree(d16); cudaFree(dbf);
    }
    // i-quant types are not served by the bf16 path
    {
        float* d = dalloc<float>(256); uint8_t* s = dalloc<uint8_t>(256);
        CHECK(dequant_bf16(16, s, (uint16_t*)d, 256, 0) != cudaSuccess);
        cudaGetLastError(); cudaFree(d); cudaFree(s);
    }
}

static void test_s2_canonical() {
    using namespace baloon::kg;
    // plane addressing: block b element j reads global element b*64 + j (no bound on j needed); outputs injective
    std::vector<uint8_t> plane(4096); for (auto& x : plane) x = (uint8_t)rng();
    auto pc = [&](uint32_t i) { return (uint32_t)(plane[i / 4] >> ((i % 4) * 2) & 3); };
    for (uint32_t b = 0; b < 60; ++b) for (uint32_t j = 0; j < 64; ++j) {
        CHECK(((plane[b * 16 + j / 4] >> ((j % 4) * 2)) & 3) == pc(b * 64 + j));
        CHECK(b * 16 <= b * 16 + j / 4 && b * 16 + j / 4 < (b + 1) * 16);
    }
    for (int code = 0; code < 4; ++code) { const int m = code - 1; CHECK(m == -1 || m == 0 || m == 1 || m == 2); }
    for (uint32_t nb = 1; nb < 5000; nb += 37) {
        const uint32_t g = (nb + 255) / 256; CHECK(nb <= g * 256 && (g - 1) * 256 < nb);
    }
    for (long long nb : {1LL, 2LL, 255LL, 256LL, 257LL, 1000LL}) {
        std::vector<uint8_t> codes((size_t)nb * 16); for (auto& x : codes) x = (uint8_t)rng();
        std::vector<float> sc((size_t)nb);
        for (auto& s : sc) s = (float)((int)(rng() % 2001) - 1000) / 37.f;
        sc[0] = 0.f;
        uint8_t* dc = dup(codes); float* ds = dup(sc);
        float* dout = dalloc<float>((size_t)nb * 64 + 64);
        cudaMemset(dout, 0xFF, ((size_t)nb * 64 + 64) * 4);
        CHECK(dequant_s2(dc, ds, dout, nb, 0) == cudaSuccess && sync_ok());
        const std::vector<float> o = back(dout, (size_t)nb * 64 + 64);
        int bad = 0;
        for (long long b = 0; b < nb; ++b) for (int j = 0; j < 64; ++j) {
            const uint32_t i = (uint32_t)(b * 64 + j);
            const int code = (codes[i / 4] >> ((i % 4) * 2)) & 3;
            const float want = (float)(code - 1) * sc[(size_t)b];   // ONE multiply of the integer multiplier
            if (fbits(o[(size_t)i]) != fbits(want)) ++bad;
            if (code == 1 && o[(size_t)i] != 0.f) ++bad;            // code 1 is an exact zero
        }
        for (size_t e = (size_t)nb * 64; e < o.size(); ++e) if (fbits(o[e]) != 0xFFFFFFFFu) ++bad;   // nothing past the plane
        if (bad && g_fail < 40) std::printf("  dequant_s2 n_blocks=%lld: %d mismatches\n", nb, bad);
        CHECK(bad == 0);
        // n_blocks <= 0 launches nothing
        cudaMemset(dout, 0xFF, 64 * 4);
        CHECK(dequant_s2(dc, ds, dout, 0, 0) == cudaSuccess && sync_ok());
        CHECK(dequant_s2(dc, ds, dout, -3, 0) == cudaSuccess && sync_ok());
        const std::vector<float> z = back(dout, 64);
        for (float v : z) CHECK(fbits(v) == 0xFFFFFFFFu);
        cudaFree(dc); cudaFree(ds); cudaFree(dout);
    }
    // the interleaved Q2_0 block form (fp16 scale + 16 code bytes) decodes to exactly {-d, 0, d, 2d}
    {
        const long long nb = 300;
        std::vector<uint8_t> blk((size_t)nb * 18); for (auto& x : blk) x = (uint8_t)rng();
        for (long long b = 0; b < nb; ++b) put16(&blk[(size_t)b * 18], rand_scale_f16());
        uint8_t* db = dup(blk); float* dout = dalloc<float>((size_t)nb * 64);
        CHECK(dequant_q2_0(db, dout, nb, 0) == cudaSuccess && sync_ok());
        const std::vector<float> o = back(dout, (size_t)nb * 64);
        int bad = 0;
        for (long long b = 0; b < nb; ++b) {
            float ref[64]; baloon::artifact::dequantize_q2_0(&blk[(size_t)b * 18], ref);
            const float d = baloon::artifact::fp16_to_fp32(baloon::artifact::read_u16(&blk[(size_t)b * 18]));
            for (int j = 0; j < 64; ++j) {
                const float v = o[(size_t)(b * 64 + j)];
                if (v != ref[j] || !(v == -d || v == 0.f || v == d || v == 2 * d)) ++bad;
            }
        }
        CHECK(bad == 0);
        cudaFree(db); cudaFree(dout);
    }
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("no GPU device available\n"); return 1; }
    if (cudaSetDevice(0) != cudaSuccess) { std::printf("cudaSetDevice(0) failed\n"); return 1; }
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    std::printf("device 0: %s (%d devices)\n", p.name, ndev);
    test_ggml_block_dequant(); std::printf("[ggml_block_dequant] done, failures so far %d\n", g_fail);
    test_s2_canonical();       std::printf("[s2_canonical_dequant] done, failures so far %d\n", g_fail);
    std::printf("u63af gpu: %d checks, %d failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}


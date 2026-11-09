// Unit ub6cd72ef49 invariants: fp16 bits (host == device), block dequantizers, GGUF header arithmetic,
// shard/expert/row geometry, BF16 widening, byte<->unicode map, weight segment accounting and chunking.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <set>
#include <cmath>
#include <cstring>
#include <string>
#include <chrono>
#include <vector>
#include "baloon/f16_bits.hpp"
#include "baloon/artifact_dequant.hpp"
#include "baloon/gguf_lazy_loader.hpp"
#include "baloon/gguf_reader.hpp"
#include "baloon/core_weights.hpp"
#include "baloon/byte_level_bpe_tokenizer.hpp"

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

using namespace baloon;

__global__ void k_f16(const float* in, uint16_t* out, const uint16_t* hin, float* hout, int n, int nh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f16bits::f16_from_f32(in[i]);
    if (i < nh) hout[i] = f16bits::f32_from_f16(hin[i]);
}


static void put_le(std::vector<uint8_t>& b, uint64_t v, int n) { for (int i = 0; i < n; ++i) b.push_back((uint8_t)(v >> (8 * i))); }
static void put_str(std::vector<uint8_t>& b, const std::string& t) { put_le(b, t.size(), 8); b.insert(b.end(), t.begin(), t.end()); }
static void put_kv_u32(std::vector<uint8_t>& b, const std::string& k, uint32_t v) { put_str(b, k); put_le(b, 4, 4); put_le(b, v, 4); }
static uint64_t g_last_hdr_end = 0;
static uint64_t hdr_end_of(const std::vector<uint8_t>&) { return g_last_hdr_end; }
// GGUF v3 image with the six guard keys; drop bit i omits key i (0 experts,1 used,2 heads,3 kv,4 layers,5 hidden).
static std::vector<uint8_t> make_guard_img(const std::string& arch, uint32_t ex, uint32_t used, uint32_t nh, uint32_t nkv,
                                           uint32_t nl, uint32_t hid, unsigned drop = 0, const std::string& pad = "") {
    std::vector<uint8_t> b; put_le(b, 0x46554747u, 4); put_le(b, 3, 4); put_le(b, 0, 8);
    put_le(b, 1 + 6 - __builtin_popcount(drop) + (pad.empty() ? 0 : 1), 8);
    put_str(b, "general.architecture"); put_le(b, 8, 4); put_str(b, arch);
    if (!pad.empty()) { put_str(b, "x.pad"); put_le(b, 8, 4); put_str(b, pad); }
    const uint32_t v[6] = {ex, used, nh, nkv, nl, hid};
    const char* k[6] = {".expert_count", ".expert_used_count", ".attention.head_count", ".attention.head_count_kv", ".block_count", ".embedding_length"};
    for (int i = 0; i < 6; ++i) if (!(drop & (1u << i))) put_kv_u32(b, arch + k[i], v[i]);
    g_last_hdr_end = b.size();
    while (b.size() % 32) b.push_back(0);
    return b;
}
static bool guard_check(const std::vector<uint8_t>& im, std::string* why) {
    auto g = gguf_reader::GgufFile::parse(im.data(), im.size());
    return g && gguf_reader::Qwen4ExpGuard{}.check(*g, why);
}
static uint32_t g_rng = 987654321u;
static uint8_t rnd8() { g_rng = g_rng * 1664525u + 1013904223u; return (uint8_t)(g_rng >> 24); }
static bool near(float a, float b) { return std::fabs(a - b) <= 1e-5f * std::fmax(1.f, std::fabs(b)); }
static void set_h(uint8_t* p, uint16_t h) { p[0] = (uint8_t)h; p[1] = (uint8_t)(h >> 8); }
static float H(uint16_t h) { return f16bits::f32_from_f16(h); }

static void test_arch_guard() {
    std::string w;
    // only GGUF version 3 is handled: every other version is refused by the reader
    for (uint32_t ver : {0u, 1u, 2u, 4u, 0x03000000u, 0xFFFFFFFFu}) {
        auto im = make_guard_img("qwen4exp", 512, 10, 24, 2, 48, 2560);
        for (int i = 0; i < 4; ++i) im[4 + i] = (uint8_t)(ver >> (8 * i));
        CHECK(!gguf_reader::GgufFile::parse(im.data(), im.size()));
    }
    CHECK(guard_check(make_guard_img("qwen4exp", 512, 10, 24, 2, 48, 2560), &w));
    // any other architecture string is refused, even qwen-prefixed ones with identical constants
    for (const char* a : {"qwen35moe", "qwen2moe", "qwen3moe", "qwen", "qwen4exp2", "qwen4ex", "Qwen4exp", "llama", ""})
        CHECK(!guard_check(make_guard_img(a, 512, 10, 24, 2, 48, 2560), &w));
    // every one of the six constants is individually enforced, and every key is required
    uint32_t good[6] = {512, 10, 24, 2, 48, 2560};
    for (int i = 0; i < 6; ++i) {
        for (int delta : {-1, 1}) {
            uint32_t v[6]; std::memcpy(v, good, sizeof v); v[i] += delta;
            CHECK(!guard_check(make_guard_img("qwen4exp", v[0], v[1], v[2], v[3], v[4], v[5]), &w));
        }
        CHECK(!guard_check(make_guard_img("qwen4exp", 512, 10, 24, 2, 48, 2560, 1u << i), &w));
    }
    // refusal reasons
    CHECK(!guard_check(make_guard_img("qwen35moe", 512, 10, 24, 2, 48, 2560), &w) && w == "architecture is not qwen4exp");
    CHECK(!guard_check(make_guard_img("qwen4exp", 512, 10, 24, 2, 48, 2048), &w) && w == "embedding_length mismatch");
    // CLI-level check_architecture is exact-match too
    auto im = make_guard_img("qwen4exp", 512, 10, 24, 2, 48, 2560);
    auto g = gguf_reader::GgufFile::parse(im.data(), im.size());
    CHECK(g && gguf_reader::check_architecture(*g, "qwen4exp") && !gguf_reader::check_architecture(*g, "qwen"));
}

static void test_block_dequantizers() {
    const uint16_t ONE = 0x3C00, HALF = 0x3800;
    // q4_0: y[j] = lo-8, y[j+16] = hi-8
    { uint8_t b[18] = {}; set_h(b, ONE); for (int j = 0; j < 16; ++j) b[2 + j] = (uint8_t)(j | ((15 - j) << 4));
      float y[32]; artifact::dequantize_q4_0(b, y);
      for (int j = 0; j < 16; ++j) { CHECK(y[j] == (float)(j - 8)); CHECK(y[16 + j] == (float)(7 - j)); } }
    // q5_0: element e = nibble + 16*bit(e) - 16, bit(e) is bit e of the 32-bit high word
    { uint8_t b[22] = {}; set_h(b, HALF); uint32_t qh = 0xA5C3F00Fu; for (int i = 0; i < 4; ++i) b[2 + i] = (uint8_t)(qh >> (8 * i));
      for (int j = 0; j < 16; ++j) b[6 + j] = (uint8_t)((j * 7 + 3) & 0xFF);
      float y[32]; artifact::dequantize_q5_0(b, y);
      for (int e = 0; e < 32; ++e) { int nib = e < 16 ? (b[6 + e] & 15) : (b[6 + e - 16] >> 4);
        CHECK(y[e] == 0.5f * (float)(nib + 16 * (int)((qh >> e) & 1) - 16)); } }
    // q2_0: 64 elements, 4 two-bit codes per byte (lowest first), symbol = code-1, scaled by d
    { uint8_t b[18] = {}; set_h(b, HALF); for (int j = 0; j < 16; ++j) b[2 + j] = (uint8_t)(0xE4 ^ (j & 1 ? 0x1B : 0));
      float y[64]; artifact::dequantize_q2_0(b, y);
      for (int e = 0; e < 64; ++e) CHECK(y[e] == 0.5f * (float)(((b[2 + e / 4] >> (2 * (e % 4))) & 3) - 1)); }
    // IQ4_XS (random data, independent per-element formula)
    for (int rep = 0; rep < 20; ++rep) {
        uint8_t b[136]; for (auto& x : b) x = rnd8(); set_h(b, 0x3400 | (rnd8() & 0x3F));
        float y[256]; artifact::dequantize_iq4_xs(b, y);
        uint16_t sh = (uint16_t)(b[2] | (b[3] << 8));
        for (int e = 0; e < 256; ++e) { int ib = e / 32, j = e % 32;
            int ls = ((b[4 + ib / 2] >> (4 * (ib & 1))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
            uint8_t q = b[8 + 16 * ib + (j % 16)]; int code = j < 16 ? (q & 15) : (q >> 4);
            CHECK(near(y[e], H((uint16_t)(b[0] | (b[1] << 8))) * (float)(ls - 32) * (float)artifact::kvalues_iq4nl[code])); }
    }
    // Q4_K / Q5_K (random data)
    for (int rep = 0; rep < 20; ++rep) for (int five = 0; five < 2; ++five) {
        uint8_t b[176]; for (auto& x : b) x = rnd8(); set_h(b, 0x3000 | (rnd8() & 0x3F)); set_h(b + 2, 0x3000 | (rnd8() & 0x3F));
        const uint8_t* sc = b + 4; const uint8_t* qh = b + 16; const uint8_t* qs = b + (five ? 48 : 16);
        float y[256]; if (five) artifact::dequantize_q5_K(b, y); else artifact::dequantize_q4_K(b, y);
        float d = H((uint16_t)(b[0] | (b[1] << 8))), dm = H((uint16_t)(b[2] | (b[3] << 8)));
        for (int e = 0; e < 256; ++e) { int g = e / 32, l = e % 32, j = g / 2;
            int s6, m6;
            if (g < 4) { s6 = sc[g] & 63; m6 = sc[g + 4] & 63; }
            else { s6 = (sc[g + 4] & 15) | ((sc[g - 4] >> 6) << 4); m6 = (sc[g + 4] >> 4) | ((sc[g] >> 6) << 4); }
            uint8_t q = qs[32 * j + l]; int v = (g & 1) ? (q >> 4) : (q & 15);
            if (five) v += ((qh[l] >> (2 * j + (g & 1))) & 1) ? 16 : 0;
            CHECK(near(y[e], d * (float)s6 * (float)v - dm * (float)m6)); }
    }
    // Q3_K
    for (int rep = 0; rep < 20; ++rep) {
        uint8_t b[110]; for (auto& x : b) x = rnd8(); set_h(b + 108, 0x3000 | (rnd8() & 0x3F));
        const uint8_t* hm = b; const uint8_t* q = b + 32; const uint8_t* s = b + 96;
        float y[256]; artifact::dequantize_q3_K(b, y);
        float d = H((uint16_t)(b[108] | (b[109] << 8)));
        for (int e = 0; e < 256; ++e) { int n = e / 128, r = e % 128, j = r / 32, half = (r % 32) / 16, l = r % 16;
            int si = 8 * n + 2 * j + half, grp = si / 4, k = si % 4;
            int lo = ((grp & 1) ? s[4 + k] : s[k]) >> ((grp >> 1) ? 4 : 0) & 15, hi = (s[8 + k] >> (2 * grp)) & 3;
            int scale = (lo | (hi << 4)) - 32;
            int code = (q[32 * n + 16 * half + l] >> (2 * j)) & 3;
            int bit = (hm[16 * half + l] >> (4 * n + j)) & 1;
            CHECK(near(y[e], d * (float)scale * (float)(code - (bit ? 0 : 4)))); }
    }
    // Q6_K
    for (int rep = 0; rep < 20; ++rep) {
        uint8_t b[210]; for (auto& x : b) x = rnd8(); set_h(b + 208, 0x3000 | (rnd8() & 0x3F));
        const uint8_t* ql = b; const uint8_t* qh = b + 128; const int8_t* sc = (const int8_t*)(b + 192);
        float y[256]; artifact::dequantize_q6_K(b, y);
        float d = H((uint16_t)(b[208] | (b[209] << 8)));
        for (int e = 0; e < 256; ++e) { int h = e / 128, r = e % 128, qd = r / 32, l = r % 32;
            int lo = (ql[64 * h + l + (qd & 1) * 32] >> (4 * (qd >> 1))) & 15, hi = (qh[32 * h + l] >> (2 * qd)) & 3;
            int q = (lo | (hi << 4)) - 32;
            CHECK(near(y[e], d * (float)sc[8 * h + l / 16 + 2 * qd] * (float)q)); }
    }
    // the block table is consistent with the GGUF geometry table
    size_t n; const artifact::BlockType* t = artifact::block_types(&n);
    CHECK(n == 10);
    for (size_t i = 0; i < n; ++i) { bool found = false;
        for (uint32_t ty = 0; ty < 64 && !found; ++ty) { auto g = gguf::block_geometry(ty); found = g.el == t[i].elems && g.by == t[i].bytes; }
        CHECK(found);
        // table entry == direct dequantizer on random bytes
        uint8_t blk[256]; for (auto& x : blk) x = rnd8(); blk[1] &= 0x3B; blk[0] &= 0xFF;
        float a[256], c[256]; t[i].fn(blk, a);
        const artifact::BlockType* f = artifact::find_block_type(t[i].name); CHECK(f == &t[i]);
        f->fn(blk, c); CHECK(std::memcmp(a, c, t[i].elems * 4) == 0); }
}

static void test_gpu_f16() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("no GPU device\n"); ++g_fail; return; }
    cudaSetDevice(0);
    std::vector<float> in; std::vector<uint16_t> hin(65536);
    uint32_t s = 12345u;
    for (int i = 0; i < 200000; ++i) { s = s * 1664525u + 1013904223u; in.push_back(f16bits::float_of(s)); }
    for (float f : {0.f, -0.f, 1.f, 65504.f, 65520.f, 1e-8f, 6e-8f, 3e-5f, 1e30f}) in.push_back(f);
    for (int i = 0; i < 65536; ++i) hin[i] = (uint16_t)i;
    int n = (int)in.size();
    float *din, *dho; uint16_t *dout, *dhi;
    CHECK(cudaMalloc(&din, n * 4) == cudaSuccess); CHECK(cudaMalloc(&dout, n * 2) == cudaSuccess);
    CHECK(cudaMalloc(&dhi, 65536 * 2) == cudaSuccess); CHECK(cudaMalloc(&dho, 65536 * 4) == cudaSuccess);
    cudaMemcpy(din, in.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dhi, hin.data(), 65536 * 2, cudaMemcpyHostToDevice);
    k_f16<<<(n + 255) / 256, 256>>>(din, dout, dhi, dho, n, 65536);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<uint16_t> o(n); std::vector<float> ho(65536);
    cudaMemcpy(o.data(), dout, n * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(ho.data(), dho, 65536 * 4, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < n; ++i) bad += (o[i] != f16bits::f16_from_f32(in[i]));
    CHECK(bad == 0);
    bad = 0;
    for (int i = 0; i < 65536; ++i) {
        float h = f16bits::f32_from_f16((uint16_t)i);
        if (std::memcmp(&h, &ho[i], 4) != 0 && !(h != h && ho[i] != ho[i])) ++bad;
    }
    CHECK(bad == 0);
    // keep the device open and busy for a couple of seconds, re-checking parity on every pass
    auto t0 = std::chrono::steady_clock::now();
    int passes = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 2.0) {
        k_f16<<<(n + 255) / 256, 256>>>(din, dout, dhi, dho, n, 65536);
        CHECK(cudaDeviceSynchronize() == cudaSuccess);
        cudaMemcpy(o.data(), dout, n * 2, cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; i += 997) CHECK(o[i] == f16bits::f16_from_f32(in[i]));
        ++passes;
    }
    CHECK(passes > 0);
    cudaFree(din); cudaFree(dout); cudaFree(dhi); cudaFree(dho);
}

static void test_f16_host() {
    // round trip: every non-NaN half survives widen -> narrow; NaN stays NaN; overflow saturates to inf (never NaN)
    for (uint32_t i = 0; i < 65536; ++i) {
        uint16_t h = (uint16_t)i; float f = f16bits::f32_from_f16(h);
        bool nan = ((h >> 10) & 31) == 31 && (h & 1023);
        uint16_t b = f16bits::f16_from_f32(f);
        if (nan) CHECK(((b >> 10) & 31) == 31 && (b & 1023)); else CHECK(b == h);
    }
    CHECK(f16bits::f16_from_f32(1e30f) == 0x7C00);
    CHECK(f16bits::f16_from_f32(-1e30f) == 0xFC00);
    CHECK(f16bits::f16_from_f32(65520.f) == 0x7C00);          // rounds up past max -> inf
    CHECK(f16bits::f16_from_f32(65504.f) == 0x7BFF);
    // bf16 widening: bits << 16 (low 16 bits zero, high 16 == pattern)
    for (uint32_t h = 0; h < 65536; h += 7) {
        float f = artifact::bf16_to_fp32((uint16_t)h); uint32_t u = f16bits::bits_of(f);
        CHECK((u & 0xFFFFu) == 0 && (u >> 16) == h && u == h * 65536u);
    }
}

static void test_dequant() {
    CHECK(artifact::kvalues_iq4nl[0] == -127 && artifact::kvalues_iq4nl[15] == 113);
    uint8_t buf[4] = {0x34, 0x12, 0, 0};
    CHECK(artifact::read_u16(buf) == 0x1234);
    // geometry table: 13 encodings, all positive, expected bytes
    struct G { uint32_t t; uint64_t el, by; } want[] = {
        {gguf::kF32, 1, 4}, {gguf::kF16, 1, 2}, {gguf::kBF16, 1, 2}, {gguf::kQ4_0, 32, 18}, {gguf::kQ5_0, 32, 22},
        {gguf::kQ8_0, 32, 34}, {gguf::kIQ4_NL, 32, 18}, {gguf::kQ3_K, 256, 110}, {gguf::kQ4_K, 256, 144},
        {gguf::kQ5_K, 256, 176}, {gguf::kQ6_K, 256, 210}, {gguf::kIQ4_XS, 256, 136}, {gguf::kQ2_0, 64, 18}};
    CHECK(sizeof(want) / sizeof(want[0]) == 13);
    for (auto& w : want) { auto g = gguf::block_geometry(w.t); CHECK(g.el == w.el && g.by == w.by && g.el > 0 && g.by > 0); }
    CHECK(gguf::block_geometry(9999).el == 0);
    // Q8_0 block: scale 0.5 (fp16 0x3800), codes -3..: y = d * q
    uint8_t blk[34] = {0x00, 0x38};
    for (int i = 0; i < 32; ++i) blk[2 + i] = (uint8_t)(int8_t)(i - 16);
    float y[32]; artifact::dequantize_q8_0(blk, y);
    for (int i = 0; i < 32; ++i) CHECK(y[i] == 0.5f * (float)(i - 16));
    // IQ4_NL block: scale 1.0 (0x3C00); nibble k -> kvalues[k]; low nibbles are elements 0..15, high 16..31
    uint8_t nl[18] = {0x00, 0x3C};
    for (int i = 0; i < 16; ++i) nl[2 + i] = (uint8_t)(i | ((15 - i) << 4));
    float z[32]; artifact::dequantize_iq4_nl(nl, z);
    for (int i = 0; i < 16; ++i) { CHECK(z[i] == (float)artifact::kvalues_iq4nl[i]); CHECK(z[16 + i] == (float)artifact::kvalues_iq4nl[15 - i]); }
    // f16/bf16/f32 row dequantizers
    uint8_t h2[4] = {0x00, 0x3C, 0x00, 0xC0}; float hf[2]; artifact::dequantize_f16(h2, hf, 2);
    CHECK(hf[0] == 1.f && hf[1] == -2.f);
    uint8_t b2[4] = {0x80, 0x3F, 0x00, 0xC0}; artifact::dequantize_bf16(b2, hf, 2);
    CHECK(hf[0] == 1.f && hf[1] == -2.f);
    float src[2] = {3.5f, -7.25f}; artifact::dequantize_f32((const uint8_t*)src, hf, 2);
    CHECK(hf[0] == 3.5f && hf[1] == -7.25f);
}

static void test_gguf_header() {
    using gguf_reader::Cursor;
    uint8_t mem[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    Cursor c(mem, 16);
    CHECK(c.need(16) && c.pos == 0);
    uint32_t a = c.u32(); CHECK(a == 0x04030201u && c.pos == 4);          // successful read advances by exactly n
    c.u64(); CHECK(c.pos == 12 && c.ok);
    CHECK(c.pos <= c.n);
    c.u64(); CHECK(!c.ok && c.pos == 12);                                   // overrun fails, no OOB advance
    Cursor d(mem, 16); d.pos = 16; CHECK(d.need(0)); CHECK(!d.need(1));
    // data_start = ceil(pos/align)*align : >= pos, < pos+align, multiple of align, idempotent on multiples
    // data_start of really parsed images: aligned, >= end of the header, < end + align, tensor data fits
    for (size_t pad = 0; pad < 70; ++pad) {
        std::vector<uint8_t> im = make_guard_img("qwen4exp", 512, 10, 24, 2, 48, 2560, 0, std::string(pad, 'p'));
        auto g = gguf_reader::GgufFile::parse(im.data(), im.size());
        CHECK(g.has_value());
        if (g) CHECK(g->data_start % 32 == 0 && g->data_start <= im.size() && g->data_start + 32 > hdr_end_of(im));
    }
    CHECK(gguf_reader::kMaxArrayElems == (1ull << 24));
    CHECK(gguf_reader::kDefaultAlignment == 32);
}


static void test_shard_loader() {
    for (uint64_t el : {1ull, 32ull, 64ull, 256ull}) {
        uint64_t by = el == 1 ? 4 : el * 3 / 8 + 2;
        for (uint64_t per : {256ull * 64, 256ull * 128}) {
            if (per % el) continue;
            for (uint64_t e = 0; e < 40; ++e) {
                // experts tile: e+1 starts where e ends; start == e * bytes; python spelling equal
                CHECK(gguf::expert_start(e + 1, per, el, by) == gguf::expert_start(e, per, el, by) + gguf::expert_bytes(per, el, by));
                CHECK(gguf::expert_start(e, per, el, by) == gguf::expert_start_py(e, per, el, by));
                CHECK(gguf::expert_start(e, per, el, by) + gguf::expert_bytes(per, el, by) <= 40 * gguf::expert_bytes(per, el, by));
            }
            CHECK(gguf::tensor_bytes(40 * per, el, by) == 40 * gguf::expert_bytes(per, el, by));
        }
    }
    CHECK(gguf::ple_row_bytes() == 90);
    CHECK(gguf::whole_blocks(160, 32) && !gguf::whole_blocks(161, 32));
    CHECK(gguf::shards_complete(5, 5) && !gguf::shards_complete(4, 5) && !gguf::shards_complete(6, 5));
    // shard discovery over a file list
    std::vector<std::string> names = {"m-00001-of-00003.gguf", "m-00002-of-00003.gguf", "m-00003-of-00003.gguf"};
    auto ok = gguf::discover_shards_in(names, names[0]); CHECK(ok && ok->size() == 3);
    names.pop_back(); auto miss = gguf::discover_shards_in(names, names[0]); CHECK(!miss);
}

static void test_tokenizer_map() {
    const auto& m = bpe::byte_map();
    std::set<uint32_t> seen; int self = 0, shifted = 0; uint32_t expect = 256;
    for (unsigned b = 0; b < 256; ++b) {
        seen.insert(m[b]);
        bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        if (printable) { CHECK(m[b] == b); ++self; }
        else { CHECK(m[b] == expect++); ++shifted; CHECK(m[b] >= 256 && m[b] < 324); }
        CHECK(bpe::codepoint_to_byte(m[b]) == (int)b);
    }
    CHECK(seen.size() == 256 && self == 188 && shifted == 68);
    // decode(encode(s)) == s for every byte, including concatenation (homomorphism)
    std::string all; for (int b = 0; b < 256; ++b) all.push_back((char)b);
    bool okk = false; std::string mapped = bpe::bytes_to_mapped(all);
    CHECK(bpe::mapped_to_bytes(mapped, &okk) == all && okk);
    std::string a = all.substr(0, 100), b2 = all.substr(100);
    bool o1 = false, o2 = false;
    CHECK(bpe::mapped_to_bytes(bpe::bytes_to_mapped(a) + bpe::bytes_to_mapped(b2), &o1) ==
          bpe::mapped_to_bytes(bpe::bytes_to_mapped(a), &o1) + bpe::mapped_to_bytes(bpe::bytes_to_mapped(b2), &o2));
    bool oe = false; CHECK(bpe::mapped_to_bytes("", &oe).empty());
}

static void test_weight_segments() {
    // fp16 scale plane: source half the destination; segment totals consistent
    for (uint64_t codes : {0ull, 256ull, 4096ull}) for (uint64_t sc : {0ull, 2ull, 64ull, 1024ull}) for (uint64_t off : {0ull, 16ull}) {
        core::WIndexRow r; r.codes = codes; r.scales = sc; r.offset = off; r.scales_fp16 = true;
        r.src_bytes = codes + sc / 2 + off; r.dst_bytes = codes + sc + off;
        std::vector<core::WSeg> segs;
        CHECK(core::plan_segments(r, segs));
        uint64_t s = 0, d = 0; for (auto& g : segs) { s += g.src_bytes; d += g.dst_bytes; }
        CHECK(s == r.src_bytes && d == r.dst_bytes);
        CHECK(d == s + sc / 2);
        core::WIndexRow bad = r; bad.src_bytes += 1; std::vector<core::WSeg> sg2;
        CHECK(!core::plan_segments(bad, sg2));                                // totals mismatch refused
        core::WIndexRow cp = r; cp.scales_fp16 = false; cp.src_bytes = cp.dst_bytes = codes + sc + off;
        std::vector<core::WSeg> sg3; CHECK(core::plan_segments(cp, sg3));
    }
    // chunking: <= remainder, <= 8 MiB, widening chunk even and positive while >= 2 bytes remain
    for (uint64_t rem : {0ull, 1ull, 2ull, 3ull, 1000ull, 8388607ull, 8388608ull, 8388609ull, 20000001ull}) {
        uint64_t c = core::chunk_len_of(rem);
        CHECK(c <= rem && c <= 8388608ull);
        if (rem % 2 == 0) CHECK(c % 2 == 0);
        if (rem >= 2) CHECK(c >= 2);
        uint64_t w = c % 2 ? c - 1 : c; CHECK(w % 2 == 0 && w <= rem && (rem < 2 || w > 0));
        uint64_t left = rem, iters = 0;                                         // widening loop terminates
        while (left >= 2) { uint64_t cc = core::chunk_len_of(left); cc -= cc % 2; CHECK(cc > 0); left -= cc; ++iters; }
        CHECK(left <= 1 && iters <= rem / 2 + 1);
    }
    // compact placement: cursor stays 256-aligned and never overlaps
    uint64_t at = 0;
    for (uint64_t dst : {1ull, 255ull, 256ull, 257ull, 5000ull, 90ull}) {
        uint64_t next = at + core::weight_align256(dst);
        CHECK(next % 256 == 0 && at + dst <= next); at = next;
    }
}

int main() {
    test_gpu_f16();
    test_f16_host();
    test_dequant();
    test_block_dequantizers();
    test_arch_guard();
    test_gguf_header();
    test_shard_loader();
    test_tokenizer_map();
    test_weight_segments();
    if (g_fail) { std::printf("ub6c: %d check(s) FAILED\n", g_fail); return 1; }
    std::printf("ub6c: all checks passed\n");
    return 0;
}

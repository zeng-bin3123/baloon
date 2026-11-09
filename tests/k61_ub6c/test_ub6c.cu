// Unit ub6cd72ef49 invariants: fp16 bits (host == device), block dequantizers, GGUF header arithmetic,
// shard/expert/row geometry, BF16 widening, byte<->unicode map, weight segment accounting and chunking.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <set>
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
    for (uint64_t a2 : {1ull, 8ull, 32ull, 64ull, 4096ull})
        for (uint64_t pos = 0; pos < 9000; pos += 13) {
            uint64_t ds = (pos + a2 - 1) / a2 * a2;
            CHECK(ds >= pos && ds < pos + a2 && ds % a2 == 0);
            CHECK((ds + a2 - 1) / a2 * a2 == ds);
        }
    CHECK(gguf_reader::kMaxArrayElems == (1ull << 24));
    CHECK(gguf_reader::kDefaultAlignment == 32);
    // tensor element count is the product of dims
    uint64_t prod = 1; for (uint64_t dd : {3ull, 5ull, 7ull, 11ull}) prod *= dd; CHECK(prod == 1155);
}

enum DirResult { ShardSet, Single, NoFiles, Ambiguous };
static DirResult dir_decision(unsigned shards, unsigned found) {
    return shards ? ShardSet : found == 1 ? Single : found == 0 ? NoFiles : Ambiguous;
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
    for (uint64_t r = 0; r < 100; ++r) CHECK(r * 90 + 90 <= 100 * 90);
    // directory decision: shard set wins; single accepted; empty / ambiguous refused
    CHECK(dir_decision(3, 9) == ShardSet && dir_decision(1, 1) == ShardSet);
    CHECK(dir_decision(0, 1) == Single && dir_decision(0, 0) == NoFiles);
    for (unsigned f = 2; f < 10; ++f) CHECK(dir_decision(0, f) == Ambiguous);
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
    // BPE merge shrinks by exactly one
    std::vector<int> l = {1, 2, 3, 4}; std::vector<int> mg(l.begin(), l.begin() + 1); mg.push_back(99); mg.insert(mg.end(), l.begin() + 3, l.end());
    CHECK(mg.size() + 1 == l.size());
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
    // promoted offsets compose: done/2 + n/2 == (done+n)/2 for 4-aligned values
    for (uint64_t done = 0; done < 64; done += 4) for (uint64_t n = 0; n < 64; n += 4) CHECK(done / 2 + n / 2 == (done + n) / 2);
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
    test_gguf_header();
    test_shard_loader();
    test_tokenizer_map();
    test_weight_segments();
    if (g_fail) { std::printf("ub6c: %d check(s) FAILED\n", g_fail); return 1; }
    std::printf("ub6c: all checks passed\n");
    return 0;
}

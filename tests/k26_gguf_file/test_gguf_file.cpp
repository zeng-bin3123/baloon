// Unit 26 tests: file-backed lazy GGUF loader invariants (GgufLazyLoader theory), checked against independently built bytes.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>
#include "../unit14/check.hpp"
#include "baloon/gguf_file_loader.hpp"

using namespace baloon;
using gguf_file::Gguf;

static void put32(std::vector<uint8_t>& b, uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 255); }
static void put64(std::vector<uint8_t>& b, uint64_t v) { for (int i = 0; i < 8; ++i) b.push_back((v >> (8 * i)) & 255); }
static void putstr(std::vector<uint8_t>& b, const std::string& s) { put64(b, s.size()); b.insert(b.end(), s.begin(), s.end()); }

struct T { std::string name; std::vector<uint64_t> shape; uint32_t type; std::vector<uint8_t> data; };

static std::vector<uint8_t> pattern(size_t n, unsigned seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = (uint8_t)(seed * 37 + i * 7 + i / 256 * 13);
    return v;
}
// Writes a GGUF v3 file (no metadata, alignment 32) holding `ts`.
static bool write_gguf(const std::string& path, const std::vector<T>& ts) {
    std::vector<uint8_t> h;
    put32(h, 0x46554747u); put32(h, 3); put64(h, ts.size()); put64(h, 0);
    std::vector<uint64_t> offs; uint64_t cur = 0;
    for (auto& t : ts) { offs.push_back(cur); cur += (t.data.size() + 31) / 32 * 32; }
    for (size_t i = 0; i < ts.size(); ++i) {
        putstr(h, ts[i].name); put32(h, (uint32_t)ts[i].shape.size());
        for (auto d : ts[i].shape) put64(h, d);
        put32(h, ts[i].type); put64(h, offs[i]);
    }
    while (h.size() % 32) h.push_back(0);
    for (size_t i = 0; i < ts.size(); ++i) { h.insert(h.end(), ts[i].data.begin(), ts[i].data.end()); while (h.size() % 32) h.push_back(0); }
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(h.data(), 1, h.size(), f) == h.size();
    std::fclose(f);
    return ok;
}

static std::string tmpdir() {
    char tpl[] = "/tmp/k26_gguf_XXXXXX";
    return std::string(mkdtemp(tpl));
}

static void test_geometry_arithmetic() {
    // block geometry of every listed type, including ggml id 42 = Q2_0 (64 elements / 18 bytes)
    struct G { uint32_t t; uint64_t el, by; } g[] = {{0, 1, 4}, {1, 1, 2}, {30, 1, 2}, {2, 32, 18}, {6, 32, 22}, {8, 32, 34}, {20, 32, 18},
                                                     {11, 256, 110}, {12, 256, 144}, {13, 256, 176}, {14, 256, 210}, {23, 256, 136}, {42, 64, 18}};
    for (auto& x : g) { auto b = gguf::block_geometry(x.t); CHECK(b.el == x.el && b.by == x.by); }
    CHECK(gguf::block_geometry(99).el == 0);
    // routed Q2_0 expert tensor [2560, 640, 512]
    const uint64_t per = 2560ull * 640;
    CHECK(gguf::expert_bytes(per, 64, 18) == 460800);
    CHECK(3 * gguf::expert_bytes(per, 64, 18) == 1382400);
    CHECK(512 * gguf::expert_bytes(per, 64, 18) == 235929600ull);
    CHECK(gguf::row_bytes(160, 32, 18) == 90);
}

static bool bytes_eq(const gguf::Span& s, const std::vector<uint8_t>& all, size_t off, size_t len) {
    return s.len == len && off + len <= all.size() && std::memcmp(s.ptr, all.data() + off, len) == 0;
}

static void test_bf16() {
    const uint16_t in[] = {0x3F80, 0xBF80, 0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0x0001, 0x4049};
    float out[9];
    gguf_file::bf16_to_f32(in, 9, out);
    CHECK(out[0] == 1.0f && out[1] == -1.0f && out[2] == 0.0f && !std::signbit(out[2]));
    CHECK(out[3] == 0.0f && std::signbit(out[3]));
    CHECK(std::isinf(out[4]) && out[4] > 0 && std::isinf(out[5]) && out[5] < 0);
    CHECK(std::isnan(out[6]));
    for (int i = 0; i < 9; ++i) { uint32_t u; std::memcpy(&u, &out[i], 4); CHECK(u == (uint32_t)in[i] << 16); }
    CHECK(std::fabs(out[8] - 3.140625f) < 1e-7f);
    // exhaustive: widening then keeping the high half returns the input
    for (uint32_t v = 0; v < 65536; ++v) {
        uint16_t h = (uint16_t)v; float f; gguf_file::bf16_to_f32(&h, 1, &f);
        uint32_t u; std::memcpy(&u, &f, 4); CHECK((u >> 16) == v && (u & 0xFFFF) == 0);
    }
}

static void test_dequant_flat() {
    // F32 / F16 / BF16 / Q8_0 against hand-computed values
    float f32[3] = {1.5f, -2.25f, 1e-3f}; uint8_t raw[12]; std::memcpy(raw, f32, 12);
    float o[64];
    CHECK(gguf_file::dequant_flat(0, raw, 3, o) && o[0] == 1.5f && o[1] == -2.25f && o[2] == 1e-3f);
    const uint8_t h16[] = {0x00, 0x3C, 0x00, 0xC0, 0x00, 0x38};  // 1.0, -2.0, 0.5
    CHECK(gguf_file::dequant_flat(1, h16, 3, o) && o[0] == 1.0f && o[1] == -2.0f && o[2] == 0.5f);
    const uint8_t b16[] = {0x80, 0x3F, 0x00, 0xC0};  // 1.0, -2.0
    CHECK(gguf_file::dequant_flat(30, b16, 2, o) && o[0] == 1.0f && o[1] == -2.0f);
    uint8_t q8[34]; q8[0] = 0x00; q8[1] = 0x38;  // scale 0.5
    for (int i = 0; i < 32; ++i) q8[2 + i] = (uint8_t)(int8_t)(i - 16);
    CHECK(gguf_file::dequant_flat(8, q8, 32, o));
    for (int i = 0; i < 32; ++i) CHECK(o[i] == 0.5f * (float)(i - 16));
    // two blocks: second block scale 2.0
    uint8_t q8b[68]; std::memcpy(q8b, q8, 34); q8b[34] = 0x00; q8b[35] = 0x40;
    for (int i = 0; i < 32; ++i) q8b[36 + i] = (uint8_t)(int8_t)(3);
    float o2[64];
    CHECK(gguf_file::dequant_flat(8, q8b, 64, o2));
    CHECK(o2[0] == -8.0f && o2[32] == 6.0f && o2[63] == 6.0f);
    // partial blocks and unknown types are refused and write nothing
    float sentinel[64]; for (auto& s : sentinel) s = 123.f;
    CHECK(!gguf_file::dequant_flat(8, q8b, 33, sentinel) && sentinel[0] == 123.f);
    CHECK(!gguf_file::dequant_flat(42, q8b, 63, sentinel) && sentinel[0] == 123.f);
    CHECK(!gguf_file::dequant_flat(99, q8b, 32, sentinel) && sentinel[0] == 123.f);
    CHECK(!gguf_file::dequant_flat(0, nullptr, 1, sentinel));
    // Q2_0 (id 42) is a known type: two blocks equal the per-block scalar dequantiser
    std::vector<uint8_t> q2 = pattern(36, 5); q2[0] = 0; q2[1] = 0x3C; q2[18] = 0; q2[19] = 0x38;
    float a[128], b[128];
    CHECK(gguf_file::dequant_flat(42, q2.data(), 128, a));
    artifact::dequantize_q2_0(q2.data(), b); artifact::dequantize_q2_0(q2.data() + 18, b + 64);
    CHECK(std::memcmp(a, b, sizeof a) == 0);
    // every quantised type: flat dequant equals block-by-block dequant
    struct Q { uint32_t t; uint64_t el, by; void (*fn)(const uint8_t*, float*); } qs[] = {
        {2, 32, 18, artifact::dequantize_q4_0}, {6, 32, 22, artifact::dequantize_q5_0}, {20, 32, 18, artifact::dequantize_iq4_nl},
        {11, 256, 110, artifact::dequantize_q3_K}, {12, 256, 144, artifact::dequantize_q4_K}, {13, 256, 176, artifact::dequantize_q5_K},
        {14, 256, 210, artifact::dequantize_q6_K}, {23, 256, 136, artifact::dequantize_iq4_xs}};
    for (auto& q : qs) {
        std::vector<uint8_t> d = pattern(q.by * 3, q.t);
        for (uint64_t blk = 0; blk < 3; ++blk) { d[blk * q.by] = 0; d[blk * q.by + 1] = 0x30; }  // finite small fp16 scale
        std::vector<float> flat(q.el * 3), ref(q.el * 3);
        CHECK(gguf_file::dequant_flat(q.t, d.data(), q.el * 3, flat.data()));
        for (uint64_t blk = 0; blk < 3; ++blk) q.fn(d.data() + blk * q.by, ref.data() + blk * q.el);
        CHECK(std::memcmp(flat.data(), ref.data(), flat.size() * 4) == 0);
    }
}

static void test_files() {
    const std::string dir = tmpdir();
    // tensors: shard 1: w (F32 [4,3]), moe (Q8_0 [64,2,5]: 128 elems / expert = 4 blocks = 136 bytes)
    //          shard 2: emb (IQ4_NL [160,7]: 90-byte rows), bf (BF16 [8,2]), odd (Q8_0 [16,4]: width not a block multiple, 64 elems)
    T w{"w", {4, 3}, 0, pattern(48, 1)};
    T moe{"moe", {64, 2, 5}, 8, pattern(5 * 136, 2)};
    T emb{"emb", {160, 7}, 20, pattern(7 * 90, 3)};
    T bf{"bf", {8, 2}, 30, {}};
    for (int i = 0; i < 16; ++i) { uint16_t v = (uint16_t)(0x3F00 + i * 16); bf.data.push_back(v & 255); bf.data.push_back(v >> 8); }
    T odd{"odd", {16, 4}, 8, pattern(2 * 34, 4)};
    // fix scales of the Q8_0 data so dequantisation is finite
    for (size_t b = 0; b < 20; ++b) { moe.data[b * 34] = 0; moe.data[b * 34 + 1] = 0x38; }
    const std::string p1 = dir + "/m-00001-of-00002.gguf", p2 = dir + "/m-00002-of-00002.gguf";
    CHECK(write_gguf(p1, {w, moe}));
    CHECK(write_gguf(p2, {emb, bf, odd}));

    // --- shard discovery: complete set found in index order, from either member, missing sibling rejected
    auto s = gguf_file::discover_shards(p1);
    CHECK(s && s->size() == 2 && (*s)[0] == p1 && (*s)[1] == p2);
    auto s2 = gguf_file::discover_shards(p2);
    CHECK(s2 && s2->size() == 2 && (*s2)[0] == p1);
    CHECK(!gguf_file::discover_shards(dir + "/nothing-00001-of-00002.gguf"));
    CHECK(!gguf_file::discover_shards(dir + "/missing.gguf"));        // plain file that does not exist
    {
        const std::string q1 = dir + "/x-00001-of-00003.gguf", q3 = dir + "/x-00003-of-00003.gguf";
        CHECK(write_gguf(q1, {w}) && write_gguf(q3, {w}));
        CHECK(!gguf_file::discover_shards(q1));                       // 00002 missing
        Gguf g; std::string err; CHECK(!g.open(q1, &err) && !err.empty());
        const std::string q2p = dir + "/x-00002-of-00003.gguf";
        T w2{"w2", {4, 3}, 0, pattern(48, 9)}; T w3{"w3", {4, 3}, 0, pattern(48, 8)};
        CHECK(write_gguf(q1, {w}) && write_gguf(q2p, {w2}) && write_gguf(q3, {w3}));
        CHECK(gguf_file::discover_shards(q1) && gguf_file::discover_shards(q1)->size() == 3);
    }
    const std::string single = dir + "/single.gguf";
    CHECK(write_gguf(single, {w}));
    auto s3 = gguf_file::discover_shards(single);
    CHECK(s3 && s3->size() == 1 && (*s3)[0] == single);

    // --- the reader
    Gguf g; std::string err;
    CHECK(g.open(p1, &err));
    CHECK(g.shard_count() == 2 && g.has("w") && g.has("moe") && g.has("emb") && g.has("bf") && !g.has("zzz"));
    auto tw = g.tensor("w"); CHECK(tw && bytes_eq(*tw, w.data, 0, 48));
    CHECK(!g.tensor("zzz"));
    // experts: adjacent, first at 0, last ends at the tensor end, per-expert bytes = per / el * by
    for (uint64_t e = 0; e < 5; ++e) { auto sp = g.expert("moe", e); CHECK(sp && bytes_eq(*sp, moe.data, e * 136, 136)); }
    auto e0 = g.expert("moe", 0), e1 = g.expert("moe", 1), e4 = g.expert("moe", 4);
    CHECK(e0 && e1 && e0->ptr + e0->len == e1->ptr);
    CHECK(e4 && e4->ptr + e4->len == g.tensor("moe")->ptr + g.tensor("moe")->len);
    CHECK(!g.expert("moe", 5) && !g.expert("moe", 1000) && !g.expert("nope", 0));
    std::vector<float> ef;
    CHECK(g.expert_f32("moe", 2, ef) && ef.size() == 128);
    for (int b = 0; b < 4; ++b) for (int i = 0; i < 32; ++i) {
        const float want = 0.5f * (float)(int8_t)moe.data[2 * 136 + b * 34 + 2 + i];
        CHECK(ef[b * 32 + i] == want);
    }
    // rows: row_bytes = width / el * by, order preserved, repeats allowed, out of range rejected
    std::vector<uint8_t> rb;
    CHECK(g.rows("emb", {6, 0, 3, 3}, rb) && rb.size() == 360);
    CHECK(std::memcmp(rb.data(), emb.data.data() + 6 * 90, 90) == 0);
    CHECK(std::memcmp(rb.data() + 90, emb.data.data(), 90) == 0);
    CHECK(std::memcmp(rb.data() + 180, emb.data.data() + 3 * 90, 90) == 0 && std::memcmp(rb.data() + 270, emb.data.data() + 270, 90) == 0);
    CHECK(!g.rows("emb", {7}, rb) && rb.empty());
    CHECK(!g.rows("emb", {0, 7}, rb));
    CHECK(g.rows("emb", {}, rb) && rb.empty());
    CHECK(!g.rows("moe", {0}, rb));                                   // 3-D tensor
    CHECK(!g.rows("odd", {0}, rb));                                   // width 16 is not a multiple of 32
    CHECK(!g.expert("odd", 0));                                       // 16 elements per expert: not a whole block
    std::vector<float> rf;
    CHECK(g.rows_f32("emb", {1, 2}, rf) && rf.size() == 320);
    {
        std::vector<float> ref(320);
        CHECK(gguf_file::dequant_flat(20, emb.data.data() + 90, 160, ref.data()));
        CHECK(gguf_file::dequant_flat(20, emb.data.data() + 180, 160, ref.data() + 160));
        CHECK(std::memcmp(rf.data(), ref.data(), 320 * 4) == 0);
    }
    std::vector<float> bff;
    CHECK(g.rows_f32("bf", {1}, bff) && bff.size() == 8);
    for (int i = 0; i < 8; ++i) { uint16_t v = (uint16_t)(0x3F00 + (8 + i) * 16); uint32_t u = (uint32_t)v << 16; float f; std::memcpy(&f, &u, 4); CHECK(bff[i] == f); }
    g.close();
    CHECK(g.shard_count() == 0 && !g.has("w"));

    // --- malformed input is rejected, not mapped
    const std::string bad = dir + "/bad.gguf";
    { FILE* f = std::fopen(bad.c_str(), "wb"); std::fputs("not a gguf file at all, definitely not", f); std::fclose(f); }
    Gguf gb; CHECK(!gb.open(bad, &err) && !err.empty());
    // truncated copy of a good file (tensor data cut)
    { FILE* f = std::fopen(p1.c_str(), "rb"); std::vector<uint8_t> all(1 << 16); size_t n = std::fread(all.data(), 1, all.size(), f); std::fclose(f);
      FILE* o2 = std::fopen((dir + "/trunc.gguf").c_str(), "wb"); std::fwrite(all.data(), 1, n - 40, o2); std::fclose(o2); }
    Gguf gt; CHECK(!gt.open(dir + "/trunc.gguf"));
    // duplicate tensor name across shards
    const std::string d1 = dir + "/d-00001-of-00002.gguf", d2 = dir + "/d-00002-of-00002.gguf";
    CHECK(write_gguf(d1, {w}) && write_gguf(d2, {w}));
    Gguf gd; CHECK(!gd.open(d1, &err) && err.find("duplicate") != std::string::npos);
    std::string cmd = "rm -rf " + dir; (void)!std::system(cmd.c_str());
}

int main() {
    test_geometry_arithmetic();
    test_bf16();
    test_dequant_flat();
    test_files();
    std::printf("%ld checks, %ld failed\n", testh::checks(), testh::failures());
    if (testh::failures()) return 1;
    std::printf("UNIT 26 TESTS PASSED\n");
    return 0;
}

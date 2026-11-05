// Unit pb7e2533792 tests: GGUF reader invariants (build an image byte by byte, parse it back, check every rule).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "../unit14/check.hpp"
#include "baloon/gguf_reader.hpp"
#include "baloon/gguf_lazy_loader.hpp"

using namespace baloon::gguf_reader;
using Bytes = std::vector<uint8_t>;

static void put(Bytes& b, uint64_t v, int n) { for (int i = 0; i < n; ++i) b.push_back((uint8_t)(v >> (8 * i))); }
static void pstr(Bytes& b, const std::string& s) { put(b, s.size(), 8); b.insert(b.end(), s.begin(), s.end()); }
static void kv_str(Bytes& b, const std::string& k, const std::string& v) { pstr(b, k); put(b, 8, 4); pstr(b, v); }
static void kv_u32(Bytes& b, const std::string& k, uint32_t v) { pstr(b, k); put(b, 4, 4); put(b, v, 4); }

struct Img { Bytes b; uint64_t nkv = 0; };

static Bytes build(const std::string& arch, uint32_t experts, uint32_t used, bool dup = false, uint32_t ttype = 8) {
    Bytes b; put(b, kMagic, 4); put(b, 3, 4); put(b, 1, 8);
    put(b, 4 + (dup ? 1 : 0), 8);
    kv_str(b, "general.architecture", arch);
    kv_u32(b, arch + ".expert_count", experts);
    kv_u32(b, arch + ".expert_used_count", used);
    pstr(b, "tok.scores"); put(b, 9, 4); put(b, 6, 4); put(b, 2, 8);
    uint32_t f1 = 0x3fc00000u, f2 = 0xc0000000u; put(b, f1, 4); put(b, f2, 4);
    if (dup) kv_u32(b, arch + ".expert_count", 9);
    pstr(b, "w"); put(b, 2, 4); put(b, 64, 8); put(b, 2, 8); put(b, ttype, 4); put(b, 0, 8);
    while (b.size() % 32) b.push_back(0);
    const auto g = block_geometry(ttype);
    b.resize(b.size() + (g.el ? 128 / g.el * g.by : 0), 0xAB);
    return b;
}

int main() {
    // geometry facts
    CHECK(block_geometry(8).el == 32 && block_geometry(8).by == 34);
    CHECK(block_geometry(12).el == 256 && block_geometry(12).by == 144);
    CHECK(block_geometry(100).el == 64 && block_geometry(100).by == 18);
    CHECK(block_geometry(999).el == 0);
    // agreement with the lazy loader on every type it knows
    for (uint32_t t : {0u, 1u, 2u, 6u, 8u, 11u, 12u, 13u, 14u, 20u, 23u, 30u, 100u}) {
        CHECK(block_geometry(t).el == baloon::gguf::block_geometry(t).el);
        CHECK(block_geometry(t).by == baloon::gguf::block_geometry(t).by);
    }
    CHECK(std::strcmp(ggml_type_name(12), "Q4_K") == 0);
    CHECK(std::strcmp(ggml_type_name(100), "Q2_0") == 0);
    CHECK(std::strcmp(ggml_type_name(77), "UNKNOWN") == 0);
    for (uint32_t t = 0; t < 31; ++t) CHECK((block_geometry(t).el == 0) == (std::strcmp(ggml_type_name(t), "UNKNOWN") == 0) || t == 4 || t == 5);

    // cursor bounds
    { uint8_t d[3] = {1, 2, 3}; Cursor c(d, 3); CHECK(c.u16() == 0x0201); CHECK(c.ok); (void)c.u16(); CHECK(!c.ok); CHECK(c.u32() == 0); }
    { Bytes s; put(s, 1000, 8); Cursor c(s.data(), s.size()); (void)c.str(); CHECK(!c.ok); }

    // read_value: signed widening, float, bool, nested array
    { Bytes b; put(b, 0xFF, 1); Cursor c(b.data(), b.size()); auto v = read_value(c, MetaType::I8); CHECK(v.i == -1 && c.ok); }
    { Bytes b; put(b, 0x8000, 2); Cursor c(b.data(), b.size()); auto v = read_value(c, MetaType::I16); CHECK(v.i == -32768); }
    { Bytes b; put(b, 0xFFFFFFFFull, 4); Cursor c(b.data(), b.size()); auto v = read_value(c, MetaType::U32); CHECK(v.u == 0xFFFFFFFFull && v.i == 0xFFFFFFFFll); }
    { Bytes b; put(b, 2, 1); Cursor c(b.data(), b.size()); (void)read_value(c, MetaType::Bool); CHECK(!c.ok); }
    { Bytes b; put(b, 0x3ff8000000000000ull, 8); Cursor c(b.data(), b.size()); auto v = read_value(c, MetaType::F64); CHECK(v.f == 1.5); }
    { Bytes b; put(b, 5, 4); put(b, 1ull << 40, 8); Cursor c(b.data(), b.size()); (void)read_value(c, MetaType::Array); CHECK(!c.ok); }

    // full parse
    Bytes img = build("qwen35moe", 4, 2);
    auto f = GgufFile::parse(img.data(), img.size());
    CHECK(f.has_value());
    if (f) {
        CHECK(f->version == 3 && f->alignment == 32 && f->data_start % 32 == 0);
        CHECK(f->tensors.size() == 1 && f->tensors[0].name == "w" && f->tensors[0].nbytes() == 128 / 32 * 34);
        CHECK(f->data_start + f->tensors[0].nbytes() == img.size());
        CHECK(f->get_string("general.architecture").value_or("") == "qwen35moe");
        CHECK(f->get_uint("qwen35moe.expert_count").value_or(0) == 4);
        auto sc = f->find_meta("tok.scores");
        CHECK(sc && sc->arr.size() == 2 && sc->arr[0].f == 1.5 && sc->arr[1].f == -2.0);
        CHECK(f->meta_order.size() == 4 && f->meta_order[0] == "general.architecture");
        CHECK(!f->get_string("nope") && !f->get_uint("general.architecture"));
        CHECK(check_architecture(*f, "qwen35moe") && !check_architecture(*f, "llama"));
        Qwen4ExpGuard g; std::string why;
        CHECK(g.check(*f, &why));
        Qwen4ExpGuard g8; g8.expected_experts = 8; CHECK(!g8.check(*f, &why) && why == "expert_count mismatch");
    }
    // guard rejections
    { Bytes i2 = build("llama", 4, 2); auto g = GgufFile::parse(i2.data(), i2.size()); CHECK(g && !check_architecture(*g, "qwen35moe")); Qwen4ExpGuard q; CHECK(!q.check(*g)); }
    { Bytes i2 = build("qwen2moe", 4, 5); auto g = GgufFile::parse(i2.data(), i2.size()); CHECK(g && !Qwen4ExpGuard().check(*g)); }
    { Bytes i2 = build("qwen2moe", 4, 0); auto g = GgufFile::parse(i2.data(), i2.size()); CHECK(g && !Qwen4ExpGuard().check(*g)); }
    { Bytes i2 = build("qwen2moe", 3, 1); auto g = GgufFile::parse(i2.data(), i2.size()); CHECK(g && !Qwen4ExpGuard().check(*g)); }
    // malformed inputs are rejected
    { Bytes bad = img; bad[0] ^= 1; CHECK(!GgufFile::parse(bad.data(), bad.size())); }
    { Bytes bad = img; bad[4] = 1; CHECK(!GgufFile::parse(bad.data(), bad.size())); }
    { Bytes bad = img; bad.pop_back(); CHECK(!GgufFile::parse(bad.data(), bad.size())); }
    { Bytes d = build("qwen2moe", 4, 2, true); CHECK(!GgufFile::parse(d.data(), d.size())); }
    { Bytes d = build("qwen2moe", 4, 2, false, 77); CHECK(!GgufFile::parse(d.data(), d.size())); }
    for (size_t cut = 0; cut + 1 < img.size() && cut < 400; cut += 7) CHECK(!GgufFile::parse(img.data(), cut));
    // agreement with the lazy loader's tensor directory
    if (f) {
        baloon::gguf::LazyImage li(img.data(), img.size());
        CHECK(li.ok() && li.index().data_start == f->data_start && li.index().tensors.size() == f->tensors.size());
    }
    std::printf("%ld checks, %ld failures\n", testh::checks(), testh::failures());
    if (testh::failures()) return 1;
    std::printf("GGUF READER UNIT TESTS PASSED\n");
    return 0;
}

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

// drop: bit0 expert_used_count, bit1 head_count, bit2 head_count_kv, bit3 block_count are omitted.
static Bytes build(const std::string& arch, uint32_t experts, uint32_t used, bool dup = false, uint32_t ttype = 8,
                   uint32_t heads = 24, uint32_t kvh = 2, uint32_t layers = 48, uint32_t drop = 0) {
    Bytes b; put(b, kMagic, 4); put(b, 3, 4); put(b, 1, 8);
    put(b, 7 + (dup ? 1 : 0) - __builtin_popcount(drop), 8);
    kv_str(b, "general.architecture", arch);
    kv_u32(b, arch + ".expert_count", experts);
    if (!(drop & 1)) kv_u32(b, arch + ".expert_used_count", used);
    if (!(drop & 2)) kv_u32(b, arch + ".attention.head_count", heads);
    if (!(drop & 4)) kv_u32(b, arch + ".attention.head_count_kv", kvh);
    if (!(drop & 8)) kv_u32(b, arch + ".block_count", layers);
    pstr(b, "tok.scores"); put(b, 9, 4); put(b, 6, 4); put(b, 2, 8);
    uint32_t f1 = 0x3fc00000u, f2 = 0xc0000000u; put(b, f1, 4); put(b, f2, 4);
    if (dup) kv_u32(b, arch + ".expert_count", 9);
    pstr(b, "w"); put(b, 2, 4); put(b, 64, 8); put(b, 2, 8); put(b, ttype, 4); put(b, 0, 8);
    while (b.size() % 32) b.push_back(0);
    const auto g = block_geometry(ttype);
    b.resize(b.size() + (g.el ? 128 / g.el * g.by : 0), 0xAB);
    return b;
}

static Qwen4ExpGuard fix4() { Qwen4ExpGuard g; g.expected_experts = 4; g.expected_used = 2; return g; }

int main() {
    // geometry facts
    CHECK(block_geometry(8).el == 32 && block_geometry(8).by == 34);
    CHECK(block_geometry(12).el == 256 && block_geometry(12).by == 144);
    CHECK(block_geometry(42).el == 64 && block_geometry(42).by == 18);
    CHECK(block_geometry(999).el == 0);
    CHECK(block_geometry(100).el == 0 && std::strcmp(ggml_type_name(100), "UNKNOWN") == 0);  // 100 is no longer a type
    CHECK(baloon::gguf::kQ2_0 == 42 && baloon::gguf::block_geometry(42).el == 64 && baloon::gguf::block_geometry(42).by == 18);
    CHECK(2 + 64 * 2 / 8 == 18);
    // agreement with the lazy loader on every type it knows
    for (uint32_t t : {0u, 1u, 2u, 6u, 8u, 11u, 12u, 13u, 14u, 20u, 23u, 30u, 42u}) {
        CHECK(block_geometry(t).el == baloon::gguf::block_geometry(t).el);
        CHECK(block_geometry(t).by == baloon::gguf::block_geometry(t).by);
    }
    CHECK(std::strcmp(ggml_type_name(12), "Q4_K") == 0);
    CHECK(std::strcmp(ggml_type_name(42), "Q2_0") == 0);
    CHECK(std::strcmp(ggml_type_name(77), "UNKNOWN") == 0);
    // named but unsized types (refused by the parser): Q8_K (15), IQ1_S (19) and the plain integer/F64 ids 24..28
    for (uint32_t t = 0; t < 31; ++t) {
        const bool named_unsized = t == 15 || t == 19 || (t >= 24 && t <= 28);
        CHECK((block_geometry(t).el == 0) == (std::strcmp(ggml_type_name(t), "UNKNOWN") == 0 || named_unsized) || t == 4 || t == 5);
        if (named_unsized) CHECK(block_geometry(t).el == 0 && std::strcmp(ggml_type_name(t), "UNKNOWN") != 0);
    }

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
        CHECK(f->meta_order.size() == 7 && f->meta_order[0] == "general.architecture");
        CHECK(!f->get_string("nope") && !f->get_uint("general.architecture"));
        CHECK(check_architecture(*f, "qwen35moe") && !check_architecture(*f, "llama"));
        Qwen4ExpGuard g = fix4(); std::string why;
        CHECK(g.check(*f, &why));
        CHECK(!Qwen4ExpGuard().check(*f, &why) && why == "expert_count mismatch");   // default is the 512-expert guard
        Qwen4ExpGuard g8; g8.expected_experts = 8; CHECK(!g8.check(*f, &why) && why == "expert_count mismatch");
    }
    // guard rejections
    { Bytes i2 = build("llama", 4, 2); auto g = GgufFile::parse(i2.data(), i2.size()); CHECK(g && !check_architecture(*g, "qwen35moe")); Qwen4ExpGuard q = fix4(); CHECK(!q.check(*g)); }
    { Bytes i2 = build("qwen2moe", 4, 5); auto g = GgufFile::parse(i2.data(), i2.size()); CHECK(g && !fix4().check(*g)); }
    { Bytes i2 = build("qwen2moe", 4, 0); auto g = GgufFile::parse(i2.data(), i2.size()); CHECK(g && !fix4().check(*g)); }
    { Bytes i2 = build("qwen2moe", 3, 1); auto g = GgufFile::parse(i2.data(), i2.size()); CHECK(g && !fix4().check(*g)); }
    // the guard constants: defaults are 512 experts / top-10 / 24 heads over 2 kv heads / 48 layers
    { Qwen4ExpGuard d; CHECK(d.expected_experts == 512 && d.expected_used == 10 && d.expected_heads == 24 &&
                              d.expected_kv_heads == 2 && d.expected_layers == 48);
      CHECK(kGuardExperts == 512 && kGuardTopK == 10 && kGuardHeads == 24 && kGuardKvHeads == 2 && kGuardLayers == 48); }
    auto gchk = [&](Bytes im, std::string* why) { auto g = GgufFile::parse(im.data(), im.size()); return g && Qwen4ExpGuard().check(*g, why); };
    { std::string w; CHECK(gchk(build("qwen4exp", 512, 10), &w)); CHECK(gchk(build("qwen35moe", 512, 10), &w)); }
    { std::string w; CHECK(!gchk(build("qwen4exp", 4, 2), &w) && w == "expert_count mismatch");
      CHECK(!gchk(build("qwen4exp", 511, 10), &w) && w == "expert_count mismatch");
      CHECK(!gchk(build("qwen4exp", 513, 10), &w) && w == "expert_count mismatch");
      CHECK(!gchk(build("qwen4exp", 512, 9), &w) && w == "expert_used_count mismatch");
      CHECK(!gchk(build("qwen4exp", 512, 11), &w) && w == "expert_used_count mismatch");
      CHECK(!gchk(build("qwen4exp", 512, 0), &w) && w == "expert_used_count out of range");
      CHECK(!gchk(build("qwen4exp", 512, 513), &w) && w == "expert_used_count out of range");
      CHECK(!gchk(build("qwen4exp", 512, 10, false, 8, 23, 2, 48), &w) && w == "head_count mismatch");
      CHECK(!gchk(build("qwen4exp", 512, 10, false, 8, 25, 2, 48), &w) && w == "head_count mismatch");
      CHECK(!gchk(build("qwen4exp", 512, 10, false, 8, 24, 1, 48), &w) && w == "head_count_kv mismatch");
      CHECK(!gchk(build("qwen4exp", 512, 10, false, 8, 24, 3, 48), &w) && w == "head_count_kv mismatch");
      CHECK(!gchk(build("qwen4exp", 512, 10, false, 8, 24, 2, 47), &w) && w == "block_count mismatch");
      CHECK(!gchk(build("qwen4exp", 512, 10, false, 8, 24, 2, 49), &w) && w == "block_count mismatch");
      CHECK(!gchk(build("llama", 512, 10), &w) && w == "not a qwen architecture");
      CHECK(!gchk(build("qwen4exp", 512, 10, false, 8, 24, 2, 48, 1), &w) && w == "missing expert_used_count");
      CHECK(!gchk(build("qwen4exp", 512, 10, false, 8, 24, 2, 48, 2), &w) && w == "missing head_count");
      CHECK(!gchk(build("qwen4exp", 512, 10, false, 8, 24, 2, 48, 4), &w) && w == "missing head_count_kv");
      CHECK(!gchk(build("qwen4exp", 512, 10, false, 8, 24, 2, 48, 8), &w) && w == "missing block_count"); }
    // each field of the guard is individually effective on a conforming image
    { Bytes im = build("qwen4exp", 512, 10); auto g = GgufFile::parse(im.data(), im.size()); CHECK(g);
      if (g) { std::string w; Qwen4ExpGuard x;
        x = Qwen4ExpGuard(); x.expected_used = 9; CHECK(!x.check(*g, &w));
        x = Qwen4ExpGuard(); x.expected_heads = 12; CHECK(!x.check(*g, &w));
        x = Qwen4ExpGuard(); x.expected_kv_heads = 4; CHECK(!x.check(*g, &w));
        x = Qwen4ExpGuard(); x.expected_layers = 24; CHECK(!x.check(*g, &w)); } }
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

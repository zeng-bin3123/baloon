#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <unistd.h>
#include <vector>
#include "baloon/gguf_lazy_loader.hpp"
#include "baloon/q2_gguf_writer.hpp"
#include "check.hpp"
using namespace baloon;
using testh::Rng;

static std::vector<uint8_t> pattern(size_t n, uint8_t salt) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = (uint8_t)(i * 7 + 3 + salt * 13 + (i >> 8));
    return v;
}

void test_gguf_geometry() {
    Rng r(10);
    const uint32_t types[] = {gguf::kF32, gguf::kF16, gguf::kBF16, gguf::kQ2_0, gguf::kQ4_0, gguf::kQ5_0, gguf::kQ8_0,
                              gguf::kQ3_K, gguf::kQ4_K, gguf::kQ5_K, gguf::kQ6_K, gguf::kIQ4_NL, gguf::kIQ4_XS};
    for (uint32_t t : types) {
        const auto g = gguf::block_geometry(t);
        CHECK(g.el > 0 && g.by > 0);
        for (int it = 0; it < 200; ++it) {
            const uint64_t k = 1 + r.below(50), n = 1 + r.below(40);
            // experts_tile: n experts of el*k elements total n times one expert's bytes
            CHECK(gguf::tensor_bytes(g.el * k * n, g.el, g.by) == n * gguf::expert_bytes(g.el * k, g.el, g.by));
            const uint64_t per = g.el * k;
            for (uint64_t e = 0; e < n; ++e) {
                CHECK(gguf::expert_start(e + 1, per, g.el, g.by) == gguf::expert_start(e, per, g.el, g.by) + gguf::expert_bytes(per, g.el, g.by));
            }
            CHECK(gguf::expert_start(n - 1, per, g.el, g.by) + gguf::expert_bytes(per, g.el, g.by) == n * gguf::expert_bytes(per, g.el, g.by));
            const uint64_t rb = gguf::row_bytes(g.el * k, g.el, g.by);
            for (uint64_t row = 0; row < n; ++row) {
                CHECK((row + 1) * rb <= n * rb);
                CHECK((row + 1) * rb == row * rb + rb);
            }
            CHECK(gguf::whole_blocks(g.el * k, g.el));
            CHECK((gguf::tensor_bytes(g.el * k, g.el, g.by) / g.by) * g.el == g.el * k);  // bytes recover elements
        }
    }
    CHECK(!gguf::whole_blocks(100, 64) && gguf::whole_blocks(128, 64));
    CHECK(gguf::block_geometry(9999).el == 0);
    // known sizes
    CHECK(gguf::tensor_bytes(64, 64, 18) == 18 && gguf::tensor_bytes(256, 256, 210) == 210);
}

void test_gguf_shapes_widen_range() {
    Rng r(11);
    for (int it = 0; it < 200; ++it) {
        std::vector<uint64_t> s(1 + r.below(5));
        for (auto& x : s) x = r.below(1000);
        CHECK(gguf::numpy_shape(gguf::numpy_shape(s)) == s);
    }
    CHECK((gguf::numpy_shape({7, 5, 3}) == std::vector<uint64_t>{3, 5, 7}));  // fused [in,out,experts] -> (experts,out,in)
    for (uint32_t h = 0; h < 65536; ++h) {
        const uint32_t bits = gguf::bf16_widen_bits((uint16_t)h);
        CHECK(bits / 65536 == h && bits % 65536 == 0);
        const float f = gguf::bf16_widen((uint16_t)h);
        uint32_t back; std::memcpy(&back, &f, 4);
        if (!std::isnan(f)) CHECK(back == bits);
    }
    CHECK(gguf::shards_complete(5, 5) && !gguf::shards_complete(4, 5) && !gguf::shards_complete(6, 5));
    CHECK(gguf::row_in_range(0, 1) && !gguf::row_in_range(1, 1) && !gguf::row_in_range(5, 3));
}

void test_gguf_shards() {
    CHECK(gguf::shard_name("model", 2, 5) == "model-00002-of-00005.gguf");
    std::string p; unsigned a = 0, b = 0;
    CHECK(gguf::parse_shard_name("model-00002-of-00005.gguf", &p, &a, &b) && p == "model" && a == 2 && b == 5);
    CHECK(!gguf::parse_shard_name("model.gguf", &p, &a, &b));
    CHECK(!gguf::parse_shard_name("model-0002-of-00005.gguf", &p, &a, &b));
    CHECK(!gguf::parse_shard_name("model-0000x-of-00005.gguf", &p, &a, &b));
    std::vector<std::string> names;
    for (unsigned i = 1; i <= 4; ++i) names.push_back(gguf::shard_name("m", i, 4));
    auto ok = gguf::discover_shards_in(names, names[0]);
    CHECK(ok && ok->size() == 4 && (*ok)[3] == names[3]);
    auto ok2 = gguf::discover_shards_in(names, names[2]);
    CHECK(ok2 && ok2->size() == 4);
    std::vector<std::string> missing = {names[0], names[1], names[3]};
    CHECK(!gguf::discover_shards_in(missing, missing[0]));  // missing shard rejected
    CHECK(!gguf::discover_shards_in(names, "single.gguf"));
    // filesystem wrapper (POSIX)
    char tmpl[] = "/tmp/baloon_unit2_shardsXXXXXX";
    char* dir = mkdtemp(tmpl);
    CHECK(dir != nullptr);
    if (dir) {
        const std::string d = dir;
        for (unsigned i = 1; i <= 3; ++i) std::ofstream(d + "/" + gguf::shard_name("w", i, 3)).put('x');
        auto found = gguf::discover_shards(d + "/" + gguf::shard_name("w", 1, 3));
        CHECK(found && found->size() == 3 && (*found)[2] == d + "/" + gguf::shard_name("w", 3, 3));
        std::remove((d + "/" + gguf::shard_name("w", 2, 3)).c_str());
        CHECK(!gguf::discover_shards(d + "/" + gguf::shard_name("w", 1, 3)));  // missing shard rejected
        auto single = gguf::discover_shards(d + "/plain.gguf");
        CHECK(single && single->size() == 1);
        std::remove((d + "/" + gguf::shard_name("w", 1, 3)).c_str());
        std::remove((d + "/" + gguf::shard_name("w", 3, 3)).c_str());
        rmdir(d.c_str());
    }
}

void test_gguf_lazy_image() {
    // fused expert tensor [in=128, out=3, experts=5] Q2_0, rows tensor [96, 7] Q4_0, plus a scalar-ish F32 tensor
    q2w::TensorSpec fused{"blk.0.ffn_gate_exps.weight", {128, 3, 5}, gguf::kQ2_0, pattern(540, 1)};
    q2w::TensorSpec rows{"tok.weight", {96, 7}, gguf::kQ4_0, pattern(7 * 54, 2)};
    q2w::TensorSpec vec{"norm.weight", {10}, gguf::kF32, pattern(40, 3)};
    std::vector<q2w::Kv> kvs = {{"general.architecture", std::string("test")}, {"a.u32", uint32_t(5)},
                                {"a.u64", uint64_t(1) << 40}, {"a.f", 1.5f}, {"a.b", true}};
    std::vector<uint8_t> img; std::string err;
    CHECK(q2w::write_file(kvs, {fused, rows, vec}, &img, &err));
    gguf::LazyImage li(img.data(), img.size());
    CHECK(li.ok());
    CHECK(li.index().version == 3 && li.index().tensors.size() == 3);
    CHECK(li.index().data_start % 32 == 0);
    for (auto& t : li.index().tensors) CHECK(t.offset % 32 == 0);

    for (uint64_t e = 0; e < 5; ++e) {
        auto sp = li.read_expert("blk.0.ffn_gate_exps.weight", e);
        CHECK(sp && sp->len == 108);
        if (sp) CHECK(std::memcmp(sp->ptr, fused.data.data() + e * 108, 108) == 0);
    }
    CHECK(!li.read_expert("blk.0.ffn_gate_exps.weight", 5));  // out-of-range expert rejected
    CHECK(!li.read_expert("missing", 0));
    // experts tile the tensor with no gap / overlap
    auto whole = li.read_tensor("blk.0.ffn_gate_exps.weight");
    CHECK(whole && whole->len == 540 && std::memcmp(whole->ptr, fused.data.data(), 540) == 0);
    auto e0 = li.read_expert("blk.0.ffn_gate_exps.weight", 0), e4 = li.read_expert("blk.0.ffn_gate_exps.weight", 4);
    CHECK(e0 && e4 && e0->ptr == whole->ptr && e4->ptr + e4->len == whole->ptr + whole->len);

    for (uint64_t r0 = 0; r0 < 7; ++r0)
        for (uint64_t c = 1; r0 + c <= 7; ++c) {
            auto sp = li.read_rows("tok.weight", r0, c);
            CHECK(sp && sp->len == c * 54 && std::memcmp(sp->ptr, rows.data.data() + r0 * 54, c * 54) == 0);
        }
    CHECK(!li.read_rows("tok.weight", 7, 1));
    CHECK(!li.read_rows("tok.weight", 3, 5));
    CHECK(!li.read_rows("tok.weight", 0, 0));
    CHECK(!li.read_rows("norm.weight", 0, 1));  // not 2-D
    auto v = li.read_tensor("norm.weight");
    CHECK(v && v->len == 40 && std::memcmp(v->ptr, vec.data.data(), 40) == 0);
    // shape reversal view of the fused tensor
    const auto* t = li.index().find("blk.0.ffn_gate_exps.weight");
    CHECK(t && gguf::numpy_shape(t->shape) == (std::vector<uint64_t>{5, 3, 128}));

    // malformed inputs
    std::vector<uint8_t> bad = img; bad[0] = 'X';
    CHECK(!gguf::LazyImage(bad.data(), bad.size()).ok());
    bad = img; bad[4] = 9;  // bad version
    CHECK(!gguf::LazyImage(bad.data(), bad.size()).ok());
    uint64_t needed = 0;
    for (auto& tt : li.index().tensors) needed = std::max<uint64_t>(needed, li.index().data_start + tt.offset + tt.nbytes());
    CHECK(gguf::parse_header(img.data(), needed).has_value());
    for (uint64_t sz = 0; sz < needed; sz += 1) CHECK(!gguf::parse_header(img.data(), sz).has_value());
}

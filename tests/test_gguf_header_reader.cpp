#include <cstdio>
#include <map>
#include <set>

#include "baloon/gguf_header_reader.hpp"

using namespace baloon::gguf;

static int g_fail = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_fail;                                                            \
        }                                                                        \
    } while (0)

static void test_geometry() {
    struct Row { uint32_t t, e, b; };
    const Row table[] = {
        {0, 1, 4},    {1, 1, 2},     {30, 1, 2},    {2, 32, 18},   {3, 32, 20},  {6, 32, 22},
        {7, 32, 24},  {8, 32, 34},   {9, 32, 36},   {10, 256, 84}, {11, 256, 110}, {12, 256, 144},
        {13, 256, 176}, {14, 256, 210}, {16, 256, 66}, {17, 256, 74}, {18, 256, 98},
        {20, 32, 18}, {21, 256, 110}, {22, 256, 82}, {23, 256, 136}, {29, 256, 56}, {42, 64, 18}};
    std::set<uint32_t> supported;
    for (const Row& r : table) {
        supported.insert(r.t);
        auto g = block_geometry(r.t);
        CHECK(g.has_value());
        if (g) CHECK(g->elems == r.e && g->bytes == r.b);
    }
    // Exactly the table is supported; everything else is rejected.
    for (uint32_t t = 0; t < 2000; ++t) CHECK(block_geometry(t).has_value() == (supported.count(t) > 0));
    CHECK(!block_geometry(4) && !block_geometry(5) && !block_geometry(100));
    // Positivity for all supported ids below 64.
    for (uint32_t t = 0; t < 64; ++t)
        if (auto g = block_geometry(t)) CHECK(g->elems > 0 && g->bytes > 0);
    // Q2_0 layout: 2 + 16 = 18 and 16 * 4 = 64.
    CHECK(block_geometry(42) == std::optional<BlockGeometry>(BlockGeometry{64, 18}));
    CHECK(2 + 16 == 18 && 16 * 4 == 64);
    CHECK(block_geometry(20) == block_geometry(2));
    CHECK(block_geometry(11) == block_geometry(21));
    for (uint32_t t : {10u, 11u, 12u, 13u, 14u}) CHECK(block_geometry(t)->elems == 256);
    // Row bytes.
    CHECK(row_bytes(42, 4096) == std::optional<uint64_t>(1152));
    for (uint64_t n : {0ull, 1ull, 7ull, 4096ull, 123457ull}) {
        CHECK(row_bytes(0, n) == std::optional<uint64_t>(n * 4));
        CHECK(row_bytes(1, n) == std::optional<uint64_t>(n * 2));
    }
    CHECK(!row_bytes(4, 64));
}

static void test_tensor_elements() {
    CHECK(tensor_elements({}) == 1);
    for (uint64_t a = 0; a < 20; ++a)
        for (uint64_t b = 0; b < 20; ++b) {
            CHECK(tensor_elements({a, b}) == a * b);
            // multiplicative over concatenation
            CHECK(tensor_elements({a, b, 3, 5}) == tensor_elements({a, b}) * tensor_elements({3, 5}));
        }
    CHECK(tensor_elements({3, 0, 5}) == 0);
}

static void test_align() {
    CHECK(align_up(100, 32) == 128);
    for (uint64_t align = 1; align <= 70; ++align)
        for (uint64_t pos = 0; pos <= 2000; ++pos) {
            uint64_t a = align_up(pos, align);
            CHECK(a % align == 0);
            CHECK(a >= pos);
            CHECK(a < pos + align);
        }
    for (uint64_t align = 1; align <= 70; ++align)
        for (uint64_t k = 0; k <= 50; ++k) CHECK(align_up(k * align, align) == k * align);
    // data_start: must not exceed the file size; align 0 rejected.
    CHECK(data_start(100, 32, 128) == std::optional<uint64_t>(128));
    CHECK(!data_start(100, 32, 127));
    CHECK(!data_start(100, 0, 1000));
    CHECK(data_start(0, 32, 0) == std::optional<uint64_t>(0));
}

static void test_cursor() {
    for (uint64_t size = 0; size <= 24; ++size)
        for (uint64_t pos = 0; pos <= size; ++pos)
            for (uint64_t n = 0; n <= 30; ++n) {
                Cursor c{pos, size};
                auto r = c.read(n);
                // fails exactly when it would overrun
                CHECK(r.has_value() == (pos + n <= size));
                if (r) {
                    CHECK(r->pos == pos + n);       // advances by exactly n
                    CHECK(r->pos <= r->size);        // in bounds
                    CHECK(r->size == size);          // size preserved
                }
                // advance() leaves the cursor untouched on failure
                Cursor d = c;
                bool ok = d.advance(n);
                CHECK(ok == r.has_value());
                CHECK(ok ? d.pos == pos + n : d.pos == pos);
                // read of zero is a no-op
                CHECK(c.read(0).has_value() && c.read(0)->pos == pos);
                // composition
                for (uint64_t m = 0; m <= 10 && r; ++m) {
                    auto r2 = r->read(m);
                    if (r2) {
                        auto both = c.read(n + m);
                        CHECK(both.has_value() && both->pos == r2->pos && both->size == r2->size);
                    }
                }
            }
    // overflow safety
    Cursor big{UINT64_MAX - 1, UINT64_MAX};
    CHECK(!big.read(2));
    CHECK(big.read(1).has_value());
}

static void test_array_bounds() {
    CHECK(kMaxArrayLen == 16777216ull);
    CHECK(array_len_ok(0) && array_len_ok(kMaxArrayLen) && !array_len_ok(kMaxArrayLen + 1));
    CHECK(depth_ok(0) && depth_ok(1) && depth_ok(2) && !depth_ok(3) && !depth_ok(100));
}

static MetadataLookup lookup_from(const std::map<std::string, uint64_t>& m) {
    return [m](const std::string& k) -> std::optional<uint64_t> {
        auto it = m.find(k);
        if (it == m.end()) return std::nullopt;
        return it->second;
    };
}

static void test_guard() {
    Guard g;
    CHECK(g.reqs().size() == 6);
    std::map<std::string, uint64_t> good;
    for (auto& r : g.reqs()) good[r.first] = r.second;
    const std::optional<std::string> arch = std::string("qwen4exp");

    // accepts exactly when everything matches
    CHECK(arch_guard(g, arch, lookup_from(good)).empty());
    // missing / wrong architecture refused regardless of metadata
    CHECK(!arch_guard(g, std::nullopt, lookup_from(good)).empty());
    CHECK(!arch_guard(g, std::string("llama"), lookup_from(good)).empty());
    CHECK(!arch_guard(g, std::string(""), lookup_from(good)).empty());
    // soundness: each requirement removed or altered is refused
    for (auto& r : g.reqs()) {
        auto missing = good;
        missing.erase(r.first);
        CHECK(!arch_guard(g, arch, lookup_from(missing)).empty());
        auto off = good;
        off[r.first] = r.second + 1;
        CHECK(!arch_guard(g, arch, lookup_from(off)).empty());
    }
    // extra unrelated keys do not matter
    auto extra = good;
    extra["something.else"] = 7;
    CHECK(arch_guard(g, arch, lookup_from(extra)).empty());
    // non-default guard
    Guard g2;
    g2.experts = 64;
    CHECK(!arch_guard(g2, arch, lookup_from(good)).empty());
    good["qwen4exp.expert_count"] = 64;
    CHECK(arch_guard(g2, arch, lookup_from(good)).empty());
}

int main() {
    test_geometry();
    test_tensor_elements();
    test_align();
    test_cursor();
    test_array_bounds();
    test_guard();
    if (g_fail) {
        std::fprintf(stderr, "gguf_header_reader: %d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("gguf_header_reader: all checks passed\n");
    return 0;
}


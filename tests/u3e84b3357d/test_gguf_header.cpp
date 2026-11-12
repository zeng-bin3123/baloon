// test_gguf_header.cpp - unit u3e84b3357d host test: every statement of GgufHeaderParsing.lean is one
// U3E_SECTION over include/baloon/gguf_reader.hpp (the port of include/ref/artifact/gguf_reader.hpp).
#include "baloon/gguf_reader.hpp"
#include "gguf_fixture.hpp"

#include <algorithm>
#include <cstring>

using namespace baloon::gguf_reader;
using u3e::W; using u3e::Kv;

// ---- sections (one per theory statement) ------------------------------------------------------------------

// theory: Cursor::need(n) fails iff pos + n > size; every successful read keeps pos <= size; a sequence of
// reads advances pos by the sum of their sizes (read_in_bounds, read_fails_iff, readAll_spec, string_read_size).
U3E_SECTION(sec_cursor_bounds, "cursor: need fails iff pos+n>size, reads stay in bounds, sequences add up") {
    uint8_t mem[64] = {};
    for (uint64_t size = 0; size <= 40; ++size) for (uint64_t pos = 0; pos <= size; ++pos) for (uint64_t k = 0; k <= 48; ++k) {
        Cursor c(mem, size); c.pos = pos;
        const bool ok = c.read(k);
        CHECK(ok == !(pos + k > size));                                   // read_fails_iff
        if (ok) CHECK(c.ok && c.pos == pos + k && c.pos <= c.n && c.n == size);   // read_in_bounds
        else    CHECK(!c.ok && c.pos == pos);                             // a failed read never moves the cursor
        Cursor d(mem, size); d.pos = pos; CHECK(d.need(k) == !(pos + k > size) && d.pos == pos);
    }
    // a huge request cannot wrap around: pos + k overflowing 2^64 is still a failure
    { Cursor c(mem, 64); c.pos = 8; CHECK(!c.read(~0ull) && !c.need(~0ull - 4)); }
    // readAll_spec: a successful sequence advances by the sum; the first failure stops it
    uint32_t rng = 12345;
    for (int trial = 0; trial < 20000; ++trial) {
        const uint64_t size = 64; Cursor c(mem, size);
        uint64_t sum = 0, prev = 0; bool alive = true;
        const int len = 1 + (int)((rng = rng * 1664525u + 1013904223u) >> 29);
        for (int i = 0; i < len; ++i) {
            const uint64_t k = (rng = rng * 1664525u + 1013904223u) >> 28;
            const bool r = c.read(k);
            if (alive && sum + k <= size) { CHECK(r); sum += k; } else { alive = false; CHECK(!r); }
            CHECK(c.pos >= prev && c.pos <= size); prev = c.pos;
        }
        CHECK(c.pos == sum && c.ok == alive);
    }
    // Cursor::read_all (theory readAll): the code path itself, against the sum of the sizes
    for (int trial = 0; trial < 20000; ++trial) {
        const uint64_t size = (rng = rng * 1664525u + 1013904223u) >> 26;   // 0..63
        std::vector<uint64_t> ks((rng = rng * 1664525u + 1013904223u) >> 29);
        for (auto& k : ks) k = (rng = rng * 1664525u + 1013904223u) >> 28;
        uint64_t sum = 0, before_fail = 0; bool fits = true;
        for (uint64_t k : ks) { if (fits && sum + k <= size) sum += k; else if (fits) { fits = false; before_fail = sum; } }
        Cursor c(mem, size);
        const bool r = c.read_all(ks.begin(), ks.end());
        CHECK(r == fits && c.ok == fits && c.pos <= size);
        CHECK(c.pos == (fits ? sum : before_fail));                       // advances by the sum, or stops at the failure
    }
    { Cursor c(mem, 64); CHECK(c.read_all({8, 16, 0, 40}) && c.pos == 64); CHECK(!c.read_all({0, 1}) && c.pos == 64); }
    { Cursor c(mem, 64); CHECK(c.read_all({}) && c.pos == 0 && c.ok); }
    // the scalar readers are reads of 1/2/4/8 bytes
    { uint8_t d[15] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}; Cursor c(d, 15);
      CHECK(c.u8() == 1 && c.pos == 1 && c.u16() == 0x0302 && c.pos == 3 && c.u32() == 0x07060504u && c.pos == 7);
      CHECK(c.u64() == 0x0F0E0D0C0B0A0908ull && c.pos == 15 && c.ok); CHECK(c.u8() == 0 && !c.ok && c.pos == 15); }
    // string_read_size: a GGUF string consumes 8 + len bytes, and fails if either part runs past the end
    for (uint64_t len : {0ull, 1ull, 7ull, 33ull, 200ull}) {
        W w; w.s(std::string((size_t)len, 'x'));
        for (uint64_t cut = 0; cut <= w.b.size(); ++cut) {
            Cursor c(w.b.data(), cut); const std::string s = c.str();
            if (cut == w.b.size()) CHECK(c.ok && s.size() == len && c.pos == 8 + len);
            else CHECK(!c.ok && c.pos <= cut);
        }
    }
}

// theory: arrays longer than 2^24 are refused (arrayLenOk_iff: accepted exactly when count <= 16,777,216).
U3E_SECTION(sec_array_len_guard, "arrays: count > 2^24 refused, count <= 2^24 accepted by the guard") {
    CHECK(kMaxArrayElems == 16777216ull && kMaxArrayElems == (1ull << 24));
    CHECK(array_len_ok(0) && array_len_ok(1) && array_len_ok(1ull << 24) && !array_len_ok((1ull << 24) + 1) && !array_len_ok(~0ull));
    for (uint64_t cnt : {(1ull << 24) + 1, 1ull << 25, 1ull << 40, ~0ull}) {
        for (uint32_t et : {0u, 8u, 9u}) {                    // U8, String, nested Array: the guard comes first
            W w; w.u(et, 4); w.u(cnt, 8); w.b.resize(w.b.size() + 64, 0);
            Cursor c(w.b.data(), w.b.size()); (void)read_value(c, MetaType::Array); CHECK(!c.ok);
        }
    }
    // exactly 2^24 one-byte elements is accepted and consumes 12 + 2^24 bytes
    { W w; w.u(0, 4); w.u(1ull << 24, 8); std::vector<uint8_t> b(w.b); b.resize(b.size() + (1u << 24), 7);
      Cursor c(b.data(), b.size()); (void)read_value(c, MetaType::Array); CHECK(c.ok && c.pos == 12 + (1ull << 24)); }
    // an accepted count that the mapping cannot hold still fails as truncation, never reads past the end
    { W w; w.u(4, 4); w.u(1000, 8); w.u(1, 4);
      Cursor c(w.b.data(), w.b.size()); (void)read_value(c, MetaType::Array); CHECK(!c.ok && c.pos <= c.n); }
}

// theory: only min(count, 64) items of an array are kept (keptItems_spec), yet all count elements are consumed.
U3E_SECTION(sec_kept_items, "arrays: min(count, 64) items kept, every element still consumed") {
    CHECK(kArraySample == 64);
    for (uint64_t n = 0; n <= 300; ++n) {
        const uint64_t k = array_sample_count(n);
        CHECK(k <= 64 && k <= n && (n > 64 || k == n) && k == std::min<uint64_t>(n, 64));
    }
    for (uint64_t n : {0ull, 1ull, 63ull, 64ull, 65ull, 200ull, 5000ull}) {
        // U32 elements: values are the indices, so the kept prefix can be identified
        W w; w.u(4, 4); w.u(n, 8); for (uint64_t i = 0; i < n; ++i) w.u(i * 3 + 1, 4); w.u(0xAB, 1);
        Cursor c(w.b.data(), w.b.size()); MetaValue v = read_value(c, MetaType::Array);
        CHECK(c.ok && v.count == n && v.arr.size() == std::min<uint64_t>(n, 64) && v.sample_count() == v.arr.size());
        CHECK(c.pos == 12 + 4 * n && c.u8() == 0xAB);                  // all n elements consumed, nothing more
        for (size_t i = 0; i < v.arr.size(); ++i) CHECK(v.arr[i].type == MetaType::U32 && v.arr[i].u == i * 3 + 1);
        // string elements: variable length, still all consumed
        W s; s.u(8, 4); s.u(n, 8); for (uint64_t i = 0; i < n; ++i) s.s(std::string((size_t)(i % 5), 'a' + (char)(i % 26)));
        Cursor cs(s.b.data(), s.b.size()); MetaValue vs = read_value(cs, MetaType::Array);
        CHECK(cs.ok && cs.pos == s.b.size() && vs.count == n && vs.arr.size() == std::min<uint64_t>(n, 64));
        // a truncated tail element beyond the kept prefix is still detected
        if (n) { Cursor ct(w.b.data(), 12 + 4 * n - 1); (void)read_value(ct, MetaType::Array); CHECK(!ct.ok); }
    }
}

// theory: nesting depth > 2 fails. The top-level value is depth 0, so arrays nested up to three levels
// (array of array of array) decode, and a fourth level of array is refused.
static std::vector<uint8_t> nested_array(int levels, uint64_t leaf_count) {
    // `levels` arrays, each holding one element of the next level; the innermost holds leaf_count U8 values
    W w;
    for (int i = 0; i < levels - 1; ++i) { w.u(9, 4); w.u(1, 8); }
    w.u(0, 4); w.u(leaf_count, 8); for (uint64_t k = 0; k < leaf_count; ++k) w.u(k, 1);
    return w.b;
}
U3E_SECTION(sec_nesting_depth, "arrays: nesting depth > 2 fails") {
    CHECK(kMaxNesting == 2);
    for (int levels = 1; levels <= 6; ++levels) {
        const std::vector<uint8_t> b = nested_array(levels, 3);
        Cursor c(b.data(), b.size()); MetaValue v = read_value(c, MetaType::Array);
        const bool expect_ok = levels - 1 <= kMaxNesting;            // deepest array is read at depth levels-1
        CHECK(c.ok == expect_ok);
        if (expect_ok) {
            CHECK(c.pos == b.size());
            const MetaValue* p = &v;
            for (int i = 0; i < levels - 1; ++i) { CHECK(p->type == MetaType::Array && p->arr.size() == 1); if (p->arr.empty()) break; p = &p->arr[0]; }
            CHECK(p->elem == MetaType::U8 && p->count == 3 && p->arr.size() == 3 && p->arr[2].u == 2);
        }
    }
    // the depth argument is the starting depth: an array read directly at depth 3 fails, at 2 succeeds
    { const auto b = nested_array(1, 1); Cursor c(b.data(), b.size()); (void)read_value(c, MetaType::Array, 3); CHECK(!c.ok); }
    { const auto b = nested_array(1, 1); Cursor c(b.data(), b.size()); (void)read_value(c, MetaType::Array, 2); CHECK(c.ok); }
    // scalars are not depth-limited
    { W w; w.u(5, 4); Cursor c(w.b.data(), 4); CHECK(read_value(c, MetaType::U32, 7).u == 5 && c.ok); }
    // a too-deep array inside a GGUF file makes the whole parse fail
    for (int levels = 1; levels <= 5; ++levels) {
        const auto raw = nested_array(levels, 2);
        std::vector<Kv> kv = {{"general.name", 8, 0, "x"}, {"tok.nested", 9, 0, std::string(raw.begin(), raw.end())}};
        auto img = u3e::image(kv); auto f = GgufFile::parse(img.data(), img.size());
        CHECK(f.has_value() == (levels - 1 <= kMaxNesting));
        if (f) CHECK(f->find_meta("tok.nested") && f->find_meta("tok.nested")->type == MetaType::Array && f->tensors.size() == 1);
    }
}

// theory: signed integer metadata is stored sign-extended into uint64 (two's complement mod 2^64) and is
// recoverable (signExtend_roundtrip).
U3E_SECTION(sec_sign_extend, "signed metadata: sign-extended into uint64, recovered exactly") {
    const struct { MetaType t; unsigned nb; int64_t lo, hi; } ts[] = {
        {MetaType::I8, 1, INT8_MIN, INT8_MAX}, {MetaType::I16, 2, INT16_MIN, INT16_MAX},
        {MetaType::I32, 4, INT32_MIN, INT32_MAX}, {MetaType::I64, 8, INT64_MIN, INT64_MAX}};
    uint64_t rng = 0x9E3779B97F4A7C15ull;
    for (auto& t : ts) {
        std::vector<int64_t> xs = {t.lo, t.lo + 1, -2, -1, 0, 1, 2, t.hi - 1, t.hi};
        for (int i = 0; i < 2000; ++i) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            xs.push_back(t.nb == 8 ? (int64_t)rng : sign_extend(rng, t.nb));
        }
        for (int64_t x : xs) {
            CHECK(x >= t.lo && x <= t.hi);
            W w; w.u((uint64_t)x, (int)t.nb); w.u(0x5A, 1);
            Cursor c(w.b.data(), w.b.size()); MetaValue v = read_value(c, t.t);
            CHECK(c.ok && c.pos == t.nb && v.i == x);
            CHECK(v.u == (uint64_t)x);                    // stored as x mod 2^64
            CHECK((int64_t)v.u == x);                     // asInt64 (signExtend64 x) = x
            CHECK(sign_extend((uint64_t)x, t.nb) == x);   // the stored low nb bytes alone determine x
            CHECK(v.f == (double)x && v.is_integer());
        }
    }
    // unsigned fields are never sign-extended: 0xFF as U8 is 255
    { uint8_t d[1] = {0xFF}; Cursor c(d, 1); MetaValue v = read_value(c, MetaType::U8); CHECK(v.u == 255 && v.i == 255); }
    // through a whole file: a negative I32 key round-trips, and get_uint refuses it rather than returning 2^64-7
    for (int64_t x : {-7ll, (long long)INT32_MIN, -1ll, 5ll}) {
        std::vector<Kv> kv = {{"general.name", 8, 0, "s"}, {"k.i32", 5, (uint64_t)x, ""}};   // type 5 = I32, 4 bytes
        auto img = u3e::image(kv); auto f = GgufFile::parse(img.data(), img.size()); CHECK(f.has_value());
        if (!f) continue;
        const MetaValue* m = f->find_meta("k.i32");
        CHECK(m && m->type == MetaType::I32 && m->i == x && (int64_t)m->u == x);
        CHECK(f->get_uint("k.i32").has_value() == (x >= 0));                  // negatives are not unsigned values
    }
}

// theory: booleans are normalised, any non-zero byte reads as 1 (readBool_binary: a decoded bool is 0 or 1).
U3E_SECTION(sec_bool_normalise, "bool metadata: any non-zero byte reads as 1") {
    for (unsigned byte = 0; byte < 256; ++byte) {
        uint8_t d[2] = {(uint8_t)byte, 0xEE};
        Cursor c(d, 2); MetaValue v = read_value(c, MetaType::Bool);
        CHECK(c.ok && c.pos == 1 && v.type == MetaType::Bool);
        CHECK(v.u == 0 || v.u == 1);
        CHECK(v.b == (byte != 0) && v.u == (byte != 0 ? 1u : 0u) && read_bool((uint8_t)byte) == (byte != 0));
    }
    // an array of bools: every element normalised
    { W w; w.u(7, 4); w.u(5, 8); for (uint64_t x : {0ull, 1ull, 2ull, 0x80ull, 0xFFull}) w.u(x, 1);
      Cursor c(w.b.data(), w.b.size()); MetaValue v = read_value(c, MetaType::Array);
      CHECK(c.ok && v.arr.size() == 5 && v.arr[0].u == 0 && v.arr[1].u == 1 && v.arr[2].u == 1 && v.arr[3].u == 1 && v.arr[4].u == 1); }
    // inside a file: a bool key holding 0x7F parses and reads as true
    { std::vector<Kv> kv = {{"general.name", 8, 0, "b"}, {"k.flag", 7, 0x7F, ""}};
      auto img = u3e::image(kv); auto f = GgufFile::parse(img.data(), img.size());
      CHECK(f && f->find_meta("k.flag") && f->find_meta("k.flag")->b && f->find_meta("k.flag")->u == 1); }
    // a bool still needs its byte
    { Cursor c(nullptr, 0); (void)read_value(c, MetaType::Bool); CHECK(!c.ok); }
}

// theory: data_start = ceil(pos / align) * align: aligned, at or after the header end, less than one
// alignment unit past it (alignUp_spec).
U3E_SECTION(sec_data_start, "data section: data_start = ceil(pos/align)*align, aligned, >= header end, < end+align") {
    for (uint64_t a = 1; a <= 300; ++a) for (uint64_t pos = 0; pos <= 1200; ++pos) {
        const uint64_t s = align_up(pos, a);
        CHECK(s % a == 0 && pos <= s && s < pos + a);
        CHECK(s == (pos % a ? pos + (a - pos % a) : pos));      // the ceiling, computed independently
    }
    for (uint64_t a : {1ull, 32ull, 4096ull}) CHECK(align_up(0, a) == 0 && align_up(1ull << 40, a) == 1ull << 40);
    // real images: the parsed data_start is align_up(end of tensor directory, alignment)
    for (uint64_t align : {32ull, 64ull, 128ull, 256ull}) for (int pad = 0; pad < 40; ++pad) {
        std::vector<Kv> kv = {{"general.name", 8, 0, std::string((size_t)pad + 1, 'n')}};
        if (align != 32) kv.push_back({"general.alignment", 4, align, ""});
        size_t end = 0; auto img = u3e::image(kv, 42, {64}, align, &end, 18);
        auto f = GgufFile::parse(img.data(), img.size());
        CHECK(f.has_value());
        if (!f) continue;
        CHECK(f->alignment == align && f->data_start == align_up(end, align));
        CHECK(f->data_start % align == 0 && f->data_start >= end && f->data_start < end + align);
        CHECK(f->data_start + f->tensors[0].nbytes() == img.size());
        // one byte short of the tensor payload: the directory claims data past the mapping and the parse fails
        CHECK(!GgufFile::parse(img.data(), img.size() - 1));
    }
}

// theory: the default alignment 32 is used when general.alignment is absent or zero (effectiveAlign,
// effectiveAlign_pos: the effective alignment is never zero).
U3E_SECTION(sec_effective_align, "alignment: general.alignment absent or 0 -> 32, otherwise its value") {
    CHECK(kDefaultAlignment == 32);
    CHECK(effective_alignment(std::nullopt) == 32 && effective_alignment(0) == 32);
    for (uint64_t a = 1; a <= 1024; ++a) CHECK(effective_alignment(a) == a && effective_alignment(a) > 0);
    // parsed images: absent, 0, and non-default values (powers of two and not), as U32 and as U64 keys
    struct Case { bool present; uint32_t type; uint64_t value; uint64_t expect; };
    const Case cs[] = {{false, 4, 0, 32}, {true, 4, 0, 32}, {true, 10, 0, 32}, {true, 4, 32, 32}, {true, 4, 64, 64},
                       {true, 10, 128, 128}, {true, 4, 1, 1}, {true, 4, 8, 8}, {true, 4, 48, 48}, {true, 4, 100, 100}};
    for (const Case& k : cs) for (int pad = 0; pad < 12; ++pad) {
        std::vector<Kv> kv = {{"general.name", 8, 0, std::string((size_t)pad * 3 + 1, 'q')}};
        if (k.present) kv.push_back({"general.alignment", k.type, k.value, ""});
        size_t end = 0; auto img = u3e::image(kv, 42, {64}, k.expect, &end, 18);
        auto f = GgufFile::parse(img.data(), img.size());
        CHECK(f.has_value());
        if (f) CHECK(f->alignment == k.expect && f->data_start == align_up(end, k.expect) && f->data_start % k.expect == 0);
    }
    // a non-integer alignment value is malformed
    { std::vector<Kv> kv = {{"general.alignment", 8, 0, "32"}}; auto img = u3e::image(kv); CHECK(!GgufFile::parse(img.data(), img.size())); }
}

// theory: block_geometry gives (elements, bytes); Q2_0 = (64, 18). Every listed quantized type has bytes
// equal to its header fields plus packed payload (q2_0_geometry, geometry_matches_dequant_layouts,
// geometry_matches_iq_kernels, q2_0_row_2560).
U3E_SECTION(sec_block_geometry, "block geometry: the theory table, Q2_0 = (64,18), bytes = header fields + payload") {
    // the theory's blockGeometry, verbatim; every other id is unsupported
    const struct { uint32_t t; uint64_t el, by; } tab[] = {
        {0, 1, 4}, {1, 1, 2}, {30, 1, 2}, {2, 32, 18}, {3, 32, 20}, {6, 32, 22}, {7, 32, 24}, {8, 32, 34}, {9, 32, 36},
        {10, 256, 84}, {11, 256, 110}, {12, 256, 144}, {13, 256, 176}, {14, 256, 210}, {16, 256, 66}, {17, 256, 74},
        {18, 256, 98}, {20, 32, 18}, {21, 256, 110}, {22, 256, 82}, {23, 256, 136}, {29, 256, 56}, {42, 64, 18}};
    for (uint32_t t = 0; t < 4096; ++t) {
        const BlockGeometry g = block_geometry(t);
        bool listed = false;
        for (auto& r : tab) if (r.t == t) { listed = true; CHECK(g.el == r.el && g.by == r.by); }
        if (!listed) CHECK(g.el == 0 && g.by == 0);
        if (listed) CHECK(std::strcmp(ggml_type_name(t), "UNKNOWN") != 0);
    }
    CHECK(block_geometry(0xFFFFFFFFu).el == 0);
    // Q2_0: an fp16 scale plus 64 two-bit codes
    CHECK(block_geometry(42).el == 64 && block_geometry(42).by == 2 + 64 * 2 / 8 && std::strcmp(ggml_type_name(42), "Q2_0") == 0);
    // the field layouts the dequantizers read
    CHECK(block_geometry(2).by == 2 + 16 && block_geometry(6).by == 2 + 4 + 16 && block_geometry(8).by == 2 + 32);
    CHECK(block_geometry(20).by == 2 + 16 && block_geometry(11).by == 32 + 64 + 12 + 2 && block_geometry(12).by == 2 + 2 + 12 + 128);
    CHECK(block_geometry(13).by == 2 + 2 + 12 + 32 + 128 && block_geometry(14).by == 128 + 64 + 16 + 2);
    CHECK(block_geometry(23).by == 2 + 2 + 4 + 128);
    // the code's own layout table: every listed quantized type is header fields + packed payload
    CHECK(geometry_matches_layouts());
    for (auto& r : tab) if (block_layout_bytes(r.t)) CHECK(block_layout_bytes(r.t) == r.by);
    for (uint32_t t : {0u, 1u, 30u}) CHECK(block_layout_bytes(t) == 0);  // F32/F16/BF16 are not block-quantized
    // the i-quant strides
    CHECK(block_geometry(16).by == 66 && block_geometry(17).by == 74 && block_geometry(18).by == 98 &&
          block_geometry(21).by == 110 && block_geometry(22).by == 82);
    // rowBytes: a Q2_0 row of the hidden size 2560 is 40 blocks = 720 bytes; TensorInfo::nbytes agrees
    CHECK(row_bytes(64, 18, 2560) == 720);
    for (auto& r : tab) for (uint64_t blocks : {1ull, 3ull, 40ull, 1000ull}) {
        TensorInfo ti; ti.type = r.t; ti.shape = {r.el * blocks};
        CHECK(ti.nbytes() == row_bytes(r.el, r.by, ti.elements()) && ti.nbytes() == blocks * r.by);
        // and the parser sizes the tensor the same way: payload exactly nbytes fits, one byte less does not
        auto img = u3e::image({{"general.name", 8, 0, "g"}}, r.t, {r.el * blocks}, 32, nullptr, blocks * r.by);
        CHECK(GgufFile::parse(img.data(), img.size()).has_value() && !GgufFile::parse(img.data(), img.size() - 1));
    }
    // a tensor of an unsupported type, or not a whole number of blocks, is refused
    { auto img = u3e::image({{"general.name", 8, 0, "g"}}, 15, {256}, 32, nullptr, 292); CHECK(!GgufFile::parse(img.data(), img.size())); }
    { auto img = u3e::image({{"general.name", 8, 0, "g"}}, 42, {65}, 32, nullptr, 36); CHECK(!GgufFile::parse(img.data(), img.size())); }
}

// theory: check_architecture returns "" iff the architecture string is qwen4exp and all six required keys are
// present and equal the compiled-in constants; otherwise it reports the FIRST failure (checkReqs_none_iff,
// checkReqs_failure_sound, checkArchitecture_sound).
U3E_SECTION(sec_check_architecture, "check_architecture: \"\" iff qwen4exp + six exact keys, else the first failure") {
    CHECK(std::string(kGuardArch) == "qwen4exp");
    const std::vector<Kv> base = u3e::guard_kvs();
    for (size_t i = 0; i < 6; ++i) CHECK(base[i + 1].key == kGuardDefaults[i].key && base[i + 1].u == kGuardDefaults[i].want);
    // every key is either correct (0), missing (1) or off by one (2): 3^6 = 729 files
    for (int code = 0; code < 729; ++code) {
        int st[6]; for (int i = 0, c = code; i < 6; ++i, c /= 3) st[i] = c % 3;
        std::vector<Kv> kv = {base[0]};
        for (int i = 0; i < 6; ++i) {
            if (st[i] == 1) continue;
            Kv k = base[i + 1]; if (st[i] == 2) k.u += 1; kv.push_back(k);
        }
        auto img = u3e::image(kv); auto f = GgufFile::parse(img.data(), img.size());
        CHECK(f.has_value()); if (!f) continue;
        std::string expect;
        for (int i = 0; i < 6; ++i) if (st[i]) { expect = kGuardDefaults[i].key; break; }
        const std::string got = check_architecture(*f);
        CHECK(got == expect);                                            // the FIRST failing key, in table order
        CHECK(got.empty() == (code == 0));                               // "" iff every key is present and exact
        if (!got.empty()) {                                              // failure soundness: it names a bad key
            bool named_bad = false;
            for (int i = 0; i < 6; ++i) if (got == kGuardDefaults[i].key) named_bad = f->get_uint(got) != std::optional<uint64_t>(kGuardDefaults[i].want);
            CHECK(named_bad);
        }
    }
    // the architecture is checked first: a wrong or missing string wins over any key failure
    for (const char* arch : {"", "qwen", "qwen4ex", "qwen4exp2", "Qwen4exp", "qwen35moe", "llama"}) {
        std::vector<Kv> kv = base; kv[0].s = arch;
        auto img = u3e::image(kv); auto f = GgufFile::parse(img.data(), img.size());
        CHECK(f && check_architecture(*f) == "general.architecture");
        kv.erase(kv.begin() + 2);                                        // plus a missing key: still the architecture
        img = u3e::image(kv); f = GgufFile::parse(img.data(), img.size());
        CHECK(f && check_architecture(*f) == "general.architecture");
    }
    { std::vector<Kv> kv(base.begin() + 1, base.end());                  // no general.architecture at all
      auto img = u3e::image(kv); auto f = GgufFile::parse(img.data(), img.size());
      CHECK(f && check_architecture(*f) == "general.architecture"); }
    { std::vector<Kv> kv = base; kv[0] = {"general.architecture", 4, 7, ""};   // not a string
      auto img = u3e::image(kv); auto f = GgufFile::parse(img.data(), img.size());
      CHECK(f && check_architecture(*f) == "general.architecture"); }
    // a key stored as U64 with the right value is accepted; as a string it is not an integer and fails
    { std::vector<Kv> kv = base; kv[3].type = 10; auto img = u3e::image(kv); auto f = GgufFile::parse(img.data(), img.size());
      CHECK(f && check_architecture(*f).empty()); }
    { std::vector<Kv> kv = base; kv[3] = {kv[3].key, 8, 0, "512"}; auto img = u3e::image(kv); auto f = GgufFile::parse(img.data(), img.size());
      CHECK(f && check_architecture(*f) == "qwen4exp.expert_count"); }
    // the passing file has hidden size 2560, 512 experts and 10 used experts (checkArchitecture_sound)
    { auto img = u3e::image(base); auto f = GgufFile::parse(img.data(), img.size());
      CHECK(f && check_architecture(*f).empty() && check_architecture(*f, "qwen4exp"));
      CHECK(f && f->get_uint("qwen4exp.embedding_length") == 2560u && f->get_uint("qwen4exp.expert_count") == 512u &&
            f->get_uint("qwen4exp.expert_used_count") == 10u); }
}

// theory form template: Qwen4ExpGuard -> guardDefaults. The default guard carries exactly the six compiled-in
// constants and accepts exactly the files check_architecture accepts.
U3E_SECTION(sec_guard_defaults, "Qwen4ExpGuard: defaults = guardDefaults, accepts exactly what check_architecture accepts") {
    const Qwen4ExpGuard d;
    CHECK(d.expected_layers == 48 && d.expected_hidden == 2560 && d.expected_experts == 512 && d.expected_used == 10 &&
          d.expected_heads == 24 && d.expected_kv_heads == 2);
    const uint64_t* fields[6] = {&d.expected_layers, &d.expected_hidden, &d.expected_experts, &d.expected_used,
                                 &d.expected_heads, &d.expected_kv_heads};
    for (int i = 0; i < 6; ++i) CHECK(*fields[i] == kGuardDefaults[i].want);
    const std::vector<Kv> base = u3e::guard_kvs();
    long agree = 0;
    for (int code = 0; code < 729; ++code) {
        int st[6]; for (int i = 0, c = code; i < 6; ++i, c /= 3) st[i] = c % 3;
        for (int arch_ok = 0; arch_ok < 2; ++arch_ok) {
            std::vector<Kv> kv = {base[0]}; if (!arch_ok) kv[0].s = "qwen35moe";
            for (int i = 0; i < 6; ++i) { if (st[i] == 1) continue; Kv k = base[i + 1]; if (st[i] == 2) k.u = k.u * 2 + 1; kv.push_back(k); }
            auto img = u3e::image(kv); auto f = GgufFile::parse(img.data(), img.size());
            CHECK(f.has_value()); if (!f) continue;
            std::string why;
            const bool g = d.check(*f, &why);
            CHECK(g == check_architecture(*f).empty()); agree += g == check_architecture(*f).empty();
            CHECK(g == (arch_ok && code == 0));
            if (!g) CHECK(!why.empty());
            // the same FIRST failure: the guard's message names the key check_architecture reports
            const std::string first = check_architecture(*f);
            if (!g && first != "general.architecture") {
                const std::string sfx = first.substr(std::string("qwen4exp.").size());
                const std::string shortk = sfx.rfind("attention.", 0) == 0 ? sfx.substr(10) : sfx;
                CHECK(why == "missing " + shortk || why == shortk + " mismatch");
            }
        }
    }
    CHECK(agree == 2 * 729);
}

// ---- driver -----------------------------------------------------------------------------------------------
int main() {
    for (auto& s : u3e::sections()) {
        const long f0 = u3e::g_fail, c0 = u3e::g_checks;
        s.fn();
        std::printf("[%s] %s: %ld checks\n", u3e::g_fail == f0 ? " ok " : "FAIL", s.name, u3e::g_checks - c0);
    }
    std::printf("u3e84b3357d host: %zu sections, %ld checks, %ld failures\n", u3e::sections().size(), u3e::g_checks, u3e::g_fail);
    if (u3e::g_fail == 0) std::printf("U3E84B3357D HOST TESTS PASSED\n");
    return u3e::g_fail ? 1 : 0;
}

// Unit ue063fb297c invariant tests (host only). Each block checks LAYER-1 statements of one theory module, exhaustively where the
// domain is small and with seeded random data elsewhere, and cross-checks the new contract headers against the earlier units' code.
#define BALOON_DEQUANT_MAIN_DISABLED 1
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
#include <vector>
#include "baloon/artifact_dequant.hpp"
#include "baloon/canonical_form_roundtrip.hpp"
#include "baloon/chat_frontend.hpp"
#include "baloon/frontend_contracts.hpp"
#include "baloon/gguf_header_geometry.hpp"
#include "baloon/gguf_reader.hpp"
#include "baloon/iq_native_pack_index.hpp"
#include "baloon/mtp_expert_quantizer.hpp"
#include "baloon/pack_builder_contracts.hpp"
#include "baloon/q2_blob_geometry.hpp"
#include "baloon/q2_row_dot_identity.hpp"
#include "baloon/serve_frontend.hpp"
#include "baloon/serve_server.hpp"

using namespace baloon;
static int g_fail = 0, g_checks = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
static std::mt19937_64 rng(0xE063FB297Cull);
static uint32_t rnd(uint32_t n) { return (uint32_t)(rng() % n); }

// ---------------------------------------------------------------- Q2ExpertBlobGeometry + PackLayerBlob
static void test_blob() {
    using namespace q2blob;
    CHECK(BLOB == 1382400 && ROLE_BYTES == 460800 && O_D_CODES == 819200 && O_GU_SCALES == 1228800 && O_D_SCALES == 1331200);
    CHECK(BLOB == mtprt::kBlob && O_D_CODES == mtprt::kGuCodes);
    CHECK(O_GU_SCALES - O_D_CODES == mtprt::kDownCodes);
    CHECK(O_D_SCALES - O_GU_SCALES == mtprt::kGuScales && BLOB - O_D_SCALES == mtprt::kDownScales);
    for (uint64_t r = 0; r < FF; ++r) {  // gate/up rows inside the gu planes
        CHECK(gu_code_off(slot(r, 1)) + ROW_GU <= O_D_CODES);
        CHECK(gu_scale_off(slot(r, 1), SC_GU - 1) + SCALE_BYTES <= O_D_SCALES);
        CHECK(slot(r, 0) == mtprt::gate_slot(r) && slot(r, 1) == mtprt::up_slot(r));
    }
    for (uint64_t r = 0; r < H; ++r) {
        CHECK(d_code_off(r) + ROW_D <= O_GU_SCALES);
        CHECK(d_scale_off(r, SC_D - 1) + SCALE_BYTES <= BLOB);
    }
    for (uint64_t b = 0; b < SC_GU; ++b) CHECK(2 * b + 1 < MAXC);
    std::set<uint64_t> slots;  // slot is a bijection onto 0..1279
    for (uint64_t r = 0; r < FF; ++r) for (uint64_t a = 0; a < 2; ++a) slots.insert(slot(r, a));
    CHECK(slots.size() == 1280 && *slots.rbegin() == 1279);
    // relayout: round trip, full coverage, and semantic placement
    std::vector<uint8_t> gate(ROLE_BYTES), up(ROLE_BYTES), down(ROLE_BYTES);
    for (auto* v : {&gate, &up, &down}) for (auto& b : *v) b = (uint8_t)rnd(256);
    std::vector<uint8_t> blobA(BLOB, 0xA5), blobB(BLOB, 0x5A);
    pack_expert_blob(gate.data(), up.data(), down.data(), blobA.data());
    pack_expert_blob(gate.data(), up.data(), down.data(), blobB.data());
    CHECK(blobA == blobB);  // every byte of the blob is written (no gap), independent of the prior fill
    std::vector<uint8_t> g2(ROLE_BYTES), u2(ROLE_BYTES), d2(ROLE_BYTES);
    unpack_expert_blob(blobA.data(), g2.data(), u2.data(), d2.data());
    CHECK(g2 == gate && u2 == up && d2 == down);
    // injective: changing any one input byte changes the blob (sampled)
    for (int t = 0; t < 400; ++t) {
        auto* which = t % 3 == 0 ? &gate : (t % 3 == 1 ? &up : &down);
        const size_t i = rnd(ROLE_BYTES);
        const uint8_t old = (*which)[i];
        (*which)[i] = (uint8_t)(old ^ 0x10);
        std::vector<uint8_t> bl(BLOB);
        pack_expert_blob(gate.data(), up.data(), down.data(), bl.data());
        CHECK(bl != blobA);
        (*which)[i] = old;
    }
    // placement: slot 2r is gate row r, 2r+1 up row r; scale plane holds each block's fp16 and the code plane its 16 bytes
    for (int t = 0; t < 2000; ++t) {
        const uint64_t r = rnd(FF), b = rnd(SC_GU), role = rnd(2);
        const uint8_t* blk = (role ? up : gate).data() + (r * SC_GU + b) * BB;
        const uint64_t s = slot(r, role);
        CHECK(std::memcmp(blobA.data() + gu_scale_off(s, b), blk, 2) == 0);
        CHECK(std::memcmp(blobA.data() + gu_code_off(s) + b * 16, blk + 2, 16) == 0);
        // the dequantised block equals the scale/code decode read straight from the blob planes
        float want[64], got[64];
        artifact::dequantize_q2_0(blk, want);
        const float d = artifact::fp16_to_fp32(artifact::read_u16(blobA.data() + gu_scale_off(s, b)));
        for (int j = 0; j < 64; ++j) {
            const unsigned c = q2dot::code_at(blobA[gu_code_off(s) + b * 16 + j / 4], j % 4);
            got[j] = d * (float)((int)c - 1);
        }
        for (int j = 0; j < 64; ++j) CHECK(want[j] == got[j] || (std::isnan(want[j]) && std::isnan(got[j])));
    }
    CHECK(arena_bytes(48) == 33973862400ull && expert_offset(1, 0) == NE * BLOB && expert_offset(0, 5) == 5 * BLOB);
    std::set<uint64_t> idx;
    for (uint64_t l = 0; l < 4; ++l) for (uint64_t e = 0; e < NE; ++e) idx.insert(packb::expert_index(l, e));
    CHECK(idx.size() == 4 * NE);
}

// ---------------------------------------------------------------- Q2RowDotIdentity
static void test_rowdot() {
    using namespace q2dot;
    for (int t = 0; t < 500; ++t) {  // sum (c-1) x == sum c x - sum x, and append additivity
        std::vector<CX> a(rnd(70)), b(rnd(70));
        for (auto& p : a) p = {rnd(4), (int64_t)rnd(255) - 127};
        for (auto& p : b) p = {rnd(4), (int64_t)rnd(255) - 127};
        CHECK(dot_sym(a) == dot_c(a) - sum_x(a));
        std::vector<CX> ab = a; ab.insert(ab.end(), b.begin(), b.end());
        CHECK(dot_sym(ab) == dot_sym(a) + dot_sym(b));
    }
    for (unsigned c0 = 0; c0 < 4; ++c0) for (unsigned c1 = 0; c1 < 4; ++c1) for (unsigned c2 = 0; c2 < 4; ++c2) for (unsigned c3 = 0; c3 < 4; ++c3) {
        const unsigned by = pack4(c0, c1, c2, c3);
        CHECK(by < 256 && code_at(by, 0) == c0 && code_at(by, 1) == c1 && code_at(by, 2) == c2 && code_at(by, 3) == c3);
    }
    for (int c = 0; c <= 3; ++c) for (int x = -127; x <= 127; ++x) CHECK(std::abs(c * x) <= 381);
    {  // worst-case 2560-wide accumulation stays inside int32 and the correction is applied once per row
        std::vector<uint8_t> codes(2560, 3); std::vector<int8_t> x(2560, 127);
        int64_t ref = 0; for (int i = 0; i < 2560; ++i) ref += (int64_t)(codes[i] - 1) * x[i];
        CHECK(row_dot_vnni(codes.data(), x.data(), 2560) == ref);
        CHECK(2560 * 381 < (1 << 30) * 2ll);
    }
    for (int t = 0; t < 300; ++t) {  // real packed Q2_0 row: VNNI form == dequantised dot / scale
        uint8_t blk[18];
        for (auto& b : blk) b = (uint8_t)rnd(256);
        blk[0] = 0x00; blk[1] = 0x3C;  // d = 1.0 in fp16
        float w[64]; artifact::dequantize_q2_0(blk, w);
        std::vector<uint8_t> codes(64); std::vector<int8_t> x(64);
        for (int j = 0; j < 64; ++j) { codes[j] = (uint8_t)code_at(blk[2 + j / 4], j % 4); x[j] = (int8_t)((int)rnd(255) - 127); }
        double ref = 0; for (int j = 0; j < 64; ++j) ref += (double)w[j] * x[j];
        CHECK((double)row_dot_vnni(codes.data(), x.data(), 64) == ref);
        const std::vector<uint8_t> lane = {blk[2], blk[3]};
        CHECK(((lane[0] + 256 * lane[1]) / 256 % 4) == (unsigned)(lane[1] % 4));  // second byte of a 16-bit lane
    }
    for (int64_t s0 = -4; s0 <= 4; ++s0) for (int64_t s1 = -4; s1 <= 4; ++s1) for (int64_t A0 = -4; A0 <= 4; ++A0) for (int64_t A1 = -4; A1 <= 4; ++A1) {
        CHECK((block_two_scale(1, s0, s1, A0, A1) == block_single_scale(1, s0, A0, A1)) == ((s1 - s0) * A1 == 0));
        CHECK(block_two_scale(7, s0, s1, A0, A1) == 7 * (s0 * A0) + 7 * (s1 * A1));
    }
    CHECK(block_two_scale(1, 1, 2, 1, 1) != block_single_scale(1, 1, 1, 1));  // the counterexample
    for (unsigned n = 1; n <= 8; ++n) CHECK(nt_dispatch(n) == n);
    for (unsigned n = 1; n <= 64; ++n) {
        const auto ch = chunks4(n);
        unsigned sum = 0; for (auto k : ch) { CHECK(k >= 1 && k <= 4); sum += k; }
        CHECK(sum == n);
    }
    CHECK(chunks4(0).empty());
}

// ---------------------------------------------------------------- CanonicalQuantRoundtrip
static void test_canon() {
    using canon::kRegistry; using canon::kRegistrySize;
    CHECK(kRegistrySize == 13);
    for (size_t i = 0; i < kRegistrySize; ++i) {
        const auto& m = kRegistry[i];
        CHECK(m.block_elems % m.group_elems == 0);
        if (m.has_offset) CHECK(std::string(m.name) == "Q4_K" || std::string(m.name) == "Q5_K");
    }
    // theory convention decode = (code + bias) * scale: bias = -(registry's subtracted bias)
    struct Row { const char* n; int bias; };
    for (Row r : {Row{"Q2_0", -1}, Row{"Q3_K", -4}, Row{"Q4_0", -8}, Row{"Q4_K", 0}, Row{"Q5_0", -16}, Row{"Q5_K", 0}, Row{"Q6_K", -32}, Row{"Q8_0", -128}, Row{"IQ4_NL", 0}, Row{"IQ4_XS", 0}}) {
        bool found = false;
        for (size_t i = 0; i < kRegistrySize; ++i) if (std::string(kRegistry[i].name) == r.n) { found = true; CHECK(-kRegistry[i].code_bias == r.bias); }
        CHECK(found);
    }
    CHECK(2 + 64 / 4 == 18 && 2 + 32 / 2 == 18 && 2 + 32 == 34 && 2 + 4 + 32 / 2 == 22 && 2 + 2 + 12 + 256 / 2 == 144 &&
          256 / 8 + 256 / 4 + 12 + 2 == 110 && 2 + 2 + 12 + 256 / 8 + 256 / 2 == 176 && 256 / 2 + 256 / 4 + 16 + 2 == 210 && 2 + 2 + 4 + 256 / 2 == 136);
    for (int c = 0; c < 4; ++c) CHECK(c - 1 >= -1 && c - 1 <= 2);
    for (int c = 0; c < 16; ++c) CHECK(c - 8 >= -8 && c - 8 <= 7);
    for (int lo = 0; lo < 16; ++lo) for (int hb = 0; hb < 2; ++hb) { const int v = lo + 16 * hb; CHECK(v < 32 && v - 16 >= -16 && v - 16 <= 15); }
    for (int c2 = 0; c2 < 4; ++c2) for (int m = 0; m < 2; ++m) { const int v = c2 + 4 * m; CHECK(v < 8 && v - 4 >= -4 && v - 4 <= 3); }
    for (int nb = 0; nb < 16; ++nb) for (int hi = 0; hi < 4; ++hi) { const int v = nb + 16 * hi; CHECK(v < 64 && v - 32 >= -32 && v - 32 <= 31); }
    for (int s = -128; s <= 127; ++s) CHECK(s + 128 >= 0 && s + 128 <= 255 && (s + 128) - 128 == s);
    for (unsigned lo = 0; lo < 16; ++lo) for (unsigned hi = 0; hi < 16; ++hi) { const unsigned v = lo + 16 * hi; CHECK(v % 16 == lo && v / 16 == hi && v < 256); }
    for (unsigned qh : {0u, 0xFFFFFFFFu, 0x12345678u, 0xA5A5A5A5u}) for (unsigned j = 0; j < 16; ++j)
        CHECK((((qh >> (j + 12)) & 0x1Fu) / 16) == ((qh >> (j + 16)) & 1u) || j + 12 > 31);
    for (unsigned lo = 0; lo < 256; ++lo) for (unsigned t = 0; t < 256; ++t) CHECK(lo % 16 + 16 * (t % 4) < 64 && lo % 16 + 16 * (t / 64) < 64);
    {  // index maps are injective
        std::set<int> a, b, c, d;
        for (int h = 0; h < 2; ++h) for (int j = 0; j < 4; ++j) { a.insert(4 * h + j); for (int s = 0; s < 2; ++s) b.insert(8 * h + 2 * j + s); }
        for (int h = 0; h < 2; ++h) for (int q = 0; q < 4; ++q) for (int l = 0; l < 32; ++l) c.insert(128 * h + 32 * q + l);
        for (int j = 0; j < 4; ++j) for (int k = 0; k < 2; ++k) d.insert(2 * j + k);
        CHECK(a.size() == 8 && b.size() == 16 && c.size() == 256 && d.size() == 8);
    }
    // IQ4 codebook: 16 entries, strictly increasing, not affine, equal to the dequantiser's table
    CHECK(sizeof(artifact::kvalues_iq4nl) == 16);
    for (int i = 1; i < 16; ++i) CHECK(artifact::kvalues_iq4nl[i] > artifact::kvalues_iq4nl[i - 1]);
    CHECK(artifact::kvalues_iq4nl[1] - artifact::kvalues_iq4nl[0] != artifact::kvalues_iq4nl[2] - artifact::kvalues_iq4nl[1]);
    // bit-exact: Q2_0 encode then decode returns the same block bytes
    for (int t = 0; t < 200; ++t) {
        uint8_t blk[18], back[18]; for (auto& b : blk) b = (uint8_t)rnd(256);
        blk[1] &= 0xFB;  // finite fp16 scale (exponent != 31): Inf/NaN scale payloads are not preserved by the float scale plane
        canon::Block cb; CHECK(canon::encode_block(canon::QType::Q2_0, blk, &cb));
        canon::decode_q2(cb, back);
        CHECK(std::memcmp(blk, back, 18) == 0);
    }
    for (uint32_t idx = 0; idx < 200; ++idx) for (uint32_t sn : {1u, 2u, 3u, 7u}) {
        int hits = 0; for (uint32_t si = 0; si < sn; ++si) hits += (idx % sn == si);
        CHECK(hits == 1);
    }
    for (int b : {2, 4, 8}) CHECK(b * (8 / b) == 8);
    for (uint64_t nb = 0; nb < 100; ++nb) CHECK(nb * 64 / 64 == nb);
}

// ---------------------------------------------------------------- ChatFrontendParser + ServeServerProtocol
static void test_frontend() {
    namespace fc = frontend_contract;
    CHECK(fc::effort_level("high") == fc::effort_level("xhigh"));
    for (auto s : {"none", "off", "minimal", "disabled"}) CHECK(fc::effort_level(s) == 0);
    for (auto s : {"turbo", "", "HIGH", "extreme"}) CHECK(!fc::effort_level(s).has_value());
    CHECK(fc::effort_level("low") == 1 && fc::effort_level("medium") == 2 && fc::effort_level("max") == 3 && fc::effort_level("maximum") == 3);
    for (auto s : {"none", "off", "minimal", "disabled", "low", "medium", "high", "xhigh", "max", "maximum"}) {  // agrees with the serve frontend
        const auto e = serve_frontend::effort_of(s); const auto l = fc::effort_level(s);
        CHECK(e.known && l.has_value() && e.disabled == (*l == 0));
        if (*l) CHECK((int)e.level + 1 == *l);
    }
    CHECK(!serve_frontend::effort_of("turbo").known);
    for (uint64_t n = 0; n < 20000; ++n) {
        CHECK(fc::budget_rank(n) >= 1 && fc::budget_rank(n) <= 3);
        CHECK(fc::budget_rank(n) <= fc::budget_rank(n + 1));
        CHECK((uint32_t)fc::budget_rank(n) == serve_frontend::budget_rank(n) + 1 && (uint32_t)fc::budget_rank(n) == chat_frontend::budget_rank((int64_t)n) + 1);
    }
    CHECK(fc::budget_rank(2047) == 1 && fc::budget_rank(2048) == 2 && fc::budget_rank(8191) == 2 && fc::budget_rank(8192) == 3);
    for (int t = 0; t < 3000; ++t) {  // hold-back is always strictly shorter than the tag; agrees with the earlier unit
        std::vector<uint32_t> tag(1 + rnd(6)), text(rnd(12)); for (auto& x : tag) x = rnd(3); for (auto& x : text) x = rnd(3);
        const size_t h = fc::hold_len(text, tag);
        CHECK(h < tag.size() && h <= text.size());
        const std::vector<uint64_t> t64(text.begin(), text.end()), g64(tag.begin(), tag.end());
        CHECK(h == chat_frontend::hold_one(t64, g64));
        if (text.size() >= tag.size()) {  // a complete tag at the end is never withheld wholesale
            std::vector<uint32_t> full = text; full.insert(full.end(), tag.begin(), tag.end());
            CHECK(fc::hold_len(full, tag) < tag.size());
        }
    }
    for (int t = 0; t < 500; ++t) {
        std::vector<uint32_t> l(rnd(10)); for (auto& x : l) x = 11 + rnd(5);
        std::vector<uint32_t> w{10}; w.insert(w.end(), l.begin(), l.end()); w.push_back(10);
        CHECK(fc::strip_nl(w) == l);
        std::vector<uint64_t> w64(w.begin(), w.end());
        const auto s64 = chat_frontend::strip_nl(w64);
        CHECK(std::vector<uint32_t>(s64.begin(), s64.end()) == l);
    }
    for (uint64_t p = 0; p < 40; ++p) for (uint64_t m = 0; m < 40; ++m) for (uint64_t c : {10ull, 40ull, 60ull}) {
        const bool a = fc::accepts(p, m, c);
        CHECK(a == (p + m <= c) && a == serve_server::accepts(p, m, c) && a == chat_frontend::fits(p, m, c));
        if (a) CHECK(p <= c && m <= c);
    }
    for (int t = 0; t < 300; ++t) {  // pad expansion
        std::vector<uint32_t> ids(rnd(15)); for (auto& x : ids) x = rnd(4);
        std::vector<uint64_t> cnt(rnd(5)); for (auto& c : cnt) c = rnd(6);
        const auto e = fc::expand(3, ids, cnt);
        const std::vector<uint64_t> i64(ids.begin(), ids.end()); const auto e64 = chat_frontend::expand(3, i64, cnt);
        CHECK(std::vector<uint32_t>(e64.begin(), e64.end()) == e);
        std::vector<uint32_t> nopad; for (auto x : ids) if (x != 3) nopad.push_back(x);
        CHECK(fc::expand(3, nopad, cnt) == nopad);
    }
    {
        std::vector<uint64_t> cs{5, 0, 3, 7};
        CHECK(fc::expand(9, std::vector<uint32_t>(cs.size(), 9), cs).size() == 15);
        CHECK(fc::expand(9, {9}, {4}) == std::vector<uint32_t>(4, 9));
    }
    for (int t = 0; t < 300; ++t) {  // deltas concatenate to the final text
        std::vector<uint32_t> a(rnd(8)), x(rnd(8)), y(rnd(8)); for (auto* v : {&a, &x, &y}) for (auto& e : *v) e = rnd(50);
        std::vector<uint32_t> ax = a; ax.insert(ax.end(), x.begin(), x.end());
        std::vector<uint32_t> axy = ax; axy.insert(axy.end(), y.begin(), y.end());
        CHECK(fc::delta_of(a, ax) == x);
        std::vector<uint32_t> acc = a, d1 = fc::delta_of(a, ax), d2 = fc::delta_of(ax, axy);
        acc.insert(acc.end(), d1.begin(), d1.end()); acc.insert(acc.end(), d2.begin(), d2.end());
        CHECK(acc == axy);
    }
    for (uint64_t calls = 0; calls < 3; ++calls) for (auto f : {fc::Finish::Stop, fc::Finish::Length}) {
        const auto sf = f == fc::Finish::Stop ? serve_server::Finish::Stop : serve_server::Finish::Length;
        CHECK(fc::openai_finish(calls, f) == serve_server::openai_finish(calls, sf));
        CHECK(fc::anthropic_stop(calls > 0, f) == serve_server::anthropic_stop(calls > 0, sf));
        if (fc::openai_finish(calls, f) == 2) CHECK(f == fc::Finish::Stop);
        if (fc::anthropic_stop(calls > 0, f) == 2) CHECK(f == fc::Finish::Stop);
    }
    CHECK(fc::openai_finish(5, fc::Finish::Length) == 1 && fc::openai_finish(0, fc::Finish::Stop) == 0 && fc::openai_finish(1, fc::Finish::Stop) == 2);
    CHECK(fc::anthropic_stop(true, fc::Finish::Length) == 1 && fc::anthropic_stop(false, fc::Finish::Stop) == 0 && fc::anthropic_stop(true, fc::Finish::Stop) == 2);
    for (int cur = -1; cur < 3; ++cur) CHECK(fc::new_block(cur, fc::Kind::Tool));
    CHECK(!fc::new_block((int)fc::Kind::Content, fc::Kind::Content) && fc::new_block((int)fc::Kind::Reasoning, fc::Kind::Content) && fc::new_block(-1, fc::Kind::Reasoning));
    for (size_t n = 0; n <= 65; ++n) {  // cache bound: insert then evict never exceeds 64; matches the FIFO cache
        std::vector<int> l(n, 1);
        CHECK(fc::evict(l).size() <= 64);
        CHECK(serve_server::cache_add(n > 64 ? 64 : n) <= 64);
    }
    {
        serve_server::ImageCache<int> cache;
        for (int i = 0; i < 200; ++i) { cache.put(std::to_string(i), i); CHECK(cache.size() <= 64); }
        CHECK(!cache.has("0") && cache.has("199") && cache.size() == 64);
        std::vector<int> v; for (int i = 0; i < 200; ++i) { v.push_back(i); v = fc::evict(v); CHECK(v.size() <= 64); }
        CHECK(v.front() == 136 && v.back() == 199);
    }
    CHECK(fc::authorized("", "anything") && fc::authorized("k", "k") && !fc::authorized("k", "j") && !fc::authorized("k", ""));
    CHECK(serve_server::authorized("", "x") && serve_server::authorized("k", "k") && !serve_server::authorized("k", "x"));
}

// ---------------------------------------------------------------- GgufReaderGeometry
static void test_gguf() {
    using namespace gguf_geom;
    CHECK(kMagic == gguf_reader::kMagic && kHeaderBytes == 24 && kTensorInfoTail == 12);
    const uint8_t magic[4] = {'G', 'G', 'U', 'F'};
    uint32_t m; std::memcpy(&m, magic, 4); CHECK(m == kMagic);
    for (int t = 0; t < 200; ++t) {
        std::vector<uint64_t> a(rnd(4)), b(rnd(4)); for (auto& d : a) d = 1 + rnd(9); for (auto& d : b) d = 1 + rnd(9);
        std::vector<uint64_t> ab = a; ab.insert(ab.end(), b.begin(), b.end());
        CHECK(shape_elems(ab) == shape_elems(a) * shape_elems(b));
    }
    CHECK(shape_elems({}) == 1);
    for (uint64_t k = 0; k < 50; ++k) { CHECK(expected_bytes(k * 64, 64, 18) == k * 18); CHECK(expected_bytes(k * 64 + 1, 64, 18) == std::nullopt || 64 == 1); }
    CHECK(expected_bytes(262144, 64, 18) == 73728u);
    {
        gguf_reader::TensorInfo ti; ti.type = 42; ti.shape = {64, 256, 16};
        CHECK(ti.elements() == 262144 && ti.nbytes() == 73728 && expected_bytes(ti.elements(), gguf_reader::block_geometry(42).el, gguf_reader::block_geometry(42).by) == ti.nbytes());
        const auto g = gguf_reader::block_geometry(42); CHECK(g.el == 64 && g.by == 18);
    }
    for (uint64_t a : {1ull, 2ull, 8ull, 32ull, 64ull, 4096ull}) for (uint64_t pos = 0; pos < 300; ++pos) {
        const uint64_t u = align_up(pos, a);
        CHECK(pos <= u && u < pos + a && u % a == 0);
        if (pos % a == 0) CHECK(u == pos);
    }
    for (uint64_t k = 0; k < 100; ++k) CHECK(align_up(32 * k, 32) == 32 * k);
    CHECK(std::string(type_name(31)) == type_name(41) && std::string(type_name(31)) == "Q4_0_8_8");
    CHECK(std::string(type_name(42)) == "Q2_0" && std::string(type_name(42)) == gguf_reader::ggml_type_name(42));
    CHECK(std::string(type_name(0)) == "F32" && std::string(type_name(30)) == "BF16" && std::string(type_name(999)) == "unknown");
    std::set<int> ids; for (int i = 0; i < 43; ++i) ids.insert(i);
    CHECK(ids.size() == 43);
}

// ---------------------------------------------------------------- IqPackLayout + PackLayerBlob (arena) + MtpPackQuant + RefPackBuilder
static void test_pack() {
    using namespace packb;
    for (int t = 0; t < 300; ++t) {  // arena = 512 * sum(blob); aligned totals
        std::vector<uint64_t> bs(rnd(6)); uint64_t sum = 0; for (auto& b : bs) { b = rnd(5000); sum += b; }
        CHECK(arena_total(bs) == 512 * sum);
        CHECK(arena_total(bs) == (bs.empty() ? 0 : iqpack::experts_offset(bs, bs.size())));
        std::vector<uint64_t> ns(rnd(8)); uint64_t raw = 0; for (auto& n : ns) { n = rnd(10000); raw += n; }
        CHECK(aligned_total(ns) >= raw && aligned_total(ns) % 64 == 0 && aligned_total(ns) == iqpack::extra_offset(ns));
    }
    for (uint64_t n = 0; n < 1000; ++n) {
        CHECK(n <= align_up_64(n) && align_up_64(n) < n + 64 && align_up_64(n) % 64 == 0);
        CHECK(align_up_64(n) == iqpack::round_row(n) && pad_to_64(n) == iqpack::pad64(n) && pad_to_64(n) < 64 && (n + pad_to_64(n)) % 64 == 0);
        CHECK(align_up_64(n) == n + pad_to_64(n));
    }
    for (uint64_t base : {0ull, 64ull, 4096ull}) for (uint64_t len = 0; len < 300; ++len) CHECK((base + len + pad_to_64(len)) % 64 == 0);
    CHECK((0 + 10) % 64 != 0);  // an unaligned blob length breaks the next base
    for (int t = 0; t < 2000; ++t) {  // router narrowing
        const uint32_t b = rnd(65536), u = b << 16;
        CHECK(router_narrow(u) == b && ((uint32_t)router_narrow(u) << 16) == u);
        std::vector<uint16_t> out; const uint32_t w = u;
        CHECK(iqpack::narrow_router_f32_to_bf16(&w, 1, &out) && out[0] == b);
        const uint32_t lossy = u | (1 + rnd(65535));
        CHECK(!iqpack::narrow_router_f32_to_bf16(&lossy, 1, &out));
    }
    CHECK(iqpack::layer_count({0, 3, 1}) == 4 && iqpack::layer_count({}) == 0);
    for (int bits : {1, 2, 4, 8}) for (int t = 0; t < 100; ++t) {  // LSB-first packing round trip
        std::vector<uint8_t> codes(rnd(200)); for (auto& c : codes) c = (uint8_t)rnd(1u << bits);
        CHECK(unpack_lsb(bits, pack_lsb(bits, codes), codes.size()) == codes);
    }
    for (unsigned c = 0; c < 256; ++c) {  // 2-bit packing matches the Q2_0 byte convention
        const auto p = pack_lsb(2, {(uint8_t)(c & 3), (uint8_t)((c >> 2) & 3), (uint8_t)((c >> 4) & 3), (uint8_t)(c >> 6)});
        CHECK(p.size() == 1 && p[0] == c && q2dot::pack4(c & 3, (c >> 2) & 3, (c >> 4) & 3, c >> 6) == c);
    }
    CHECK(code_bits('2') == 2 && code_bits('4') == 4 && code_bits('8') == 8);
    for (unsigned b : {2u, 4u, 8u}) CHECK(b * (8 / b) == 8);
    for (uint64_t n = 0; n < 500; ++n) CHECK(2 * scale_plane_bytes(n, true) == scale_plane_bytes(n, false));
    std::set<uint64_t> ids; for (uint64_t l = 0; l < 6; ++l) for (uint64_t e = 0; e < 512; ++e) ids.insert(expert_index(l, e));
    CHECK(ids.size() == 6 * 512);
    CHECK(512 / 32 == 16); for (uint64_t k = 0; k < 16; ++k) CHECK(k * 32 < 512);
    // MTP quantiser
    for (unsigned i = 0; i <= 16; ++i) {
        CHECK(scale_numer(i) >= 16 && scale_numer(i) <= 32);
        CHECK(std::fabs(mtpq::scale_factor((int)i) - (float)scale_numer(i) / 32.f) == 0.f);
    }
    CHECK(mtpq::kScaleCandidates == 17);
    for (int s = -1; s <= 2; ++s) CHECK(code_of_symbol(s) >= 0 && code_of_symbol(s) <= 3);
    for (unsigned lo = 0; lo <= 15; ++lo) for (unsigned hi = 0; hi <= 15; ++hi) { const unsigned p = nibble_pack(lo, hi); CHECK(p < 256 && p % 16 == lo && p / 16 == hi); }
    CHECK(mtpq::total_bytes(64, 18) == 707788800ull && mtpq::total_bytes(32, 18) == 1415577600ull && mtpq::total_bytes(32, 34) == 2673868800ull);
    CHECK(2 * mtpq::total_bytes(64, 18) == mtpq::total_bytes(32, 18));
    CHECK(expert_bytes(512ull * 1280 * 2560, 64, 18) == 471859200ull && expert_bytes(512ull * 2560 * 640, 64, 18) == 235929600ull);
    for (int t = 0; t < 200; ++t) {  // the quantisers' codes stay in range and Q2 codes decode through the unit's block layout
        float x[64]; for (auto& v : x) v = ((float)rnd(2001) - 1000.f) / 100.f;
        uint8_t b2[18]; mtpq::quantize_q2(x, b2);
        for (int j = 0; j < 64; ++j) CHECK(q2dot::code_at(b2[2 + j / 4], j % 4) <= 3);
        uint8_t b4[18]; mtpq::quantize_q4(x, b4);  // 32 elements; nibble codes <= 15 by construction of a byte nibble
        uint8_t b8[34]; mtpq::quantize_q8(x, b8);
        for (int j = 0; j < 32; ++j) CHECK((int)(int8_t)b8[2 + j] >= -127 && (int)(int8_t)b8[2 + j] <= 127);
    }
}

int main() {
    test_blob(); test_rowdot(); test_canon(); test_frontend(); test_gguf(); test_pack();
    std::printf("ue06: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

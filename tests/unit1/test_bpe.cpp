#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <string>
#include <vector>
#include "baloon/byte_level_bpe_tokenizer.hpp"
#include "check.hpp"

using namespace baloon::bpe;

void test_bpe_byte_map() {
    int shifted = 0, same = 0;
    std::set<uint32_t> images;
    uint32_t next_shift = 256;
    for (unsigned b = 0; b < 256; ++b) {
        const uint32_t cp = byte_to_codepoint((uint8_t)b);
        if (printable(b)) { CHECK(cp == b); ++same; }
        else { CHECK(cp == next_shift); ++next_shift; ++shifted; }       // 256 + n in byte order
        CHECK(cp < 324);
        CHECK(images.insert(cp).second);                                 // injective
        CHECK(codepoint_to_byte(cp) == (int)b);                          // decode inverts encode
    }
    CHECK(shifted == 68 && same == 188 && images.size() == 256);
    CHECK(!printable(32) && !printable(10) && byte_to_codepoint(32) == 288 && byte_to_codepoint(10) == 266);
    CHECK(byte_to_codepoint(127) == 289 && byte_to_codepoint(173) == 256 + 67);
    for (unsigned b : {33u, 126u, 161u, 172u, 174u, 255u}) CHECK(printable(b));
    for (unsigned b : {0u, 32u, 127u, 160u, 173u}) CHECK(!printable(b));
    CHECK(codepoint_to_byte(400) == -1 && codepoint_to_byte(324) == -1 && codepoint_to_byte(0x4E2D) == -1);
}

void test_bpe_lossless_bytes() {
    std::mt19937_64 rng(1);
    for (int it = 0; it < 20000; ++it) {
        std::string s(rng() % 40, '\0');
        for (auto& c : s) c = (char)rng();                                // arbitrary bytes, valid UTF-8 or not
        bool ok = false;
        const std::string m = bytes_to_mapped(s);
        std::vector<uint32_t> cps;
        CHECK(utf8_decode(m, cps) && cps.size() == s.size());             // one code point per byte
        CHECK(mapped_to_bytes(m, &ok) == s && ok);
    }
    bool ok = true;
    mapped_to_bytes(std::string("\xC4\x90") /* U+0110 = 272 -> a byte */, &ok);
    CHECK(ok);
    std::string bad;
    utf8_append(400, bad);
    mapped_to_bytes(bad, &ok);
    CHECK(!ok);
    mapped_to_bytes(std::string("\xFF\xFE"), &ok);
    CHECK(!ok);
    mapped_to_bytes(std::string("\xC4"), &ok);                            // truncated sequence
    CHECK(!ok);
    // every UTF-8 text survives: mapped form decodes back to the original UTF-8 bytes
    const std::string text = "h\xC3\xA9llo \xE4\xB8\xAD\xE6\x96\x87 \xF0\x9F\x99\x82\n";
    CHECK(mapped_to_bytes(bytes_to_mapped(text), &ok) == text && ok);
}

void test_bpe_merge_step() {
    for (size_t n = 0; n <= 10; ++n) {
        std::vector<int> l(n);
        for (size_t i = 0; i < n; ++i) l[i] = (int)i + 1;
        if (n < 2) { for (size_t i = 0; i + 2 <= n; ++i) CHECK(false); continue; }   // no pair in a short word
        for (size_t i = 0; i + 2 <= n; ++i) {
            const auto r = merge_at<int>(l, i, 99);
            CHECK(r.size() == n - 1);                                     // shortens by exactly one
            CHECK(r.size() < l.size());
            CHECK(r[i] == 99);
            for (size_t k = 0; k < i; ++k) CHECK(r[k] == l[k]);
            for (size_t k = i + 2; k < n; ++k) CHECK(r[k - 1] == l[k]);
        }
    }
}

void test_bpe_special_and_bookkeeping() {
    for (bool p : {false, true}) {
        CHECK(matched(kTypeUserDefined, p));
        CHECK(!matched(kTypeNormal, p));
        for (int t : {0, 1, 2, 5, 6}) CHECK(!matched(t, p));
    }
    CHECK(!matched(kTypeControl, false) && matched(kTypeControl, true));
    CHECK(kNumMerges < kVocabSize && 256 + kNumMerges <= kVocabSize && kVocabSize == 248320 && kNumMerges == 247587);
    for (size_t n = 0; n < 10; ++n) CHECK(valid_merge(n) == (n == 2));
    std::string a, b;
    CHECK(parse_merge_line("\xC4\xA0t he", &a, &b) && a == "\xC4\xA0t" && b == "he");
    CHECK(!parse_merge_line("a", &a, &b) && !parse_merge_line("a b c", &a, &b) && !parse_merge_line("", &a, &b));
    CHECK(!parse_merge_line("a  b", &a, &b) && !parse_merge_line(" b", &a, &b) && !parse_merge_line("a ", &a, &b));
    for (int64_t id : {(int64_t)-1, (int64_t)-100, (int64_t)INT64_MIN}) CHECK(!id_in_range(id, 10));
    CHECK(id_in_range(0, 10) && id_in_range(9, 10) && !id_in_range(10, 10) && !id_in_range(0, 0));
}

namespace {
// Tiny vocabulary: 256 byte tokens (ids 0..255 in byte order) + merged tokens + two specials.
void build(ByteLevelBpe& t, const std::vector<std::pair<std::string, std::string>>& merges, std::string* err) {
    std::vector<std::pair<std::string, int>> toks;
    for (int b = 0; b < 256; ++b) { std::string s; utf8_append(byte_to_codepoint((uint8_t)b), s); toks.emplace_back(s, kTypeNormal); }
    std::vector<std::string> lines;
    for (auto& m : merges) { toks.emplace_back(m.first + m.second, kTypeNormal); lines.push_back(m.first + " " + m.second); }
    toks.emplace_back("<|endoftext|>", kTypeControl);
    toks.emplace_back("<|user|>", kTypeUserDefined);
    *err = t.load(toks, lines);
}
std::string g(const std::string& raw) { return bytes_to_mapped(raw); }
}  // namespace

void test_bpe_encode_decode() {
    ByteLevelBpe t;
    std::string err;
    build(t, {{g("h"), g("e")}, {g("l"), g("l")}, {g("he"), g("ll")}, {g("hell"), g("o")}, {g("a"), g("a")}, {g(" "), g("w")}}, &err);
    CHECK_MSG(err.empty(), "%s", err.c_str());
    CHECK(t.merge_count() == 6 && t.vocab_size() == 256 + 6 + 2);
    std::vector<int32_t> ids;
    CHECK(t.encode_word("hello", &ids) && ids.size() == 1 && ids[0] == 256 + 3);         // fully merged
    CHECK(t.encode_word("hell", &ids) && ids.size() == 1 && ids[0] == 256 + 2);
    CHECK(t.encode_word("help", &ids) && ids.size() == 3);                               // he, l, p
    CHECK(t.encode_word("aaa", &ids) && ids.size() == 2 && ids[0] == 256 + 4 && ids[1] == (int)'a');  // leftmost first
    CHECK(t.encode_word("", &ids) && ids.empty());
    CHECK(t.encode_word("x", &ids) && ids.size() == 1 && ids[0] == 'x');
    CHECK(t.encode_word(" w", &ids) && ids.size() == 1 && ids[0] == 256 + 5);
    std::mt19937_64 rng(4);
    const char alpha[] = "helo a w";
    for (int it = 0; it < 5000; ++it) {
        std::string w(rng() % 12, ' ');
        for (auto& c : w) c = (rng() % 4 == 0) ? (char)rng() : alpha[rng() % 8];
        CHECK(t.encode_word(w, &ids));
        CHECK(ids.size() <= w.size());                                                   // merges never lengthen
        std::vector<int64_t> wide(ids.begin(), ids.end());
        std::string back;
        CHECK(t.decode(wide, &back) && back == w);                                       // lossless round trip
        for (int32_t id : ids) CHECK(id_in_range(id, t.vocab_size()));
    }
    // decode range checks
    std::string out;
    CHECK(!t.decode({-1}, &out));
    CHECK(!t.decode({(int64_t)t.vocab_size()}, &out));
    CHECK(!t.decode({0, 1, 1000000}, &out));
    CHECK(t.decode({}, &out) && out.empty());
    // special literals
    auto off = t.special_literals(false), on = t.special_literals(true);
    CHECK(off.size() == 1 && off[0].first == "<|user|>");
    CHECK(on.size() == 2);
    // merge over unknown token is rejected at load
    ByteLevelBpe t2;
    CHECK(!t2.load({{"a", kTypeNormal}}, {"a b"}).empty());
    CHECK(!t2.load({{"a", kTypeNormal}, {"b", kTypeNormal}}, {"a b c"}).empty());
    CHECK(t2.load({{"a", kTypeNormal}, {"b", kTypeNormal}, {"ab", kTypeNormal}}, {"a b"}).empty());
}

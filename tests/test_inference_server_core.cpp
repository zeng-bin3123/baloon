#include <cstdio>
#include <string>
#include <vector>

#include "baloon/inference_server_core.hpp"

using namespace baloon::server;

static int g_fail = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            if (g_fail < 50) std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_fail;                                                            \
        }                                                                        \
    } while (0)

static void test_context() {
    for (uint64_t p = 0; p <= 40; ++p)
        for (uint64_t m = 0; m <= 40; ++m)
            for (uint64_t c = 0; c <= 80; ++c) {
                bool f = fits_context(p, m, c);
                CHECK(f == (p + m <= c));          // sound and complete
                if (f && p > 0) CHECK(fits_context(p - 1, m, c));  // shorter prompt still admitted
            }
    CHECK(!fits_context(UINT64_MAX, 1, UINT64_MAX));  // no overflow wrap
    CHECK(!fits_context(1, UINT64_MAX, 10));
}

static void test_finish() {
    for (uint64_t calls : {0ull, 1ull, 5ull}) {
        CHECK(openai_finish(calls, Finish::Length) == OpenFinish::Length);  // length wins even with calls
        CHECK(openai_finish(calls, Finish::Stop) == (calls > 0 ? OpenFinish::ToolCalls : OpenFinish::Stop));
    }
    for (bool u : {false, true}) {
        CHECK(anthropic_stop(u, Finish::Length) == AnthStop::MaxTokens);
        CHECK(anthropic_stop(u, Finish::Stop) == (u ? AnthStop::ToolUse : AnthStop::EndTurn));
        // tool_use iff used tool and stop
        for (Finish f : {Finish::Stop, Finish::Length})
            CHECK((anthropic_stop(u, f) == AnthStop::ToolUse) == (u && f == Finish::Stop));
    }
}

static void test_auth() {
    for (const char* p : {"", "a", "secret", "SECRET"}) CHECK(authorized("", p));  // open access
    CHECK(authorized("secret", "secret"));
    for (const char* p : {"", "secre", "secrets", "SECRET", " secret", "x"}) CHECK(!authorized("secret", p));
}

static void test_pads() {
    const int32_t PAD = 7;
    {
        auto r = expand_pads(PAD, {1, 7, 2, 7}, {3, 2});
        CHECK((r.ids == std::vector<int32_t>{1, 7, 7, 7, 2, 7, 7}) && r.unused.empty());
    }
    // text-only prompt unchanged, counts untouched
    {
        std::vector<int32_t> ids{1, 2, 3, 4, 5};
        auto r = expand_pads(PAD, ids, {9, 9});
        CHECK(r.ids == ids && (r.unused == std::vector<uint32_t>{9, 9}));
        auto r2 = expand_pads(PAD, {}, {});
        CHECK(r2.ids.empty() && r2.unused.empty());
    }
    CHECK((expand_pads(PAD, {PAD}, {4}).ids == std::vector<int32_t>(4, PAD)));
    // pad with no remaining count is kept as is; unused counts reported
    {
        auto r = expand_pads(PAD, {PAD, 1, PAD}, {2});
        CHECK((r.ids == std::vector<int32_t>{7, 7, 1, 7}) && r.unused.empty());
        auto r2 = expand_pads(PAD, {1}, {2, 3});
        CHECK((r2.unused == std::vector<uint32_t>{2, 3}));
    }
    // length law: |out| = |ids| - pads_consumed + sum(consumed counts) (+ unmatched pads kept)
    for (int mask = 0; mask < 256; ++mask) {
        std::vector<int32_t> ids;
        size_t pads = 0;
        for (int i = 0; i < 8; ++i) {
            bool isp = (mask >> i) & 1;
            ids.push_back(isp ? PAD : 100 + i);
            pads += isp;
        }
        for (size_t ncounts = 0; ncounts <= 4; ++ncounts) {
            std::vector<uint32_t> counts;
            for (size_t i = 0; i < ncounts; ++i) counts.push_back(static_cast<uint32_t>(i * 2));  // includes 0
            auto r = expand_pads(PAD, ids, counts);
            size_t used = pads < ncounts ? pads : ncounts;
            size_t expect = ids.size() - used;
            for (size_t i = 0; i < used; ++i) expect += counts[i];
            CHECK(r.ids.size() == expect);
            CHECK(r.unused.size() == ncounts - used);
            // non-pad tokens keep their relative order
            std::vector<int32_t> a, b;
            for (int32_t t : ids) if (t != PAD) a.push_back(t);
            for (int32_t t : r.ids) if (t != PAD) b.push_back(t);
            CHECK(a == b);
        }
    }
}

static void test_cache() {
    std::vector<int> c;
    BoundedCache<int> bc;
    for (int i = 0; i < 300; ++i) {
        c = cache_insert(c, i);
        bc.insert(i);
        CHECK(c.size() <= 64 && bc.size() <= 64);
        CHECK(c.size() == bc.size());
        CHECK(c.back() == i);
        size_t n = c.size();
        // oldest evicted first: the cache holds exactly the most recent n inserts in order
        for (size_t k = 0; k < n; ++k) {
            CHECK(c[k] == i - static_cast<int>(n - 1 - k));
            CHECK(bc.entries()[k] == c[k]);
        }
    }
    CHECK(c.size() == 64 && kVisionCacheCap == 64);
}

// Decode a byte prefix as a lenient UTF-8 decoder: a trailing incomplete sequence becomes U+FFFD.
static std::string decode_prefix(const std::string& bytes, size_t k) {
    std::string p = bytes.substr(0, k);
    size_t i = p.size();
    size_t back = 0;
    while (i > 0 && back < 4 && (static_cast<unsigned char>(p[i - 1]) & 0xC0) == 0x80) { --i; ++back; }
    if (i > 0) {
        unsigned char lead = static_cast<unsigned char>(p[i - 1]);
        size_t need = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
        size_t have = back + 1;
        if (need > 1 && have < need) {
            p.erase(i - 1);
            p += "\xEF\xBF\xBD";
        }
    }
    return p;
}

static void test_detok() {
    CHECK(ends_with_replacement("ab\xEF\xBF\xBD") && !ends_with_replacement("ab") && !ends_with_replacement(""));
    // model-level behaviour
    {
        auto s = detok_push("hello", 2, true);
        CHECK(s.emitted.empty() && s.sent == 2);
        auto t = detok_push("hello", 2, false);
        CHECK(t.emitted == "llo" && t.sent == 5);
        CHECK(std::string("hello").substr(0, 2) + t.emitted == "hello");
    }
    const std::string text = "ab \xC3\xA9 x \xE4\xB8\xAD\xE6\x96\x87 y \xF0\x9F\x98\x80 z";
    Detokenizer d;
    std::string all;
    for (size_t k = 0; k <= text.size(); ++k) {
        std::string dec = decode_prefix(text, k);
        size_t before = d.sent();
        std::string delta = d.push(dec);
        if (ends_with_replacement(dec)) {
            CHECK(delta.empty() && d.sent() == before);  // nothing while partial
        } else {
            CHECK(d.sent() == dec.size());               // everything decoded is sent
        }
        all += delta;
    }
    CHECK(all == text);  // concatenated deltas equal the decoded text
    CHECK(all.find("\xEF\xBF\xBD") == std::string::npos);
    // repeated push of unchanged text emits nothing the second time
    Detokenizer e;
    CHECK(e.push("abc") == "abc" && e.push("abc").empty() && e.push("abcd") == "d");
}

static void test_blocks() {
    using BK = BlockKind;
    CHECK(block_starts(std::nullopt, BK::Text) && block_starts(std::nullopt, BK::Thinking) &&
          block_starts(std::nullopt, BK::ToolUse));
    CHECK(!block_starts(BK::Text, BK::Text) && !block_starts(BK::Thinking, BK::Thinking));
    CHECK(block_starts(BK::Text, BK::Thinking) && block_starts(BK::Thinking, BK::Text));
    for (BK o : {BK::Thinking, BK::Text, BK::ToolUse}) CHECK(block_starts(o, BK::ToolUse));  // always for tool_use
    CHECK(block_starts(std::optional<BK>(BK::ToolUse), BK::Text));
}

int main() {
    test_context();
    test_finish();
    test_auth();
    test_pads();
    test_cache();
    test_detok();
    test_blocks();
    if (g_fail) {
        std::fprintf(stderr, "inference_server_core: %d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("inference_server_core: all checks passed\n");
    return 0;
}


#include <cstdio>
#include <string>
#include <vector>

#include "baloon/chat_frontend_parsing.hpp"

using namespace baloon::frontend;

static int g_fail = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            if (g_fail < 50) std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_fail;                                                            \
        }                                                                        \
    } while (0)

static Kw lvl(Level l) { return Kw{Kw::Kind::Level, l}; }
static Kw nothink() { return Kw{Kw::Kind::NoThink, Level::Low}; }

static void test_effort() {
    for (const char* s : {"none", "off", "minimal", "disabled", "false"}) {
        auto k = effort_kwargs(std::string(s));
        CHECK(k.has_value() && *k == nothink());
    }
    CHECK(*effort_kwargs(std::string("low")) == lvl(Level::Low));
    CHECK(*effort_kwargs(std::string("medium")) == lvl(Level::Medium));
    CHECK(*effort_kwargs(std::string("high")) == lvl(Level::XHigh));
    CHECK(*effort_kwargs(std::string("xhigh")) == lvl(Level::XHigh));
    // max / maximum are aliases of high
    CHECK(effort_kwargs(std::string("max")) == effort_kwargs(std::string("high")));
    CHECK(effort_kwargs(std::string("maximum")) == effort_kwargs(std::string("high")));
    // unknown spellings are rejected, never a crash
    for (const char* s : {"", "HIGH", "ultra", "hi", "0", "true", " low", "low ", "Medium", "xhigh\n"})
        CHECK(!effort_kwargs(std::string(s)).has_value());
    CHECK(!effort_kwargs(EffortKey::Unknown).has_value());
    // accepted exactly when the key is not Unknown
    for (EffortKey k : {EffortKey::None, EffortKey::Off, EffortKey::Minimal, EffortKey::Disabled, EffortKey::False,
                        EffortKey::Low, EffortKey::Medium, EffortKey::High, EffortKey::XHigh, EffortKey::Max,
                        EffortKey::Maximum, EffortKey::Unknown})
        CHECK(effort_kwargs(k).has_value() == (k != EffortKey::Unknown));
}

static void test_budget() {
    CHECK(budget_effort(0) == Level::Low && budget_effort(2047) == Level::Low);
    CHECK(budget_effort(2048) == Level::Medium && budget_effort(8191) == Level::Medium);
    CHECK(budget_effort(8192) == Level::XHigh && budget_effort(1u << 30) == Level::XHigh);
    int prev = 0;
    for (uint64_t n = 0; n < 20000; ++n) {
        int r = level_rank(budget_effort(n));
        CHECK(r >= prev);  // monotone
        prev = r;
    }
}

// Brute-force definition of hold for the test.
static size_t hold_ref(const std::string& text, const std::string& tag) {
    size_t best = 0;
    for (size_t k = 1; k + 1 <= tag.size() && k <= text.size(); ++k)
        if (text.compare(text.size() - k, k, tag, 0, k) == 0) best = k;
    return best;
}

static std::vector<std::string> all_strings(const std::string& alpha, size_t maxlen) {
    std::vector<std::string> out{""};
    size_t start = 0;
    for (size_t len = 1; len <= maxlen; ++len) {
        size_t end = out.size();
        for (size_t i = start; i < end; ++i)
            for (char c : alpha) out.push_back(out[i] + c);
        start = end;
    }
    return out;
}

static void test_hold() {
    CHECK(hold(std::vector<int>{9, 1, 2}, std::vector<int>{1, 2, 3, 4}) == 2);
    CHECK(hold(std::vector<int>{9, 8, 7}, std::vector<int>{1, 2, 3, 4}) == 0);
    CHECK(hold(std::string("abc"), std::string("")) == 0);
    auto texts = all_strings("ab", 8);
    auto tags = all_strings("ab", 5);
    for (const auto& tag : tags)
        for (const auto& text : texts) {
            size_t h = hold(text, tag);
            CHECK(h == hold_ref(text, tag));
            if (!tag.empty()) CHECK(h < tag.size());  // strictly smaller than the tag
            CHECK(h <= text.size());
            if (h > 0) CHECK(text.compare(text.size() - h, h, tag, 0, h) == 0);  // really a prefix
        }
}

static void test_scanner() {
    const std::string tag = "<tool_call>";
    const std::string src = "hello <tool <tool_ca world <tool_call>name=x";
    // one-shot reference
    std::string ref_out;
    {
        TagScanner s(tag);
        bool f = s.feed(src, ref_out);
        CHECK(f);
        CHECK(ref_out == "hello <tool <tool_ca world ");
        CHECK(s.buffer() == "name=x");
    }
    // every 2-way and 3-way chunking gives identical emitted text; never emits a partial tag at end
    for (size_t i = 0; i <= src.size(); ++i)
        for (size_t j = i; j <= src.size(); ++j) {
            TagScanner s(tag);
            std::string out;
            bool found = false;
            for (const std::string& part : {src.substr(0, i), src.substr(i, j - i), src.substr(j)}) {
                if (found) break;
                std::string before = out;
                found = s.feed(part, out);
                if (!found) {
                    // held-back text is a proper prefix of the tag
                    CHECK(s.buffer().size() < tag.size());
                    CHECK(tag.compare(0, s.buffer().size(), s.buffer()) == 0);
                }
            }
            if (found) CHECK(out == ref_out);
            // bytes after the tag may span chunks; the emitted text must match once found
        }
    // stream ending in a partial tag: flush releases it as content
    {
        TagScanner s(tag);
        std::string out;
        CHECK(!s.feed("abc <tool_c", out));
        CHECK(out == "abc ");
        CHECK(s.buffer() == "<tool_c");
        s.flush(out);
        CHECK(out == "abc <tool_c" && s.buffer().empty());
    }
    // no tag at all: everything is emitted, nothing lost, regardless of chunking
    for (size_t i = 0; i <= 12; ++i) {
        std::string text = "plain text!!";
        TagScanner s(tag);
        std::string out;
        s.feed(text.substr(0, i), out);
        s.feed(text.substr(i), out);
        s.flush(out);
        CHECK(out == text);
    }
}

static std::string strip_ref(const std::string& v) {
    std::string r = v;
    if (!r.empty() && r.front() == '\n') r.erase(0, 1);
    if (!r.empty() && r.back() == '\n') r.pop_back();
    return r;
}

static void test_trim() {
    CHECK(trim_value("\nabc\n") == "abc");
    CHECK(trim_value("\n\nabc\n\n") == "\nabc\n");  // only one on each side
    CHECK(trim_value("\n") == "" && trim_value("") == "" && trim_value("\n\n") == "");
    CHECK(trim_value("abc") == "abc");
    for (const auto& v : all_strings("a\n", 9)) {
        std::string t = trim_value(v);
        CHECK(t == strip_ref(v));
        CHECK(t.size() <= v.size());       // never lengthens
        CHECK(v.size() <= t.size() + 2);   // removes at most two characters
    }
}

static void test_finish() {
    CHECK(finish_kind(PState::Reasoning) == EvKind::Reasoning);
    CHECK(finish_kind(PState::Content) == EvKind::Content);
    CHECK(finish_kind(PState::Call) == EvKind::Content);
    for (PState s : {PState::Reasoning, PState::Content, PState::Call}) CHECK(finish_kind(s) != EvKind::ToolCall);
}

int main() {
    test_effort();
    test_budget();
    test_hold();
    test_scanner();
    test_trim();
    test_finish();
    if (g_fail) {
        std::fprintf(stderr, "chat_frontend_parsing: %d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("chat_frontend_parsing: all checks passed\n");
    return 0;
}


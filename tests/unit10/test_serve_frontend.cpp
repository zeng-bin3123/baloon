#include <string>
#include <vector>
#include "baloon/serve_frontend.hpp"
#include "check.hpp"
using namespace baloon::serve_frontend;

void test_sf_effort() {
    for (const char* s : {"none", "off", "minimal", "disabled", "false"}) { auto e = effort_of(s); CHECK(e.known && e.disabled); }
    { auto e = effort_of("low"); CHECK(e.known && !e.disabled && e.level == Level::Low); }
    { auto e = effort_of("medium"); CHECK(e.known && !e.disabled && e.level == Level::Medium); }
    for (const char* s : {"high", "xhigh", "max", "maximum"}) { auto e = effort_of(s); CHECK(e.known && !e.disabled && e.level == Level::XHigh); }
    for (const char* s : {"turbo", "", "HIGH", "mid"}) CHECK(!effort_of(s).known);
}
void test_sf_budget() {
    CHECK(budget_rank(0) == 0); CHECK(budget_rank(2047) == 0); CHECK(budget_rank(2048) == 1);
    CHECK(budget_rank(8191) == 1); CHECK(budget_rank(8192) == 2); CHECK(budget_rank(1u << 30) == 2);
    CHECK(budget_level(100) == Level::Low && budget_level(5000) == Level::Medium && budget_level(9000) == Level::XHigh);
    uint32_t prev = 0;
    for (uint64_t n = 0; n < 20000; ++n) { uint32_t r = budget_rank(n); CHECK(r >= prev); CHECK(r <= 2); prev = r; }
}
void test_sf_hold() {
    const std::string tag = "</think>";
    CHECK(hold("abc", tag) == 0);
    CHECK(hold("abc<", tag) == 1);
    CHECK(hold("abc</thi", tag) == 5);
    CHECK(hold("abc</think", tag) == 7);        // longest proper prefix
    CHECK(hold("abc</think>", tag) == 0);       // complete tag is never held
    CHECK(hold("<<", tag) == 1);
    CHECK(hold("", tag) == 0);
    testh::Rng r(7);
    const char al[] = "<>/thinkx";
    for (int it = 0; it < 20000; ++it) {
        std::string s; for (uint32_t i = r.below(14); i > 0; --i) s += al[r.below(9)];
        size_t h = hold(s, tag);
        CHECK(h < tag.size());
        CHECK(h <= s.size());
        if (h) CHECK(s.substr(s.size() - h) == tag.substr(0, h));
        for (size_t k = h + 1; k < tag.size() && k <= s.size(); ++k) CHECK(s.substr(s.size() - k) != tag.substr(0, k));  // maximal
    }
    std::vector<int> t = {1, 2, 3}, x = {9, 1, 2};
    CHECK(hold(x, t) == 2);
}
void test_sf_parser() {
    CHECK(next(PState::Reasoning, Ev::ThinkEnd) == PState::Content);
    CHECK(next(PState::Content, Ev::CallStart) == PState::Call);
    CHECK(next(PState::Call, Ev::CallEnd) == PState::Content);
    for (PState s : {PState::Content, PState::Call})
        for (Ev e : {Ev::ThinkEnd, Ev::CallStart, Ev::CallEnd, Ev::Other}) CHECK(next(s, e) != PState::Reasoning);
    CHECK(strip_one("\nabc") == "abc"); CHECK(strip_one("\n\nabc") == "\nabc"); CHECK(strip_one("abc") == "abc"); CHECK(strip_one("") == "");

    OutputParser p;
    Delta d = p.feed("think</thi");
    CHECK(d.reasoning == "think" && d.content.empty());  // partial tag held
    d = p.feed("nk>hello <tool_ca");
    CHECK(d.reasoning.empty() && d.content == "hello ");
    CHECK(p.state() == PState::Content);
    d = p.feed("ll>body</tool_call> tail");
    CHECK(d.call == "body" && d.calls_completed == 1 && d.content == " tail");
    CHECK(p.calls() == 1 && p.state() == PState::Content);
    d = p.feed("</think>stay");  // never returns to reasoning
    CHECK(d.content == "</think>stay" && p.state() == PState::Content);
    OutputParser q(false);
    d = q.feed("abc<"); d = q.finish();
    CHECK(d.content == "<");  // flushed at the end
}
void test_sf_parser_random() {
    testh::Rng r(99);
    const std::string full = "reason A</think>hello <tool_call>f(x)</tool_call> mid <tool_call>g</tool_call>end <";
    for (int it = 0; it < 3000; ++it) {
        OutputParser p;
        std::string R, C, K;
        PState prev = PState::Reasoning; bool bad = false;
        size_t i = 0;
        while (i < full.size()) {
            size_t n = 1 + r.below(7); if (i + n > full.size()) n = full.size() - i;
            Delta d = p.feed(full.substr(i, n)); i += n;
            R += d.reasoning; C += d.content; K += d.call;
            if (prev != PState::Reasoning && p.state() == PState::Reasoning) bad = true;
            prev = p.state();
        }
        Delta d = p.finish(); R += d.reasoning; C += d.content; K += d.call;
        CHECK(!bad);
        CHECK(R == "reason A");
        CHECK(C == "hello  mid end <");
        CHECK(K == "f(x)g");
        CHECK(p.calls() == 2);
    }
}

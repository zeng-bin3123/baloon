#include <string>
#include <vector>
#include "baloon/serve_server.hpp"
#include "check.hpp"
using namespace baloon::serve_server;

void test_ss_guard() {
    testh::Rng r(3);
    for (int i = 0; i < 20000; ++i) {
        uint64_t p = r.below(5000), m = r.below(5000), c = r.below(8000);
        bool a = accepts(p, m, c);
        CHECK(a == (p + m <= c));
        if (a) CHECK(p + m <= c); else CHECK(c < p + m);
    }
    CHECK(accepts(10, 0, 10)); CHECK(!accepts(10, 1, 10)); CHECK(accepts(0, 0, 0));
}
void test_ss_detok() {
    testh::Rng r(4);
    uint64_t sent = 0, len = 0;
    for (int i = 0; i < 50000; ++i) {
        len += r.below(5); bool held = r.below(3) == 0;
        uint64_t ns = detok_push(sent, len, held), d = delta_len(sent, len, held);
        CHECK(ns >= sent); CHECK(ns <= len); CHECK(sent + d == ns);
        if (held) CHECK(ns == sent && d == 0); else CHECK(ns == len);
        sent = ns;
    }
}
void test_ss_run() {
    auto stop = [](uint32_t t) { return t == 2; };
    CHECK(run_n(stop, {}) == std::make_pair((uint64_t)0, false));
    CHECK(run_n(stop, {5, 6, 7}) == std::make_pair((uint64_t)3, false));
    CHECK(run_n(stop, {5, 2, 7}) == std::make_pair((uint64_t)2, true));
    CHECK(run_n(stop, {2}) == std::make_pair((uint64_t)1, true));  // the stop token counts
    testh::Rng r(5);
    for (int i = 0; i < 5000; ++i) {
        std::vector<uint32_t> v(r.below(20)); for (auto& x : v) x = r.below(5);
        auto res = run_n(stop, v);
        CHECK(res.first <= v.size());
        if (!v.empty()) CHECK(res.first >= 1);
        // matches the recursive definition of the theory
        uint64_t n = 0; bool s = false; for (size_t k = 0; k < v.size(); ++k) { ++n; if (v[k] == 2) { s = true; break; } }
        CHECK(res.first == n && res.second == s);
    }
}
void test_ss_finish() {
    for (uint64_t c = 1; c < 5; ++c) { CHECK(openai_finish(c, Finish::Stop) == 2); }
    CHECK(openai_finish(0, Finish::Stop) == 0);
    for (uint64_t c = 0; c < 5; ++c) CHECK(openai_finish(c, Finish::Length) == 1);
    CHECK(anthropic_stop(true, Finish::Stop) == 2); CHECK(anthropic_stop(false, Finish::Stop) == 0);
    CHECK(anthropic_stop(true, Finish::Length) == 1); CHECK(anthropic_stop(false, Finish::Length) == 1);
}
void test_ss_cache_auth() {
    size_t s = 0; for (int i = 0; i < 200; ++i) { s = cache_add(s); CHECK(s <= 64); }
    CHECK(s == 64); CHECK(cache_add(63) == 64);
    ImageCache<int> c;
    for (int i = 0; i < 300; ++i) { c.put("k" + std::to_string(i), i); CHECK(c.size() <= 64); }
    CHECK(c.size() == 64 && !c.has("k235") && c.has("k236") && c.has("k299") && *c.get("k299") == 299);
    c.put("k299", 7); CHECK(c.size() == 64 && *c.get("k299") == 7);
    CHECK(pick_max(5, 9) == 5); CHECK(pick_max(0, 9) == 9); CHECK(pick_max(0, 0) == 1024);
    for (uint64_t a = 0; a < 4; ++a) for (uint64_t b = 0; b < 4; ++b) CHECK(pick_max(a, b) > 0);
    CHECK(authorized("", "")); CHECK(authorized("", "anything"));
    CHECK(authorized("k", "k")); CHECK(!authorized("k", "")); CHECK(!authorized("k", "x"));
}

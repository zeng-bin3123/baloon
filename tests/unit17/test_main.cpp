#include <algorithm>
#include <cstdio>
#include "check.hpp"
#include "baloon/chat_frontend.hpp"
#include "baloon/prefill_session.hpp"
#include "baloon/serve_frontend.hpp"
using namespace baloon;
using V = std::vector<uint64_t>;

static void test_budget() {
    for (int64_t n = -5000; n < 20000; ++n) {
        const uint32_t r = chat_frontend::budget_rank(n);
        CHECK(r < 3);
        CHECK(chat_frontend::budget_rank(n) <= chat_frontend::budget_rank(n + 1));
        if (n >= 0) CHECK(r == serve_frontend::budget_rank((uint64_t)n));
    }
    CHECK(chat_frontend::budget_rank(2047) == 0 && chat_frontend::budget_rank(2048) == 1);
    CHECK(chat_frontend::budget_rank(8191) == 1 && chat_frontend::budget_rank(8192) == 2);
}
static void test_hold() {
    testh::Rng rng(7);
    for (int it = 0; it < 20000; ++it) {
        V tag(1 + rng.below(6)), text(rng.below(14));
        for (auto& x : tag) x = rng.below(3);
        for (auto& x : text) x = rng.below(3);
        const size_t h = chat_frontend::hold_one(text, tag);
        CHECK(h < tag.size());                       // proper prefix
        CHECK(h <= text.size());
        if (h) CHECK(std::equal(text.end() - (long)h, text.end(), tag.begin()));
        for (size_t n = h + 1; n < tag.size() && n <= text.size(); ++n)   // longest
            CHECK(!std::equal(text.end() - (long)n, text.end(), tag.begin()));
        CHECK(h == serve_frontend::hold(text, tag));  // agrees with the incremental parser's hold-back
        const size_t e = chat_frontend::emitted_len(text.size(), h);
        CHECK(e + h == text.size());                  // held + emitted == buffer
    }
    CHECK(chat_frontend::hold_one({}, {1, 2}) == 0);
    CHECK(chat_frontend::hold_one({5, 1}, {1, 2}) == 1);
    CHECK(chat_frontend::hold_one({1, 2}, {1, 2}) == 0);  // a full tag is not a proper prefix
}
static void test_strip() {
    CHECK((chat_frontend::strip_nl({10, 1, 2, 10}) == V{1, 2}));
    CHECK((chat_frontend::strip_nl({10, 10, 1, 10, 10}) == V{10, 1, 10}));
    CHECK((chat_frontend::strip_nl({10}) == V{}));
    CHECK((chat_frontend::strip_nl({}) == V{}));
    CHECK((chat_frontend::strip_nl({1, 2}) == V{1, 2}));
    testh::Rng rng(3);
    for (int it = 0; it < 5000; ++it) {
        V l(rng.below(8)); for (auto& x : l) x = rng.below(12);
        const V s = chat_frontend::strip_nl(l);
        CHECK(s.size() <= l.size() && l.size() <= s.size() + 2);
        CHECK(chat_frontend::drop_trail(l).size() <= l.size() && l.size() <= chat_frontend::drop_trail(l).size() + 1);
        CHECK(chat_frontend::drop_lead(l).size() + 1 >= l.size());
    }
}
static void test_expand() {
    const uint64_t pad = 99;
    testh::Rng rng(11);
    for (int it = 0; it < 5000; ++it) {
        V ids(rng.below(12)); for (auto& x : ids) x = rng.below(4) == 0 ? pad : rng.below(50);
        V cs(rng.below(5)); for (auto& x : cs) x = 1 + rng.below(6);
        CHECK(chat_frontend::expand(pad, ids, {}) == ids);                 // no images: unchanged
        V nopad; for (auto x : ids) if (x != pad) nopad.push_back(x);
        CHECK(chat_frontend::expand(pad, nopad, cs) == nopad);             // no pads: unchanged
        const V out = chat_frontend::expand(pad, ids, cs);
        size_t np = 0; for (auto x : ids) np += x == pad;
        size_t want = ids.size() - np, k = 0;
        for (size_t i = 0; i < np; ++i) want += i < cs.size() ? cs[k++] : 1;
        CHECK(out.size() == want);
        V a, b; for (auto x : out) if (x != pad) a.push_back(x); for (auto x : ids) if (x != pad) b.push_back(x);
        CHECK(a == b);                                                     // non-pad tokens keep order
    }
    CHECK((chat_frontend::expand(9, {1, 9, 2, 9}, {3, 2}) == V{1, 9, 9, 9, 2, 9, 9}));
}
static void test_fits_delta() {
    for (uint64_t p = 0; p < 40; ++p) for (uint64_t m = 0; m < 40; ++m) for (uint64_t c = 0; c < 80; c += 3) {
        const bool f = chat_frontend::fits(p, m, c);
        CHECK(f == (p + m <= c));
        if (f) CHECK(p <= c);
    }
    for (uint64_t t = 0; t < 50; ++t) for (uint64_t s = 0; s <= t; ++s) CHECK(s + chat_frontend::delta_len(t, s) == t);
}
static void test_prefill() {
    for (uint64_t chunk = 0; chunk < 9; ++chunk) for (uint64_t n = 0; n < 40; ++n) {
        std::vector<int> seen(n, 0); uint64_t next = 0; bool ordered = true, bounded = true;
        prefill_session::Prefill p(chunk);
        const auto st = p.run(n, [&](uint64_t s, uint64_t l) {
            if (s != next) ordered = false;
            if (l == 0 || l > p.chunk()) bounded = false;
            for (uint64_t i = s; i < s + l && i < n; ++i) ++seen[i];
            next = s + l;
        });
        CHECK(ordered && bounded);
        CHECK(std::all_of(seen.begin(), seen.end(), [](int x) { return x == 1; }));
        CHECK(st.tokens == n);
        CHECK(st.chunks == (n + p.chunk() - 1) / p.chunk());
        CHECK(st.max_chunk <= p.chunk());
    }
}
void run_test(const char* name, void (*fn)()) {
    const long before = testh::failures(); fn();
    std::printf("[%s] %s\n", testh::failures() == before ? " ok " : "FAIL", name); std::fflush(stdout);
}
int main() {
    run_test("test_budget", test_budget); run_test("test_hold", test_hold); run_test("test_strip", test_strip);
    run_test("test_expand", test_expand); run_test("test_fits_delta", test_fits_delta); run_test("test_prefill", test_prefill);
    std::printf("checks=%ld failures=%ld\n", testh::checks(), testh::failures());
    if (testh::failures()) { std::printf("UNIT17 TESTS FAILED\n"); return 1; }
    std::printf("UNIT17 TESTS PASSED\n"); return 0;
}

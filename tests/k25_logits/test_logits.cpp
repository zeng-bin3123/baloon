#include <cstdio>
#include <cstdint>
#include <string>
#include "baloon/logits_selection.hpp"
using namespace baloon::logits;
static long g_checks = 0, g_fail = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); } } while (0)

static uint64_t P(const char* s) { uint64_t v = 0; return parse_stride(s, &v) ? v : 0; }
static void test_parse() {
    CHECK(P("1") == 1); CHECK(P("7") == 7); CHECK(P("0010") == 10); CHECK(P("4294967295") == 4294967295ull);
    uint64_t v = 99;
    const char* bad[] = {"", "0", "00", "-1", "+1", " 1", "1 ", "1.5", "abc", "1x", "4294967296", "99999999999999999999999"};
    for (const char* b : bad) { v = 99; CHECK(!parse_stride(b, &v)); CHECK(v == 99); }
    CHECK(!parse_stride("5", nullptr));
}
static void test_selection() {
    for (uint64_t s = 1; s <= 40; ++s)
        for (uint64_t total = 0; total <= 200; ++total) {
            uint64_t cnt = 0; bool prev_ok = true;
            for (uint64_t r = 0; r < total; ++r) {
                const bool sel = selected(r, s);
                cnt += sel;
                CHECK(sel == (r % s == 0));
                (void)prev_ok;
            }
            CHECK(row_count(total, s) == cnt);                      // count agrees with the predicate
            if (total > 0) CHECK(selected(0, s));                   // first row always kept
            if (s == 1) CHECK(row_count(total, s) == total);        // stride 1 keeps all
            CHECK(row_count(total, s) <= total);
            CHECK(row_count(total, s) * s >= total);                // ceil: s * count covers total
            if (total > 0) CHECK((row_count(total, s) - 1) * s < total);
        }
    // consecutive selected rows are exactly `stride` apart
    uint64_t last = 0; bool first = true;
    for (uint64_t r = 0; r < 1000; ++r) if (selected(r, 13)) { if (!first) CHECK(r - last == 13); last = r; first = false; }
}
static void test_invalid() {
    CHECK(row_count(10, 0) == 0); CHECK(!selected(0, 0)); CHECK(!selected(5, 0));
    CHECK(row_count(UINT64_MAX, 1) == UINT64_MAX);
    CHECK(row_count(UINT64_MAX, UINT64_MAX) == 1);
    CHECK(row_count(UINT64_MAX, 2) == (UINT64_MAX / 2) + 1);       // no overflow from total + stride - 1
    CHECK(selected(UINT64_MAX - 1, 2) && !selected(UINT64_MAX, 2));
}
int main() {
    test_parse(); test_selection(); test_invalid();
    std::printf("%ld checks, %ld failed\n", g_checks, g_fail);
    if (g_fail) return 1;
    std::printf("UNIT 25 TESTS PASSED\n");
    return 0;
}

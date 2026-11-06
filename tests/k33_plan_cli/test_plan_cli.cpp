// Unit 33 invariants of the plan CLI: strict u64 parsing (no wrap, no junk), exit-code contract (0 closes / 1 refuses / 2
// usage), output equals the library's printable form, --device/--reserve equals vram_pool_bytes, and the real binary
// (path in argv[1]) behaves the same as run().
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <sys/wait.h>
#include "baloon/plan_cli.hpp"

using namespace baloon::plan;
static long g_checks = 0, g_fails = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fails; std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)

static bool P(const char* s, uint64_t& v) { return cli::parse_u64(s, v); }

static int capture(const std::vector<std::string>& a, std::string& o, std::string& e) {
    std::FILE* fo = std::tmpfile(); std::FILE* fe = std::tmpfile();
    const int rc = cli::run(a, fo, fe);
    auto slurp = [](std::FILE* f, std::string& s) { s.clear(); std::rewind(f); char b[512]; size_t n; while ((n = std::fread(b, 1, sizeof b, f)) > 0) s.append(b, n); std::fclose(f); };
    slurp(fo, o); slurp(fe, e);
    return rc;
}

static void test_parse() {
    uint64_t v = 7;
    CHECK(P("0", v) && v == 0); CHECK(P("123", v) && v == 123); CHECK(P("18446744073709551615", v) && v == UINT64_MAX);
    CHECK(!P("18446744073709551616", v));                       // 2^64 does not wrap to 0
    CHECK(!P("99999999999999999999999999", v));
    CHECK(!P("340282366920938463463374607431768211456", v));    // 2^128: must not wrap the wide accumulator to 0
    CHECK(!P("340282366920938463463374607431768211461", v));    // 2^128 + 5
    CHECK(!P("100000000000000000000000000000000000000000000000000", v));
    CHECK(P("1k", v) && v == 1024); CHECK(P("3M", v) && v == 3ull << 20); CHECK(P("32g", v) && v == 32ull << 30);
    CHECK(P("17179869183g", v) && v == 17179869183ull << 30);   // largest g count that fits
    CHECK(!P("17179869184g", v));                               // suffix shift overflow refused
    CHECK(!P("18014398509481984k", v));                         // 2^54 k = 2^64
    CHECK(P("18014398509481983k", v));
    CHECK(!P("", v)); CHECK(!P("-1", v)); CHECK(!P("+1", v)); CHECK(!P("1x", v)); CHECK(!P("1kk", v)); CHECK(!P("k", v));
    CHECK(!P(" 1", v)); CHECK(!P("1 ", v)); CHECK(!P("0x10", v)); CHECK(!P("1.5", v));
    // randomized oracle: decimal round trip
    uint64_t x = 88172645463325252ull;
    for (int i = 0; i < 2000; ++i) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; CHECK(P(std::to_string(x).c_str(), v) && v == x); }
}

static void test_run_matches_library() {
    Geometry g; Costs c;
    c.dense = 1000; c.embd = 500; c.workspace = 64;
    const uint64_t pool = 4ull << 30;
    std::string o, e;
    int rc = capture({"1024", "4g", "dense=1000", "embd=500", "workspace=64"}, o, e);
    const Result r = make_plan(1024, g, c, pool);
    CHECK(std::holds_alternative<Plan>(r));
    CHECK(rc == 0); CHECK(e.empty()); CHECK(o == to_string(r) + "\n");
    // refusal: pool too small for the fixed costs
    rc = capture({"1024", "10", "dense=1000"}, o, e);
    CHECK(rc == 1); CHECK(o.empty()); CHECK(e.find("does not close") != std::string::npos);
    Costs c2; c2.dense = 1000;
    CHECK(e == to_string(make_plan(1024, g, c2, 10)) + "\n");
    rc = capture({"1", "1g", "expert_blob=0"}, o, e);          // zero blob refuses
    CHECK(rc == 1 && e.find("zero") != std::string::npos);
    rc = capture({"1", "1g", "expert_blob=2g"}, o, e);         // blob larger than the pool: no whole slot
    CHECK(rc == 1 && e.find("no whole expert slot") != std::string::npos);
    // unrepresentable cost refuses instead of wrapping
    rc = capture({"18446744073709551615", "1g"}, o, e);
    CHECK(rc == 1);
}

static void test_device_reserve() {
    std::string o, e, o2, e2;
    CHECK(capture({"2048", "--device", "16g", "--reserve", "1g"}, o, e) == 0);
    CHECK(capture({"2048", "15g"}, o2, e2) == 0);
    CHECK(o == o2);
    CHECK(capture({"2048", "--device", "1g", "--reserve", "2g"}, o, e) == 1);   // reserve > device: pool 0, truncating
    CHECK(e.find("pool=0 ") != std::string::npos);
    CHECK(capture({"2048", "--device", "16g"}, o, e) == 0);                     // reserve defaults to 0
    CHECK(capture({"2048", "16g", "--device", "16g"}, o, e) == 2);              // conflicting pool sources
}

static void test_usage_errors() {
    std::string o, e;
    const std::vector<std::vector<std::string>> bad = {
        {}, {"1024"}, {"1024", "1g", "extra"}, {"abc", "1g"}, {"1024", "-5"}, {"1024", "1g", "bogus=1"},
        {"1024", "1g", "n_layers=x"}, {"1024", "1g", "--nope"}, {"1024", "--device"}, {"1024", "--device", "zz"},
        {"1024", "1g", "dense=18446744073709551616"}, {"1024", "--reserve", "1g"},
    };
    for (const auto& a : bad) { CHECK(capture(a, o, e) == 2); CHECK(o.empty()); CHECK(e.find("usage") != std::string::npos); }
}

static void test_fields_reach_planner() {
    // 1. every key lands in exactly its own field (distinct value per key; no other field changes)
    const char* names[] = {"n_layers", "n_qsa", "n_kv_heads", "head_dim", "kv_group", "qsa_block", "idx_key_heads", "idx_key_dim",
                           "ssm_state", "ssm_v_heads", "ssm_conv", "ssm_channels", "expert_blob", "dense", "embd", "mtp",
                           "workspace", "state"};
    auto snap = [](const cli::Config& c) {
        return std::vector<uint64_t>{c.g.n_layers, c.g.n_qsa, c.g.n_kv_heads, c.g.head_dim, c.g.kv_group, c.g.qsa_block,
                                     c.g.idx_key_heads, c.g.idx_key_dim, c.g.ssm_state, c.g.ssm_v_heads, c.g.ssm_conv,
                                     c.g.ssm_channels, c.c.expert_blob, c.c.dense, c.c.embd, c.c.mtp, c.c.workspace, c.c.state};
    };
    cli::Config def; const std::vector<uint64_t> base = snap(def);
    for (size_t i = 0; i < 18; ++i) {
        cli::Config c; std::string err;
        const uint64_t val = 1000003 + 7 * i;
        CHECK(cli::parse_args({"1", "1g", std::string(names[i]) + "=" + std::to_string(val)}, c, err));
        const std::vector<uint64_t> got = snap(c);
        for (size_t j = 0; j < 18; ++j) CHECK(got[j] == (j == i ? val : base[j]));
    }
    // 2. keys that feed the printed plan change the printed plan
    std::string base_o, e, o;
    CHECK(capture({"4096", "8g"}, base_o, e) == 0);
    const char* keys[] = {"n_qsa=20", "n_kv_heads=4", "head_dim=512", "kv_group=32", "qsa_block=8", "idx_key_dim=256",
                          "idx_key_heads=2", "dense=1m", "embd=1m", "mtp=1m", "workspace=1m", "state=1m", "expert_blob=5m"};
    for (const char* k : keys) { CHECK(capture({"4096", "8g", k}, o, e) == 0); CHECK(o != base_o); }
    // 3. state_bytes of the parsed geometry equals the library's value for the same numbers
    cli::Config c; std::string err;
    CHECK(cli::parse_args({"1", "1g", "n_layers=10", "n_qsa=4", "ssm_state=16", "ssm_v_heads=3", "ssm_conv=5", "ssm_channels=7"}, c, err));
    CHECK(state_bytes(c.g) == 6ull * (16 * 3 * 16 * 4 + 4 * 7 * 4));
}

static int run_bin(const std::string& bin, const std::string& args, std::string& out) {
    out.clear();
    std::FILE* p = popen((bin + " " + args + " 2>&1").c_str(), "r");
    if (!p) return -1;
    char b[512]; size_t n;
    while ((n = std::fread(b, 1, sizeof b, p)) > 0) out.append(b, n);
    const int st = pclose(p);
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

static void test_binary(const std::string& bin) {
    std::string out, lib_o, lib_e;
    int rc = run_bin(bin, "1024 4g dense=1000", out);
    CHECK(rc == capture({"1024", "4g", "dense=1000"}, lib_o, lib_e)); CHECK(rc == 0); CHECK(out == lib_o);
    rc = run_bin(bin, "1024 10 dense=1000", out);
    CHECK(rc == 1); CHECK(out.find("does not close") != std::string::npos);
    rc = run_bin(bin, "", out);
    CHECK(rc == 2); CHECK(out.find("usage") != std::string::npos);
    rc = run_bin(bin, "1024 18446744073709551616", out);
    CHECK(rc == 2);
}

int main(int argc, char** argv) {
    test_parse(); test_run_matches_library(); test_device_reserve(); test_usage_errors(); test_fields_reach_planner();
    if (argc > 1) test_binary(argv[1]); else { ++g_fails; std::printf("FAIL: binary path (argv[1]) missing\n"); }
    std::printf("checks: %ld\n", g_checks);
    if (g_fails) { std::printf("UNIT 33 TESTS FAILED (%ld)\n", g_fails); return 1; }
    std::printf("UNIT 33 TESTS PASSED\n");
    return 0;
}

// suffix_drafter_test: legacy invariant body (unchanged, entry renamed) followed by the unit-34 integration layer.
// suffix_drafter_test: suffix-drafter invariants (unit 9 + unit 13 test bodies).
#include "../k39_integration/run_harness.hpp"
void test_sd_basic(); void test_sd_random(); void test_sd_table();
void test_sd_theory_table(); void test_sd_theory_match_and_propose(); void test_sd_theory_ways();
void test_sd_ways_and_keys(); void test_sd_probe_mask(); void test_sd_match_loop(); void test_sd_clamps(); void test_sd_continuation(); void test_sd_end_to_end();
void test_port_mix(); void test_port_find_slot_and_reset();
static int legacy_main() {
    RUN(test_sd_basic); RUN(test_sd_random); RUN(test_sd_table);
    RUN(test_sd_theory_table); RUN(test_sd_theory_match_and_propose); RUN(test_sd_theory_ways);
    RUN(test_sd_ways_and_keys); RUN(test_sd_probe_mask); RUN(test_sd_match_loop); RUN(test_sd_clamps); RUN(test_sd_continuation); RUN(test_sd_end_to_end);
    RUN(test_port_mix); RUN(test_port_find_slot_and_reset);
    return finish("suffix_drafter_test");
}

// ============================================================================================================
// Unit-34 integration layer. Everything below is host-side reference code and invariant checks for this
// executable; the legacy body above keeps running unchanged and its verdict is combined with this one.
// ============================================================================================================
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <numeric>
#define IG_UNUSED [[maybe_unused]]
IG_UNUSED static long ig_checks = 0;
IG_UNUSED static long ig_failures = 0;
IG_UNUSED static void ig_check(bool ok, const char* what) { ++ig_checks; if (!ok) { ++ig_failures; std::printf("  invariant violated: %s\n", what); } }
IG_UNUSED static double ig_rel_l1(const float* a, const float* b, size_t n) {
    double num = 0, den = 0; for (size_t i = 0; i < n; ++i) { num += std::fabs((double)a[i] - (double)b[i]); den += std::fabs((double)b[i]); }
    return den > 0 ? num / den : num;
}

// The legacy body runs unchanged with its output untouched (an external launch tracer may read this process's stdout).
template <class F> IG_UNUSED static int ig_legacy(F f) { return f(); }
IG_UNUSED static int ig_finish(const char* tag, int legacy_rc) {
    std::printf("%s: port-layer checks=%ld violations=%ld legacy_rc=%d\n", tag, ig_checks, ig_failures, legacy_rc);
    if (ig_checks == 0 || ig_failures != 0) { std::printf("%s: port-layer invariants NOT satisfied\n", tag); return 1; }
    if (legacy_rc != 0) return legacy_rc;
    std::printf("%s: port layer PASS\n", tag); return 0;
}
#include "baloon/suffix_drafter.hpp"
struct IgSim { uint64_t proposed = 0, accepted = 0, rounds = 0; };
// Whitespace-separated token ids from a text file.
static std::vector<uint32_t> ig_read_ids(const std::string& path) { std::ifstream f(path); std::vector<uint32_t> v; unsigned long x; while (f >> x) v.push_back((uint32_t)x); return v; }
// Draft-and-verify replay: propose up to k tokens, accept the matching prefix, commit accepted+1 true tokens.
static IgSim ig_simulate(const std::vector<uint32_t>& ids, uint64_t k) {
    baloon::suffix::SuffixDrafter dr; IgSim s; size_t pos = 0;
    for (; pos < 8 && pos < ids.size(); ++pos) dr.append(ids[pos]);
    while (pos < ids.size()) {
        const std::vector<uint32_t> p = dr.propose(k); size_t acc = 0;
        while (acc < p.size() && pos + acc < ids.size() && p[acc] == ids[pos + acc]) ++acc;
        s.proposed += p.size(); s.accepted += acc; ++s.rounds;
        for (size_t i = 0; i < acc + 1 && pos < ids.size(); ++i) dr.append(ids[pos++]);
    }
    return s;
}
static void ig_unit_tests() {
    baloon::suffix::SuffixDrafter d;
    ig_check(d.size() == 0 && d.propose(4).empty(), "an empty history proposes nothing");
    d.append(1); d.append(2); ig_check(d.propose(4).empty() && d.last_match_len() == 0, "fewer than three tokens never match");
    for (int r = 0; r < 4; ++r) for (uint32_t t : {10u, 11u, 12u, 13u, 14u}) d.append(t);
    ig_check(d.size() == 22, "history length");
    d.append(10); d.append(11); d.append(12);
    const auto p = d.propose(2);
    ig_check(p.size() == 2 && p[0] == 13 && p[1] == 14, "the continuation of the earlier occurrence is proposed");
    ig_check(d.last_match_len() >= 3, "match length reaches the minimum");
    d.reset(); ig_check(d.size() == 0 && d.propose(4).empty(), "reset forgets the history");
    ig_check(d.min_match() >= 3 && d.max_match() >= d.min_match(), "match limits are consistent");
}
static void ig_run() {
    ig_unit_tests();
    std::vector<uint32_t> rep; for (int r = 0; r < 60; ++r) for (uint32_t t : {5u, 9u, 2u, 7u, 3u, 8u}) rep.push_back(t);
    const IgSim a = ig_simulate(rep, 4);
    ig_check(a.accepted <= a.proposed && a.proposed > 0, "accepted never exceeds proposed");
    ig_check((double)a.accepted / (double)a.proposed > 0.9, "a periodic stream is drafted almost perfectly");
    ig_check(a.rounds < rep.size() / 2, "drafting commits several tokens per round on a periodic stream");
    std::mt19937 rng(4); std::vector<uint32_t> rnd(600); for (auto& t : rnd) t = rng() % 5000;
    const IgSim b = ig_simulate(rnd, 4);
    ig_check(b.accepted <= b.proposed && (b.proposed == 0 || (double)b.accepted / (double)b.proposed < 0.5), "a random stream is not drafted well");
    const std::string tmp = "/tmp/ig_ids_" + std::to_string((long)rep.size());
    { std::ofstream f(tmp); f << "1 2 3\n40000 5\n"; }
    const auto ids = ig_read_ids(tmp); ig_check(ids.size() == 5 && ids[3] == 40000, "id file parsing"); std::remove(tmp.c_str());
}
int main() { int rc = ig_legacy([&] { return legacy_main(); }); ig_run(); return ig_finish("suffix_drafter_test", rc); }

// controller_test: legacy invariant body (unchanged, entry renamed) followed by the unit-34 integration layer.
// controller_test: speculative-decoding controller invariants (unit 9 + unit 13 test bodies).
#include "../k39_integration/run_harness.hpp"
void test_sc_indexing(); void test_sc_expected(); void test_sc_choose_learn();
void test_sc_theory_tables_and_indexing(); void test_sc_theory_observe_and_commit(); void test_sc_theory_gain_threshold();
void test_pm_clamp_and_buckets(); void test_pm_expected(); void test_pm_best_gate(); void test_pm_observe_and_clip(); void test_pm_controller();
void test_port_expected(); void test_port_choose_argmax();
static int legacy_main() {
    RUN(test_sc_indexing); RUN(test_sc_expected); RUN(test_sc_choose_learn);
    RUN(test_sc_theory_tables_and_indexing); RUN(test_sc_theory_observe_and_commit); RUN(test_sc_theory_gain_threshold);
    RUN(test_pm_clamp_and_buckets); RUN(test_pm_expected); RUN(test_pm_best_gate); RUN(test_pm_observe_and_clip); RUN(test_pm_controller);
    RUN(test_port_expected); RUN(test_port_choose_argmax);
    return finish("controller_test");
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
#include "baloon/spec_controller.hpp"
static const char* ig_name(baloon::spec::Drafter d) { return d == baloon::spec::Drafter::None ? "none" : d == baloon::spec::Drafter::Mtp ? "mtp" : "lookup"; }
static void ig_run() {
    namespace sp = baloon::spec;
    ig_check(!std::strcmp(ig_name(sp::Drafter::None), "none") && !std::strcmp(ig_name(sp::Drafter::Mtp), "mtp") && !std::strcmp(ig_name(sp::Drafter::Lookup), "lookup"), "drafter names");
    const double ones[3] = {1, 1, 1}, halves[3] = {0.5, 0.5, 0.5};
    ig_check(sp::expected(ones, 3) == 4.0, "certain acceptance commits k+1 tokens");
    ig_check(std::fabs(sp::expected(halves, 2) - 1.75) < 1e-12, "expected tokens = 1 + running products");
    ig_check(sp::committed(5, 3) == 4 && sp::committed(0, 3) == 1 && sp::seen(3, 3) == 3, "committed / seen counts");
    ig_check(sp::step_idx(0) == 0 && sp::step_idx(100) == sp::K_MAX && sp::bucket(4) == 0 && sp::bucket(5) == 1 && sp::bucket(8) == 2 && sp::bucket(16) == 3, "indexing and buckets");
    sp::Controller c;
    ig_check(c.expected(sp::Source::None, 4) == 1.0 && c.expected(sp::Source::Mtp, 0) == 1.0, "no draft commits exactly one token");
    double prev = 1.0; bool mono = true; for (uint64_t k = 1; k <= sp::K_MAX; ++k) { const double e = c.expected(sp::Source::Mtp, k); if (e < prev) mono = false; prev = e; }
    ig_check(mono, "expected tokens never decrease with a longer MTP chain");
    ig_check(c.choose(0, 0, 0).drafter == sp::Drafter::None, "nothing available means plain decoding");
    const double before = c.mtp_p[0];
    for (int i = 0; i < 50; ++i) c.observe(sp::Drafter::Mtp, 4, 4);
    ig_check(c.mtp_p[0] > before && c.mtp_p[0] <= 1.0, "full acceptance raises the MTP estimate toward 1");
    sp::Controller c2; for (int i = 0; i < 50; ++i) c2.observe(sp::Drafter::Mtp, 4, 0);
    ig_check(c2.mtp_p[0] < before, "total rejection lowers the MTP estimate");
    ig_check(sp::keep_draft(106, 100) && !sp::keep_draft(104, 100), "a draft must beat plain decoding by the margin");
}
int main() { int rc = ig_legacy([&] { return legacy_main(); }); ig_run(); return ig_finish("controller_test", rc); }

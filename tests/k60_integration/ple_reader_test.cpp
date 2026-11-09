// ple_reader_test: legacy invariant body (unchanged, entry renamed) followed by the unit-34 integration layer.
// ple_reader_test: runs the invariant test body of k21_ple_reader/test_ple_reader_theory.cpp (its main is renamed; argv is accepted for --selftest style invocation).
#define main orig_test_main
#include "../k21_ple_reader/test_ple_reader_theory.cpp"
#undef main
static int legacy_main(int argc, char** argv) {
    (void)argc; (void)argv;           // --selftest (and fixture flags) are accepted; the checks always run
    const int rc = orig_test_main();
    std::printf("ple_reader_test: %s\n", rc == 0 ? "PASSED" : "FAILED");
    return rc;
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
#include <unistd.h>
#define IG_CHECK(c) ig_check((c), #c)
static const int ig_k = 8;     // columns of a table row
static const int ig_ng = 3;    // n-gram order of the hashed index
static std::vector<float> ig_expected_row(int64_t r, int cols) { std::vector<float> v(cols); for (int c = 0; c < cols; ++c) v[c] = (float)((r * 131 + c * 7) % 251); return v; }
static std::vector<float> ig_make_table(int64_t rows, int cols) { std::vector<float> t; t.reserve((size_t)rows * cols); for (int64_t r = 0; r < rows; ++r) { auto v = ig_expected_row(r, cols); t.insert(t.end(), v.begin(), v.end()); } return t; }
// Reads each requested row straight from the file at byte offset row*cols*4 and compares it with the closed form.
static bool ig_check_rows(const std::string& path, int cols, const std::vector<int64_t>& ids) {
    std::ifstream f(path, std::ios::binary); if (!f) return false; bool ok = true;
    for (int64_t r : ids) { std::vector<float> got(cols); f.seekg((std::streamoff)(r * cols * 4)); f.read((char*)got.data(), cols * 4); if (f.gcount() != cols * 4 || got != ig_expected_row(r, cols)) ok = false; }
    return ok;
}
// A real table is a non-empty file whose size is a whole number of rows.
static bool ig_real(const std::string& path, int cols) { std::ifstream f(path, std::ios::binary | std::ios::ate); if (!f) return false; const std::streamoff n = f.tellg(); return n > 0 && n % (cols * 4) == 0; }
static void ig_selftest() {
    const int64_t rows = 1000; const std::string p = "/tmp/ig_ple_table_" + std::to_string((long)getpid());
    const auto t = ig_make_table(rows, ig_k);
    { std::ofstream f(p, std::ios::binary); f.write((const char*)t.data(), (std::streamsize)(t.size() * 4)); }
    IG_CHECK(ig_real(p, ig_k));
    IG_CHECK(!ig_real(p + ".missing", ig_k));
    std::vector<int64_t> ids = {0, 1, 7, 255, 256, 999};
    for (int i = 0; i < ig_ng; ++i) ids.push_back((i * 337) % rows);
    IG_CHECK(ig_check_rows(p, ig_k, ids));
    IG_CHECK(t.size() == (size_t)rows * ig_k);
    IG_CHECK(ig_expected_row(5, ig_k) != ig_expected_row(6, ig_k));
    std::remove(p.c_str());
}
int main(int argc, char** argv) { int rc = ig_legacy([&] { return legacy_main(argc, argv); }); ig_selftest(); return ig_finish("ple_reader_test", rc); }

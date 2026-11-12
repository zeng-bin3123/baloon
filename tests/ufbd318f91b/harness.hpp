// harness.hpp - section registry shared by the unit ufbd318f91b CPU and GPU suites. Each theory statement adds one
// UF_SECTION; main() runs them in registration order and counts CHECK failures.
#pragma once
#include <cstdio>
#include <vector>

namespace uf {
struct Section { const char* name; const char* claim; void (*fn)(); };
inline std::vector<Section>& sections() { static std::vector<Section> v; return v; }
inline int& failures() { static int n = 0; return n; }
struct Reg { Reg(const char* n, const char* c, void (*f)()) { sections().push_back({n, c, f}); } };
inline int run_all(const char* banner) {
    for (const Section& s : sections()) {
        const int before = failures();
        s.fn();
        std::printf("[%s] %s - \"%s\"\n", failures() == before ? " ok " : "FAIL", s.name, s.claim);
    }
    if (failures()) { std::printf("%d check(s) failed\n", failures()); return 1; }
    std::printf("%s (%zu sections)\n", banner, sections().size());
    return 0;
}
}  // namespace uf

#define UF_SECTION(id, claim)                         \
    static void id();                                 \
    static uf::Reg id##_reg(#id, claim, &id);         \
    static void id()
#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) { std::printf("  check failed %s:%d: %s\n", __FILE__, __LINE__, #cond); ++uf::failures(); } \
    } while (0)

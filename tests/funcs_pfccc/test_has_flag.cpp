// has_flag(flags, f): membership check in the CPU flag set (used by cpu_info and _cpuid_avx512_full).
// Ref: setup.py _has_flag()
#include <cassert>
#include <set>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    std::set<std::string> flags = {"avx2", "fma", "avx512f"};
    assert(baloon::setup::has_flag(flags, "avx2"));
    assert(baloon::setup::has_flag(flags, "fma"));
    assert(!baloon::setup::has_flag(flags, "avx512bw"));
    // "avx" prefix does NOT match "avx2"
    assert(!baloon::setup::has_flag(flags, "avx"));
    // empty set
    assert(!baloon::setup::has_flag({}, "avx2"));
    return 0;
}

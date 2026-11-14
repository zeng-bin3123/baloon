// _cpuid_avx512_full(flags): true only when all 8 required flags are present.
// Required: avx512f, avx512bw, avx512dq, avx512vl, avx512_vbmi, avx512_vnni, f16c, fma.
// Ref: setup.py _cpuid_avx512_full(flags)
#include <cassert>
#include <set>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    using baloon::setup::_cpuid_avx512_full;
    const std::set<std::string> full = {
        "avx512f","avx512bw","avx512dq","avx512vl","avx512_vbmi","avx512_vnni","f16c","fma"};
    assert(_cpuid_avx512_full(full));
    // dropping any single required flag must fail
    const char* need[] = {"avx512f","avx512bw","avx512dq","avx512vl","avx512_vbmi","avx512_vnni","f16c","fma"};
    for (const char* n : need) {
        std::set<std::string> partial = full;
        partial.erase(n);
        assert(!_cpuid_avx512_full(partial));
    }
    // avx512fp16 must NOT satisfy avx512f (prefix match not allowed)
    std::set<std::string> wrong = full;
    wrong.erase("avx512f"); wrong.insert("avx512fp16");
    assert(!_cpuid_avx512_full(wrong));
    assert(!_cpuid_avx512_full({}));
    return 0;
}

// strata/core/cpu_features.cpp — CPUID-based AVX-512 feature detection
//
// Queries CPUID leaf 7 for the five AVX-512 sub-features the expert
// kernel requires. On CPUs that don't support leaf 7, all features
// report as absent.

#include "strata/core/cpu_features.hpp"
#include <cstdio>
#include <cstdlib>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace strata {

CpuFeatures cpu_features() {
    CpuFeatures f;

#if defined(_MSC_VER)
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 0);
    if (regs[0] < 7) return f;
    __cpuidex(regs, 7, 0);
    unsigned ebx = static_cast<unsigned>(regs[1]);
    unsigned ecx = static_cast<unsigned>(regs[2]);
#else
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    __cpuid_count(0, 0, eax, ebx, ecx, edx);
    if (eax < 7) return f;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
#endif

    f.avx512f    = static_cast<int>((ebx >> 16) & 1u);
    f.avx512bw   = static_cast<int>((ebx >> 30) & 1u);
    f.avx512vl   = static_cast<int>((ebx >> 31) & 1u);
    f.avx512_vnni = static_cast<int>((ecx >> 11) & 1u);
    f.avx512_vbmi = static_cast<int>((ecx >>  1) & 1u);

    return f;
}

std::string CpuFeatures::missing_report() const {
    std::string result;
    auto append = [&](const char* name) {
        if (!result.empty()) result += ", ";
        result += name;
    };
    if (!avx512f)     append("AVX-512F");
    if (!avx512bw)    append("AVX-512BW");
    if (!avx512vl)    append("AVX-512VL");
    if (!avx512_vnni) append("AVX-512-VNNI");
    if (!avx512_vbmi) append("AVX-512-VBMI");
    return result;
}

void cpu_require_expert_support() {
    CpuFeatures f = cpu_features();
    if (f.expert_usable()) return;

    std::string missing = f.missing_report();
    fprintf(stderr, "FATAL: CPU lacks required AVX-512 features for expert kernel: %s\n",
            missing.c_str());
    fprintf(stderr, "The expert MLP kernel requires AVX-512F + BW + VL + VNNI + VBMI.\n");
    exit(1);
}

}  // namespace strata


#pragma once
// strata/core/cpu_features.hpp — CPUID-based AVX-512 feature detection
//
// The expert MLP kernel requires AVX-512F, AVX-512BW, AVX-512VL,
// AVX-512-VNNI, and AVX-512-VBMI. All five are required.
// Provides a diagnostic naming each missing feature individually.

#include <string>

namespace strata {

// ─── CpuFeatures ─────────────────────────────────────────────────────
// Probed once at startup. All fields default to 0 (not present).
struct CpuFeatures {
    int avx512f    = 0;   // EBX bit 16
    int avx512bw   = 0;   // EBX bit 30
    int avx512vl   = 0;   // EBX bit 31
    int avx512_vnni = 0;  // ECX bit 11
    int avx512_vbmi = 0;  // ECX bit  1

    // True iff all five required features are present.
    bool expert_usable() const {
        return avx512f && avx512bw && avx512vl && avx512_vnni && avx512_vbmi;
    }

    // Returns a comma-separated list of missing features, or empty if all present.
    std::string missing_report() const;
};

// ─── cpu_features ────────────────────────────────────────────────────
// Queries CPUID leaf 7 and returns the feature flags.
CpuFeatures cpu_features();

// ─── cpu_require_expert_support ──────────────────────────────────────
// Terminates the process with a diagnostic if any required feature is missing.
void cpu_require_expert_support();

}  // namespace strata


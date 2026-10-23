// strata/core/device_main.cpp — GPU capability and memory plan reporter
// Target: Iluvatar BI-V100 (ivcore10)
//
// Standalone tool: reports GPU properties, validates memory plan against
// actual cudaMemGetInfo, and enforces the compute capability floor.

#include "strata/core/device.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

int main() {
    std::string err;
    strata::DeviceInfo info;

    if (!strata::device_init(0, info, err)) {
        fprintf(stderr, "ERROR: %s\n", err.c_str());
        return 1;
    }

    printf("=== strata device report ===\n");
    printf("SMs:  %d\n", info.sm_count);
    printf("VRAM: %.2f GiB total, %.2f GiB free\n",
           static_cast<double>(info.vram_total) / 1073741824.0,
           static_cast<double>(info.vram_free) / 1073741824.0);
    printf("CC:   %d.%d\n", info.cc_major, info.cc_minor);

    // Query live memory info
    size_t free_bytes = 0, total_bytes = 0;
    if (strata::device_mem_info(free_bytes, total_bytes, err)) {
        printf("Live: %.2f GiB free / %.2f GiB total\n",
               static_cast<double>(free_bytes) / 1073741824.0,
               static_cast<double>(total_bytes) / 1073741824.0);
    }

    // Memory plan validation: 4.52 GiB dense + expert arena
    const double dense_gib = 4.522;
    const double dense_bytes = dense_gib * 1073741824.0;
    if (static_cast<double>(free_bytes) < dense_bytes) {
        printf("WARNING: free VRAM (%.2f GiB) < dense weight plan (%.2f GiB)\n",
               static_cast<double>(free_bytes) / 1073741824.0, dense_gib);
    } else {
        double remaining = (static_cast<double>(free_bytes) - dense_bytes) / 1073741824.0;
        printf("Plan:  %.2f GiB dense, %.2f GiB available for expert cache\n",
               dense_gib, remaining);
    }

    printf("=== done ===\n");
    return 0;
}


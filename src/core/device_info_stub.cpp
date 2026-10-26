// ============================================================================
// device_info_stub.cpp — Host-only stub (no CUDA runtime)
//
// Provides the same C API as device_info.cu so the Python wrapper and build
// system can be verified on machines that lack the CoreX SDK.  All GPU
// queries return "no device".
// ============================================================================

#include "baloon/config.h"
#include <cstdio>
#include <cstring>

extern "C" {

struct BaloonDeviceInfo {
    int   device_count;
    int   warp_size;
    int   max_threads;
    int   sm_count;
    int   compute_major;
    int   compute_minor;
    int   max_shmem_per_block;
    size_t total_mem;
    size_t free_mem;
    char  name[256];
};

int baloon_device_count() { return 0; }

int baloon_query_device(int dev, BaloonDeviceInfo* info) {
    BALOON_UNUSED(dev);
    if (!info) return -1;
    memset(info, 0, sizeof(*info));
    strncpy(info->name, "(host stub — no GPU)", sizeof(info->name) - 1);
    return 0;
}

int baloon_smoke_test(int dev) {
    BALOON_UNUSED(dev);
    printf("Host stub: smoke test skipped (no GPU)\n");
    return 0;  // pass trivially
}

} // extern "C"

#ifndef BALOON_LIB_ONLY

int main() {
    printf("=== Baloon Device Info (host stub) ===\n");
    printf("Platform : %s\n", BALOON_PLATFORM_STR);
    printf("WARP_SIZE: %d (compile-time)\n", WARP_SIZE);
    printf("Devices  : %d (no CoreX SDK)\n", baloon_device_count());
    printf("\n");
    printf("This is a host-only build for syntax/link verification.\n");
    printf("For GPU execution, build on a machine with CoreX SDK at\n");
    printf("  /usr/local/corex/bin/clang++\n");
    printf("\nHost stub smoke test: PASS\n");
    return 0;
}

#endif


// device_main.cpp - lists the visible devices; `--selftest` also checks each device's arena and poison fill.
// usage: device_main [--require N] [--selftest]   exit 0 ok, 1 failure, 2 usage.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "baloon/rt_device.hpp"

using namespace baloon::rt;

std::string human(uint64_t bytes) {
    static const char* unit[] = {"B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"};
    if (bytes < 1024) return std::to_string(bytes) + " B";
    double v = (double)bytes; int u = 0;
    while (v >= 1024.0 && u < 6) { v /= 1024.0; ++u; }
    char buf[48]; std::snprintf(buf, sizeof buf, "%.1f %s", v, unit[u]);
    return buf;
}

#ifndef RT_DEVICE_MAIN_DISABLED
static bool selftest(int ord) {
    const uint64_t cap = 1ull << 20;
    DevArena a(ord, cap);
    if (!a.ok()) return false;
    void* p = a.alloc(4096);
    if (!p || ((uintptr_t)p % kArenaAlign) != 0) return false;
    a.reset(true);
    std::vector<uint8_t> back(cap);
    if (cudaMemcpy(back.data(), a.base(), cap, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    for (uint8_t b : back) if (b != DevArena::kPoisonByte) return false;
    return true;
}

int main(int argc, char** argv) {
    int require = 1; bool st = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--require") && i + 1 < argc) require = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--selftest")) st = true;
        else { std::fprintf(stderr, "usage: %s [--require N] [--selftest]\n", argv[0]); return 2; }
    }
    const int n = device_count();
    std::printf("devices: %d\n", n);
    int rc = 0;
    for (int i = 0; i < n; ++i) {
        DeviceDesc d; std::string err;
        if (!query_device(i, &d, &err)) { std::printf("device %d: %s\n", i, err.c_str()); rc = 1; continue; }
        std::printf("device %d: %s, %s total, %s free, warp %d, %d SMs, cc %d.%d\n", d.ordinal, d.name.c_str(),
                    human(d.total_mem).c_str(), human(d.free_mem).c_str(), d.warp, d.sm_count, d.cc_major, d.cc_minor);
        if (st) { const bool ok = selftest(i); std::printf("device %d selftest: %s\n", i, ok ? "ok" : "FAILED"); if (!ok) rc = 1; }
    }
    if (n < require) { std::fprintf(stderr, "need %d device(s), found %d\n", require, n); rc = 1; }
    return rc;
}
#endif

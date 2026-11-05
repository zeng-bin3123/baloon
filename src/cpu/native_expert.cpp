// native_expert.cpp - one-time selection of the native CPU expert backend. Built with the AVX-512 ISA flags like the
// other src/cpu sources; the probe itself uses cpuid/xgetbv through kern::cpu_features(), never an AVX-512 instruction.
#include <atomic>
#include <mutex>

#include "baloon/cpu_native_traits.hpp"

namespace baloon {
namespace {

std::once_flag g_once;
std::atomic<int> g_backend{(int)NativeBackend::Unavailable};

void init_once() {
    const kern::CpuFeatures f = kern::cpu_features();
    NativeBackend b = NativeBackend::Unavailable;
    if (f.usable()) b = NativeBackend::Avx512;
    else if (f.avx2) b = NativeBackend::Avx2;
    g_backend.store((int)b, std::memory_order_release);
}

}  // namespace

void native_expert_init() { std::call_once(g_once, init_once); }

NativeBackend native_backend() {
    native_expert_init();
    return (NativeBackend)g_backend.load(std::memory_order_acquire);
}

}  // namespace baloon

// native_expert.cpp - per-format traits, one-time backend probe and the native CPU expert entry points (reference layout).
#include <atomic>
#include <mutex>

#include "baloon/u11_expert.hpp"

namespace baloon {
namespace u11 {
namespace native {

namespace {
std::once_flag g_once;
std::atomic<int> g_backend{0};
}  // namespace

FmtTraits traits(NativeFmt f) {
    switch (f) {
        case NativeFmt::Q2_0:    return {f, 42, 64, 18, 32, false, true};
        case NativeFmt::IQ2_XXS: return {f, 16, 256, 66, 256, true, true};
        case NativeFmt::IQ2_XS:  return {f, 17, 256, 74, 256, true, true};
        case NativeFmt::IQ3_XXS: return {f, 18, 256, 98, 256, true, true};
        case NativeFmt::IQ3_S:   return {f, 21, 256, 110, 256, true, true};
        case NativeFmt::IQ2_S:   return {f, 22, 256, 82, 256, true, true};
        default:                 return {NativeFmt::None, 0, 0, 0, 0, false, false};
    }
}

static void probe() {
    const kern::CpuFeatures c = kern::cpu_features();
    g_backend.store(c.usable() ? 2 : c.avx2 ? 1 : 0, std::memory_order_release);
}
void init_once() { std::call_once(g_once, probe); }
int backend() { init_once(); return g_backend.load(std::memory_order_acquire); }

NativeFmt native_fmt(int t) { return baloon::native_fmt(t); }
bool native_experts_available(int t, int64_t e, int64_t f) { return baloon::native_experts_available(t, e, f); }
bool native_quant_act(NativeFmt f, const float* x, uint32_t n, uint32_t nt, void* out) { return baloon::native_quant_act(f, x, n, nt, out); }
bool native_quant_h(NativeFmt f, const float* h, uint32_t n, uint32_t nt, void* out) { return baloon::native_quant_h(f, h, n, nt, out); }
bool native_gu_rows(NativeFmt f, const uint8_t* blob, uint32_t e, uint32_t ff, uint32_t r0, uint32_t r1, const void* xq, uint32_t nt,
                    float* act, size_t st) { return baloon::native_gu_rows(f, blob, e, ff, r0, r1, xq, nt, act, st); }
bool native_down_rows(NativeFmt f, const uint8_t* blob, uint32_t e, uint32_t ff, uint32_t r0, uint32_t r1, const void* hq, uint32_t nt,
                      float* out, size_t st) { return baloon::native_down_rows(f, blob, e, ff, r0, r1, hq, nt, out, st); }

}  // namespace native
}  // namespace u11
}  // namespace baloon

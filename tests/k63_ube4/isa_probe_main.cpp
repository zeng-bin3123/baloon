// isa_probe_main.cpp - tiny host executable linking the AVX-512 expert kernel object (src/cpu/expert_q2.cpp) and the explicit
// FMA helpers, all built with the reference ISA flags. It runs one quantize + one Q2_0 row dot + one fused multiply-add so the
// zmm / vpermb / vpdpbusd / vcvtph2ps / vfmadd instruction paths are reachable from a standalone binary.
#include <cstdio>
#include <cstdint>
#include <vector>
#include "baloon/cpu_expert_layout.hpp"
namespace ube4 { void fma_vec16(const float*, const float*, const float*, float*, uint32_t); }
int main() {
    using namespace baloon;
    std::vector<float> x(64, 0.5f), y(64, 2.0f), z(64, 1.0f), o(64);
    ube4::fma_vec16(x.data(), y.data(), z.data(), o.data(), 64);
    int8_t q[64]; float d[2]; int32_t hx[2];
    act_quant_any(x.data(), 64, q, d, hx);
    std::vector<uint8_t> w(18, 0);
    float out[1] = {0};
    const bool ok = q2_rows_any(w.data(), 0, 1, 1, 1, q, 64, d, hx, 2, out, 1);
    std::printf("isa probe: fma=%g rows_ok=%d feature=%s\n", o[0], (int)ok, kern::cpu_features().reason());
    return (o[0] == 2.0f) ? 0 : 1;
}

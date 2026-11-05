// rt_testdata.hpp - deterministic generators shared by the runtime CLIs and tests.
#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "baloon/rt_kernels.hpp"

namespace baloon {
namespace rt {

struct XorRng {
    uint64_t s;
    explicit XorRng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    float sym() { return (float)((double)(next() >> 11) / 9007199254740992.0 * 2.0 - 1.0); }   // [-1,1)
};

inline std::vector<uint16_t> make_bf16(XorRng& r, uint64_t n) { std::vector<uint16_t> v(n); for (auto& e : v) e = f32_to_bf16_trunc(r.sym()); return v; }
inline std::vector<float> make_f32(XorRng& r, uint64_t n) { std::vector<float> v(n); for (auto& e : v) e = r.sym(); return v; }

}  // namespace rt
}  // namespace baloon

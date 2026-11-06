// prefill_device_api.hpp - host API of the prefill CUDA kernels in src/prefill_kernels.cu. Pure C++ (no CUDA headers).
// Return: 0 ok, -1 rejected by host validation (nothing launched), >0 a cudaError_t value.
#pragma once
#include <cstdint>

namespace baloon {
namespace pfk_dev {

// out[i] = t*10240 + c*2560 + d of the device-side decode of i, for i < T*10240.
int launch_hc_decode(int T, int32_t* out);
// counts[e] += 1 for every (lane, i) -> expert lane + 32*i; counts has 512 entries.
int launch_route_cover(int32_t* counts);
// Ternary expert dequant: blob (1,382,400 bytes) -> gu_out[1280*2560] and d_out[2560*640] floats; value = (code-1)*scale.
int launch_expert_dequant(const uint8_t* blob, float* gu_out, float* d_out);
// q[i] = clamp(rint(x[i]*inv), -127, 127)
int launch_kv_quant(const float* x, int8_t* q, int64_t n, float inv);

}  // namespace pfk_dev
}  // namespace baloon

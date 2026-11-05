// f16_decode_bf16.hpp - integer models of the fp16 decoder's subnormal normalisation / exponent re-bias and of the
// bf16 round-to-nearest-even (host + device), plus their GPU entry points.
//   norm_shift(fuel, m, s): shift m left until bit 10 is set, counting shifts; stops when the fuel runs out.
//   For every non-zero subnormal mantissa m < 1024, norm_shift(10, m, 0) ends with 1024 <= m' < 2048, s <= 10, m' == m << s.
//   subnormal_exp_field(s) = 113 - s (113 = 127 - 15 + 1), always a valid normal exponent (103..113).
//   normal_exp_field(ex)   = ex - 15 + 127, evaluated as a SIGNED quantity (== ex + 112, 113..142 for ex 1..30);
//                            truncated unsigned arithmetic would be wrong (nat_truncation_hazard).
//   bf16_round(i) = (i + ((i >> 16) & 1) + 0x7FFF) >> 16 in 64-bit arithmetic (no carry-out loss).
#pragma once
#include <cstdint>
#include "baloon/f16_bits.hpp"

namespace baloon {
namespace f16bits {

struct NormShift { uint32_t m; uint32_t s; };

BALOON_BITS_HD inline NormShift norm_shift(uint32_t fuel, uint32_t m, uint32_t s) {
    while (fuel > 0u) {
        if (m >= 1024u) break;
        m *= 2u; ++s; --fuel;
    }
    return NormShift{m, s};
}

constexpr uint32_t kSubnormalBias = 127u - 15u + 1u;   // 113

BALOON_BITS_HD constexpr int32_t subnormal_exp_field(uint32_t s) { return (int32_t)kSubnormalBias - (int32_t)s; }
BALOON_BITS_HD constexpr int32_t normal_exp_field(int32_t ex) { return ex - 15 + 127; }

BALOON_BITS_HD constexpr uint64_t bf16_round(uint64_t i) { return (i + ((i >> 16) & 1u) + 0x7FFFu) >> 16; }

}  // namespace f16bits

namespace f16_decode_dev {
// For each half bit pattern h[i] (any 16-bit value): model_field[i] = f32 exponent field the model predicts
// (0 for +-0, 113 - s for subnormals, 255 for inf/NaN, ex - 15 + 127 for normals), norm_m[i]/norm_s[i] = norm_shift(10, mant, 0)
// (mant = h & 0x3FF), real_bits[i] = f32 bits of f32_from_f16(h[i]).
int f16_decode_step(const uint16_t* h, int32_t* model_field, uint32_t* norm_m, uint32_t* norm_s, uint32_t* real_bits,
                    int64_t n, void* stream);
// For each fp32 bit pattern bits[i] (non-NaN): model[i] = bf16_round(bits) (may be 65536 on carry out of the top),
// real[i] = round_f32_to_bf16_bits.
int bf16_round_step(const uint32_t* bits, uint32_t* model, uint16_t* real, int64_t n, void* stream);
}  // namespace f16_decode_dev
}  // namespace baloon

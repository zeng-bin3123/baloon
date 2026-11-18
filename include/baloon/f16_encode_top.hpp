// f16_encode_top.hpp - the top-level split of the fp32 -> fp16 encoder (host + device) and its GPU entry point.
//   encode_top(sign, rawexp, man) returns the final half bits when the input is inf/NaN (raw exponent 255 is tested FIRST,
//   it also satisfies the overflow test) or a finite overflow (saturates to a signed inf, mantissa field zero, never NaN),
//   and -1 for "normal or subnormal path".
#pragma once
#include <cstdint>
#include "baloon/f16_bits.hpp"

namespace baloon {
namespace f16bits {

BALOON_BITS_HD inline int32_t encode_top(uint32_t sign, uint32_t rawexp, uint32_t man) {
    if (rawexp == 255u) return (int32_t)(sign + 31744u + (man != 0u ? 512u : 0u));
    if ((int)rawexp - 127 + 15 >= 31) return (int32_t)(sign + 31744u);
    return -1;
}

}  // namespace f16bits
}  // namespace baloon

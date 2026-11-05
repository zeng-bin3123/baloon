// dequant_check.cpp - standalone self-check of the dense dequantisers and layout sizes (compiled out when
// BALOON_DEQUANT_MAIN_DISABLED is defined).
#ifndef BALOON_DEQUANT_MAIN_DISABLED
#include <cstdio>
#include "baloon/f16_bits_rounding.hpp"
int main() {
    using namespace baloon;
    uint8_t blk[18] = {0x00, 0x3C};  // d = 1.0
    for (int i = 0; i < 16; ++i) blk[2 + i] = (uint8_t)((i << 4) | (15 - i));
    float y[32];
    f16bits::dequantize_iq4_nl(blk, y);
    for (int j = 0; j < 16; ++j)
        if (y[j] != (float)dequant::kvalues_iq4nl[15 - j] || y[j + 16] != (float)dequant::kvalues_iq4nl[j]) { std::puts("iq4_nl mismatch"); return 1; }
    std::puts("dequant_check ok");
    return 0;
}
#endif

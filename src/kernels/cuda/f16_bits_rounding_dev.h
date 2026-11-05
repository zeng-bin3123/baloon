#pragma once
#include <cstdint>
namespace f16dev {
int launch_round(const uint32_t* in, uint16_t* half_out, uint16_t* bf_out, int32_t* top_out, uint32_t* rne_out, int n);
}

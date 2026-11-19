// kernels.cu - bf16_gemv_kernel and embed_gather_kernel moved to dense_bank.cpp and embed_head.cpp
// (both compiled as CUDA with $CUFLAGS); this file intentionally defines no __global__ kernels.
#include "baloon/rt_kernels.hpp"

namespace baloon {
namespace rt {
}  // namespace rt
}  // namespace baloon

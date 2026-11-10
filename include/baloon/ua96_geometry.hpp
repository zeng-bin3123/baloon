// ua96_geometry.hpp - device-side execution of the launch index maps of the gated-residual read and the GDN step kernels
// (theory modules FusedGatedResidualRead, GatedResidualReference, FusedGdnStepNorm, GdnReferenceKernels). Each kernel runs
// the (block, warp/lane, register) -> row/column map of the real launch geometry and counts how often every output
// element is claimed; a correct map claims each element exactly once (cover + unique), never out of range.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace baloon {
namespace ua96 {

constexpr int kN = 2560, kHc = 4, kD = 10240, kLr = 320, kS = 128;
constexpr int kDownBlocks = 40, kUpBlocks = 80, kUpCols = 32, kUpmBlocks = 160, kUpmCols = 16;
constexpr int kStepRg = 4, kStepRpg = 32, kConvChannels = (2 * 2 + 4) * 128;

struct CoverBuffers {      // all zero-initialised device counters
    uint32_t* down;        // kLr          : lo rows of the fused down kernel
    uint32_t* up;          // kD           : stacked rows (stream, column) of the up kernel
    uint32_t* upm;         // kD           : same for the multi-token up kernel
    uint32_t* gate;        // kD           : gate-kernel rows, ceil(D/8) blocks of 8 warps
    uint32_t* step_rows;   // kS * kS      : (row, column) cells of the GDN state owned by (row group, register, thread)
    uint32_t* conv;        // kConvChannels: conv channels, blocks of 128 threads
    uint32_t* oob;         // 1            : out-of-range claims (must stay 0)
};

cudaError_t cover_all(const CoverBuffers& b, cudaStream_t s = 0);

}  // namespace ua96
}  // namespace baloon

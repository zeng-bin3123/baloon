// prefill_kernel_geometry.hpp - index, layout and clamping arithmetic of the prefill CUDA kernels (hyper-connection
// read/write, GDN conv/recurrence, MoE routing/dequant, QSA helpers). Host/device usable.
#pragma once
#include <cstdint>
#include "baloon/f16_conversion.hpp"

namespace baloon {
namespace pfk {

constexpr int kThreads = 256;
constexpr int kHcD = 10240, kHcN = 2560, kHc = 4;           // residual stream [T][4][2560]
BALOON_HD inline int64_t blocks_for(int64_t n) { return (n + 255) / 256; }
BALOON_HD inline int64_t route_blocks(int64_t T) { return (T + 7) / 8; }  // 8 warps per block

// Residual thread decode i -> (t, c, d)
struct Thc { int64_t t; int c; int d; };
BALOON_HD inline Thc hc_decode(int64_t i) { return {i / kHcD, (int)(i % kHcD / kHcN), (int)(i % kHcN)}; }
BALOON_HD inline int64_t gr_mix_read(int64_t i, int c) { return i / kHcN * kHcD + (int64_t)c * kHcN + i % kHcN; }
BALOON_HD inline int gr_norm_idx(int64_t row, int d) { return (int)(row % kHc) * kHcN + d; }

// GDN
constexpr int kGdnChannels = 10240, kGdnConvBlock = 128, kGdnRows = 128, kGdnRowGroup = 32, kGdnHeads = 48;
BALOON_HD inline int gdn_row(int rg, int r) { return rg * kGdnRowGroup + r; }
BALOON_HD inline int rec_key_idx(int head, int tid) { return 16 * 128 + head % 16 * 128 + tid; }
BALOON_HD inline int rec_value_idx(int head, int col) { return 2 * 16 * 128 + head * 128 + col; }
BALOON_HD inline int l2_idx(int head, int tid) { return head * 128 + tid; }

// MoE
constexpr int kExperts = 512, kLanes = 32, kPerLane = 16;
BALOON_HD inline int lane_expert(int lane, int i) { return lane + i * kLanes; }
constexpr int64_t kGuRows = 1280, kGuBytes = 640, kDRows = 2560, kDBytes = 160;
constexpr int64_t O_GU_CODES = 0;
constexpr int64_t O_D_CODES = kGuRows * kGuBytes;                       // 819200
constexpr int64_t O_GU_SC = O_D_CODES + kDRows * kDBytes;               // 1228800
constexpr int64_t O_D_SC = O_GU_SC + kGuRows * 40 * 2;                  // 1331200
constexpr int64_t kExpertBlob = O_D_SC + kDRows * 10 * 2;               // 1382400
BALOON_HD inline int64_t gu_scale_idx(int64_t row, int64_t byte) { return row * 40 + byte * 4 / 64; }
BALOON_HD inline int64_t d_scale_idx(int64_t row, int64_t byte) { return row * 10 + byte * 4 / 64; }
BALOON_HD inline int64_t gu_out_idx(int64_t row, int64_t byte, int k) { return row * 2560 + byte * 4 + k; }
BALOON_HD inline int64_t d_out_idx(int64_t row, int64_t byte, int k) { return row * 640 + byte * 4 + k; }
constexpr int64_t kExpertThreads = kGuRows * kGuBytes + kDRows * kDBytes;  // 1,228,800 code bytes
BALOON_HD inline int code_field(unsigned c, int k) { return (int)((c >> (2 * k)) & 3u); }

// QSA helpers
BALOON_HD inline int split_q_idx(int h, int d) { return h * 512 + d; }
BALOON_HD inline int gate_attn_idx(int h, int d) { return h * 512 + 256 + d; }
BALOON_HD inline int kv_group(int z) { return z / 2; }
BALOON_HD inline int64_t kv_page_row(int64_t page, int kvh) { return page * 2 + kvh; }
// INT8 clamp used by kv_append: never -128.
BALOON_HD inline int clamp_q(int64_t q) { return q < -127 ? -127 : (q > 127 ? 127 : (int)q); }

}  // namespace pfk
}  // namespace baloon

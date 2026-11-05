// kern_cu_blocks.cuh - device-side element decoders of the ggml block formats used by the unit-14 kernels
// (Q4_0 2, Q5_0 6, Q8_0 8, IQ4_NL 20, Q3_K 11, Q4_K 12, Q5_K 13, Q6_K 14, IQ4_XS 23, Q2_0 42).
// dq_elem(type, block, i) returns the dequantized value of element i of one block, written from the block layouts
// (same layouts as include/baloon/scalar_dequantizers.hpp, which is the host reference).
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#include "baloon/f16_bits.hpp"

namespace baloon {
namespace kern {

constexpr int T_Q4_0 = 2, T_Q5_0 = 6, T_Q8_0 = 8, T_Q3_K = 11, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_IQ4_NL = 20,
              T_IQ4_XS = 23, T_Q2_0 = 42;

// IQ4_NL codebook (shared by IQ4_NL and IQ4_XS).
__device__ __constant__ int8_t kv_iq4nl_dev[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

__device__ __forceinline__ float rd_h(const uint8_t* p) { return f16bits::f32_from_f16((uint16_t)(p[0] | (p[1] << 8))); }

__host__ __device__ constexpr int block_elems_of(int type) {
    return (type == T_Q4_0 || type == T_Q5_0 || type == T_Q8_0 || type == T_IQ4_NL) ? 32
         : type == T_Q2_0 ? 64
         : (type == T_Q3_K || type == T_Q4_K || type == T_Q5_K || type == T_Q6_K || type == T_IQ4_XS) ? 256 : 0;
}
__host__ __device__ constexpr int block_bytes_of(int type) {
    return type == T_Q4_0 ? 18 : type == T_Q5_0 ? 22 : type == T_Q8_0 ? 34 : type == T_IQ4_NL ? 18 : type == T_Q2_0 ? 18
         : type == T_Q3_K ? 110 : type == T_Q4_K ? 144 : type == T_Q5_K ? 176 : type == T_Q6_K ? 210 : type == T_IQ4_XS ? 136 : 0;
}

__device__ __forceinline__ void scale_min_k4_dev(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

__device__ __forceinline__ float dq_elem(int type, const uint8_t* b, int i) {
    switch (type) {
        case T_Q4_0: {
            const float d = rd_h(b);
            const int q = i < 16 ? (b[2 + i] & 0xF) : (b[2 + i - 16] >> 4);
            return d * (float)(q - 8);
        }
        case T_IQ4_NL: {
            const float d = rd_h(b);
            const int q = i < 16 ? (b[2 + i] & 0xF) : (b[2 + i - 16] >> 4);
            return d * (float)kv_iq4nl_dev[q];
        }
        case T_Q5_0: {
            const float d = rd_h(b);
            const uint32_t qh = (uint32_t)b[2] | ((uint32_t)b[3] << 8) | ((uint32_t)b[4] << 16) | ((uint32_t)b[5] << 24);
            const int j = i < 16 ? i : i - 16;
            const int lo = i < 16 ? (b[6 + j] & 0xF) : (b[6 + j] >> 4);
            const int hi = (int)((qh >> (i < 16 ? j : j + 16)) & 1u) << 4;
            return d * (float)((lo | hi) - 16);
        }
        case T_Q8_0: return rd_h(b) * (float)(int8_t)b[2 + i];
        case T_Q2_0: return rd_h(b) * (float)(((b[2 + i / 4] >> (2 * (i % 4))) & 3) - 1);
        case T_Q4_K: case T_Q5_K: {
            const float d = rd_h(b), dmin = rd_h(b + 2);
            const int g = i / 64, w = i % 64, half = w / 32, l = w % 32;
            int s, m;
            scale_min_k4_dev(2 * g + half, b + 4, s, m);
            int q;
            if (type == T_Q4_K) {
                const uint8_t by = b[16 + 32 * g + l];
                q = half ? (by >> 4) : (by & 0xF);
            } else {
                const uint8_t by = b[48 + 32 * g + l];
                q = (half ? (by >> 4) : (by & 0xF)) + ((b[16 + l] & (1u << (2 * g + half))) ? 16 : 0);
            }
            return d * (float)s * (float)q - dmin * (float)m;
        }
        case T_Q6_K: {
            const int n = i / 128, w = i % 128, quad = w / 32, l = w % 32, is = l / 16;
            const uint8_t ql = b[64 * n + l + ((quad & 1) ? 32 : 0)];
            const int nib = quad < 2 ? (ql & 0xF) : (ql >> 4);
            const int hi = (b[128 + 32 * n + l] >> (2 * quad)) & 3;
            const int sc = (int8_t)b[192 + 8 * n + is + 2 * quad];
            return rd_h(b + 208) * (float)sc * (float)((nib | (hi << 4)) - 32);
        }
        case T_Q3_K: {
            const int n = i / 128, w = i % 128, j = w / 32, h16 = (w % 32) / 16, l = w % 16;
            const int si = 8 * n + 2 * j + h16;
            const uint8_t* s12 = b + 96;
            const int lo = si < 8 ? (s12[si] & 0xF) : (s12[si - 8] >> 4);
            const int hs = (s12[8 + (si % 4)] >> (2 * (si / 4))) & 3;
            const int sc = (lo | (hs << 4)) - 32;
            const int qi = 32 * n + h16 * 16 + l;
            const int q = (b[32 + qi] >> (2 * j)) & 3;
            const bool hm = (b[h16 * 16 + l] & (1u << (4 * n + j))) != 0;
            return rd_h(b + 108) * (float)sc * (float)(q - (hm ? 0 : 4));
        }
        case T_IQ4_XS: {
            const int ib = i / 32, w = i % 32, j = w % 16;
            const uint16_t sh = (uint16_t)(b[2] | (b[3] << 8));
            const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
            const uint8_t by = b[8 + 16 * ib + j];
            const int nib = w < 16 ? (by & 0xF) : (by >> 4);
            return rd_h(b) * (float)(ls - 32) * (float)kv_iq4nl_dev[nib];
        }
        default: return 0.f;
    }
}

}  // namespace kern
}  // namespace baloon

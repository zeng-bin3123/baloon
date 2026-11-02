// bl_gdn_reference_kernels.h - launch geometry and addressing of the non-projection parts of the gated delta net.
#pragma once
#include <vector>
#include "bl_nat.h"

namespace baloon::gdn_reference {

constexpr nat JTHREADS = 32;
constexpr nat MAX_H = 1;

constexpr nat gridX(nat S) { return (S + 31) / 32; }
constexpr nat gridY(nat h_v) { return (h_v + MAX_H - 1) / MAX_H; }
constexpr nat headsInBlock(nat by, nat h_v) { return nmin(MAX_H, sat_sub(h_v, by)); }
constexpr nat qkHead(nat h, nat h_k) { return nmod(h, h_k); }
// (i, h, j) element of the (S, h_v, S) state; a thread owns column pointer h*S+j and strides h_v*S per row.
constexpr nat stateIndex(nat i, nat h, nat j, nat S, nat h_v) { return (i * h_v + h) * S + j; }
constexpr nat stateViaColumn(nat i, nat h, nat j, nat S, nat h_v) { return h * S + j + i * (h_v * S); }
constexpr bool stepSizeOk(nat S) { return S <= 128; }

constexpr nat convStateIndex(nat c, nat i, nat dc) { return c * (dc - 1) + i; }
constexpr nat convWeightIndex(nat c, nat i, nat dc) { return c * dc + i; }
constexpr nat convGridBlocks(nat C) { return (C + 255) / 256; }
constexpr nat betaGridBlocks(nat n) { return (n + 127) / 128; }
constexpr bool l2ColsOk(nat cols) { return cols <= 1024; }

// Sliding causal conv window (d_conv - 1 history values): drop the oldest, append x.
inline std::vector<long> slideList(const std::vector<long>& st, long x) {
  std::vector<long> r(st.begin() + (st.empty() ? 0 : 1), st.end());
  r.push_back(x);
  return r;
}

}  // namespace baloon::gdn_reference

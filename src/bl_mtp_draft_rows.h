// bl_mtp_draft_rows.h - row bookkeeping of the MTP draft layer.
#pragma once
#include "bl_nat.h"

namespace baloon::mtp_draft_rows {

// Bump allocation of n elements of size sz rounds to 256 bytes.
constexpr nat bumpUsed(nat used, nat n, nat sz) { return used + (n * sz + 255) / 256 * 256; }
// Attention window capacity rounds up to 64 cells.
constexpr nat cap64(nat w) { return (w + 63) / 64 * 64; }

// Staging holds 2*maxT rows: [0, maxT) catch-up, row 2*maxT-1 the full-layer row,
// chain step j (1 .. maxT-2) uses row maxT + j - 1.
constexpr nat fullRow(nat maxT) { return sat_sub(2 * maxT, 1); }
constexpr nat chainRow(nat maxT, nat j) { return sat_sub(maxT + j, 1); }
constexpr nat stagedRows(nat maxT) { return 2 * maxT; }

// Prefill walks the prompt in chunks of at most maxT and always makes progress while cells remain.
constexpr nat chunkT(nat maxT, nat n, nat c) { return nmin(maxT, sat_sub(n, c)); }
constexpr nat prefillChunks(nat maxT, nat n) {
  nat c = 0, k = 0;
  while (c < n) {
    nat t = chunkT(maxT, n, c);
    if (t == 0) return k;  // maxT == 0: no progress possible
    c += t; ++k;
  }
  return k;
}

}  // namespace baloon::mtp_draft_rows

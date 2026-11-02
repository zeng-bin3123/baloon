// bl_gdn_qsa_layer_layout.h - buffer-carving arithmetic and stage selection of the layer composition.
#pragma once
#include <vector>
#include "bl_nat.h"

namespace baloon::gdn_qsa_layer_layout {

// Q8_K buffers: n/256 blocks of 292 bytes.
constexpr nat q8kBytes(nat n) { return n / 256 * 292; }
constexpr nat align16(nat v) { return (v + 15) / 16 * 16; }

// Bump cursor: every take is rounded to 16 bytes, so the total covers the sum of the parts.
struct Cursor {
  nat used = 0;
  nat take(nat count, nat sz) {  // returns the start offset of the carved region
    nat off = used;
    used = takeUsed(used, count, sz);
    return off;
  }
  static constexpr nat takeUsed(nat used, nat count, nat sz) { return align16(used + count * sz); }
};
constexpr nat takeUsed(nat used, nat count, nat sz) { return Cursor::takeUsed(used, count, sz); }
inline nat carveTotal(const std::vector<nat>& parts) {
  nat t = 0;
  for (nat p : parts) t += align16(p);
  return t;
}

// Paged KV state holds ceil(a / b) * b >= a cells for page size b.
constexpr nat ceilDivMul(nat a, nat b) { return b ? (a + b - 1) / b * b : 0; }

// RoPE row pos_base+pos must be below max_cells; then row*h + j (j < h) stays inside the table.
constexpr bool ropeRowOk(nat posBase, nat pos, nat maxCells) { return posBase + pos < maxCells; }

// Stage i (0..4) under stage prefix pfx: positive prefix -> runs iff pfx >= i+1;
// prefix 0 -> by half: stages 0-2 for halves 0/1, stages 3-4 for halves 0/2.
constexpr bool runStage(nat pfx, nat half, nat i) {
  if (pfx > 0) return pfx >= i + 1;
  if (i < 3) return half == 0 || half == 1;
  return half == 0 || half == 2;
}

// Per-layer dump region ends inside the whole dump buffer of (nl + 1) regions of n bytes.
constexpr nat dumpSlotEnd(nat l, nat n) { return (l + 1) * n + n; }
constexpr nat dumpBufferBytes(nat nl, nat n) { return (nl + 1) * n; }

}  // namespace baloon::gdn_qsa_layer_layout

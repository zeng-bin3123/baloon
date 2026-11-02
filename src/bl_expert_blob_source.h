// bl_expert_blob_source.h - index arithmetic of file-backed and arena-backed expert sources.
#pragma once
#include <optional>
#include "bl_nat.h"

namespace baloon::expert_blob_source {

// Flat blob index of (layer l, expert e). An expert id >= ne aliases the next layer, so checked() refuses it.
constexpr nat flat(nat ne, nat l, nat e) { return l * ne + e; }
constexpr std::optional<nat> checked(nat nl, nat ne, nat l, nat e) {
  if (l >= nl || e >= ne) return std::nullopt;
  return flat(ne, l, e);
}

// Offset of an expert's blob inside its layer's arena registration slice.
constexpr nat sliceOff(nat bb, nat e) { return e * bb; }
// The arena is one max-blob longer than the file data so a max-blob copy from any blob start stays inside.
constexpr nat arenaBytes(nat total, nat maxBlob) { return total + maxBlob; }

// GGUF chunked loading: a chunk of b experts starting at expert a is valid iff it fits in the tensor.
constexpr bool ggufChunkFits(nat per, nat a, nat b, nat ne) { return per * a + per * b <= per * ne; }
constexpr bool gguf_expert_in_range(nat per, nat a, nat b, nat ne, nat k) {
  return per > 0 && ggufChunkFits(per, a, b, ne) && k < b && (a + k) < ne;
}

}  // namespace baloon::expert_blob_source

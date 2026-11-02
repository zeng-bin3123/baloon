// bl_weights_loader_segments.h - byte arithmetic of the dense-weight loader.
#pragma once
#include "bl_nat.h"

namespace baloon::weights_loader_segments {

// Quantized tensor = codes | scales | optional offsets. An fp16 scale plane is half its engine size in the pack.
constexpr nat srcScale(nat scales, bool fp16) { return fp16 ? scales / 2 : scales; }
constexpr nat srcTotal(nat codes, nat scales, nat offs, bool fp16) { return codes + srcScale(scales, fp16) + offs; }
constexpr nat dstTotal(nat codes, nat scales, nat offs) { return codes + scales + offs; }

// Conversion chunks are at most 8 MiB of source; stage-out is twice stage-in.
constexpr nat CHUNK = 8388608;
constexpr nat STAGE_OUT = 2 * CHUNK;

// Widening never splits an fp16 element: the chunk byte count is rounded down to even.
constexpr nat widenN(nat rem) { return (nmin(CHUNK, rem) % 2 == 1) ? nmin(CHUNK, rem) - 1 : nmin(CHUNK, rem); }
constexpr nat widenOutBytes(nat n) { return n / 2 * 4; }   // f16 -> f32
constexpr nat bf16OutBytes(nat n) { return n / 4 * 2; }    // f32 -> bf16 promotion

// Compacted placement rounds each tensor's destination size to the index alignment a (> 0).
constexpr nat placeAt(nat at, nat dst, nat a) { return at + (dst + a - 1) / a * a; }

// Raw 16-bit index kinds map to engine kinds: 4 -> 1 (BF16), 5 -> 3 (F16), others unchanged.
constexpr nat engineKind(nat k) { return k == 4 ? 1 : (k == 5 ? 3 : k); }

}  // namespace baloon::weights_loader_segments

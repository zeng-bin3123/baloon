/-!
# MODULE: weight_loader_segments
Byte accounting of the dense-weight loader: per-plane segment sums with fp16 scale widening, chunked
conversions (BF16 high-half, f16 widening) and their staging-buffer bounds, and the compacted arena layout.

## Source semantics
- A quantized tensor is codes, scales and an optional offset plane. With fp16 scales the source scale plane is
  half the destination plane; the segment sums are checked against the index totals.
- Chunks are at most 8 MiB of source. BF16 conversion maps 4 source bytes to 2; f16 widening maps 2 to 4, so
  the output staging buffer is twice the input one.
- Destination offsets inside a converted run are `done / 2` (shrinking) or `done * 2` (widening).
- A skip set compacts the arena: kept rows are placed at 256-aligned running offsets.

## Form template identifiers
- `srcSum`, `dstSum` <- `sum_src`, `sum_dst`
- `widenOut`, `bf16Out` <- `out_bytes = (n/2)*4` and `(n/4)*2`
- `placeNext`        <- `at += (dst_bytes + a - 1) / a * a`
-/

namespace Strata.Weights

/-- Source bytes of a quantized tensor's three planes. -/
def srcSum (codes scales off : Nat) (fp16 : Bool) : Nat :=
  codes + (if fp16 then scales / 2 else scales) + off

/-- Destination bytes of the three planes. -/
def dstSum (codes scales off : Nat) : Nat := codes + scales + off

/-- Output bytes of widening `n` source bytes of fp16 to f32. -/
def widenOut (n : Nat) : Nat := (n / 2) * 4

/-- Output bytes of taking the high half of `n` source bytes of f32. -/
def bf16Out (n : Nat) : Nat := (n / 4) * 2

/-- Next compacted placement offset (alignment 256). -/
def placeNext (at_ bytes : Nat) : Nat := at_ + (bytes + 255) / 256 * 256

/-- Without fp16 scales the source and destination sums agree. -/
theorem srcSum_eq_dstSum (c s o : Nat) : srcSum c s o false = dstSum c s o := by
  simp [srcSum, dstSum]

/-- Widening can only grow the plane: the source sum never exceeds the destination sum. -/
theorem srcSum_le_dstSum (c s o : Nat) (fp : Bool) : srcSum c s o fp ≤ dstSum c s o := by
  unfold srcSum dstSum
  split <;> omega

/-- With an even fp16 scale plane the source is exactly half of the destination plane. -/
theorem fp16_half (s : Nat) (h : s % 2 = 0) : (s / 2) * 2 = s := by
  omega

/-- Widening an even run doubles its byte count. -/
theorem widenOut_double (n : Nat) (h : n % 2 = 0) : widenOut n = 2 * n := by
  unfold widenOut; omega

/-- The output staging buffer (twice the 8 MiB chunk) holds any widened chunk. -/
theorem widen_fits (n : Nat) (h : n ≤ 8388608) : widenOut n ≤ 2 * 8388608 := by
  unfold widenOut; omega

/-- BF16 high-half conversion halves a 4-byte-aligned run. -/
theorem bf16Out_half (n : Nat) (h : n % 4 = 0) : bf16Out n = n / 2 := by
  unfold bf16Out; omega

/-- The BF16 output fits the staging buffer. -/
theorem bf16_fits (n : Nat) (h : n ≤ 8388608) : bf16Out n ≤ 2 * 8388608 := by
  unfold bf16Out; omega

/-- A chunk size is a multiple of 4 (so converted chunks never split an element). -/
theorem chunk_mod4 : 8388608 % 4 = 0 := by
  decide

/-- Consecutive BF16 chunks tile the destination: the next chunk starts where this one's output ends. -/
theorem bf16_chunks_tile (done n : Nat) (hd : done % 4 = 0) (hn : n % 4 = 0) :
    done / 2 + bf16Out n = (done + n) / 2 := by
  unfold bf16Out; omega

/-- Consecutive widening chunks tile the destination. -/
theorem widen_chunks_tile (done n : Nat) (hd : done % 2 = 0) (hn : n % 2 = 0) :
    done * 2 + widenOut n = (done + n) * 2 := by
  unfold widenOut; omega

/-- The promoted-type size check: `4 B/elem` in and `2 B/elem` out give half the source size. -/
theorem promoted_sizes (elems : Nat) : elems * 2 * 2 = elems * 4 := by
  omega

/-- Compacted placement is 256-aligned if the previous offset was. -/
theorem placeNext_mod (a b : Nat) (h : a % 256 = 0) : placeNext a b % 256 = 0 := by
  unfold placeNext; omega

/-- Compacted placement leaves room for the tensor. -/
theorem placeNext_covers (a b : Nat) : a + b ≤ placeNext a b := by
  unfold placeNext; omega

/-- The index row has 19 fields; any other count is refused. -/
def rowOk (n : Nat) : Bool := decide (n = 19)

/-- Only exactly 19 parsed fields pass. -/
theorem rowOk_iff (n : Nat) : rowOk n = true ↔ n = 19 := by
  simp [rowOk]

end Strata.Weights

/-!
# MODULE: weight_loader_segments
Byte accounting of the dense-weight loader: quantized tensors are loaded as codes / scales / offset segments,
promoted float tensors shrink from 4 to 2 bytes per element, and fp16 scale planes widen to f32.

## Source semantics
* The segment lists must cover both recorded totals (`sum_src`, `sum_dst`) or the loader refuses.
* An fp16 scale plane has `scales_bytes / 2` source bytes and an even `scales_bytes`.
* A promoted tensor is `4*elems` in and `2*elems` out; Bf16High/ToF16 map `n` source bytes to `n/2`.
* WidenF16 maps `n` (even) source bytes to `2n` output bytes; staging output is `2*CHUNK` so it never overflows.
* Compaction aligns each kept tensor to `align` (default 256).

## Form template identifiers
* `Seg` sums -> `quantSrcTotal` / `quantDstTotal`
* `Conv::Bf16High`/`ToF16` -> `narrowOut`, `Conv::WidenF16` -> `widenOut`
* `CHUNK` -> `chunk`, `a = align` -> `align256`
-/

namespace Strata.WeightLoader

-- § quantized tensor segments

/-- Source bytes of a quantized tensor: codes, then (possibly half-width) scales, then the offset plane. -/
def quantSrcTotal (codes scales offset : Nat) (scalesFp16 : Bool) : Nat :=
  codes + (if scalesFp16 then scales / 2 else scales) + offset

/-- Destination bytes of a quantized tensor: all three planes in engine form. -/
def quantDstTotal (codes scales offset : Nat) : Nat := codes + scales + offset

/-- Without fp16 scales the two layouts have the same size. -/
theorem quant_plain_same (codes scales offset : Nat) :
    quantSrcTotal codes scales offset false = quantDstTotal codes scales offset := by
  simp [quantSrcTotal, quantDstTotal]

/-- With fp16 scales (even byte count) the destination exceeds the source by exactly the scale plane's half. -/
theorem quant_fp16_growth (codes scales offset : Nat) (heven : scales % 2 = 0) :
    quantDstTotal codes scales offset = quantSrcTotal codes scales offset true + scales / 2 := by
  simp only [quantDstTotal, quantSrcTotal, if_true]
  omega

-- § conversions

/-- Output bytes of Bf16High / ToF16 for `n` source bytes (4 B -> 2 B per element). -/
def narrowOut (n : Nat) : Nat := n / 4 * 2

/-- Output bytes of WidenF16 for `n` source bytes (2 B -> 4 B per element). -/
def widenOut (n : Nat) : Nat := n / 2 * 4

/-- The staging chunk of source bytes (8 MiB). -/
def chunk : Nat := 8388608

/-- Narrowing halves a whole number of f32 elements, and the destination offset `done/2` agrees. -/
theorem narrow_half (n : Nat) (h : n % 4 = 0) : narrowOut n = n / 2 := by
  unfold narrowOut; omega

/-- Widening doubles an even byte count exactly. -/
theorem widen_double (n : Nat) (h : n % 2 = 0) : widenOut n = 2 * n := by
  unfold widenOut; omega

/-- The widened output of one chunk fits the doubled staging buffer. -/
theorem widen_fits_stage (n : Nat) (h : n ≤ chunk) : widenOut n ≤ 2 * chunk := by
  unfold widenOut chunk at *; omega

/-- The narrowed output of one chunk fits the staging buffer. -/
theorem narrow_fits_stage (n : Nat) (h : n ≤ chunk) : narrowOut n ≤ 2 * chunk := by
  unfold narrowOut chunk at *; omega

/-- A promoted tensor of `elems` elements: the check `src = 4*elems ∧ dst = 2*elems` makes narrowing exact. -/
theorem promoted_exact (elems : Nat) : narrowOut (4 * elems) = 2 * elems := by
  unfold narrowOut; omega

/-- Widening an fp16 plane of `m` elements (2m source bytes) yields 4m bytes, twice the source. -/
theorem widen_plane (m : Nat) : widenOut (2 * m) = 4 * m := by
  unfold widenOut; omega

-- § arena compaction

/-- Alignment used when compacting the arena with a skip set. -/
def align256 (x : Nat) : Nat := (x + 255) / 256 * 256

/-- Aligned placement never shrinks a tensor. -/
theorem align256_ge (x : Nat) : x ≤ align256 x := by
  unfold align256; omega

/-- Aligned tensor sizes keep every placement 256-aligned. -/
theorem align256_mod (x : Nat) : align256 x % 256 = 0 := by
  unfold align256; omega

/-- Compacted arena size: sum of aligned sizes of the kept tensors. -/
def compactBytes : List Nat → Nat
  | [] => 0
  | b :: bs => align256 b + compactBytes bs

/-- Raw total of the kept tensors. -/
def rawBytes : List Nat → Nat
  | [] => 0
  | b :: bs => b + rawBytes bs

/-- The compacted arena always holds the kept tensors. -/
theorem compact_ge_raw (bs : List Nat) : rawBytes bs ≤ compactBytes bs := by
  induction bs with
  | nil => simp [rawBytes, compactBytes]
  | cons b bs ih =>
    simp only [rawBytes, compactBytes]
    have := align256_ge b
    omega

/-- Skipping a tensor never grows the compacted arena. -/
theorem compact_skip_le (b : Nat) (bs : List Nat) : compactBytes bs ≤ compactBytes (b :: bs) := by
  simp only [compactBytes]
  omega

end Strata.WeightLoader

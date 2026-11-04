/-!
# MODULE: weight_loader
SOURCES: src/core/weights.cpp
The dense-weight loader: reads `index.txt`, places each tensor in the arena (256-byte aligned, optionally
compacted by a skip set), and streams it plane by plane in 8 MiB chunks, converting widths on the way.

## Source semantics
* A quantized tensor is segments codes | scales | offsets; an fp16 scale plane is half as large in the pack as
  in the arena and is widened exactly (fp16 → f32).
* The segment totals are checked against the index's `src_bytes`/`dst_bytes` before any byte moves.
* BF16/F16 tensors promoted from f32 containers are 4 B/elem in and 2 B/elem out.
* Index kinds 4/5 (raw 16-bit) are copied and take the engine forms 1 and 3.
* Chunks never split an element; the widened chunk fits the 2×CHUNK output staging buffer.

## Form template identifiers
- Seg / segs → segment list ; chunkLen → `n = min(CHUNK, rem)` ; placeAll → compaction loop
- convFor → conversion selection ; mapKind → raw16 kind remap
-/

namespace Strata.WeightLoader

/-- 8 MiB of source per staging round. -/
def CHUNK : Nat := 8388608

-- § segment lists

/-- Source bytes of the scale plane. -/
def srcScaleBytes (scalesFp16 : Bool) (scales : Nat) : Nat := if scalesFp16 then scales / 2 else scales

/-- Total source bytes of the three segments. -/
def sumSrc (codes scales offset : Nat) (fp16 : Bool) : Nat := codes + srcScaleBytes fp16 scales + offset

/-- Total destination bytes of the three segments. -/
def sumDst (codes scales offset : Nat) : Nat := codes + scales + offset

/-- The destination total is the three plane sizes. -/
theorem sumDst_eq (c s o : Nat) : sumDst c s o = c + s + o := rfl

/-- With plain scales the two totals agree. -/
theorem sumSrc_plain (c s o : Nat) : sumSrc c s o false = sumDst c s o := by
  simp [sumSrc, sumDst, srcScaleBytes]

/-- With fp16 scales of even byte count, the source is exactly half the scale plane smaller. -/
theorem sumSrc_fp16 (c s o : Nat) (he : s % 2 = 0) : sumSrc c s o true + s / 2 = sumDst c s o := by
  simp only [sumSrc, sumDst, srcScaleBytes, if_true]
  omega

/-- An even fp16 scale plane survives halve-then-double. -/
theorem fp16_half_double (s : Nat) (he : s % 2 = 0) : s / 2 * 2 = s := by
  omega

-- § conversions: widen, bf16, f16

/-- Widening `n` fp16 bytes yields `2 * n` f32 bytes. -/
theorem widen_out_bytes (n : Nat) (he : n % 2 = 0) : 4 * (n / 2) = 2 * n := by
  omega

/-- Widened chunks tile the destination: `done*2 + out = (done + n)*2`. -/
theorem widen_tiles (done n : Nat) (he : n % 2 = 0) : done * 2 + 4 * (n / 2) = (done + n) * 2 := by
  omega

/-- A widened chunk fits the `2 * CHUNK` staging buffer. -/
theorem widen_fits_stage (n : Nat) (h : n ≤ CHUNK) : 4 * (n / 2) ≤ 2 * CHUNK := by
  unfold CHUNK at *
  omega

/-- Narrowing a 4-byte container to 2 bytes tiles the destination at half the source offset. -/
theorem narrow_tiles (done n : Nat) (hd : done % 4 = 0) (hn : n % 4 = 0) :
    done / 2 + 2 * (n / 4) = (done + n) / 2 := by
  omega

/-- The promoted-tensor check: 4 B/elem in and 2 B/elem out saves exactly 2 B/elem. -/
theorem promoted_saves (elems src dst : Nat) (h1 : src = elems * 4) (h2 : dst = elems * 2) :
    src - dst = elems * 2 := by
  omega

-- § chunk loop

/-- Length of the next chunk. -/
def chunkLen (rem : Nat) : Nat := min CHUNK rem

/-- While bytes remain, the next chunk is non-empty and strictly shrinks the remainder. -/
theorem chunk_progress (rem : Nat) (h : 0 < rem) : 0 < chunkLen rem ∧ rem - chunkLen rem < rem := by
  unfold chunkLen CHUNK
  omega

/-- A chunk never exceeds the staging size. -/
theorem chunk_le (rem : Nat) : chunkLen rem ≤ CHUNK := by
  unfold chunkLen; omega

/-- The chunk size is even (needed so widening never splits an element). -/
theorem chunk_even_of_full (rem : Nat) (h : CHUNK ≤ rem) : chunkLen rem % 2 = 0 := by
  unfold chunkLen CHUNK at *
  omega

/-- When the remainder is a multiple of 4, so is the next chunk. -/
theorem chunk_mult4 (rem : Nat) (h : rem % 4 = 0) : chunkLen rem % 4 = 0 := by
  unfold chunkLen CHUNK
  omega

-- § arena placement and compaction

/-- Round up to a multiple of 256 (the default alignment). -/
def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Rounding keeps all bytes. -/
theorem alignUp256_ge (n : Nat) : n ≤ alignUp256 n := by
  unfold alignUp256; omega

/-- Rounded sizes are multiples of 256. -/
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by
  unfold alignUp256; omega

/-- The compaction loop: offsets of kept rows, each advancing by its aligned size. -/
def placeAll : Nat → List Nat → Nat
  | at_, [] => at_
  | at_, x :: xs => placeAll (at_ + alignUp256 x) xs

/-- Placement from an aligned start gives an aligned pool. -/
theorem placeAll_mod : ∀ (l : List Nat) (a : Nat), a % 256 = 0 → placeAll a l % 256 = 0
  | [], a, h => by simpa [placeAll] using h
  | x :: xs, a, h => by
    have h1 := alignUp256_mod x
    have := placeAll_mod xs (a + alignUp256 x) (by omega)
    simpa [placeAll] using this

/-- The pool is at least the start plus the plain sum of sizes. -/
theorem placeAll_ge : ∀ (l : List Nat) (a : Nat), a + l.sum ≤ placeAll a l
  | [], a => by simp [placeAll]
  | x :: xs, a => by
    have h1 := alignUp256_ge x
    have h2 := placeAll_ge xs (a + alignUp256 x)
    simp only [placeAll, List.sum_cons]
    omega

/-- Each row's end is at or below the next row's start (rows never overlap). -/
theorem row_end_le_next (at_ x : Nat) : at_ + x ≤ at_ + alignUp256 x := by
  have := alignUp256_ge x
  omega

-- § index kinds

/-- Raw 16-bit index kinds 4 (BF16) and 5 (F16) take engine forms 1 and 3; others are unchanged. -/
def mapKind (k : Nat) : Nat := if k = 4 then 1 else if k = 5 then 3 else k

/-- Kind 4 becomes the BF16 engine form. -/
theorem mapKind_bf16 : mapKind 4 = 1 := by decide

/-- Kind 5 becomes the F16 engine form. -/
theorem mapKind_f16 : mapKind 5 = 3 := by decide

/-- Other kinds are untouched. -/
theorem mapKind_other (k : Nat) (h4 : k ≠ 4) (h5 : k ≠ 5) : mapKind k = k := by
  simp [mapKind, h4, h5]

/-- Conversion applied to an unquantized row. -/
inductive Conv
  | copy
  | bf16High
  | toF16

/-- Selection rule: raw 16-bit rows are copied, promoted rows convert, others copy. -/
def convFor (raw16 : Bool) (kind : Nat) : Conv :=
  if raw16 then Conv.copy else if kind = 1 then Conv.bf16High else if kind = 3 then Conv.toF16 else Conv.copy

/-- Raw 16-bit rows are never converted. -/
theorem convFor_raw (k : Nat) : convFor true k = Conv.copy := by
  simp [convFor]

/-- A promoted BF16 row takes the high half. -/
theorem convFor_bf16 : convFor false 1 = Conv.bf16High := by
  simp [convFor]

/-- A promoted F16 row converts with round-to-nearest-even. -/
theorem convFor_f16 : convFor false 3 = Conv.toF16 := by
  simp [convFor]

-- § arena sufficiency

/-- The arena check rejects exactly when the arena is smaller than the pool. -/
def arenaOk (arena pool : Nat) : Bool := decide (pool ≤ arena)

end Strata.WeightLoader

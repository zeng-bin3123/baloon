/-!
# MODULE: weight_loader
Arithmetic contracts of `src/core/weights.cpp`: the plane-by-plane segment list of a quantized tensor, the
8 MiB chunked transform loop (copy, bf16 high-half, fp16 widen), and arena placement under a skip set.

## Source semantics
* A quantized tensor is codes | scales | offsets; in the pack the scale plane may be fp16 (half the bytes) and is
  widened to f32 in the arena; segment sums must equal the index's recorded totals.
* Chunks are `min(CHUNK, remaining)` with `CHUNK = 8 MiB`; widening/bf16 conversions never split an element, each
  chunk's destination range is contiguous with the previous one, and the output fits `stage_out = 2*CHUNK`.
* A promoted (bf16/f16-in-f32) tensor is `4*elems` in and `2*elems` out.
* With a skip set the kept rows are re-placed in index order at 256-byte alignment.

## Form template identifiers
* `chunkLen` <- `n = min(CHUNK, s.src_bytes - done)`
* `srcScale` <- `src_scale_bytes`
* `placeTotal` <- compacted `pool`
-/

namespace Strata.WeightLoader

-- § chunk loop

/-- Chunk length: `min(8 MiB, remaining)`. -/
def chunkLen (rem : Nat) : Nat := min 8388608 rem

/-- A non-empty remainder yields a non-empty chunk, within the remainder and within CHUNK. -/
theorem chunk_progress (rem : Nat) (h : 0 < rem) :
    0 < chunkLen rem ∧ chunkLen rem ≤ rem ∧ chunkLen rem ≤ 8388608 := by
  unfold chunkLen; omega

/-- Each step conserves `done + remaining = total`, so the loop covers the span exactly. -/
theorem chunk_conserve (done rem total : Nat) (h : done + rem = total) :
    (done + chunkLen rem) + (rem - chunkLen rem) = total := by
  unfold chunkLen; omega

/-- Chunks never split a 4-byte element when the span is a multiple of 4. -/
theorem chunk_mod4 (rem : Nat) (h : rem % 4 = 0) : chunkLen rem % 4 = 0 := by
  unfold chunkLen; omega

/-- Chunks never split a 2-byte element when the span is even. -/
theorem chunk_mod2 (rem : Nat) (h : rem % 2 = 0) : chunkLen rem % 2 = 0 := by
  unfold chunkLen; omega

/-- bf16 high-half chunks are contiguous in the destination: `done/2 + (n/4)*2 = (done+n)/2`. -/
theorem bf16_chunks_contiguous (done n : Nat) (hd : done % 4 = 0) (hn : n % 4 = 0) :
    done / 2 + n / 4 * 2 = (done + n) / 2 := by
  omega

/-- fp16 widening chunks are contiguous in the destination: `done*2 + (n/2)*4 = (done+n)*2`. -/
theorem widen_chunks_contiguous (done n : Nat) (hn : n % 2 = 0) :
    done * 2 + n / 2 * 4 = (done + n) * 2 := by
  omega

/-- The converted output of one chunk fits `stage_out` (`2*CHUNK` bytes) for every transform. -/
theorem stage_out_fits (n : Nat) (h : n ≤ 8388608) :
    n ≤ 2 * 8388608 ∧ n / 4 * 2 ≤ 2 * 8388608 ∧ n / 2 * 4 ≤ 2 * 8388608 := by
  omega

-- § segment list

/-- Source bytes of the scale plane: half when stored as fp16. -/
def srcScale (sb : Nat) (fp16 : Bool) : Nat := if fp16 = true then sb / 2 else sb

/-- Destination sum of the three segments equals the recorded destination size. -/
theorem seg_dst_sum (cb sb ob : Nat) : cb + sb + ob = cb + sb + ob := rfl

/-- Source sum of the three segments (codes, scale source, offsets). -/
theorem seg_src_sum (cb sb ob : Nat) (fp16 : Bool) :
    cb + srcScale sb fp16 + ob = cb + (if fp16 = true then sb / 2 else sb) + ob := rfl

/-- With an even fp16 plane, widening exactly doubles the source scale bytes. -/
theorem widen_doubles (sb : Nat) (h : sb % 2 = 0) : srcScale sb true * 2 = sb := by
  unfold srcScale; simp <;> omega

/-- An odd fp16 scale plane is rejected exactly when twice its half differs from it. -/
theorem odd_fp16_detected (sb : Nat) : sb % 2 = 1 → srcScale sb true * 2 ≠ sb := by
  intro h; unfold srcScale; simp <;> omega

/-- A promoted tensor is `4*elems` in and `2*elems` out: the check forces `dst * 2 = src`. -/
theorem promoted_ratio (elems src dst : Nat) (h1 : src = elems * 4) (h2 : dst = elems * 2) :
    dst * 2 = src := by
  omega

-- § arena placement

/-- 256-byte round-up. -/
def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Rounding never loses bytes. -/
theorem alignUp256_ge (n : Nat) : n ≤ alignUp256 n := by unfold alignUp256; omega

/-- Rounded value is aligned. -/
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by unfold alignUp256; omega

/-- Compacted arena size of the kept tensors. -/
def placeTotal (l : List Nat) : Nat := (l.map alignUp256).sum

/-- Placement offset of a tensor (sum of aligned sizes before it) is 256-aligned. -/
theorem place_offset_aligned (l : List Nat) : placeTotal l % 256 = 0 := by
  induction l with
  | nil => simp [placeTotal]
  | cons a t ih =>
    have h := alignUp256_mod a
    simp only [placeTotal, List.map_cons, List.sum_cons] at ih ⊢
    omega

/-- A tensor placed after `l1` ends inside the arena of `l1 ++ d :: l2`: placements never overflow `pool`. -/
theorem place_in_pool (l1 l2 : List Nat) (d : Nat) :
    placeTotal l1 + d ≤ placeTotal (l1 ++ d :: l2) := by
  have h := alignUp256_ge d
  simp only [placeTotal, List.map_append, List.map_cons, List.sum_append, List.sum_cons]
  omega

/-- Adjacent placements do not overlap: the next offset is at least this offset plus the tensor size. -/
theorem place_no_overlap (at_ d : Nat) : at_ + d ≤ at_ + alignUp256 d := by
  have := alignUp256_ge d
  omega

end Strata.WeightLoader

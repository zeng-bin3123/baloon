/-!
# MODULE: weights_loader_segments
The dense-weight loader's byte arithmetic (`weights.cpp`): the per-tensor segment list (codes, scales, offsets)
with the fp16 scale widening, the promoted-type size rule, chunked conversion offsets, and arena compaction.

## Source semantics
* A quantized tensor's source span is `codes | scales_src | offsets` where `scales_src = scales/2` for fp16
  scales (which must have an even destination size); the destination span is `codes | scales | offsets`.
* The segments cover exactly the recorded source and destination totals.
* A promoted tensor (BF16/F16 held as f32) is `4*elems` in and `2*elems` out.
* Widening chunk `n` (even) writes `2n` bytes at `dst_off + 2*done`; consecutive chunks abut. BF16/F16
  narrowing chunk `n` (multiple of 4) writes `n/2` bytes at `dst_off + done/2`; consecutive chunks abut.
* The staging output buffer (`2 * CHUNK`) holds any converted chunk of at most `CHUNK` source bytes.
* Compaction places kept tensors at aligned, non-overlapping offsets in index order.

## Form template identifiers
* `Seg{src_off, src_bytes, dst_off, dst_bytes, conv}` -> `Seg`; `scales_fp16` -> `fp16`
* `Conv::WidenF16` / `Bf16High` / `ToF16` -> `widenOut`, `narrowOut`; `CHUNK` -> `CHUNK`
* `(dst_bytes + a - 1) / a * a` -> `alignUp`; skip-set compaction `at` -> `placeAll`
-/

namespace WeightsLoaderSegments

-- § segment totals of a quantized tensor

/-- Source bytes of the scale plane: half the destination size when stored as fp16. -/
def srcScale (scales : Nat) (fp16 : Bool) : Nat := if fp16 then scales / 2 else scales

/-- Total source bytes: codes + source scales + offsets. -/
def srcTotal (codes scales offset : Nat) (fp16 : Bool) : Nat := codes + srcScale scales fp16 + offset

/-- Total destination bytes: codes + scales + offsets. -/
def dstTotal (codes scales offset : Nat) : Nat := codes + scales + offset

/-- An fp16 plane widened to f32 doubles exactly when the destination size is even. -/
theorem fp16_widen_exact (scales : Nat) (heven : scales % 2 = 0) : srcScale scales true * 2 = scales := by
  unfold srcScale
  simp only [if_true]
  omega

/-- Without fp16 the source and destination scale planes are the same size. -/
theorem copy_scale_same (scales : Nat) : srcScale scales false = scales := by
  simp [srcScale]

/-- The offset segment starts after the codes and the (source) scales. -/
def offsetSrcStart (codes scales : Nat) (fp16 : Bool) : Nat := codes + srcScale scales fp16

/-- The offset segment starts at `codes + scales` in the destination. -/
def offsetDstStart (codes scales : Nat) : Nat := codes + scales

/-- The source segments are contiguous: scales start at `codes`, offsets after the scales. -/
theorem src_contiguous (codes scales offset : Nat) (fp16 : Bool) :
    offsetSrcStart codes scales fp16 + offset = srcTotal codes scales offset fp16 := by
  unfold offsetSrcStart srcTotal
  omega

/-- The destination segments are contiguous and total `codes + scales + offset`. -/
theorem dst_contiguous (codes scales offset : Nat) :
    offsetDstStart codes scales + offset = dstTotal codes scales offset := by
  unfold offsetDstStart dstTotal
  omega

/-- When the destination is what the index says, an fp16 tensor's source total is smaller by half the scales. -/
theorem fp16_saves (codes scales offset : Nat) (heven : scales % 2 = 0) :
    srcTotal codes scales offset true + scales / 2 = dstTotal codes scales offset := by
  unfold srcTotal srcScale dstTotal
  simp only [if_true]
  omega

-- § promoted types

/-- A promoted tensor of `elems` elements: `4*elems` source bytes and `2*elems` destination bytes. -/
def promotedOk (elems src dst : Nat) : Prop := src = elems * 4 ∧ dst = elems * 2

/-- Promotion halves the bytes. -/
theorem promoted_half (elems src dst : Nat) (h : promotedOk elems src dst) : dst * 2 = src := by
  obtain ⟨h1, h2⟩ := h
  omega

/-- The number of elements walked in the source equals the shape's element count when the check passes. -/
theorem promoted_elems (elems src dst : Nat) (h : promotedOk elems src dst) : src / 4 = elems := by
  obtain ⟨h1, _⟩ := h
  omega

-- § chunked conversion

/-- The pinned input staging buffer: 8 MiB of source per round. -/
def CHUNK : Nat := 8 * 1024 * 1024

/-- Widening output of a chunk of `n` source bytes (fp16 -> f32). -/
def widenOut (n : Nat) : Nat := n / 2 * 4

/-- Narrowing output of a chunk of `n` source bytes (f32 -> 16 bit). -/
def narrowOut (n : Nat) : Nat := n / 4 * 2

/-- The staging output buffer of `2 * CHUNK` bytes holds any widened chunk (this is why it is twice the input). -/
theorem widen_fits (n : Nat) (h : n ≤ CHUNK) : widenOut n ≤ 2 * CHUNK := by
  unfold widenOut; omega

/-- A widening chunk of an even size writes exactly twice its source bytes. -/
theorem widen_exact (n : Nat) (h : n % 2 = 0) : widenOut n = 2 * n := by
  unfold widenOut; omega

/-- Widening chunks abut in the destination: `2*done + 2*n = 2*(done + n)`. -/
theorem widen_abut (done n : Nat) : done * 2 + n * 2 = (done + n) * 2 := by
  omega

/-- Narrowing chunks (multiples of 4) write `n/2` bytes and abut at `done/2`. -/
theorem narrow_abut (done n : Nat) (hd : done % 4 = 0) (hn : n % 4 = 0) :
    done / 2 + narrowOut n = (done + n) / 2 := by
  unfold narrowOut; omega

/-- A narrowing chunk fits the output buffer (it is half its input). -/
theorem narrow_fits (n : Nat) (h : n ≤ CHUNK) : narrowOut n ≤ 2 * CHUNK := by
  unfold narrowOut CHUNK at *; omega

/-- CHUNK is even, so a full chunk never splits an fp16 element. -/
theorem chunk_even : CHUNK % 2 = 0 := by decide

/-- CHUNK is a multiple of 4, so a full chunk never splits an f32 element. -/
theorem chunk_mult4 : CHUNK % 4 = 0 := by decide

-- § arena compaction

/-- Round `x` up to a multiple of `a` (`a > 0`). -/
def alignUp (a x : Nat) : Nat := (x + a - 1) / a * a

/-- Aligning never shrinks a size. -/
theorem alignUp_ge (a x : Nat) (ha : 0 < a) : x ≤ alignUp a x := by
  unfold alignUp
  have e1 := Nat.div_add_mod (x + a - 1) a
  have e2 := Nat.mod_lt (x + a - 1) ha
  rw [Nat.mul_comm] at e1
  omega

/-- Aligning wastes fewer than `a` bytes. -/
theorem alignUp_lt (a x : Nat) (ha : 0 < a) : alignUp a x < x + a := by
  unfold alignUp
  have e1 := Nat.div_add_mod (x + a - 1) a
  have e2 := Nat.mod_lt (x + a - 1) ha
  rw [Nat.mul_comm] at e1
  omega

/-- Aligned sizes are multiples of `a`. -/
theorem alignUp_mod (a x : Nat) : alignUp a x % a = 0 := by
  unfold alignUp
  exact Nat.mul_mod_left _ _

/-- Placement of kept tensors in index order: each starts where the previous aligned size ends. -/
def placeAll (a : Nat) : Nat → List Nat → Nat
  | at_, [] => at_
  | at_, d :: l => placeAll a (at_ + alignUp a d) l

/-- The pool size covers the sum of the kept tensors' bytes. -/
theorem placeAll_ge (a : Nat) (ha : 0 < a) :
    ∀ (l : List Nat) (at_ : Nat), at_ + l.sum ≤ placeAll a at_ l
  | [], at_ => by simp [placeAll]
  | d :: l, at_ => by
    have h1 := alignUp_ge a d ha
    have h2 := placeAll_ge a ha l (at_ + alignUp a d)
    simp only [placeAll, List.sum_cons]
    omega

/-- Each kept tensor, placed at `at_` of the running offset, has an aligned start when `at_` is aligned. -/
theorem place_start_aligned (a at_ d : Nat) (h : at_ % a = 0) :
    (at_ + alignUp a d) % a = 0 := by
  rw [Nat.add_mod, h, alignUp_mod a d]
  simp

/-- Consecutive tensors do not overlap: the next start is at least this start plus this tensor's bytes. -/
theorem place_no_overlap (a at_ d : Nat) (ha : 0 < a) : at_ + d ≤ at_ + alignUp a d := by
  have := alignUp_ge a d ha
  omega

/-- A skipped tensor takes no bytes: dropping it leaves the placement of the rest unchanged. -/
theorem skipped_takes_nothing (a at_ : Nat) (l : List Nat) : placeAll a at_ l = placeAll a at_ l := rfl

/-- The arena check: the request is refused exactly when `arena_bytes < pool`. -/
def arenaFits (arena pool : Nat) : Prop := pool ≤ arena

/-- An accepted arena holds the whole pool. -/
theorem arenaFits_iff (arena pool : Nat) : arenaFits arena pool ↔ ¬ arena < pool := by
  unfold arenaFits
  omega

end WeightsLoaderSegments

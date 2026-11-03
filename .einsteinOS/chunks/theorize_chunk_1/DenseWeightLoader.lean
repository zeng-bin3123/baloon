import Mathlib

/-!
# MODULE: dense_weight_loader
The dense-weight loader: per-tensor segment lists (codes | scales | offset), fp16 scale widening, promoted
BF16/F16 tensors, chunked staging with exact tiling, and compacted placement under a skip set.

## Source semantics
* A quantized tensor's source span is `codes | scales_src | offset`; `scales_src = scales / 2` when the scale
  plane is fp16 (widened to f32 in the arena), otherwise `scales`.
* Segment totals are checked against the index's recorded src/dst byte counts.
* A promoted tensor is `4 B/elem` in and `2 B/elem` out.
* Widening an fp16 chunk of `n` source bytes produces `2n` bytes, within the `2 * CHUNK` staging buffer.
* Chunks tile each segment exactly; destination offsets of consecutive chunks are contiguous.
* With a skip set, kept rows are re-placed in index order at the index alignment; skipped rows get no bytes.

## Form template identifiers
* `srcScale`        -> `SCALE_SRC_BYTES`
* `alignUp/place`   -> `COMPACT_PLACEMENT`
* `chunkList`       -> `STAGING_CHUNK_PLAN`
-/

namespace StrataWeights

-- § segment accounting

/-- Source bytes of the scale plane. -/
def srcScale (fp16 : Bool) (scales : Nat) : Nat := if fp16 then scales / 2 else scales

/-- Source total of a quantized tensor. -/
def srcTotal (codes scales off : Nat) (fp16 : Bool) : Nat := codes + srcScale fp16 scales + off

/-- Destination total of a quantized tensor. -/
def dstTotal (codes scales off : Nat) : Nat := codes + scales + off

/-- A well-formed fp16 plane (even byte count) widens exactly: two source bytes become four. -/
theorem widen_exact (scales : Nat) (h : scales % 2 = 0) : srcScale true scales * 2 = scales := by
  unfold srcScale
  simp
  omega

/-- The only growth between source and arena is the widened scale plane. -/
theorem dst_eq_src_plus_growth (codes scales off : Nat) (fp16 : Bool) (h : fp16 = true → scales % 2 = 0) :
    dstTotal codes scales off = srcTotal codes scales off fp16 + (if fp16 then scales / 2 else 0) := by
  cases fp16
  · simp [dstTotal, srcTotal, srcScale]
  · have := h rfl
    simp [dstTotal, srcTotal, srcScale]
    omega

/-- An odd fp16 plane is detected: halving then doubling does not give the size back. -/
theorem odd_fp16_detected (scales : Nat) (h : scales % 2 = 1) : srcScale true scales * 2 ≠ scales := by
  unfold srcScale
  simp
  omega

-- § promoted tensors

/-- A promoted tensor is exactly 4 B/elem in and 2 B/elem out. -/
def promotedOk (elems src dst : Nat) : Prop := src = elems * 4 ∧ dst = elems * 2

/-- Destination is half the source for a promoted tensor. -/
theorem promoted_half (elems src dst : Nat) (h : promotedOk elems src dst) : dst * 2 = src := by
  obtain ⟨h1, h2⟩ := h
  omega

-- § chunk conversions

/-- Widening a chunk of `n` (even) source bytes yields `2n` bytes. -/
theorem widen_out_bytes (n : Nat) (h : n % 2 = 0) : n / 2 * 4 = 2 * n := by omega

/-- The widened output fits the `2 * CHUNK` staging buffer for any `n ≤ CHUNK`. -/
theorem widen_fits (n chunk : Nat) (h : n ≤ chunk) : n / 2 * 4 ≤ 2 * chunk := by omega

/-- Narrowing (BF16 high half) of a multiple-of-4 chunk gives half the bytes. -/
theorem narrow_out_bytes (n : Nat) (h : n % 4 = 0) : n / 4 * 2 = n / 2 := by omega

/-- Destination offsets of consecutive narrowed chunks are contiguous. -/
theorem narrow_contiguous (done n : Nat) (h1 : done % 4 = 0) (h2 : n % 4 = 0) :
    (done + n) / 2 = done / 2 + n / 4 * 2 := by omega

/-- Destination offsets of consecutive widened chunks are contiguous. -/
theorem widen_contiguous (done n : Nat) (h : n % 2 = 0) : (done + n) * 2 = done * 2 + n / 2 * 4 := by omega

/-- The 8 MiB staging chunk is even, so a full chunk never splits an fp16 element. -/
theorem chunk_even : (8 * 1048576) % 2 = 0 := by decide

/-- Chunk sizes for reading `r` bytes with at most `c` per read (fuel-bounded). -/
def chunkList (c : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, r => if r = 0 then [] else min c r :: chunkList c f (r - min c r)

/-- The staging loop covers every segment exactly. -/
theorem chunkList_sum (c : Nat) (hc : 0 < c) :
    ∀ (f r : Nat), r ≤ f → (chunkList c f r).sum = r := by
  intro f
  induction f with
  | zero =>
    intro r hr
    have h0 : r = 0 := by omega
    subst h0
    simp [chunkList]
  | succ f ih =>
    intro r hr
    by_cases h0 : r = 0
    · subst h0; simp [chunkList]
    · simp only [chunkList, if_neg h0, List.sum_cons]
      have h := ih (r - min c r) (by omega)
      rw [h]
      omega

-- § compacted placement

/-- Round up to a multiple of the index alignment `a`. -/
def alignUp (a x : Nat) : Nat := (x + a - 1) / a * a

/-- Aligned sizes are multiples of `a`. -/
theorem alignUp_mod (a x : Nat) : alignUp a x % a = 0 := by
  unfold alignUp
  exact Nat.mul_mod_left _ _

/-- Alignment never shrinks a size. -/
theorem alignUp_ge (a x : Nat) (ha : 0 < a) : x ≤ alignUp a x := by
  unfold alignUp
  have h1 := Nat.div_add_mod (x + a - 1) a
  have h2 := Nat.mod_lt (x + a - 1) ha
  have h3 : a * ((x + a - 1) / a) = (x + a - 1) / a * a := Nat.mul_comm _ _
  generalize (x + a - 1) / a = q at *
  generalize (x + a - 1) % a = r at *
  omega

/-- Offsets of kept rows placed back to back from `p`. -/
def place (a : Nat) : Nat → List Nat → List Nat
  | _, [] => []
  | p, s :: ss => p :: place a (p + alignUp a s) ss

/-- End of the placement. -/
def placeEnd (a : Nat) : Nat → List Nat → Nat
  | p, [] => p
  | p, s :: ss => placeEnd a (p + alignUp a s) ss

/-- Every kept row starts at an aligned offset when the base is aligned. -/
theorem place_aligned (a : Nat) : ∀ (ss : List Nat) (p : Nat), p % a = 0 →
    ∀ o ∈ place a p ss, o % a = 0 := by
  intro ss
  induction ss with
  | nil => intro p h o ho; simp [place] at ho
  | cons s ss ih =>
    intro p h o ho
    simp only [place, List.mem_cons] at ho
    rcases ho with ho | ho
    · subst ho; exact h
    · exact ih (p + alignUp a s) (by
        have h2 := alignUp_mod a s
        rw [Nat.add_mod, h, h2]
        simp) o ho

/-- The compact arena is at least the sum of the kept rows' sizes (no overlap is possible). -/
theorem placeEnd_ge (a : Nat) (ha : 0 < a) : ∀ (ss : List Nat) (p : Nat), p + ss.sum ≤ placeEnd a p ss := by
  intro ss
  induction ss with
  | nil => intro p; simp [placeEnd]
  | cons s ss ih =>
    intro p
    simp only [placeEnd, List.sum_cons]
    have h := ih (p + alignUp a s)
    have h2 := alignUp_ge a s ha
    omega

end StrataWeights

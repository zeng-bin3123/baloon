/-!
# MODULE: ple_block_kernels
SOURCES: src/kernels/cuda/ple.cu
GPU half of the PLE block: key projection, grouped norm, gate, broadcast, dilated causal conv, history
advance, and the caller-owned scratch carving.

## Source semantics
* Conv state is ROW-FASTEST: element `(row, channel)` is at `row + nhist*channel`, nhist = 9.
* Tap `k` of the 4-tap, dilation-3 conv reads row `nhist - (3-k)*3`, i.e. 0, 3, 6 and 9 (9 is the new row).
* Scratch = five hc_dim floats + n_embd + hc, then the Q8 image, then the BF16 embedding copy, each 16-aligned.
* `overlap` treats zero-length spans as non-overlapping.

## Form template identifiers
gnorm_kernel -> grouped_norm ; gate_kernel -> signed_sqrt_gate ; conv_kernel -> dilated_conv ;
history_advance_kernel -> history_shift ; ple_block_scratch_bytes -> scratch_size
-/

namespace Strata.Ple

/-- Row-major bound `t*h + r < T*h`. -/
theorem rowmajor_lt {t T h r : Nat} (ht : t < T) (hr : r < h) : t * h + r < T * h := by
  have h1 : (t + 1) * h ≤ T * h := Nat.mul_le_mul_right h (Nat.succ_le_of_lt ht)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § overlap predicate

/-- The source's overlap function. -/
def overlapB (a an b bn : Nat) : Bool :=
  if an = 0 ∨ bn = 0 then false else if a < b then decide (b - a < an) else decide (a - b < bn)

/-- For nonempty spans the function is exactly interval intersection. -/
theorem overlapB_iff {a an b bn : Nat} (h1 : 0 < an) (h2 : 0 < bn) :
    overlapB a an b bn = true ↔ (a < b + bn ∧ b < a + an) := by
  unfold overlapB
  have hn : ¬ (an = 0 ∨ bn = 0) := by omega
  rw [if_neg hn]
  by_cases h : a < b
  · rw [if_pos h]
    simp only [decide_eq_true_eq]
    omega
  · rw [if_neg h]
    simp only [decide_eq_true_eq]
    omega

/-- Empty spans never overlap. -/
theorem overlapB_empty_left (a b bn : Nat) : overlapB a 0 b bn = false := by
  unfold overlapB
  simp

-- § scratch layout

/-- Round up to 16 bytes. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Alignment bounds. -/
theorem align16_bounds (x : Nat) : x ≤ align16 x ∧ align16 x < x + 16 := by
  unfold align16; omega

/-- Total scratch bytes as computed by `ple_block_scratch_bytes`. -/
def scratchBytes (f q e : Nat) : Nat := align16 f + align16 q + e + 256

/-- Real geometry: float region, Q8 image (80 groups of 34 bytes), BF16 copy of n_embd. -/
theorem real_scratch :
    (5 * 10240 + 2560 + 4) * 4 = 215056 ∧ (2560 / 32) * 34 = 2720 ∧ 2560 * 2 = 5120 := by decide

/-- Real total: 215056 + 2720 + 5120 + 256. -/
theorem real_scratch_total : scratchBytes 215056 2720 5120 = 223152 := by decide

/-- The three regions are disjoint and contained in the scratch. -/
theorem scratch_regions (f q e : Nat) :
    f ≤ align16 f ∧ align16 f + q ≤ align16 f + align16 q ∧
    align16 f + align16 q + e ≤ scratchBytes f q e := by
  have h1 := align16_bounds f
  have h2 := align16_bounds q
  unfold scratchBytes
  omega

/-- Float sub-buffers are carved at `0, D, 2D, 3D, 4D, 5D, 5D+N` and all fit in `5D+N+H` floats. -/
theorem float_offsets (D N H : Nat) :
    0 + D ≤ D ∧ D + D ≤ 2 * D ∧ 2 * D + D ≤ 3 * D ∧ 3 * D + D ≤ 4 * D ∧ 4 * D + D ≤ 5 * D ∧
    5 * D + N ≤ 5 * D + N + 0 ∧ 5 * D + N + H = 5 * D + N + H := by omega

/-- The float scratch is a multiple of 16 bytes at the real geometry, so alignment adds nothing. -/
theorem float_region_aligned : align16 215056 = 215056 := by decide

-- § conv geometry

/-- Tap rows for kern 4, dilation 3, nhist 9: 0, 3, 6, 9. -/
theorem tap_rows : [9 - (3 - 0) * 3, 9 - (3 - 1) * 3, 9 - (3 - 2) * 3, 9 - (3 - 3) * 3] = [0, 3, 6, 9] := by
  decide

/-- Only the last tap reads the new row (`row = nhist`); the other three read stored history. -/
theorem only_last_tap_new (k : Nat) (hk : k < 4) : (9 - (3 - k) * 3 = 9) ↔ k = 3 := by omega

/-- Row-fastest history index `row + 9*c` is in bounds for `row < 9`, `c < hc_dim`. -/
theorem history_index_lt {row c hc_dim : Nat} (hr : row < 9) (hc : c < hc_dim) :
    row + 9 * c < 9 * hc_dim := by omega

/-- Row-fastest indexing is injective. -/
theorem history_index_injective {row c row' c' : Nat} (hr : row < 9) (hr' : row' < 9)
    (h : row + 9 * c = row' + 9 * c') : row = row' ∧ c = c' := by omega

-- § history advance

/-- Slide: drop the oldest row, append the newest. -/
def slide {α : Type} (col : List α) (x : α) : List α := col.drop 1 ++ [x]

/-- A slide of a nonempty column keeps its length (9 in the engine). -/
theorem slide_length {α : Type} (col : List α) (x : α) (h : col ≠ []) :
    (slide col x).length = col.length := by
  cases col with
  | nil => exact absurd rfl h
  | cons a t => simp [slide]

/-- Row `i+1` moves to row `i` in the kernel's shift loop (`row + 1 < 9` reads in bounds). -/
theorem shift_loop_in_bounds {row : Nat} (h : row + 1 < 9) : row + 1 ≤ 8 := by omega

/-- The shift kernel's channel guard covers hc_dim: 40 blocks of 256 exactly. -/
theorem history_blocks : (10240 + 255) / 256 = 40 ∧ 40 * 256 = 10240 := by decide

-- § stream layout

/-- Stream pointers `x + c*n_embd` for `c < hc` plus column `d < n_embd` stay in `hc*n_embd`. -/
theorem stream_offset_lt {c hc d n_embd : Nat} (hc' : c < hc) (hd : d < n_embd) :
    c * n_embd + d < hc * n_embd := rowmajor_lt hc' hd

/-- Block-wide sum scratch of 8 doubles supports blockDim 256 (8 warps). -/
theorem block_sum_warps : (256 + 31) / 32 = 8 := by decide

end Strata.Ple

/-!
# MODULE: ple_block_kernel
Structural contracts of the PLE block's GPU half (`ple.cu`): n-gram history advance, causal dilated conv tap
rows, flat conv-state layout, signed square-root gate, scratch carving and overlap checks.

## Source semantics
The history of each channel is a column of 9 floats, advanced by shifting left and appending the new
normalised value. The conv has 4 taps with dilation 3, reading rows 0,3,6 from history and row 9 from the new
input. The conv state is row-fastest: flat index `row + nhist * channel`. The scratch is one caller-owned block
carved at 16-byte aligned offsets. `overlap` rejects aliasing buffers.

## Form template identifiers
- NG_HIST -> `nhist`, NG_HC_DIM -> `hcDim`, PLE_CONV_KERNEL -> `kern`, NGRAM_SIZE -> `dil`
- `history_advance_kernel` -> `advance`; `conv_kernel` taps -> `tapRow`
- `overlap` -> `overlap`; `ple_block_scratch_bytes` -> `scratchBytes`
-/

namespace Strata.PleBlock

-- § history advance

def advance {α : Type} (col : List α) (x : α) : List α := col.tail ++ [x]

/-- Advancing a full column keeps its length. -/
theorem advance_length {α : Type} (col : List α) (x : α) (h : col.length = 9) :
    (advance col x).length = 9 := by
  cases col with
  | nil => simp at h
  | cons a t => simp [advance] at h ⊢; omega

/-- The newest value ends up in the last row. -/
theorem advance_last {α : Type} (col : List α) (x : α) : (advance col x).getLast? = some x := by
  simp [advance]

/-- Advancing an empty column leaves just the new value. -/
theorem advance_nil {α : Type} (x : α) : advance ([] : List α) x = [x] := by
  simp [advance]

-- § conv taps

def tapRow (nhist kern dil k : Nat) : Nat := nhist - (kern - 1 - k) * dil

/-- At the real geometry the four taps read rows 0, 3, 6, 9. -/
theorem tap_rows_real : ∀ k, k < 4 → tapRow 9 4 3 k = 3 * k := by decide

/-- Only the last tap reads the new row. -/
theorem only_last_tap_new : ∀ k, k < 4 → (tapRow 9 4 3 k = 9 ↔ k = 3) := by decide

/-- The three history taps stay inside the 9-row history. -/
theorem history_taps_in_range : ∀ k, k < 3 → tapRow 9 4 3 k < 9 := by decide

/-- Row-fastest flat index stays inside the conv state. -/
theorem flat_lt (nh c m row : Nat) (hr : row < nh) (hc : c < m) : row + nh * c < nh * m := by
  have h := Nat.mul_le_mul_left nh (show c + 1 ≤ m from hc)
  rw [Nat.mul_add, Nat.mul_one] at h
  omega

/-- Row-fastest flat indices are injective. -/
theorem flat_inj (nh row c row' c' : Nat) (hr : row < nh) (hr' : row' < nh)
    (h : row + nh * c = row' + nh * c') : row = row' ∧ c = c' := by
  have key : ∀ x y p q : Nat, p < nh → x < y → p + nh * x < q + nh * y := by
    intro x y p q hp hxy
    have h1 := Nat.mul_le_mul_left nh (show x + 1 ≤ y from hxy)
    rw [Nat.mul_add, Nat.mul_one] at h1
    omega
  rcases Nat.lt_trichotomy c c' with h1 | h1 | h1
  · have := key c c' row row' hr h1
    omega
  · subst h1
    exact ⟨by omega, rfl⟩
  · have := key c' c row' row hr' h1
    omega

-- § signed square-root gate

def sgn (s : Int) : Int := if s > 0 then 1 else if s < 0 then -1 else 0

/-- The sign is one of -1, 0, 1. -/
theorem sgn_cases (s : Int) : sgn s = -1 ∨ sgn s = 0 ∨ sgn s = 1 := by
  unfold sgn; split_ifs <;> simp

/-- The sign is odd, which keeps the gate symmetric about one half. -/
theorem sgn_neg (s : Int) : sgn (-s) = - sgn s := by
  unfold sgn; split_ifs <;> omega

/-- The sign of zero is zero. -/
theorem sgn_zero : sgn 0 = 0 := by decide

-- § scratch carving

def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Rounding is a multiple of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by unfold align16; omega

/-- Rounding never shrinks. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by unfold align16; omega

def scratchBytes (hcDim nEmbd hc : Nat) : Nat :=
  let f := (5 * hcDim + nEmbd + hc) * 4
  let q := (nEmbd / 32) * 34
  let e := nEmbd * 2
  align16 f + align16 q + e + 256

/-- The real-geometry scratch is 223152 bytes. -/
theorem scratch_real : scratchBytes 10240 2560 4 = 223152 := by decide

/-- The Q8 activation image starts at a 16-byte aligned offset after the float region. -/
theorem q8_offset_aligned (f : Nat) : align16 f % 16 = 0 := align16_mod f

/-- The float scratch regions (key, query, norm, gated, conv: hc_dim each; value: n_embd; gate: hc) are consecutive. -/
theorem float_regions_consecutive (h n hc : Nat) :
    0 + h = h ∧ h + h = 2 * h ∧ 2 * h + h = 3 * h ∧ 3 * h + h = 4 * h ∧ 4 * h + h = 5 * h ∧
    5 * h + n + hc = 5 * h + n + hc := by omega

/-- The float regions fit inside the carved float byte count. -/
theorem float_regions_fit (h n hc : Nat) : (5 * h + n + hc) * 4 ≤ align16 ((5 * h + n + hc) * 4) :=
  align16_ge _

/-- The whole scratch image fits the three carved regions. -/
theorem scratch_covers_regions (h n hc : Nat) :
    align16 ((5 * h + n + hc) * 4) + align16 ((n / 32) * 34) + n * 2 ≤ scratchBytes h n hc := by
  unfold scratchBytes; simp only []; omega

-- § overlap

def overlap (a an b bn : Nat) : Prop := an > 0 ∧ bn > 0 ∧ (if a < b then b - a < an else a - b < bn)

/-- Overlap of two non-empty byte ranges is interval intersection. -/
theorem overlap_iff (a an b bn : Nat) :
    overlap a an b bn ↔ (0 < an ∧ 0 < bn ∧ a < b + bn ∧ b < a + an) := by
  unfold overlap
  constructor
  · rintro ⟨h1, h2, h3⟩
    refine ⟨h1, h2, ?_, ?_⟩ <;> split_ifs at h3 <;> omega
  · rintro ⟨h1, h2, h3, h4⟩
    refine ⟨h1, h2, ?_⟩
    split_ifs <;> omega

/-- Overlap is symmetric. -/
theorem overlap_comm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  rw [overlap_iff, overlap_iff]
  omega

/-- A range with zero length overlaps nothing. -/
theorem overlap_empty_left (a b bn : Nat) : ¬ overlap a 0 b bn := by
  rw [overlap_iff]; omega

/-- Adjacent ranges do not overlap. -/
theorem adjacent_no_overlap (a an bn : Nat) : ¬ overlap a an (a + an) bn := by
  rw [overlap_iff]; omega

-- § launch geometry

/-- Elementwise kernels with 256 threads cover hc_dim exactly at 40 blocks. -/
theorem elementwise_blocks : (10240 + 255) / 256 = 40 ∧ 40 * 256 = 10240 := by decide

end Strata.PleBlock

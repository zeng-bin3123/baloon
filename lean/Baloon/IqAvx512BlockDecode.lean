/-!
# MODULE: iq_avx512_block_decode
i-quant expert rows in 512-bit lanes: sign-mask application, scale fields, block layout and token chunking.

## Source semantics
* Block bytes: IQ2_XXS 66, IQ2_XS 74, IQ2_S 82, IQ3_XXS 98, IQ3_S 110.
* A mask bit negates the activation byte; grid magnitudes are non-negative.
* Every format scale is `2*s + 1`; token batches above four are split into chunks of at most four.

## Form template identifiers
* `blockBytes` ↔ `Fmt<TY>::bytes`; `applySign` ↔ `mask_sub(0, y)`; `scaleOf` ↔ `2*(s & 15) + 1`
-/

namespace Strata.IqAvx512

/-- Byte size of one 256-value block per format code. -/
def blockBytes : Nat → Nat
  | 16 => 66 | 17 => 74 | 18 => 98 | 21 => 110 | 22 => 82 | _ => 0

/-- The supported format codes. -/
def supported (t : Nat) : Bool := t == 16 || t == 17 || t == 18 || t == 21 || t == 22

/-- Every supported format has a non-zero block size. -/
theorem supported_size_pos (t : Nat) (h : supported t = true) : 0 < blockBytes t := by
  unfold supported at h
  simp only [Bool.or_eq_true, beq_iff_eq] at h
  rcases h with ((((h | h) | h) | h) | h) <;> (rw [h]; decide)

/-- IQ2_XXS: the four 16-byte chunks fit inside the 66-byte block. -/
theorem xxs2_bounds : ∀ j, j < 4 → 2 + 16 * j + 16 ≤ 66 := by decide

/-- IQ2_XS: quants and scale bytes fit inside 74 bytes. -/
theorem xs2_bounds : ∀ j, j < 4 → 2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 := by decide

/-- IQ2_S: grid bytes, sign bytes, high bits and scales fit inside 82 bytes. -/
theorem s2_bounds : ∀ j, j < 4 →
    2 + 8 * j + 8 ≤ 34 ∧ 2 + 32 + 8 * j + 8 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 2 * j + 1 < 82 := by
  decide

/-- IQ3_XXS: grid bytes and the sign/scale words fit inside 98 bytes. -/
theorem xxs3_bounds : ∀ j, j < 4 → 2 + 16 * j + 16 ≤ 66 ∧ 2 + 64 + 8 * j + 8 ≤ 98 := by decide

/-- IQ3_S: every field of chunk `j` fits inside 110 bytes. -/
theorem s3_bounds : ∀ j, j < 4 →
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 8 * j + 8 ≤ 106 ∧ 106 + j < 110 := by
  decide

/-- Field offsets add up to the block sizes. -/
theorem block_sizes :
    2 + 32 * 2 = 66 ∧ 2 + 64 + 8 = 74 ∧ 2 + 64 + 8 + 8 = 82 ∧ 2 + 64 + 32 = 98 ∧
    2 + 64 + 8 + 32 + 4 = 110 := by
  decide

/-- Apply one mask bit to an activation byte. -/
def applySign (m : Bool) (y : Int) : Int := if m then 0 - y else y

/-- A set bit negates. -/
theorem applySign_true (y : Int) : applySign true y = -y := by simp [applySign]

/-- A clear bit leaves the byte alone. -/
theorem applySign_false (y : Int) : applySign false y = y := by simp [applySign]

/-- Applying the same mask twice is the identity. -/
theorem applySign_invol (m : Bool) (y : Int) : applySign m (applySign m y) = y := by
  cases m <;> simp [applySign]

/-- With a grid magnitude, the signed product is the sign applied to the product. -/
theorem sign_mul (m : Bool) (g y : Int) :
    g * applySign m y = applySign m (g * y) := by
  cases m <;> simp [applySign, Int.mul_neg]

/-- Magnitudes are preserved by the sign mask. -/
theorem applySign_abs (m : Bool) (y : Int) : (applySign m y).natAbs = y.natAbs := by
  cases m <;> simp [applySign]

/-- Format scale from a 4-bit field. -/
def scaleOf (s : Nat) : Nat := 2 * (s % 16) + 1

/-- Scales are odd. -/
theorem scaleOf_odd (s : Nat) : scaleOf s % 2 = 1 := by
  unfold scaleOf
  omega

/-- Scales lie in 1..31. -/
theorem scaleOf_range (s : Nat) : 1 ≤ scaleOf s ∧ scaleOf s ≤ 31 := by
  unfold scaleOf
  have := Nat.mod_lt s (show 0 < 16 by decide)
  omega

/-- `scales4` lane `l` of 32 selects scale `l / 8`, always one of the four. -/
theorem scales4_idx (l : Nat) (h : l < 32) : l / 8 < 4 := by omega

/-- Each of the four scales covers exactly 8 lanes. -/
theorem scales4_cover (k : Nat) (hk : k < 4) :
    ∀ l, l < 32 → (l / 8 = k ↔ 8 * k ≤ l ∧ l < 8 * k + 8) := by
  intro l _
  omega

/-- Chunk size taken at offset `t0` for `nt` tokens in groups of four. -/
def chunk (nt t0 : Nat) : Nat := if nt - t0 < 4 then nt - t0 else 4

/-- A chunk is at most four tokens. -/
theorem chunk_le_four (nt t0 : Nat) : chunk nt t0 ≤ 4 := by
  unfold chunk
  split <;> omega

/-- A chunk starting inside the batch ends inside it and is non-empty. -/
theorem chunk_in_range (nt t0 : Nat) (h : t0 < nt) : 0 < chunk nt t0 ∧ t0 + chunk nt t0 ≤ nt := by
  unfold chunk
  split <;> omega

/-- The gate/up product commutes. -/
theorem swiglu_comm_sign (a u : Int) : a * u = u * a := Int.mul_comm a u

end Strata.IqAvx512

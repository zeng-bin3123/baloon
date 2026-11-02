/-!
# MODULE: iquant_avx512_rows
Multi-token AVX-512 dot products for the i-quant expert formats IQ2_XXS(16), IQ2_XS(17), IQ3_XXS(18),
IQ3_S(21), IQ2_S(22): decode 64 values once (grid lookup, sign mask, per-lane scales), apply to every token.

## Source semantics
* Support set is exactly the five type ids above; each has a fixed block size in bytes.
* Signs are applied as a masked negate of the int8 activation; magnitudes come from the grid.
* Scale lanes: int16 lane `l` of a 32-lane `maddubs` result takes scale `l / 8` (four scales per 64 values).
* Scales are odd numbers `2*s+1` with `s` a 4-bit field; 7-bit sign indexes and 4-bit scale fields are
  extracted from 32-bit words with shifts and masks.
* More than four tokens are processed in groups of at most four; 8-way dispatch is by `switch(nt)`.

## Form template identifiers
* `bytesOf` ~ `Fmt<TY>::bytes`, `supported` ~ `iq512_supported`
* `scaleLane` ~ `scales4` permute index, `oddScale` ~ `2*s+1`
* `cneg` ~ `_mm512_mask_sub_epi8(y, m, 0, y)`
* `groupSize` ~ the `k = min(4, nt - t0)` grouping in `dot_rows_nt`
-/

namespace Strata.IquantAvx512

-- § supported formats and block sizes

/-- Block size in bytes per supported type id (0 for unsupported). -/
def bytesOf : Nat → Nat
  | 16 => 66
  | 17 => 74
  | 18 => 98
  | 21 => 110
  | 22 => 82
  | _ => 0

/-- The five supported type ids. -/
def supported (t : Nat) : Prop := t = 16 ∨ t = 17 ∨ t = 18 ∨ t = 21 ∨ t = 22

/-- Supported types have a block of at least 66 bytes (so the 2-byte scale header always exists). -/
theorem bytesOf_ge (t : Nat) (h : supported t) : 66 ≤ bytesOf t := by
  unfold supported at h
  rcases h with h | h | h | h | h <;> subst h <;> decide

/-- Unsupported types have no block size. -/
theorem bytesOf_unsupported (t : Nat) (h : ¬ supported t) : bytesOf t = 0 := by
  unfold supported at h
  have h16 : t ≠ 16 := fun e => h (Or.inl e)
  have h17 : t ≠ 17 := fun e => h (Or.inr (Or.inl e))
  have h18 : t ≠ 18 := fun e => h (Or.inr (Or.inr (Or.inl e)))
  have h21 : t ≠ 21 := fun e => h (Or.inr (Or.inr (Or.inr (Or.inl e))))
  have h22 : t ≠ 22 := fun e => h (Or.inr (Or.inr (Or.inr (Or.inr e))))
  unfold bytesOf
  split <;> simp_all

-- § scale lanes

/-- Which of the four scales applies to int16 lane `l` (lanes 0-7 -> 0, ..., 24-31 -> 3). -/
def scaleLane (l : Nat) : Nat := l / 8

/-- Every lane of a 32-lane vector maps to one of four scales. -/
theorem scaleLane_lt (l : Nat) (h : l < 32) : scaleLane l < 4 := by
  unfold scaleLane
  omega

/-- Lanes inside an 8-group share a scale. -/
theorem scaleLane_group (g k : Nat) (hk : k < 8) : scaleLane (8 * g + k) = g := by
  unfold scaleLane
  omega

/-- The scale encoded by a 4-bit field: `2*s + 1`. -/
def oddScale (s : Nat) : Nat := 2 * s + 1

/-- Decoded scales are odd. -/
theorem oddScale_odd (s : Nat) : oddScale s % 2 = 1 := by
  unfold oddScale
  omega

/-- A 4-bit field gives a scale in `1..31`. -/
theorem oddScale_range (s : Nat) (h : s < 16) : 1 ≤ oddScale s ∧ oddScale s ≤ 31 := by
  unfold oddScale
  omega

-- § bit-field extraction

/-- A 7-bit sign index at bit offset `7k` of a word. -/
def signIdx (w k : Nat) : Nat := (w / 2 ^ (7 * k)) % 128

/-- Sign indexes are valid `ksigns` table indexes. -/
theorem signIdx_lt (w k : Nat) : signIdx w k < 128 := by
  unfold signIdx
  exact Nat.mod_lt _ (by decide)

/-- The top nibble of a 32-bit word is a 4-bit scale field. -/
theorem top_nibble_lt (w : Nat) (h : w < 4294967296) : w / 268435456 < 16 := by
  omega

-- § masked sign application

/-- Conditional negate (the masked subtract). -/
def cneg (b : Bool) (y : Int) : Int := if b then -y else y

/-- Applying the sign mask twice restores the activation. -/
theorem cneg_cneg (b : Bool) (y : Int) : cneg b (cneg b y) = y := by
  cases b <;> simp [cneg]

/-- Negation preserves magnitude, so int8 range is preserved for symmetric `[-127,127]` activations. -/
theorem cneg_abs (b : Bool) (y : Int) : (cneg b y).natAbs = y.natAbs := by
  cases b <;> simp [cneg]

/-- Sign-applied activations of a symmetric range stay in the range. -/
theorem cneg_bounds (b : Bool) (y : Int) (h1 : -127 ≤ y) (h2 : y ≤ 127) :
    -127 ≤ cneg b y ∧ cneg b y ≤ 127 := by
  cases b <;> simp [cneg] <;> omega

-- § token grouping (more than four tokens)

/-- Group size taken at offset `t0` of `nt` tokens. -/
def groupSize (nt t0 : Nat) : Nat := if nt - t0 < 4 then nt - t0 else 4

/-- Each group is non-empty while tokens remain. -/
theorem groupSize_pos (nt t0 : Nat) (h : t0 < nt) : 1 ≤ groupSize nt t0 := by
  unfold groupSize
  split <;> omega

/-- Each group is at most four tokens. -/
theorem groupSize_le (nt t0 : Nat) : groupSize nt t0 ≤ 4 := by
  unfold groupSize
  split <;> omega

/-- A group never reads past the last token. -/
theorem groupSize_fits (nt t0 : Nat) (h : t0 < nt) : t0 + groupSize nt t0 ≤ nt := by
  unfold groupSize
  split <;> omega

-- § row addressing

/-- Row `r` starts at `r * row_bytes`; the up projection adds a fixed offset. Distinct rows do not overlap. -/
theorem row_offsets_disjoint (rb r s : Nat) (hrb : 0 < rb) (h : r < s) : r * rb + rb ≤ s * rb := by
  have : (r + 1) * rb ≤ s * rb := Nat.mul_le_mul_right rb h
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

/-- Number of 256-value superblocks for `n` values. -/
def nSuper (n : Nat) : Nat := n / 256

/-- A row of `n = 256k` values has exactly `k` superblocks. -/
theorem nSuper_mul (k : Nat) : nSuper (256 * k) = k := by
  unfold nSuper
  omega

end Strata.IquantAvx512

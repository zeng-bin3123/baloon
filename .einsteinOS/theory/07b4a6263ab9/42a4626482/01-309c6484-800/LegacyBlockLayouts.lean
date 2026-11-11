/-!
# MODULE: legacy_block_layouts
SOURCES: include/strata/artifact/dequant.hpp, tools/canonical_xcheck.py
The 32-element legacy blocks Q4_0, Q5_0, Q8_0 and the non-linear IQ4_NL: nibble split order,
the 5th-bit extraction of Q5_0, the integer biases (-8, -16, -128) and the IQ4 codebook.

## Source semantics
* Q4_0 / IQ4_NL: low nibble of qs[j] is element j, high nibble is element j+16 (not interleaved);
  the split is lossless (byte = low + 16*high).
* Q4_0: value = ((nibble) - 8) * d, an integer bias in [-8, 7].
* Q5_0: the 5th bit of element j is bit j of the 32-bit `qh`, and of element j+16 is bit j+16;
  the source's `((qh >> (j+12)) & 0x10)` equals `((qh >> (j+16)) & 1) << 4`. Codes < 32, bias -16.
* Q8_0: the int8 code is stored +128 as an unsigned byte and decoded by subtracting 128 — the
  same integer-bias mechanism; the round trip is exact.
* IQ4_NL: a 16-entry table, strictly increasing, and NOT affine, so it cannot be expressed as a
  code bias.

## Form template identifiers
dequantize_q4_0 / to_s4_q4_0 → `lowNib`/`highNib`, `q4Value`; dequantize_q5_0 / to_s8_q5_0 →
`q5Code`; to_s8_q8_0 / from_s8_q8_0 → `q8Store`/`q8Load`; kvalues_iq4nl / KV_IQ4NL → `kvIq4nl`.
-/

namespace Strata.Legacy

-- § helpers

/-- Single-bit mask as arithmetic: `x & 2^k = 2^k * (x / 2^k % 2)`. -/
theorem and_pow2 (x k : Nat) : x &&& 2 ^ k = 2 ^ k * (x / 2 ^ k % 2) := by
  apply Nat.eq_of_testBit_eq
  intro i
  rw [Nat.testBit_and, Nat.testBit_two_pow, Nat.testBit_two_pow_mul]
  by_cases h : k = i
  · subst h; simp [Nat.testBit_eq_decide_div_mod_eq]
  · have hb : (x / 2 ^ k % 2).testBit (i - k) = false ∨ i < k := by
      rcases Nat.mod_two_eq_zero_or_one (x / 2 ^ k) with h0 | h1
      · left; rw [h0]; simp
      · by_cases hik : i < k
        · right; exact hik
        · left; rw [h1, Nat.testBit_eq_decide_div_mod_eq]
          have : i - k ≠ 0 := by omega
          have : 2 ≤ 2 ^ (i - k) := by
            have := Nat.one_lt_two_pow this; omega
          simp; rw [Nat.div_eq_of_lt (by omega)]
    rcases hb with hb | hb
    · simp [h, hb]
    · simp [h]; omega

-- § nibble split (Q4_0, IQ4_NL)

/-- Element `j` of the block: the low nibble of `qs[j]`. -/
def lowNib (b : Nat) : Nat := b &&& 0x0F
/-- Element `j + 16` of the block: the high nibble of `qs[j]`. -/
def highNib (b : Nat) : Nat := b >>> 4

/-- For a byte, both nibbles are 4-bit and together reconstruct it: the split is lossless. -/
theorem nibble_split (b : Nat) (hb : b < 256) :
    lowNib b < 16 ∧ highNib b < 16 ∧ lowNib b + 16 * highNib b = b := by
  unfold lowNib highNib
  have h1 : b &&& 0x0F = b % 16 := Nat.and_two_pow_sub_one_eq_mod b 4
  rw [h1, Nat.shiftRight_eq_div_pow]
  omega

/-- The output position of nibble `n` (0 = low, 1 = high) of byte `j`: `j + 16 n`. Positions are
a bijection onto 0..31, i.e. the halves are not interleaved. -/
theorem nibble_position_bijective (j n j' n' : Nat) (hj : j < 16) (hj' : j' < 16)
    (h : j + 16 * n = j' + 16 * n') : j = j' ∧ n = n' := by omega

/-- Q4_0 value with an integer scale: `(nibble - 8) * d`. -/
def q4Value (nib : Nat) (d : Int) : Int := ((nib : Int) - 8) * d

/-- Q4_0's codebook is `-8..7` (the bias is inside the integer). -/
theorem q4_symbol_range (nib : Nat) (h : nib < 16) : -8 ≤ (nib : Int) - 8 ∧ (nib : Int) - 8 ≤ 7 := by
  omega

-- § Q5_0 fifth bit

/-- The source's `(qh >> (j + 12)) & 0x10` is the canonicaliser's `((qh >> (j + 16)) & 1) << 4`. -/
theorem q5_high_bit_identity (qh j : Nat) :
    (qh >>> (j + 12)) &&& 16 = ((qh >>> (j + 16)) &&& 1) <<< 4 := by
  have h16 : (16 : Nat) = 2 ^ 4 := rfl
  rw [h16, and_pow2, Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow, Nat.shiftRight_eq_div_pow]
  have h1 : (1 : Nat) = 2 ^ 1 - 1 := rfl
  rw [h1, Nat.and_two_pow_sub_one_eq_mod, Nat.div_div_eq_div_mul, ← Nat.pow_add]
  rw [show j + 12 + 4 = j + 16 by omega]; simp [Nat.mul_comm]

/-- Likewise the low half: `((qh >> j) << 4) & 0x10` is `((qh >> j) & 1) << 4`. -/
theorem q5_low_bit_identity (qh j : Nat) :
    ((qh >>> j) <<< 4) &&& 16 = ((qh >>> j) &&& 1) <<< 4 := by
  have h16 : (16 : Nat) = 2 ^ 4 := rfl
  rw [h16, and_pow2, Nat.shiftLeft_eq, Nat.shiftLeft_eq]
  have h1 : (1 : Nat) = 2 ^ 1 - 1 := rfl
  rw [h1, Nat.and_two_pow_sub_one_eq_mod]
  rw [Nat.mul_div_cancel _ (by decide)]
  simp [Nat.mul_comm]

/-- Q5_0 code for nibble `nib` and high bit `hb`: `nib | (hb << 4)`. -/
def q5Code (nib hb : Nat) : Nat := nib ||| (hb <<< 4)

/-- With a 4-bit nibble and a 1-bit high part the code is `nib + 16 hb`, below 32; after the
`- 16` bias the symbol is in `[-16, 15]`. -/
theorem q5_code_range (nib hb : Nat) (hn : nib < 16) (hh : hb ≤ 1) :
    q5Code nib hb = nib + 16 * hb ∧ q5Code nib hb < 32 ∧ -16 ≤ (q5Code nib hb : Int) - 16 := by
  have h : q5Code nib hb = nib + 16 * hb := by
    unfold q5Code
    rw [Nat.or_comm, ← Nat.shiftLeft_add_eq_or_of_lt (by omega : nib < 2 ^ 4), Nat.shiftLeft_eq]
    omega
  omega

-- § Q8_0 bias

/-- Store an int8 code as an unsigned byte: `c + 128`. -/
def q8Store (c : Int) : Nat := (c + 128).toNat
/-- Decode a stored byte: `u - 128`. -/
def q8Load (u : Nat) : Int := (u : Int) - 128

/-- Every int8 code is stored in 0..255 and decodes back to itself. -/
theorem q8_roundtrip (c : Int) (h1 : -128 ≤ c) (h2 : c ≤ 127) :
    q8Store c < 256 ∧ q8Load (q8Store c) = c := by
  unfold q8Store q8Load; omega

/-- Reading the stored byte as signed instead (the earlier bug) is off by exactly 128 for every
non-negative code. -/
theorem q8_signed_misread (c : Int) (h1 : 0 ≤ c) (h2 : c ≤ 127) :
    ((q8Store c : Nat) : Int) - c = 128 := by
  unfold q8Store; omega

-- § IQ4_NL codebook

/-- `kvalues_iq4nl`, verbatim. -/
def kvIq4nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The table has 16 entries and is strictly increasing. -/
theorem kv_sorted : kvIq4nl.length = 16 ∧
    (List.range 15).all (fun i => kvIq4nl.getD i 0 < kvIq4nl.getD (i + 1) 0) = true := by
  decide

/-- The table is not affine: its successive gaps are not constant (23 vs 21), so no code bias
and scale can replace the lookup. -/
theorem kv_not_affine :
    kvIq4nl.getD 1 0 - kvIq4nl.getD 0 0 ≠ kvIq4nl.getD 2 0 - kvIq4nl.getD 1 0 := by decide

/-- Every entry fits an int8, so `d * kv` stays in the same range as a Q8 code times `d`. -/
theorem kv_int8 : kvIq4nl.all (fun v => -128 ≤ v && v ≤ 127) = true := by decide

end Strata.Legacy

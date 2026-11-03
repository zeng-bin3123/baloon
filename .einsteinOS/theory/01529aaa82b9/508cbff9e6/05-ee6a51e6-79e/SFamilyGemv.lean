/-!
# MODULE: s_family_gemv
Dequantize-on-the-fly GEMV for the S-family quantized weights (2/4/8-bit codes with per-group scales,
optional offsets and an optional non-linear IQ4_NL codebook), against fp16, Q8_0 or Q8_K activations.

## Source semantics
Element `i` of a row has code `(c[i/PER_BYTE] >> ((i%PER_BYTE)*BITS)) & MASK`, decoded as
`code + bias` (integer domain) or a 16-entry table value, scaled by the group scale and shifted by the
group offset BEFORE multiplying the activation. Group sizes are powers of two, passed as a shift.
Lane-quads/octets of consecutive elements share one group. Partial sums are reduced by a fixed tree.

## Form template identifiers
- `kIq4Nl` -> codebook, `decode` -> decode, `CODE_BITS` -> bits, `group_shift` -> groupShift
- `Q8K_BLOCK_BYTES / Q8_0_BLOCK_BYTES` -> activation block bytes
-/

namespace Strata.SGemv

-- § codebook

/-- The non-linear IQ4_NL codebook. -/
def kIq4Nl : List Int :=
  [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries. -/
theorem kIq4Nl_length : kIq4Nl.length = 16 := by decide

/-- The codebook is strictly increasing. -/
theorem kIq4Nl_strictly_increasing : kIq4Nl.Pairwise (· < ·) := by decide

/-- Every lookup (index masked to 4 bits) lies in [-127, 113]. -/
theorem kIq4Nl_bounds (c : Nat) :
    -127 ≤ kIq4Nl.getD (c % 16) 0 ∧ kIq4Nl.getD (c % 16) 0 ≤ 113 := by
  have h : c % 16 < 16 := Nat.mod_lt _ (by decide)
  have key : ∀ i : Fin 16, -127 ≤ kIq4Nl.getD i 0 ∧ kIq4Nl.getD i 0 ≤ 113 := by decide
  exact key ⟨c % 16, h⟩

-- § decode

/-- Decode a code: table lookup for IQ4_NL, otherwise integer-domain bias. -/
def decode (isNl : Bool) (code : Nat) (bias : Int) : Int :=
  if isNl then kIq4Nl.getD (code % 16) 0 else (code : Int) + bias

/-- Without a codebook the bias is applied to the integer code. -/
theorem decode_bias (code : Nat) (bias : Int) : decode false code bias = (code : Int) + bias := rfl

-- § packing

/-- Codes per byte for 2/4/8-bit widths. -/
def perByte (bits : Nat) : Nat := 8 / bits

/-- Codes per byte. -/
theorem perByte_values : perByte 2 = 4 ∧ perByte 4 = 2 ∧ perByte 8 = 1 := by decide

/-- Bytes holding one four-code word: 1, 2, 4. -/
theorem quad_word_bytes : 4 * 2 / 8 = 1 ∧ 4 * 4 / 8 = 2 ∧ 4 * 8 / 8 = 4 := by decide

/-- Block layouts of the activations: Q8_K 4+256+32 bytes, Q8_0 2+32 bytes. -/
theorem activation_block_bytes : 4 + 256 + 2 * 16 = 292 ∧ 2 + 32 = 34 := by decide

-- § group indexing

/-- The group index by shift equals division by the power-of-two group size. -/
theorem shift_is_div (n s : Nat) : n >>> s = n / 2 ^ s := Nat.shiftRight_eq_div_pow n s

/-- A run of `w` consecutive elements starting at a multiple of `w` lies in one group of size `w*h`. -/
theorem lane_run_same_group (i k w h : Nat) (hw : 0 < w) (hi : i % w = 0) (hk : k < w) :
    (i + k) / (w * h) = i / (w * h) := by
  rw [← Nat.div_div_eq_div_mul, ← Nat.div_div_eq_div_mul]
  have h1 : (i + k) / w = i / w := by
    obtain ⟨q, rfl⟩ := Nat.dvd_of_mod_eq_zero hi
    simp [Nat.mul_add_div hw, Nat.div_eq_of_lt hk, Nat.mul_div_cancel_left q hw]
  rw [h1]

/-- A 16-element lane run never straddles a Q8_K block (256 elements). -/
theorem octet_fits_q8k (i : Nat) (h : i % 16 = 0) : i % 256 + 16 ≤ 256 := by omega

/-- A 16-element lane run never straddles a Q8_0 block (32 elements). -/
theorem octet_fits_q8_0 (i : Nat) (h : i % 16 = 0) : i % 32 + 16 ≤ 32 := by omega

-- § reduction

/-- The sixteen-accumulator tree equals the flat sum (exact arithmetic). -/
theorem acc16_tree (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 : Int) :
    (((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))) +
      (((a8 + a9) + (a10 + a11)) + ((a12 + a13) + (a14 + a15))) =
    a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 := by
  omega

/-- Four-accumulator tree equals the flat sum. -/
theorem acc4_tree (a b c d : Int) : (a + b) + (c + d) = a + b + c + d := by omega

/-- One warp per row: eight rows per 256-thread block cover every row. -/
theorem warp_rows_cover (nOut : Nat) : nOut ≤ ((nOut + 7) / 8) * 8 := by omega

end Strata.SGemv

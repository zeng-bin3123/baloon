/-!
# MODULE: dequant_to_half_bf16
SOURCES: src/kernels/cuda/dequant_bf16.cu
Dequantizing ggml block formats (Q2_0, Q4_0, Q5_0, Q8_0, IQ4_NL, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS) to bf16, fp16 or
fp32, one thread per 32-element group; the i-quants are delegated to iq_kernels.

## Source semantics
Row stride is cols/be*bb bytes; each row has cols/32 groups. Group g of a row lives in block g / (be/32) at
sub-group g % (be/32). K-quant scale/min unpacking yields 6-bit values. fp32 -> bf16 rounds to nearest even via
u += 0x7fff + ((u>>16)&1); result u>>16.

## Form template identifiers
geometry -> "format table"; bf16Round -> "round-to-nearest-even truncation"; scaleMin -> "6-bit scale unpack".
-/
namespace Strata.DequantBf16

-- § geometry
/-- (elements, bytes) per block, or (0,0). -/
def beOf (t : Nat) : Nat :=
  if t = 2 ∨ t = 6 ∨ t = 8 ∨ t = 20 then 32 else if t = 42 then 64 else
  if t = 11 ∨ t = 12 ∨ t = 13 ∨ t = 14 ∨ t = 23 then 256 else 0

/-- Bytes per block. -/
def bbOf (t : Nat) : Nat :=
  if t = 2 then 18 else if t = 6 then 22 else if t = 8 then 34 else if t = 20 then 18
  else if t = 11 then 110 else if t = 12 then 144 else if t = 13 then 176 else if t = 14 then 210
  else if t = 23 then 136 else if t = 42 then 18 else 0

/-- `dequant_bf16_supported`. -/
def supp (t : Nat) : Bool := beOf t != 0

/-- Every supported block is a whole number of 32-element groups. -/
theorem groups_per_block (t : Nat) (ht : t < 64) (h : supp t = true) : beOf t % 32 = 0 := by
  revert t
  decide

/-- Group decomposition: block index and sub-group reassemble the group index (256-wide blocks). -/
theorem group_decomp_256 (g : Nat) : 8 * (g / 8) + g % 8 = g := by
  omega

/-- Same for Q2_0 (two groups per 64-element block). -/
theorem group_decomp_q2_0 (g : Nat) : 2 * (g / 2) + g % 2 = g := by
  omega

/-- The sub-group of a 256-wide block is below 8 (so its element offset is below 256). -/
theorem subgroup_lt (g : Nat) : (g % 8) * 32 + 32 ≤ 256 := by
  omega

/-- The i-quant types dispatched elsewhere are disjoint from this file's table. -/
theorem iq_disjoint (t : Nat) (ht : t < 64) (h : t = 16 ∨ t = 17 ∨ t = 18 ∨ t = 21 ∨ t = 22 ∨ t = 29) :
    supp t = false := by
  revert t
  decide

-- § K-quant scale unpacking
/-- High-index scale (j >= 4): low nibble of q[j+4] plus two high bits of q[j-4] shifted up four. -/
def scaleHi (lowByte hiByte : Nat) : Nat := lowByte % 16 + (hiByte / 64) * 16

/-- Unpacked 6-bit scales stay below 64. -/
theorem scaleHi_lt (a b : Nat) (ha : a < 256) (hb : b < 256) : scaleHi a b < 64 := by
  unfold scaleHi
  omega

/-- Low-index scale (j < 4): the low six bits of the byte. -/
theorem scaleLo_lt (a : Nat) : a % 64 < 64 := by
  omega

-- § bf16 rounding
/-- `f2bf` on the raw 32-bit pattern (no overflow modeled: finite inputs). -/
def bf16Round (u : Nat) : Nat := (u + 32767 + (u / 65536) % 2) / 65536

/-- Exactly representable inputs (low half zero) are preserved. -/
theorem bf16_exact (u : Nat) (h : u % 65536 = 0) : bf16Round u = u / 65536 := by
  unfold bf16Round
  omega

/-- Below the halfway point the result truncates. -/
theorem bf16_round_down (u : Nat) (h : u % 65536 < 32768) : bf16Round u = u / 65536 := by
  unfold bf16Round
  omega

/-- Above the halfway point the result rounds up. -/
theorem bf16_round_up (u : Nat) (h : 32768 < u % 65536) : bf16Round u = u / 65536 + 1 := by
  unfold bf16Round
  omega

/-- At exactly the halfway point the result is even (ties to even). -/
theorem bf16_tie_even (u : Nat) (h : u % 65536 = 32768) : bf16Round u % 2 = 0 := by
  obtain ⟨a, ha⟩ : ∃ a, u = 65536 * a + 32768 := ⟨u / 65536, by omega⟩
  subst ha
  have h1 : (65536 * a + 32768) / 65536 = a := Nat.div_eq_of_lt_le (by omega) (by omega)
  unfold bf16Round
  rw [h1]
  rcases Nat.mod_two_eq_zero_or_one a with h2 | h2
  · have key : (65536 * a + 32768 + 32767 + a % 2) / 65536 = a :=
      Nat.div_eq_of_lt_le (by omega) (by omega)
    rw [key]
    omega
  · have key : (65536 * a + 32768 + 32767 + a % 2) / 65536 = a + 1 :=
      Nat.div_eq_of_lt_le (by omega) (by omega)
    rw [key]
    omega

/-- Launch grid covers every group. -/
theorem grid_covers (total : Nat) : total ≤ (total + 255) / 256 * 256 := by
  omega

end Strata.DequantBf16

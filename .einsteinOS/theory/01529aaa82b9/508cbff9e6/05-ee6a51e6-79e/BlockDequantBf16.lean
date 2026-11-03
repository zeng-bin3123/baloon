/-!
# MODULE: block_dequant_bf16
Dequantization of ggml block formats (Q2_0, Q4_0, Q5_0, Q8_0, Q3_K..Q6_K, IQ4_NL, IQ4_XS) to BF16/FP16/FP32,
one 32-element group per thread, with round-to-nearest-even BF16 conversion.

## Source semantics
A row has `cols / 32` groups; group `gi` of a 256-element format lives in block `gi / 8`, sub-group `gi % 8`;
for Q2_0 (64-element blocks) in block `gi / 2`, offset `(gi % 2) * 32`. Output index of (row r, group gi) is
`(r * groups_per_row + gi) * 32`. BF16 conversion is `(u + 0x7fff + ((u >> 16) & 1)) >> 16` on 32-bit words.

## Form template identifiers
- `geometry` -> blockGeom, `group32` -> group decoding, `f2bf` -> toBf16
- `scale_min_k4` -> scaleMinK4
-/

namespace Strata.BlockDequant

-- § geometry

/-- (elements, bytes) per block. -/
def blockGeom : Nat → Option (Nat × Nat)
  | 2 => some (32, 18) | 6 => some (32, 22) | 8 => some (32, 34) | 20 => some (32, 18)
  | 11 => some (256, 110) | 12 => some (256, 144) | 13 => some (256, 176)
  | 14 => some (256, 210) | 23 => some (256, 136) | 42 => some (64, 18) | _ => none

/-- Q4_K and Q5_K geometry. -/
theorem k_quant_geometry : blockGeom 12 = some (256, 144) ∧ blockGeom 13 = some (256, 176) := by decide

/-- IQ i-quants are not handled here. -/
theorem iq_types_absent : blockGeom 16 = none ∧ blockGeom 29 = none := by decide

/-- Groups of a 256-element block: `gi / 8` and `gi % 8` reassemble `gi`. -/
theorem k_group_decomposition (gi : Nat) : (gi / 8) * 8 + gi % 8 = gi := by omega

/-- Q2_0 group `gi` is block `gi/2` at element offset `32 * (gi % 2)`, always inside the 64-element block. -/
theorem q2_0_group_offset (gi j : Nat) (hj : j < 32) : (gi % 2) * 32 + j < 64 := by omega

-- § output indexing

/-- Distinct (row, group) pairs map to distinct output slots. -/
theorem row_group_index_inj (G r gi r' gi' : Nat) (hG : 0 < G) (h : gi < G) (h' : gi' < G)
    (e : r * G + gi = r' * G + gi') : r = r' ∧ gi = gi' := by
  have d1 : (r * G + gi) / G = r := by
    rw [Nat.mul_comm r G, Nat.mul_add_div hG, Nat.div_eq_of_lt h, Nat.add_zero]
  have d2 : (r' * G + gi') / G = r' := by
    rw [Nat.mul_comm r' G, Nat.mul_add_div hG, Nat.div_eq_of_lt h', Nat.add_zero]
  have hr : r = r' := by rw [← d1, ← d2, e]
  subst hr
  omega

/-- Row byte size is block count times block bytes (Q6_K example, 2560 columns). -/
theorem q6k_row_bytes : 2560 / 256 * 210 = 2100 := by decide

-- § per-format code ranges

/-- Q4_0 codes decode to [-8, 7]. -/
theorem q4_0_range (c : Nat) (h : c < 16) : -8 ≤ (c : Int) - 8 ∧ (c : Int) - 8 ≤ 7 := by omega

/-- Q5_0 five-bit codes decode to [-16, 15]. -/
theorem q5_0_range (c : Nat) (h : c < 32) : -16 ≤ (c : Int) - 16 ∧ (c : Int) - 16 ≤ 15 := by omega

/-- Q6_K assembles a six-bit code from a nibble and two high bits. -/
theorem q6k_assemble (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) : lo + 16 * hi < 64 := by omega

/-- Q6_K codes decode to [-32, 31]. -/
theorem q6k_range (q : Nat) (h : q < 64) : -32 ≤ (q : Int) - 32 ∧ (q : Int) - 32 ≤ 31 := by omega

/-- Q5_K assembles a five-bit code from a nibble and one high bit. -/
theorem q5k_assemble (nib hb : Nat) (hn : nib < 16) (hh : hb < 2) : nib + 16 * hb < 32 := by omega

/-- Q2_0 codes decode to [-1, 2]. -/
theorem q2_0_range (c : Nat) (h : c < 4) : -1 ≤ (c : Int) - 1 ∧ (c : Int) - 1 ≤ 2 := by omega

/-- A byte splits into two nibbles (low, high), each below 16, which reassemble the byte. -/
theorem nibble_split (b : Nat) (h : b < 256) : b % 16 < 16 ∧ b / 16 < 16 ∧ b % 16 + 16 * (b / 16) = b := by
  omega

/-- IQ4_XS six-bit scale `ls` minus 32 lies in [-32, 31]. -/
theorem iq4xs_scale_range (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) :
    -32 ≤ ((lo + 16 * hi : Nat) : Int) - 32 ∧ ((lo + 16 * hi : Nat) : Int) - 32 ≤ 31 := by omega

-- § BF16 conversion

/-- Round-to-nearest-even BF16 conversion on a 32-bit word (with 32-bit wrap-around). -/
def toBf16 (u : Nat) : Nat := ((u + 32767 + (u / 65536) % 2) % 4294967296) / 65536

/-- The result is a 16-bit pattern. -/
theorem toBf16_lt (u : Nat) : toBf16 u < 65536 := by
  unfold toBf16
  omega

/-- Values already representable in BF16 convert exactly. -/
theorem toBf16_exact (h : Nat) (hh : h < 65536) : toBf16 (65536 * h) = h := by
  unfold toBf16
  have h1 : 65536 * h / 65536 = h := by omega
  rw [h1]
  have h2 : 65536 * h + 32767 + h % 2 < 4294967296 := by omega
  rw [Nat.mod_eq_of_lt h2]
  omega

/-- Ties round to even: an even upper half stays. -/
theorem toBf16_tie_even (h : Nat) (hh : h < 65535) (he : h % 2 = 0) :
    toBf16 (65536 * h + 32768) = h := by
  unfold toBf16
  omega

/-- Ties round to even: an odd upper half rounds up. -/
theorem toBf16_tie_odd (h : Nat) (hh : h < 65535) (ho : h % 2 = 1) :
    toBf16 (65536 * h + 32768) = h + 1 := by
  unfold toBf16
  omega

/-- Grid covers every group with 256 threads per block. -/
theorem grid_cover (total : Nat) : total ≤ ((total + 255) / 256) * 256 := by omega

end Strata.BlockDequant

/-!
# MODULE: artifact_dequant_reference
Scalar reference dequantizers (Q2_0, Q8_0, Q4_0, Q5_0, IQ4_NL, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS): block
geometry, field-size accounting, index bounds for every packed access, value ranges of the integer symbols,
and the IQ4_NL codebook.

## Source semantics
* Q2_0: block = fp16 `d` + 16 code bytes = 64 elements / 18 bytes; code `c ∈ {0..3}` ↦ symbol `c - 1`; element
  `j` reads byte `j/4`, bit offset `2 (j % 4)`.
* Each format's field sizes add up to its byte size.
* K-quant scale/min unpack (`get_scale_min_k4`) yields 6-bit values.
* Q3_K scales are `s - 32` (signed), the low-2-bit symbol is offset by 4 when the high bit is clear.
* IQ4_NL / IQ4_XS use a strictly increasing 16-entry codebook.
* bf16 → fp32 is a 16-bit left shift; `read_u16` composes two bytes little-endian.

## Form template identifiers
dequantize_q2_0 … dequantize_iq4_xs → block_dequantizer, kvalues_iq4nl → nonlinear_codebook,
get_scale_min_k4 → scale_min_unpack, read_u16 → le_u16, fp16_to_fp32/bf16_to_fp32 → half_to_float.
-/

namespace Strata.Artifact.Dequant

-- § block geometry

/-- (elements per block, bytes per block) for each format. -/
def geom : String → Option (Nat × Nat)
  | "Q2_0" => some (64, 18)
  | "Q8_0" => some (32, 34)
  | "Q4_0" => some (32, 18)
  | "Q5_0" => some (32, 22)
  | "IQ4_NL" => some (32, 18)
  | "Q6_K" => some (256, 210)
  | "Q4_K" => some (256, 144)
  | "Q5_K" => some (256, 176)
  | "Q3_K" => some (256, 110)
  | "IQ4_XS" => some (256, 136)
  | _ => none

/-- Field-size accounting: every block size is the sum of its fields. -/
theorem field_sizes_add_up :
    18 = 2 + 16 ∧ 34 = 2 + 32 ∧ 22 = 2 + 4 + 16 ∧ 210 = 128 + 64 + 16 + 2 ∧ 144 = 2 + 2 + 12 + 128 ∧
      176 = 2 + 2 + 12 + 32 + 128 ∧ 110 = 32 + 64 + 12 + 2 ∧ 136 = 2 + 2 + 4 + 128 := by decide

/-- The 64-element Q2_0 block stores 4 codes per byte over 16 bytes. -/
theorem q2_0_codes_fill_block : 16 * 4 = 64 := by decide

-- § Q2_0 indexing and symbols

/-- Byte index and bit offset of element `j` of a Q2_0 block stay in range. -/
theorem q2_0_index_bounds (j : Nat) (h : j < 64) : j / 4 < 16 ∧ (j % 4) * 2 ≤ 6 := by omega

/-- The (byte, slot) pair determines `j` uniquely: `4 * (j/4) + j%4 = j`. -/
theorem q2_0_index_unique (j : Nat) : 4 * (j / 4) + j % 4 = j := by omega

/-- Code `c < 4` maps to a symbol in {-1, 0, 1, 2}. -/
theorem q2_0_symbol_range (c : Int) (h0 : 0 ≤ c) (h1 : c < 4) :
    c - 1 = -1 ∨ c - 1 = 0 ∨ c - 1 = 1 ∨ c - 1 = 2 := by omega

/-- Every dequantized value is one of `-d, 0, d, 2d` (as integer multiples of `d`): the structural check
that `--check` asserts. -/
theorem q2_0_values_multiples (c d : Int) (h0 : 0 ≤ c) (h1 : c < 4) :
    (c - 1) * d = -d ∨ (c - 1) * d = 0 ∨ (c - 1) * d = d ∨ (c - 1) * d = 2 * d := by
  have : c = 0 ∨ c = 1 ∨ c = 2 ∨ c = 3 := by omega
  rcases this with rfl | rfl | rfl | rfl <;> simp <;> omega

-- § 4/5-bit linear formats

/-- Q4_0 symbols: a nibble minus 8 lies in [-8, 7]. -/
theorem q4_0_symbol_range (n : Int) (h0 : 0 ≤ n) (h1 : n < 16) : -8 ≤ n - 8 ∧ n - 8 ≤ 7 := by omega

/-- Q5_0: a 5-bit value (nibble plus one high bit) minus 16 lies in [-16, 15]. -/
theorem q5_0_symbol_range (lo hi : Nat) (hl : lo < 16) (hh : hi < 2) :
    (lo + 16 * hi : Int) - 16 ≥ -16 ∧ (lo + 16 * hi : Int) - 16 ≤ 15 := by omega

/-- Q4_0/Q5_0/IQ4_NL: element `j` and `j+16` of a 32-block come from nibble `j`, `j < 16`. -/
theorem nibble_pair_cover (j : Nat) (h : j < 16) : j < 32 ∧ j + 16 < 32 ∧ j + 16 ≥ 16 := by omega

/-- The two nibbles of a byte recombine to it. -/
theorem nibbles_recombine (b : Nat) (h : b < 256) : b % 16 + 16 * (b / 16) = b := by omega

-- § K-quants

/-- `get_scale_min_k4` for `j < 4`: low six bits of two bytes, hence below 64. -/
theorem scale_min_small (a b : Nat) : a % 64 < 64 ∧ b % 64 < 64 := by omega

/-- `get_scale_min_k4` for `j ≥ 4`: low nibble of one byte plus the top two bits of another, shifted by 4,
stays below 64. -/
theorem scale_min_large (q4 qm : Nat) (h1 : q4 < 256) (h2 : qm < 256) :
    q4 % 16 + 16 * (qm / 64) < 64 := by omega

/-- The same for the min half: high nibble plus top two bits shifted by 4. -/
theorem scale_min_large_min (q4 qm : Nat) (h1 : q4 < 256) (h2 : qm < 256) :
    q4 / 16 + 16 * (qm / 64) < 64 := by omega

/-- Q4_K / Q5_K advance four 64-element groups, using scale indices `is, is+1` with `is = 0,2,4,6`: all in 0..7. -/
theorem q45k_scale_index (g : Nat) (h : g < 4) : 2 * g < 8 ∧ 2 * g + 1 < 8 := by omega

/-- Q5_K high-bit masks `u1 = 1 << 2g`, `u2 = 2 << 2g` are distinct powers of two fitting a byte. -/
theorem q5k_masks : (List.range 4).map (fun g => 1 <<< (2 * g)) = [1, 4, 16, 64] ∧
    (List.range 4).map (fun g => 2 <<< (2 * g)) = [2, 8, 32, 128] := by decide

/-- Q4_K output index within its 256-element block: group `g` writes 32 + 32 elements at `64 g`. -/
theorem q4k_output_cover (g l : Nat) (hg : g < 4) (hl : l < 32) : 64 * g + l < 256 ∧ 64 * g + 32 + l < 256 := by
  omega

/-- Q4_K/Q5_K `qs` reads (`+16`/`+48` header) stay in bounds. -/
theorem q45k_reads (g l : Nat) (hg : g < 4) (hl : l < 32) :
    16 + 32 * g + l < 144 ∧ 48 + 32 * g + l < 176 := by omega

/-- Q6_K: each 128-half uses scale indices `8h + l/16 + {0,2,4,6}`, all below 16. -/
theorem q6k_scale_index (h l : Nat) (hh : h < 2) (hl : l < 32) :
    8 * h + l / 16 + 6 < 16 := by omega

/-- Q6_K reads: `ql` ≤ index 127, `qh` ≤ 191, `scales` ≤ 207, `d` at 208. -/
theorem q6k_reads (h l : Nat) (hh : h < 2) (hl : l < 32) :
    64 * h + l + 32 < 128 ∧ 128 + 32 * h + l < 192 ∧ 192 + 8 * h + l / 16 + 6 < 208 := by omega

/-- Q6_K symbol: a 6-bit value minus 32 lies in [-32, 31]. -/
theorem q6k_symbol_range (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) :
    (lo + 16 * hi : Int) - 32 ≥ -32 ∧ (lo + 16 * hi : Int) - 32 ≤ 31 := by omega

/-- Q3_K: the low-2-bit value minus 4 (high bit clear) or minus 0 lies in [-4, 3]. -/
theorem q3k_symbol_range (q : Int) (hm : Bool) (h0 : 0 ≤ q) (h1 : q < 4) :
    -4 ≤ q - (if hm then 0 else 4) ∧ q - (if hm then 0 else 4) ≤ 3 := by
  cases hm <;> simp <;> omega

/-- Q3_K scales are signed: a 6-bit scale minus 32 lies in [-32, 31]. -/
theorem q3k_scale_range (s : Int) (h0 : 0 ≤ s) (h1 : s < 64) : -32 ≤ s - 32 ∧ s - 32 ≤ 31 := by omega

/-- Q3_K reads: scales at 96..107, d at 108; `q` offset `32 + 32 n + l + 16`. -/
theorem q3k_reads (n l : Nat) (hn : n < 2) (hl : l < 16) : 32 + 32 * n + l + 16 < 96 := by omega

/-- Q3_K mask shifts across both halves: 8 distinct bits 1,2,…,128 (no reset between halves). -/
theorem q3k_mask_sequence : (List.range 8).map (fun i => 1 <<< i) = [1, 2, 4, 8, 16, 32, 64, 128] := by decide

/-- IQ4_XS: a six-bit scale index `lo | (hi << 4)` is below 64, and `ls - 32` is in [-32, 31]. -/
theorem iq4xs_scale (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) :
    lo + 16 * hi < 64 ∧ (lo + 16 * hi : Int) - 32 ≥ -32 ∧ (lo + 16 * hi : Int) - 32 ≤ 31 := by omega

/-- IQ4_XS group `ib` reads 16 `qs` bytes at `8 + 16 ib`, inside the 136-byte block. -/
theorem iq4xs_reads (ib j : Nat) (hib : ib < 8) (hj : j < 16) : 8 + 16 * ib + j < 136 := by omega

/-- IQ4_XS scale nibble selection: `scales_l[ib/2]` shifted by `4 (ib % 2)`, index below 4. -/
theorem iq4xs_scale_l_index (ib : Nat) (h : ib < 8) : ib / 2 < 4 ∧ 4 * (ib % 2) ≤ 4 := by omega

-- § the IQ4_NL codebook

/-- The non-linear codebook of ggml's `kvalues_iq4nl`. -/
def kvalues : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- Sixteen entries, strictly increasing. -/
theorem kvalues_sorted : kvalues.length = 16 ∧ List.Pairwise (· < ·) kvalues := by decide

/-- Entries fit int8, so the table can be stored as `int8_t`. -/
theorem kvalues_int8 : ∀ v ∈ kvalues, -128 ≤ v ∧ v ≤ 127 := by decide

-- § element types

/-- `read_u16` of two bytes is below 2^16. -/
theorem read_u16_lt (p0 p1 : Nat) (h0 : p0 < 256) (h1 : p1 < 256) : p0 + 256 * p1 < 65536 := by omega

/-- bf16 → fp32 places the 16 bits in the top half of a 32-bit word. -/
theorem bf16_shift_fits (h : Nat) (hh : h < 65536) : h * 65536 < 4294967296 := by omega

end Strata.Artifact.Dequant

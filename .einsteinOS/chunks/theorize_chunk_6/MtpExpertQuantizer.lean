/-!
# MODULE: mtp_expert_quantizer
Block-quantisers for the MTP draft block's routed experts (Q2_0 with per-block MSE scale search, ggml-reference
Q4_0 and Q8_0): code ranges, byte packing, candidate-scale grid, block sizes and the resulting tensor sizes.

## Source semantics
* Q2_0: 64-element blocks, grid `{-1,0,1,2} * d`; stored code is `q + 1` in 0..3, four codes per byte
  (`c0 | c1<<2 | c2<<4 | c3<<6`), block = fp16 scale + 16 bytes = 18 bytes. The scale search tries
  `d = amax * f` for 17 factors `f` evenly spaced in `[1/2, 1]`.
* Q4_0: 32-element blocks, `q = min(15, trunc(x/d + 8.5))`, first 16 elements in low nibbles and last 16 in
  high nibbles; 18 bytes per block.
* Q8_0: 32-element blocks of int8 plus fp16 scale = 34 bytes; scale `amax/127`.
* Experts keep the fused layout `gate_up_proj [512,1280,2560]` and `down_proj [512,2560,640]`, quantised along
  the last axis, which must be a multiple of the block size.

## Form template identifiers
q2_block := 64 ; q2_block_bytes := 18 ; q4_block := 32 ; q8_block_bytes := 34 ; grid_steps := 17 ;
gate_up_shape := [512,1280,2560] ; down_shape := [512,2560,640]
-/

namespace Strata.MtpQuant

-- § block geometry

/-- Q2_0 block: 2 + 16 bytes cover 64 elements at 2 bits each. -/
theorem q2_block : 2 + 64 / 4 = 18 := by
  decide

/-- Q4_0 block: 2 + 16 bytes cover 32 elements at 4 bits each. -/
theorem q4_block : 2 + 32 / 2 = 18 := by
  decide

/-- Q8_0 block: 2 + 32 bytes. -/
theorem q8_block : 2 + 32 = 34 := by
  decide

/-- Both expert inner dimensions are whole numbers of 64-element (and so 32-element) blocks. -/
theorem inner_dims_divisible : 2560 % 64 = 0 ∧ 640 % 64 = 0 ∧ 2560 % 32 = 0 ∧ 640 % 32 = 0 := by
  decide

-- § Q2_0 codes

/-- Clamp of the rounded level to the grid `{-1, 0, 1, 2}`. -/
def clampQ2 (q : Int) : Int := max (-1) (min 2 q)

/-- The clamped level is always on the grid. -/
theorem clampQ2_range (q : Int) : -1 ≤ clampQ2 q ∧ clampQ2 q ≤ 2 := by
  unfold clampQ2
  omega

/-- The stored code `q + 1` is in 0..3, so it fits two bits. -/
theorem q2_code_range (q : Int) : 0 ≤ clampQ2 q + 1 ∧ clampQ2 q + 1 ≤ 3 := by
  unfold clampQ2
  omega

/-- The decoder's `code - 1` recovers the clamped level exactly. -/
theorem q2_code_roundtrip (q : Int) : (clampQ2 q + 1) - 1 = clampQ2 q := by
  omega

/-- Levels already on the grid are untouched by the clamp. -/
theorem clampQ2_id (q : Int) (h1 : -1 ≤ q) (h2 : q ≤ 2) : clampQ2 q = q := by
  unfold clampQ2
  omega

/-- Four 2-bit codes pack into a byte (`c0 | c1<<2 | c2<<4 | c3<<6`) and unpack to themselves. -/
theorem q2_pack_unpack (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) < 256 ∧ (c0 + 4 * c1 + 16 * c2 + 64 * c3) % 4 = c0 ∧
      ((c0 + 4 * c1 + 16 * c2 + 64 * c3) / 4) % 4 = c1 ∧ ((c0 + 4 * c1 + 16 * c2 + 64 * c3) / 16) % 4 = c2 ∧
      ((c0 + 4 * c1 + 16 * c2 + 64 * c3) / 64) % 4 = c3 := by
  omega

-- § scale search grid

/-- Candidate scale factor numerators: factor `k` is `(16 + k) / 32`, `k = 0..16`. -/
def factorNum (k : Nat) : Nat := 16 + k

/-- The 17 factors span exactly `[16/32, 32/32] = [1/2, 1]`. -/
theorem factor_span : factorNum 0 = 16 ∧ factorNum 16 = 32 := by
  decide

/-- Every candidate factor is between one half and one. -/
theorem factor_in_range (k : Nat) (h : k < 17) : 16 ≤ factorNum k ∧ factorNum k ≤ 32 := by
  unfold factorNum
  omega

/-- Factors are strictly increasing in `k`, so the grid has 17 distinct values. -/
theorem factor_strict (k j : Nat) (h : k < j) : factorNum k < factorNum j := by
  unfold factorNum
  omega

/-- At the largest factor the scale equals `amax` (the ggml reference), so the reference is on the grid. -/
theorem factor_reference : factorNum 16 = 32 := by
  decide

/-- At the smallest factor the `+2` level reaches `amax` (`2 * (amax/2) = amax`). -/
theorem factor_plus_two_reaches_max : 2 * factorNum 0 = 32 := by
  decide

-- § Q4_0 / Q8_0 codes

/-- Q4_0 code from `trunc(x*inv + 8.5)` capped at 15. -/
def clampQ4 (t : Nat) : Nat := min 15 t

/-- The Q4_0 code fits a nibble. -/
theorem q4_code_range (t : Nat) : clampQ4 t < 16 := by
  unfold clampQ4
  omega

/-- First half of the block in low nibbles, second half in high nibbles: pack and unpack. -/
theorem q4_pack_unpack (lo hi : Nat) (hl : lo < 16) (hh : hi < 16) :
    (lo + 16 * hi) < 256 ∧ (lo + 16 * hi) % 16 = lo ∧ (lo + 16 * hi) / 16 = hi := by
  omega

/-- The packed nibble byte `j` carries elements `j` and `j + 16` of the 32-element block. -/
theorem q4_element_positions (j : Nat) (h : j < 16) : j < 16 ∧ 16 ≤ j + 16 ∧ j + 16 < 32 := by
  omega

/-- Q8_0 level is an int8 in `[-127, 127]` when `amax/127` is the scale. -/
def clampQ8 (q : Int) : Int := max (-127) (min 127 q)

/-- Q8_0 levels never reach `-128`. -/
theorem q8_level_range (q : Int) : -127 ≤ clampQ8 q ∧ clampQ8 q ≤ 127 := by
  unfold clampQ8
  omega

-- § tensor sizes

/-- Elements of the fused gate/up tensor. -/
def guElems : Nat := 512 * 1280 * 2560

/-- Elements of the down tensor. -/
def downElems : Nat := 512 * 2560 * 640

/-- The down tensor holds half the elements of gate/up. -/
theorem down_is_half : 2 * downElems = guElems := by
  decide

/-- Q2_0 bytes of the gate/up tensor. -/
theorem q2_gate_up_bytes : guElems / 64 * 18 = 471859200 := by
  decide

/-- Q2_0 bytes of the down tensor. -/
theorem q2_down_bytes : downElems / 64 * 18 = 235929600 := by
  decide

/-- Q2_0 expert total, about 0.71 GB as the header comment states. -/
theorem q2_total_bytes : guElems / 64 * 18 + downElems / 64 * 18 = 707788800 := by
  decide

/-- Q4_0 expert total, about 1.42 GB. -/
theorem q4_total_bytes : guElems / 32 * 18 + downElems / 32 * 18 = 1415577600 := by
  decide

/-- Q8_0 expert total, about 2.67 GB. -/
theorem q8_total_bytes : guElems / 32 * 34 + downElems / 32 * 34 = 2673868800 := by
  decide

/-- The three formats are ordered by size: Q2_0 < Q4_0 < Q8_0. -/
theorem format_size_order : 707788800 < 1415577600 ∧ 1415577600 < 2673868800 := by
  decide

/-- Each expert slice is a whole number of Q2_0 blocks (so slicing by expert is block aligned). -/
theorem expert_slice_blocks : (1280 * 2560) % 64 = 0 ∧ (2560 * 640) % 64 = 0 := by
  decide

end Strata.MtpQuant

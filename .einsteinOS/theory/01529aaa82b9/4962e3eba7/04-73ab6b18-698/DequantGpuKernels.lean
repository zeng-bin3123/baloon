/-!
# MODULE: dequant_gpu_kernels
GPU dequantiser indexing: one thread per 32-element group, per-type block/sub-block decomposition,
agreement with the sequential CPU order, launch grid, bf16 round-to-nearest-even.

## Source semantics
`group32<TYPE>` decodes group gi of a row. Q2_0: block gi/2, element offset (gi%2)*32. K-quants:
block gi/8, sub-group gi%8. Q3_K mask is 1 << (n*4+jj); Q5_K mask is 1 << (2*j64+hi); Q4_K and Q5_K
scale index is gi. The grid is ceil(total / 256). `f2bf` adds 0x7fff plus the lsb of the kept part
and truncates 16 bits, which is round-to-nearest-even.

## Form template identifiers
group32 -> decode_group ; f2bf -> f32_to_bf16 ; grid -> launch_grid
-/

set_option maxRecDepth 100000

namespace Strata.DequantGpu

-- § Per-type decomposition

/-- Q2_0 group gi of a row: block gi/2, element base (gi%2)*32; elements stay below 64. -/
theorem q2_group (gi j : Nat) (hj : j < 32) : (gi % 2) * 32 + j < 64 := by omega

/-- Q2_0 groups tile a block of 64: group 2b covers [0,32), group 2b+1 covers [32,64). -/
theorem q2_tile (b : Nat) : (2 * b) / 2 = b ∧ (2 * b + 1) / 2 = b ∧ (2 * b) % 2 = 0 ∧
    (2 * b + 1) % 2 = 1 := by omega

/-- Q3_K: the GPU scale index n*8 + jj*2 + half equals the CPU sequential counter
`2*(4n + jj) + half`. -/
theorem q3k_scale_idx (n jj half : Nat) :
    n * 8 + jj * 2 + half = 2 * (4 * n + jj) + half := by omega

/-- Q3_K sub-group gi < 8 splits as gi = 4n + jj. -/
theorem q3k_split (gi : Nat) (h : gi < 8) : gi / 4 * 4 + gi % 4 = gi := by omega

/-- Q3_K: the GPU mask exponent n*4+jj equals gi, matching the CPU's hoisted mask shifted once
per step. -/
theorem q3k_mask_exp (gi : Nat) (h : gi < 8) : (gi / 4) * 4 + gi % 4 = gi := by omega

/-- Q3_K mask exponent stays below 8, so the mask fits a byte. -/
theorem q3k_mask_fits (gi : Nat) (h : gi < 8) : (1 <<< gi) < 256 := by
  have : ∀ k < 8, (1 <<< k) < 256 := by decide
  exact this gi h

/-- Q4_K / Q5_K: scale index 2*(gi/2) + gi%2 equals gi, as the CPU's `is + {0,1}`. -/
theorem q45k_scale_idx (gi : Nat) : 2 * (gi / 2) + gi % 2 = gi := by omega

/-- Q5_K: the exponent 2*j64 + hi equals gi, so `1 << (2*j64+hi)` is the CPU's u1 / u2 pair. -/
theorem q5k_mask_exp (gi : Nat) : 2 * (gi / 2) + gi % 2 = gi := by omega

/-- Q5_K qs offset 32*j64 stays in the 128-byte payload for j64 < 4. -/
theorem q5k_qs_off (j64 l : Nat) (hj : j64 < 4) (hl : l < 32) : 32 * j64 + l < 128 := by omega

/-- Q6_K: scale index 8n + l/16 + 2qu stays below 16. -/
theorem q6k_scale (n qu l : Nat) (hn : n < 2) (hq : qu < 4) (hl : l < 32) :
    8 * n + l / 16 + 2 * qu < 16 := by omega

/-- Q6_K: ql index 64n + l + 32 stays inside 128. -/
theorem q6k_ql (n l : Nat) (hn : n < 2) (hl : l < 32) : 64 * n + l + 32 < 128 := by omega

/-- Q6_K: qh index 32n + l stays inside 64. -/
theorem q6k_qh (n l : Nat) (hn : n < 2) (hl : l < 32) : 32 * n + l < 64 := by omega

/-- IQ4_XS: group ib has qs at 16*ib, below 128 for ib < 8. -/
theorem iq4xs_qs (ib j : Nat) (hi : ib < 8) (hj : j < 16) : 16 * ib + j < 128 := by omega

/-- Output offset of group gi is gi*32, and groups tile gpr*32 columns. -/
theorem out_tile (gi gpr : Nat) (h : gi < gpr) : gi * 32 + 32 ≤ gpr * 32 := by
  have : (gi + 1) * 32 ≤ gpr * 32 := Nat.mul_le_mul_right 32 h
  omega

-- § Geometry check

/-- Every supported block size is a multiple of 32, so cols % be = 0 implies cols % 32 = 0. -/
theorem cols_mult32 (be cols : Nat) (hb : be = 32 ∨ be = 64 ∨ be = 256) (h : cols % be = 0) :
    cols % 32 = 0 := by
  rcases hb with rfl | rfl | rfl <;> omega

/-- For Q2_0, row bytes cols/64*18 equals (cols/32)/2 * 18 : groups pair into blocks. -/
theorem q2_row_groups (cols : Nat) (h : cols % 64 = 0) : cols / 64 = (cols / 32) / 2 := by omega

-- § Launch grid

/-- Ceil division grid covers all groups. -/
theorem grid_covers (total : Nat) : total ≤ (total + 255) / 256 * 256 := by omega

/-- And leaves no empty trailing block. -/
theorem grid_tight (total : Nat) (h : 0 < total) : ((total + 255) / 256 - 1) * 256 < total := by omega

/-- Zero total gives an empty grid, which is illegal to launch: the host rejects rows <= 0. -/
theorem grid_zero : (0 + 255) / 256 = 0 := by decide

-- § bf16 rounding

/-- Result of `(u + 0x7fff + ((u >> 16) & 1)) >> 16`, with u = 65536 q + r and lsb = q % 2. -/
def bf16Round (q r : Nat) : Nat := (65536 * q + r + 32767 + q % 2) / 65536

/-- Below half: truncate. -/
theorem bf16_below (q r : Nat) (h : r < 32768) : bf16Round q r = q := by
  unfold bf16Round; omega

/-- Above half: round up. -/
theorem bf16_above (q r : Nat) (h1 : 32768 < r) (h2 : r < 65536) : bf16Round q r = q + 1 := by
  unfold bf16Round; omega

/-- Exact tie rounds to even. -/
theorem bf16_tie_even (q : Nat) (h : q % 2 = 0) : bf16Round q 32768 = q := by
  unfold bf16Round; omega

/-- Exact tie, odd: rounds up to even. -/
theorem bf16_tie_odd (q : Nat) (h : q % 2 = 1) : bf16Round q 32768 = q + 1 := by
  unfold bf16Round; omega

/-- A tie always lands on an even result. -/
theorem bf16_tie_result_even (q : Nat) : bf16Round q 32768 % 2 = 0 := by
  rcases Nat.mod_two_eq_zero_or_one q with h | h
  · rw [bf16_tie_even q h]; exact h
  · rw [bf16_tie_odd q h]; omega

end Strata.DequantGpu

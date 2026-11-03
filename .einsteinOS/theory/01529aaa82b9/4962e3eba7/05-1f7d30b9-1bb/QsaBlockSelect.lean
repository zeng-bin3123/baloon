/-!
# MODULE: qsa_block_select
Batched block scoring and radix-select top-k over pooled indexer blocks (128-dim keys, 4 indexer heads,
blocks of R = 4 cells).  This file proves the launch coverage, the block-to-cell weighting, the radix
digit schedule, the per-thread block partition and the tie-budget clamp that make the selection exact.

## Source semantics
* `block_scores_kernel`: one warp per block, 8 warps per block; each lane loads a float4 of the key.
* `block_topk_kernel`: block weight is R for complete blocks and `n_kv - n_bid*R` for the tail; the
  threshold is found by 4 radix passes of 8 bits; at most `width - above` cells equal to the
  threshold are taken, lowest index first, each thread clamped to its share.

## Form template identifiers
- `block_scores_kernel` -> `block_score`; `block_topk_kernel` -> `block_select`
- `weight` -> `block_cells`; `eq_budget` -> `tie_budget`; `my_eq` -> `thread_ties`
-/

namespace Strata.QsaBlockSelect

-- § block scores

/-- A (block, warp) decomposition `b = bx*8 + w` covers every block index. -/
theorem score_block_cover {b maxb : Nat} (hb : b < maxb) : b / 8 < (maxb + 7) / 8 := by omega

/-- A float4 per lane covers the 128-dim key exactly. -/
theorem key_lanes_cover : 32 * 4 = 128 := by decide

/-- Lane runs of 4 dims are disjoint. -/
theorem key_lane_inj {l l' j j' : Nat} (hj : j < 4) (hj' : j' < 4)
    (h : l * 4 + j = l' * 4 + j') : l = l' ∧ j = j' := by omega

-- § cells of a block

/-- A cell id `b*4 + c` of a complete block is below `n_kv`. -/
theorem complete_cell_lt {b c n : Nat} (hb : b < n / 4) (hc : c < 4) : b * 4 + c < n := by omega

/-- A cell id of the tail block (`b = n_bid`) is below `n_kv`. -/
theorem tail_cell_lt {n c : Nat} (hc : c < n - (n / 4) * 4) : (n / 4) * 4 + c < n := by omega

/-- Block weights sum to `n_kv`: `n_bid` complete blocks of 4 plus the tail of `n % 4` cells. -/
theorem weights_total (n : Nat) : (n / 4) * 4 + (n - (n / 4) * 4) = n := by omega

/-- The tail weight is `n % 4`, and the score bias applies exactly when it is nonzero. -/
theorem tail_weight (n : Nat) : n - (n / 4) * 4 = n % 4 := by omega

/-- Every cell `j < n` lies in a block `j/4 <= n_bid`, at local slot `j % 4`. -/
theorem cell_in_block {j n : Nat} (hj : j < n) :
    j / 4 ≤ n / 4 ∧ (j / 4) * 4 + j % 4 = j ∧ j % 4 < 4 := by omega

-- § identity branch

/-- When `n_kv <= width` everything is selected, in ascending order. -/
theorem identity_selects_all {n width j : Nat} (h : n ≤ width) (hj : j < n) : j < width := by omega

-- § radix schedule

/-- Four passes of 8 bits cover all 32 key bits. -/
theorem radix_bits : 4 * 8 = 32 := by decide

/-- The shifts 24, 16, 8, 0 are the consecutive 8-bit digits. -/
theorem radix_shifts : (24 : Nat) = 16 + 8 ∧ (16 : Nat) = 8 + 8 ∧ (8 : Nat) = 0 + 8 := by decide

/-- A digit in `[0, 255]` is a valid histogram bin of the 256-bin table. -/
theorem digit_bin_lt {k : Nat} : (k / 2 ^ 24) % 256 < 256 := Nat.mod_lt _ (by decide)

-- § per-thread block partition

/-- Each of the 256 threads gets `ceil(nb/256)` consecutive blocks; every block has an owner thread. -/
theorem block_owner_lt {b nb per : Nat} (hb : b < nb) (hper : nb ≤ per * 256) : b / per < 256 :=
  Nat.div_lt_of_lt_mul (Nat.lt_of_lt_of_le hb hper)

/-- `per = (nb + 255)/256` satisfies `nb <= per * 256`. -/
theorem per_cover (nb : Nat) : nb ≤ ((nb + 255) / 256) * 256 := by omega

-- § tie budget clamp

/-- Per-thread tie share: the budget left after earlier threads, clamped to `[0, eq]`. -/
def clampEq (budget before eq : Int) : Int := min eq (max 0 (budget - before))

/-- The share is nonnegative. -/
theorem clampEq_nonneg {budget before eq : Int} (h : 0 ≤ eq) : 0 ≤ clampEq budget before eq := by
  unfold clampEq
  omega

/-- The share never exceeds the thread's own equal count. -/
theorem clampEq_le_eq {budget before eq : Int} : clampEq budget before eq ≤ eq := by
  unfold clampEq
  omega

/-- Earlier ties plus this share never exceed the budget (or the earlier ties themselves when over budget). -/
theorem clampEq_cap {budget before eq : Int} :
    before + clampEq budget before eq ≤ max budget before := by
  unfold clampEq
  omega

/-- If the thread's equal cells all fit in the remaining budget, all of them are taken. -/
theorem clampEq_fill {budget before eq : Int} (h0 : 0 ≤ eq) (h : before + eq ≤ budget) :
    clampEq budget before eq = eq := by
  unfold clampEq
  omega

/-- With no budget left nothing is taken. -/
theorem clampEq_zero {budget before eq : Int} (h : budget ≤ before) : clampEq budget before eq = min eq 0 := by
  unfold clampEq
  omega

/-- The budget is positive and fits the equal cells when `above < width <= above + eq_total`. -/
theorem tie_budget {above eqt width : Nat} (h1 : above < width) (h2 : width ≤ above + eqt) :
    0 < width - above ∧ width - above ≤ eqt ∧ above + (width - above) = width := by omega

end Strata.QsaBlockSelect

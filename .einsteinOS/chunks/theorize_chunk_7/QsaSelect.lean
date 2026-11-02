/-!
# MODULE: qsa_select
Block-level indexer scoring weights and radix top-k bookkeeping for batched QSA selection.

## Source semantics
Blocks 0..n_bid score; block b < n_bid carries 4 cells, the last carries n_kv - 4*n_bid = n_kv%4 cells (it may be empty). Total weight is n_kv. The radix select fixes the threshold key 8 bits at a time over 4 passes (shifts 24,16,8,0); equal-key cells are taken lowest index first up to the budget. If n_kv <= width everything is selected as the identity.

## Form template identifiers
block_topk_kernel -> block_topk, weight -> block_cells, order_key -> order_key
-/
namespace Strata.QsaSelect

/-- Cells carried by block b. -/
def weight (nBid nKv b : Nat) : Nat := if b < nBid then 4 else nKv - nBid * 4

-- § weights

/-- Total weight of blocks 0..n_bid is n_kv. -/
theorem total_weight (nKv : Nat) : 4 * (nKv / 4) + (nKv - (nKv / 4) * 4) = nKv := by omega

/-- The last block's weight is n_kv mod 4. -/
theorem last_weight (nKv : Nat) : weight (nKv / 4) nKv (nKv / 4) = nKv % 4 := by unfold weight; simp; omega

/-- Full blocks carry 4 cells. -/
theorem full_weight (nBid nKv b : Nat) (h : b < nBid) : weight nBid nKv b = 4 := by unfold weight; simp [h]

/-- Cell ids of full blocks are below n_kv. -/
theorem full_cell_lt (nKv b c : Nat) (hb : b < nKv / 4) (hc : c < 4) : b * 4 + c < nKv := by omega

/-- Cell ids of the tail block are below n_kv. -/
theorem tail_cell_lt (nKv c : Nat) (hc : c < nKv % 4) : (nKv / 4) * 4 + c < nKv := by omega

/-- Block and cell recompose the id. -/
theorem cell_decomp (i : Nat) : (i / 4) * 4 + i % 4 = i := by omega

-- § radix passes

/-- Four 8-bit passes cover 32 bits. -/
theorem radix_bits : 4 * 8 = 32 := by decide

/-- The pass shifts are 24, 16, 8, 0. -/
theorem radix_shifts : [24, 16, 8, 0].map (fun s => s / 8) = [3, 2, 1, 0] := by decide

/-- A digit of a key is below 256. -/
theorem digit_lt (k : Nat) : (k / 2 ^ 24) % 256 < 256 := by omega

-- § tie budget

/-- Cells taken at the threshold by one thread: clamp of (budget - before) to [0, eq]. -/
def myEq (budget before eq : Int) : Int := max 0 (min eq (budget - before))

/-- A thread never takes more than it has. -/
theorem myEq_le (b a e : Int) (he : 0 ≤ e) : myEq b a e ≤ e := by unfold myEq; omega

/-- A thread never takes a negative count. -/
theorem myEq_nonneg (b a e : Int) : 0 ≤ myEq b a e := by unfold myEq; omega

/-- Earlier threads plus this one stay within the budget when the budget is not yet exhausted. -/
theorem myEq_budget (b a e : Int) (h : a ≤ b) : a + myEq b a e ≤ b := by unfold myEq; omega

-- § grids

/-- The scoring grid (8 warps per block) covers max_blocks. -/
theorem score_grid (mb : Nat) : mb ≤ (mb + 7) / 8 * 8 := by omega

/-- Each lane's float4 covers 4 of the 128 dims, 32 lanes give all of them. -/
theorem lane_dims (lane j : Nat) (hl : lane < 32) (hj : j < 4) : lane * 4 + j < 128 := by omega

end Strata.QsaSelect

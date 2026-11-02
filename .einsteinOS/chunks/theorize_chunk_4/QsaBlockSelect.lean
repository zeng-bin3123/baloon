/-!
# MODULE: qsa_block_select
Batched block scoring (relu'd per-head dots, +1e9 for an incomplete tail) and block-level radix top-k that
maps selected blocks to cell ids (4 cells per block).

## Source semantics
Block b < n_bid has weight 4; block n_bid has weight n_kv - 4 n_bid. If n_kv <= width everything is selected.
Radix select works on 8-bit digits at shifts 24, 16, 8, 0 to find the largest threshold key with weighted
count >= width; equal-key cells fill the remaining budget lowest index first.

## Form template identifiers
block_scores -> block_score; block_topk -> block_select
-/
namespace Strata.QsaSelect

/-- Cells in block `b`. -/
def weight (nKv nBid b : Nat) : Nat := if b < nBid then 4 else nKv - nBid * 4

/-- The last block weight is the tail size. -/
theorem weight_last (nKv : Nat) : weight nKv (nKv / 4) (nKv / 4) = nKv % 4 := by
  unfold weight; split <;> omega

/-- Completed cells plus the tail give n_kv. -/
theorem weights_total (nKv : Nat) : (nKv / 4) * 4 + weight nKv (nKv / 4) (nKv / 4) = nKv := by
  unfold weight; split <;> omega

/-- Cell id of the c-th cell of block b. -/
def cellId (b c : Nat) : Nat := b * 4 + c

/-- Cell ids decompose back into block and offset. -/
theorem cell_block (b c : Nat) (h : c < 4) : cellId b c / 4 = b ∧ cellId b c % 4 = c := by
  unfold cellId; omega

/-- Four 8-bit radix passes cover 32 bits. -/
theorem radix_cover : [24, 16, 8, 0].map (fun s => s + 8) = [32, 24, 16, 8] := by decide

/-- Equal-key budget never exceeds what remains. -/
theorem eq_budget_le (width above : Nat) (h : above ≤ width) : width - above ≤ width := by omega

/-- Per-thread equal quota is clamped between zero and its own count. -/
theorem my_eq_clamp (budget before eq : Nat) : min eq (budget - before) ≤ eq := by omega

/-- Block-per-warp grid covers all blocks. -/
theorem grid_cover (mb : Nat) : mb ≤ ((mb + 7) / 8) * 8 := by omega

end Strata.QsaSelect

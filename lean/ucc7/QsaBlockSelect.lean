/-!
# MODULE: qsa_block_select
SOURCES: src/kernels/cuda/qsa_select.cu
Batched block scoring (relu of per-head dots against the pooled key, spare/dead key for the tail) and an
exact radix-select top-k over BLOCKS where each block carries a cell weight (4, or the tail length), emitting
cell ids in ascending order with the lowest-index tie rule.

## Source semantics
* Block `b <= n_bid`; weight is `4` for `b < n_bid` and `n_kv - 4*n_bid` for the tail; weights sum to `n_kv`.
* The radix select fixes 8 key bits per pass (shifts 24, 16, 8, 0) with a weighted histogram.
* `eq_budget = width - above`; per-thread tie quota is `clamp(eq_budget - eq_before, 0, eq)`.
* If `n_kv <= width` every cell is selected (identity ascending).

## Form template identifiers
* `R` -> block_cells, `blockWeight` -> block_cell_count, `takeQuota` -> per_thread_tie_quota.
-/

namespace Ref.QsaSelect

/-- Cells per block. -/
def R : Nat := 4

/-- Weight of block `b`: full blocks have 4 cells, the last has the remainder. -/
def blockWeight (n_kv n_bid b : Nat) : Nat := if b < n_bid then 4 else n_kv - n_bid * 4

/-- Sum of the weights of the first `n` full blocks. -/
def fullSum : Nat → Nat
  | 0 => 0
  | n + 1 => fullSum n + 4

/-- The full-block sum is `4n`. -/
theorem fullSum_eq (n : Nat) : fullSum n = 4 * n := by
  induction n with
  | zero => rfl
  | succ k ih => simp only [fullSum, ih]; omega

/-- Weights over blocks `0..n_bid` sum to `n_kv` when `n_bid = n_kv / 4`. -/
theorem weights_total (n_kv n_bid : Nat) (h : n_bid = n_kv / 4) :
    fullSum n_bid + blockWeight n_kv n_bid n_bid = n_kv := by
  rw [fullSum_eq]
  unfold blockWeight
  rw [if_neg (by omega)]
  omega

/-- A block's weight is at most 4 (tail strictly below 4). -/
theorem weight_le (n_kv n_bid b : Nat) (h : n_bid = n_kv / 4) : blockWeight n_kv n_bid b ≤ 4 := by
  unfold blockWeight; split <;> omega

/-- The tail weight is zero iff `n_kv % 4 = 0` (an empty trailing block). -/
theorem tail_weight_zero_iff (n_kv n_bid : Nat) (h : n_bid = n_kv / 4) :
    blockWeight n_kv n_bid n_bid = 0 ↔ n_kv % 4 = 0 := by
  unfold blockWeight; rw [if_neg (by omega)]; omega

/-- Emitted cell ids `b*4 + c` (`c < weight`) stay below `n_kv`. -/
theorem cell_id_lt (n_kv n_bid b c : Nat) (h : n_bid = n_kv / 4) (hb : b ≤ n_bid)
    (hc : c < blockWeight n_kv n_bid b) : b * 4 + c < n_kv := by
  unfold blockWeight at hc; split at hc <;> omega

/-- Block-score grid: `ceil(max_blocks/8)` x `nq` blocks, block `bx*8 + warp`. -/
theorem score_grid_covers (max_blocks : Nat) : max_blocks ≤ (max_blocks + 7) / 8 * 8 := by omega

/-- Every block index maps to a (grid x, warp) pair. -/
theorem score_block_cover (b : Nat) : ∃ bx w, w < 8 ∧ bx * 8 + w = b := by
  exact ⟨b / 8, b % 8, by omega, by omega⟩

/-- A key whose tail block is empty keeps the +1e9 bias off: bias iff `n_kv % 4 ≠ 0`. -/
theorem bias_iff (n_kv n_bid : Nat) (h : n_bid = n_kv / 4) : n_bid * 4 < n_kv ↔ n_kv % 4 ≠ 0 := by omega

/-- Identity early exit: when `n_kv <= width` the output `j` for `j < n_kv` is the full range. -/
theorem identity_selection (n_kv width j : Nat) (h : n_kv ≤ width) (hj : j < n_kv) : j < width := by omega

/-- Digit extraction: a 32-bit key splits into four 8-bit digits that reassemble it. -/
theorem digits_roundtrip (k : Nat) (h : k < 4294967296) :
    (k / 16777216) * 16777216 + ((k / 65536) % 256) * 65536 + ((k / 256) % 256) * 256 + k % 256 = k := by
  omega

/-- Each digit is below 256, so the 256-bin histogram suffices. -/
theorem digit_lt (k s : Nat) : (k / s) % 256 < 256 := Nat.mod_lt _ (by omega)

/-- The top digit of a 32-bit key is also below 256. -/
theorem top_digit_lt (k : Nat) (h : k < 4294967296) : k / 16777216 < 256 := by omega

/-- Four passes of 8 bits consume exactly 32 bits. -/
theorem passes_cover : 4 * 8 = 32 := by decide

/-- Per-thread tie quota: the part of the budget left after the earlier threads, clamped to `eq`. -/
def takeQuota (budget before eq : Nat) : Nat := min eq (budget - before)

/-- Quotas telescope: `min budget (before+eq) = min budget before + quota`. -/
theorem quota_telescopes (budget before eq : Nat) :
    min budget (before + eq) = min budget before + takeQuota budget before eq := by
  unfold takeQuota; omega

/-- A quota never exceeds the thread's own tie count. -/
theorem quota_le_eq (budget before eq : Nat) : takeQuota budget before eq ≤ eq := by
  unfold takeQuota; omega

/-- When earlier threads already used the budget, the quota is zero. -/
theorem quota_zero (budget before eq : Nat) (h : budget ≤ before) : takeQuota budget before eq = 0 := by
  unfold takeQuota; omega

/-- If total ties reach the budget, the final prefix hits the budget exactly. -/
theorem quota_total (budget before eq : Nat) (h : budget ≤ before + eq) :
    min budget before + takeQuota budget before eq = budget := by
  unfold takeQuota; omega

/-- Selected count of a thread is `gt + quota`; with the global counts it equals the width. -/
theorem selected_total (cgt budget width : Nat) (h : budget = width - cgt) (hg : cgt ≤ width) :
    cgt + budget = width := by omega

/-- Thread partitioning of the block range: `per = ceil(nb/256)` makes 256 threads cover `nb` blocks. -/
theorem per_thread_cover (nb per : Nat) (h : per = (nb + 255) / 256) : nb ≤ 256 * per := by omega

end Ref.QsaSelect

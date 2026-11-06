/-!
# MODULE: qsa_topk_order_key
SOURCES: src/kernels/cuda/qsa.cu, src/kernels/cuda/qsa_select.cu
The monotone 32-bit order key used by the exact top-k threshold search, the selection-count arithmetic
(strictly-above plus first equal cells by ascending index) and the per-block cell weights of the
block-level radix select.

## Source semantics
* `order_key(b) = b & 0x80000000 ? ~b : b | 0x80000000` is a total order on float bit patterns.
* The selection emits all cells strictly above the threshold plus `width - c_gt` threshold-equal cells.
* A block of the block-level select weighs `r` cells, except the last (tail) block which weighs
  `n_kv - n_bid * r`; weights sum to `n_kv`.

## Form template identifiers
orderKey <- order_key, selCount <- gt + min(eq, eq_budget), eqBudget <- eq_budget = width - c_gt,
blockWeight <- weight(b) lambda in block_topk_kernel.
-/

namespace Strata.QsaTopk

-- § order key (32-bit patterns as Nat)

/-- `order_key` on a bit pattern `b < 2^32`; `~b` is `2^32 - 1 - b`. -/
def orderKey (b : Nat) : Nat :=
  if b ≥ 2147483648 then 4294967295 - b else b + 2147483648

/-- Keys stay 32-bit. -/
theorem orderKey_lt {b : Nat} (h : b < 4294967296) : orderKey b < 4294967296 := by
  unfold orderKey; split <;> omega

/-- Non-negative patterns keep their order. -/
theorem orderKey_mono_nonneg {a b : Nat} (hab : a ≤ b) (hb : b < 2147483648) :
    orderKey a ≤ orderKey b := by
  unfold orderKey; split <;> split <;> omega

/-- Negative patterns reverse their order (larger magnitude is smaller). -/
theorem orderKey_mono_neg {a b : Nat} (ha : 2147483648 ≤ a) (hab : a ≤ b)
    (hb : b < 4294967296) : orderKey b ≤ orderKey a := by
  have hb' : 2147483648 ≤ b := by omega
  have h1 : orderKey a = 4294967295 - a := by unfold orderKey; rw [if_pos ha]
  have h2 : orderKey b = 4294967295 - b := by unfold orderKey; rw [if_pos hb']
  omega

/-- Every negative pattern sorts below every non-negative one. -/
theorem orderKey_neg_lt_nonneg {a b : Nat} (ha : 2147483648 ≤ a) (ha' : a < 4294967296)
    (hb : b < 2147483648) : orderKey a < orderKey b := by
  unfold orderKey; split <;> split <;> omega

/-- Finite patterns map strictly above 0, the key reserved for NaN. -/
theorem orderKey_finite_pos {b : Nat} (h : b < 4286578688) : 0 < orderKey b := by
  unfold orderKey; split <;> omega

-- § selection count

/-- Cells emitted: those strictly above the threshold plus the equal cells that fit. -/
def selCount (gt eq width : Nat) : Nat := gt + min eq (width - gt)

/-- Equal cells the selection has room for. -/
def eqBudget (gt width : Nat) : Nat := width - gt

/-- When the threshold is the width-th largest key, exactly `width` cells are emitted. -/
theorem selCount_exact {gt eq width : Nat} (h1 : gt ≤ width) (h2 : width ≤ gt + eq) :
    selCount gt eq width = width := by
  unfold selCount; omega

/-- With distinct scores (`gt = width`) the budget at the threshold is zero. -/
theorem eqBudget_zero {gt width : Nat} (h : width ≤ gt) : eqBudget gt width = 0 := by
  unfold eqBudget; omega

/-- Never more than `width` cells are emitted when `gt <= width`. -/
theorem selCount_le_width {gt eq width : Nat} (h1 : gt ≤ width) : selCount gt eq width ≤ width := by
  unfold selCount; omega

-- § block weights

/-- Weight (cells) of block `b`. -/
def blockWeight (nKv nBid r b : Nat) : Nat := if b < nBid then r else nKv - nBid * r

/-- Prefix sum of block weights. -/
def weightSum (nKv nBid r : Nat) : Nat → Nat
  | 0 => 0
  | m + 1 => weightSum nKv nBid r m + blockWeight nKv nBid r m

/-- The first `m <= n_bid` blocks weigh `m * r`. -/
theorem weightSum_full {nKv nBid r m : Nat} (hm : m ≤ nBid) :
    weightSum nKv nBid r m = m * r := by
  induction m with
  | zero => simp [weightSum]
  | succ m ih =>
    have h1 := ih (by omega)
    have h2 : m < nBid := by omega
    have h3 : (m + 1) * r = m * r + r := by rw [Nat.add_mul, Nat.one_mul]
    simp [weightSum, blockWeight, h2, h1] <;> omega

/-- All `n_bid + 1` blocks together weigh exactly `n_kv`. -/
theorem weightSum_total {nKv nBid r : Nat} (h : nBid * r ≤ nKv) :
    weightSum nKv nBid r (nBid + 1) = nKv := by
  have h1 := weightSum_full (nKv := nKv) (nBid := nBid) (r := r) (m := nBid) (Nat.le_refl _)
  simp [weightSum, blockWeight, h1] <;> omega

/-- The tail block is empty when `n_kv` is a multiple of `r`. -/
theorem tail_weight_zero {nKv nBid r : Nat} (h : nKv = nBid * r) :
    blockWeight nKv nBid r nBid = 0 := by
  simp [blockWeight, h]

end Strata.QsaTopk

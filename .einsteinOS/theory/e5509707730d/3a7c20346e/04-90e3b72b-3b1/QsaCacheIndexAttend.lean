/-!
# MODULE: qsa_cache_index_attend
SOURCES: src/kernels/cuda/qsa.cu
The QSA cache, indexer and attention: paged KV append, indexer key pooling with a spare slot, per-block
scores mapped onto cells, exact top-k by threshold search with an ascending tie rule, KV gather and a
one-block-per-head attention, plus the step-state (pos, n_kv, n_bid, width) derived from one position.

## Source semantics
* From `pos`: `n_kv = pos+1`, `n_bid = n_kv / r`, `width = min(n_kv, top_k + r - 1)`.
* Cells of block `b < n_bid` are `[b*r, b*r+r)`; the incomplete tail is `[n_bid*r, n_kv)` and is biased +1e9.
* top-k = every cell strictly above the width-th key, then the first `width - c_gt` cells equal to it.
* The f32 order key maps -0 to +0, NaN to 0, and flips negatives so unsigned comparison orders floats.
* GQA pairing is `kv = h / (n_head / n_head_kv)` (NOT `h % n_head_kv`).
* gate uses the SECOND half of the doubled query row.

## Form template identifiers
* `selWidth` -> selection_width, `stepNBid` -> completed_blocks, `orderKey` -> float_order_key.
-/

namespace Strata.Qsa

/-- Selection width: `min(n_kv, top_k + r - 1)`. -/
def selWidth (n topk r : Nat) : Nat := min n (topk + r - 1)

/-- The width never exceeds the cell count. -/
theorem selWidth_le_n (n topk r : Nat) : selWidth n topk r ≤ n := by unfold selWidth; omega

/-- The width never exceeds `top_k + r - 1`. -/
theorem selWidth_le_cap (n topk r : Nat) : selWidth n topk r ≤ topk + r - 1 := by unfold selWidth; omega

/-- Below the bound the selection is the identity (everything selected). -/
theorem selWidth_identity (n topk r : Nat) (h : n ≤ topk + r - 1) : selWidth n topk r = n := by
  unfold selWidth; omega

/-- With `top_k = 2048`, `r = 4` the cap is 2051 (matches the literal in the native score kernel). -/
theorem selWidth_real (n : Nat) : selWidth n 2048 4 = if n < 2051 then n else 2051 := by
  unfold selWidth; split <;> omega

/-- Completed blocks for `r = 4`. -/
def stepNBid (pos : Nat) : Nat := (pos + 1) / 4

/-- Completed blocks cover no more than the cells present. -/
theorem nbid_cells_le (pos : Nat) : stepNBid pos * 4 ≤ pos + 1 := by unfold stepNBid; omega

/-- The incomplete tail is shorter than one block. -/
theorem tail_lt_block (pos : Nat) : pos + 1 - stepNBid pos * 4 < 4 := by unfold stepNBid; omega

/-- A tail exists iff `n_kv % 4 ≠ 0` (the +1e9 bias condition). -/
theorem tail_iff (pos : Nat) : stepNBid pos * 4 < pos + 1 ↔ (pos + 1) % 4 ≠ 0 := by
  unfold stepNBid; omega

/-- Cells of a completed block lie below `n_kv`. -/
theorem full_block_cells_lt (pos b c : Nat) (hb : b < stepNBid pos) (hc : c < 4) : b * 4 + c < pos + 1 := by
  unfold stepNBid at hb; omega

/-- Every cell is in exactly one block: block `cell / 4`, which is `<= n_bid`. -/
theorem cell_block_le (pos cell : Nat) (h : cell < pos + 1) : cell / 4 ≤ stepNBid pos := by
  unfold stepNBid; omega

/-- Counts: with `c_gt < w <= c_gt + c_eq` exactly `w` cells are selected. -/
theorem topk_selected_count (cgt ceq w : Nat) (h1 : cgt < w) (h2 : w ≤ cgt + ceq) :
    cgt + min (w - cgt) ceq = w := by omega

/-- The tie budget is positive exactly when the strict-above set is short of the width. -/
theorem eq_budget_pos (cgt w : Nat) (h : cgt < w) : 0 < w - cgt := by omega

/-- Order key of a float bit pattern `b < 2^32`: positives get the top bit set, negatives are complemented. -/
def orderKey (b : Nat) : Nat := if b < 2147483648 then b + 2147483648 else 4294967295 - b

/-- Keys are 32-bit values. -/
theorem orderKey_lt (b : Nat) (h : b < 4294967296) : orderKey b < 4294967296 := by
  unfold orderKey; split <;> omega

/-- Among non-negative floats the key is monotone in the bit pattern. -/
theorem orderKey_mono_pos (a b : Nat) (ha : a < 2147483648) (hb : b < 2147483648) (h : a ≤ b) :
    orderKey a ≤ orderKey b := by
  unfold orderKey; rw [if_pos ha, if_pos hb]; omega

/-- Among negative floats a larger magnitude (bit pattern) gives a smaller key. -/
theorem orderKey_anti_neg (a b : Nat) (ha : 2147483648 ≤ a) (hb : 2147483648 ≤ b)
    (hb' : b < 4294967296) (h : a ≤ b) : orderKey b ≤ orderKey a := by
  unfold orderKey; rw [if_neg (by omega), if_neg (by omega)]; omega

/-- Every negative key is below every non-negative key. -/
theorem orderKey_neg_lt_pos (a b : Nat) (ha : 2147483648 ≤ a) (ha' : a < 4294967296)
    (hb : b < 2147483648) : orderKey a < orderKey b := by
  unfold orderKey; rw [if_neg (by omega), if_pos hb]; omega

/-- Page-local slot is below the page size. -/
theorem slot_lt (pos ps : Nat) (h : 0 < ps) : pos % ps < ps := Nat.mod_lt _ h

/-- Indexer: a block completes exactly when `slot = r-1 = 3`. -/
theorem block_complete_iff (pos : Nat) : pos % 4 = 3 ↔ (pos + 1) % 4 = 0 := by omega

/-- Completing a block, the spare row `b+1` is within `max_cells/4 + 1` pooled rows. -/
theorem spare_row_in_pool (pos maxc : Nat) (hp : pos < maxc) (h3 : pos % 4 = 3) :
    pos / 4 + 1 < maxc / 4 + 1 := by omega

/-- The tail has `r-1 = 3` rows: every stored slot indexes inside `3*idx_dim` floats (idx_dim = 128). -/
theorem tail_index_lt (slot d : Nat) (hs : slot < 3) (hd : d < 128) : slot * 128 + d < 3 * 128 := by omega

/-- GQA pairing with 24 heads and 2 KV heads (group of 12) is a valid KV head. -/
theorem gqa_kv_lt (h : Nat) (hh : h < 24) : h / 12 < 2 := by omega

/-- The division pairing differs from a modulo pairing (head 1 maps to kv 0, not 1). -/
theorem gqa_not_mod : (1 : Nat) / 12 ≠ 1 % 2 := by decide

/-- The gate half starts exactly one head_dim into the doubled row (head_dim = 256). -/
theorem gate_second_half (h d : Nat) (hd : d < 256) :
    h * 512 + 256 ≤ h * 512 + 256 + d ∧ h * 512 + 256 + d < (h + 1) * 512 := by omega

/-- Gather decomposition (2 KV heads, 64 uint2 per row): flat index splits into id, head, quad. -/
theorem gather_decomp (i : Nat) :
    i / 128 * 128 + (i % 128) / 64 * 64 + (i % 128) % 64 = i := by omega

/-- Gather: with `i < n_ids * 128` the id is `< n_ids`. -/
theorem gather_id_lt (i n : Nat) (h : i < n * 128) : i / 128 < n := by omega

/-- Gather: the head part is below 2 and the quad below 64. -/
theorem gather_parts_lt (i : Nat) : (i % 128) / 64 < 2 ∧ (i % 128) % 64 < 64 := by omega

/-- Launch ceil: a grid of `ceil(n/256)` blocks covers `n` threads. -/
theorem grid_for_covers (n : Nat) : n ≤ (n + 255) / 256 * 256 := by omega

/-- Attention shared memory per head is `(max_ids + 32)` floats; the 32 reduction slots precede the weights. -/
theorem attend_smem (max_ids : Nat) : (max_ids + 32) * 4 = max_ids * 4 + 128 := by omega

end Strata.Qsa

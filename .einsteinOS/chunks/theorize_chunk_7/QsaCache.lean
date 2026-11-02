/-!
# MODULE: qsa_cache
Step-state derivation, paged KV addressing, indexer key pooling slots and the exact top-k selection counts of the QSA cache kernels.

## Source semantics
From a position pos: n_kv = pos+1, n_bid = n_kv / idx_block, width = min(n_kv, top_k + r - 1). A cell with pos%4 = 3 completes a block (pooled row b = pos/4). The top-k keeps every cell strictly above the threshold key and the first (width - c_gt) cells equal to it, giving exactly width ids. The order key maps float bits to an unsigned total order.

## Form template identifiers
qsa_step_fill -> step_fill, order_key -> order_key, topk_kernel -> topk_select
-/
namespace Strata.QsaCache

-- § step fill

/-- Selection width. -/
def selWidth (nkv topK r : Nat) : Nat := min nkv (topK + r - 1)

/-- Width never exceeds the cache length. -/
theorem width_le_nkv (nkv topK r : Nat) : selWidth nkv topK r ≤ nkv := by unfold selWidth; omega

/-- Width never exceeds top_k + r - 1. -/
theorem width_le_cap (nkv topK r : Nat) : selWidth nkv topK r ≤ topK + r - 1 := by unfold selWidth; omega

/-- Below the cap the selection is the identity. -/
theorem width_identity (nkv topK r : Nat) (h : nkv ≤ topK + r - 1) : selWidth nkv topK r = nkv := by unfold selWidth; omega

/-- Positive length gives positive width when the cap is positive. -/
theorem width_pos (nkv topK r : Nat) (h : 0 < nkv) (hc : 0 < topK + r - 1) : 0 < selWidth nkv topK r := by unfold selWidth; omega

-- § paged addressing

/-- Page and slot recompose the position. -/
theorem page_slot (pos ps : Nat) : ps * (pos / ps) + pos % ps = pos := Nat.div_add_mod pos ps

/-- The slot is inside the page. -/
theorem slot_lt (pos ps : Nat) (h : 0 < ps) : pos % ps < ps := Nat.mod_lt _ h

-- § indexer pooling (block size 4)

/-- A cell completes a block exactly when its slot is 3. -/
theorem completes_iff (pos : Nat) : pos % 4 = 3 ↔ (pos + 1) % 4 = 0 := by omega

/-- On completion the number of completed blocks is the block index plus one. -/
theorem completed_count (pos : Nat) (h : pos % 4 = 3) : (pos + 1) / 4 = pos / 4 + 1 := by omega

/-- Slots below 3 are stored in the raw tail; the completing slot is not. -/
theorem tail_slot (pos : Nat) (h : pos % 4 < 3) : pos % 4 ≠ 3 := by omega

/-- The rotation position of a block is its first cell's position. -/
def blockPos (posBase b : Nat) : Nat := posBase + b * 4

/-- Consecutive blocks are 4 positions apart. -/
theorem blockPos_succ (posBase b : Nat) : blockPos posBase (b + 1) = blockPos posBase b + 4 := by unfold blockPos; omega

-- § top-k counts

/-- Equal-to-threshold budget. -/
def eqBudget (width cgt : Nat) : Nat := width - cgt

/-- Strictly-greater cells plus the equal cells taken total exactly width. -/
theorem select_total (width cgt ceq : Nat) (h1 : cgt < width) (h2 : width ≤ cgt + ceq) : cgt + min ceq (eqBudget width cgt) = width := by unfold eqBudget; omega

/-- The threshold search guarantees the budget is positive. -/
theorem budget_pos (width cgt : Nat) (h : cgt < width) : 0 < eqBudget width cgt := by unfold eqBudget; omega

-- § order key

/-- Order key of float bits: negatives flip all bits, non-negatives set the sign bit. -/
def orderKey (b : BitVec 32) : BitVec 32 := if b.msb = true then ~~~b else b ||| 0x80000000#32

/-- Non-negative floats order like their bits. -/
theorem key_nonneg_mono (a b : BitVec 32) (ha : a.msb = false) (hb : b.msb = false) (h : a ≤ b) : orderKey a ≤ orderKey b := by sorry

/-- Every negative key is below every non-negative key. -/
theorem key_sign_sep (a b : BitVec 32) (ha : a.msb = true) (hb : b.msb = false) : orderKey a < orderKey b := by sorry

end Strata.QsaCache

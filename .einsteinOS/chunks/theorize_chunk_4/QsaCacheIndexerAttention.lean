/-!
# MODULE: qsa_cache_indexer_attention
QSA paged KV append, block-pooled indexer keys, indexer scoring, exact top-k by threshold search, gather, attention.

## Source semantics
Step state is derived from one position: n_kv = pos+1, n_bid = n_kv / r, width = min(n_kv, top_k + r - 1).
A pooled block completes when `pos % r = r-1`. Top-k picks all cells above the threshold key plus the first
`width - c_gt` equal cells by ascending index. Query head h uses kv head `h / (n_head / n_head_kv)`.

## Form template identifiers
qsa -> sparse_block_attention; kStep* -> step_state fields; r -> idx_block (4 in practice)
-/
namespace Strata.Qsa

/-- Selection width. -/
def selWidth (nKv topK r : Nat) : Nat := min nKv (topK + r - 1)

/-- Width never exceeds the number of cells. -/
theorem selWidth_le (nKv topK r : Nat) : selWidth nKv topK r ≤ nKv := by unfold selWidth; omega

/-- Below the bound the selection is the identity. -/
theorem selWidth_identity (nKv topK r : Nat) (h : nKv ≤ topK + r - 1) : selWidth nKv topK r = nKv := by
  unfold selWidth; omega

/-- Threshold selection emits exactly `width` cells: all above, plus the equal budget. -/
theorem selection_count (gt eq width : Nat) (h1 : gt ≤ width) (h2 : width ≤ gt + eq) :
    gt + min eq (width - gt) = width := by omega

/-- Pooled block completion for r = 4: the next position starts a new block. -/
theorem pool_complete (pos : Nat) (h : pos % 4 = 3) :
    (pos + 1) % 4 = 0 ∧ (pos + 1) / 4 = pos / 4 + 1 := by omega

/-- Completed blocks never cover more cells than exist. -/
theorem nbid_cover (pos : Nat) : ((pos + 1) / 4) * 4 ≤ pos + 1 := by omega

/-- Incomplete tail size plus completed cells equals n_kv. -/
theorem tail_size (nKv : Nat) : (nKv / 4) * 4 + nKv % 4 = nKv := by omega

/-- GQA mapping for 24 heads over 2 kv heads stays in range. -/
theorem gqa_range (h : Nat) (hh : h < 24) : h / (24 / 2) < 2 := by omega

/-- Slot of a position inside its page is in range. -/
theorem page_slot_lt (pos ps : Nat) (hp : 0 < ps) : pos % ps < ps := Nat.mod_lt _ hp

/-- Pool of positions: page index times page size plus slot reconstructs the position. -/
theorem page_decomp (pos ps : Nat) : (pos / ps) * ps + pos % ps = pos := by
  have := Nat.div_add_mod pos ps
  rw [Nat.mul_comm] at this
  omega

/-- Uint2 gather copies four halves at a time, so head_dim must be a multiple of 4. -/
theorem uint2_per (hd : Nat) (h : hd % 4 = 0) : (hd / 4) * 4 = hd := by omega

end Strata.Qsa

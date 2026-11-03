/-!
# MODULE: qsa_attention_contracts
Cache layout, selection width, per-token step state, GQA head map, tie-broken top-k order and gate split of the
QSA (sparse full-attention) layer.

## Source semantics
* `qsa_selection_width(n_kv) = min(n_kv, idx_top_k + idx_block - 1)`; below the bound selection is the identity.
* `step = [pos, pos+1, (pos+1)/idx_block, width]` is derived from `pos` in one place.
* Physical KV row: `(page*n_head_kv + h)*page_size + t % page_size`, page = `table[t / page_size]`.
* q->kv head map is integer division `q / (n_head/n_head_kv)`, not `q % n_head_kv`.
* top-k ties break by ascending index.
* Gate is the second half of each head's `2*head_dim` block.

## Form template identifiers
* qsa_selection_width -> `selection_width`
* qsa_step_fill -> `step_fill`
* cell_block / n_bid -> `cell_block`, `completed_blocks`
* topk_512 -> `topk_select`
-/

namespace Strata.Qsa

-- § Selection width

/-- Selection width. -/
def selWidth (topK block nKv : Nat) : Nat := min nKv (topK + block - 1)

/-- The width never exceeds the cache. -/
theorem selWidth_le_nKv (topK block nKv : Nat) : selWidth topK block nKv ≤ nKv := by
  unfold selWidth
  omega

/-- The width never exceeds the budget plus incomplete tail. -/
theorem selWidth_le_budget (topK block nKv : Nat) : selWidth topK block nKv ≤ topK + block - 1 := by
  unfold selWidth
  omega

/-- Below the bound the selection is the identity (all cells). -/
theorem selWidth_identity (topK block nKv : Nat) (h : nKv ≤ topK + block - 1) :
    selWidth topK block nKv = nKv := by
  unfold selWidth
  omega

/-- For the real geometry the bound is 2051, not 2048. -/
theorem real_bound : 2048 + 4 - 1 = 2051 := by decide

/-- At 2051 cells the selection is still the identity; at 2052 it is 2051. -/
theorem real_width_edges : selWidth 2048 4 2051 = 2051 ∧ selWidth 2048 4 2052 = 2051 := by decide

-- § Step state

/-- Step buffer in the order pos, n_kv, n_bid, width. -/
def stepFill (topK block pos : Nat) : List Nat :=
  [pos, pos + 1, (pos + 1) / block, selWidth topK block (pos + 1)]

/-- The step buffer has four entries. -/
theorem stepFill_length (topK block pos : Nat) : (stepFill topK block pos).length = 4 := by
  unfold stepFill
  rfl

/-- Completed blocks never cover more than the cached cells. -/
theorem completed_blocks_le (block pos : Nat) (hb : 0 < block) : ((pos + 1) / block) * block ≤ pos + 1 := by
  exact Nat.div_mul_le_self (pos + 1) block

/-- The incomplete tail has fewer than `block` cells. -/
theorem tail_lt_block (block pos : Nat) (hb : 0 < block) : (pos + 1) - ((pos + 1) / block) * block < block := by
  have h1 := Nat.div_add_mod (pos + 1) block
  have h2 := Nat.mod_lt (pos + 1) hb
  rw [Nat.mul_comm block ((pos + 1) / block)] at h1
  omega

/-- Step buffer bytes: four int32. -/
theorem step_bytes : 4 * 4 = 16 := by decide

-- § Paged KV layout

/-- Logical cell split into page and in-page slot recombines exactly. -/
theorem page_split (t pageSize : Nat) (hp : 0 < pageSize) : (t / pageSize) * pageSize + t % pageSize = t := by
  have h := Nat.div_add_mod t pageSize
  rw [Nat.mul_comm pageSize (t / pageSize)] at h
  exact h

/-- The in-page slot is within the page. -/
theorem slot_lt (t pageSize : Nat) (hp : 0 < pageSize) : t % pageSize < pageSize := Nat.mod_lt _ hp

/-- Physical row of cell `t`, head `h`. -/
def physRow (page nHeadKv h pageSize t : Nat) : Nat := (page * nHeadKv + h) * pageSize + t % pageSize

/-- Page size 1 is the unpaged configuration: the row is just `page*n_head_kv + h`. -/
theorem physRow_unpaged (page nHeadKv h t : Nat) : physRow page nHeadKv h 1 t = page * nHeadKv + h := by
  unfold physRow
  have : t % 1 = 0 := Nat.mod_one t
  rw [this]
  omega

/-- Same page and head, adjacent cells give adjacent rows (a head's cells are contiguous in a page). -/
theorem physRow_contiguous (page nHeadKv h pageSize t : Nat) (h1 : (t + 1) % pageSize ≠ 0)
    (hp : 0 < pageSize) :
    physRow page nHeadKv h pageSize (t + 1) = physRow page nHeadKv h pageSize t + 1 := by
  unfold physRow
  have hd := Nat.div_add_mod t pageSize
  have hm := Nat.mod_lt t hp
  have hlt : t % pageSize + 1 < pageSize := by
    by_contra hc
    have he : t % pageSize + 1 = pageSize := by omega
    apply h1
    have h2 : t + 1 = pageSize * (t / pageSize + 1) := by
      rw [Nat.mul_add]
      omega
    rw [h2]
    exact Nat.mul_mod_right pageSize _
  have h3 : t + 1 = (t % pageSize + 1) + pageSize * (t / pageSize) := by omega
  have a : (t + 1) % pageSize = t % pageSize + 1 := by
    rw [h3, Nat.add_mul_mod_self_left]
    exact Nat.mod_eq_of_lt hlt
  rw [a]
  omega

/-- 32K context cache size: 12 layers x 2 heads x 256 dims x 2 (K,V) x 2 bytes x 32768 cells. -/
theorem kv_cache_bytes : 12 * 2 * 256 * 2 * 2 * 32768 = 805306368 := by decide

-- § GQA head map

/-- The integer-division map sends the 24 query heads onto 2 kv heads in two blocks of 12. -/
theorem gqa_in_range : ∀ q : Fin 24, q.val / 12 < 2 := by decide

/-- It is not the modulo map: query head 1 maps to kv head 0 but `1 % 2 = 1`. -/
theorem gqa_not_mod : (1 : Nat) / 12 ≠ 1 % 2 := by decide

/-- The two maps agree on a block-boundary-free example, so a single test with q = 0 cannot tell them apart. -/
theorem gqa_agree_at_zero : (0 : Nat) / 12 = 0 % 2 := by decide

-- § Top-k ordering

/-- `a` precedes `b`: larger score first, ties by ascending index. -/
def before (a b : Int × Nat) : Prop := b.1 < a.1 ∨ (a.1 = b.1 ∧ a.2 < b.2)

/-- No element precedes itself. -/
theorem before_irrefl (a : Int × Nat) : ¬ before a a := by
  unfold before
  omega

/-- Precedence is asymmetric. -/
theorem before_asymm (a b : Int × Nat) (h : before a b) : ¬ before b a := by
  unfold before at *
  omega

/-- Elements with different indices are always comparable, so the order is total on cells. -/
theorem before_total (a b : Int × Nat) (hne : a.2 ≠ b.2) : before a b ∨ before b a := by
  unfold before
  omega

/-- Precedence is transitive. -/
theorem before_trans (a b c : Int × Nat) (h1 : before a b) (h2 : before b c) : before a c := by
  unfold before at *
  omega

-- § Tail bias and gate split

/-- Per-cell block map: cells of completed blocks map to their block, tail cells to the spare slot `n_bid`. -/
def cellBlock (j block nBid : Nat) : Nat := min (j / block) nBid

/-- The block index is within the pooled rows (0..n_bid). -/
theorem cellBlock_le (j block nBid : Nat) : cellBlock j block nBid ≤ nBid := by
  unfold cellBlock
  omega

/-- A tail cell (j at or past the last completed block's end) maps to the spare slot. -/
theorem cellBlock_tail (j block nBid : Nat) (hb : 0 < block) (h : nBid * block ≤ j) :
    cellBlock j block nBid = nBid := by
  unfold cellBlock
  have : nBid ≤ j / block := (Nat.le_div_iff_mul_le hb).2 h
  omega

/-- Gate element index `head_dim + d` is inside the head's `2*head_dim` block and past the query half. -/
theorem gate_index (headDim d : Nat) (hd : d < headDim) : headDim ≤ headDim + d ∧ headDim + d < 2 * headDim := by
  omega

/-- Top-k single-block cap: the largest width at 32768 cells is 2051. -/
theorem topk_cap : selWidth 2048 4 32768 = 2051 := by decide

end Strata.Qsa

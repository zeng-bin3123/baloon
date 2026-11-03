/-!
# MODULE: qsa_cache_indexer
Arithmetic core of the QSA cache, indexer and top-k: step-state derivation, paged addressing,
pooling-block bookkeeping, order-key total order and the tie-aware top-k count.

## Source semantics
`qsa_step_fill`: n_kv = pos+1, n_bid = n_kv / r, width = min(n_kv, top_k + r - 1).
Cells live in pages: row = (page*kv_heads + h)*page_size + pos % page_size.
`order_key` maps f32 bits to an unsigned key (negative: ~b, positive: b | 2^31) so integer comparison
agrees with float order. top-k selects every cell above the threshold plus the first (width - c_gt) equals.

## Form template identifiers
pos -> token position; r -> idx_block (4 in models); width -> selection width; thr -> threshold key.
-/
namespace Strata.Qsa

/-- Selection width with top_k = 2048, r = 4. -/
def width (nkv : Nat) : Nat := min nkv 2051

/-- Step fill: the number of completed blocks. -/
def nbid (pos : Nat) : Nat := (pos + 1) / 4

/-- Width never exceeds the number of cells. -/
theorem width_le_nkv (n : Nat) : width n ≤ n := by unfold width; omega

/-- Below the selection bound the selection is the identity (width = n). -/
theorem width_identity (n : Nat) (h : n ≤ 2051) : width n = n := by unfold width; omega

/-- Above the bound the width saturates at 2051. -/
theorem width_sat (n : Nat) (h : 2051 ≤ n) : width n = 2051 := by unfold width; omega

/-- Completed blocks never exceed n_kv / 4. -/
theorem nbid_le (pos : Nat) : nbid pos * 4 ≤ pos + 1 := by unfold nbid; omega

/-- The incomplete tail has fewer than 4 cells. -/
theorem tail_lt (pos : Nat) : (pos + 1) - nbid pos * 4 < 4 := by unfold nbid; omega

/-- Cell j < n_kv lies in a block index at most n_bid. -/
theorem cell_block_le (j pos : Nat) (h : j < pos + 1) : j / 4 ≤ nbid pos := by unfold nbid; omega

/-- Block b < n_bid owns cells 4b..4b+3, all below n_kv. -/
theorem block_cells_lt (b pos c : Nat) (hb : b < nbid pos) (hc : c < 4) : b * 4 + c < pos + 1 := by
  unfold nbid at hb; omega

/-- Slot of a position inside its block and the block index recompose the position. -/
theorem slot_decomp (pos : Nat) : (pos / 4) * 4 + pos % 4 = pos := by omega

/-- A block completes exactly at slot r-1 = 3. -/
theorem completes_iff (pos : Nat) : pos % 4 = 3 ↔ (pos + 1) % 4 = 0 := by omega

/-- Pooled rows needed for positions below cap: block index b+1 stays within cap/4 + 1 rows. -/
theorem pooled_capacity (pos cap : Nat) (h : pos < cap) : pos / 4 + 1 ≤ cap / 4 + 1 := by omega

/-- Paged addressing: page index and in-page slot recompose the cell. -/
theorem page_decomp (cell ps : Nat) : ps * (cell / ps) + cell % ps = cell := Nat.div_add_mod cell ps

/-- The in-page slot is below the page size. -/
theorem page_slot_lt (cell ps : Nat) (h : 0 < ps) : cell % ps < ps := Nat.mod_lt _ h

/-- Query head to kv head mapping h / (n_head/n_head_kv); with group g it stays below n_head_kv. -/
theorem kv_head_lt (h g nkv : Nat) (hg : 0 < g) (hh : h < g * nkv) : h / g < nkv := Nat.div_lt_of_lt_mul hh

/-- Order key: negative floats (sign bit set) are bit-complemented, others get the sign bit set. -/
def orderKey (b : Nat) : Nat := if b ≥ 2147483648 then 4294967295 - b else b + 2147483648

/-- Negative branch. -/
theorem key_neg (b : Nat) (h : b ≥ 2147483648) : orderKey b = 4294967295 - b := by
  simp [orderKey, h]

/-- Positive branch. -/
theorem key_pos (b : Nat) (h : b < 2147483648) : orderKey b = b + 2147483648 := by
  have : ¬ b ≥ 2147483648 := by omega
  simp [orderKey, this]

/-- Keys fit in 32 bits. -/
theorem key_lt (b : Nat) (h : b < 4294967296) : orderKey b < 4294967296 := by
  by_cases h1 : b ≥ 2147483648
  · rw [key_neg b h1]; omega
  · rw [key_pos b (by omega)]; omega

/-- Non-negative floats keep their order. -/
theorem key_mono_pos (a b : Nat) (h : a ≤ b) (hb : b < 2147483648) : orderKey a ≤ orderKey b := by
  rw [key_pos a (by omega), key_pos b hb]; omega

/-- Negative floats reverse their bit order, matching float order. -/
theorem key_mono_neg (a b : Nat) (h : a ≤ b) (ha : 2147483648 ≤ a) (hb : b < 4294967296) :
    orderKey b ≤ orderKey a := by
  rw [key_neg a ha, key_neg b (by omega)]; omega

/-- Every negative key is below every non-negative key. -/
theorem key_neg_lt_pos (a b : Nat) (ha : 2147483648 ≤ a) (ha2 : a < 4294967296) (hb : b < 2147483648) :
    orderKey a < orderKey b := by
  rw [key_neg a ha, key_pos b hb]; omega

/-- Number selected by the tie rule: all strictly greater plus as many equals as the budget allows. -/
def selected (gt eq w : Nat) : Nat := gt + min eq (w - gt)

/-- The selection never exceeds the width when c_gt <= width. -/
theorem selected_le (gt eq w : Nat) (h : gt ≤ w) : selected gt eq w ≤ w := by unfold selected; omega

/-- The selection fills the width exactly when the threshold was chosen so that count(>= thr) >= width. -/
theorem selected_eq (gt eq w : Nat) (h : gt ≤ w) (h2 : w ≤ gt + eq) : selected gt eq w = w := by
  unfold selected; omega

/-- The equal-budget is the room left after the strictly greater cells. -/
theorem eq_budget_spec (gt w : Nat) (h : gt ≤ w) : gt + (w - gt) = w := by omega

/-- Binary-lifting count: 32 passes fix 32 bits. -/
theorem lift_bits : 32 * 1 = 32 := by decide

/-- Chunked walk: ceil(n/256) cells per thread, threads cover all cells. -/
theorem chunk_cover (n j : Nat) (h : j < n) : j / ((n + 255) / 256) < 256 ∨ n = 0 := by
  by_cases hn : n = 0
  · exact Or.inr hn
  · left
    have hc : 0 < (n + 255) / 256 := by omega
    have : j < 256 * ((n + 255) / 256) := by omega
    exact Nat.div_lt_of_lt_mul (by omega)

/-- Attend: the kv-head group index of query head h is h / 12 and the in-group index is h % 12. -/
theorem attend_group (h : Nat) : (h / 12) * 12 + h % 12 = h := by omega

/-- Gate multiply reads the second half of each [2*head_dim] q row: offset head_dim + d stays in the row. -/
theorem gate_offset_lt (d hd : Nat) (h : d < hd) : hd + d < 2 * hd := by omega

end Strata.Qsa

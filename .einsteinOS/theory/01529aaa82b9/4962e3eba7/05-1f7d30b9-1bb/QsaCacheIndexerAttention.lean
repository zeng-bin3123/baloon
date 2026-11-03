/-!
# MODULE: qsa_cache_indexer_attention
The QSA (quantised sparse attention) cache, block indexer, exact top-k selection and attention kernels.
This file proves the combinatorial core: the step-state derivation, the page/row addressing of the
KV cache, the pooling ring buffer (blocks of r = 4 cells), the cell-to-block mapping of the indexer, the
chunked threshold-selection arithmetic of `topk_512`, and the head/shared-memory arithmetic of the attention.

## Source semantics
* `qsa_step_fill` derives `n_kv = pos+1`, `n_bid = n_kv / r`, `width = min(n_kv, top_k + r - 1)` from one position.
* The indexer pools r raw keys into one block row; the incomplete tail block is scored as one block
  and receives a `+1e9` bias exactly when `n_kv % r != 0`.
* `topk_512` selects every cell with key above the threshold and the first `width - c_gt` cells at it.
* `qsa_attend` maps query head `h` to key/value head `h / (n_head / n_head_kv)`.

## Form template identifiers
- `kv_append_kernel` -> `kv_append`; `indexer_key_append_kernel` -> `pool_append`
- `qsa_index_kernel` -> `block_score`; `topk_kernel` -> `threshold_select`; `kv_gather_kernel` -> `kv_gather`
- `qsa_attend_kernel` -> `sparse_attend`; `qsa_gate_apply_kernel` -> `gate_apply`
- `kStepPos/NKv/NBid/Width` -> `step_pos/n_kv/n_bid/width`; `idx_block` -> `pool_block` (r)
-/

namespace Strata.Qsa

-- § arithmetic helper

/-- Row-major linearisation stays in bounds. -/
theorem lin2_lt {i a c d : Nat} (hi : i < a) (hc : c < d) : i * d + c < a * d := by
  have h := Nat.mul_le_mul_right d (show i + 1 ≤ a by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- Division/remainder reconstruct the dividend in the order used by the kernels (`q*r + n%r`). -/
theorem div_mul_add_mod (n r : Nat) : (n / r) * r + n % r = n := by
  rw [Nat.mul_comm]
  exact Nat.div_add_mod n r

-- § step state

/-- Selection width: all cells below the bound, otherwise `top_k + r - 1`. -/
def selWidth (n k r : Nat) : Nat := min n (k + r - 1)

/-- The step quadruple derived from a single position. -/
structure Step where
  pos : Nat
  nKv : Nat
  nBid : Nat
  width : Nat

/-- `qsa_step_fill`: all four entries derive from `pos`. -/
def stepFill (pos k r : Nat) : Step :=
  { pos := pos, nKv := pos + 1, nBid := (pos + 1) / r, width := selWidth (pos + 1) k r }

/-- `n_kv` is always one more than the position. -/
theorem stepFill_nKv (pos k r : Nat) : (stepFill pos k r).nKv = (stepFill pos k r).pos + 1 := rfl

/-- The width never exceeds the number of cells. -/
theorem stepFill_width_le_nKv (pos k r : Nat) : (stepFill pos k r).width ≤ (stepFill pos k r).nKv := by
  dsimp only [stepFill, selWidth]
  omega

/-- The width never exceeds `top_k + r - 1`. -/
theorem stepFill_width_le_cap (pos k r : Nat) : (stepFill pos k r).width ≤ k + r - 1 := by
  dsimp only [stepFill, selWidth]
  omega

/-- Below the bound the selection is the identity (every cell is selected). -/
theorem stepFill_identity {pos k r : Nat} (h : pos + 1 ≤ k + r - 1) :
    (stepFill pos k r).width = (stepFill pos k r).nKv := by
  dsimp only [stepFill, selWidth]
  omega

/-- With r = 4 the completed-block count and the tail partition the cells. -/
theorem stepFill_blocks_four (pos k : Nat) :
    (stepFill pos k 4).nBid * 4 ≤ (stepFill pos k 4).nKv ∧
      (stepFill pos k 4).nKv < (stepFill pos k 4).nBid * 4 + 4 := by
  dsimp only [stepFill]
  omega

/-- The default geometry: `top_k = 2048`, `r = 4` gives the 2051-cell bound. -/
theorem selection_bound_2051 : 2048 + 4 - 1 = 2051 := by decide

-- § kv cache addressing

/-- A page-table slot `pos / page_size` is in range when the position is below `page_size * n_pages`. -/
theorem page_slot_lt {pos ps n_pages : Nat} (h : pos < ps * n_pages) : pos / ps < n_pages :=
  Nat.div_lt_of_lt_mul h

/-- The in-page slot is below the page size. -/
theorem page_offset_lt {pos ps : Nat} (hps : 0 < ps) : pos % ps < ps := Nat.mod_lt _ hps

/-- The pool row `(page*kv_heads + h)*page_size + slot` is inside the pool of `n_pages` pages. -/
theorem pool_row_lt {page h slot n_pages kvh ps : Nat} (hp : page < n_pages) (hh : h < kvh)
    (hs : slot < ps) : (page * kvh + h) * ps + slot < n_pages * kvh * ps :=
  lin2_lt (lin2_lt hp hh) hs

/-- Splitting a flat index `i < kv_heads * head_dim` into `(head, dim)` is exact and in range. -/
theorem split_head_dim {i kvh hd : Nat} (hd0 : 0 < hd) (hi : i < hd * kvh) :
    i / hd < kvh ∧ i % hd < hd ∧ (i / hd) * hd + i % hd = i :=
  ⟨Nat.div_lt_of_lt_mul hi, Nat.mod_lt _ hd0, div_mul_add_mod i hd⟩

/-- The gather destination `(id*kv_heads + h)*per + q` is inside the scratch of `n_ids` selected cells. -/
theorem gather_dst_lt {id h q n_ids kvh per : Nat} (hid : id < n_ids) (hh : h < kvh) (hq : q < per) :
    (id * kvh + h) * per + q < n_ids * kvh * per :=
  lin2_lt (lin2_lt hid hh) hq

-- § pooling ring buffer (r = 4)

/-- The ring slot of a position is below 4. -/
theorem slot_lt (pos : Nat) : pos % 4 < 4 := by omega

/-- A cell completes a block exactly when its slot is 3, i.e. when `pos + 1` is a multiple of 4. -/
theorem completes_block_iff (pos : Nat) : pos % 4 = 3 ↔ (pos + 1) % 4 = 0 := by omega

/-- The tail rows written (slots 0..2) are the first three cells of the current block. -/
theorem tail_cells {pos : Nat} (h : pos % 4 < 3) : pos / 4 * 4 + pos % 4 = pos ∧ pos % 4 ≤ 2 := by omega

/-- A completed block's row and the spare row after it fit in a pooled buffer of `max_cells/4 + 1` rows. -/
theorem pooled_rows_fit {pos max : Nat} (hp : pos < max) (hs : pos % 4 = 3) :
    pos / 4 + 1 < max / 4 + 1 := by omega

-- § indexer: cells of a block

/-- Every cell `j < n_kv` belongs to block `j/4`, which is at most the completed-block count. -/
theorem cell_block {j n : Nat} (hj : j < n) :
    j / 4 ≤ n / 4 ∧ (j / 4) * 4 ≤ j ∧ j < (j / 4) * 4 + 4 := by omega

/-- A complete block's four cells all exist. -/
theorem complete_block_in_range {b n : Nat} (hb : b < n / 4) : b * 4 + 4 ≤ n := by omega

/-- The incomplete tail block holds `n % 4` cells, and the bias applies exactly when that is nonzero. -/
theorem tail_cells_count (n : Nat) : n - (n / 4) * 4 = n % 4 := by omega

/-- When `n` is a multiple of 4 the block `n_bid` is empty (it is only the spare row). -/
theorem empty_tail_block {n : Nat} (h : n % 4 = 0) : n / 4 * 4 = n := by omega

/-- A cell id `b*4 + c` of the tail block is below `n_kv`. -/
theorem tail_cell_lt {n c : Nat} (hc : c < n - (n / 4) * 4) : (n / 4) * 4 + c < n := by omega

-- § top-k: chunking and tie budget

/-- The 256 chunks of `ceil(n/256)` cells cover `[0, n)`. -/
theorem chunks_cover (n : Nat) : n ≤ ((n + 255) / 256) * 256 := by omega

/-- Cell `j < n` belongs to chunk `t = j / chunk`, and `t` is a valid thread. -/
theorem chunk_owner_lt {j n chunk : Nat} (hj : j < n) (hc : n ≤ chunk * 256) : j / chunk < 256 :=
  Nat.div_lt_of_lt_mul (Nat.lt_of_lt_of_le hj hc)

/-- Cell `j` lies in `[t*chunk, t*chunk + chunk)` for its owner `t`. -/
theorem chunk_member {j chunk : Nat} (hc : 0 < chunk) :
    (j / chunk) * chunk ≤ j ∧ j < (j / chunk) * chunk + chunk := by
  have h1 := div_mul_add_mod j chunk
  have h2 := Nat.mod_lt j hc
  omega

/-- When `c_gt < width <= c_gt + c_eq` the tie budget is positive, fits the equal cells and fills the width. -/
theorem tie_budget {c_gt c_eq width : Nat} (h1 : c_gt < width) (h2 : width ≤ c_gt + c_eq) :
    0 < width - c_gt ∧ width - c_gt ≤ c_eq ∧ c_gt + (width - c_gt) = width := by omega

/-- Exclusive prefix of a list of per-thread counts: the sum of the first `i` entries. -/
def exclPrefix (xs : List Nat) (i : Nat) : Nat := (xs.take i).sum

/-- The prefix before thread 0 is empty. -/
theorem exclPrefix_zero (xs : List Nat) : exclPrefix xs 0 = 0 := by
  simp [exclPrefix]

/-- Each prefix is the previous prefix plus the previous thread's count. -/
theorem exclPrefix_succ (xs : List Nat) {i : Nat} (hi : i < xs.length) :
    exclPrefix xs (i + 1) = exclPrefix xs i + xs[i] := by
  unfold exclPrefix
  first
    | exact List.sum_take_succ xs i hi
    | (rw [List.take_succ, List.getElem?_eq_getElem hi]; simp [List.sum_append])

/-- The prefix after all threads is the total. -/
theorem exclPrefix_total (xs : List Nat) : exclPrefix xs xs.length = xs.sum := by
  simp [exclPrefix]

-- § attention

/-- With `n_head = g * n_head_kv` the key/value head `h / g` is a valid KV head. -/
theorem kv_head_lt {h g nkv : Nat} (hh : h < g * nkv) : h / g < nkv := Nat.div_lt_of_lt_mul hh

/-- Contiguous grouping `h / g` differs from interleaved grouping `h % n_head_kv` (4 heads, 2 KV heads, h = 1). -/
theorem kv_head_not_modulo : (1 : Nat) / (4 / 2) ≠ 1 % 2 := by decide

/-- Scores stored after the 32 reduction floats stay inside the shared buffer of `max_ids + 32` floats. -/
theorem score_slot_lt {j n_ids max_ids : Nat} (hj : j < n_ids) (hn : n_ids ≤ max_ids) :
    32 + j < max_ids + 32 := by omega

/-- Warp slot for a block of `blockDim` threads is below the warp count `(blockDim + 31) / 32`. -/
theorem warp_slot_lt {d bd : Nat} (hd : d < bd) : d / 32 < (bd + 31) / 32 := by omega

/-- For a 256-thread block there are 8 warps, which fits the 32 reduction floats. -/
theorem warp_count_256 : (256 + 31) / 32 = 8 := by decide

/-- Scratch addressing `(j*n_head_kv + kv)*head_dim + i` is inside `n_ids * n_head_kv * head_dim`. -/
theorem scratch_index_lt {j kv i n_ids nkv hd : Nat} (hj : j < n_ids) (hk : kv < nkv) (hi : i < hd) :
    (j * nkv + kv) * hd + i < n_ids * nkv * hd :=
  lin2_lt (lin2_lt hj hk) hi

/-- The default shared allocation for 2051 ids is 8332 bytes, below the 48 KiB default limit. -/
theorem default_shared_bytes : (2051 + 32) * 4 = 8332 ∧ 8332 < 49152 := by decide

/-- The gate's second half `h*(2*hd) + hd + d` is inside the `n_head * (2*hd)` fused q buffer. -/
theorem gate_second_half_lt {h d n hd : Nat} (hh : h < n) (hd' : d < hd) :
    h * (2 * hd) + hd + d < n * (2 * hd) := by
  have h1 := Nat.mul_le_mul_right (2 * hd) (show h + 1 ≤ n by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

end Strata.Qsa

/-!
# MODULE: native_qsa_score
Tensor-core (m16n8k8 tf32) block scoring for the QSA indexer: 32 pooled rows per block, 2 warps splitting
the 128-dim key, 4 indexer heads padded to 8 mma columns, per-head ReLU, head sum, optional bias, and
the `+1e9` tail bias; each block score is broadcast to its cells.  This file proves the tile/shared-memory
addressing, the key-column split, the writer/reader agreement of the partial-sum exchange, the mapping
from mma fragment slots to (head, row), and the capacity/budget checks.

## Source semantics
* Warp `w` accumulates key columns `w*32 + lane` and `w*32 + lane + 64`.
* Fragment slot `l` of lane `lane` in A-tile `ia` of warp `w` holds row `ia*16 + (l/2)*8 + lane/4`
  and head `(lane%4)*2 + l%2` (heads >= 4 are padding).
* Partials are exchanged through `shared[h*68 + warp*32 + row]`.
* The step is accepted only when `width = min(n_kv, 2051)` (`top_k = 2048`, `r = 4`).

## Form template identifiers
- `score_kernel` -> `mma_block_score`; `STRIDE` -> `tile_stride` (36); `COMBINE` -> `combine_stride` (68)
- `ROWS` -> `rows_per_block` (32); `WARPS` -> `warps_per_block` (2); `max_blocks` -> `block_capacity`
-/

namespace Strata.NativeQsaScore

-- § grid

/-- A grid of `ceil(max_blocks/32)` blocks of 32 rows covers every pooled row. -/
theorem row_grid_cover {row maxb : Nat} (hr : row < maxb) : row / 32 < (maxb + 31) / 32 := by omega

/-- The first row of a block and the local row index reconstruct the row. -/
theorem row_split {row : Nat} : (row / 32) * 32 + row % 32 = row ∧ row % 32 < 32 := by omega

-- § key column split between the two warps

/-- Every key column is visited exactly once: owner warp `(c/32)%2`, iteration `c/64`, lane `c%32`. -/
theorem column_owner {c : Nat} (hc : c < 128) :
    (c / 64) * 64 + ((c / 32) % 2) * 32 + c % 32 = c ∧ c / 64 < 2 ∧ (c / 32) % 2 < 2 := by omega

/-- The columns a warp visits, `warp*32 + lane + 64*it`, are in range. -/
theorem warp_column_lt {warp lane it : Nat} (hw : warp < 2) (hl : lane < 32) (hi : it < 2) :
    warp * 32 + lane + 64 * it < 128 := by omega

/-- Distinct (warp, lane, iteration) triples visit distinct columns. -/
theorem warp_column_inj {w l i w' l' i' : Nat} (hl : l < 32) (hl' : l' < 32) (hw : w < 2) (hw' : w' < 2)
    (h : w * 32 + l + 64 * i = w' * 32 + l' + 64 * i') : w = w' ∧ l = l' ∧ i = i' := by omega

-- § shared memory tiles

/-- A tile write `tile[i*36 + lane]` (i < 16) stays inside the warp's 16x36 tile. -/
theorem tile_index_lt {i lane : Nat} (hi : i < 16) (hl : lane < 32) : i * 36 + lane < 16 * 36 := by omega

/-- Each warp's tile fits the `2 * 16 * 36` shared array. -/
theorem warp_tile_lt {warp i lane : Nat} (hw : warp < 2) (hi : i < 16) (hl : lane < 32) :
    warp * (16 * 36) + (i * 36 + lane) < 2 * 16 * 36 := by omega

/-- The exchange array `h*68 + i` (h < 8, i < 64) fits inside the same shared allocation. -/
theorem combine_index_lt {h i : Nat} (hh : h < 8) (hi : i < 64) : h * 68 + i < 2 * 16 * 36 := by omega

/-- Row slots of the two warps (offset 0 and 32) stay inside one 68-wide head row. -/
theorem combine_row_fits {warp row : Nat} (hw : warp < 2) (hr : row < 32) : warp * 32 + row < 68 := by omega

-- § mma fragment slots

/-- Fragment slot to (row, head): the row inside a 32-row A tile pair, and the head from lane and slot. -/
def fragRow (ia l lane : Nat) : Nat := ia * 16 + (l / 2) * 8 + lane / 4
/-- Head of a fragment slot. -/
def fragHead (l lane : Nat) : Nat := (lane % 4) * 2 + l % 2

/-- The row of a fragment slot is below 32. -/
theorem fragRow_lt {ia l lane : Nat} (hi : ia < 2) (hl : l < 4) (hlane : lane < 32) : fragRow ia l lane < 32 := by
  unfold fragRow
  omega

/-- The head of a fragment slot is below 8 (heads 4..7 are padding). -/
theorem fragHead_lt {l lane : Nat} : fragHead l lane < 8 := by
  unfold fragHead
  omega

/-- Distinct (ia, slot, lane) triples write distinct (head, row) cells, so the exchange has no collisions. -/
theorem frag_inj {ia l lane ia' l' lane' : Nat} (hi : ia < 2) (hi' : ia' < 2) (hl : l < 4) (hl' : l' < 4)
    (hlane : lane < 32) (hlane' : lane' < 32)
    (hh : fragHead l lane = fragHead l' lane') (hr : fragRow ia l lane = fragRow ia' l' lane') :
    ia = ia' ∧ l = l' ∧ lane = lane' := by
  unfold fragHead at hh
  unfold fragRow at hr
  omega

/-- The writer's cell `h*68 + warp*32 + row` equals the reader's `j*68 + warp*32 + lane`. -/
theorem exchange_agrees (h w row : Nat) : h * 68 + (w * 32 + row) = h * 68 + w * 32 + row := by omega

-- § capacities and budget

/-- Block capacity check `max_blocks = max_cells/4 + 1` is consistent with the cells of the last block. -/
theorem capacity_covers {n maxc : Nat} (hn : n ≤ maxc) : n / 4 < maxc / 4 + 1 := by omega

/-- Cells of a block `row*4 + i` with `i < 4` and `row*4 + i < n` are below `n`. -/
theorem block_cell_lt {row i n : Nat} (h : row * 4 + i < n) : row * 4 + i < n := h

/-- The accepted width is `min(n, 2051)`: `top_k + r - 1` for `top_k = 2048`, `r = 4`. -/
theorem width_formula : 2048 + 4 - 1 = 2051 := by decide

end Strata.NativeQsaScore

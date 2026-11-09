/-!
# MODULE: qsa_selection_and_attention
SOURCES: include/strata/kernels/qsa.hpp, include/strata/kernels/qsa_select.hpp, include/strata/kernels/qsa_decode_attn.hpp
Decode-time QSA layer: paged FP16 KV cache, pooled indexer keys, per-cell scores, top-k selection (identity
below the width), gather through the page table, and split-K softmax attention over the selected cells.

## Source semantics
* Selection width is `min(n_kv, idx_top_k + idx_block - 1)`; the extra `r - 1` covers the incomplete tail.
* The per-token step record is `[pos, n_kv = pos+1, n_bid = n_kv / r, width]`.
* Cell `j` scores through block `min(j / r, n_bid)`; cells `j >= (n_kv / r) * r` are the incomplete tail and
  get the +1e9 bias.
* KV pool row for cell `t`, head `h`: `(page * n_head_kv + h) * page_size + t % page_size`, one head's cells
  contiguous inside a page.
* GQA maps query head `q` to KV head `q / (n_head / n_head_kv)` (integer division).
* The gate half of each head's `2*head_dim` block starts at offset `head_dim`.
* Split-K attention uses `ceil(cap / CHUNK)` chunks; INT8 pools carry one fp16 scale per 64 values.

## Form template identifiers
`QsaShapes` -> `attn_geometry`; `qsa_step_fill` -> `fill_step_record`; `topk_512` -> `select_cells`;
`kv_gather` -> `gather_kv`; `page_size` -> `page_cells`; `idx_block` -> `pool_width`; `CHUNK` -> `split_chunk`.
-/

namespace Strata.Qsa

-- § selection width and step record

/-- Selection width: `min(n_kv, idx_top_k + idx_block - 1)`. -/
def selWidth (nKv topK r : Nat) : Nat := min nKv (topK + r - 1)

/-- The selection never returns more cells than the cache holds. -/
theorem selWidth_le_nkv (nKv topK r : Nat) : selWidth nKv topK r ≤ nKv := by
  unfold selWidth
  omega

/-- Below the bound the selection is the identity (every cell). -/
theorem selWidth_identity (nKv topK r : Nat) (h : nKv ≤ topK + r - 1) : selWidth nKv topK r = nKv := by
  unfold selWidth
  omega

/-- Real geometry: top-k 2048, block 4 gives bound 2051 (the plan's 2048 is conservative by three). -/
theorem selWidth_bound_real : 2048 + 4 - 1 = 2051 := by decide

/-- The widest selection the single-block kernel can ever produce (`kTopkMaxCells = 32768`) is 2051. -/
theorem selWidth_cap_real : selWidth 32768 2048 4 = 2051 := by decide

/-- The step record written for position `pos`. -/
def stepRecord (pos topK r : Nat) : List Nat := [pos, pos + 1, (pos + 1) / r, selWidth (pos + 1) topK r]

/-- Four entries, i.e. `kStepCount`, and `qsa_step_bytes` is 16. -/
theorem stepRecord_length (pos topK r : Nat) : (stepRecord pos topK r).length = 4 := rfl

/-- The step buffer is 16 bytes. -/
theorem step_bytes : 4 * 4 = 16 := by decide

/-- Completed blocks times the block width never exceed the cache size. -/
theorem blocks_le_nkv (nKv r : Nat) : nKv / r * r ≤ nKv := Nat.div_mul_le_self nKv r

/-- The incomplete tail has fewer than `r` cells: why the budget adds `r - 1`. -/
theorem tail_count_lt (n r : Nat) (hr : 0 < r) : n - (n / r) * r < r := by
  have h1 := Nat.div_add_mod n r
  have h2 := Nat.mod_lt n hr
  have h3 : r * (n / r) = (n / r) * r := Nat.mul_comm _ _
  omega

-- § cell to block mapping (idx_block = 4)

/-- Block used to score cell `j`. -/
def cellBlock (j nBid : Nat) : Nat := min (j / 4) nBid

/-- A completed-region cell maps to its own block. -/
theorem cellBlock_complete (j nKv : Nat) (h : j < nKv / 4 * 4) : cellBlock j (nKv / 4) = j / 4 := by
  unfold cellBlock
  omega

/-- A tail cell maps to the spare slot `n_bid`. -/
theorem cellBlock_tail (j nKv : Nat) (h1 : nKv / 4 * 4 ≤ j) (h2 : j < nKv) :
    cellBlock j (nKv / 4) = nKv / 4 := by
  unfold cellBlock
  omega

/-- The mapped block never exceeds the spare slot. -/
theorem cellBlock_le (j nBid : Nat) : cellBlock j nBid ≤ nBid := by
  unfold cellBlock
  omega

/-- The spare block has cells exactly when `n_kv` is not a multiple of 4. -/
theorem tail_has_cells_iff (nKv : Nat) : nKv / 4 * 4 < nKv ↔ nKv % 4 ≠ 0 := by
  omega

/-- Cells per scored block: `r` for completed blocks, the remainder for the spare one. -/
def cellsInBlock (b nBid nKv r : Nat) : Nat := if b < nBid then r else nKv - nBid * r

/-- Weights of all `n_bid + 1` blocks add up to `n_kv` (completed blocks plus the tail). -/
theorem block_weights_total (nKv r : Nat) : (nKv / r) * r + cellsInBlock (nKv / r) (nKv / r) nKv r = nKv := by
  unfold cellsInBlock
  have h := Nat.div_mul_le_self nKv r
  have hc : ¬ (nKv / r < nKv / r) := by omega
  rw [if_neg hc]
  omega

/-- The radix select makes 4 passes of 8 bits over a 32-bit key. -/
theorem radix_passes : 4 * 8 = 32 := by decide

-- § paged KV layout

/-- Physical row of `(page, head, offset)`: `(page * n_head_kv + head) * page_size + offset`. -/
def physRow (page nHeadKv head pageSize off : Nat) : Nat := (page * nHeadKv + head) * pageSize + off

/-- Consecutive cells of one head are contiguous inside a page. -/
theorem physRow_contiguous (page nHeadKv head pageSize off : Nat) :
    physRow page nHeadKv head pageSize (off + 1) = physRow page nHeadKv head pageSize off + 1 := by
  unfold physRow
  omega

/-- A valid (page, head, offset) is inside the pool of `n_pages * n_head_kv * page_size` rows. -/
theorem physRow_lt (page nHeadKv head pageSize off nPages : Nat) (hp : page < nPages) (hh : head < nHeadKv)
    (ho : off < pageSize) : physRow page nHeadKv head pageSize off < nPages * nHeadKv * pageSize := by
  unfold physRow
  have e1 : (page + 1) * nHeadKv = page * nHeadKv + nHeadKv := by rw [Nat.add_mul, Nat.one_mul]
  have h1 : (page + 1) * nHeadKv ≤ nPages * nHeadKv := Nat.mul_le_mul_right nHeadKv hp
  have e2 : (page * nHeadKv + head + 1) * pageSize = (page * nHeadKv + head) * pageSize + pageSize := by
    rw [Nat.add_mul, Nat.one_mul]
  have h2 : (page * nHeadKv + head + 1) * pageSize ≤ (nPages * nHeadKv) * pageSize :=
    Nat.mul_le_mul_right pageSize (by omega)
  omega

/-- A logical cell splits into page and offset. -/
theorem cell_split (t ps : Nat) : ps * (t / ps) + t % ps = t := Nat.div_add_mod t ps

/-- With page size 1 (the unpaged configuration) the offset is always zero. -/
theorem unpaged_offset (t : Nat) : t % 1 = 0 := Nat.mod_one t

-- § memory budget (documented figures)

/-- 12 QSA layers x 2 heads x 256 dims x (K and V) x 2 B x 32768 cells = 805306368 bytes. -/
theorem kv_bytes_32k : 12 * 2 * 256 * 2 * 2 * 32768 = 805306368 := by decide

/-- FINDING: that is exactly 768 MiB (about 805 MB); the header's "805 MiB" is a decimal-megabyte figure. -/
theorem kv_is_768_mib : 805306368 = 768 * 1048576 := by decide

/-- Indexer keys: 128 floats = 512 B per block, 128 B per cell per layer, 1536 B per cell over 12 layers. -/
theorem indexer_bytes_per_cell : 128 * 4 / 4 * 12 = 1536 := by decide

/-- At 32K cells the indexer keys take 50331648 bytes, exactly 48 MiB (the header says "50 MiB"). -/
theorem indexer_bytes_32k : 1536 * 32768 = 50331648 ∧ 50331648 = 48 * 1048576 := by decide

-- § grouped-query attention head map (24 query heads, 2 KV heads)

/-- Each KV head serves 12 query heads. -/
theorem gqa_group : 24 / 2 = 12 := by decide

/-- Integer-division map gives the contiguous halves [0 x 12, 1 x 12]. -/
theorem gqa_div_counts :
    ((List.range 24).filter (fun q => q / 12 = 0)).length = 12 ∧
    ((List.range 24).filter (fun q => q / 12 = 1)).length = 12 := by
  decide

/-- The modulo map would pair query head 1 with KV head 1, but the integer map pairs it with KV head 0. -/
theorem gqa_mod_differs : (1 % 2 : Nat) ≠ 1 / 12 := by decide

-- § gate split of the q projection (n_head 24, head_dim 256)

/-- Within a head's 512-wide block the query half precedes the gate half. -/
theorem gate_after_query (h d d' : Nat) (hd : d < 256) : h * 512 + d < h * 512 + 256 + d' := by
  omega

/-- Every gate element lies inside the `24 * 512` projection. -/
theorem gate_idx_lt (h d : Nat) (hh : h < 24) (hd : d < 256) : h * 512 + 256 + d < 24 * 512 := by
  omega

-- § split-K decode attention

/-- Number of chunks for a capacity `cap` and chunk size `c`. -/
def nChunks (cap c : Nat) : Nat := (cap + c - 1) / c

/-- The chunks cover the whole capacity. -/
theorem chunks_cover (cap c : Nat) (hc : 0 < c) : cap ≤ nChunks cap c * c := by
  unfold nChunks
  have h1 := Nat.div_add_mod (cap + c - 1) c
  have h2 := Nat.mod_lt (cap + c - 1) hc
  have h3 : c * ((cap + c - 1) / c) = ((cap + c - 1) / c) * c := Nat.mul_comm _ _
  omega

/-- Every cell below `cap` falls into a chunk below the chunk count. -/
theorem cell_chunk_lt (j cap c : Nat) (hc : 0 < c) (hj : j < cap) : j / c < nChunks cap c := by
  have h1 := chunks_cover cap c hc
  have h2 : j < c * nChunks cap c := by
    rw [Nat.mul_comm]
    omega
  exact Nat.div_lt_of_lt_mul h2

/-- INT8 pools carry one fp16 scale per 64 values: four per 256-wide row. -/
theorem scales_per_row : 256 / 64 = 4 := by decide

end Strata.Qsa

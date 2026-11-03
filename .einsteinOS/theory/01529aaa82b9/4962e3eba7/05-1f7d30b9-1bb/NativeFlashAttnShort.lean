/-!
# MODULE: native_flash_attn_short
Short-context decode FlashAttention (24 query heads, 2 KV heads, head_dim 256, f16 K/V, up to 256 cells)
specialised from llama.cpp's vector kernel: 128 threads (32 x 4), the context padded to 256 and walked in
two tiles of 128 cells.  This file proves the thread-to-cell and thread-to-dimension bijections, the
K/V/Q/shared-memory index bounds, the output coverage (including the NaN fill on an invalid step) and
the step-validity condition.

## Source semantics
* `kv = head / 12`; cells are addressed `(cell*2 + kv)*256 + d`.
* Per tile, cell = `base + warp*32 + (lane/8)*8 + row` with `row < 8`.
* Dimensions: `d = 2*(i0 + (lane%8)*4 + j)` with `i0 in {0,32,64,96}`, `j < 4`, plus `d+1`.
* A step is valid when `1 <= width <= max_context`, `n_kv = width`, `pos = width-1`, `n_bid = width/4`.

## Form template identifiers
- `attend` -> `short_flash_attend`; `vkq` -> `value_accumulator`; `tile` -> `weight_tile`
- `qreg` -> `query_registers`; `kStepWidth/NKv/Pos/NBid` -> `step_width/n_kv/pos/n_bid`
-/

namespace Strata.NativeFlashAttn

-- § arithmetic helper

/-- Row-major linearisation stays in bounds. -/
theorem lin2_lt {i a c d : Nat} (hi : i < a) (hc : c < d) : i * d + c < a * d := by
  have h := Nat.mul_le_mul_right d (show i + 1 ≤ a by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

-- § heads

/-- Query head `head < 24` maps to KV head `head / 12 < 2`. -/
theorem kv_head_lt {head : Nat} (h : head < 24) : head / 12 < 2 := by omega

-- § cells per tile

/-- Per-tile cell offset `warp*32 + (lane/8)*8 + row` is below 128. -/
theorem tile_cell_lt {warp lane row : Nat} (hw : warp < 4) (hl : lane < 32) (hr : row < 8) :
    warp * 32 + (lane / 8) * 8 + row < 128 := by omega

/-- Distinct (warp, lane group, row) triples address distinct cells of a tile. -/
theorem tile_cell_inj {w g r w' g' r' : Nat} (hg : g < 4) (hg' : g' < 4) (hr : r < 8) (hr' : r' < 8)
    (h : w * 32 + g * 8 + r = w' * 32 + g' * 8 + r') : w = w' ∧ g = g' ∧ r = r' := by omega

/-- Two tiles of 128 cells cover the padded length 256. -/
theorem tiles_cover : 2 * 128 = 256 := by decide

/-- The score slot `tid` of the weight tile is below 128 and below the 4096-float tile. -/
theorem score_slot_lt {tid : Nat} (h : tid < 128) : tid < 4 * 4 * 256 := by omega

-- § dimensions

/-- The dimension pair `(d, d+1)` with `d = 2*(32*a + 4*m + j)` stays below 256. -/
theorem dim_pair_lt {a m j : Nat} (ha : a < 4) (hm : m < 8) (hj : j < 4) :
    2 * (32 * a + m * 4 + j) + 1 < 256 := by omega

/-- Distinct (block a, lane m, element j) triples address distinct dimension pairs, so the 128 pairs are covered once. -/
theorem dim_pair_inj {a m j a' m' j' : Nat} (hm : m < 8) (hm' : m' < 8) (hj : j < 4) (hj' : j' < 4)
    (h : 2 * (32 * a + m * 4 + j) = 2 * (32 * a' + m' * 4 + j')) : a = a' ∧ m = m' ∧ j = j' := by omega

/-- The register slot `i0/8 + j = 4a + j` is below 16. -/
theorem register_slot_lt {a j : Nat} (ha : a < 4) (hj : j < 4) : 4 * a + j < 16 := by omega

-- § K/V addressing

/-- K/V element `(cell*2 + kv)*256 + d` is inside a cache of `capacity` cells (2 KV heads, 256 dims). -/
theorem kv_index_lt {cell kv d capacity : Nat} (hc : cell < capacity) (hk : kv < 2) (hd : d < 256) :
    (cell * 2 + kv) * 256 + d < capacity * 2 * 256 :=
  lin2_lt (lin2_lt hc hk) hd

/-- A cache of `capacity` cells holds `capacity * 1024` bytes in f16. -/
theorem kv_bytes (capacity : Nat) : capacity * 2 * 256 * 2 = capacity * 1024 := by omega

/-- A valid width never exceeds the cache capacity because `width <= max_context <= 256 <= capacity`. -/
theorem width_le_capacity {width ctx capacity : Nat} (h1 : width ≤ ctx) (h2 : ctx ≤ 256) (h3 : 256 ≤ capacity) :
    width ≤ capacity := by omega

-- § shared-memory reduction

/-- The cross-warp reduction write `warp*1024 + group*256 + 2*(i0 + m*4) + 2j + 1` stays below 4096. -/
theorem reduce_write_lt {warp group a m j : Nat} (hw : warp < 4) (hg : group < 4) (ha : a < 4) (hm : m < 8)
    (hj : j < 4) : warp * 1024 + group * 256 + 2 * (32 * a + m * 4) + 2 * j + 1 < 4096 := by omega

/-- The final reduction read `w*1024 + group*256 + i0 + tid` with `i0 in {0,128}`, `tid < 128` stays below 4096. -/
theorem reduce_read_lt {w group i0 tid : Nat} (hw : w < 4) (hg : group < 4) (hi : i0 = 0 ∨ i0 = 128)
    (ht : tid < 128) : w * 1024 + group * 256 + i0 + tid < 4096 := by omega

/-- The two output passes `i0 + tid` cover each of the 256 output dims exactly once. -/
theorem output_cover {d : Nat} (hd : d < 256) :
    (d < 128 ∧ d = 0 + d) ∨ (128 ≤ d ∧ d - 128 < 128 ∧ d = 128 + (d - 128)) := by omega

/-- On an invalid step each of the 128 threads writes two NaNs at `tid` and `tid + 128`, covering all 256. -/
theorem invalid_fill_cover {tid : Nat} (h : tid < 128) : tid < 256 ∧ tid + 128 < 256 := by omega

-- § step validity

/-- Step validity predicate of the kernel. -/
def valid (width nkv pos nbid ctx : Nat) : Prop :=
  1 ≤ width ∧ width ≤ ctx ∧ nkv = width ∧ pos + 1 = width ∧ nbid = width / 4

/-- A step produced by `qsa_step_fill` is valid whenever the position is below 256 and `top_k >= 256`. -/
theorem fill_is_valid {pos k : Nat} (hp : pos < 256) (hk : 256 ≤ k) :
    valid (min (pos + 1) (k + 4 - 1)) (pos + 1) pos ((pos + 1) / 4) 256 := by
  unfold valid
  omega

/-- A validated width indexes only cells that exist: `cell < width <= 256`. -/
theorem valid_cells_in_cache {cell width ctx : Nat} (hv : width ≤ ctx) (hctx : ctx ≤ 256) (hc : cell < width) :
    cell < 256 := by omega

end Strata.NativeFlashAttn

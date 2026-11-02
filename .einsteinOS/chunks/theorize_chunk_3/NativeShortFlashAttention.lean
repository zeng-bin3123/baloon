/-!
# MODULE: native_short_flash_attention
Index geometry and step-validity contract of the specialised single-token FlashAttention-vec kernel:
24 query heads of dimension 256, 2 KV heads (12 query heads per KV head), contexts 1..256, fp16 K/V,
128 threads (32 lanes x 4 warps) per head, padded KV length 256 processed in tiles of 128 cells.

## Source semantics
- Block = query head; `kv = head / 12`. K/V element (cell, kv, d) is at `(cell*2 + kv)*256 + d` halves.
- The step buffer is valid iff `1 <= width <= max_context`, `nkv == width`, `pos == width-1`, `nbid == width/4`;
  otherwise the kernel writes NaN and an unsupported-step status.
- Each tile handles cells `base + warp*32 + (lane & ~7) + row` for row in 0..7; cells >= width are masked.
- The final reduction sums 4 warps x 4 lane groups of the shared tile (4*4*256 floats) per output element.

## Form template identifiers
- width -> context_len, nkv -> kv_len, pos -> last_position, nbid -> index_blocks, mask -> attn_mask
- head -> query_head, kv -> kv_head, cell -> kv_cell
-/

namespace Strata.NativeFlashAttn

-- § helpers

/-- Row-major index bound. -/
theorem idx_lt {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § validity

/-- The step-validity predicate. -/
def validStep (width nkv pos nbid maxc : Int) : Prop :=
  1 ≤ width ∧ width ≤ maxc ∧ nkv = width ∧ pos = width - 1 ∧ nbid = width / 4

/-- Valid steps under the launcher bound have width at most 256. -/
theorem valid_width_le_256 (width nkv pos nbid maxc : Int) (hm : maxc ≤ 256)
    (h : validStep width nkv pos nbid maxc) : width ≤ 256 := by
  obtain ⟨_, h2, _, _, _⟩ := h
  omega

/-- Valid steps have the last position strictly below the context length. -/
theorem valid_pos_lt_width (width nkv pos nbid maxc : Int)
    (h : validStep width nkv pos nbid maxc) : pos < width := by
  obtain ⟨_, _, _, h4, _⟩ := h
  omega

/-- Cells below `width` are below the padded length 256 (padded cells are masked). -/
theorem live_cells_below_padding (cell width : Nat) (hw : width ≤ 256) (hc : cell < width) :
    cell < 256 := by
  omega

-- § head mapping

/-- 24 query heads over 2 KV heads is exactly 12 per KV head. -/
theorem heads_per_kv : 24 / 2 = 12 := by decide

/-- A query head below 24 maps to a KV head below 2. -/
theorem kv_head_lt (head : Nat) (h : head < 24) : head / 12 < 2 := by
  omega

/-- Query heads of one KV group are 12 consecutive heads. -/
theorem kv_group_range (head : Nat) (h : head < 24) :
    (head / 12) * 12 ≤ head ∧ head < (head / 12) * 12 + 12 := by
  omega

-- § cell / lane mapping inside a tile

/-- `lane & ~7` equals `(lane/8)*8`. -/
theorem lane_mask_eq (lane : Nat) : lane - lane % 8 = (lane / 8) * 8 := by
  omega

/-- Tile-local cell of (warp, lane, row) is below 128. -/
theorem tile_cell_lt (warp lane row : Nat) (hw : warp < 4) (hl : lane < 32) (hr : row < 8) :
    warp * 32 + (lane / 8) * 8 + row < 128 := by
  omega

/-- Tile-local cells are distinct for distinct (warp, lane group, row). -/
theorem tile_cell_inj (w g r w' g' r' : Nat) (hg : g < 4) (hr : r < 8) (hg' : g' < 4) (hr' : r' < 8)
    (h : w * 32 + g * 8 + r = w' * 32 + g' * 8 + r') : w = w' ∧ g = g' ∧ r = r' := by
  omega

/-- Every tile-local cell is produced by some (warp, lane group, row). -/
theorem tile_cell_cover (c : Nat) (h : c < 128) :
    ∃ w g r, w < 4 ∧ g < 4 ∧ r < 8 ∧ c = w * 32 + g * 8 + r :=
  ⟨c / 32, (c % 32) / 8, c % 8, by omega, by omega, by omega, by omega⟩

/-- Two tiles of 128 cover the padded length 256. -/
theorem tiles_cover_padding : 2 * 128 = 256 := by decide

/-- Absolute cell = tile base + local cell stays below 256 for base in {0, 128}. -/
theorem abs_cell_lt (base local_ : Nat) (hb : base = 0 ∨ base = 128) (hl : local_ < 128) :
    base + local_ < 256 := by
  omega

-- § K/V addressing

/-- K/V element address bound: (cell*2 + kv)*256 + d < capacity*2*256. -/
theorem kv_index_lt (cell kv d capacity : Nat) (hc : cell < capacity) (hk : kv < 2) (hd : d < 256) :
    (cell * 2 + kv) * 256 + d < capacity * 2 * 256 :=
  idx_lt (idx_lt hc hk) hd

/-- K/V capacity in halves times 2 bytes gives 1024 bytes per cell (the span checked by the launcher). -/
theorem kv_bytes_per_cell : 2 * 256 * 2 = 1024 := by decide

/-- Query/vkq register slot of (i0, j) with i0 = 32*m is below 16. -/
theorem qreg_slot_lt (m j : Nat) (hm : m < 4) (hj : j < 4) : (32 * m) / 8 + j < 16 := by
  omega

/-- Dimension pair index d = 2*(i+j) with i = 32*m + 4*l has d+1 < 256. -/
theorem dim_pair_lt (m l j : Nat) (hm : m < 4) (hl : l < 8) (hj : j < 4) :
    2 * (32 * m + 4 * l + j) + 1 < 256 := by
  omega

/-- The 128 (m, l, j) triples enumerate distinct dimension pairs. -/
theorem dim_pair_inj (m l j m' l' j' : Nat) (hl : l < 8) (hj : j < 4) (hl' : l' < 8) (hj' : j' < 4)
    (h : 32 * m + 4 * l + j = 32 * m' + 4 * l' + j') : m = m' ∧ l = l' ∧ j = j' := by
  omega

/-- Mask index of a live cell is below 256 (the mask has 256 halves). -/
theorem mask_index_lt (cell width : Nat) (hw : width ≤ 256) (hc : cell < width) : cell < 256 := by
  omega

-- § output reduction

/-- Shared-tile reduction index `w*1024 + group*256 + i0 + tid` is below 4096 (the tile size). -/
theorem reduce_index_lt (w group i0 tid : Nat) (hw : w < 4) (hg : group < 4)
    (hi : i0 = 0 ∨ i0 = 128) (ht : tid < 128) : w * 1024 + group * 256 + i0 + tid < 4096 := by
  omega

/-- Shared tile of 4*4*256 floats. -/
theorem tile_floats : 4 * 4 * 256 = 4096 := by decide

/-- Output elements `i0 + tid` for i0 in {0,128}, tid < 128 cover [0,256) once. -/
theorem output_cover (e : Nat) (h : e < 256) :
    ∃ i0 tid, (i0 = 0 ∨ i0 = 128) ∧ tid < 128 ∧ e = i0 + tid := by
  by_cases h' : e < 128
  · exact ⟨0, e, Or.inl rfl, h', by omega⟩
  · exact ⟨128, e - 128, Or.inr rfl, by omega, by omega⟩

/-- Output address of (head, element) is below 24*256. -/
theorem out_index_lt (head e : Nat) (hh : head < 24) (he : e < 256) : head * 256 + e < 24 * 256 :=
  idx_lt hh he

/-- Warp slots for the cross-warp max/sum are 0..3 and fit the 32-entry shared arrays. -/
theorem warp_slots_fit (warp : Nat) (h : warp < 4) : warp < 32 := by
  omega

-- § launcher contract

/-- Supported geometry of the launcher. -/
def supportedShape (n_head n_head_kv head_dim capacity max_context : Int) : Prop :=
  n_head = 24 ∧ n_head_kv = 2 ∧ head_dim = 256 ∧ 256 ≤ capacity ∧ 1 ≤ max_context ∧ max_context ≤ 256

/-- Supported shapes have capacity for every live cell. -/
theorem supported_capacity_covers (a b c cap m : Int) (h : supportedShape a b c cap m) : m ≤ cap := by
  obtain ⟨_, _, _, h4, _, h6⟩ := h
  omega

end Strata.NativeFlashAttn

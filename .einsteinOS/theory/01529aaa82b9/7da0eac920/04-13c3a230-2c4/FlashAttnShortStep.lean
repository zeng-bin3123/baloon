/-!
# MODULE: flash_attn_short_step
Single-token decode FlashAttention for Q = 24 heads x 256, KV = 2 heads x 256, contexts up to 256 cells, F16 KV
cache, one 128-thread block (32 lanes x 4 warps) per query head, online softmax with a final cross-warp merge.

## Source semantics
* Step validity: width in [1, max_context], nkv = width, pos = width - 1, nbid = width / 4; otherwise the
  output is NaN and status = unsupported.
* Query head h reads KV head h / 12, which is below 2.
* In a 128-cell tile, (warp, lane/8, row) enumerates the 128 cells exactly once; masked cells (>= width) contribute -inf.
* The 128-element head dimension of float2 pairs is covered by (i0 in {0,32,64,96}, lane%8, j<4) exactly once.
* All shared-memory indices stay below the 4096-float tile; K/V reads stay inside capacity*512 halves.
* The launcher requires capacity >= 256, context <= 256, disjoint aligned spans.

## Form template identifiers
`attend` → decode_attention_kernel ; step[kStepWidth] → kv_width ; kv = head/12 → kv_head_of_query ;
`native_flash_attn_short_step` → flash_attn_launch ; `validate_spans` → span_validation.
-/

namespace Strata.FlashAttnShort

-- § step validity

/-- The step-validity predicate over (width, nkv, pos, nbid, max_context). -/
def validStep (width nkv pos nbid maxc : Nat) : Prop :=
  1 ≤ width ∧ width ≤ maxc ∧ nkv = width ∧ pos + 1 = width ∧ nbid = width / 4

/-- A valid step with max_context ≤ 256 has at most 256 cells. -/
theorem valid_width_le (w nkv pos nb mc : Nat) (h : validStep w nkv pos nb mc) (hm : mc ≤ 256) : w ≤ 256 := by
  unfold validStep at h; omega

/-- Valid steps have a positive width. -/
theorem valid_width_pos (w nkv pos nb mc : Nat) (h : validStep w nkv pos nb mc) : 0 < w := by
  unfold validStep at h; omega

/-- The newest cell index is width - 1 and is a valid cell. -/
theorem valid_pos_lt (w nkv pos nb mc : Nat) (h : validStep w nkv pos nb mc) : pos < w := by
  unfold validStep at h; omega

-- § head mapping

/-- Query head maps to a KV head below 2. -/
theorem kv_head_lt (head : Nat) (h : head < 24) : head / 12 < 2 := by omega

/-- Query q-index head*256 + d is within the 24*256 query buffer. -/
theorem q_index_lt (head d : Nat) (hh : head < 24) (hd : d < 256) : head * 256 + d < 24 * 256 := by omega

-- § tile coverage

/-- A cell offset within a 128-cell tile. -/
def cellOffset (warp g row : Nat) : Nat := warp * 32 + g * 8 + row

/-- Cell offsets are below 128. -/
theorem cell_offset_lt (w g r : Nat) (hw : w < 4) (hg : g < 4) (hr : r < 8) : cellOffset w g r < 128 := by
  unfold cellOffset; omega

/-- Cell offsets are injective: each cell is scored exactly once per tile. -/
theorem cell_offset_unique (w g r w' g' r' : Nat) (h1 : w < 4) (h2 : g < 4) (h3 : r < 8) (h4 : w' < 4)
    (h5 : g' < 4) (h6 : r' < 8) (h : cellOffset w g r = cellOffset w' g' r') : w = w' ∧ g = g' ∧ r = r' := by
  unfold cellOffset at h; omega

/-- Lane l maps to group l/8 and in-group lane l%8; the lane's own score cell is warp*32+lane. -/
theorem score_cell_is_tid (w l : Nat) (hl : l < 32) : cellOffset w (l / 8) (l % 8) = w * 32 + l := by
  unfold cellOffset; omega

/-- Reduction over `k0` (step 4) and `lane/8` covers each local cell 0..31 of a warp exactly once. -/
theorem local_cell_cover (k0 g : Nat) (hk : k0 < 32) (hk4 : k0 % 4 = 0) (hg : g < 4) : k0 + g < 32 := by omega

/-- Distinct (k0, g) pairs give distinct local cells. -/
theorem local_cell_unique (k0 g k0' g' : Nat) (h1 : k0 % 4 = 0) (h2 : k0' % 4 = 0) (h3 : g < 4) (h4 : g' < 4)
    (h : k0 + g = k0' + g') : k0 = k0' ∧ g = g' := by omega

/-- Two 128-cell tiles cover the 256 padded length. -/
theorem tiles_cover : 2 * 128 = 256 := rfl

-- § head-dimension coverage

/-- Pair index i = i0 + 4*m + j, i0 = 32a, over a<4, m<8, j<4 lies in [0,128). -/
theorem pair_index_lt (a m j : Nat) (ha : a < 4) (hm : m < 8) (hj : j < 4) : 32 * a + 4 * m + j < 128 := by omega

/-- The pair index decomposition is unique. -/
theorem pair_index_unique (a m j a' m' j' : Nat) (h1 : m < 8) (h2 : j < 4) (h3 : m' < 8) (h4 : j' < 4)
    (h : 32 * a + 4 * m + j = 32 * a' + 4 * m' + j') : a = a' ∧ m = m' ∧ j = j' := by omega

/-- Element index d = 2*(pair) and d+1 stay below 256. -/
theorem elem_index_lt (p : Nat) (h : p < 128) : 2 * p + 1 < 256 := by omega

/-- qreg slot i0/8 + j with i0 ∈ {0,32,64,96}, j<4 is below 16. -/
theorem qreg_slot_lt (a j : Nat) (ha : a < 4) (hj : j < 4) : 4 * a + j < 16 := by omega

/-- qreg slot: i0/8 = 4a for i0 = 32a, so slot = 4a + j. -/
theorem qreg_slot_value (a j : Nat) : (32 * a) / 8 + j = 4 * a + j := by omega

/-- qreg slots are unique across (a, j). -/
theorem qreg_slot_unique (a j a' j' : Nat) (h1 : j < 4) (h2 : j' < 4) (h : 4 * a + j = 4 * a' + j') :
    a = a' ∧ j = j' := by omega

-- § memory bounds

/-- K/V half index (cell*2 + kv)*256 + d is below capacity*512 for cell < 256 ≤ capacity. -/
theorem kv_index_lt (cell kv d cap : Nat) (hc : cell < 256) (hk : kv < 2) (hd : d < 256) (hcap : 256 ≤ cap) :
    (cell * 2 + kv) * 256 + d < cap * 512 := by omega

/-- Mask index cell < 256 fits the 256-half mask span. -/
theorem mask_index_lt (cell : Nat) (h : cell < 256) : cell < 256 := h

/-- Shared tile index of the final merge stays below 4096. -/
theorem merge_tile_lt (w g a m j : Nat) (hw : w < 4) (hg : g < 4) (ha : a < 4) (hm : m < 8) (hj : j < 4) :
    w * 1024 + g * 256 + 2 * (32 * a + 4 * m) + 2 * j + 1 < 4096 := by omega

/-- Final output index i0 + tid for i0 ∈ {0,128}, tid<128 is below 256. -/
theorem out_index_lt (i0 tid : Nat) (h0 : i0 = 0 ∨ i0 = 128) (ht : tid < 128) : i0 + tid < 256 := by omega

/-- The merge reads w*1024 + group*256 + i0 + tid below 4096. -/
theorem merge_read_lt (w g i0 tid : Nat) (hw : w < 4) (hg : g < 4) (h0 : i0 = 0 ∨ i0 = 128) (ht : tid < 128) :
    w * 1024 + g * 256 + i0 + tid < 4096 := by omega

/-- Launch capacity bytes: capacity * 1024 = capacity * 512 halves * 2. -/
theorem kv_bytes (cap : Nat) : cap * 1024 = cap * 512 * 2 := by omega

-- § span overlap

/-- Byte-span overlap as in the launcher. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap; omega

/-- Spans laid end to end do not overlap. -/
theorem overlap_end_to_end (a an b bn : Nat) (h : a + an ≤ b) : ¬ overlap a an b bn := by
  unfold overlap; omega

/-- A non-empty span overlaps itself. -/
theorem overlap_self (a n : Nat) (h : 0 < n) : overlap a n a n := by
  unfold overlap; omega

end Strata.FlashAttnShort

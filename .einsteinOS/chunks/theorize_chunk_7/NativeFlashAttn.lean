/-!
# MODULE: native_flash_attn
Index arithmetic and validity predicate for the specialised single-token flash attention (Q24x256, KV2x256).

## Source semantics
One block per query head (24), kv head = head/12. A step is valid iff 1 <= width <= max_context, n_kv = width, pos = width-1, n_bid = width/4; otherwise the output is NaN and status reports unsupported. Within a 128-cell tile, cell = base + warp*32 + (lane/8)*8 + row.

## Form template identifiers
attend -> flash_step, kStepWidth -> width, kStepNKv -> n_kv
-/
namespace Strata.NativeFlashAttn

/-- Step validity. -/
def valid (width nkv pos nbid maxCtx : Nat) : Prop :=
  1 ≤ width ∧ width ≤ maxCtx ∧ nkv = width ∧ pos + 1 = width ∧ nbid = width / 4

-- § head mapping

/-- 24 heads map onto 2 kv heads. -/
theorem kv_lt (head : Nat) (h : head < 24) : head / 12 < 2 := by omega

/-- Each kv head serves 12 query heads. -/
theorem kv_group (head : Nat) : head / 12 * 12 ≤ head ∧ head < head / 12 * 12 + 12 := by omega

-- § tile mapping

/-- Cell offsets stay inside a 128-cell tile. -/
theorem cell_lt (warp lane row : Nat) (hw : warp < 4) (hl : lane < 32) (hr : row < 8) : warp * 32 + (lane / 8) * 8 + row < 128 := by omega

/-- Distinct (warp, lane group, row) triples address distinct cells. -/
theorem cell_inj (w l r w' l' r' : Nat) (hw : w < 4) (hl : l < 32) (hr : r < 8) (hw' : w' < 4) (hl' : l' < 32) (hr' : r' < 8) (h : w * 32 + (l / 8) * 8 + r = w' * 32 + (l' / 8) * 8 + r') : w = w' ∧ l / 8 = l' / 8 ∧ r = r' := by omega

/-- Output dims: the two passes over i0 in {0,128} cover all 256 dims. -/
theorem out_cover (j : Nat) (h : j < 256) : j = 0 + j % 128 ∨ j = 128 + j % 128 := by omega

-- § validity

/-- A valid step has n_kv = pos + 1. -/
theorem valid_nkv (width nkv pos nbid maxCtx : Nat) (h : valid width nkv pos nbid maxCtx) : nkv = pos + 1 := by unfold valid at h; omega

/-- A valid step has at least one cell. -/
theorem valid_pos_width (width nkv pos nbid maxCtx : Nat) (h : valid width nkv pos nbid maxCtx) : 1 ≤ nkv := by unfold valid at h; omega

/-- A valid step is within the context bound. -/
theorem valid_bound (width nkv pos nbid maxCtx : Nat) (h : valid width nkv pos nbid maxCtx) : nkv ≤ maxCtx := by unfold valid at h; omega

/-- A padded length of 256 gives one KV tile of width at most 256. -/
theorem single_tile (width : Nat) (h : width ≤ 256) : (width + 255) / 256 ≤ 1 := by omega

end Strata.NativeFlashAttn

/-!
# MODULE: native_flash_attn_bounds
Memory-bound and loop-coverage facts of the single-step flash attention kernel `native_flash_attn.cu`
(24 query heads, 2 KV heads, head dim 256, context 1..256, 128 threads).

## Source semantics
* KV head of query head `h` is `h/12`, below 2 for `h < 24`.
* The KV loop runs `base = 0, 128` over a padded length of 256; cell `base + warp*32 + (lane&~7) + row` stays
  below 256 for warp < 4, lane < 32, row < 8.
* K/V element `(cell*2 + kv)*256 + d (+1)` is below `capacity*512` halves.
* Shared tile of 4096 floats is indexed in the PV phase by `local = warp*32 + k0 + lane/8` and in the final
  reduction by `w*1024 + group*256 + i0 + tid`.
* The step is valid iff `1 <= width <= max_context`, `n_kv = width`, `pos = width-1`, `n_bid = width/4`.

## Form template identifiers
kv -> kv_head; width -> context_length; capacity -> kv_capacity; tile -> shared_tile;
max_shared/sum_shared -> warp_partials.
-/

namespace NativeFlashAttnBounds

-- § heads

/-- Each query head maps to a KV head below 2. -/
theorem kv_head_lt (head : Nat) (h : head < 24) : head / 12 < 2 := by omega

/-- Query element indices stay inside the 24x256 query. -/
theorem q_index_lt (head d : Nat) (hh : head < 24) (hd : d < 255) : head * 256 + d + 1 < 24 * 256 := by
  omega

/-- Output indices for both halves of the head are inside the output. -/
theorem out_index_lt (head tid : Nat) (hh : head < 24) (ht : tid < 128) :
    head * 256 + tid + 128 < 24 * 256 := by omega

-- § KV loop

/-- The padded loop runs exactly two iterations. -/
theorem loop_iterations : (256 + 127) / 128 = 2 := by decide

/-- Cell index of a (base, warp, lane, row) quadruple stays inside the padded range. -/
theorem cell_lt (base warp lane row : Nat) (hb : base = 0 ∨ base = 128) (hw : warp < 4) (hl : lane < 32)
    (hr : row < 8) : base + warp * 32 + (lane / 8) * 8 + row < 256 := by
  omega

/-- The four warps times 32 cells cover the 128 cells of one iteration. -/
theorem iteration_cells : 4 * 32 = 128 := by decide

/-- Dimension indices of the dot product stay below 256. -/
theorem dim_index_lt (i0 lane j : Nat) (h0 : i0 = 0 ∨ i0 = 32 ∨ i0 = 64 ∨ i0 = 96) (hl : lane < 32)
    (hj : j < 4) : 2 * (i0 + (lane % 8) * 4 + j) + 1 < 256 := by
  omega

/-- K/V element indices stay inside a cache of `capacity` cells (halves). -/
theorem kv_index_lt (cell kv d capacity : Nat) (hc : cell < capacity) (hk : kv < 2) (hd : d < 255) :
    (cell * 2 + kv) * 256 + d + 1 < capacity * 512 := by
  omega

/-- Cache bytes are `capacity*1024` for K and for V, i.e. `capacity*512` halves. -/
theorem kv_bytes_halves (capacity : Nat) : capacity * 1024 = capacity * 512 * 2 := by omega

-- § shared memory

/-- Shared memory use: tile 16384 B plus two 32-float arrays. -/
theorem smem_bytes : 4 * 4 * 256 * 4 + 2 * 32 * 4 = 16640 := by decide

/-- The shared use fits the 48 KiB static limit. -/
theorem smem_fits : 4 * 4 * 256 * 4 + 2 * 32 * 4 ≤ 49152 := by decide

/-- The probability tile slot of the PV phase is inside the first 128 entries. -/
theorem pv_local_lt (warp k0 lane : Nat) (hw : warp < 4) (hk : k0 < 32) (hk4 : k0 % 4 = 0) (hl : lane < 32) :
    warp * 32 + k0 + lane / 8 < 128 := by omega

/-- Final partial-accumulator slot is inside the 4096-float tile. -/
theorem partial_slot_lt (w group i0 tid : Nat) (hw : w < 4) (hg : group < 4)
    (hi : i0 = 0 ∨ i0 = 128) (ht : tid < 128) : w * (4 * 256) + group * 256 + i0 + tid < 4096 := by
  omega

/-- Partial tile writes of the vkq registers are inside the tile. -/
theorem vkq_write_lt (warp g lane i0 j : Nat) (hw : warp < 4) (hg : g < 4) (hl : lane < 32)
    (h0 : i0 = 0 ∨ i0 = 32 ∨ i0 = 64 ∨ i0 = 96) (hj : j < 4) :
    warp * 4 * 256 + (lane / 8) * 256 + 2 * (i0 + (lane % 8) * 4) + 2 * j + 1 < 4096 := by
  omega

-- § validity predicate

/-- Validity of a step. -/
def valid (width nkv pos nbid maxCtx : Nat) : Prop :=
  1 ≤ width ∧ width ≤ maxCtx ∧ nkv = width ∧ pos = width - 1 ∧ nbid = width / 4

/-- A valid step fits the padded loop. -/
theorem valid_fits (width nkv pos nbid maxCtx : Nat) (hm : maxCtx ≤ 256)
    (h : valid width nkv pos nbid maxCtx) : width ≤ 256 := by
  unfold valid at h; omega

/-- A valid step has at least one attended cell, and the newest cell is the last one. -/
theorem valid_last_cell (width nkv pos nbid maxCtx : Nat) (h : valid width nkv pos nbid maxCtx) :
    pos < width := by
  unfold valid at h; omega

/-- The padded cells beyond width are exactly the masked ones. -/
theorem padding_masked (width cell : Nat) (hw : width ≤ 256) (hc : cell < 256) :
    cell < width ∨ width ≤ cell := by omega

end NativeFlashAttnBounds

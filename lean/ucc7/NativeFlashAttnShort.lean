/-!
# MODULE: native_flash_attn_short
SOURCES: src/kernels/cuda/native_flash_attn.cu
Short-context FlashAttention vector kernel specialised from llama.cpp: 24 query heads of dim 256, 2 KV heads,
one block of 128 threads per head, online softmax over the padded length 256 in two tiles of 128 cells, with a
device-side validity check on the step state that writes a status word and NaN-fills the output on failure.

## Source semantics
* `kv = head / 12`; K/V cell row is `(cell*2 + kv)*256`; cells `>= width` are masked to -inf.
* A step is valid iff `1 <= width <= max_context`, `nkv = width`, `pos = width-1`, `n_bid = width/4`.
* Tile `base` runs over `{0, 128}`; cell = `base + warp*32 + (lane/8)*8 + row`, `row < 8`.
* The cross-warp combine uses `4*4*256` floats of shared memory.
* `max_context <= 256`, `capacity >= 256`, `capacity*1024` must not overflow `size_t`.

## Form template identifiers
* `HD` -> head_dim, `PAD` -> padded_length, `validStep` -> step_validity_predicate.
-/

namespace Ref.NativeFlashAttn

/-- Head dimension. -/
def HD : Nat := 256
/-- Padded context length. -/
def PAD : Nat := 256

/-- Step validity predicate. -/
def validStep (width nkv pos nbid maxc : Nat) : Prop :=
  1 ≤ width ∧ width ≤ maxc ∧ nkv = width ∧ pos + 1 = width ∧ nbid = width / 4

/-- A valid step has `n_kv - 1 = pos` and at most 256 cells for `max_context <= 256`. -/
theorem valid_width_le (width nkv pos nbid maxc : Nat) (h : validStep width nkv pos nbid maxc)
    (hm : maxc ≤ 256) : width ≤ 256 ∧ nkv = pos + 1 := by
  unfold validStep at h; omega

/-- The padded length is two tiles of 128 cells. -/
theorem tiles : PAD / 128 = 2 := by decide

/-- GQA: a head `< 24` pairs with a KV head `< 2`. -/
theorem kv_lt (head : Nat) (h : head < 24) : head / 12 < 2 := by omega

/-- Every padded cell is covered by exactly one (tile, warp, lane-group, row). -/
theorem cell_cover (cell : Nat) (h : cell < 256) :
    ∃ base warp lane row, (base = 0 ∨ base = 128) ∧ warp < 4 ∧ lane < 32 ∧ row < 8 ∧
      cell = base + warp * 32 + (lane / 8) * 8 + row := by
  refine ⟨cell / 128 * 128, (cell % 128) / 32, ((cell % 32) / 8) * 8, cell % 8, ?_, ?_, ?_, ?_, ?_⟩
  · omega
  · omega
  · omega
  · omega
  · omega

/-- Distinct (tile, warp, lane-group, row) give distinct cells (no double counting). -/
theorem cell_unique (b w g r b' w' g' r' : Nat) (hb : b = 0 ∨ b = 128) (hb' : b' = 0 ∨ b' = 128)
    (hw : w < 4) (hw' : w' < 4) (hg : g < 4) (hg' : g' < 4) (hr : r < 8) (hr' : r' < 8)
    (h : b + w * 32 + g * 8 + r = b' + w' * 32 + g' * 8 + r') :
    b = b' ∧ w = w' ∧ g = g' ∧ r = r' := by omega

/-- K/V element address `(cell*2 + kv)*256 + d` is below `capacity * 512` elements. -/
theorem kv_addr_lt (cell kv d capacity : Nat) (hc : cell < capacity) (hk : kv < 2) (hd : d < 256) :
    (cell * 2 + kv) * 256 + d < capacity * 512 := by omega

/-- One cell holds `2 KV heads * 256 dims * 2 bytes = 1024` bytes. -/
theorem cell_bytes : 2 * 256 * 2 = 1024 := by decide

/-- The host's byte-size computation cannot overflow under the capacity guard. -/
theorem kv_bytes_no_overflow (cap : Nat) (h : cap ≤ 18446744073709551615 / 1024) :
    cap * 1024 ≤ 18446744073709551615 := by omega

/-- The V loop: `local = warp*32 + k0 + lane/8` with `k0 ∈ {0,4,..,28}` reaches every one of 128 locals. -/
theorem v_local_cover (loc : Nat) (h : loc < 128) :
    ∃ warp k0 g, warp < 4 ∧ k0 < 32 ∧ k0 % 4 = 0 ∧ g < 4 ∧ loc = warp * 32 + k0 + g := by
  refine ⟨loc / 32, ((loc % 32) / 4) * 4, loc % 4, ?_, ?_, ?_, ?_, ?_⟩ <;> omega

/-- Shared tile size is `4*4*256 = 4096` floats (16 KB). -/
theorem tile_size : 4 * 4 * 256 = 4096 ∧ 4096 * 4 = 16384 := by decide

/-- Combine write index `warp*1024 + (lane/8)*256 + 2*(i0 + (lane%8)*4) + 2j + 1` is inside the tile. -/
theorem tile_write_lt (warp g l8 i0 j : Nat) (hw : warp < 4) (hg : g < 4) (hl : l8 < 8)
    (hi : i0 < 128) (h32 : i0 % 32 = 0) (hj : j < 4) :
    warp * 1024 + g * 256 + 2 * (i0 + l8 * 4 + j) + 1 < 4096 := by omega

/-- Combine read index `w*1024 + group*256 + i0 + tid` is inside the tile (`i0 + tid < 256`). -/
theorem tile_read_lt (w group col : Nat) (hw : w < 4) (hg : group < 4) (hc : col < 256) :
    w * 1024 + group * 256 + col < 4096 := by omega

/-- Output columns: `tid` and `tid + 128` for `tid < 128` cover all 256 head dims exactly once. -/
theorem out_cover (d : Nat) (h : d < 256) : ∃ tid half, tid < 128 ∧ half < 2 ∧ d = tid + 128 * half := by
  exact ⟨d % 128, d / 128, by omega, by omega, by omega⟩

/-- Output row base `head*256` is disjoint across heads. -/
theorem out_rows_disjoint (head head' d d' : Nat) (hd : d < 256) (hd' : d' < 256)
    (h : head * 256 + d = head' * 256 + d') : head = head' ∧ d = d' := by omega

/-- Per-head query offset `head*256 + d` lies inside `24*256` floats. -/
theorem q_addr_lt (head d : Nat) (hh : head < 24) (hd : d < 256) : head * 256 + d < 24 * 256 := by omega

/-- Scale `1/sqrt(256) = 1/16`: `16 * 16 = 256`. -/
theorem scale_is_inv_sqrt : 16 * 16 = HD := by decide

/-- Out-of-range cells are masked: a cell `>= width` is excluded from the valid set. -/
theorem masked_cell (cell width : Nat) (h : width ≤ cell) : ¬ cell < width := by omega

/-- With `width <= 256` every valid cell lies inside the padded tiles. -/
theorem valid_cell_in_pad (cell width : Nat) (h : cell < width) (hw : width ≤ 256) : cell < PAD := by
  unfold PAD; omega

/-- Status word domain: the kernel writes either success or the unsupported-step code, never both:
modelled as a boolean choice from validity. -/
def statusOf (valid : Bool) (ok bad : Nat) : Nat := if valid then ok else bad

/-- A valid step reports the success code. -/
theorem status_valid (ok bad : Nat) : statusOf true ok bad = ok := rfl

/-- An invalid step reports the unsupported-step code. -/
theorem status_invalid (ok bad : Nat) : statusOf false ok bad = bad := rfl

end Ref.NativeFlashAttn

/-!
# MODULE: native_flash_attn_short_step
Specialised short-context FlashAttention decode (`src/kernels/cuda/native_flash_attn.cu`):
24 query heads x 256 dims, 2 KV heads, contexts 1..256, 24 blocks of 32x4 threads, padded length 256.

## Source semantics
* The step is valid iff `1 <= width <= max_context`, `n_kv = width`, `pos = width - 1`, `n_bid = width / 4`;
  otherwise the kernel reports `UnsupportedStep` and writes NaN.
* `kv = head / 12`; K/V element `(cell*2 + kv)*256 + d`; output `head*256 + tid (+128)`.
* A 128-cell tile is covered by `warp*32 + (lane/8)*8 + row` with warp<4, lane/8<4, row<8; two tiles cover 256.
* The scale is `0.0625 = 1/sqrt 256`; `tile` holds 4*4*256 floats.

## Form template identifiers
* `Valid`        <- the `valid` predicate computed from `step`
* `cellOf`       <- `base + warp*32 + (lane & ~7) + row`
* `kElem`        <- `(cell * 2 + kv) * 256 + d`
-/

namespace NativeFlashAttnShortStep

-- § step validity

/-- The step-state validity test of the kernel. -/
def Valid (maxCtx pos nKv nBid width : Nat) : Prop :=
  1 ≤ width ∧ width ≤ maxCtx ∧ nKv = width ∧ pos + 1 = width ∧ nBid = width / 4

/-- A step filled from a position (`n_kv = pos+1`, `width = min n_kv 2051`) is valid for contexts up to 256. -/
theorem valid_of_fill (maxCtx pos : Nat) (h1 : pos + 1 ≤ maxCtx) (h2 : maxCtx ≤ 256) :
    Valid maxCtx pos (pos + 1) ((pos + 1) / 4) (min (pos + 1) 2051) := by
  unfold Valid
  omega

/-- Valid steps are consistent: the position is the last cell. -/
theorem valid_pos (maxCtx pos nKv nBid width : Nat) (h : Valid maxCtx pos nKv nBid width) :
    pos = width - 1 := by
  unfold Valid at h
  omega

/-- A step whose `n_kv` disagrees with the width is rejected. -/
theorem invalid_of_nKv_ne (maxCtx pos nKv nBid width : Nat) (h : nKv ≠ width) :
    ¬ Valid maxCtx pos nKv nBid width := by
  unfold Valid
  omega

-- § heads and buffers

/-- Each of the 24 heads reads one of the 2 KV heads. -/
theorem kv_lt (head : Nat) (h : head < 24) : head / 12 < 2 := by
  omega

/-- Element offset of K/V dimension `d` (and `d+1`) of `cell` stays inside a `capacity`-cell buffer. -/
theorem kElem_lt (cell kv d capacity : Nat) (hc : cell < capacity) (hk : kv < 2) (hd : d + 1 < 256) :
    (cell * 2 + kv) * 256 + d + 1 ≤ capacity * 512 := by
  omega

/-- The byte size `capacity * 1024` is `capacity * 2 heads * 256 dims * 2 bytes`. -/
theorem kv_bytes_eq (capacity : Nat) : capacity * 2 * 256 * 2 = capacity * 1024 := by
  omega

/-- Output writes `head*256 + i0 + tid` for `i0 ∈ {0, 128}`, `tid < 128` stay in the 24x256 output. -/
theorem out_index_lt (head tid i0 : Nat) (hh : head < 24) (ht : tid < 128)
    (hi : i0 = 0 ∨ i0 = 128) : head * 256 + i0 + tid < 6144 := by
  omega

/-- Shared-memory tile writes of the final reduction stay inside `4*4*256` floats. -/
theorem tile_index_lt (warp lane i0 j : Nat) (hw : warp < 4) (hl : lane < 32)
    (hi : i0 = 0 ∨ i0 = 32 ∨ i0 = 64 ∨ i0 = 96) (hj : j < 4) :
    warp * 4 * 256 + (lane / 8) * 256 + 2 * (i0 + (lane % 8) * 4) + 2 * j + 1 < 4096 := by
  omega

-- § tile coverage

/-- Cell handled by (tile base, warp, lane group, row). -/
def cellOf (base w g r : Nat) : Nat := base + w * 32 + g * 8 + r

/-- Every thread-assigned cell lies inside the padded length 256. -/
theorem cellOf_lt (base w g r : Nat) (hb : base = 0 ∨ base = 128) (hw : w < 4) (hg : g < 4)
    (hr : r < 8) : cellOf base w g r < 256 := by
  unfold cellOf
  omega

/-- Every cell below 256 is handled by some (tile, warp, group, row). -/
theorem cellOf_surj (c : Nat) (hc : c < 256) :
    ∃ base w g r, (base = 0 ∨ base = 128) ∧ w < 4 ∧ g < 4 ∧ r < 8 ∧ cellOf base w g r = c := by
  refine ⟨c / 128 * 128, c % 128 / 32, c % 32 / 8, c % 8, ?_, ?_, ?_, ?_, ?_⟩
  · omega
  · omega
  · omega
  · omega
  · unfold cellOf
    omega

/-- ... and by exactly one: no two threads process the same cell. -/
theorem cellOf_inj (b1 w1 g1 r1 b2 w2 g2 r2 : Nat)
    (hb1 : b1 = 0 ∨ b1 = 128) (hw1 : w1 < 4) (hg1 : g1 < 4) (hr1 : r1 < 8)
    (hb2 : b2 = 0 ∨ b2 = 128) (hw2 : w2 < 4) (hg2 : g2 < 4) (hr2 : r2 < 8)
    (h : cellOf b1 w1 g1 r1 = cellOf b2 w2 g2 r2) :
    b1 = b2 ∧ w1 = w2 ∧ g1 = g2 ∧ r1 = r2 := by
  unfold cellOf at h
  omega

/-- Two tiles of 128 cover the padded length of 256. -/
theorem tiles_cover : 256 / 128 = 2 := by
  decide

/-- The scale `1/16` squares to `1/256`, i.e. `1/sqrt(head_dim)` for `head_dim = 256`. -/
theorem scale_sq : 16 * 16 = 256 := by
  decide

end NativeFlashAttnShortStep

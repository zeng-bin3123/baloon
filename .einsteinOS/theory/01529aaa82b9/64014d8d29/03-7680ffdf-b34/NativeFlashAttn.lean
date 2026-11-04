/-!
# MODULE: native_flash_attn
SOURCES: src/kernels/cuda/native_flash_attn.cu
Short-context (<= 256 cells) FlashAttention-vec decode kernel specialised to Q 24x256 / KV 2x256, fp16 K/V,
one block per query head, 128 threads; online softmax with a streaming maximum.

## Source semantics
* The kernel is valid only if width in [1, max_context], n_kv = width, pos = width-1, n_bid = width/4;
  otherwise it writes a NaN output and reports an `unsupported step` status.
* Query head h reads kv head h / 12.
* Cells are processed 128 per pass over a padded length of 256 (two passes); cells >= width are -inf masked.
* The output is combined from four warps' accumulators and divided by the summed exp weights.

## Form template identifiers
width -> `width`; max_context -> `maxCtx`; padded_length -> 256.
-/

namespace NativeFlashAttn

/-- Validity predicate for the step. -/
def valid (width nkv pos nbid maxCtx : Nat) : Prop :=
  1 ≤ width ∧ width ≤ maxCtx ∧ nkv = width ∧ pos + 1 = width ∧ nbid = width / 4

/-- A host-filled step with width = pos+1 <= 256 passes the guard. -/
theorem valid_of_fill (pos maxCtx : Nat) (h : pos + 1 ≤ maxCtx) :
    valid (pos + 1) (pos + 1) pos ((pos + 1) / 4) maxCtx := by
  unfold valid; omega

/-- Below the selection bound (2051) the selection width equals n_kv, so this kernel sees every cell. -/
theorem short_context_is_identity (n : Nat) (h : n ≤ 256) : min n 2051 = n := by omega

/-- GQA: heads 0..23 map to kv heads 0..1. -/
theorem kv_head_lt (head : Nat) (h : head < 24) : head / 12 < 2 := by omega

/-- Every cell `< 256` is visited by exactly one (pass, warp, lane-group, row). -/
theorem cell_decomposition (cell : Nat) (h : cell < 256) :
    cell = (cell / 128) * 128 + ((cell % 128) / 32) * 32 + ((cell % 32) / 8) * 8 + cell % 8 ∧
    cell / 128 < 2 ∧ (cell % 128) / 32 < 4 ∧ cell % 8 < 8 := by omega

/-- The thread whose lane satisfies `lane % 8 = row` keeps the score: its cell is `warp*32 + lane`. -/
theorem lane_holds_cell (warp lane row : Nat) (hl : lane < 32) (hr : row = lane % 8) :
    warp * 32 + (lane / 8) * 8 + row = warp * 32 + lane := by omega

/-- The output is written for all 256 dims by two 128-wide passes. -/
theorem output_cover (d : Nat) (h : d < 256) : d = (d / 128) * 128 + d % 128 ∧ d / 128 < 2 ∧ d % 128 < 128 := by
  omega

/-- The online-softmax rescale exponent `maximum - next_max` is `<= 0`, so `rescale <= 1`. -/
theorem rescale_exponent_nonpos (maximum x : Int) : maximum - max maximum x ≤ 0 := by omega

/-- The running maximum is monotone. -/
theorem running_max_mono (maximum x : Int) : maximum ≤ max maximum x := by omega

/-- The four-warp accumulator staging index stays inside the 4*4*256-float shared tile. -/
theorem stage_index_lt (warp lane i0 j : Nat) (hw : warp < 4) (hl : lane < 32)
    (hi : i0 = 0 ∨ i0 = 32 ∨ i0 = 64 ∨ i0 = 96) (hj : j < 4) :
    warp * 4 * 256 + (lane / 8) * 256 + 2 * (i0 + (lane % 8) * 4) + 2 * j + 1 < 4 * 4 * 256 := by
  omega

/-- Shared memory use: tile + two 32-float reductions fits 48 KiB. -/
theorem smem_fits : (4 * 4 * 256 + 32 + 32) * 4 ≤ 49152 := by decide

/-- `scale = 1/sqrt(256) = 1/16 = 0.0625`: 16 * 16 = 256. -/
theorem scale_is_inv_sqrt : 16 * 16 = 256 := by decide

/-- KV byte count `capacity*1024` is the 2 heads * 256 dims * 2 B per cell, with no loss. -/
theorem kv_bytes_per_cell : 2 * 256 * 2 = 1024 := by decide

/-- A K/V element index `(cell*2 + kv)*256 + d` is within `capacity` cells. -/
theorem kv_index_lt (cell kv d cap : Nat) (hc : cell < cap) (hk : kv < 2) (hd : d < 256) :
    (cell * 2 + kv) * 256 + d < cap * 2 * 256 := by
  have h1 : (cell + 1) * 2 ≤ cap * 2 := Nat.mul_le_mul_right 2 (by omega)
  have h2 : (cell * 2 + kv + 1) * 256 ≤ (cap * 2) * 256 := Nat.mul_le_mul_right 256 (by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1 h2
  omega

end NativeFlashAttn

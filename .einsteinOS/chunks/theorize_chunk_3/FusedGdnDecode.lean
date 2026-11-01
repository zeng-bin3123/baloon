/-!
# MODULE: fused_gdn_decode
Index, launch-geometry and packing contracts of the fused gated-delta-net decode kernels: the one-block-per-head
state step with RMS-gated output (`gdn_step_norm_kernel`), the depthwise conv + SiLU + L2 kernel
(`gdn_conv_l2_kernel`) and the alpha/beta projection kernel (`gdn_ab_kernel`).

## Source semantics
* The state is `(S, h_v, S)` with `S = 128`; block `head` owns one head, thread `(col, rg)` owns column `col` and the
  32 rows `rg*32 .. rg*32+31`, so the 4 row groups tile the 128 rows exactly once.
* `q`/`k` are read through `head % h_k` (modulo head pairing), which needs `h_v % h_k = 0` and `h_k > 0`.
* The conv kernel has one thread per channel, `channels % 128 = 0`; history holds 3 taps per channel, weights 4.
* The ab kernel handles `2*h_v` rows (alpha rows first, then beta rows), 8 warps per block, `n_embd % 8 = 0`;
  bf16 weights are unpacked from `uint32` with `<< 16` (low half) and `& 0xffff0000` (high half).

## Form template identifiers
* `S` / `RG` / `RPG`   : state size, row groups, rows per thread
* `rowOf`              : (row group, register slot) -> state row
* `stateIndex`         : (row, head, col) -> flat index `(row*h_v + head)*S + col`
* `ab_row`             : global warp row of the ab kernel
-/

namespace Strata.FusedGdn

-- § constants

/-- State size (rows = cols). -/
def S : Nat := 128
/-- Row groups per block. -/
def RG : Nat := 4
/-- Rows per thread. -/
def RPG : Nat := 32

/-- The row groups tile the state rows. -/
theorem rg_mul_rpg : RG * RPG = S := by decide

/-- The block has `S * RG = 512` threads, matching `__launch_bounds__(S * RG)`. -/
theorem block_threads : S * RG = 512 := by decide

-- § row ownership

/-- State row held in register slot `r` of row group `rg`. -/
def rowOf (rg r : Nat) : Nat := rg * 32 + r

/-- Every (row group, slot) pair names a valid state row. -/
theorem rowOf_lt (rg r : Nat) (hg : rg < 4) (hr : r < 32) : rowOf rg r < 128 := by
  unfold rowOf; omega

/-- Distinct (row group, slot) pairs name distinct rows: no two threads of a column write the same element. -/
theorem rowOf_inj (rg r rg' r' : Nat) (hr : r < 32) (hr' : r' < 32)
    (h : rowOf rg r = rowOf rg' r') : rg = rg' ∧ r = r' := by
  unfold rowOf at h; omega

/-- Every row is owned by exactly the (row group, slot) pair `(row / 32, row % 32)`. -/
theorem rowOf_surj (row : Nat) (h : row < 128) :
    rowOf (row / 32) (row % 32) = row ∧ row / 32 < 4 ∧ row % 32 < 32 := by
  unfold rowOf; omega

-- § thread numbering

/-- `tid = rg * 128 + col` is below the 512-thread block, and `tid < 128` exactly for row group 0. -/
theorem tid_split (rg col : Nat) (hg : rg < 4) (hc : col < 128) :
    rg * 128 + col < 512 ∧ (rg * 128 + col < 128 ↔ rg = 0) := by
  constructor <;> omega

/-- The warps of row group 0 are warps 0..3, so the four-entry `wsum` read is covered. -/
theorem wsum_slot_lt (tid : Nat) (h : tid < 128) : tid / 32 < 4 := by omega

/-- The declared `wsum[S*RG/32]` array has room for every warp index of the block. -/
theorem wsum_capacity (tid : Nat) (h : tid < 512) : tid / 32 < 128 * 4 / 32 := by omega

-- § state addressing

/-- A product bound used for flat indices of the shape `(a*b + c)*d + e`. -/
theorem idx3_lt (A B D a b c e : Nat) (hb : a < A) (hc : c < B) (he : e < D) :
    (a * B + c) * D + e < A * B * D := by
  have h1 : a * B + c < A * B := by
    have h : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B hb
    rw [Nat.add_mul, Nat.one_mul] at h
    omega
  have h2 : (a * B + c + 1) * D ≤ A * B * D := Nat.mul_le_mul_right D h1
  rw [Nat.add_mul, Nat.one_mul] at h2
  omega

/-- Flat index of state element `(row, head, col)` in the `(S, h_v, S)` layout. -/
def stateIndex (h_v row head col : Nat) : Nat := (row * h_v + head) * 128 + col

/-- Every in-range state access stays inside the `S * h_v * S` state buffer. -/
theorem stateIndex_lt (h_v row head col : Nat) (hr : row < 128) (hh : head < h_v) (hc : col < 128) :
    stateIndex h_v row head col < 128 * h_v * 128 := by
  unfold stateIndex
  exact idx3_lt 128 h_v 128 row h_v head col hr hh hc

-- § head pairing

/-- `head % h_k` is a valid q/k head. -/
theorem pair_head_lt (head h_k : Nat) (hk : 0 < h_k) : head % h_k < h_k := Nat.mod_lt _ hk

/-- Staged q/k reads `qh*128 + tid` stay inside the `h_k * 128` q/k buffers. -/
theorem qk_read_lt (qh h_k tid : Nat) (hq : qh < h_k) (ht : tid < 128) : qh * 128 + tid < h_k * 128 := by
  omega

-- § conv + SiLU + L2 kernel

/-- Channel index `blk*128 + t` is below `channels` for every launched thread. -/
theorem conv_channel_lt (blk t channels : Nat) (hc : channels % 128 = 0) (hb : blk < channels / 128)
    (ht : t < 128) : blk * 128 + t < channels := by omega

/-- History taps `c*3 .. c*3+2` stay inside the `3 * channels` history. -/
theorem conv_hist_lt (c channels : Nat) (h : c < channels) : c * 3 + 2 < 3 * channels := by omega

/-- Weight taps `c*4 .. c*4+3` stay inside the `4 * channels` weights. -/
theorem conv_weight_lt (c channels : Nat) (h : c < channels) : c * 4 + 3 < 4 * channels := by omega

/-- Blocks `< qk_heads` are exactly the q/k blocks, and the validated bound keeps them inside the grid. -/
theorem qk_blocks_in_grid (blk qk_heads channels : Nat) (hq : qk_heads ≤ channels / 128) (hb : blk < qk_heads) :
    blk < channels / 128 := by omega

/-- The L2 reduction uses `S/32 = 4` partials, one per warp of a 128-thread block. -/
theorem conv_partial_slots (t : Nat) (h : t < 128) : t / 32 < 128 / 32 := by omega

-- § alpha/beta projection kernel

/-- Global row handled by warp `warp` of block `blk` (8 warps per block). -/
def abRow (blk warp : Nat) : Nat := blk * 8 + warp

/-- The `(2*h_v + 7)/8` blocks cover every one of the `2*h_v` rows. -/
theorem ab_row_covered (h_v r : Nat) (hr : r < 2 * h_v) :
    r / 8 < (2 * h_v + 7) / 8 ∧ abRow (r / 8) (r % 8) = r := by
  unfold abRow; omega

/-- Beta rows (`row ≥ h_v`) map into `[0, h_v)` after subtracting `h_v`. -/
theorem beta_row_lt (h_v row : Nat) (hr : row < 2 * h_v) (hb : h_v ≤ row) : row - h_v < h_v := by omega

/-- With `n % 8 = 0` the 8-wide chunks cover the whole reduction axis exactly. -/
theorem ab_chunks_exact (n : Nat) (h : n % 8 = 0) : (n / 8) * 8 = n := by omega

/-- Lane-strided loop `j = lane, lane+32, ...` visits each chunk exactly once. -/
theorem lane_stride_unique (j : Nat) : j % 32 < 32 ∧ j % 32 + 32 * (j / 32) = j := by omega

-- § bf16 unpacking from a packed word

/-- Value of the low bf16 half of a 32-bit word when shifted into the high half (`w << 16`, 32-bit wrap). -/
def lowHalfWord (w : Nat) : Nat := (w % 65536) * 65536

/-- Value of the high bf16 half in place (`w & 0xffff0000`). -/
def highHalfWord (w : Nat) : Nat := (w / 65536) * 65536

/-- The two unpacked halves lose no bits of the packed word. -/
theorem halves_recombine (w : Nat) : highHalfWord w + w % 65536 = w := by
  unfold highHalfWord; omega

/-- The shifted low half is a valid 32-bit word. -/
theorem lowHalfWord_lt (w : Nat) : lowHalfWord w < 4294967296 := by
  unfold lowHalfWord; omega

end Strata.FusedGdn

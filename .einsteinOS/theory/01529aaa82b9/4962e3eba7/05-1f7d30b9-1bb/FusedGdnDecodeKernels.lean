/-!
# MODULE: fused_gdn_decode_kernels
Fused single-token kernels of the gated delta net (GDN): causal 4-tap conv + SiLU (+ per-head L2 norm
on the q/k channels), the alpha/beta gate projections over bf16 weights, and one fused recurrence
step + RMSNorm + sigmoid-gate kernel.  This file proves the index, coverage and bound arithmetic that
every launch configuration of `fused_gdn.cu` relies on.

## Source semantics
* `gdn_step_norm_kernel`: block = one value head, 128 x 4 threads; thread (col, rg) owns 32 rows
  `rg*32 .. rg*32+31` of column `col` of the (128, h_v, 128) state.  Rows are partitioned exactly once.
* `gdn_conv_l2_kernel`: one thread per channel, `channels % 128 == 0`, 3-deep history per channel.
* `gdn_ab_kernel`: one warp per output row; rows `[0,h_v)` are alpha, `[h_v,2h_v)` are beta.
* Softplus uses the `v > 20` pass-through branch.

## Form template identifiers
- `S` -> `state_dim` (128); `RG` -> `row_groups` (4); `RPG` -> `rows_per_group` (32)
- `h_k` -> `n_key_heads`; `h_v` -> `n_value_heads`; `qh = head % h_k` -> `key_head_of`
- `hist` -> `conv_history`; `wsum` -> `warp_partial_sums`
-/

namespace Strata.FusedGdn

-- § arithmetic helper

/-- Row-major linearisation stays in bounds: `i < a`, `c < d` gives `i*d + c < a*d`. -/
theorem lin2_lt {i a c d : Nat} (hi : i < a) (hc : c < d) : i * d + c < a * d := by
  have h := Nat.mul_le_mul_right d (show i + 1 ≤ a by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

-- § constants

/-- State size `S`. -/
def stateDim : Nat := 128
/-- Number of row groups `RG`. -/
def rowGroups : Nat := 4

/-- `RPG = S / RG` is 32 rows per thread. -/
theorem rows_per_group_eq : stateDim / rowGroups = 32 := by decide

-- § step + norm kernel geometry

/-- Every (row group, local row) pair addresses a row inside the 128-row state. -/
theorem tile_row_lt {rg r : Nat} (hrg : rg < 4) (hr : r < 32) : rg * 32 + r < 128 := by omega

/-- Distinct (row group, local row) pairs address distinct rows, so no two threads write one state element. -/
theorem tile_row_inj {rg r rg' r' : Nat} (hr : r < 32) (hr' : r' < 32)
    (h : rg * 32 + r = rg' * 32 + r') : rg = rg' ∧ r = r' := by omega

/-- Every row of the state is owned by exactly one (row group, local row) pair. -/
theorem tile_row_surj {row : Nat} (hrow : row < 128) :
    row / 32 < 4 ∧ row % 32 < 32 ∧ (row / 32) * 32 + row % 32 = row := by omega

/-- The state element `((row*h_v + head)*128 + col)` lies inside the `128 x h_v x 128` buffer. -/
theorem state_index_lt {row head col h_v : Nat} (hr : row < 128) (hh : head < h_v) (hc : col < 128) :
    (row * h_v + head) * 128 + col < 128 * h_v * 128 :=
  lin2_lt (lin2_lt hr hh) hc

/-- Threads `tid < 128` are exactly the threads of row group 0 (they stage k/q and write y). -/
theorem tid_in_first_group {rg col : Nat} (hc : col < 128) : rg * 128 + col < 128 ↔ rg = 0 := by
  constructor <;> intro h <;> omega

/-- The warp slot written into `wsum` is within its 16 entries for every thread of the 512-thread block. -/
theorem tid_warp_slot {rg col : Nat} (hrg : rg < 4) (hc : col < 128) :
    (rg * 128 + col) / 32 < 16 := by omega

/-- Row group 0 spans exactly four warps, which is why the final sum reads `wsum[0..3]`. -/
theorem first_group_warps {col : Nat} (hc : col < 128) : col / 32 < 4 := by omega

/-- `wsum` has `S*RG/32 = 16` entries. -/
theorem wsum_capacity : 128 * 4 / 32 = 16 := by decide

/-- The key/query head selected by the modulo pairing is a valid head. -/
theorem key_head_lt {head h_k : Nat} (hk : 0 < h_k) : head % h_k < h_k := Nat.mod_lt _ hk

-- § conv + L2 kernel geometry

/-- Each channel is handled by exactly one (block, thread) pair when `channels % 128 = 0`. -/
theorem conv_channel_cover {channels c : Nat} (hdiv : channels % 128 = 0) (hc : c < channels) :
    c / 128 < channels / 128 ∧ c % 128 < 128 ∧ (c / 128) * 128 + c % 128 = c := by omega

/-- Every thread of every launched block addresses an in-range channel. -/
theorem conv_thread_in_range {channels b t : Nat} (hdiv : channels % 128 = 0)
    (hb : b < channels / 128) (ht : t < 128) : b * 128 + t < channels := by omega

/-- History addressing `hist[c*3 + i]` stays in the `3 * channels` buffer. -/
theorem conv_hist_index_lt {c i channels : Nat} (hc : c < channels) (hi : i < 3) :
    c * 3 + i < channels * 3 := by omega

/-- Weight addressing `w[c*4 + i]` stays in the `4 * channels` buffer. -/
theorem conv_weight_index_lt {c i channels : Nat} (hc : c < channels) (hi : i < 4) :
    c * 4 + i < channels * 4 := by omega

/-- Blocks below `qk_heads` hold whole 128-channel heads inside the channel range, so the L2 branch is per head. -/
theorem qk_block_in_range {channels qk b : Nat} (hdiv : channels % 128 = 0)
    (hq : qk ≤ channels / 128) (hb : b < qk) : b * 128 + 127 < channels := by omega

-- § alpha/beta kernel geometry

/-- Every output row is covered by the `(2*h_v + 7)/8` launched blocks of 8 warps. -/
theorem ab_row_cover {row h : Nat} (hrow : row < 2 * h) : row / 8 < (2 * h + 7) / 8 := by omega

/-- A row is either an alpha row or a beta row, with the beta index `row - h_v` in range. -/
theorem ab_row_split {row h : Nat} (hrow : row < 2 * h) : row < h ∨ (h ≤ row ∧ row - h < h) := by omega

/-- The (block, warp) decomposition of a row is recovered by division and remainder. -/
theorem ab_block_row {b w : Nat} (hw : w < 8) : (b * 8 + w) / 8 = b ∧ (b * 8 + w) % 8 = w := by omega

/-- Vector loads of 8 bf16 weights never run past a row when `n % 8 = 0`. -/
theorem ab_vector_in_row {n j : Nat} (hn : n % 8 = 0) (hj : j < n / 8) : j * 8 + 7 < n := by omega

-- § softplus branch

/-- Overflow-safe softplus selector: above 20 the input is returned unchanged. -/
def softplusSel (f : Int → Int) (v : Int) : Int := if v > 20 then v else f v

/-- Above the threshold the large-argument branch is taken. -/
theorem softplusSel_large (f : Int → Int) {v : Int} (h : 20 < v) : softplusSel f v = v := by
  unfold softplusSel
  split <;> omega

/-- At or below the threshold the real function is used. -/
theorem softplusSel_small (f : Int → Int) {v : Int} (h : v ≤ 20) : softplusSel f v = f v := by
  unfold softplusSel
  split <;> omega

end Strata.FusedGdn

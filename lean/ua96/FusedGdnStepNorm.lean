/-!
# MODULE: fused_gdn_step_norm
SOURCES: src/kernels/cuda/fused_gdn.cu
Fused gated-delta-net kernels: depthwise conv + SiLU + L2 norm, the alpha/beta gate projections, and the
state step fused with RMS-norm and sigmoid output gating. This file covers the launch geometry, index maps
and the small algebraic contracts the three kernels rely on.

## Source semantics
* The state is `(S, h_v, S)` with `S = 128`; a block owns one head, 512 threads = 128 columns x 4 row groups
  of 32 rows each, so the row groups tile the 128 rows exactly once.
* `gdn_conv_l2_kernel`: one thread per channel `c = blockIdx.x*128 + threadIdx.x`; history slides by one tap.
* `gdn_ab_kernel`: one warp per output row of the stacked `[alpha; beta]` projection (`2*h_v` rows, 8 per block).
* head pairing is `head % h_k`.

## Form template identifiers
* `S` -> state_size, `RG` -> row_groups, `RPG` -> rows_per_group, `rowOf` -> row_index_map,
  `convShift` -> history_slide, `abRow` -> stacked_projection_row.
-/

namespace Ref.FusedGdn

/-- State size (rows = cols). -/
def S : Nat := 128
/-- Row groups per head block. -/
def RG : Nat := 4
/-- Rows owned by one thread (`S / RG`). -/
def RPG : Nat := 32

/-- `RPG` really is `S / RG`. -/
theorem rpg_def : S / RG = RPG := by decide

/-- The row groups tile the state rows exactly. -/
theorem rows_cover : RG * RPG = S := by decide

/-- A step block has `S * RG = 512` threads. -/
theorem threads_per_block : S * RG = 512 := by decide

/-- Warps of row group 0 are threads 0..127, i.e. four warps, matching the four `wsum` entries read. -/
theorem warps_of_group0 : S / 32 = 4 := by decide

/-- The `wsum` scratch has `S*RG/32 = 16` slots, at least the 4 that are read. -/
theorem wsum_big_enough : 4 ≤ S * RG / 32 := by decide

/-- Row of the state handled by (row group, register index). -/
def rowOf (rg r : Nat) : Nat := rg * 32 + r

/-- Every (group, register) pair addresses a row inside the state. -/
theorem rowOf_lt (rg r : Nat) (hg : rg < 4) (hr : r < 32) : rowOf rg r < 128 := by
  unfold rowOf; omega

/-- The row map is injective: no two threads/registers touch the same row. -/
theorem rowOf_inj (rg r rg' r' : Nat) (hr : r < 32) (hr' : r' < 32)
    (h : rowOf rg r = rowOf rg' r') : rg = rg' ∧ r = r' := by
  unfold rowOf at h; omega

/-- The row map is surjective onto `0..127`: every row is owned. -/
theorem rowOf_surj (i : Nat) (hi : i < 128) :
    ∃ rg r, rg < 4 ∧ r < 32 ∧ rowOf rg r = i := by
  refine ⟨i / 32, i % 32, by omega, by omega, ?_⟩
  unfold rowOf; omega

/-- Linear state address `((row*h_v + head)*128 + col)` stays inside the `128*h_v*128` state. -/
theorem state_addr_lt (row head col h_v : Nat) (hrow : row < 128) (hh : head < h_v) (hc : col < 128) :
    (row * h_v + head) * 128 + col < 128 * h_v * 128 := by
  have h1 : (row + 1) * h_v ≤ 128 * h_v := Nat.mul_le_mul_right _ (by omega)
  have h2 : (row + 1) * h_v = row * h_v + h_v := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- Head pairing `head % h_k` is always a valid q/k head. -/
theorem qk_head_lt (head h_k : Nat) (h : 0 < h_k) : head % h_k < h_k := Nat.mod_lt _ h

/-- Conv channel index is in range when the grid is `channels / 128` blocks of 128 threads. -/
theorem conv_channel_lt (nb blk t : Nat) (hb : blk < nb) (ht : t < 128) : blk * 128 + t < nb * 128 := by
  omega

/-- The history slide: drop the oldest tap and append the newest input. -/
def convShift (h0 h1 h2 x : Int) : Int × Int × Int := (h1, h2, x)

/-- Four-tap depthwise convolution output. -/
def convOut (h0 h1 h2 x w0 w1 w2 w3 : Int) : Int := h0 * w0 + h1 * w1 + h2 * w2 + x * w3

/-- After a step the newest history slot holds exactly the input just consumed. -/
theorem convShift_newest (h0 h1 h2 x : Int) : (convShift h0 h1 h2 x).2.2 = x := rfl

/-- After a step the oldest slot holds what was the middle tap. -/
theorem convShift_oldest (h0 h1 h2 x : Int) : (convShift h0 h1 h2 x).1 = h1 := rfl

/-- With an all-zero history only the last tap contributes. -/
theorem convOut_zero_history (x w0 w1 w2 w3 : Int) : convOut 0 0 0 x w0 w1 w2 w3 = x * w3 := by
  unfold convOut; simp

/-- Two consecutive steps: the input of step one is the middle tap at step two. -/
theorem convShift_twice (h0 h1 h2 x y : Int) :
    (convShift (convShift h0 h1 h2 x).1 (convShift h0 h1 h2 x).2.1 (convShift h0 h1 h2 x).2.2 y).2.1 = x := rfl

/-- History addressing `c*3 + tap` is injective for taps below 3. -/
theorem hist_addr_inj (c t c' t' : Nat) (ht : t < 3) (ht' : t' < 3)
    (h : c * 3 + t = c' * 3 + t') : c = c' ∧ t = t' := by omega

/-- Weight addressing `c*4 + tap` is injective for taps below 4. -/
theorem weight_addr_inj (c t c' t' : Nat) (ht : t < 4) (ht' : t' < 4)
    (h : c * 4 + t = c' * 4 + t') : c = c' ∧ t = t' := by omega

/-- The stacked alpha/beta rows are covered by `(2*h_v+7)/8` blocks of 8 warps. -/
theorem ab_rows_covered (h_v row : Nat) (h : row < 2 * h_v) : row / 8 < (2 * h_v + 7) / 8 := by omega

/-- The grid supplies at least `2*h_v` rows. -/
theorem ab_grid_enough (h_v : Nat) : 2 * h_v ≤ (2 * h_v + 7) / 8 * 8 := by omega

/-- Row selector of the stacked projection: rows `>= h_v` are the beta rows. -/
def abRow (h_v row : Nat) : Nat := if h_v ≤ row then row - h_v else row

/-- The within-projection row index is always `< h_v`. -/
theorem abRow_lt (h_v row : Nat) (h : row < 2 * h_v) : abRow h_v row < h_v := by
  unfold abRow; split <;> omega

/-- `n_embd % 8 = 0` makes the 8-wide bf16 chunks cover the reduction axis exactly. -/
theorem ab_chunks_exact (n : Nat) (h : n % 8 = 0) : n / 8 * 8 = n := by omega

/-- The delta-rule write at scalar level with `k = beta = 1`: the new state equals `v`. -/
theorem delta_write_exact (dec s v : Int) : dec * s + (v - dec * s * 1) * 1 = v := by
  omega

end Ref.FusedGdn

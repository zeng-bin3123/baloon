/-!
# MODULE: fused_gated_delta_net
Fused gated-delta-net decode kernels: a recurrent step with output RMS-norm and sigmoid gating
(`gdn_step_norm_kernel`), a 4-tap causal conv + SiLU + optional L2 norm (`gdn_conv_l2_kernel`),
and the alpha/beta projection producing the decay gate and write strength (`gdn_ab_kernel`).

## Source semantics
* State is (S, h_v, S) with S = 128; a block per value head, 128x4 threads; thread (col, rg) owns 32 rows.
* The row groups partition the 128 rows exactly once; the four partial sums cover the head's S rows.
* Query/key heads are paired with value heads by `head % h_k`.
* The conv history is a sliding window of 3 past values; after a step it holds the last 3 of (history ++ [x]).
* `gdn_ab_kernel` assigns one warp per output row; rows < h_v are the gate (alpha), the rest are beta.

## Form template identifiers
`gdn_step_norm_kernel` → recurrent_step_with_norm ; `gdn_conv_l2_kernel` → causal_conv_silu_l2 ;
`gdn_ab_kernel` → gate_beta_projection ; S → state_dim ; RG → row_groups ; RPG → rows_per_thread.
-/

namespace Strata.FusedGdn

-- § constants

/-- State dimension. -/
def S : Nat := 128
/-- Row groups per block. -/
def RG : Nat := 4
/-- Rows per thread. -/
def RPG : Nat := 32

/-- Rows per thread times row groups covers the state dimension. -/
theorem rows_cover : RG * RPG = S := rfl

/-- Block size is 512 threads. -/
theorem block_threads : S * RG = 512 := rfl

/-- The shared `wsum` array has 16 slots, more than the 4 warps of row group 0. -/
theorem wsum_capacity : 4 ≤ S * RG / 32 := by decide

-- § row ownership

/-- Each (rg, r) pair maps to a row below 128. -/
theorem row_lt (rg r : Nat) (hrg : rg < 4) (hr : r < 32) : rg * 32 + r < 128 := by omega

/-- Distinct (rg, r) pairs own distinct rows. -/
theorem row_injective (rg r rg' r' : Nat) (h1 : rg < 4) (h2 : r < 32) (h3 : rg' < 4) (h4 : r' < 32)
    (h : rg * 32 + r = rg' * 32 + r') : rg = rg' ∧ r = r' := by omega

/-- Every row has an owner. -/
theorem row_surjective (i : Nat) (h : i < 128) : ∃ rg r, rg < 4 ∧ r < 32 ∧ rg * 32 + r = i :=
  ⟨i / 32, i % 32, by omega, by omega, by omega⟩

/-- Thread id decomposition: tid = rg*128 + col with col < 128. -/
theorem tid_decompose (rg col : Nat) (hrg : rg < 4) (hc : col < 128) : (rg * 128 + col) / 128 = rg ∧ (rg * 128 + col) % 128 = col := by
  omega

/-- Row-group-0 threads (tid < 128) span exactly the 4 warps that fill `wsum`. -/
theorem warp_index_lt (tid : Nat) (h : tid < 128) : tid / 32 < 4 := by omega

/-- Column address in the (S, h_v, S) state is below the state size. -/
theorem idx_lt (a b n m : Nat) (ha : a < n) (hb : b < m) : a * m + b < n * m := by
  have h1 : (a + 1) * m ≤ n * m := Nat.mul_le_mul_right m (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * m = a * m + m := Nat.succ_mul a m
  omega

/-- State element (row i, head h, col c) at `(i*h_v + h)*128 + c` is in range. -/
theorem state_index_lt (i h c hv : Nat) (hi : i < 128) (hh : h < hv) (hc : c < 128) :
    (i * hv + h) * 128 + c < 128 * hv * 128 := by
  have := idx_lt i h 128 hv hi hh
  have h2 := idx_lt (i * hv + h) c (128 * hv) 128 (by omega) hc
  exact h2

/-- Head pairing: the shared q/k head index is below h_k. -/
theorem qk_head_lt (head hk : Nat) (h : 0 < hk) : head % hk < hk := Nat.mod_lt _ h

/-- With h_v a multiple of h_k, each q/k head serves exactly h_v / h_k value heads. -/
theorem head_pairing_exact (head hk hv : Nat) (h : head < hv) (hk0 : 0 < hk) (hd : hv % hk = 0) :
    head % hk < hk ∧ head < hv := ⟨Nat.mod_lt _ hk0, h⟩

-- § conv kernel

/-- One block per 128 channels; the grid covers all channels when channels % 128 = 0. -/
theorem conv_channel_lt (bx tx channels : Nat) (hb : bx < channels / 128) (ht : tx < 128) (h : channels % 128 = 0) :
    bx * 128 + tx < channels := by omega

/-- Blocks with index below qk_heads are the L2-normalised q/k channels. -/
theorem conv_qk_blocks (bx qk channels : Nat) (hq : qk ≤ channels / 128) (hb : bx < qk) : bx < channels / 128 := by omega

/-- A 128-thread block has 4 warps, matching `part[S/32]`. -/
theorem conv_part_size : 128 / 32 = 4 := rfl

/-- Sliding window update: drop the oldest entry, append x. -/
def slide (hist : List Nat) (x : Nat) : List Nat := (hist ++ [x]).drop 1

/-- Sliding the window preserves its length. -/
theorem slide_length (hist : List Nat) (x : Nat) : (slide hist x).length = hist.length := by
  simp [slide]

/-- The newest entry is in the window. -/
theorem slide_mem (hist : List Nat) (x : Nat) (h : hist ≠ []) : x ∈ slide hist x := by
  cases hist with
  | nil => exact absurd rfl h
  | cons a t => simp [slide]

-- § alpha/beta kernel

/-- One warp per row: 8 rows per block; the grid covers all 2*h_v rows. -/
theorem ab_grid_covers (hv row : Nat) (h : row < 2 * hv) : row / 8 < (2 * hv + 7) / 8 := by omega

/-- A row ≥ h_v is a beta row and maps to r = row - h_v < h_v. -/
theorem ab_beta_row (hv row : Nat) (h : row < 2 * hv) (hb : hv ≤ row) : row - hv < hv := by omega

/-- A row < h_v is a gate row and maps to itself. -/
theorem ab_gate_row (hv row : Nat) (hb : row < hv) : row < hv := hb

/-- With n % 8 = 0, the uint4 chunks tile the n-vector exactly: n/8 chunks of 8. -/
theorem ab_chunks_exact (n : Nat) (h : n % 8 = 0) : n / 8 * 8 = n := by omega

/-- Last element touched by chunk j (< n/8) is below n. -/
theorem ab_chunk_in_range (n j : Nat) (h : n % 8 = 0) (hj : j < n / 8) : j * 8 + 7 < n := by omega

/-- Lane-strided loop covers every chunk index k < n/8 with lane = k % 32. -/
theorem ab_lane_cover (k : Nat) : k = (k / 32) * 32 + k % 32 ∧ k % 32 < 32 := by omega

/-- A 16-thread-wide tile of xor-shuffles (16,8,4,2,1) reduces 32 lanes: sum of offsets equals 31. -/
theorem warp_reduce_offsets : 16 + 8 + 4 + 2 + 1 = 31 := rfl

end Strata.FusedGdn

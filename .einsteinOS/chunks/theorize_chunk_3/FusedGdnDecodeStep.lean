/-!
# MODULE: fused_gdn_decode_step
Index geometry and launch contracts of the fused gated-delta-net decode kernels: the conv+SiLU+L2 kernel,
the alpha/beta projection kernel, and the state-update + RMSNorm*gate kernel. The kernels split the 128x128
per-head state over 4 row groups of 32 rows, one thread per (row group, column).

## Source semantics
- `gdn_step_norm_kernel`: block = head, 128x4 threads, thread (col, rg) owns rows rg*32..rg*32+31 of column col.
  State address of row i is `(i*h_v + head)*S + col`; q/k head is `head % h_k`; h_v % h_k = 0 is required.
- `gdn_conv_l2_kernel`: one thread per channel, history stride 3, weight stride 4; the first `qk_heads` blocks
  (of 128 channels) are L2-normalised.
- `gdn_ab_kernel`: one warp per output row, rows [0,h_v) are alpha (gate), rows [h_v,2h_v) are beta.

## Form template identifiers
- S -> state_dim (128), RG -> row_groups (4), RPG -> rows_per_group (32)
- head -> head_idx, h_k -> n_qk_heads, h_v -> n_v_heads, col -> column_idx, rg -> row_group_idx
-/

namespace Strata.FusedGdnDecode

/-- State size (rows = cols). -/
def S : Nat := 128
/-- Row groups per block. -/
def RG : Nat := 4
/-- Rows held in registers per thread. -/
def RPG : Nat := 32

-- § helpers

/-- Row-major index bound used throughout the address arithmetic. -/
theorem idx_lt {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § block geometry

/-- The row groups tile the state exactly. -/
theorem row_groups_tile_state : RG * RPG = S := by decide

/-- A block has 512 threads. -/
theorem block_thread_count : S * RG = 512 := by decide

/-- The per-warp partial-sum array has 16 slots (512/32). -/
theorem wsum_slot_count : S * RG / 32 = 16 := by decide

/-- Thread (rg, r) addresses a row below 128. -/
theorem thread_row_in_range (rg r : Nat) (hrg : rg < 4) (hr : r < 32) : rg * 32 + r < 128 := by
  omega

/-- Distinct (row group, register) pairs address distinct rows. -/
theorem thread_row_inj (rg r rg' r' : Nat) (hr : r < 32) (hr' : r' < 32)
    (h : rg * 32 + r = rg' * 32 + r') : rg = rg' ∧ r = r' := by
  omega

/-- Every row of the state is owned by some (row group, register) pair. -/
theorem row_cover (row : Nat) (h : row < 128) :
    ∃ rg r, rg < 4 ∧ r < 32 ∧ row = rg * 32 + r :=
  ⟨row / 32, row % 32, by omega, by omega, by omega⟩

/-- Flattened thread id is below the block size. -/
theorem tid_in_range (rg col : Nat) (hrg : rg < 4) (hcol : col < 128) : rg * 128 + col < 512 := by
  omega

/-- Only row group 0 stages the 128 q/k entries into shared memory. -/
theorem staging_threads_are_group0 (rg col : Nat) (hrg : rg < 4) (hcol : col < 128) :
    rg * 128 + col < 128 ↔ rg = 0 := by
  omega

/-- The RMS reduction uses warp slots 0..3, which fit in `wsum`. -/
theorem rms_warp_slot (col : Nat) (h : col < 128) : col / 32 < 4 := by
  omega

/-- Every warp slot index of the 512-thread block is a valid `wsum` index. -/
theorem wsum_index_bound (tid : Nat) (h : tid < 512) : tid / 32 < 16 := by
  omega

/-- Row-group-0 warps are exactly threads 0..127: their lane-0 slots are 0..3. -/
theorem rms_slots_cover_group0 (tid : Nat) (h : tid < 128) : tid / 32 < 4 := by
  omega

-- § state addressing

/-- Query/key heads are paired by modulo, and the pairing stays in range. -/
theorem qk_head_in_range (head h_k : Nat) (hk : 0 < h_k) : head % h_k < h_k :=
  Nat.mod_lt _ hk

/-- State element address bound for the (row, head, col) layout `(S, h_v, S)`. -/
theorem state_index_lt (row head col h_v : Nat) (hrow : row < 128) (hhead : head < h_v)
    (hcol : col < 128) : (row * h_v + head) * 128 + col < 128 * h_v * 128 :=
  idx_lt (idx_lt hrow hhead) hcol

/-- Stepping from row `rg*32` by `r` rows adds `r * (h_v*128)` to the address. -/
theorem state_row_stride (rg r head col h_v : Nat) :
    ((rg * 32 + r) * h_v + head) * 128 + col =
      ((rg * 32) * h_v + head) * 128 + col + r * (h_v * 128) := by
  simp only [Nat.add_mul, Nat.mul_assoc]
  omega

/-- Output / v / z addressing `head*S + col` stays inside `h_v * S`. -/
theorem out_index_lt (head col h_v : Nat) (hhead : head < h_v) (hcol : col < 128) :
    head * 128 + col < h_v * 128 :=
  idx_lt hhead hcol

-- § conv + L2 kernel

/-- With channels a multiple of 128, block `b` thread `t` addresses channel below `channels`. -/
theorem conv_channel_in_range (channels b t : Nat) (hc : channels % 128 = 0)
    (hb : b < channels / 128) (ht : t < 128) : b * 128 + t < channels := by
  omega

/-- History (stride 3) and weight (stride 4) addresses stay in range. -/
theorem conv_strides_in_range (channels c : Nat) (hc : c < channels) :
    c * 3 + 2 < 3 * channels ∧ c * 4 + 3 < 4 * channels := by
  omega

/-- The first `qk_heads` blocks are exactly the channels below `qk_heads*128`. -/
theorem l2_applies_to_qk_channels (qk_heads c : Nat) :
    c / 128 < qk_heads ↔ c < qk_heads * 128 := by
  omega

/-- The L2 partial array has 4 slots, one per warp of a 128-thread block. -/
theorem l2_warp_slots (t : Nat) (h : t < 128) : t / 32 < 4 := by
  omega

-- § alpha/beta kernel

/-- Grid size covers all `2*h_v` rows (8 warps per block). -/
theorem ab_grid_covers (h_v : Nat) : 2 * h_v ≤ (2 * h_v + 7) / 8 * 8 := by
  omega

/-- A row index below `2*h_v` is either an alpha row or a beta row with local index below `h_v`. -/
theorem ab_row_split (row h_v : Nat) (h : row < 2 * h_v) :
    (row < h_v ∧ row < h_v) ∨ (h_v ≤ row ∧ row - h_v < h_v) := by
  omega

/-- Each lane handles 8-element chunks; with `n % 8 = 0` the chunks tile `n` exactly. -/
theorem ab_chunks_tile (n j : Nat) (hn : n % 8 = 0) (hj : j < n / 8) : j * 8 + 7 < n := by
  omega

/-- The warp index within a 256-thread block is below 8. -/
theorem ab_warp_in_block (t : Nat) (h : t < 256) : t / 32 < 8 := by
  omega

-- § launch contract

/-- `fused_gdn_step_norm` precondition: `h_v` is a positive multiple of `h_k`. -/
def stepNormArgsValid (h_k h_v : Nat) : Prop := 0 < h_k ∧ 0 < h_v ∧ h_v % h_k = 0

/-- Valid arguments give a launch of `h_v` blocks, each with an in-range q/k head. -/
theorem step_norm_valid_heads (h_k h_v head : Nat) (hv : stepNormArgsValid h_k h_v) :
    head % h_k < h_k := by
  obtain ⟨hk, _, _⟩ := hv
  exact qk_head_in_range head h_k hk

/-- `fused_gdn_conv_l2` precondition: channels is a multiple of 128 and `qk_heads <= channels/128`. -/
def convArgsValid (channels qk_heads : Nat) : Prop :=
  channels % 128 = 0 ∧ qk_heads ≤ channels / 128

/-- Under the contract, the normalised channel range lies inside the channel range. -/
theorem conv_qk_range_valid (channels qk_heads : Nat) (h : convArgsValid channels qk_heads) :
    qk_heads * 128 ≤ channels := by
  obtain ⟨h1, h2⟩ := h
  omega

end Strata.FusedGdnDecode

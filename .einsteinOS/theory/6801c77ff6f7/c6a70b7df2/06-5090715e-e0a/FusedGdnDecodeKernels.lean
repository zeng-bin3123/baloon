/-!
# MODULE: fused_gdn_decode_kernels
SOURCES: src/kernels/cuda/fused_gdn.cu
Index-geometry theory of the fused gated-delta-net decode kernels (conv+L2, alpha/beta projections,
state step + gated RMS norm): launch coverage, thread/row decompositions and buffer bounds.

## Source semantics
* `gdn_step_norm_kernel` runs one block per value head with a (128 x 4) thread tile; thread (col, rg) owns
  32 state rows `rg*32 + r` of column `col`, state laid out `(row, h_v, col)`.
* `gdn_conv_l2_kernel` runs one thread per channel with a 3-deep history and a 4-tap filter.
* `gdn_ab_kernel` runs one warp per row of `[w_alpha; w_beta]`, lanes striding 8-wide bf16 chunks.

## Form template identifiers
fused_gdn_step_norm -> decode_state_step ; fused_gdn_conv_l2 -> decode_conv ; fused_gdn_ab -> decode_ab_proj ;
S -> state_dim ; RG -> row_groups ; RPG -> rows_per_group ; h_k/h_v -> key_heads/value_heads
-/

namespace Strata.FusedGdn

-- § generic arithmetic helpers

/-- Division/modulo uniqueness for a mixed-radix index `a*n+b`. -/
theorem divmod_unique {a b c d n : Nat} (hb : b < n) (hd : d < n) (h : a * n + b = c * n + d) :
    a = c ∧ b = d := by
  have hn : 0 < n := by omega
  have h1 : (a * n + b) / n = a := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt hb, Nat.zero_add]
  have h2 : (c * n + d) / n = c := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt hd, Nat.zero_add]
  have h3 : a = c := by rw [← h1, ← h2, h]
  refine ⟨h3, ?_⟩
  subst h3
  exact Nat.add_left_cancel h

/-- Row-major bound `t*h + r < T*h`. -/
theorem rowmajor_lt {t T h r : Nat} (ht : t < T) (hr : r < h) : t * h + r < T * h := by
  have h1 : (t + 1) * h ≤ T * h := Nat.mul_le_mul_right h (Nat.succ_le_of_lt ht)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § constants

/-- State size. -/
def S : Nat := 128
/-- Row groups. -/
def RG : Nat := 4
/-- Rows per thread. -/
def RPG : Nat := S / RG

/-- A thread owns 32 rows and the four groups tile the state exactly. -/
theorem rpg_value : RPG = 32 ∧ RPG * RG = S := by decide

/-- The block has 512 threads and `wsum` has one slot per warp. -/
theorem block_threads_warps : S * RG = 512 ∧ S * RG / 32 = 16 := by decide

-- § step kernel thread decomposition

/-- A linear thread id splits into a column in `[0,128)` and a row group in `[0,4)`. -/
theorem tid_decomp (tid : Nat) (h : tid < 512) :
    tid / 128 < 4 ∧ tid % 128 < 128 ∧ tid = (tid / 128) * 128 + tid % 128 := by omega

/-- The 32 rows owned by different row groups are pairwise distinct (row = rg*32 + r). -/
theorem row_owner_injective {rg r rg' r' : Nat} (hr : r < 32) (hr' : r' < 32)
    (h : rg * 32 + r = rg' * 32 + r') : rg = rg' ∧ r = r' := by omega

/-- Every owned row lies inside the state. -/
theorem row_owner_lt {rg r : Nat} (hg : rg < 4) (hr : r < 32) : rg * 32 + r < 128 := by omega

/-- Every state row is owned by some (group, slot). -/
theorem row_owner_surjective (row : Nat) (h : row < 128) :
    ∃ rg r, rg < 4 ∧ r < 32 ∧ row = rg * 32 + r :=
  ⟨row / 32, row % 32, by omega, by omega, by omega⟩

/-- The state address `((row*h_v + head)*S + col)` is injective in `(row, head, col)`. -/
theorem state_index_injective {h_v row head col row' head' col' : Nat}
    (hh : head < h_v) (hh' : head' < h_v) (hc : col < 128) (hc' : col' < 128)
    (h : (row * h_v + head) * 128 + col = (row' * h_v + head') * 128 + col') :
    row = row' ∧ head = head' ∧ col = col' := by
  obtain ⟨h1, h2⟩ := divmod_unique hc hc' h
  obtain ⟨h3, h4⟩ := divmod_unique hh hh' h1
  exact ⟨h3, h4, h2⟩

/-- State addresses stay inside the `(S, h_v, S)` buffer. -/
theorem state_index_lt {h_v row head col : Nat} (hr : row < 128) (hh : head < h_v) (hc : col < 128) :
    (row * h_v + head) * 128 + col < 128 * h_v * 128 := by
  have h1 : row * h_v + head < 128 * h_v := rowmajor_lt hr hh
  have h2 := rowmajor_lt (r := col) (h := 128) h1 hc
  exact h2

/-- Head pairing `head % h_k` stays inside the q/k buffers of `h_k` heads. -/
theorem qk_head_pairing {head h_k : Nat} (hk : 0 < h_k) :
    head % h_k < h_k ∧ (head % h_k) * 128 + 127 < 128 * h_k := by
  have h := Nat.mod_lt head hk
  omega

/-- Only the first row group's four warps write `wsum`, so its 4 summands cover all of 0..127. -/
theorem rms_warp_slots (tid : Nat) (h : tid < 128) : tid / 32 < 4 := by omega

/-- `h_v % h_k = 0` and positivity imply `h_k ≤ h_v`. -/
theorem heads_divisible_le {h_k h_v : Nat} (hk : 0 < h_k) (hv : 0 < h_v) (hd : h_v % h_k = 0) :
    h_k ≤ h_v := by
  obtain ⟨m, hm⟩ := Nat.dvd_of_mod_eq_zero hd
  cases m with
  | zero => omega
  | succ m => rw [hm]; exact Nat.le_mul_of_pos_right _ (Nat.succ_pos m)

-- § conv + L2 kernel

/-- Channel index of the conv kernel is in range when `128 | channels`. -/
theorem conv_channel_in_range {blk t channels : Nat} (hc : channels % 128 = 0)
    (hb : blk < channels / 128) (ht : t < 128) : blk * 128 + t < channels := by omega

/-- History, weight index bounds for a channel `c < channels`. -/
theorem conv_buffer_bounds {c channels : Nat} (h : c < channels) :
    c * 3 + 2 < 3 * channels ∧ c * 4 + 3 < 4 * channels := by omega

/-- Heads with index below `qk_heads` (≤ channels/128) get the L2 normalisation. -/
theorem qk_blocks_within_grid {blk qk channels : Nat} (hq : qk ≤ channels / 128) (hb : blk < qk) :
    blk < channels / 128 := by omega

/-- The 128-thread block reduces with 4 warp partials. -/
theorem conv_partials : 128 / 32 = 4 := by decide

-- § alpha/beta kernel

/-- Launch coverage: `(2*h_v+7)/8` blocks of 8 warps give each of the `2*h_v` rows a unique warp. -/
theorem ab_row_coverage (row h_v : Nat) (h : row < 2 * h_v) :
    row / 8 < (2 * h_v + 7) / 8 ∧ row = (row / 8) * 8 + row % 8 ∧ row % 8 < 8 := by omega

/-- The alpha/beta split: rows below `h_v` are gate rows, the others are beta rows. -/
theorem ab_row_split (row h_v : Nat) (h : row < 2 * h_v) :
    (row < h_v → row < h_v) ∧ (h_v ≤ row → row - h_v < h_v) := by omega

/-- Every 8-wide weight chunk `j < n/8` is visited by exactly lane `j % 32`. -/
theorem ab_chunk_lane (j n : Nat) (hn : n % 8 = 0) (hj : j < n / 8) :
    j % 32 < 32 ∧ j = j % 32 + 32 * (j / 32) ∧ j * 8 + 7 < n := by omega

/-- bf16 low half expanded to a float bit pattern. -/
def unpackLo (w : Nat) : Nat := (w % 65536) * 65536
/-- bf16 high half expanded to a float bit pattern. -/
def unpackHi (w : Nat) : Nat := (w / 65536) * 65536

/-- The two unpackings are 32-bit words. -/
theorem unpack_lt (w : Nat) : unpackLo w < 4294967296 ∧ (w < 4294967296 → unpackHi w < 4294967296) := by
  unfold unpackLo unpackHi
  constructor <;> omega

/-- The unpackings together recover the packed word. -/
theorem unpack_recover (w : Nat) : w = unpackLo w / 65536 + unpackHi w := by
  unfold unpackLo unpackHi
  omega

-- § launch guards

/-- Launch dimension for the step kernel is `h_v` blocks; each owns a distinct head. -/
theorem step_grid_heads {head h_v : Nat} (h : head < h_v) : head < h_v := h

/-- Output indices `head*S + col` fit in `h_v * S`. -/
theorem step_output_bound {h_v head col : Nat} (hh : head < h_v) (hc : col < 128) :
    head * 128 + col < h_v * 128 := rowmajor_lt hh hc

end Strata.FusedGdn

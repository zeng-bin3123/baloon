/-!
# MODULE: fused_gdn_step_norm_kernels
Row-group sharding, channel/head block mapping and bf16 chunk decoding of the fused gated-delta-net kernels
(`gdn_step_norm_kernel`, `gdn_conv_l2_kernel`, `gdn_ab_kernel`).

## Source semantics
`S = 128`.  The step kernel runs one block per head with `128 x 4` threads: thread `(col, rg)` keeps 32
state rows `rg*32 .. rg*32+31` of column `col` in registers; the four row-group partials are summed through
shared memory, so the result equals the sequential contraction over all 128 rows.  The conv kernel runs one
128-channel block per (q/k/v) head and L2-normalises only the first `qk_heads` blocks.  The `ab` kernel
assigns one warp per output row of the stacked `[alpha; beta]` projection and decodes bf16 weights eight at
a time from `uint4` chunks (even element = low half shifted up, odd element = high half masked).

## Form template identifiers
- `S`, `RG`, `RPG`         -> state size, row groups, rows per thread
- `sumRange f n`           -> sequential reduction of `f 0 .. f (n-1)`
- `stateIdx h_v head row col` -> address of state[row][head][col]
-/

namespace Strata.Kernels.FusedGdn

-- § arithmetic helpers

/-- Row-major bound. -/
theorem mul_add_lt_aux {a b c n : Nat} (ha : a < b) (hc : c < n) : a * n + c < b * n := by
  have h : (a + 1) * n ≤ b * n := Nat.mul_le_mul_right n ha
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- Division recovers the row. -/
theorem div_of_split_aux {a c n : Nat} (hc : c < n) : (a * n + c) / n = a := by
  have hn : 0 < n := by omega
  have h1 := Nat.add_mul_div_right c a hn
  have h0 : c / n = 0 := Nat.div_eq_of_lt hc
  have e : a * n + c = c + a * n := Nat.add_comm _ _
  rw [e]
  omega

/-- Row-major addresses are unique. -/
theorem split_unique_aux {a a' c c' n : Nat} (hc : c < n) (hc' : c' < n)
    (h : a * n + c = a' * n + c') : a = a' ∧ c = c' := by
  have h1 : (a * n + c) / n = a := div_of_split_aux hc
  have h2 : (a' * n + c') / n = a' := div_of_split_aux hc'
  have ha : a = a' := by rw [← h1, ← h2, h]
  subst ha
  exact ⟨rfl, by omega⟩

-- § constants

/-- State size. -/
def S : Nat := 128
/-- Row groups per column. -/
def RG : Nat := 4
/-- Rows held by one thread. -/
def RPG : Nat := 32

/-- The row groups tile the state exactly. -/
theorem row_groups_tile : RG * RPG = S := by
  decide

/-- `wsum` has one slot per warp of the 512-thread block. -/
theorem wsum_size : S * RG / 32 = 16 := by
  decide

/-- Block of `128 x 4` threads. -/
theorem block_threads : S * RG = 512 := by
  decide

-- § sequential reduction and row-group sharding

/-- Sequential sum of `f 0 .. f (n-1)`. -/
def sumRange (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => sumRange f n + f n

/-- A range sum splits at any point. -/
theorem sumRange_add (f : Nat → Int) (n m : Nat) :
    sumRange f (n + m) = sumRange f n + sumRange (fun i => f (n + i)) m := by
  induction m with
  | zero => simp [sumRange]
  | succ m ih =>
    show sumRange f (n + m) + f (n + m) =
      sumRange f n + (sumRange (fun i => f (n + i)) m + f (n + m))
    rw [ih]
    omega

/-- Four row groups of 32 rows add up to the full 128-row contraction. -/
theorem rows_four_way (f : Nat → Int) :
    sumRange f 128 =
      sumRange (fun r => f (0 * 32 + r)) 32 + sumRange (fun r => f (1 * 32 + r)) 32 +
      sumRange (fun r => f (2 * 32 + r)) 32 + sumRange (fun r => f (3 * 32 + r)) 32 := by
  have h1 : sumRange f 128 = sumRange f 96 + sumRange (fun r => f (3 * 32 + r)) 32 :=
    sumRange_add f 96 32
  have h2 : sumRange f 96 = sumRange f 64 + sumRange (fun r => f (2 * 32 + r)) 32 :=
    sumRange_add f 64 32
  have h3 : sumRange f 64 = sumRange f 32 + sumRange (fun r => f (1 * 32 + r)) 32 :=
    sumRange_add f 32 32
  have h4 : sumRange f 32 = sumRange f 0 + sumRange (fun r => f (0 * 32 + r)) 32 :=
    sumRange_add f 0 32
  have h0 : sumRange f 0 = 0 := rfl
  omega

-- § thread and row indexing

/-- Rows split uniquely into (row group, row-in-group). -/
theorem row_partition (i : Nat) (hi : i < 128) :
    i / 32 < 4 ∧ i % 32 < 32 ∧ i = (i / 32) * 32 + i % 32 := by
  omega

/-- A thread's row `rg*32 + r` is a valid state row. -/
theorem row_lt {rg r : Nat} (hrg : rg < 4) (hr : r < 32) : rg * 32 + r < 128 := by
  omega

/-- Distinct (row group, r) give distinct rows. -/
theorem row_unique {rg rg' r r' : Nat} (hr : r < 32) (hr' : r' < 32)
    (h : rg * 32 + r = rg' * 32 + r') : rg = rg' ∧ r = r' :=
  split_unique_aux hr hr' h

/-- Linear thread id `rg*128 + col` is below 512 and determines `(rg, col)`. -/
theorem tid_unique {rg rg' col col' : Nat} (hc : col < 128) (hc' : col' < 128)
    (h : rg * 128 + col = rg' * 128 + col') : rg = rg' ∧ col = col' :=
  split_unique_aux hc hc' h

/-- Thread ids stay below the block size. -/
theorem tid_lt {rg col : Nat} (hrg : rg < 4) (hc : col < 128) : rg * 128 + col < 512 := by
  omega

/-- Threads below 128 are exactly row group 0, and their warp id indexes `wsum[0..3]`. -/
theorem group0_threads {rg col : Nat} (hc : col < 128) (hrg : rg < 4)
    (h : rg * 128 + col < 128) : rg = 0 ∧ (rg * 128 + col) / 32 < 4 := by
  omega

/-- Address of `state[row][head][col]` in the `(S, h_v, S)` layout. -/
def stateIdx (h_v head row col : Nat) : Nat := (row * h_v + head) * 128 + col

/-- Every in-range address is inside the state allocation. -/
theorem stateIdx_lt {h_v head row col : Nat} (hh : head < h_v) (hr : row < 128) (hc : col < 128) :
    stateIdx h_v head row col < 128 * h_v * 128 := by
  unfold stateIdx
  have h1 : row * h_v + head < 128 * h_v := mul_add_lt_aux hr hh
  exact mul_add_lt_aux h1 hc

/-- Distinct `(row, head, col)` have distinct addresses. -/
theorem stateIdx_inj {h_v head head' row row' col col' : Nat} (hh : head < h_v) (hh' : head' < h_v)
    (hc : col < 128) (hc' : col' < 128)
    (e : stateIdx h_v head row col = stateIdx h_v head' row' col') :
    row = row' ∧ head = head' ∧ col = col' := by
  unfold stateIdx at e
  obtain ⟨e1, e2⟩ := split_unique_aux hc hc' e
  obtain ⟨e3, e4⟩ := split_unique_aux hh hh' e1
  exact ⟨e3, e4, e2⟩

-- § conv/l2 kernel geometry

/-- Channel `c = bx*128 + tx` of block `bx < channels/128` is inside the channel range. -/
theorem channel_lt {channels bx tx : Nat} (hdiv : channels % 128 = 0) (hbx : bx < channels / 128)
    (htx : tx < 128) : bx * 128 + tx < channels := by
  omega

/-- The 3-row history of channel `c` is inside the `3*channels` buffer. -/
theorem hist_index_lt {c channels j : Nat} (hc : c < channels) (hj : j < 3) :
    c * 3 + j < channels * 3 :=
  mul_add_lt_aux hc hj

/-- The 4 conv taps of channel `c` are inside the `4*channels` weight buffer. -/
theorem tap_index_lt {c channels j : Nat} (hc : c < channels) (hj : j < 4) :
    c * 4 + j < channels * 4 :=
  mul_add_lt_aux hc hj

/-- A launch with `qk_heads <= channels/128` normalises only whole heads. -/
theorem qk_blocks_in_grid {channels qk_heads bx : Nat} (h : qk_heads ≤ channels / 128)
    (hbx : bx < qk_heads) : bx < channels / 128 := by
  omega

-- § alpha/beta kernel geometry

/-- Stacked row `row < 2*h_v` splits into alpha (`row < h_v`) or beta (`row - h_v`) rows. -/
theorem ab_row_split {row h_v : Nat} (h : row < 2 * h_v) :
    (if row ≥ h_v then row - h_v else row) < h_v := by
  split <;> omega

/-- The 8-warps-per-block grid covers all `2*h_v` rows. -/
theorem ab_grid_cover (h_v : Nat) : 2 * h_v ≤ ((2 * h_v + 7) / 8) * 8 := by
  omega

/-- Element `e` of a row lives in `uint4` chunk `e/8`, word `(e%8)/2`, half `e%2`. -/
theorem chunk_decomposition {e n : Nat} (hn : n % 8 = 0) (he : e < n) :
    e / 8 < n / 8 ∧ (e % 8) / 2 < 4 ∧ e = 8 * (e / 8) + 2 * ((e % 8) / 2) + e % 2 := by
  omega

/-- Shifting a 32-bit word left by 16 keeps exactly its low half in the high half. -/
theorem low_half_shift (x : Nat) : (x * 65536) % 4294967296 = (x % 65536) * 65536 := by
  omega

/-- The two bf16 halves of a word reassemble the word. -/
theorem halves_reassemble (x : Nat) : (x / 65536) * 65536 + x % 65536 = x := by
  omega

/-- `h_v` divisible by `h_k > 0` forces `h_k <= h_v` (the validity check implies a real grouping). -/
theorem head_group_le {h_k h_v : Nat} (hv : 0 < h_v) (h : h_v % h_k = 0) : h_k ≤ h_v :=
  Nat.le_of_dvd hv (Nat.dvd_of_mod_eq_zero h)

/-- The shared q/k head `head % h_k` is in range. -/
theorem qk_head_lt {head h_k : Nat} (hk : 0 < h_k) : head % h_k < h_k :=
  Nat.mod_lt head hk

end Strata.Kernels.FusedGdn

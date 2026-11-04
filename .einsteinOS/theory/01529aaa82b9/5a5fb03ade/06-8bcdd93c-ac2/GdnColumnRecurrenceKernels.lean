/-!
# MODULE: gdn_column_recurrence_kernels
Address, launch-geometry and conv-window contracts of the gated delta-net's non-projection kernels
(`gdn_step_kernel`, `gdn_conv_kernel`, `gdn_l2_kernel`, `gdn_out_norm_kernel`, `gdn_beta_gate_kernel`).

## Source semantics
The state is `(S, h_v, S)` with `j` fastest.  One thread owns one `(h, j)` column and walks `i` in `0..S-1`;
no barrier is needed because every access is within one column.  The grid is `(ceil(S/32), ceil(h_v/MAX_H))`
with `MAX_H = 1` (latency hiding: 192 blocks instead of 24).  The conv state slides: drop the oldest row,
append the newest; kernel tap 0 multiplies the oldest row, the last tap multiplies the new input.
q/k heads are paired with value heads by `h % h_k`.

## Form template identifiers
- `state` -> `stateIdx S h_v i h j`            (flat address of state[i][h][j])
- `JTHREADS`, `MAX_H`                           (block shape knobs)
- `conv_state`/`kW`                             (`slide`, `dot`)
-/

namespace Strata.Kernels.GdnColumn

-- § arithmetic helpers

/-- `a < b` and `c < n` give `a*n + c < b*n`: the basic row-major bound. -/
theorem mul_add_lt_aux {a b c n : Nat} (ha : a < b) (hc : c < n) : a * n + c < b * n := by
  have h : (a + 1) * n ≤ b * n := Nat.mul_le_mul_right n ha
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- Division recovers the row of a row-major address. -/
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

-- § constants and addressing

/-- Threads along `j`. -/
def JTHREADS : Nat := 32

/-- Heads staged per block (latency-hiding knob). -/
def MAX_H : Nat := 1

/-- Flat address of `state[i][h][j]` in the `(S, h_v, S)` layout with `j` fastest. -/
def stateIdx (S h_v i h j : Nat) : Nat := (i * h_v + h) * S + j

/-- Every in-range state address is inside the `S*h_v*S` allocation. -/
theorem stateIdx_lt {S h_v i h j : Nat} (hi : i < S) (hh : h < h_v) (hj : j < S) :
    stateIdx S h_v i h j < S * h_v * S := by
  unfold stateIdx
  have h1 : i * h_v + h < S * h_v := mul_add_lt_aux hi hh
  exact mul_add_lt_aux h1 hj

/-- Distinct `(i, h, j)` never alias: columns are disjoint, so one thread per column needs no barrier. -/
theorem stateIdx_inj {S h_v i i' h h' j j' : Nat} (hh : h < h_v) (hh' : h' < h_v)
    (hj : j < S) (hj' : j' < S)
    (e : stateIdx S h_v i h j = stateIdx S h_v i' h' j') : i = i' ∧ h = h' ∧ j = j' := by
  unfold stateIdx at e
  obtain ⟨e1, e2⟩ := split_unique_aux hj hj' e
  obtain ⟨e3, e4⟩ := split_unique_aux hh hh' e1
  exact ⟨e3, e4, e2⟩

/-- Two threads of different columns touch different addresses at every step `i`. -/
theorem column_private {S h_v i i' h h' j j' : Nat} (hh : h < h_v) (hh' : h' < h_v)
    (hj : j < S) (hj' : j' < S) (hne : h ≠ h' ∨ j ≠ j') :
    stateIdx S h_v i h j ≠ stateIdx S h_v i' h' j' := by
  intro e
  obtain ⟨_, e2, e3⟩ := stateIdx_inj hh hh' hj hj' e
  rcases hne with h1 | h1
  · exact h1 e2
  · exact h1 e3

/-- The paired q/k row `(h % h_k)*S + i` is inside the `h_k*S` q/k buffer. -/
theorem qk_index_lt {h h_k S i : Nat} (hk : 0 < h_k) (hi : i < S) :
    (h % h_k) * S + i < h_k * S :=
  mul_add_lt_aux (Nat.mod_lt h hk) hi

/-- Value-side address `h*S + j` stays inside the `h_v*S` buffer. -/
theorem value_index_lt {h h_v S j : Nat} (hh : h < h_v) (hj : j < S) :
    h * S + j < h_v * S :=
  mul_add_lt_aux hh hj

-- § launch geometry

/-- Every column `j < S` is covered by exactly the block/thread pair `(j/32, j%32)`. -/
theorem column_cover (S j : Nat) (hj : j < S) :
    j / JTHREADS < (S + JTHREADS - 1) / JTHREADS ∧ j % JTHREADS < JTHREADS ∧
      (j / JTHREADS) * JTHREADS + j % JTHREADS = j := by
  unfold JTHREADS
  omega

/-- Every head is covered by block row `h / MAX_H`, slot `h % MAX_H`, inside that block's `nh`. -/
theorem head_cover (h_v h : Nat) (hh : h < h_v) :
    h / MAX_H < (h_v + MAX_H - 1) / MAX_H ∧ h / MAX_H * MAX_H + h % MAX_H = h ∧
      h % MAX_H < min MAX_H (h_v - h / MAX_H * MAX_H) := by
  unfold MAX_H
  omega

/-- The tuned grid at the real geometry: 192 blocks. -/
theorem grid_blocks_tuned :
    ((128 + JTHREADS - 1) / JTHREADS) * ((48 + MAX_H - 1) / MAX_H) = 192 := by
  decide

/-- The untuned `MAX_H = 8` grid had only 24 blocks. -/
theorem grid_blocks_untuned :
    ((128 + JTHREADS - 1) / JTHREADS) * ((48 + 8 - 1) / 8) = 24 := by
  decide

/-- 24 single-warp blocks leave half of 48 SMs idle; 192 blocks give four per SM. -/
theorem sm_occupancy : 24 < 48 ∧ 192 = 4 * 48 := by
  decide

/-- Total columns `S * h_v` do not depend on the packing. -/
theorem thread_count_fixed : 128 * 48 = 6144 := by
  decide

/-- A 1-D launch with `blocks = ceil(n/B)` covers `n` (B = 128, as in the beta gate). -/
theorem launch_cover_128 (n : Nat) : n ≤ ((n + 127) / 128) * 128 := by
  omega

/-- Same for the 256-thread conv launch. -/
theorem launch_cover_256 (n : Nat) : n ≤ ((n + 255) / 256) * 256 := by
  omega

/-- The staged `ks`/`qs` arrays hold 128 entries, so `S <= 128` suffices for the guard. -/
theorem staged_fit {S i : Nat} (hS : S ≤ 128) (hi : i < S) : i < 128 := by
  omega

/-- A warp striding a row of `cols <= 1024` does at most 32 iterations per lane. -/
theorem warp_stride_bound (cols : Nat) (h : cols ≤ 1024) : (cols + 31) / 32 ≤ 32 := by
  omega

/-- Each element `i < cols` is visited by lane `i % 32` at iteration `i / 32`, exactly once. -/
theorem warp_stride_cover (cols i : Nat) (hi : i < cols) :
    i % 32 < 32 ∧ (i / 32) * 32 + i % 32 = i := by
  omega

-- § conv window

/-- Integer dot product of two lists (truncating to the shorter). -/
def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- The conv state after one step: drop the oldest row, append the newest. -/
def slide (st : List Int) (x : Int) : List Int := st.tail ++ [x]

/-- With `d_conv-1` stored rows (non-empty) the slide keeps the state size. -/
theorem slide_length (st : List Int) (x : Int) (h : st ≠ []) :
    (slide st x).length = st.length := by
  cases st with
  | nil => exact absurd rfl h
  | cons a as => simp [slide]

/-- The slid state is the window `[state | x]` without its first (oldest) row. -/
theorem slide_eq_drop (st : List Int) (x : Int) (h : st ≠ []) :
    slide st x = (st ++ [x]).drop 1 := by
  cases st with
  | nil => exact absurd rfl h
  | cons a as => simp [slide]

/-- The last tap multiplies the new input; the earlier taps run over the stored rows. -/
theorem dot_snoc (st w' : List Int) (x wl : Int) (h : st.length = w'.length) :
    dot (st ++ [x]) (w' ++ [wl]) = dot st w' + x * wl := by
  induction st generalizing w' with
  | nil =>
    cases w' with
    | nil => simp [dot]
    | cons b bs => simp at h
  | cons a as ih =>
    cases w' with
    | nil => simp at h
    | cons b bs =>
      have h' : as.length = bs.length := by simpa using h
      simp only [List.cons_append, dot]
      rw [ih bs h']
      omega

/-- The kernel's output `sum_{i<dc-1} st[i]*w[i] + x*w[dc-1]` is the dot of the full window. -/
theorem conv_out_is_window_dot (st w' : List Int) (x wl : Int) (h : st.length = w'.length) :
    dot st w' + x * wl = dot (st ++ [x]) (w' ++ [wl]) :=
  (dot_snoc st w' x wl h).symm

end Strata.Kernels.GdnColumn

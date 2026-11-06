/-!
# MODULE: gdn_reference_kernels
SOURCES: src/kernels/cuda/gdn.cu
Non-projection parts of the gated delta net: the recurrence step (one thread per state column), the sliding
causal conv, per-row L2 norm, per-head gated RMS norm and the beta sigmoid gate.

## Source semantics
* State layout is `(S, h_v, S)` with `j` fastest, so column (h, j) is at `h*S + j + i*(h_v*S)`.
* Each thread owns one (h, j) column end to end: no barrier is needed inside the recurrence.
* Conv state slides: drop oldest, append newest; the new input lands in the last tap.
* `MAX_H = 1` heads per block (a latency-hiding choice, not a capacity one).

## Form template identifiers
gdn_step_kernel -> delta_rule_step ; gdn_conv_kernel -> causal_conv_step ; gdn_l2_kernel -> l2_row_norm ;
gdn_out_norm_kernel -> gated_out_norm ; gdn_beta_gate_kernel -> beta_sigmoid
-/

namespace Strata.Gdn

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

-- § step kernel geometry

/-- Threads along `j`. -/
def JTHREADS : Nat := 32
/-- Heads per block. -/
def MAX_H : Nat := 1

/-- The grid is `ceil(S/32) x ceil(h_v/MAX_H)`: at the real size 4 x 48, i.e. 192 blocks. -/
theorem real_grid : (128 + 31) / 32 = 4 ∧ (48 + 1 - 1) / 1 = 48 ∧ 4 * 48 = 192 := by decide

/-- Every column `j < S` is assigned to a block along x and a lane. -/
theorem column_coverage (j S : Nat) (h : j < S) :
    j / 32 < (S + 31) / 32 ∧ j % 32 < 32 ∧ j = (j / 32) * 32 + j % 32 := by omega

/-- With one head per block, block `by` handles exactly head `by` (and nothing beyond `h_v`). -/
theorem head_per_block (by_ h_v : Nat) (hb : by_ < h_v) :
    min MAX_H (h_v - by_ * MAX_H) = 1 ∧ by_ * MAX_H < h_v := by
  unfold MAX_H
  omega

/-- Heads handled by a block are below `h_v`. -/
theorem staged_head_lt (by_ hi h_v : Nat) (hb : by_ < h_v) (hi_lt : hi < min MAX_H (h_v - by_ * MAX_H)) :
    by_ * MAX_H + hi < h_v := by
  unfold MAX_H at *
  omega

/-- Column address `h*S + j + i*(h_v*S)` equals the flat `(i*h_v + h)*S + j` address. -/
theorem column_address_flat (i h j h_v S : Nat) :
    i * (h_v * S) + (h * S + j) = (i * h_v + h) * S + j := by
  rw [Nat.add_mul, Nat.mul_assoc]
  omega

/-- The column map is injective in `(i, h, j)`: different threads never touch the same cell. -/
theorem column_address_injective {h_v i h j i' h' j' : Nat} (hh : h < h_v) (hh' : h' < h_v)
    (hj : j < 128) (hj' : j' < 128)
    (e : i * (h_v * 128) + (h * 128 + j) = i' * (h_v * 128) + (h' * 128 + j')) :
    i = i' ∧ h = h' ∧ j = j' := by
  rw [column_address_flat, column_address_flat] at e
  obtain ⟨a, b⟩ := divmod_unique hj hj' e
  obtain ⟨c, d⟩ := divmod_unique hh hh' a
  exact ⟨c, d, b⟩

/-- The staged `ks`/`qs` rows have 128 slots, so `S ≤ 128` is exactly the guard in `gdn_step`. -/
theorem staged_row_fits (i S : Nat) (hS : S ≤ 128) (hi : i < S) : i < 128 := by omega

/-- Modulo head pairing stays below `h_k`. -/
theorem modulo_pairing (h h_k : Nat) (hk : 0 < h_k) : h % h_k < h_k := Nat.mod_lt h hk

/-- Staging loop: element `i < S` of a row is loaded by lane `i % 32` in iteration `i / 32`. -/
theorem staging_owner (i : Nat) : i % 32 < 32 ∧ i = i % 32 + 32 * (i / 32) := by omega

-- § sliding conv

/-- Sliding the conv state keeps its length. -/
theorem slide_length {α : Type} (st : List α) (x : α) (h : st ≠ []) :
    ((st.drop 1) ++ [x]).length = st.length := by
  cases st with
  | nil => exact absurd rfl h
  | cons a t => simp

/-- The newest input is the last element after a slide. -/
theorem slide_last {α : Type} (st : List α) (x : α) :
    ((st.drop 1) ++ [x]).getLast? = some x := by
  simp

/-- With `d_conv = 4` the state has 3 rows and the state index `c*(dc-1) + i` stays in `3*C`. -/
theorem conv_state_bound {c C i : Nat} (hc : c < C) (hi : i < 3) : c * 3 + i < 3 * C := by omega

/-- The conv kernel's block count covers all channels. -/
theorem conv_blocks_cover (c C : Nat) (h : c < C) : c / 256 < (C + 255) / 256 := by omega

-- § norms and beta gate

/-- The L2 kernel's 32 threads stride columns; every `i < cols` has one owner. -/
theorem l2_owner (i : Nat) : i % 32 < 32 ∧ i = i % 32 + 32 * (i / 32) := by omega

/-- The guard `cols ≤ 1024` makes each lane's trip count at most 32. -/
theorem l2_trip_count (cols : Nat) (h : cols ≤ 1024) : (cols + 31) / 32 ≤ 32 := by omega

/-- Per-head rows `h*S + i` fit within `h_v*S`. -/
theorem head_row_bound {h_v h S i : Nat} (hh : h < h_v) (hi : i < S) : h * S + i < h_v * S :=
  rowmajor_lt hh hi

/-- The beta gate launches `ceil(n/128)` blocks of 128 threads and covers every head. -/
theorem beta_gate_cover (i n : Nat) (h : i < n) : i / 128 < (n + 127) / 128 := by omega

end Strata.Gdn

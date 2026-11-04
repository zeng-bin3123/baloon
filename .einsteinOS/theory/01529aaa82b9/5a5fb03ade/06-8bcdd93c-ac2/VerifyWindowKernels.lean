/-!
# MODULE: verify_window_kernels
Multi-token (verify window) variants of the GDN kernels plus the gather/select/steps utilities.

## Source semantics
The window of token `t` is the last four entries of `[hist0, hist1, hist2, x_0, ..., x_t]`; entry `s` of this
sequence is `hist[s]` for `s < 3` and `x[s-3]` otherwise.  Committing `n` tokens keeps the last three entries
of `[hist | x_0 .. x_{n-1}]` (`n <= 0` leaves the history unchanged).  `gdn_step_norm_multi` replays tokens
`0 .. n-1` (`n = *n_keep` or `T`), writes outputs only for `t >= t_out_begin`, and writes the state back only
when `n_keep` is given and `n > 0`.  `window_ids` clips the attention window to `start = max(n_kv - window, 0)`.
`dense_steps` stores `(c, c+1, (c+1)/4, c+1)` per cell.  Grid-stride loops cover each index exactly once.

## Form template identifiers
- `seqAt hist x s`     -> entry `s` of `[hist | x]`
- `commit`             -> history after keeping `n` tokens
- `windowStart/Width`  -> `window_ids` clipping
-/

namespace Strata.Kernels.Verify

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

-- § conv window over [hist | x]

/-- Entry `s` of the sequence `[hist0, hist1, hist2, x_0, x_1, ...]`. -/
def seqAt {α : Type} (hist : Nat → α) (x : Nat → α) (s : Nat) : α :=
  if s < 3 then hist s else x (s - 3)

/-- The newest window entry of token `t` is `x_t`. -/
theorem window_last {α : Type} (hist x : Nat → α) (t : Nat) : seqAt hist x (t + 3) = x t := by
  unfold seqAt
  rw [if_neg (by omega), Nat.add_sub_cancel]

/-- Committing zero tokens leaves the history unchanged. -/
theorem commit_zero {α : Type} (hist x : Nat → α) (j : Nat) (hj : j < 3) :
    seqAt hist x (0 + j) = hist j := by
  unfold seqAt
  rw [if_pos (by omega)]
  simp

/-- Committing `n` tokens then `m` more equals committing `n + m` at once. -/
theorem commit_compose {α : Type} (hist x : Nat → α) (n m j : Nat) :
    seqAt (fun j => seqAt hist x (n + j)) (fun i => x (n + i)) (m + j) =
      seqAt hist x (n + (m + j)) := by
  unfold seqAt
  by_cases h1 : m + j < 3
  · by_cases h2 : n + (m + j) < 3
    · simp [h1, h2]
    · simp [h1, h2]
  · have h2 : ¬ n + (m + j) < 3 := by omega
    have e : n + (m + j - 3) = n + (m + j) - 3 := by omega
    simp [h1, h2, e]

/-- Window and `x`/`hist` addresses: history channel `c*3 + j` and token row `(s-3)*C + c`. -/
theorem token_index_lt {s C c T : Nat} (hs : s < T + 3) (h3 : 3 ≤ s) (hc : c < C) :
    (s - 3) * C + c < T * C :=
  mul_add_lt_aux (by omega) hc

-- § step kernel indexing and replay

/-- Replayed tokens `t < t_out_begin` produce no output; outputs are exactly `[t_out_begin, n)`. -/
theorem outputs_count {n t_out_begin : Nat} (h : t_out_begin ≤ n) :
    ((List.range n).filter (fun t => decide (t_out_begin ≤ t))).length = n - t_out_begin := by
  induction n with
  | zero =>
    have : t_out_begin = 0 := by omega
    subst this
    simp
  | succ n ih =>
    by_cases hc : t_out_begin ≤ n
    · have ih' := ih hc
      rw [List.range_succ, List.filter_append]
      simp [hc, ih']
      omega
    · have e : t_out_begin = n + 1 := by omega
      subst e
      rw [List.range_succ, List.filter_append]
      have hnil : (List.range n).filter (fun t => decide (n + 1 ≤ t)) = [] := by
        apply List.filter_eq_nil_iff.mpr
        intro a ha
        have h1 := List.mem_range.mp ha
        have h2 : ¬ (n + 1 ≤ a) := by omega
        simp [h2]
      have hn : ¬ (n + 1 ≤ n) := by omega
      rw [hnil]
      simp [hn]

/-- The state is written back only when `n_keep` is supplied and positive. -/
def writeBack (hasKeep : Bool) (n : Nat) : Bool := hasKeep && decide (0 < n)

/-- Without `n_keep` the state is never written back. -/
theorem no_keep_no_writeback (n : Nat) : writeBack false n = false := by
  rfl

/-- With `n_keep = 0` the state is not written back either. -/
theorem zero_keep_no_writeback : writeBack true 0 = false := by
  decide

/-- Channel layout `C = 2*qk + value_dim`: q, k and v indices stay in their own thirds. -/
theorem qkv_layout {S h_k h_v qh head col C : Nat} (hqh : qh < h_k) (hh : head < h_v) (hc : col < S)
    (hC : C = 2 * (S * h_k) + S * h_v) :
    qh * S + col < S * h_k ∧ S * h_k + (qh * S + col) < 2 * (S * h_k) ∧
      2 * (S * h_k) + (head * S + col) < C := by
  have e1 : S * h_k = h_k * S := Nat.mul_comm _ _
  have e2 : S * h_v = h_v * S := Nat.mul_comm _ _
  have a1 := mul_add_lt_aux hqh hc
  have a2 := mul_add_lt_aux hh hc
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- Gate/beta slot `t*h_v + head` is inside `T*h_v`. -/
theorem gate_index_lt {t T head h_v : Nat} (ht : t < T) (hh : head < h_v) :
    t * h_v + head < T * h_v :=
  mul_add_lt_aux ht hh

-- § window_ids and dense_steps

/-- First visible position. -/
def windowStart (n_kv window : Nat) : Nat := n_kv - window

/-- Visible width. -/
def windowWidth (n_kv window : Nat) : Nat := n_kv - windowStart n_kv window

/-- The visible width never exceeds the window and the range ends at `n_kv`. -/
theorem window_bounds (n_kv window : Nat) :
    windowWidth n_kv window ≤ window ∧ windowStart n_kv window + windowWidth n_kv window = n_kv := by
  unfold windowWidth windowStart
  omega

/-- A short history is not clipped. -/
theorem window_short (n_kv window : Nat) (h : n_kv ≤ window) :
    windowStart n_kv window = 0 ∧ windowWidth n_kv window = n_kv := by
  unfold windowWidth windowStart
  omega

/-- Visible ids `start + j` (for `j < width`) are real positions `< n_kv`. -/
theorem window_ids_lt {n_kv window j : Nat} (hj : j < windowWidth n_kv window) :
    windowStart n_kv window + j < n_kv := by
  unfold windowWidth windowStart at *
  omega

/-- `dense_steps` slots `i*4 + k` (k < 4) are inside `4n`, and `(c+1)/4 <= c+1`. -/
theorem dense_steps_index {i n k c : Nat} (hi : i < n) (hk : k < 4) :
    i * 4 + k < n * 4 ∧ (c + 1) / 4 ≤ c + 1 :=
  ⟨mul_add_lt_aux hi hk, by omega⟩

-- § grid-stride loops and gathers

/-- A grid-stride loop with stride `s > 0` visits index `i` at thread `i % s`, iteration `i / s`. -/
theorem gridstride_cover {s : Nat} (hs : 0 < s) (i : Nat) :
    i % s < s ∧ (i / s) * s + i % s = i := by
  refine ⟨Nat.mod_lt i hs, ?_⟩
  have h := Nat.div_add_mod i s
  have e : s * (i / s) = (i / s) * s := Nat.mul_comm _ _
  omega

/-- That visit is unique. -/
theorem gridstride_unique {s t t' k k' : Nat} (ht : t < s) (ht' : t' < s)
    (h : k * s + t = k' * s + t') : k = k' ∧ t = t' :=
  split_unique_aux ht ht' h

/-- `gather_rows`: source address `ids[r]*row16 + o` is inside `nrows*row16` for valid ids. -/
theorem gather_src_lt {id nrows row16 o : Nat} (hid : id < nrows) (ho : o < row16) :
    id * row16 + o < nrows * row16 :=
  mul_add_lt_aux hid ho

/-- `gather_rows`: destination `r*row16 + o` splits back into `(r, o)`. -/
theorem gather_dst_unique {r r' o o' row16 : Nat} (ho : o < row16) (ho' : o' < row16)
    (h : r * row16 + o = r' * row16 + o') : r = r' ∧ o = o' :=
  split_unique_aux ho ho' h

/-- `embedding_gather_dev`: a stored code field has `bits` bits. -/
theorem code_field_lt (byte shift bits : Nat) : (byte / 2 ^ shift) % 2 ^ bits < 2 ^ bits :=
  Nat.mod_lt _ (Nat.two_pow_pos bits)

/-- With 4-bit codes two codes share a byte: index `i` is byte `i/2`, nibble `i%2`. -/
theorem nibble_split (i : Nat) : 8 / 4 = 2 ∧ i = 2 * (i / 2) + i % 2 := by
  omega

/-- Per-token row base `token*row_codes` plus an in-row offset stays inside the table. -/
theorem token_row_lt {token vocab row_codes o : Nat} (ht : token < vocab) (ho : o < row_codes) :
    token * row_codes + o < vocab * row_codes :=
  mul_add_lt_aux ht ho

-- § misc launch checks

/-- `ident_hits` runs one block of 1024 threads; `n <= 1024` guarantees `threadIdx < n` covers all ids. -/
theorem ident_hits_cover {n i : Nat} (hn : n ≤ 1024) (hi : i < n) : i < 1024 := by
  omega

/-- `rebase_ptrs` assigns `base + k*bytes`; consecutive blobs are adjacent and disjoint. -/
theorem rebase_adjacent (base bytes k : Nat) :
    (base + k * bytes) + bytes = base + (k + 1) * bytes := by
  rw [Nat.add_mul, Nat.one_mul]
  omega

end Strata.Kernels.Verify

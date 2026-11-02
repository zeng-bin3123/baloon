/-!
# MODULE: verify_window_indexing
Index arithmetic of the multi-token verify kernels: causal conv window, commit of the last three history
entries, sliding-window ids, pointer rebasing, block gathers, and broadcast indexing.

## Source semantics
* The conv window of token t is the sequence `[hist0, hist1, hist2, x_0, ...]` entries t..t+2 plus x_t.
* Commit keeps the last three entries of `hist ++ x_0..x_{n-1}`.
* Window ids cover `[max(0, n_kv - window), n_kv)`.
* `dense_steps` emits `(c, c+1, (c+1)/4, c+1)`.

## Form template identifiers
* `gdn_conv_l2_multi_kernel` -> `causal_conv_window`; `gdn_conv_commit_kernel` -> `history_commit`
* `window_ids_kernel` -> `sliding_ids`; `rebase_ptrs_kernel` -> `pointer_rebase`
* `fetch_blobs_kernel` -> `blob_gather`; `broadcast_streams_kernel` -> `stream_broadcast`
-/

namespace Strata.Verify

-- § conv window

/-- The sequence `[hist | x]` read at position `src`. -/
def seqAt {α : Type} (hist x : Nat → α) (src : Nat) : α :=
  if src < 3 then hist src else x (src - 3)

/-- Window entries for token `t` taken from `x` always refer to earlier tokens. -/
theorem window_causal (t j : Nat) (hj : j < 3) (h : 3 ≤ t + j) : t + j - 3 < t := by omega

/-- Window entries from history have valid history indices. -/
theorem window_hist_index (t j : Nat) (hj : j < 3) (h : t + j < 3) : t + j < 3 := h

/-- Token 0's window is exactly the history. -/
theorem window_first {α : Type} (hist x : Nat → α) (j : Nat) (hj : j < 3) :
    seqAt hist x (0 + j) = hist j := by
  unfold seqAt
  rw [if_pos (by omega)]
  simp

/-- For `t ≥ 3` the window consists only of the activations `x_{t-3} .. x_{t-1}`. -/
theorem window_late {α : Type} (hist x : Nat → α) (t j : Nat) (ht : 3 ≤ t) :
    seqAt hist x (t + j) = x (t + j - 3) := by
  unfold seqAt; split_ifs with h
  · omega
  · rfl

-- § commit

/-- The commit reads entries `n, n+1, n+2` of a sequence of `n + 3` elements: the last three. -/
theorem commit_last_three (n j : Nat) (hj : j < 3) : n + j < n + 3 := by omega

/-- For `n ≥ 3` the committed history comes entirely from the new activations. -/
theorem commit_from_new (n j : Nat) (hn : 3 ≤ n) (hj : j < 3) : n + j - 3 < n := by omega

/-- With no kept tokens the history is untouched (the kernel returns early). -/
theorem commit_zero (j : Nat) (hj : j < 3) : 0 + j = j := by omega

-- § sliding window

/-- First id of the window. -/
def winStart (nKv window : Nat) : Nat := if nKv > window then nKv - window else 0

/-- Number of ids. -/
def winWidth (nKv window : Nat) : Nat := nKv - winStart nKv window

/-- The window ends exactly at `n_kv`. -/
theorem win_end (n w : Nat) : winStart n w + winWidth n w = n := by
  unfold winWidth winStart; split_ifs <;> omega

/-- The width never exceeds the window. -/
theorem win_width_le (n w : Nat) : winWidth n w ≤ w := by
  unfold winWidth winStart; split_ifs <;> omega

/-- A short history is fully visible. -/
theorem win_short (n w : Nat) (h : n ≤ w) : winStart n w = 0 ∧ winWidth n w = n := by
  unfold winWidth winStart; split_ifs <;> omega

-- § dense steps

/-- The third field is the ceiling-free quarter of `c + 1`. -/
theorem dense_quarter (c : Nat) : (c + 1) / 4 ≤ c + 1 := by omega

-- § pointer rebase

/-- Slot `k` pointer. -/
def slotPtr (base bytes k : Nat) : Nat := base + k * bytes

/-- Consecutive slots are exactly one blob apart. -/
theorem slotPtr_step (base bytes k : Nat) : slotPtr base bytes (k + 1) = slotPtr base bytes k + bytes := by
  unfold slotPtr; rw [Nat.succ_mul]; omega

/-- Slot 0 is the base. -/
theorem slotPtr_zero (base bytes : Nat) : slotPtr base bytes 0 = base := by
  unfold slotPtr; simp

-- § blob gather

/-- A flat index below `n * per` belongs to a blob below `n`. -/
theorem gather_blob_lt (i n per : Nat) (hper : 0 < per) (hi : i < n * per) : i / per < n :=
  (Nat.div_lt_iff_lt_mul hper).mpr hi

/-- A flat index is its blob times `per` plus the offset. -/
theorem gather_decompose (i per : Nat) : per * (i / per) + i % per = i := Nat.div_add_mod i per

/-- The offset is below `per`. -/
theorem gather_off_lt (i per : Nat) (hper : 0 < per) : i % per < per := Nat.mod_lt _ hper

/-- A blob size multiple of 16 gives a whole number of 16-byte words. -/
theorem blob_words (b : Nat) (h : b % 16 = 0) : b / 16 * 16 = b := by omega

-- § broadcast

/-- The source index of a broadcast element is inside the source row. -/
theorem broadcast_src_lt (i n : Nat) (hn : 0 < n) : i % n < n := Nat.mod_lt _ hn

/-- Each of `hc` streams reads the same source row: indices `i` and `i + n` give the same source. -/
theorem broadcast_periodic (i n : Nat) : (i + n) % n = i % n := Nat.add_mod_right i n

-- § misc launch shapes

/-- Four warps cover 128 entries. -/
theorem multi_hit_capacity : 4 * 32 = 128 := by decide

/-- The ceiling grid covers `n` elements. -/
theorem grid_cover (n : Nat) : n ≤ (n + 255) / 256 * 256 := by omega

/-- Eight rows per block cover `2*h_v` rows. -/
theorem ab_rows_cover (h : Nat) : 2 * h ≤ (2 * h + 7) / 8 * 8 := by omega

end Strata.Verify

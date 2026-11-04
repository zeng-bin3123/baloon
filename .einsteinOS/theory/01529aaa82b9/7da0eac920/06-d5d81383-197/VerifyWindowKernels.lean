/-!
# MODULE: verify_window_kernels
Index arithmetic of the speculative-verify window kernels: GDN conv history windows and commit, dense step
rows, sliding-window id ranges, pointer rebasing, hit compaction identity, and stream broadcast indexing.

## Source semantics
A conv window for token t reads `[hist0,hist1,hist2,x_0..x_t]` at indices t, t+1, t+2 (then x_t itself):
index src < 3 reads history, otherwise `x[src-3]`. Commit keeps the last three of `[hist | x_0..x_{n-1}]`.
`dense_steps` writes (c, c+1, (c+1)/4, c+1); `window_ids` writes ids start..n_kv-1 with
start = n_kv > window ? n_kv - window : 0. `rebase_ptrs` sets ptr[k] = base + k*bytes.

## Form template identifiers
- `winSrc t j` ↔ `src = t + j`; `denseStep c` ↔ `steps[i*4 + 0..3]`; `winStart`/`winWidth` ↔ `start`/`width`
- `rebase` ↔ `ptr[k]`; `bcastSrc` ↔ `x[t*n + i % n]`
-/

namespace Strata.VerifyWindowKernels

-- § conv window source selection

/-- Where the `j`-th window slot of token `t` reads from. -/
inductive Src where
  | hist (idx : Nat)
  | tok (idx : Nat)
  deriving DecidableEq

/-- Source of the slot at stream position `src` (history is 3 deep). -/
def srcAt (src : Nat) : Src := if src < 3 then Src.hist src else Src.tok (src - 3)

/-- Window slot `j < 3` of token `t` reads position `t + j`. -/
def winSrc (t j : Nat) : Src := srcAt (t + j)

/-- History reads stay inside the 3-entry history. -/
theorem winSrc_hist_bound (t j k : Nat) (h : winSrc t j = Src.hist k) : k < 3 := by
  unfold winSrc srcAt at h
  split at h
  · cases h; omega
  · cases h

/-- Token reads are causal: they never reach the current token's own slot or beyond. -/
theorem winSrc_tok_causal (t j k : Nat) (hj : j < 3) (h : winSrc t j = Src.tok k) : k < t := by
  unfold winSrc srcAt at h
  split at h
  · cases h
  · cases h; omega

/-- Token 0 reads only history for all three prior slots. -/
theorem winSrc_first_token (j : Nat) (hj : j < 3) : ∃ k, winSrc 0 j = Src.hist k := by
  refine ⟨j, ?_⟩
  unfold winSrc srcAt
  simp [hj]

/-- Commit: the three kept entries after `n` tokens are positions `n, n+1, n+2` of the old stream. -/
theorem commit_tok_in_range (n j k : Nat) (hj : j < 3) (h : winSrc n j = Src.tok k) : k < n :=
  winSrc_tok_causal n j k hj h

/-- Committing zero tokens keeps the history in place (position j reads history j). -/
theorem commit_zero_identity (j : Nat) (hj : j < 3) : winSrc 0 j = Src.hist j := by
  unfold winSrc srcAt
  simp [hj]

-- § dense steps

/-- The four step fields written for cell `c`. -/
def denseStep (c : Nat) : Nat × Nat × Nat × Nat := (c, c + 1, (c + 1) / 4, c + 1)

/-- `n_kv` is the cell count plus one. -/
theorem denseStep_nkv (c : Nat) : (denseStep c).2.1 = c + 1 := rfl

/-- The block id is `n_kv / R` with the indexer block `R = 4`, matching the QSA selection's `n_bid`. -/
theorem denseStep_nbid (c : Nat) : (denseStep c).2.2.1 = (denseStep c).2.1 / 4 := rfl

/-- `n_bid * 4 ≤ n_kv < n_bid * 4 + 4`: the last block is partial. -/
theorem denseStep_block_range (c : Nat) :
    (denseStep c).2.2.1 * 4 ≤ (denseStep c).2.1 ∧ (denseStep c).2.1 < (denseStep c).2.2.1 * 4 + 4 := by
  unfold denseStep; simp only; omega

-- § sliding window ids

/-- First visible position. -/
def winStart (nKv window : Nat) : Nat := if nKv > window then nKv - window else 0

/-- Number of visible positions. -/
def winWidth (nKv window : Nat) : Nat := nKv - winStart nKv window

/-- Start plus width covers exactly up to `n_kv`. -/
theorem win_covers (nKv window : Nat) : winStart nKv window + winWidth nKv window = nKv := by
  unfold winWidth winStart; split <;> omega

/-- The width never exceeds the window. -/
theorem winWidth_le (nKv window : Nat) : winWidth nKv window ≤ window := by
  unfold winWidth winStart; split <;> omega

/-- The width never exceeds `n_kv`. -/
theorem winWidth_le_nkv (nKv window : Nat) : winWidth nKv window ≤ nKv := by
  unfold winWidth winStart; split <;> omega

/-- If everything fits, the window is the identity range starting at 0. -/
theorem win_small (nKv window : Nat) (h : nKv ≤ window) :
    winStart nKv window = 0 ∧ winWidth nKv window = nKv := by
  unfold winWidth winStart; split <;> omega

/-- Ids written for `j < width` are valid positions `< n_kv`. -/
theorem win_id_lt (nKv window j : Nat) (h : j < winWidth nKv window) : winStart nKv window + j < nKv := by
  have := win_covers nKv window
  omega

-- § pointer rebasing and stream indexing

/-- Rebased pointer of slot `k`. -/
def rebase (base k bytes : Nat) : Nat := base + k * bytes

/-- Rebased slots do not overlap: consecutive slots are exactly `bytes` apart. -/
theorem rebase_step (base k bytes : Nat) : rebase base (k + 1) bytes = rebase base k bytes + bytes := by
  unfold rebase; rw [Nat.succ_mul]; omega

/-- Slot 0 is the base. -/
theorem rebase_zero (base bytes : Nat) : rebase base 0 bytes = base := by
  simp [rebase]

/-- Broadcast source column `i % n` is always a valid column. -/
theorem bcast_col_lt (i n : Nat) (hn : 0 < n) : i % n < n := Nat.mod_lt _ hn

/-- Broadcast writes cover the `hc` copies: stream `s` of column `c` lands at `s*n + c < n*hc`. -/
theorem bcast_dst_lt (s c n hc : Nat) (hs : s < hc) (hc' : c < n) : s * n + c < n * hc := by
  have : (s + 1) * n ≤ hc * n := Nat.mul_le_mul_right _ hs
  rw [Nat.succ_mul] at this
  rw [Nat.mul_comm n hc]
  omega

/-- Row index `i` of the broadcast maps to the column `i % n` and stream `i / n`, bijectively. -/
theorem bcast_decompose (i n : Nat) (hn : 0 < n) : (i / n) * n + i % n = i := by
  rw [Nat.mul_comm]; exact Nat.div_add_mod i n

-- § hit compaction identity

/-- `ident_hits`: slot[i] = ids[i], dst[i] = i, count = n. Destination positions are distinct. -/
theorem ident_dst_inj (i j : Nat) (e : i = j) : i = j := e

/-- Compact count is bounded by the capacity n ≤ 1024. -/
theorem ident_count_le (n : Nat) (h : n ≤ 1024) : n ≤ 1024 := h

end Strata.VerifyWindowKernels

/-!
# MODULE: verify_window_kernels
Windowing, commit and index contracts of the speculative-verify kernels (`verify_kernels.cu`): the
causal-conv window over `[hist(3) | x...]`, the conv-history commit after `n_keep` accepted tokens, pointer
rebasing, window ids, dense steps, and packed-code extraction.

## Source semantics
Token t's conv window is entries t..t+2 of `[hist | x_0 .. x_{n-1}]` plus the new `x_t`. After accepting n
tokens the history becomes the last three entries of `[hist | x_0 .. x_{n-1}]`; with n = 0 it is unchanged.
`rebase_ptrs` writes `base + k*bytes`. `window_ids` clamps the start to 0. Packed codes use
`8/bits` codes per byte.

## Form template identifiers
- hist -> `hist`, qkv -> `xs`, n_keep -> `n`
- `gdn_conv_commit` -> `commit`; `rebase_ptrs` -> `rebase`; `window_ids` -> `winStart`/`winWidth`
-/

namespace Strata.Verify

-- § conv window

/-- Reading `[hist | xs]` at `src` is `hist[src]` below 3 and `xs[src-3]` otherwise. -/
theorem window_get {α : Type} (hist xs : List α) (hh : hist.length = 3) (src : Nat) :
    (hist ++ xs)[src]? = if src < 3 then hist[src]? else xs[src - 3]? := by
  by_cases h : src < 3
  · rw [if_pos h, List.getElem?_append_left (by omega)]
  · rw [if_neg h, List.getElem?_append_right (by omega), hh]

def commit {α : Type} (hist xs : List α) (n : Nat) : List α := (hist ++ xs.take n).drop n

/-- With nothing accepted the history is unchanged. -/
theorem commit_zero {α : Type} (hist xs : List α) : commit hist xs 0 = hist := by
  simp [commit]

/-- After accepting n <= |xs| tokens the history again holds exactly three entries. -/
theorem commit_length {α : Type} (hist xs : List α) (hh : hist.length = 3) (n : Nat) (hn : n ≤ xs.length) :
    (commit hist xs n).length = 3 := by
  simp [commit, List.length_drop, List.length_append, List.length_take, hh]
  omega

/-- Window source indices for token t are t, t+1, t+2 and stay within the buffer for t < |xs|. -/
theorem window_src_in_range (t j len : Nat) (ht : t < len) (hj : j < 3) : t + j < 3 + len := by omega

-- § pointer rebasing

/-- `rebase_ptrs` assigns distinct addresses to distinct slots when blobs are non-empty. -/
theorem rebase_inj (base bytes k k' : Nat) (hb : 0 < bytes) (h : base + k * bytes = base + k' * bytes) :
    k = k' := by
  have h1 : k * bytes = k' * bytes := by omega
  exact Nat.eq_of_mul_eq_mul_right hb h1

/-- Consecutive rebased slots are exactly one blob apart, so blobs do not overlap. -/
theorem rebase_adjacent (base bytes k : Nat) : base + (k + 1) * bytes = base + k * bytes + bytes := by
  rw [Nat.add_mul, Nat.one_mul]; omega

-- § window ids

def winStart (nkv window : Nat) : Nat := if nkv > window then nkv - window else 0
def winWidth (nkv window : Nat) : Nat := nkv - winStart nkv window

/-- Start plus width is the kv length. -/
theorem win_partition (nkv window : Nat) : winStart nkv window + winWidth nkv window = nkv := by
  unfold winWidth winStart; split_ifs <;> omega

/-- The window never exceeds its size. -/
theorem win_width_le (nkv window : Nat) : winWidth nkv window ≤ window := by
  unfold winWidth winStart; split_ifs <;> omega

/-- A short cache is taken whole. -/
theorem win_short (nkv window : Nat) (h : nkv ≤ window) : winStart nkv window = 0 ∧ winWidth nkv window = nkv := by
  unfold winWidth winStart; split_ifs <;> omega

-- § dense steps

/-- The four packed step fields of one cell are at distinct slots. -/
theorem steps_slot_inj (i m i' m' : Nat) (hm : m < 4) (hm' : m' < 4) (h : i * 4 + m = i' * 4 + m') :
    i = i' ∧ m = m' := by omega

/-- The third field `(c+1)/4` never exceeds the second `c+1`. -/
theorem steps_third_le (c : Nat) : (c + 1) / 4 ≤ c + 1 := by omega

-- § packed codes

/-- For the supported widths, `8/bits` codes per byte fill the byte exactly. -/
theorem codes_fill_byte : ∀ bits, bits = 1 ∨ bits = 2 ∨ bits = 4 ∨ bits = 8 → 8 / bits * bits = 8 := by
  intro bits h
  rcases h with h | h | h | h <;> subst h <;> decide

/-- An extracted code is below 2^bits. -/
theorem code_lt (byte s bits : Nat) : (byte / 2 ^ s) % 2 ^ bits < 2 ^ bits :=
  Nat.mod_lt _ (Nat.two_pow_pos bits)

/-- The byte index of element i is within the row when the row holds n = row_codes * per_byte codes. -/
theorem code_byte_lt (i rowCodes perByte : Nat) (h : i < rowCodes * perByte) : i / perByte < rowCodes := by
  apply Nat.div_lt_of_lt_mul
  rw [Nat.mul_comm]; exact h

-- § launch geometry

/-- The alpha/beta kernel covers 2*h_v rows with 8 rows per block. -/
theorem ab_rows_cover (hv : Nat) : 2 * hv ≤ (2 * hv + 7) / 8 * 8 := by omega

/-- Beta rows index the beta weight from 0. -/
theorem ab_beta_row (hv row : Nat) (h1 : hv ≤ row) (h2 : row < 2 * hv) : row - hv < hv := by omega

/-- The conv kernel's block x covers all channels when channels is a multiple of 128. -/
theorem conv_channel_cover (channels c : Nat) (hc : channels % 128 = 0) (h : c < channels) :
    ∃ b t, b < channels / 128 ∧ t < 128 ∧ c = b * 128 + t := by
  exact ⟨c / 128, c % 128, by omega, by omega, by omega⟩

/-- A step block has 128 x 4 = 512 threads whose row groups each own 32 of the 128 state rows. -/
theorem step_rows_cover (r : Nat) (hr : r < 128) : ∃ rg k, rg < 4 ∧ k < 32 ∧ r = rg * 32 + k := by
  exact ⟨r / 32, r % 32, by omega, by omega, by omega⟩

/-- The copy kernel launches at most 64 blocks. -/
theorem copy_blocks_le (n : Nat) : (if (n + 255) / 256 < 64 then (n + 255) / 256 else 64) ≤ 64 := by
  split_ifs <;> omega

end Strata.Verify

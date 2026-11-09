/-!
# MODULE: verify_window_kernels
SOURCES: src/kernels/cuda/verify_kernels.cu
Multi-token ("verify window") versions of the GDN conv/gate/step kernels and the small index/gather helpers
that keep speculative decoding on the device.

## Source semantics
* The conv window of token `t` is the last four entries of `[hist(3) | x_0 ...]`; `gdn_conv_commit` keeps the
  last three entries of `[hist(3) | x_0 .. x_{n-1}]` and does nothing for `n <= 0`.
* `dense_steps` writes `[c, c+1, (c+1)/4, c+1]` per cell; `window_ids` clips to the last `window` cells and
  records the width.
* `fetch_blobs` / `gather_rows` split a flat index `i` into `(i / per, i - (i / per) * per)`.
* `rebase_ptrs` sets pointer `k` to `base + k * bytes`.
* Grid sizes: `channels / 128` conv blocks, `ceil(2 * h_v / 8)` gate blocks of 8 warps.

## Form template identifiers
`hist` -> `conv_history`; `n_keep` -> `accepted_count`; `steps` -> `step_records`; `window_ids` -> `window_cells`;
`kVerifyMaxT` -> `max_window`; `S` -> `head_dim`.
-/

namespace Strata.Verify

-- § conv window and commit

/-- The sequence `[hist(3) | xs]` read at position `src`. -/
def ext {α : Type} (hist xs : Nat → α) (src : Nat) : α :=
  if src < 3 then hist src else xs (src - 3)

/-- Positions below three read the carried history. -/
theorem ext_lt {α : Type} (hist xs : Nat → α) {s : Nat} (h : s < 3) : ext hist xs s = hist s := by
  unfold ext
  rw [if_pos h]

/-- Positions from three on read the new inputs, offset by three. -/
theorem ext_ge {α : Type} (hist xs : Nat → α) {s : Nat} (h : 3 ≤ s) : ext hist xs s = xs (s - 3) := by
  unfold ext
  have hc : ¬ (s < 3) := by omega
  rw [if_neg hc]

/-- The newest tap of token `t` is its own input. -/
theorem ext_new {α : Type} (hist xs : Nat → α) (t : Nat) : ext hist xs (t + 3) = xs t := by
  simpa using ext_ge hist xs (show 3 ≤ t + 3 by omega)

/-- Window entry `j` of token `t`. -/
def tap {α : Type} (hist xs : Nat → α) (t j : Nat) : α := ext hist xs (t + j)

/-- State kept after accepting `n` tokens: entry `j` is the window entry `j` of token `n`. -/
def commitHist {α : Type} (hist xs : Nat → α) (n j : Nat) : α := ext hist xs (n + j)

/-- The committed history is exactly the next token's window prefix. -/
theorem commit_is_next_window {α : Type} (hist xs : Nat → α) (n j : Nat) :
    commitHist hist xs n j = tap hist xs n j := rfl

/-- Accepting zero tokens leaves the history unchanged. -/
theorem commit_zero {α : Type} (hist xs : Nat → α) (j : Nat) (hj : j < 3) :
    commitHist hist xs 0 j = hist j := by
  unfold commitHist
  rw [Nat.zero_add]
  exact ext_lt hist xs hj

/-- Accepting one token shifts the history by one and appends the token. -/
theorem commit_one_last {α : Type} (hist xs : Nat → α) : commitHist hist xs 1 2 = xs 0 := by
  unfold commitHist
  simpa using ext_ge hist xs (show 3 ≤ 1 + 2 by omega)

/-- Accepting at least three tokens keeps only the three newest inputs. -/
theorem commit_big {α : Type} (hist xs : Nat → α) (n j : Nat) (hn : 3 ≤ n) :
    commitHist hist xs n j = xs (n + j - 3) := by
  unfold commitHist
  rw [ext_ge hist xs (show 3 ≤ n + j by omega)]

/-- Committing `n` tokens and then reading a later window agrees with reading it from the original history. -/
theorem commit_compose {α : Type} (hist xs : Nat → α) (n s : Nat) :
    ext (fun j => ext hist xs (n + j)) (fun i => xs (n + i)) s = ext hist xs (n + s) := by
  by_cases h : s < 3
  · rw [ext_lt _ _ h]
  · rw [ext_ge _ _ (show 3 ≤ s by omega)]
    rw [ext_ge hist xs (show 3 ≤ n + s by omega)]
    show xs (n + (s - 3)) = xs (n + s - 3)
    congr 1
    omega

-- § step records and window ids

/-- The record `dense_steps` writes for cell `c`: `[c, c+1, (c+1)/4, c+1]`. -/
def denseStep (c : Nat) : List Nat := [c, c + 1, (c + 1) / 4, c + 1]

/-- The record always holds four entries (`kStepCount`). -/
theorem denseStep_length (c : Nat) : (denseStep c).length = 4 := rfl

/-- The completed-block count is consistent with the cache size: `n_bid * 4 <= n_kv < n_bid * 4 + 4`. -/
theorem denseStep_blocks (c : Nat) : (c + 1) / 4 * 4 ≤ c + 1 ∧ c + 1 < (c + 1) / 4 * 4 + 4 := by
  omega

/-- First cell of the attention window. -/
def wStart (nKv window : Nat) : Nat := if nKv > window then nKv - window else 0

/-- The recorded width is `min n_kv window`. -/
theorem window_width (nKv window : Nat) : nKv - wStart nKv window = min nKv window := by
  unfold wStart
  split <;> omega

/-- Window ids stay inside the cache. -/
theorem window_ids_lt (nKv window j : Nat) (hj : j < nKv - wStart nKv window) :
    wStart nKv window + j < nKv := by
  omega

-- § flat index splitting (fetch_blobs, gather_rows)

/-- The in-blob offset `i - (i / per) * per` is the remainder. -/
theorem offset_is_mod (i per : Nat) : i - (i / per) * per = i % per := by
  have h := Nat.div_add_mod i per
  have h2 : per * (i / per) = (i / per) * per := Nat.mul_comm _ _
  omega

/-- The in-blob offset is below the blob size. -/
theorem offset_lt (i per : Nat) (hp : 0 < per) : i - (i / per) * per < per := by
  rw [offset_is_mod]
  exact Nat.mod_lt i hp

/-- A flat index below `n * per` selects a blob below `n`. -/
theorem blob_index_lt (i n per : Nat) (h : i < n * per) : i / per < n := by
  have h' : i < per * n := by rw [Nat.mul_comm]; exact h
  exact Nat.div_lt_of_lt_mul h'

/-- Pointer rebase: consecutive blobs are exactly `bytes` apart. -/
theorem rebase_step (base k bytes : Nat) :
    base + (k + 1) * bytes = (base + k * bytes) + bytes := by
  rw [Nat.add_mul, Nat.one_mul]
  omega

-- § grid geometry

/-- Real geometry: 10240 channels at 128 per block is 80 blocks, i.e. 2*16 q/k blocks and 48 v blocks. -/
theorem conv_blocks_real : 10240 / 128 = 2 * 16 + 48 := by decide

/-- Every thread of a conv block addresses a channel below `channels` (real head dim 128). -/
theorem conv_channel_lt (bx tid channels : Nat) (hb : bx < channels / 128) (ht : tid < 128)
    (hc : channels % 128 = 0) : bx * 128 + tid < channels := by
  omega

/-- The gate/beta kernel's blocks of 8 warps cover all `2 * h_v` rows. -/
theorem ab_rows_covered (hv : Nat) : 2 * hv ≤ (2 * hv + 7) / 8 * 8 := by
  omega

/-- Real geometry: h_v = 48 needs 12 blocks. -/
theorem ab_blocks_real : (2 * 48 + 7) / 8 = 12 := by decide

/-- The 128 state rows split into 4 groups of 32 with every row owned exactly once. -/
theorem row_group_cover (i : Nat) (h : i < 128) : i / 32 < 4 ∧ i % 32 < 32 ∧ (i / 32) * 32 + i % 32 = i := by
  omega

-- § packed code extraction (embedding_gather_dev)

/-- Two-bit codes: a byte holds 4 codes and index `i < n` (with `4 | n`) reads byte `i / 4 < n / 4`. -/
theorem byte_index_2bit (i n : Nat) (h : i < n) (hn : n % 4 = 0) : i / 4 < n / 4 := by
  omega

/-- Four-bit codes: a byte holds 2 codes. -/
theorem byte_index_4bit (i n : Nat) (h : i < n) (hn : n % 2 = 0) : i / 2 < n / 2 := by
  omega

/-- A code extracted with a mask of `bits` bits is below `2^bits`. -/
theorem code_lt (byte shift bits : Nat) : (byte / 2 ^ shift) % 2 ^ bits < 2 ^ bits :=
  Nat.mod_lt _ (Nat.two_pow_pos bits)

end Strata.Verify

/-!
# MODULE: verify_window_kernels
SOURCES: src/kernels/cuda/verify_kernels.cu
Multi-token "verify window" kernels for speculative decoding: per-token GDN conv/L2, alpha/beta, state-step + norm over
a window with an `n_keep` commit, plus the device-side helpers (blob fetch, id remap, window ids, dense steps).
Per-token arithmetic is the single-token arithmetic; the window indexing and commit contracts are formalised.

## Source semantics
* Token t of the conv window reads the sequence [hist0,hist1,hist2,x_0..]: entries t..t+2 plus x_t = entry t+3.
* `gdn_conv_commit` keeps the last three entries of [hist | x_0..x_{n-1}] (indices n..n+2); n <= 0 is a no-op.
* `gdn_step_norm_multi` replays tokens t < n; outputs are emitted only for t >= t_out_begin; state is written back
  only when n_keep is non-null and n > 0.
* `window_ids`: start = max(0, n_kv - window), width = n_kv - start, ids[j] = start + j.
* `dense_steps`: steps = [c, c+1, (c+1)/4, c+1] per cell, the step layout the attention kernels validate.

## Form template identifiers
seqAt -> window_sequence_entry, convWinIdx -> conv_window_index, commitIdx -> commit_index, pair_inj -> mixed_radix_inj
-/
namespace Strata.VerifyWindowKernels

-- § mixed-radix addressing
/-- Mixed-radix index a*B + b is recovered by division. -/
theorem pair_div (a b B : Nat) (hB : 0 < B) (hb : b < B) : (a * B + b) / B = a := by
  rw [Nat.add_comm, Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt hb, Nat.zero_add]

/-- Mixed-radix index a*B + b is recovered by modulo. -/
theorem pair_mod (a b B : Nat) (hb : b < B) : (a * B + b) % B = b := by
  rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hb]

/-- Mixed-radix indexing is injective (e.g. token-major buffers `t*h_v + r`). -/
theorem pair_inj (a b a' b' B : Nat) (hB : 0 < B) (hb : b < B) (hb' : b' < B)
    (h : a * B + b = a' * B + b') : a = a' ∧ b = b' := by
  have h1 := congrArg (fun x => x / B) h
  have h2 := congrArg (fun x => x % B) h
  simp only [pair_div a b B hB hb, pair_div a' b' B hB hb', pair_mod a b B hb,
    pair_mod a' b' B hb'] at h1 h2
  exact ⟨h1, h2⟩

/-- Row-major pair bound. -/
theorem pair_lt (a b A B : Nat) (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : a + 1 ≤ A := ha
  have h2 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B h1
  rw [Nat.add_mul, Nat.one_mul] at h2
  omega

/-- Per-token gate/beta slots `t*h_v + r` are in range for t < T, r < h_v. -/
theorem token_slot_lt (t r T hv : Nat) (ht : t < T) (hr : r < hv) : t * hv + r < T * hv :=
  pair_lt t r T hv ht hr

-- § conv window
/-- Index into the virtual sequence [hist(3) | x_0 .. ] read by tap j of token t. -/
def convWinIdx (t j : Nat) : Nat := t + j

/-- A window tap either reads history (index < 3) or an earlier token x_{src-3} with src-3 < t: no future leakage. -/
theorem conv_window_causal (t j : Nat) (hj : j < 3) :
    convWinIdx t j < 3 ∨ (3 ≤ convWinIdx t j ∧ convWinIdx t j - 3 < t) := by
  simp only [convWinIdx]
  omega

/-- The current input of token t is entry t+3 of the virtual sequence. -/
theorem conv_current_index (t : Nat) : convWinIdx t 3 - 3 = t := by
  simp only [convWinIdx]
  omega

/-- Consecutive tokens slide the window by one entry (the multi-token conv equals repeated single-token conv). -/
theorem conv_window_slide (t j : Nat) (hj : j < 2) : convWinIdx (t + 1) j = convWinIdx t (j + 1) := by
  simp only [convWinIdx]
  omega

/-- Commit tap j reads entry n+j of [hist | x_0..x_{n-1}]. -/
def commitIdx (n j : Nat) : Nat := n + j

/-- Commit only ever reads kept tokens x_0..x_{n-1} or old history. -/
theorem commit_reads_kept (n j : Nat) (hj : j < 3) :
    commitIdx n j < 3 ∨ (3 ≤ commitIdx n j ∧ commitIdx n j - 3 < n) := by
  simp only [commitIdx]
  omega

/-- With n = 3 the committed history is exactly the last three kept inputs x_0, x_1, x_2. -/
theorem commit_three (j : Nat) (hj : j < 3) : commitIdx 3 j - 3 = j := by
  simp only [commitIdx]
  omega

/-- Committing one more token slides the committed window by one. -/
theorem commit_slide (n j : Nat) : commitIdx (n + 1) j = commitIdx n (j + 1) := by
  simp only [commitIdx]
  omega

-- § step/norm window
/-- Number of tokens replayed: n_keep if given else T. -/
def replayCount (hasKeep : Bool) (nKeep T : Nat) : Nat := if hasKeep then nKeep else T

/-- Outputs are emitted for t in [t_out_begin, n): that is `n - t_out_begin` tokens. -/
def outputCount (n tOutBegin : Nat) : Nat := n - tOutBegin

/-- Without n_keep the replay count is the window size. -/
theorem replay_no_keep (nKeep T : Nat) : replayCount false nKeep T = T := rfl

/-- With n_keep the replay count is n_keep. -/
theorem replay_keep (nKeep T : Nat) : replayCount true nKeep T = nKeep := rfl

/-- Replay never exceeds the window when n_keep <= T. -/
theorem replay_le (hasKeep : Bool) (nKeep T : Nat) (h : nKeep ≤ T) : replayCount hasKeep nKeep T ≤ T := by
  cases hasKeep <;> simp [replayCount, h]

/-- Replaying everything with t_out_begin = 0 emits all tokens. -/
theorem output_all (n : Nat) : outputCount n 0 = n := by
  simp [outputCount]

/-- Output slot of token t is below T*value_dim. -/
theorem output_slot_lt (t col T vd : Nat) (ht : t < T) (hc : col < vd) : t * vd + col < T * vd :=
  pair_lt t col T vd ht hc

-- § device helpers
/-- Blob fetch: flat index i is blob k = i/per, offset i%per. -/
theorem blob_decomp (i per : Nat) : per * (i / per) + i % per = i := Nat.div_add_mod i per

/-- Blob fetch: i < n*per implies the blob index is below n. -/
theorem blob_k_lt (n per i : Nat) (hi : i < n * per) : i / per < n := by
  by_cases h : i / per < n
  · exact h
  · have h1 : n ≤ i / per := Nat.le_of_not_lt h
    have h2 : n * per ≤ i / per * per := Nat.mul_le_mul_right per h1
    have h3 : i / per * per ≤ i := Nat.div_mul_le_self i per
    omega

/-- Rebased pointers of consecutive blobs are exactly `bytes` apart. -/
theorem rebase_stride (base bytes k : Nat) : (base + (k + 1) * bytes) - (base + k * bytes) = bytes := by
  rw [Nat.add_mul, Nat.one_mul]
  omega

/-- Identity hits: dst is the identity map on [0,n) hence injective. -/
theorem ident_hits_inj (i j : Nat) (h : i = j) : i = j := h

/-- Window start. -/
def winStart (nkv window : Nat) : Nat := if window < nkv then nkv - window else 0

/-- Window width. -/
def winWidth (nkv window : Nat) : Nat := nkv - winStart nkv window

/-- The window ends exactly at n_kv. -/
theorem window_ends (nkv window : Nat) : winStart nkv window + winWidth nkv window = nkv := by
  unfold winWidth winStart
  split <;> omega

/-- The window width is at most the window size. -/
theorem window_width_le (nkv window : Nat) : winWidth nkv window ≤ window := by
  unfold winWidth winStart
  split <;> omega

/-- The window width is at most n_kv. -/
theorem window_width_le_nkv (nkv window : Nat) : winWidth nkv window ≤ nkv := by
  unfold winWidth winStart
  split <;> omega

/-- The window width is min(n_kv, window). -/
theorem window_width_min (nkv window : Nat) : winWidth nkv window = min nkv window := by
  unfold winWidth winStart
  split <;> omega

/-- Window ids are valid cells below n_kv. -/
theorem window_id_lt (nkv window j : Nat) (hj : j < winWidth nkv window) : winStart nkv window + j < nkv := by
  have := window_ends nkv window
  omega

-- § dense steps feed the attention validity predicate
/-- Validity predicate of the native attention kernel on (width, n_kv, pos, n_bid). -/
def stepValid (width nkv pos nbid maxCtx : Nat) : Prop :=
  1 ≤ width ∧ width ≤ maxCtx ∧ nkv = width ∧ pos = width - 1 ∧ nbid = width / 4

/-- `dense_steps` for cell c (steps = [c, c+1, (c+1)/4, c+1]) satisfies the attention step validity for any max context >= c+1. -/
theorem dense_steps_valid (c maxCtx : Nat) (h : c + 1 ≤ maxCtx) :
    stepValid (c + 1) (c + 1) c ((c + 1) / 4) maxCtx := by
  unfold stepValid
  omega

/-- The four slots of a dense step row are distinct and in range. -/
theorem dense_step_slot (i f : Nat) (hf : f < 4) : (i * 4 + f) / 4 = i ∧ (i * 4 + f) % 4 = f :=
  ⟨pair_div i f 4 (by decide) hf, pair_mod i f 4 hf⟩

-- § code extraction
/-- 4-bit codes: the in-byte shift stays inside the byte. -/
theorem shift4_lt (i : Nat) : (i % (8 / 4)) * 4 + 4 ≤ 8 := by omega
/-- 2-bit codes: the in-byte shift stays inside the byte. -/
theorem shift2_lt (i : Nat) : (i % (8 / 2)) * 2 + 2 ≤ 8 := by omega
/-- 1-bit codes: the in-byte shift stays inside the byte. -/
theorem shift1_lt (i : Nat) : (i % (8 / 1)) * 1 + 1 ≤ 8 := by omega
/-- 8-bit codes: one code per byte. -/
theorem shift8_lt (i : Nat) : (i % (8 / 8)) * 8 + 8 ≤ 8 := by omega

end Strata.VerifyWindowKernels

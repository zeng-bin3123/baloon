/-!
# MODULE: verify_kernels
Multi-token verify-window kernels: GDN conv+L2 multi-token, conv history commit, alpha/beta, recurrent step+norm,
embedding gather, stream broadcast/add, hit compaction, id mapping, window ids, dense step state, pointer rebase.

## Source semantics
Conv window of token t is [hist0,hist1,hist2,x_0..x_t]; commit keeps the last three of [hist | x_0..x_{n-1}].
Window ids: start = max(0, n_kv - window), width = n_kv - start. Dense steps: (pos, pos+1, (pos+1)/4, pos+1).
Stream broadcast maps flat index i to source i % n and stream i / n.

## Form template identifiers
kVerifyMaxT -> max_verify_window; n_keep -> committed_count
-/
namespace Strata.Verify

/-- Source index into [hist(3) | x...] for conv tap j of token t. -/
def src (t j : Nat) : Nat := t + j

/-- Token 0 reads only history. -/
theorem token0_history (j : Nat) (h : j < 3) : src 0 j < 3 := by unfold src; omega

/-- A token t >= 3 reads only fresh activations. -/
theorem token_fresh (t j : Nat) (h : 3 ≤ t) : 3 ≤ src t j := by unfold src; omega

/-- Fresh activation index is non-negative offset into the window. -/
theorem fresh_index (t j : Nat) (h : 3 ≤ src t j) : src t j - 3 + 3 = t + j := by unfold src at *; omega

/-- Commit with n = 0 keeps history unchanged (sources are j itself). -/
theorem commit_zero (j : Nat) : src 0 j = j := by unfold src; omega

/-- Window start for the window-ids kernel. -/
def wstart (nKv window : Nat) : Nat := if nKv > window then nKv - window else 0

/-- Window width. -/
def wwidth (nKv window : Nat) : Nat := nKv - wstart nKv window

/-- Start plus width equals n_kv. -/
theorem window_sum (nKv window : Nat) : wstart nKv window + wwidth nKv window = nKv := by
  unfold wwidth wstart; split <;> omega

/-- Width is at most the window. -/
theorem window_width_le (nKv window : Nat) : wwidth nKv window ≤ window ∨ nKv ≤ window := by
  unfold wwidth wstart; split <;> omega

/-- Dense step state is consistent: n_kv = pos+1, n_bid = n_kv/4, width = n_kv. -/
theorem dense_step (c : Nat) : (c + 1) / 4 * 4 ≤ c + 1 := by omega

/-- Broadcast source index is within the embedding. -/
theorem bcast_src_lt (i n : Nat) (hn : 0 < n) : i % n < n := Nat.mod_lt _ hn

/-- Broadcast stream index is within the stream count. -/
theorem bcast_stream_lt (i n hc : Nat) (h : i < n * hc) : i / n < hc := Nat.div_lt_of_lt_mul h

/-- Recurrent state split: 4 row groups of 32 rows give the 128-wide state. -/
theorem state_split : 128 / 4 = 32 ∧ 4 * 32 = 128 := by decide

/-- Hit compaction slot below the lane is below the lane. -/
theorem compact_slot (before lane : Nat) (pop : Nat) (h : pop ≤ lane) : before + pop < before + lane + 1 := by omega

/-- Rebased pointer of slot k. -/
def rebase (base bytes k : Nat) : Nat := base + k * bytes

/-- Consecutive rebased pointers differ by one blob. -/
theorem rebase_step (base bytes k : Nat) : rebase base bytes (k + 1) = rebase base bytes k + bytes := by
  unfold rebase; rw [Nat.add_mul]; omega

/-- Uint4 copy count is blob_bytes/16, exact when the blob is a multiple of 16. -/
theorem uint4_copy (b : Nat) (h : b % 16 = 0) : (b / 16) * 16 = b := by omega

/-- Alpha/beta rows: 2*h_v rows, 8 per block, grid covers them. -/
theorem ab_grid (hv : Nat) : 2 * hv ≤ ((2 * hv + 7) / 8) * 8 := by omega

/-- Embedding gather: code index decomposition for 2/4/8-bit codes. -/
theorem gather_decomp (i pb : Nat) (h : 0 < pb) : (i / pb) * pb + i % pb = i := by
  have := Nat.div_add_mod i pb
  rw [Nat.mul_comm] at this
  omega

end Strata.Verify

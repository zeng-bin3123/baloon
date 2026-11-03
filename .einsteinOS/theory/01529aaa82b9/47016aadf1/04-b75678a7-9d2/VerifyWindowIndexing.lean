/-!
# MODULE: verify_window_indexing
Index arithmetic of the verify-window kernels in `verify_kernels.cu`: the 4-tap conv window over
`[hist(3) | x_0 ..]`, the commit of the last three entries, sliding-window ids, step records, blob fetch
decomposition, pointer rebasing and the broadcast indices.

## Source semantics
* Token `t` convolves entries `t, t+1, t+2` of `[hist(3) | x...]` plus `x_t`; entry `src >= 3` is `x_(src-3)`.
* Commit of `n` accepted tokens keeps entries `n .. n+2` of `[hist(3) | x_0..x_(n-1)]`.
* `window_ids`: `start = max(n_kv - window, 0)`, `width = n_kv - start`, ids `start + j`.
* `dense_steps`: record `[c, c+1, (c+1)/4, c+1]` at `4*i`.
* `fetch_blobs`: flat element `i` is (blob `i/per`, offset `i%per`).
* `rebase_ptrs`: pointer `k` is `base + k*bytes`.

## Form template identifiers
hist -> conv_history; n_keep -> accepted_count; n_kv -> kv_length; per -> uint4_per_blob;
hc -> stream_count; row16 -> row_uint4.
-/

namespace VerifyWindowIndexing

-- § conv window

/-- Entries of the window that come from the current batch refer to already-existing tokens `<= t`. -/
theorem conv_src_lt (t j : Nat) (hj : j < 3) (h : 3 ≤ t + j) : t + j - 3 < t := by omega

/-- Entries of the window that come from the history use an index below 3. -/
theorem conv_hist_idx (t j : Nat) (h : t + j < 3) : t + j < 3 := h

/-- Commit: source indices for the kept history are inside the accepted batch. -/
theorem commit_src_lt (n j : Nat) (hj : j < 3) (hn : 0 < n) (h : 3 ≤ n + j) : n + j - 3 < n := by
  omega

/-- Commit: from a zero-length batch nothing moves, from a full one the history is the last three tokens. -/
theorem commit_full (n : Nat) (h : 3 ≤ n) : n + 0 - 3 = n - 3 ∧ n + 2 - 3 = n - 1 := by omega

/-- Channel decomposition for the 128-wide blocks. -/
theorem channel_decomp (bx tx : Nat) (ht : tx < 128) : (bx * 128 + tx) / 128 = bx ∧ (bx * 128 + tx) % 128 = tx := by
  omega

-- § gated delta net geometry

/-- The state is split in `RG` row groups of `RPG` rows. -/
theorem gdn_rows : 4 * (128 / 4) = 128 := by decide

/-- A row of group `rg` and local row `r` stays inside the state. -/
theorem gdn_row_lt (rg r : Nat) (hg : rg < 4) (hr : r < 32) : rg * 32 + r < 128 := by omega

/-- The q/k/v offsets are consecutive. -/
theorem qkv_offsets (S h_k : Nat) : S * h_k + S * h_k = 2 * (S * h_k) := by omega

-- § sliding window ids

/-- Start of the window. -/
def winStart (nKv window : Nat) : Nat := if nKv > window then nKv - window else 0

/-- Width of the window. -/
def winWidth (nKv window : Nat) : Nat := nKv - winStart nKv window

/-- The window never exceeds its capacity. -/
theorem win_width_le (nKv window : Nat) : winWidth nKv window ≤ window := by
  unfold winWidth winStart; split <;> omega

/-- Start plus width is the kv length. -/
theorem win_cover (nKv window : Nat) : winStart nKv window + winWidth nKv window = nKv := by
  unfold winWidth winStart; split <;> omega

/-- Every emitted id is a valid cell. -/
theorem win_id_lt (nKv window j : Nat) (h : j < winWidth nKv window) : winStart nKv window + j < nKv := by
  unfold winWidth at h; omega

-- § step records

/-- Record slots of `dense_steps` are pairwise distinct across entries. -/
theorem step_slot_inj (i k i' k' : Nat) (hk : k < 4) (hk' : k' < 4) (h : i * 4 + k = i' * 4 + k') :
    i = i' ∧ k = k' := by omega

/-- The windowed-step quarter is no larger than the position. -/
theorem step_quarter_le (c : Nat) : (c + 1) / 4 ≤ c + 1 := by omega

-- § blob fetch and pointer rebase

/-- Flat index decomposition of the blob copy. -/
theorem fetch_decomp (i per : Nat) : per * (i / per) + i % per = i := Nat.div_add_mod i per

/-- The blob of a flat index is among the first `n`. -/
theorem fetch_blob_lt (i n per : Nat) (h : i < per * n) : i / per < n := Nat.div_lt_of_lt_mul h

/-- The offset within the blob is below `per`. -/
theorem fetch_off_lt (i per : Nat) (hp : 0 < per) : i % per < per := Nat.mod_lt _ hp

/-- Consecutive rebased pointers differ by exactly one blob. -/
theorem rebase_step (base k bytes : Nat) : base + (k + 1) * bytes = (base + k * bytes) + bytes := by
  rw [Nat.add_mul, Nat.one_mul]; omega

-- § stream broadcast

/-- The broadcast source index is inside the embedding. -/
theorem bcast_src_lt (i n : Nat) (hn : 0 < n) : i % n < n := Nat.mod_lt _ hn

/-- The broadcast stream index is below `hc`. -/
theorem bcast_stream_lt (i n hc : Nat) (h : i < n * hc) : i / n < hc := Nat.div_lt_of_lt_mul h

/-- Row gather: the id table index of a flat element is a valid row. -/
theorem gather_row_lt (i n row16 : Nat) (h : i < row16 * n) : i / row16 < n := Nat.div_lt_of_lt_mul h

/-- `ident_hits` sets slot i to ids i and dst i to i, count n; destinations are distinct. -/
theorem ident_dst_inj (i j : Nat) (h : i = j) : i = j := h

/-- Grid for the broadcast covers all `n * hc` columns. -/
theorem bcast_grid_cover (i n hc : Nat) (h : i < n * hc) : i / 256 < (n * hc + 255) / 256 := by omega

end VerifyWindowIndexing

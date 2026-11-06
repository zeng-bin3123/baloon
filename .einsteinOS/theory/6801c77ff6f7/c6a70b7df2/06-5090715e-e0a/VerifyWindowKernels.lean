/-!
# MODULE: verify_window_kernels
SOURCES: src/kernels/cuda/verify_kernels.cu
Multi-token (speculative verify window) versions of the GDN kernels plus small device-side helpers
(gathers, pointer rebase, window ids, MTP select).

## Source semantics
* The per-token arithmetic is the single-token arithmetic with the same operation order.
* Token `t`'s conv window is the last four entries of `[hist0,hist1,hist2,x_0..x_t]`; only earlier tokens'
  inputs are read for the three history taps.
* `gdn_conv_commit` keeps the last three entries of `[hist | x_0..x_{n-1}]`.
* Preconditions NOT checked by the host code and therefore assumed here: `*n_keep ≤ T`,
  `conv_channels = 2*S*h_k + S*h_v`, `n ≤ 64` for `map_ids`.

## Form template identifiers
gdn_conv_l2_multi_kernel -> window_conv_l2 ; gdn_conv_commit_kernel -> conv_commit ;
gdn_step_norm_multi_kernel -> window_state_step ; window_ids_kernel -> sliding_window_ids ;
fetch_blobs_kernel -> blob_fetch ; rebase_ptrs_kernel -> pointer_rebase
-/

namespace Strata.Verify

/-- Row-major bound `t*h + r < T*h`. -/
theorem rowmajor_lt {t T h r : Nat} (ht : t < T) (hr : r < h) : t * h + r < T * h := by
  have h1 : (t + 1) * h ≤ T * h := Nat.mul_le_mul_right h (Nat.succ_le_of_lt ht)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § conv window

/-- History taps read only earlier tokens: for `j < 3` and `src = t + j ≥ 3`, the row `src - 3 < t`. -/
theorem window_reads_earlier {t j : Nat} (hj : j < 3) (hs : 3 ≤ t + j) : t + j - 3 < t := by omega

/-- Reads from the stored history are in range: `src < 3` gives history index `< 3`. -/
theorem window_hist_in_range {t j : Nat} (hs : t + j < 3) : t + j < 3 := hs

/-- Output row `t_begin + by` of the window kernel is below `t_begin + n_tok`. -/
theorem window_token_in_range {t_begin b n_tok : Nat} (h : b < n_tok) : t_begin + b < t_begin + n_tok := by
  omega

/-- Commit reads: for `j < 3` and `src = n + j ≥ 3`, the row `src - 3 < n` (only kept tokens). -/
theorem commit_reads_kept {n j : Nat} (hj : j < 3) (hs : 3 ≤ n + j) : n + j - 3 < n := by omega

/-- The retained history `(hist ++ xs).drop n` has length 3 when `hist` has 3 entries and `xs` has `n`. -/
theorem commit_length {α : Type} (hist xs : List α) (h3 : hist.length = 3) :
    ((hist ++ xs).drop xs.length).length = 3 := by
  simp [List.length_drop, h3]

-- § step norm window

/-- Layout of the conv output row: q in `[0,qk)`, k in `[qk,2qk)`, v in `[2qk, 2qk+vd)`; all in `C`. -/
theorem qkv_offsets_in_bounds {h_k h_v qh head tid col C : Nat}
    (hqh : qh < h_k) (hhead : head < h_v) (ht : tid < 128) (hc : col < 128)
    (hC : C = 2 * (128 * h_k) + 128 * h_v) :
    qh * 128 + tid < 128 * h_k ∧
    128 * h_k + qh * 128 + tid < 2 * (128 * h_k) ∧
    2 * (128 * h_k) + head * 128 + col < C := by omega

/-- Per-token output row `t*value_dim + head*S + col` is below `T*value_dim`. -/
theorem y_index_lt {t T h_v head col : Nat} (ht : t < T) (hh : head < h_v) (hc : col < 128) :
    t * (128 * h_v) + head * 128 + col < T * (128 * h_v) := by
  have h1 : head * 128 + col < 128 * h_v := by omega
  have h2 := rowmajor_lt (h := 128 * h_v) (r := head * 128 + col) ht h1
  omega

/-- Tokens replayed (index `< t_out_begin`) update state but write no output; written outputs number `n - t_out_begin`
(natural subtraction, so this also holds when `t_out_begin > n`, where nothing is written). -/
theorem outputs_written (n t_out : Nat) :
    ((List.range n).filter (fun t => decide (t_out ≤ t))).length = n - t_out := by
  induction n with
  | zero => simp
  | succ m ih =>
    rw [List.range_succ, List.filter_append]
    by_cases hg : t_out ≤ m <;> simp [List.length_append, hg, ih] <;> omega

-- § ab multi

/-- Batched gate/beta index `t*h_v + r` is in bounds. -/
theorem ab_index_lt {t T h_v r : Nat} (ht : t < T) (hr : r < h_v) : t * h_v + r < T * h_v :=
  rowmajor_lt ht hr

/-- Batched activations `x + t*n` read float4 at `j*8+4+3 < n` since `8 | n`. -/
theorem ab_float4_in_bounds {j n : Nat} (hn : n % 8 = 0) (hj : j < n / 8) : j * 8 + 7 < n := by omega

-- § embedding gather

/-- For code widths 1, 2, 4, 8 the packing is exact: `(8/bits)*bits = 8` and the in-byte shift is `< 8`. -/
theorem code_packing (bits i : Nat) (hb : bits = 1 ∨ bits = 2 ∨ bits = 4 ∨ bits = 8) :
    (8 / bits) * bits = 8 ∧ (i % (8 / bits)) * bits < 8 := by
  rcases hb with h | h | h | h <;> subst h <;> omega

/-- Group of element `i` is below the number of groups when `group_elems | n`. -/
theorem group_in_range {i n g : Nat} (hg : 0 < g) (hn : n % g = 0) (hi : i < n) : i / g < n / g := by
  have h : i / g < n / g := Nat.div_lt_div_of_lt_of_dvd (Nat.dvd_of_mod_eq_zero hn) hi
  exact h

-- § broadcast, blobs, rebase

/-- Broadcast: output index `t*n*hc + i` is in bounds for `i < n*hc`. -/
theorem broadcast_index_lt {t T n hc i : Nat} (ht : t < T) (hi : i < n * hc) :
    t * (n * hc) + i < T * (n * hc) := rowmajor_lt ht hi

/-- Source column `i % n` is a valid column when `n > 0`. -/
theorem broadcast_source_col {i n : Nat} (hn : 0 < n) : i % n < n := Nat.mod_lt _ hn

/-- `fetch_blobs`: with `per > 0`, `i < n*per` gives blob `i / per < n` and offset `i % per < per`. -/
theorem fetch_split {i n per : Nat} (hp : 0 < per) (h : i < n * per) :
    i / per < n ∧ i % per < per ∧ i = (i / per) * per + i % per := by
  have h1 : i / per < n := Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm] at h; exact h)
  have h3 := Nat.div_add_mod i per
  refine ⟨h1, Nat.mod_lt _ hp, ?_⟩
  rw [Nat.mul_comm]
  exact h3.symm

/-- `rebase_ptrs`: consecutive rebased pointers are separated by at least one blob. -/
theorem rebase_separated {base k k' bytes : Nat} (h : k < k') :
    base + k * bytes + bytes ≤ base + k' * bytes := by
  have h1 : (k + 1) * bytes ≤ k' * bytes := Nat.mul_le_mul_right bytes (Nat.succ_le_of_lt h)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

/-- `rebase_ptrs` launches 128 threads, so it handles at most 128 blobs. -/
theorem rebase_capacity {k n : Nat} (h : k < n) (hn : n ≤ 128) : k < 128 := by omega

-- § small helpers

/-- Window ids: `start + width = n_kv`, `width ≤ window`, `width ≤ n_kv`. -/
theorem window_width (n_kv window : Nat) :
    let start := if n_kv > window then n_kv - window else 0
    start + (n_kv - start) = n_kv ∧ n_kv - start ≤ window ∧ n_kv - start ≤ n_kv := by
  simp only
  split <;> omega

/-- Dense steps: the four entries of query `i` occupy `i*4 .. i*4+3` and stay below `4n`. -/
theorem dense_steps_bounds {i n k : Nat} (hi : i < n) (hk : k < 4) : i * 4 + k < 4 * n := by omega

/-- `ident_hits` requires `n ≤ 1024` threads; thread `i < n` is then valid. -/
theorem ident_threads {i n : Nat} (hi : i < n) (hn : n ≤ 1024) : i < 1024 := by omega

/-- Grid-stride loops: with stride `s`, `i = i % s + s*(i/s)`, so each `i` has one owner. -/
theorem stride_owner (i s : Nat) : i % s + s * (i / s) = i := Nat.mod_add_div i s

/-- `copy_indexed` caps blocks at 64. -/
theorem copy_blocks_cap (n : Nat) : (if (n + 255) / 256 < 64 then (n + 255) / 256 else 64) ≤ 64 := by
  split <;> omega

/-- Top-prob reduction: 1024 threads are 32 warps, matching `part[32]`. -/
theorem top_prob_warps : 1024 / 32 = 32 := by decide

end Strata.Verify

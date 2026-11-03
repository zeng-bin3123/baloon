/-!
# MODULE: gdn_reference_kernels
The unfused gated-delta-net kernels: the column-parallel recurrence step, the sliding 1-D convolution,
per-row L2 normalisation, the beta sigmoid gate and the output RMSNorm/gate.  This file proves the
coverage and index facts of the launch configuration and the list-level semantics of the conv slide.

## Source semantics
* The state layout is `(S, h_v, S)` with `j` fastest; thread `(h, j)` owns one column end to end.
* `MAX_H` heads share a block; `h0 + hi < h_v` is enforced through `nh = min(MAX_H, h_v - h0)`.
* The conv state holds the `d_conv - 1` oldest inputs; the new input lands in the last tap and the
  state slides by one.

## Form template identifiers
- `gdn_step_kernel` -> `recurrence_step`; `gdn_conv_kernel` -> `conv_slide`; `gdn_l2_kernel` -> `row_l2_norm`
- `gdn_out_norm_kernel` -> `out_rms_gate`; `gdn_beta_gate_kernel` -> `beta_sigmoid`
- `S` -> `state_dim`; `MAX_H` -> `heads_per_block`; `JTHREADS` -> `column_threads`
-/

namespace Strata.GdnReference

-- § arithmetic helper

/-- Row-major linearisation stays in bounds. -/
theorem lin2_lt {i a c d : Nat} (hi : i < a) (hc : c < d) : i * d + c < a * d := by
  have h := Nat.mul_le_mul_right d (show i + 1 ≤ a by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

-- § launch geometry

/-- Every column `j < S` is handled by some thread of the `ceil(S/32)` column blocks. -/
theorem column_cover {S j : Nat} (hj : j < S) : j / 32 < (S + 31) / 32 ∧ j % 32 < 32 := by omega

/-- With 8 heads per block every head lies in one of the `ceil(h_v/8)` blocks. -/
theorem heads_cover_eight {h hv : Nat} (hh : h < hv) :
    h / 8 < (hv + 7) / 8 ∧ h % 8 < 8 ∧ (h / 8) * 8 + h % 8 = h := by omega

/-- With one head per block the staged head count is exactly 1. -/
theorem staged_heads_one {h0 hv : Nat} (h : h0 < hv) : min 1 (hv - h0) = 1 := by omega

/-- With 8 heads per block the staged head count is between 1 and 8 for every launched block. -/
theorem staged_heads_eight {by' hv : Nat} (h : by' < (hv + 7) / 8) :
    1 ≤ min 8 (hv - by' * 8) ∧ min 8 (hv - by' * 8) ≤ 8 := by omega

/-- Block counts for S=128, h_v=48: 24 blocks with 8 heads per block, 192 with one head per block. -/
theorem grid_block_counts : ((128 + 31) / 32) * ((48 + 7) / 8) = 24 ∧ ((128 + 31) / 32) * 48 = 192 := by
  decide

/-- Staged rows of `k`/`q` (arrays of 128) are never indexed out of bounds when `S ≤ 128`. -/
theorem staged_in_bounds {S i : Nat} (hS : S ≤ 128) (hi : i < S) : i < 128 := by omega

/-- Modulo head pairing yields a valid key/query head. -/
theorem key_head_lt {h h_k : Nat} (hk : 0 < h_k) : h % h_k < h_k := Nat.mod_lt _ hk

-- § state addressing

/-- The two forms of the column address agree: `(i*h_v + h)*S + j = i*(h_v*S) + (h*S + j)`. -/
theorem stride_form (i hv h S j : Nat) : (i * hv + h) * S + j = i * (hv * S) + (h * S + j) := by
  rw [Nat.add_mul, Nat.mul_assoc]
  omega

/-- Column address stays inside the `S x h_v x S` state. -/
theorem column_index_lt {i h j hv S : Nat} (hi : i < S) (hh : h < hv) (hj : j < S) :
    (i * hv + h) * S + j < S * hv * S :=
  lin2_lt (lin2_lt hi hh) hj

-- § sliding convolution (list model)

/-- Dot product of two integer lists (truncating to the shorter). -/
def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- Conv output: `[conv_state | x]` against the kernel taps. -/
def convOut (st w : List Int) (x : Int) : Int := dot (st ++ [x]) w

/-- Conv state update: drop the oldest entry, append the newest input. -/
def convSlide (st : List Int) (x : Int) : List Int := st.drop 1 ++ [x]

/-- Sliding keeps the state length whenever the state is nonempty (`d_conv >= 2`). -/
theorem convSlide_length (st : List Int) (x : Int) (h : st ≠ []) :
    (convSlide st x).length = st.length := by
  cases st with
  | nil => exact absurd rfl h
  | cons a rest => simp [convSlide]

/-- The newest input is the last element of the new state. -/
theorem convSlide_last (st : List Int) (x : Int) : (convSlide st x).reverse.head? = some x := by
  simp [convSlide]

/-- Dropping the oldest and appending the newest, spelled out for a nonempty state. -/
theorem convSlide_cons (a : Int) (rest : List Int) (x : Int) : convSlide (a :: rest) x = rest ++ [x] := by
  simp [convSlide]

/-- With `d_conv = 1` the code stores to `st[d_conv - 2]`, i.e. index -1: the kernel needs `d_conv >= 2`. -/
theorem conv_single_tap_index_negative : ((1 : Int) - 2) < 0 := by decide

/-- With `d_conv = 2` the shift loop is empty and only the final store remains. -/
theorem conv_two_tap_shift_loop_empty : 2 - 2 = 0 := by decide

-- § row-wise reductions

/-- A strided loop `i = tid; i < cols; i += 32` assigns each element to exactly one of 32 threads. -/
theorem strided_owner {i : Nat} : i % 32 < 32 ∧ (i / 32) * 32 + i % 32 = i := by omega

/-- The beta-gate launch (128 threads per block) covers every head. -/
theorem beta_gate_cover {n i : Nat} (hi : i < n) : i / 128 < (n + 127) / 128 := by omega

end Strata.GdnReference

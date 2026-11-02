/-!
# MODULE: gdn_reference_kernels
Launch geometry and addressing of the non-projection parts of the gated delta net: one-step recurrence
(one thread per (head, j) state column), the sliding 4-tap causal convolution, L2 normalisation, the
output RMS-norm * gate, and the in-place beta sigmoid.

## Source semantics
- State layout `(S, h_v, S)` with j fastest: element (i, h, j) at `(i*h_v + h)*S + j`; the recurrence touches only
  one (h, j) column per thread so it needs no barrier.
- Grid is (ceil(S/32), ceil(h_v/MAX_H)) with MAX_H = 1 (latency-hiding choice): one head per block row.
- Conv state per channel holds d_conv-1 values; the new input lands in the last tap and the window slides.
- `gdn_step` aborts for S > 128 (shared staging is [MAX_H][128]).

## Form template identifiers
- S -> state_dim, h_k -> n_qk_heads, h_v -> n_v_heads, dc -> d_conv, JTHREADS -> j_threads
- gdn_step -> delta_step, gdn_conv_step -> causal_conv_step, gdn_beta_gate -> beta_sigmoid
-/

namespace Strata.GdnReference

def JTHREADS : Nat := 32
def MAX_H : Nat := 1

-- § helpers

/-- Row-major index bound. -/
theorem idx_lt {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § step kernel geometry

/-- With MAX_H = 1 each block row stages exactly one head. -/
theorem one_head_per_block (by_ h_v : Nat) (h : by_ < h_v) : min 1 (h_v - by_) = 1 := by
  omega

/-- Grid y extent with MAX_H = 1 equals h_v. -/
theorem grid_y_is_h_v (h_v : Nat) : (h_v + MAX_H - 1) / MAX_H = h_v := by
  unfold MAX_H
  omega

/-- Grid x extent covers every column j < S. -/
theorem grid_x_covers (S j : Nat) (hj : j < S) : j / 32 < (S + 31) / 32 := by
  omega

/-- A column index inside S lies in block j/32 at lane j%32. -/
theorem column_split (j : Nat) : (j / 32) * 32 + j % 32 = j := by
  omega

/-- Head pairing (MODULO) stays in range. -/
theorem head_pair_in_range (h h_k : Nat) (hk : 0 < h_k) : h % h_k < h_k := Nat.mod_lt _ hk

/-- Element address of state row i for column pointer `h*S + j` and stride `h_v*S`. -/
theorem state_elem_address (i h j S h_v : Nat) :
    h * S + j + i * (h_v * S) = (i * h_v + h) * S + j := by
  simp only [Nat.add_mul, Nat.mul_assoc]
  omega

/-- State element address stays inside S*h_v*S. -/
theorem state_elem_lt (i h j S h_v : Nat) (hi : i < S) (hh : h < h_v) (hj : j < S) :
    (i * h_v + h) * S + j < S * h_v * S :=
  idx_lt (idx_lt hi hh) hj

/-- Output address `h*S + j` stays inside `h_v*S`. -/
theorem out_elem_lt (h j S h_v : Nat) (hh : h < h_v) (hj : j < S) : h * S + j < h_v * S :=
  idx_lt hh hj

/-- The `S > 128` guard: accepted sizes fit the staged shared arrays. -/
theorem staged_rows_fit (S i : Nat) (hS : S ≤ 128) (hi : i < S) : i < 128 := by
  omega

-- § conv kernel

/-- Sliding the window drops the oldest value and appends the new one (d_conv = 4 case). -/
def slide3 (st : Int × Int × Int) (x : Int) : Int × Int × Int := (st.2.1, st.2.2, x)

/-- After a slide the window is [s1, s2, x]. -/
theorem slide3_eq (a b c x : Int) : slide3 (a, b, c) x = (b, c, x) := rfl

/-- Three slides leave exactly the last three inputs. -/
theorem slide3_three (s : Int × Int × Int) (x y z : Int) :
    slide3 (slide3 (slide3 s x) y) z = (x, y, z) := rfl

/-- Generic slide on lists keeps the length when non-empty. -/
def slideList (st : List Int) (x : Int) : List Int := st.tail ++ [x]

/-- The generic slide preserves length of a non-empty window. -/
theorem slideList_length (st : List Int) (x : Int) (h : st ≠ []) :
    (slideList st x).length = st.length := by
  cases st with
  | nil => exact absurd rfl h
  | cons a t => simp [slideList]

/-- Per-channel state address bound for stride dc-1. -/
theorem conv_state_in_range (c i C dc : Nat) (hc : c < C) (hi : i < dc - 1) :
    c * (dc - 1) + i < C * (dc - 1) :=
  idx_lt hc hi

/-- Per-channel weight address bound for stride dc. -/
theorem conv_weight_in_range (c i C dc : Nat) (hc : c < C) (hi : i < dc) :
    c * dc + i < C * dc :=
  idx_lt hc hi

/-- Grid for the conv kernel covers all channels (256 per block). -/
theorem conv_grid_covers (C : Nat) : C ≤ (C + 255) / 256 * 256 := by
  omega

-- § beta gate

/-- Grid for the in-place beta sigmoid covers all heads (128 per block). -/
theorem beta_grid_covers (n : Nat) : n ≤ (n + 127) / 128 * 128 := by
  omega

-- § norm kernels

/-- Output norm: one block per head, address `h*S + i` stays in range for every strided i. -/
theorem out_norm_in_range (h i S h_v : Nat) (hh : h < h_v) (hi : i < S) : h * S + i < h_v * S :=
  idx_lt hh hi

/-- L2 norm warp reduction width is guarded: accepted column counts are at most 1024. -/
theorem l2_cols_guard (cols : Nat) (h : cols ≤ 1024) : cols < 1025 := by
  omega

end Strata.GdnReference

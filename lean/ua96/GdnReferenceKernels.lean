/-!
# MODULE: gdn_reference_kernels
SOURCES: src/kernels/cuda/gdn.cu
The gated delta-net's non-projection parts: the recurrence step (one thread per (head, column) of the
state), the sliding conv, per-row L2 norm, beta sigmoid gate and output RMS-norm gating.

## Source semantics
* State layout `(S, h_v, S)` with the column index `j` fastest; a thread owns one column end to end, so the
  recurrence needs no barrier.
* The decay is applied BEFORE the rank-1 update; the read-out uses the UPDATED state.
* `MAX_H = 1`: grid `(S/32, h_v)`; the thread count is `S*h_v` independent of packing.
* conv: `kernel[0]` meets the OLDEST state row, the new input meets the LAST tap, the state slides.
* `beta` must be passed through a sigmoid before the step (the C1 bug).

## Form template identifiers
* `JTHREADS` -> column_lanes_per_block, `MAX_H` -> heads_per_block, `decayThenUpdate` -> delta_rule_column.
-/

namespace Ref.Gdn

/-- Threads along `j`. -/
def JTHREADS : Nat := 32
/-- Heads staged per block (latency-hiding knob). -/
def MAX_H : Nat := 1

/-- With `S = 128` there are 4 column blocks per head. -/
theorem col_blocks : (128 + 32 - 1) / 32 = 4 := by decide

/-- Every column `j < 128` is owned by some (block, lane). -/
theorem col_cover (j : Nat) (h : j < 128) : ∃ bx tx, bx < 4 ∧ tx < 32 ∧ bx * 32 + tx = j := by
  refine ⟨j / 32, j % 32, ?_, ?_, ?_⟩ <;> omega

/-- Ownership is unique, so no two threads write the same column. -/
theorem col_unique (bx tx bx' tx' : Nat) (ht : tx < 32) (ht' : tx' < 32)
    (h : bx * 32 + tx = bx' * 32 + tx') : bx = bx' ∧ tx = tx' := by omega

/-- With `MAX_H = 1` the head grid dimension equals `h_v`. -/
theorem grid_y_eq (h_v : Nat) : (h_v + MAX_H - 1) / MAX_H = h_v := by
  unfold MAX_H; omega

/-- The block count for `h_v = 48` is 192 (4 warps on every SM instead of 24 blocks). -/
theorem blocks_for_48 : 4 * ((48 + MAX_H - 1) / MAX_H) = 192 := by decide

/-- The staged q/k rows hold at most 128 entries, so any `S <= 128` indexes inside them. -/
theorem staging_in_bounds (S i : Nat) (hS : S ≤ 128) (hi : i < S) : i < 128 := by omega

/-- Head pairing is a modulo pairing. -/
theorem src_head_lt (h h_k : Nat) (hk : 0 < h_k) : h % h_k < h_k := Nat.mod_lt _ hk

/-- One column of the recurrence, abstracted to a single state row with scalar key/query. -/
def decayThenUpdate (s dec k v b : Int) : Int :=
  let sd := dec * s
  sd + k * ((v - sd * k) * b)

/-- With unit key and unit write strength the updated state equals the value (delta rule exactness). -/
theorem delta_exact (s dec v : Int) : decayThenUpdate s dec 1 v 1 = v := by
  unfold decayThenUpdate; simp only [Int.mul_one]; omega

/-- Zero write strength keeps the decayed state: decay happens before (and independently of) the update. -/
theorem zero_beta_only_decays (s dec k v : Int) : decayThenUpdate s dec k v 0 = dec * s := by
  unfold decayThenUpdate; simp

/-- Four-tap conv with the kernel's tap convention (oldest history meets kernel tap 0). -/
def convOut (st0 st1 st2 x w0 w1 w2 w3 : Int) : Int := st0 * w0 + st1 * w1 + st2 * w2 + x * w3

/-- The slide: `st[i] := st[i+1]` for `i < dc-2`, last slot receives the input. -/
def convSlide (st1 st2 x : Int) : Int × Int × Int := (st1, st2, x)

/-- The newest input is stored in the last state slot. -/
theorem convSlide_last (st1 st2 x : Int) : (convSlide st1 st2 x).2.2 = x := rfl

/-- Per-channel state offsets `c*3 + i` (d_conv = 4) are injective across channels. -/
theorem conv_state_inj (c i c' i' : Nat) (hi : i < 3) (hi' : i' < 3)
    (h : c * 3 + i = c' * 3 + i') : c = c' ∧ i = i' := by omega

/-- Conv launch covers all channels: `ceil(C/256) * 256 >= C`. -/
theorem conv_grid_covers (C : Nat) : C ≤ (C + 255) / 256 * 256 := by omega

/-- Beta-gate launch covers all heads. -/
theorem beta_grid_covers (n : Nat) : n ≤ (n + 127) / 128 * 128 := by omega

/-- `gdn_l2_norm` accepts only `cols <= 1024`, which a single warp strided loop covers in `<= 32` steps. -/
theorem l2_steps (cols : Nat) (h : cols ≤ 1024) : (cols + 31) / 32 ≤ 32 := by omega

/-- The L2 warp-strided loop reaches every element of the row. -/
theorem l2_stride_cover (i cols : Nat) (h : i < cols) :
    ∃ lane step, lane < 32 ∧ i = lane + 32 * step := by
  exact ⟨i % 32, i / 32, by omega, by omega⟩

end Ref.Gdn

/-!
# MODULE: gdn_reference_kernels
The non-projection parts of the gated delta-net as separate kernels: the column recurrence (`gdn_step_kernel`), the
sliding depthwise conv (`gdn_conv_kernel`), L2 norm, beta sigmoid and gated RMS output norm.  This module models the
recurrence algebraically over `Int` lists and proves it equals the fused form used by the faster kernels.

## Source semantics
* One thread owns one (head, column) of the `(S, h_v, S)` state and needs no barrier; it decays the column, contracts it
  with `k`, forms `d = (v - sk) * beta`, applies the rank-1 update and reads out against `q` using the UPDATED state.
* Decay comes BEFORE the update: `sk = sum_i (s_i * dec) * k_i = dec * sum_i s_i * k_i`.
* The conv state holds `d_conv - 1` taps; the window is `state ++ [x]`, the output is its dot with the channel's
  kernel (oldest tap first), and the new state is the window without its oldest element.
* `MAX_H` is a latency-hiding knob only: heads per block changes the grid, never the set of (head, column) pairs.

## Form template identifiers
* `dot`            : sum of pairwise products
* `decayedState`   : `s * dec` elementwise
* `updateState`    : `dec * s + k * d`
* `convStep`       : sliding window step on a list state
-/

namespace Strata.GdnRef

-- § algebra of the recurrence

/-- Pairwise dot product of two integer lists (zip-truncating). -/
def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- Pass 1 of the kernel: the state after decay, `s_i * dec`. -/
def decayedState (dec : Int) (s : List Int) : List Int := s.map (fun x => x * dec)

/-- Contracting the decayed state against `k` equals `dec` times the raw contraction. -/
theorem decayed_dot (dec : Int) (s k : List Int) : dot (decayedState dec s) k = dec * dot s k := by
  unfold decayedState
  induction s generalizing k with
  | nil => simp [dot]
  | cons a as ih =>
    cases k with
    | nil => simp [dot]
    | cons b bs =>
      simp only [List.map_cons, dot]
      rw [ih, Int.mul_add, Int.mul_comm a dec, Int.mul_assoc]

/-- Write strength `d = (v - sk) * beta` with `sk` taken from the decayed state. -/
def deltaTwoPass (dec : Int) (s k : List Int) (v beta : Int) : Int :=
  (v - dot (decayedState dec s) k) * beta

/-- The fused kernels compute the same write strength as `(v - dec * (s . k)) * beta`. -/
theorem delta_fused_eq (dec : Int) (s k : List Int) (v beta : Int) :
    deltaTwoPass dec s k v beta = (v - dec * dot s k) * beta := by
  unfold deltaTwoPass
  rw [decayed_dot]

/-- Rank-1 update of one state column: `s_i * dec + k_i * d` (the two-pass form). -/
def updateTwoPass (dec : Int) (s k : List Int) (d : Int) : List Int :=
  List.zipWith (fun a b => a * dec + b * d) s k

/-- Rank-1 update in the fused form `dec * s_i + k_i * d`. -/
def updateFused (dec : Int) (s k : List Int) (d : Int) : List Int :=
  List.zipWith (fun a b => dec * a + b * d) s k

/-- The two update forms agree elementwise (multiplication commutes). -/
theorem update_forms_eq (dec : Int) (s k : List Int) (d : Int) :
    updateTwoPass dec s k d = updateFused dec s k d := by
  unfold updateTwoPass updateFused
  congr 1
  funext a b
  rw [Int.mul_comm a dec]

/-- The update never changes the column length when `s` and `k` have equal length. -/
theorem update_length (dec : Int) (s k : List Int) (d : Int) (h : s.length = k.length) :
    (updateTwoPass dec s k d).length = s.length := by
  unfold updateTwoPass
  simp [List.length_zipWith, h]

-- § sliding convolution

/-- Window `state ++ [x]` fed to the dot with the channel kernel. -/
def convWindow (st : List Int) (x : Int) : List Int := st ++ [x]

/-- Convolution output: dot of the window with the kernel (kernel[0] reads the oldest tap). -/
def convOut (st : List Int) (x : Int) (w : List Int) : Int := dot (convWindow st x) w

/-- State after the step: drop the oldest tap, append the newest. -/
def convNext (st : List Int) (x : Int) : List Int := (convWindow st x).tail

/-- For a non-empty state the slide is `tail st ++ [x]`. -/
theorem convNext_eq (a : Int) (r : List Int) (x : Int) : convNext (a :: r) x = r ++ [x] := by
  simp [convNext, convWindow]

/-- The slide keeps the state length. -/
theorem convNext_length (a : Int) (r : List Int) (x : Int) :
    (convNext (a :: r) x).length = (a :: r).length := by
  simp [convNext, convWindow]

/-- The newest input always lands in the last tap of the window. -/
theorem convWindow_last (st : List Int) (x : Int) : (convWindow st x).getLast? = some x := by
  simp [convWindow]

-- § launch geometry

/-- With `S % 32 = 0` the column index `bx*32 + tx` covers `[0, S)` once. -/
theorem col_cover (S j : Nat) (hS : S % 32 = 0) (hj : j < S) :
    j / 32 < S / 32 ∧ (j / 32) * 32 + j % 32 = j := by omega

/-- With `MAX_H = 1` every block's head is valid and exactly one head is staged. -/
theorem head_block_one (h_v blk : Nat) (hb : blk < (h_v + 1 - 1) / 1) : blk * 1 < h_v := by omega

/-- With `MAX_H = 8` the first head of every block is valid. -/
theorem head_block_eight (h_v blk : Nat) (hb : blk < (h_v + 8 - 1) / 8) : blk * 8 < h_v := by omega

/-- Grid sizes quoted in the source: 24 blocks for `MAX_H = 8`, 192 for `MAX_H = 1` at `h_v = 48`. -/
theorem grid_sizes : (128 / 32) * (48 / 8) = 24 ∧ (128 / 32) * (48 / 1) = 192 := by decide

/-- Thread count is independent of `MAX_H`: `S * h_v = 6144` columns at the production shape. -/
theorem columns_fixed : 128 * 48 = 6144 := by decide

/-- A 32-thread strided loop over `cols` entries visits entry `i` from thread `i % 32`. -/
theorem warp_stride_owner (i : Nat) : i % 32 < 32 ∧ i % 32 + 32 * (i / 32) = i := by omega

/-- Conv channels: `blocks = (C + 255) / 256` of 256 threads cover every channel. -/
theorem conv_blocks_cover (C c : Nat) (h : c < C) : c / 256 < (C + 255) / 256 := by omega

/-- State slot `c*(dc-1) + i` of the conv state stays inside `C * (dc-1)`. -/
theorem conv_state_lt (C dc c i : Nat) (hc : c < C) (hi : i < dc - 1) : c * (dc - 1) + i < C * (dc - 1) := by
  have h : (c + 1) * (dc - 1) ≤ C * (dc - 1) := Nat.mul_le_mul_right (dc - 1) hc
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- Warp-shuffle L2 norm is only used for `cols ≤ 1024`: at most 32 partials per row. -/
theorem l2_width_bound (cols : Nat) (h : cols ≤ 1024) : (cols + 31) / 32 ≤ 32 := by omega

-- § beta gate

/-- The beta gate is applied to `h_v` scalars, one launch of `(h_v + 127) / 128` blocks of 128 threads. -/
theorem beta_gate_cover (h_v i : Nat) (h : i < h_v) : i / 128 < (h_v + 127) / 128 := by omega

end Strata.GdnRef

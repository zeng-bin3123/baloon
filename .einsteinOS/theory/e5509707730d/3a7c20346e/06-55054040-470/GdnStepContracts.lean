/-!
# MODULE: gdn_step_contracts
SOURCES: include/strata/kernels/gdn.hpp
Gated delta-net layer pieces between the projections: the delta-rule state step, the causal conv step with
carried state, the L2 norm, the beta sigmoid and the closing per-head norm.

## Source semantics
* Real geometry: S = 128, h_k = 16, h_v = 48, conv channels C = 2*S*h_k + S*h_v = 10240, d_conv = 4;
  48 GDN layers and 12 QSA layers.
* State layout is (S, H_v, S) = (i, h, j) with j fastest: flat index `(i * h_v + h) * S + j`.
* Head pairing is MODULO (`h % h_k`), not interleaved.
* Recurrence per column: decay first (`st = dec * state`), `sk = k . st`, `d = (v - sk) * beta`,
  `state' = st + k * d`, `o = state' . q`. The kernel form `g*s + k*((v - g*(s*k))*beta)` is the same number.
* Conv: output `sum_i inp[i] * kW[c*d_conv + i]` over `inp = [conv_state | x]`; the new state is the last
  `d_conv - 1` entries of `inp`.
* L2 norm adds eps to the squared norm (no division by the width); the closing norm is per head row.

## Form template identifiers
`gdn_step` -> `delta_rule_step`; `gdn_conv_step` -> `causal_conv_step`; `S` -> `head_dim`; `h_k` -> `n_key_heads`;
`h_v` -> `n_value_heads`; `conv_state` -> `conv_carry`; `kW` -> `conv_weight`.
-/

namespace Strata.Gdn

-- § geometry

/-- Conv channels for the real artifact: q and k (16 heads of 128 each) plus v (48 heads of 128). -/
theorem conv_channels_real : 2 * 128 * 16 + 128 * 48 = 10240 := by decide

/-- The value heads are a whole multiple of the key heads (3 value heads per key head). -/
theorem heads_multiple : 48 % 16 = 0 ∧ 48 / 16 = 3 := by decide

/-- Layer split of the artifact: 48 GDN layers and 12 QSA layers. -/
theorem layer_split : 48 + 12 = 60 := by decide

/-- Conv weight has `channels * d_conv` entries. -/
theorem conv_weight_len_real : 10240 * 4 = 40960 := by decide

-- § state layout (S, H_v, S), j fastest

/-- Flat index of state element `(i, h, j)`. -/
def stateIdx (S hv i h j : Nat) : Nat := (i * hv + h) * S + j

/-- Walking `j` is contiguous: the coalescing property the layout was chosen for. -/
theorem stateIdx_contiguous (S hv i h j : Nat) : stateIdx S hv i h (j + 1) = stateIdx S hv i h j + 1 := by
  unfold stateIdx
  omega

/-- Every valid element lands inside the `S * H_v * S` buffer. -/
theorem stateIdx_lt (S hv i h j : Nat) (hi : i < S) (hh : h < hv) (hj : j < S) :
    stateIdx S hv i h j < S * hv * S := by
  unfold stateIdx
  have e1 : (i + 1) * hv = i * hv + hv := by rw [Nat.add_mul, Nat.one_mul]
  have h1 : (i + 1) * hv ≤ S * hv := Nat.mul_le_mul_right hv hi
  have e2 : (i * hv + h + 1) * S = (i * hv + h) * S + S := by rw [Nat.add_mul, Nat.one_mul]
  have h2 : (i * hv + h + 1) * S ≤ (S * hv) * S := Nat.mul_le_mul_right S (by omega)
  omega

/-- The row-group addressing of the multi-token kernel, `base + r * (h_v * S)` with
`base = ((g*32) * h_v + head) * S + col`, is the same element as `stateIdx` of row `g*32 + r`. -/
theorem row_group_address (g r hv head S col : Nat) :
    ((g * 32) * hv + head) * S + col + r * (hv * S) = ((g * 32 + r) * hv + head) * S + col := by
  simp only [Nat.add_mul, Nat.mul_assoc]
  omega

-- § head pairing

/-- Modulo pairing always names a valid key head. -/
theorem pair_mod_lt (h hk : Nat) (hpos : 0 < hk) : h % hk < hk := Nat.mod_lt h hpos

/-- The two conventions disagree on a real head: value head 1 pairs with key head 1 (modulo) but 0 (interleave). -/
theorem pairing_distinguishable : (1 % 16 : Nat) ≠ 1 / (48 / 16) := by decide

-- § delta-rule recurrence

/-- Kernel form of one scalar state update (decay applied inside the delta). -/
def stepKernel (g b s k v : Int) : Int := g * s + k * ((v - g * (s * k)) * b)

/-- Reference form: decay first, then the rank-one update. -/
def stepRef (g b s k v : Int) : Int := (g * s) + k * ((v - k * (g * s)) * b)

/-- The two forms are the same number. -/
theorem kernel_eq_ref (g b s k v : Int) : stepKernel g b s k v = stepRef g b s k v := by
  unfold stepKernel stepRef
  have h : g * (s * k) = k * (g * s) := by
    rw [Int.mul_comm s k]
    exact Int.mul_left_comm g k s
  rw [h]

/-- With zero write strength the step is pure decay. -/
theorem step_beta_zero (g s k v : Int) : stepKernel g 0 s k v = g * s := by
  unfold stepKernel
  omega

/-- With no decay, unit key and unit write strength the delta rule overwrites the state with the value. -/
theorem step_overwrite (s v : Int) : stepKernel 1 1 s 1 v = v := by
  unfold stepKernel
  omega

-- § causal conv step

/-- Input column `i` of `[conv_state | x]` with `d - 1` carried columns. -/
def convInp {α : Type} (st : Nat → α) (x : α) (d i : Nat) : α := if i < d - 1 then st i else x

/-- The new state entry `j` is input column `j + 1`: the oldest column is dropped. -/
def convNext {α : Type} (st : Nat → α) (x : α) (d j : Nat) : α := convInp st x d (j + 1)

/-- Tap 0 reads the OLDEST carried column. -/
theorem convInp_oldest {α : Type} (st : Nat → α) (x : α) (d : Nat) (hd : 2 ≤ d) :
    convInp st x d 0 = st 0 := by
  unfold convInp
  have hc : 0 < d - 1 := by omega
  rw [if_pos hc]

/-- The newest input lands in the LAST state slot. -/
theorem convNext_last {α : Type} (st : Nat → α) (x : α) (d : Nat) (hd : 2 ≤ d) :
    convNext st x d (d - 2) = x := by
  unfold convNext convInp
  have hc : ¬ (d - 2 + 1 < d - 1) := by omega
  rw [if_neg hc]

/-- Every other state slot is shifted down by one column. -/
theorem convNext_shift {α : Type} (st : Nat → α) (x : α) (d j : Nat) (h : j + 2 < d) :
    convNext st x d j = st (j + 1) := by
  unfold convNext convInp
  have hc : j + 1 < d - 1 := by omega
  rw [if_pos hc]

/-- Weight index `c * d + i` (tap `i` of channel `c`, `i` fastest) stays inside `channels * d`. -/
theorem convWeightIdx_lt (c d i channels : Nat) (hc : c < channels) (hi : i < d) :
    c * d + i < channels * d := by
  have e : (c + 1) * d = c * d + d := by rw [Nat.add_mul, Nat.one_mul]
  have h : (c + 1) * d ≤ channels * d := Nat.mul_le_mul_right d hc
  omega

-- § norms

/-- Row-major element `h * S + d` of a `(h_v, S)` tensor stays in range, so the closing norm is per head row. -/
theorem headRowIdx_lt (h S d hv : Nat) (hh : h < hv) (hd : d < S) : h * S + d < hv * S := by
  have e : (h + 1) * S = h * S + S := by rw [Nat.add_mul, Nat.one_mul]
  have h1 : (h + 1) * S ≤ hv * S := Nat.mul_le_mul_right S hh
  omega

/-- L2 norm vs the mean-based misreading: the sum-based denominator differs from the mean-based one
whenever the squared norm is non-zero (the true factor is the width, 128 here). -/
theorem l2_sum_vs_mean (sumsq : Nat) (h : 0 < sumsq) : sumsq / 128 < sumsq := by
  omega

end Strata.Gdn

/-!
# MODULE: gated_delta_net_reference
The reference (non-fused) gated delta-net kernels: one thread per (head, column) recurrence step, the
`ggml_ssm_conv` sliding conv, the double-precision per-row L2 normalisation, the beta sigmoid gate, and the
per-head RMS-norm * ssm_norm * sigmoid(z) output.

## Source semantics
* State layout (S, h_v, S) with j fastest: element (i, h, j) is at `(i*h_v + h)*S + j`.
* Grid = (ceil(S/32), ceil(h_v/MAX_H)) with MAX_H = 1 and 32 threads along j; thread j-index `bx*32 + tx < S`.
* Staging loop `i = tx; i < S; i += 32` visits every i < S exactly once.
* Decay happens BEFORE the update; the readout uses the UPDATED state.
* `gdn_step` rejects S > 128 (shared arrays are [MAX_H][128]).
* Conv: out[c] = Σ_{i<d_conv-1} st[i]*w[i] + x*w[d_conv-1]; the state slides by one.
* `beta = sigmoid(beta)` must be applied before the delta net (the C1 bug).

## Form template identifiers
`gdn_step_kernel` → delta_net_step ; `gdn_conv_kernel` → ssm_conv_step ; `gdn_l2_kernel` → row_l2_norm_f64 ;
`gdn_out_norm_kernel` → head_rms_gate ; `gdn_beta_gate_kernel` → beta_sigmoid_inplace.
-/

namespace Strata.GdnRef

-- § grid / thread mapping

/-- Grid x covers all S columns: ceil(S/32) blocks of 32. -/
theorem grid_x_covers (S j : Nat) (h : j < S) : j / 32 < (S + 31) / 32 := by omega

/-- For S = 128 the grid has 4 blocks in x. -/
theorem grid_x_128 : (128 + 31) / 32 = 4 := rfl

/-- Column index bx*32 + tx with tx < 32 splits uniquely. -/
theorem col_unique (bx tx bx' tx' : Nat) (h1 : tx < 32) (h2 : tx' < 32) (h : bx * 32 + tx = bx' * 32 + tx') :
    bx = bx' ∧ tx = tx' := by omega

/-- With MAX_H = 1, block y is the head and nh = min(1, h_v - h0) = 1 when h0 < h_v. -/
theorem nh_is_one (hv h0 : Nat) (h : h0 < hv) : min 1 (hv - h0) = 1 := by omega

/-- The y-grid ceil(h_v/1) = h_v covers all heads. -/
theorem grid_y_covers (hv : Nat) : (hv + 1 - 1) / 1 = hv := by omega

/-- Staging loop: element i < S is staged by thread i % 32 at iteration i / 32. -/
theorem stage_cover (i : Nat) : i = (i % 32) + 32 * (i / 32) := by omega

/-- The staging loop partitions [0, S): distinct (tx, m) give distinct i. -/
theorem stage_unique (tx m tx' m' : Nat) (h1 : tx < 32) (h2 : tx' < 32) (h : tx + 32 * m = tx' + 32 * m') :
    tx = tx' ∧ m = m' := by omega

/-- Head pairing: src = h % h_k < h_k. -/
theorem src_head_lt (h hk : Nat) (hp : 0 < hk) : h % hk < hk := Nat.mod_lt _ hp

/-- q/k staged row index src*S + i is below h_k*S. -/
theorem qk_stage_lt (src i hk S : Nat) (hs : src < hk) (hi : i < S) : src * S + i < hk * S := by
  have h1 : (src + 1) * S ≤ hk * S := Nat.mul_le_mul_right S (Nat.succ_le_of_lt hs)
  have h2 : (src + 1) * S = src * S + S := Nat.succ_mul src S
  omega

/-- State column address (h*S + j) + i*(h_v*S) equals (i*h_v + h)*S + j. -/
theorem state_addr_eq (i h j hv S : Nat) : h * S + j + i * (hv * S) = (i * hv + h) * S + j := by
  simp only [Nat.add_mul, Nat.mul_assoc]
  try omega

/-- State address is below S*h_v*S. -/
theorem state_addr_lt (i h j hv S : Nat) (hi : i < S) (hh : h < hv) (hj : j < S) :
    (i * hv + h) * S + j < S * hv * S := by
  have h1 : (i + 1) * hv ≤ S * hv := Nat.mul_le_mul_right hv (Nat.succ_le_of_lt hi)
  have h2 : (i + 1) * hv = i * hv + hv := Nat.succ_mul i hv
  have h3 : i * hv + h < S * hv := by omega
  have h4 : (i * hv + h + 1) * S ≤ S * hv * S := Nat.mul_le_mul_right S (Nat.succ_le_of_lt h3)
  have h5 : (i * hv + h + 1) * S = (i * hv + h) * S + S := Nat.succ_mul _ S
  omega

/-- `gdn_step` guard: S ≤ 128 ensures the staged arrays suffice. -/
theorem stage_guard (S i : Nat) (hS : S ≤ 128) (hi : i < S) : i < 128 := by omega

-- § delta-rule algebra (integer shadow)

/-- Decay then contract: sk = Σ (s_i*dec) * k_i, here for a list of rows. -/
def contract (dec : Int) (s k : List Int) : Int := ((s.zip k).map (fun p => p.1 * dec * p.2)).sum

/-- Contraction with zero decay vanishes. -/
theorem contract_zero_decay (s k : List Int) : contract 0 s k = 0 := by
  unfold contract
  induction s generalizing k with
  | nil => simp
  | cons a t ih =>
    cases k with
    | nil => simp
    | cons b u => simp [List.zip_cons_cons, ih]

/-- Contraction over empty state is zero. -/
theorem contract_nil (dec : Int) (k : List Int) : contract dec [] k = 0 := by
  simp [contract]

-- § conv

/-- Conv state slide: drop the oldest, append the newest. -/
def slide (st : List Int) (x : Int) : List Int := st.drop 1 ++ [x]

/-- Sliding preserves the state length when it is non-empty. -/
theorem slide_length (st : List Int) (x : Int) (h : st ≠ []) : (slide st x).length = st.length := by
  unfold slide
  cases st with
  | nil => exact absurd rfl h
  | cons a t => simp

/-- The newest sample is in the window. -/
theorem slide_mem (st : List Int) (x : Int) : x ∈ slide st x := by
  unfold slide; simp

/-- Conv state address c*(dc-1) + i stays within C*(dc-1). -/
theorem conv_state_lt (c i C dc : Nat) (hc : c < C) (hi : i < dc - 1) : c * (dc - 1) + i < C * (dc - 1) := by
  have h1 : (c + 1) * (dc - 1) ≤ C * (dc - 1) := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hc)
  have h2 : (c + 1) * (dc - 1) = c * (dc - 1) + (dc - 1) := Nat.succ_mul c _
  omega

/-- Conv weight address c*dc + i (i < dc) stays within C*dc. -/
theorem conv_weight_lt (c i C dc : Nat) (hc : c < C) (hi : i < dc) : c * dc + i < C * dc := by
  have h1 : (c + 1) * dc ≤ C * dc := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hc)
  have h2 : (c + 1) * dc = c * dc + dc := Nat.succ_mul c _
  omega

/-- Conv grid covers channels. -/
theorem conv_grid_covers (C : Nat) : C ≤ (C + 255) / 256 * 256 := by omega

-- § norms and gates

/-- `gdn_l2_norm` guard: cols ≤ 1024 so that 32 threads strided over cols suffice for the reduction. -/
theorem l2_guard (cols : Nat) (h : cols ≤ 1024) : cols ≤ 32 * 32 := by omega

/-- Out-norm thread stride loop partitions [0, S) like the staging loop. -/
theorem out_norm_cover (i : Nat) : i = (i % 32) + 32 * (i / 32) := by omega

/-- Beta gate grid ceil(n/128) covers n elements. -/
theorem beta_grid_covers (n : Nat) : n ≤ (n + 127) / 128 * 128 := by omega

/-- Head output address h*S + j is below h_v*S. -/
theorem out_addr_lt (h j hv S : Nat) (hh : h < hv) (hj : j < S) : h * S + j < hv * S := by
  have h1 : (h + 1) * S ≤ hv * S := Nat.mul_le_mul_right S (Nat.succ_le_of_lt hh)
  have h2 : (h + 1) * S = h * S + S := Nat.succ_mul h S
  omega

end Strata.GdnRef

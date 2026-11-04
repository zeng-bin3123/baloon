/-!
# MODULE: shared_expert_scratch_and_combine
The MoE shared expert (`silu(gate) * up`, down projection, per-token scalar sigmoid gate multiplied in) and the
final MoE combination (router-weighted routed parts plus the plain shared output). This module captures the
scratch carving, the activation-kind dispatch constraints, row indexing of the multi-token scaling and the
combine algebra.

## Source semantics
* Scratch (all regions 16-byte aligned): `gate | up | h_q8_0 | h_q8k | g`; total
  `2 a + q0 + qk + 32` with `a = round16(4 n_ff)`, `q0 = round16(34 n_ff/32)`, `qk = round16(292 n_ff/256)`.
* A down weight wanting Q8_K needs `n_ff % 256 = 0`; with `n_ff = 640` that is impossible.
* The shared output is added plain: `y_j = Σ_e w_e · parts[e,j] + shared_j`.
* Multi-token: `n_tok` in 1..8, row `t` scaled by `g[t]` at index `t * n_embd + i`.
* The scalar gate reads `n_embd` BF16 values: 2 bytes each, not 4.

## Form template identifiers
- `shared_expert_scratch_bytes`  → `scratch_size`
- `moe_combine_kernel`           → `weighted_combine_kernel`
- `scale_rows_kernel`            → `per_token_scale_kernel`
- `scalar_gate_kernel`           → `token_scalar_gate_kernel`
-/

namespace Strata.SharedExpert

-- § scratch carving

/-- Round up to a multiple of 16. -/
def round16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Rounding never shrinks and always aligns. -/
theorem round16_ok (x : Nat) : x ≤ round16 x ∧ round16 x % 16 = 0 := by
  unfold round16
  omega

/-- Offsets of the five regions. -/
theorem carve_offsets_aligned (a q0 qk : Nat) (ha : a % 16 = 0) (h0 : q0 % 16 = 0) (hk : qk % 16 = 0) :
    (a) % 16 = 0 ∧ (2 * a) % 16 = 0 ∧ (2 * a + q0) % 16 = 0 ∧ (2 * a + q0 + qk) % 16 = 0 := by
  omega

/-- Regions are in order and the scalar gate `g` (one float = 4 bytes) fits in the final 32 bytes. -/
theorem carve_fits (n_ff a q0 qk : Nat) (ha : 4 * n_ff ≤ a) (h0 : n_ff / 32 * 34 ≤ q0)
    (hk : n_ff / 256 * 292 ≤ qk) :
    4 * n_ff ≤ a ∧ a + 4 * n_ff ≤ 2 * a ∧ n_ff / 32 * 34 ≤ q0 ∧
      2 * a + q0 + qk + 4 ≤ 2 * a + q0 + qk + 32 := by
  omega

/-- For `n_ff = 640` the three regions are 2560, 680 and 584 bytes before rounding (and 2560, 688, 592 after). -/
theorem n_ff_640_sizes :
    4 * 640 = 2560 ∧ 640 / 32 * 34 = 680 ∧ 640 / 256 * 292 = 584 ∧
    round16 680 = 688 ∧ round16 584 = 592 ∧ round16 2560 = 2560 := by decide

-- § activation-kind constraints

/-- Q8_K is structurally impossible for `n_ff = 640`; Q8_0 is fine. -/
theorem q8k_impossible_for_640 : 640 % 256 ≠ 0 ∧ 640 % 32 = 0 := by decide

/-- The Q8_K image of a vector needs a whole number of 256-blocks: its byte count is `n/256 * 292`. -/
theorem q8k_bytes (m : Nat) : (256 * m) / 256 * 292 = 292 * m := by omega

/-- The Q8_0 image: `n/32 * 34` bytes. -/
theorem q8_0_bytes (m : Nat) : (32 * m) / 32 * 34 = 34 * m := by omega

-- § BF16 scalar gate

/-- BF16 weights are 2 bytes: reading them as 4-byte floats overruns the tensor by exactly its own size. -/
theorem gate_overrun : 2560 * 4 - 2560 * 2 = 5120 ∧ 2560 * 2 = 5120 := by decide

/-- Per-lane strided dot terms: thread `t` covers `i ≡ t (mod blockDim)`; each index has one owner. -/
theorem gate_stride_cover (i bd : Nat) : i = i % bd + bd * (i / bd) := (Nat.mod_add_div i bd).symm

/-- Number of warps of a 256-thread gate block: 8, matching `scratch[8]`. -/
theorem gate_warps : (256 + 31) >>> 5 = 8 := by decide

-- § multi-token scaling

/-- Row `t`, element `i` of the multi-token output is slot `t * n + i`, in range for `t < n_tok`, `i < n`. -/
theorem row_slot_lt (n_tok n t i : Nat) (ht : t < n_tok) (hi : i < n) : t * n + i < n_tok * n := by
  have h : (t + 1) * n ≤ n_tok * n := Nat.mul_le_mul_right n ht
  rw [Nat.add_mul] at h
  omega

/-- Distinct (token, element) pairs map to distinct slots: scaling by `g[t]` touches each slot once. -/
theorem row_slot_injective (n t i t' i' : Nat) (hi : i < n) (hi' : i' < n)
    (h : t * n + i = t' * n + i') : t = t' ∧ i = i' := by
  have hpos : 0 < n := by omega
  have h1 : (t * n + i) / n = t := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hi, Nat.zero_add]
  have h2 : (t' * n + i') / n = t' := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hi', Nat.zero_add]
  have ht : t = t' := by
    calc t = (t * n + i) / n := h1.symm
      _ = (t' * n + i') / n := by rw [h]
      _ = t' := h2
  subst ht
  exact ⟨rfl, by omega⟩

/-- Token count guard: `1 ≤ n_tok ≤ 8`. -/
def tokOk (n : Nat) : Bool := decide (1 ≤ n ∧ n ≤ 8)

/-- Exactly the accepted counts are 1 through 8. -/
theorem tokOk_iff (n : Nat) : tokOk n = true ↔ (1 ≤ n ∧ n ≤ 8) := by
  unfold tokOk
  simp

-- § combine algebra

/-- Sum of the first `k` router-weighted parts at column `j`. -/
def routedSum (w : Nat → Int) (parts : Nat → Int) (k : Nat) : Int :=
  match k with
  | 0 => 0
  | k + 1 => routedSum w parts k + w k * parts k

/-- The combine output: routed sum plus the plain shared value when present. -/
def combine (w parts : Nat → Int) (k : Nat) (shared : Option Int) : Int :=
  routedSum w parts k + shared.getD 0

/-- Without a shared output the combine is the routed sum. -/
theorem combine_no_shared (w parts : Nat → Int) (k : Nat) : combine w parts k none = routedSum w parts k := by
  simp [combine]

/-- The shared output is added plain, never router-weighted. -/
theorem combine_shared_plain (w parts : Nat → Int) (k : Nat) (s : Int) :
    combine w parts k (some s) = routedSum w parts k + s := by
  simp [combine]

/-- Appending one more routed expert adds exactly `w_k * part_k`. -/
theorem routedSum_succ (w parts : Nat → Int) (k : Nat) :
    routedSum w parts (k + 1) = routedSum w parts k + w k * parts k := rfl

/-- With all router weights zero the routed part vanishes. -/
theorem routedSum_zero_weights (parts : Nat → Int) (k : Nat) : routedSum (fun _ => 0) parts k = 0 := by
  induction k with
  | zero => rfl
  | succ k ih =>
    simp only [routedSum, ih]
    simp

/-- Launch guards: `k ≤ 64` accepted; the combine grid of 128-thread blocks covers `n_embd`. -/
theorem combine_grid_covers (n j : Nat) (h : j < n) : j / 128 < (n + 127) / 128 := by omega

end Strata.SharedExpert

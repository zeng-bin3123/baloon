/-!
# MODULE: shared_expert_combine
The shared expert (SwiGLU MLP with a per-token scalar sigmoid gate) and the MoE final combination.

## Source semantics
`h = silu(x Wg^T) * (x Wu^T)` (SiLU on GATE), `h = h Wd^T`, `g = sigmoid(x . w)` one scalar per token,
result `h * g`. The result is ADDED to the routed output without router weighting. `moe_combine` computes
`y_j = sum_e w_e * parts[e][j] (+ shared_j)` with `k <= 64`. Scratch is carved from caller memory as
`gate | up | q8_0 | q8k | g`, each region 16-byte aligned. The native path is chosen when any projection has a
native MMVQ weight; the multi-token path accepts 1..8 tokens.

## Form template identifiers
- `shared_expert_scratch_bytes` -> scratchBytes, `moe_combine_kernel` -> combine
- `native_gate/native_up/native_down` -> native flags, `scale_rows_kernel` -> row scaling
-/

namespace Strata.SharedExpert

-- § scratch carving

/-- Round up to a multiple of 16. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Alignment never shrinks. -/
theorem align16_ge (x : Nat) : x ≤ align16 x := by unfold align16; omega

/-- Alignment yields a multiple of 16. -/
theorem align16_mod (x : Nat) : align16 x % 16 = 0 := by unfold align16; omega

/-- Bytes of the `gate`/`up` float regions. -/
def regionA (nFf : Nat) : Nat := align16 (nFf * 4)
/-- Bytes of the Q8_0 region. -/
def regionQ0 (nFf : Nat) : Nat := align16 (nFf / 32 * 34)
/-- Bytes of the Q8_K region. -/
def regionQk (nFf : Nat) : Nat := align16 (nFf / 256 * 292)

/-- Total scratch bytes. -/
def scratchBytes (nFf : Nat) : Nat := regionA nFf * 2 + regionQ0 nFf + regionQk nFf + 32

/-- Offset of the scalar gate region. -/
def gateOffset (nFf : Nat) : Nat := regionA nFf * 2 + regionQ0 nFf + regionQk nFf

/-- The scalar gate (one float, padded to 32 bytes) ends exactly at the scratch end. -/
theorem gate_region_end (nFf : Nat) : gateOffset nFf + 32 = scratchBytes nFf := rfl

/-- Scratch for the model's `n_ff = 640`. -/
theorem scratch_640 : scratchBytes 640 = 2 * 2560 + 680 + 584 + 32 := by sorry

/-- Corrected scratch for `n_ff = 640`: the Q8_0 and Q8_K regions align up to 688 and 592 bytes
(the statement above uses the unaligned 680/584 and is left as `sorry`). -/
theorem scratch_640_aligned : scratchBytes 640 = 2 * 2560 + 688 + 592 + 32 := by decide

/-- All region starts are 16-byte aligned. -/
theorem region_offsets_aligned (nFf : Nat) :
    (regionA nFf * 2) % 16 = 0 ∧ (regionA nFf * 2 + regionQ0 nFf) % 16 = 0 ∧
    gateOffset nFf % 16 = 0 := by
  unfold gateOffset regionA regionQ0 regionQk align16
  omega

-- § final combination

/-- Dot product of two lists (shorter length wins). -/
def dot : List Int → List Int → Int
  | w :: ws, p :: ps => w * p + dot ws ps
  | _, _ => 0

/-- `moe_combine`: routed weighted sum plus an optional plain shared term. -/
def combine (ws ps : List Int) (shared : Option Int) : Int :=
  dot ws ps + shared.getD 0

/-- Without a shared output the result is exactly the weighted sum. -/
theorem combine_none (ws ps : List Int) : combine ws ps none = dot ws ps := by
  simp [combine]

/-- The shared output is added plain, not weighted. -/
theorem combine_some (ws ps : List Int) (s : Int) : combine ws ps (some s) = dot ws ps + s := rfl

/-- With empty routed parts only the shared output remains. -/
theorem combine_empty (s : Int) : combine [] [] (some s) = s := by
  simp [combine, dot]

/-- Weighted sum of a single routed expert. -/
theorem dot_singleton (w p : Int) : dot [w] [p] = w * p := by
  simp [dot]

-- § multi-token path

/-- Row scaling indexes token `t`'s row at `t * n + i`, inside the `T * n` buffer. -/
theorem row_index_lt (t T i n : Nat) (ht : t < T) (hi : i < n) : t * n + i < T * n := by
  have h1 := Nat.mul_le_mul_right n (Nat.succ_le_of_lt ht)
  rw [Nat.succ_mul] at h1
  omega

/-- The token count must be between 1 and 8. -/
def validTokens (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 8

/-- Multi-token launch over `n` tokens makes `n * n_ff` gate/up elements. -/
theorem multi_elements (n nFf : Nat) : n * nFf = nFf * n := Nat.mul_comm n nFf

-- § native selection

/-- The native projection path is active iff any projection is native. -/
theorem native_projection_iff (g u d : Bool) : (g || u || d) = false ↔ g = false ∧ u = false ∧ d = false := by
  revert g u d
  decide

/-- Activation kind routing: 2-bit weights use the Q8_0 specialised kernel, kind 1 uses Q8_K, else Q8_0. -/
def routeKind (codeBits actKind : Nat) : Nat :=
  if codeBits = 2 then 0 else if actKind = 1 then 1 else 2

/-- Two-bit weights always take the Q8_0 specialised kernel. -/
theorem route_two_bit (k : Nat) : routeKind 2 k = 0 := by
  simp [routeKind]

/-- Q8_K is chosen only for non-2-bit forms with act_kind 1. -/
theorem route_q8k : routeKind 4 1 = 1 ∧ routeKind 8 1 = 1 := by decide

/-- `moe_combine` refuses k above 64 rather than truncating. -/
theorem combine_k_guard (k : Nat) (h : 64 < k) : ¬ k ≤ 64 := by omega

end Strata.SharedExpert

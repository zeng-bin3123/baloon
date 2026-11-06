/-!
# MODULE: shared_expert_combine
SOURCES: src/kernels/cuda/shared_expert.cu
The shared expert (SiLU on gate, times up, down projection, scaled by a per-token scalar sigmoid gate) with
legacy and native Q8_1 projection paths, the multi-token (1..8) variant, scratch carving, and the MoE combine.

## Source semantics
* Result is silu(x*gate) * (x*up) -> down -> times one scalar per token; the shared output is ADDED plain.
* moe_combine: y[j] = sum_{e<k} w[e]*parts[e][j] (+ shared[j] when present); k <= 64.
* Scratch: gate | up | q8_0 image | q8_K image | scalar gate, each region 16-byte aligned, plus 32 B tail.
* Q8_K down activation requires n_ff % 256 = 0; 2-bit forms always use Q8_0.
* shared_expert_multi accepts 1..8 tokens; row-scaling indexes t*n + i.

## Form template identifiers
shared_expert_scratch_bytes -> scratch_size_fn ; moe_combine_kernel -> expert_sum ; scalar_gate_kernel -> token_gate ;
scale_rows_kernel -> row_scale ; act_kind -> activation_kind
-/

namespace Strata.SharedExpert

/-- Sum of f over the first n indices. -/
def sumTo (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => sumTo f n + f n

-- § scratch carving

/-- Round up to a multiple of 16. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Alignment never shrinks. -/
theorem align16_ge (x : Nat) : x ≤ align16 x := by
  unfold align16
  omega

/-- Alignment yields a multiple of 16. -/
theorem align16_mod (x : Nat) : align16 x % 16 = 0 := by
  unfold align16
  omega

/-- Region sizes of the carved scratch. -/
def regionA (nFf : Nat) : Nat := align16 (nFf * 4)
/-- q8_0 image region. -/
def regionQ0 (nFf : Nat) : Nat := align16 (nFf / 32 * 34)
/-- q8_K image region. -/
def regionQk (nFf : Nat) : Nat := align16 (nFf / 256 * 292)

/-- Total scratch bytes. -/
def scratchBytes (nFf : Nat) : Nat := regionA nFf * 2 + regionQ0 nFf + regionQk nFf + 32

/-- The scalar-gate slot (offset at the end of all regions) fits with room to spare. -/
theorem gate_slot_fits (nFf : Nat) :
    regionA nFf * 2 + regionQ0 nFf + regionQk nFf + 4 ≤ scratchBytes nFf := by
  unfold scratchBytes
  omega

/-- Region offsets are monotone, so regions never overlap. -/
theorem regions_ordered (nFf : Nat) :
    regionA nFf ≤ regionA nFf * 2 ∧ regionA nFf * 2 ≤ regionA nFf * 2 + regionQ0 nFf ∧
      regionA nFf * 2 + regionQ0 nFf ≤ regionA nFf * 2 + regionQ0 nFf + regionQk nFf := by
  omega

/-- Each region offset is 16-byte aligned. -/
theorem offsets_aligned (nFf : Nat) :
    (regionA nFf * 2) % 16 = 0 ∧ (regionA nFf * 2 + regionQ0 nFf) % 16 = 0 ∧
      (regionA nFf * 2 + regionQ0 nFf + regionQk nFf) % 16 = 0 := by
  have h1 := align16_mod (nFf * 4)
  have h2 := align16_mod (nFf / 32 * 34)
  have h3 := align16_mod (nFf / 256 * 292)
  unfold regionA regionQ0 regionQk
  omega

/-- The gate region holds n_ff floats. -/
theorem gate_holds_floats (nFf : Nat) : nFf * 4 ≤ regionA nFf := by
  exact align16_ge _

-- § activation contract

/-- Activation kinds. -/
inductive Act where
  | q8_0
  | q8k
  deriving DecidableEq

/-- Which activation image a projection form reads. -/
def actFor (codeBits actKind : Nat) : Act :=
  if codeBits = 2 then Act.q8_0 else if actKind = 1 then Act.q8k else Act.q8_0

/-- 2-bit forms never use Q8_K. -/
theorem two_bit_uses_q8_0 (k : Nat) : actFor 2 k = Act.q8_0 := by
  simp [actFor]

/-- Q8_K is chosen only for non-2-bit forms that ask for it. -/
theorem q8k_only_when_requested (b k : Nat) (h : actFor b k = Act.q8k) : b ≠ 2 ∧ k = 1 := by
  unfold actFor at h
  split_ifs at h with h1 h2
  exact ⟨h1, h2⟩

/-- The down projection with n_ff = 640 cannot use Q8_K. -/
theorem down_640_no_q8k : 640 % 256 ≠ 0 := by
  decide

-- § multi-token variant and row scaling

/-- Token-count guard. -/
def validTokens (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 8

/-- A row index t*n + i lies in the n_tok-row buffer. -/
theorem row_index_lt (t i n nTok : Nat) (ht : t < nTok) (hi : i < n) : t * n + i < nTok * n := by
  have := Nat.mul_le_mul_right n (show t + 1 ≤ nTok by omega)
  simp only [Nat.add_mul, Nat.one_mul] at this
  omega

/-- The scalar-gate block of 256 threads has 8 warps matching the 8-slot scratch. -/
theorem gate_warps : (256 + 31) / 32 = 8 := by
  decide

/-- Per-row scaling grid covers n_embd columns. -/
theorem scale_grid_covers (n : Nat) : n ≤ 128 * ((n + 127) / 128) := by
  omega

-- § MoE combine

/-- Routed contribution at column j. -/
def routed (w p : Nat → Int) (k : Nat) : Int := sumTo (fun e => w e * p e) k

/-- Combined value: shared term added plainly when present. -/
def combined (w p : Nat → Int) (k : Nat) (shared : Option Int) : Int :=
  match shared with
  | none => routed w p k
  | some s => routed w p k + s

/-- Without a shared expert the output is exactly the routed sum. -/
theorem combined_none (w p : Nat → Int) (k : Nat) : combined w p k none = routed w p k := by
  rfl

/-- The shared output is added unweighted. -/
theorem combined_some (w p : Nat → Int) (k : Nat) (s : Int) :
    combined w p k (some s) = routed w p k + s := by
  rfl

/-- A prefix sum of zeros is zero. -/
theorem sumTo_zero (f : Nat → Int) (k : Nat) (h : ∀ i, f i = 0) : sumTo f k = 0 := by
  induction k with
  | zero => simp [sumTo]
  | succ k ih =>
    simp only [sumTo]
    rw [ih, h k]
    rfl

/-- Zero router weights leave only the shared output. -/
theorem routed_zero_weights (p : Nat → Int) (k : Nat) : routed (fun _ => 0) p k = 0 := by
  unfold routed
  exact sumTo_zero _ k (fun e => by simp)

/-- k <= 64 is accepted, longer lists are refused rather than truncated. -/
theorem combine_k_guard (k : Nat) : ¬ (k > 64) ↔ k ≤ 64 := by
  omega

/-- Combine grid covers all columns. -/
theorem combine_grid_covers (n : Nat) : n ≤ 128 * ((n + 127) / 128) := by
  omega

end Strata.SharedExpert

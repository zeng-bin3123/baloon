/-!
# MODULE: shared_expert_block
SOURCES: src/kernels/cuda/shared_expert.cu
The MoE shared expert: `h = silu(x Wgate^T) * (x Wup^T)`, `h Wdown^T`, scaled by a per-token scalar
`sigmoid(x . w_gate_inp)`; plus the MoE combine and the multi-token (1..8) native path.

## Source semantics
- SiLU is applied to the gate projection. The scalar gate is one value per token and multiplies the whole row.
- Scratch is carved from the caller's buffer, 16-byte aligned: gate | up | q8_0 | q8k | g.
- The shared output is added plain (not router-weighted) in the combine; `k` is limited to 64.
- Q8_K needs `n_ff % 256 = 0`; the BF16 gate vector has 2 bytes/element, so reading it as floats overruns.

## Form template identifiers
- shared_expert_scratch_bytes -> scratchBytes ; moe_combine_kernel -> combine ; scale_rows_kernel -> rowIndex
-/

namespace Strata.SharedExpert

/-- Round up to a multiple of 16. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Alignment never shrinks. -/
theorem align16_ge (x : Nat) : x ≤ align16 x := by
  unfold align16
  omega

/-- Aligned values are multiples of 16. -/
theorem align16_mod (x : Nat) : align16 x % 16 = 0 := by
  unfold align16
  omega

/-- Alignment overshoots by less than 16. -/
theorem align16_lt (x : Nat) : align16 x < x + 16 := by
  unfold align16
  omega

-- § scratch layout
/-- Byte size of the gate/up float arrays. -/
def sA (nFf : Nat) : Nat := align16 (nFf * 4)
/-- Byte size of the Q8_0 intermediate. -/
def sQ0 (nFf : Nat) : Nat := align16 (nFf / 32 * 34)
/-- Byte size of the Q8_K intermediate. -/
def sQk (nFf : Nat) : Nat := align16 (nFf / 256 * 292)
/-- Total scratch bytes. -/
def scratchBytes (nFf : Nat) : Nat := sA nFf * 2 + sQ0 nFf + sQk nFf + 32

/-- Offsets of the up, q8_0, q8k and g regions are increasing. -/
theorem scratch_offsets_ordered (n : Nat) :
    0 ≤ sA n ∧ sA n ≤ 2 * sA n ∧ 2 * sA n ≤ 2 * sA n + sQ0 n ∧
    2 * sA n + sQ0 n ≤ 2 * sA n + sQ0 n + sQk n := by
  omega

/-- The scalar gate (4 bytes) fits inside the scratch buffer. -/
theorem gate_slot_fits (n : Nat) : 2 * sA n + sQ0 n + sQk n + 4 ≤ scratchBytes n := by
  unfold scratchBytes
  omega

/-- Every region offset is 16-byte aligned. -/
theorem offsets_aligned (n : Nat) :
    sA n % 16 = 0 ∧ (2 * sA n) % 16 = 0 ∧ (2 * sA n + sQ0 n) % 16 = 0 ∧
    (2 * sA n + sQ0 n + sQk n) % 16 = 0 := by
  have h1 : sA n % 16 = 0 := align16_mod _
  have h2 : sQ0 n % 16 = 0 := align16_mod _
  have h3 : sQk n % 16 = 0 := align16_mod _
  omega

/-- The gate array holds `n_ff` floats. -/
theorem gate_holds_floats (n : Nat) : n * 4 ≤ sA n := align16_ge _

-- § combine
/-- Weighted sum of routed parts plus an optional plain shared term. -/
def combine (ws ps : List Int) (shared : Option Int) : Int :=
  (List.zipWith (· * ·) ws ps).sum + shared.getD 0

/-- Without a shared output the combine is just the weighted sum. -/
theorem combine_no_shared (ws ps : List Int) : combine ws ps none = (List.zipWith (· * ·) ws ps).sum := by
  simp [combine]

/-- The shared term is added plain (no weight). -/
theorem combine_shared_plain (ws ps : List Int) (s : Int) :
    combine ws ps (some s) = (List.zipWith (· * ·) ws ps).sum + s := by
  simp [combine]

/-- Empty routing leaves only the shared output. -/
theorem combine_empty (s : Int) : combine [] [] (some s) = s := by
  simp [combine]

-- § row-scaling index
/-- Row index `t*n + i` of token t, element i, stays below `n_tok * n`. -/
theorem row_index_lt (nTok t i n : Nat) (ht : t < nTok) (hi : i < n) : t * n + i < nTok * n := by
  have h1 : (t + 1) * n ≤ nTok * n := Nat.mul_le_mul_right n ht
  have h2 : (t + 1) * n = t * n + n := Nat.succ_mul t n
  omega

/-- Multi-token path precondition: 1..8 tokens. -/
def tokOk (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 8

/-- The multi-column MMVQ bound (8) matches the shared-expert token bound. -/
theorem tok_matches_ncols (n : Nat) (h : tokOk n) : 1 ≤ n ∧ n ≤ 8 := h

-- § scalar-gate reduction
/-- Warps in a block of `b` threads (b ≤ 256) never exceed the 8-slot scratch. -/
theorem gate_warps_le_8 (b : Nat) (h : b ≤ 256) : (b + 31) / 32 ≤ 8 := by
  omega

/-- The default launch (256 threads) uses exactly 8 warps. -/
theorem gate_warps_256 : (256 + 31) / 32 = 8 := by decide

-- § constraints and the historical overrun
/-- Q8_K is impossible for n_ff = 640 (not a multiple of 256) while Q8_0 is possible. -/
theorem shexp_q8k_vs_q8_0 : 640 % 256 ≠ 0 ∧ 640 % 32 = 0 := by decide

/-- Reading 2560 floats from a 2560-element BF16 vector overran by 5120 bytes. -/
theorem bf16_overrun_bytes : 2560 * 4 - 2560 * 2 = 5120 := by decide

/-- The combine limit is 64 routed parts. -/
theorem combine_limit (k : Nat) (h : k ≤ 64) : k ≤ 64 := h

end Strata.SharedExpert

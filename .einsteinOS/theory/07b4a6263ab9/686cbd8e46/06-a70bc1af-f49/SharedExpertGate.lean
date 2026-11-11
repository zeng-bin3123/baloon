/-!
# MODULE: shared_expert_gate
SOURCES: src/kernels/cuda/shared_expert.cu
Model of the P2.S2 shared expert and the MoE combination: SwiGLU orientation (SiLU on the
gate), the per-token scalar sigmoid gate, the caller-owned scratch carve-out, the native
projection dispatch preconditions and `moe_combine`'s plain addition of the shared output.

## Source semantics
* `h = silu(x·Wg) * (x·Wu)`; `h = h·Wd`; `g = sigmoid(x · w_gate_inp)` is ONE scalar per token
  computed from the original `x`; the result is `h * g`.
* The shared output is added unweighted to the router-weighted routed sum; `k ≤ 64`.
* Scratch = gate | up | q8_0 | q8_K | g, each region 16-byte aligned, carved, never allocated.
* Any native gate/up projection (or the native BF16 gate) requires the unrounded `x_f32`;
  `shared_expert_multi` requires 1..8 tokens.
* The scalar gate reduces 2560 terms over 256 threads: a 10-term chain per thread, then a tree.

## Form template identifiers
`swiglu_kernel` → `swiglu`; `scalar_gate_kernel` → `scalarGate`; `shared_expert_scratch_bytes` →
`scratchBytes`/`offsets`; `moe_combine_kernel` → `combine`; native flags → `Dispatch`;
`scale_rows_kernel` index → `rowIndex`.
-/

namespace SharedExpertGate

-- § SwiGLU orientation

/-- `swiglu(gate, up) = silu(gate) * up` for an abstract activation `silu`. -/
def swiglu {α : Type} [Mul α] (silu : α → α) (g u : α) : α := silu g * u

/-- The orientation matters: with a non-linear activation, swapping gate and up changes the
    value (here with `x ↦ x²`, `swiglu 2 3 = 12` but `swiglu 3 2 = 18`).  The swapped reading
    is the plausible-but-wrong one recorded in `docs/semantics.md`. -/
theorem swiglu_not_symmetric :
    swiglu (fun x : Int => x * x) 2 3 ≠ swiglu (fun x : Int => x * x) 3 2 := by decide

/-- The SwiGLU is linear in `up`. -/
theorem swiglu_linear_up (silu : Int → Int) (g u v : Int) :
    swiglu silu g (u + v) = swiglu silu g u + swiglu silu g v := by
  unfold swiglu; exact Int.mul_add _ _ _

-- § scalar gate reduction

/-- Terms per thread of a strided loop over `n` elements with `t` threads (thread `i` takes
    `i, i+t, …`). -/
def termsPerThread (n t i : Nat) : Nat := (n + t - 1 - i) / t

/-- For `n_embd = 2560` and 256 threads every thread's dependency chain is 10 terms long,
    instead of the 2560-long single-thread chain the source replaced. -/
theorem gate_chain_length : ∀ i, i < 256 → termsPerThread 2560 256 i = 10 := by
  intro i h; unfold termsPerThread; omega

/-- The 256 per-thread chains partition the 2560 terms: element `e` belongs to thread `e % 256`
    as its `(e / 256)`-th term. -/
theorem gate_partition (e : Nat) (h : e < 2560) :
    e % 256 < 256 ∧ e / 256 < 10 ∧ e % 256 + 256 * (e / 256) = e := by omega

/-- The block reduction uses `(256 + 31) / 32 = 8` warp partials, matching `scratch[8]`. -/
theorem gate_warps : (256 + 31) / 32 = 8 := by decide

/-- Per-token gate: `g = sigmoid(Σ x_i w_i)`; the multiply is applied to every output element
    of that token: element `i` of token `t` sits at `t * n + i`. -/
def rowIndex (n t i : Nat) : Nat := t * n + i

/-- `scale_rows_kernel` touches each (token, element) exactly once. -/
theorem rowIndex_injective {n t1 t2 i1 i2 : Nat} (h1 : i1 < n) (h2 : i2 < n)
    (e : rowIndex n t1 i1 = rowIndex n t2 i2) : t1 = t2 ∧ i1 = i2 := by
  unfold rowIndex at e
  have hn : 0 < n := by omega
  have d1 : (i1 + n * t1) / n = t1 := by
    rw [Nat.add_mul_div_left _ _ hn, Nat.div_eq_of_lt h1, Nat.zero_add]
  have d2 : (i2 + n * t2) / n = t2 := by
    rw [Nat.add_mul_div_left _ _ hn, Nat.div_eq_of_lt h2, Nat.zero_add]
  have e' : i1 + n * t1 = i2 + n * t2 := by
    rw [Nat.mul_comm n t1, Nat.mul_comm n t2]; omega
  have ht : t1 = t2 := by rw [← d1, ← d2, e']
  subst ht
  exact ⟨rfl, by omega⟩

-- § scratch

/-- 16-byte round-up `(x + 15) & ~15`. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Q8_K block: fp32 `d`, 256 int8 codes, 16 int16 block sums. -/
theorem q8k_block_bytes : 4 + 256 + 16 * 2 = 292 := by decide

/-- `shared_expert_scratch_bytes`. -/
def scratchBytes (nFF : Nat) : Nat :=
  2 * align16 (nFF * 4) + align16 (nFF / 32 * 34) + align16 (nFF / 256 * 292) + 32

/-- Region starts: gate, up, q8_0, q8_K, g. -/
def offsets (nFF : Nat) : List Nat :=
  [0, align16 (nFF * 4), 2 * align16 (nFF * 4), 2 * align16 (nFF * 4) + align16 (nFF / 32 * 34),
   2 * align16 (nFF * 4) + align16 (nFF / 32 * 34) + align16 (nFF / 256 * 292)]

/-- Each carved region holds its payload before the next starts, every start is 16-byte
    aligned, and the 4-byte scalar `g` fits in the trailing 32 bytes. -/
theorem carve_fits (nFF : Nat) :
    nFF * 4 ≤ align16 (nFF * 4) ∧ nFF / 32 * 34 ≤ align16 (nFF / 32 * 34) ∧
    nFF / 256 * 292 ≤ align16 (nFF / 256 * 292) ∧
    2 * align16 (nFF * 4) + align16 (nFF / 32 * 34) + align16 (nFF / 256 * 292) + 4 ≤
      scratchBytes nFF ∧
    (∀ o, o ∈ offsets nFF → o % 16 = 0) := by
  refine ⟨by unfold align16; omega, by unfold align16; omega, by unfold align16; omega,
    by unfold scratchBytes; omega, ?_⟩
  intro o ho
  simp only [offsets, List.mem_cons, List.not_mem_nil, or_false] at ho
  unfold align16 at ho
  rcases ho with h | h | h | h | h <;> subst h <;> omega

/-- Q8_K for the down weight needs `n_ff % 256 = 0`; when it holds the q8_K region holds every
    super-block. -/
theorem q8k_whole (k : Nat) : (k * 256) / 256 * 292 = k * 292 := by
  rw [Nat.mul_div_cancel _ (by decide)]

-- § dispatch preconditions

/-- Which pieces run natively. -/
structure Dispatch where
  nativeBF16 : Bool
  gate : Bool
  up : Bool
  down : Bool
  hasF32 : Bool

/-- `shared_expert` accepts the call (does not throw on the `x_f32` check). -/
def accepts (d : Dispatch) : Bool := !((d.nativeBF16 || d.gate || d.up) && !d.hasF32)

/-- A native gate or up projection, or the native BF16 scalar gate, is only accepted with the
    unrounded F32 input; a native DOWN projection alone does not need it (it reads `gate`). -/
theorem accepts_iff (d : Dispatch) :
    accepts d = true ↔ ((d.nativeBF16 = true ∨ d.gate = true ∨ d.up = true) → d.hasF32 = true) := by
  cases d with
  | mk a b c e f => cases a <;> cases b <;> cases c <;> cases f <;> simp [accepts]

/-- `shared_expert_multi`'s token count guard. -/
def multiTokensOk (n : Int) : Bool := 1 ≤ n && n ≤ 8

/-- Exactly 1..8 tokens are accepted, matching the multi-column MMVQ limit. -/
theorem multiTokensOk_iff (n : Int) : multiTokensOk n = true ↔ 1 ≤ n ∧ n ≤ 8 := by
  simp [multiTokensOk]

-- § MoE combination

/-- `moe_combine_kernel` for one output column: `acc = 0; acc += w_e * p_e …; acc += shared`. -/
def combine (w p : List Int) (shared : Option Int) : Int :=
  (List.zipWith (· * ·) w p).foldl (· + ·) 0 + shared.getD 0

/-- Left fold sum starting from an accumulator. -/
theorem foldl_add_start (l : List Int) (a : Int) : l.foldl (· + ·) a = a + l.foldl (· + ·) 0 := by
  induction l generalizing a with
  | nil => simp
  | cons x xs ih =>
    simp only [List.foldl]
    rw [ih (a + x), ih (0 + x)]
    omega

/-- The shared output is added plain: the combination with a shared expert is the routed
    combination plus the shared value, with no router weight or renormalisation applied to it. -/
theorem combine_shared_plain (w p : List Int) (s : Int) :
    combine w p (some s) = combine w p none + s := by
  simp [combine]

/-- With one routed expert and no shared expert the combination is `w₀ · p₀`. -/
theorem combine_single (w0 p0 : Int) : combine [w0] [p0] none = w0 * p0 := by
  simp [combine]

/-- `moe_combine` refuses `k > 64` rather than truncating. -/
def combineAccepts (k : Nat) : Bool := k ≤ 64

end SharedExpertGate

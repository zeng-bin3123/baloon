/-!
# MODULE: shared_expert_block
SOURCES: src/kernels/cuda/shared_expert.cu
The MoE shared expert (silu on gate, times up, down projection, scaled by a per-token scalar sigmoid gate) and
the final routed-plus-shared combination.

## Source semantics
h = silu(x Wg) * (x Wu); out = (h Wd) * sigmoid(x . w_gate_inp). The routed output is sum_e w_e * part_e plus the
shared output added plain (not router-weighted). The down input is quantized to the down weight's own activation
format: Q8_K needs n_ff % 256 = 0, otherwise Q8_0. Scratch is carved from caller memory in 16-byte aligned
regions: gate | up | q8_0 | q8k | g. Native paths take 1..8 tokens; moe_combine refuses k > 64.

## Form template identifiers
align16 -> "scratch alignment"; scratchBytes -> "shared expert scratch"; dotAcc -> "routed weighted sum".
-/
namespace Strata.SharedExpert

-- § scratch carving
/-- Round up to a multiple of 16. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Alignment is a multiple of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by
  unfold align16
  omega

/-- Alignment never shrinks. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by
  unfold align16
  omega

/-- `shared_expert_scratch_bytes`. -/
def scratchBytes (nFf : Nat) : Nat :=
  align16 (nFf * 4) * 2 + align16 (nFf / 32 * 34) + align16 (nFf / 256 * 292) + 32

/-- Region offsets in carving order. -/
def offUp (nFf : Nat) : Nat := align16 (nFf * 4)
/-- Offset of the Q8_0 image. -/
def offQ80 (nFf : Nat) : Nat := align16 (nFf * 4) * 2
/-- Offset of the Q8_K image. -/
def offQ8k (nFf : Nat) : Nat := align16 (nFf * 4) * 2 + align16 (nFf / 32 * 34)
/-- Offset of the scalar gate. -/
def offG (nFf : Nat) : Nat := align16 (nFf * 4) * 2 + align16 (nFf / 32 * 34) + align16 (nFf / 256 * 292)

/-- Every carved region starts 16-byte aligned. -/
theorem offsets_aligned (nFf : Nat) :
    offUp nFf % 16 = 0 ∧ offQ80 nFf % 16 = 0 ∧ offQ8k nFf % 16 = 0 ∧ offG nFf % 16 = 0 := by
  unfold offUp offQ80 offQ8k offG align16
  omega

/-- The scalar gate slot plus its 32-byte tail ends exactly at the scratch size. -/
theorem scratch_end (nFf : Nat) : offG nFf + 32 = scratchBytes nFf := by
  unfold offG scratchBytes
  rfl

/-- For n_ff = 640 the scratch is 6432 bytes. -/
theorem scratch_640 : scratchBytes 640 = 6432 := by
  decide

/-- n_ff = 640 is a multiple of 32 but not 256: Q8_K is structurally impossible for the down input. -/
theorem down_q8k_impossible_640 : 640 % 32 = 0 ∧ 640 % 256 ≠ 0 := by
  decide

-- § native path guards
/-- Token-count guard of `shared_expert_multi`. -/
def tokensOk (n : Nat) : Bool := decide (1 ≤ n ∧ n ≤ 8)

/-- Valid token counts are exactly 1..8. -/
theorem tokensOk_iff (n : Nat) : tokensOk n = true ↔ (1 ≤ n ∧ n ≤ 8) := by
  unfold tokensOk
  simp

/-- A native projection is used when any of gate/up/down is natively supported. -/
def nativeProjection (g u d : Bool) : Bool := g || u || d

/-- Without any native projection the legacy SwiGLU is used. -/
theorem legacy_when_none : nativeProjection false false false = false := by
  decide

-- § combination
/-- Weighted accumulation `acc += w_e * part_e` in order. -/
def dotAcc : Int → List Int → List Int → Int
  | acc, [], _ => acc
  | acc, _ :: _, [] => acc
  | acc, w :: ws, p :: ps => dotAcc (acc + w * p) ws ps

/-- Starting accumulator offset passes straight through: this is why the shared output is added plain. -/
theorem dotAcc_shift (ws ps : List Int) (a c : Int) : dotAcc (a + c) ws ps = dotAcc a ws ps + c := by
  induction ws generalizing ps a with
  | nil => simp [dotAcc]
  | cons w ws ih =>
    cases ps with
    | nil => simp [dotAcc]
    | cons p ps =>
      have e : a + c + w * p = a + w * p + c := by omega
      simp only [dotAcc]
      rw [e]
      exact ih ps (a + w * p)

/-- Final combination: routed sum plus the shared output. -/
def combine (ws ps : List Int) (shared : Int) : Int := dotAcc shared ws ps

/-- The combination equals the routed sum plus the shared output. -/
theorem combine_eq (ws ps : List Int) (shared : Int) : combine ws ps shared = dotAcc 0 ws ps + shared := by
  unfold combine
  have h := dotAcc_shift ws ps 0 shared
  rw [Int.zero_add] at h
  exact h

/-- With no routed experts the output is exactly the shared output. -/
theorem combine_nil (ps : List Int) (shared : Int) : combine [] ps shared = shared := by
  simp [combine, dotAcc]

/-- The k <= 64 guard. -/
def kOk (k : Nat) : Bool := decide (0 < k ∧ k ≤ 64)

/-- Ten routed experts pass the guard; 65 is refused. -/
theorem kOk_cases : kOk 10 = true ∧ kOk 65 = false := by
  decide

end Strata.SharedExpert

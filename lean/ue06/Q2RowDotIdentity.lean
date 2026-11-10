/-!
# MODULE: q2_row_dot_identity
SOURCES: include/ref/kernels/cpu/expert.hpp, src/kernels/cpu/expert.cpp, src/kernels/cpu/q2_avx2.cpp
The algebra behind the Q2_0 row dot: codes {0..3} stand for symbols {-1,0,1,2}; the VNNI path computes
sum(c*xhat) and subtracts sum(xhat) (the `hx` correction) instead of forming (c-1).

## Source semantics
* sum (c-1) x = sum c x - sum x (integer-exact, which is why the correction can be factored out).
* One weight scale covers two activation chunks with different scales; collapsing to one scale is wrong.
* Unpack: code i of a packed byte stream is (b / 4^i) % 4; multishift lanes keep codes in order.
* Multi-token kernels process each token independently; tokens are dispatched with NT = n (1..8) and the
  AVX2 path in chunks of at most 4.

## Form template identifiers
dotSym -> symbol_dot, dotC -> code_dot, sumX -> activation_sum, ntDispatch -> token_template_select,
chunks4 -> avx2_token_chunks
-/

namespace Ref.Q2RowDot

-- § dot products over (code, activation) pairs
/-- Dot product with the symbol mapping c -> c - 1. -/
def dotSym : List (Nat × Int) → Int
  | [] => 0
  | (c, x) :: l => ((c : Int) - 1) * x + dotSym l

/-- Dot product with raw codes. -/
def dotC : List (Nat × Int) → Int
  | [] => 0
  | (c, x) :: l => (c : Int) * x + dotC l

/-- Sum of the activations. -/
def sumX : List (Nat × Int) → Int
  | [] => 0
  | (_, x) :: l => x + sumX l

/-- The Q2_0 identity: sum (c-1) x = sum c x - sum x. -/
theorem dotSym_eq_dotC_sub_sumX : ∀ l : List (Nat × Int), dotSym l = dotC l - sumX l
  | [] => by simp [dotSym, dotC, sumX]
  | (c, x) :: l => by
      have ih := dotSym_eq_dotC_sub_sumX l
      simp only [dotSym, dotC, sumX, ih, Int.sub_mul, Int.one_mul] <;> omega

/-- The symbol dot product splits over concatenation (two chunks of a block). -/
theorem dotSym_append : ∀ l₁ l₂ : List (Nat × Int), dotSym (l₁ ++ l₂) = dotSym l₁ + dotSym l₂
  | [], l₂ => by simp [dotSym]
  | (c, x) :: l₁, l₂ => by
      have ih := dotSym_append l₁ l₂
      simp only [List.cons_append, dotSym, ih] <;> omega

/-- A zero code contributes nothing, so zero-padded lanes of a narrow load are harmless. -/
theorem zero_code_contributes_nothing (x : Int) : (0 : Int) * x = 0 := Int.zero_mul x

-- § two scales per weight block
/-- Per-chunk scaling distributes over the weight-block scale. -/
theorem block_two_scale (dw s0 s1 A0 A1 : Int) :
    dw * (s0 * A0) + dw * (s1 * A1) = dw * (s0 * A0 + s1 * A1) := by
  rw [Int.mul_add]

/-- Using one chunk scale for the whole block agrees with the correct formula only if (s1-s0)*A1 = 0. -/
theorem single_scale_iff (s0 s1 A0 A1 : Int) :
    s0 * A0 + s1 * A1 = s0 * (A0 + A1) ↔ (s1 - s0) * A1 = 0 := by
  constructor
  · intro h
    rw [Int.mul_add] at h
    rw [Int.sub_mul]
    omega
  · intro h
    rw [Int.sub_mul] at h
    rw [Int.mul_add]
    omega

/-- A concrete case where the single-scale shortcut gives a wrong answer. -/
theorem single_scale_counterexample : ¬ ((1 : Int) * 1 + 2 * 1 = 1 * (1 + 1)) := by decide

-- § unpack and VNNI bounds
/-- Pack four 2-bit codes into a byte, LSB first. -/
def pack4 (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3
/-- Code 0 of a byte. -/
def code0 (b : Nat) : Nat := b % 4
/-- Code 1 of a byte. -/
def code1 (b : Nat) : Nat := b / 4 % 4
/-- Code 2 of a byte. -/
def code2 (b : Nat) : Nat := b / 16 % 4
/-- Code 3 of a byte. -/
def code3 (b : Nat) : Nat := b / 64 % 4

/-- Unpacking a packed byte returns the four codes in order. -/
theorem unpack_pack (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    code0 (pack4 c0 c1 c2 c3) = c0 ∧ code1 (pack4 c0 c1 c2 c3) = c1 ∧
      code2 (pack4 c0 c1 c2 c3) = c2 ∧ code3 (pack4 c0 c1 c2 c3) = c3 := by
  unfold code0 code1 code2 code3 pack4
  refine ⟨by omega, by omega, by omega, by omega⟩

/-- A packed byte is below 256 when its codes are 2-bit. -/
theorem pack4_lt (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    pack4 c0 c1 c2 c3 < 256 := by
  unfold pack4
  omega

/-- Across the two bytes of a 16-bit lane, position 4 is the first code of the second byte. -/
theorem lane_second_byte (b0 b1 : Nat) (h : b0 < 256) : (b0 + 256 * b1) / 256 % 4 = b1 % 4 := by
  omega

/-- A u8 code times an int8 activation stays within +-381. -/
theorem code_mul_bound (c x : Int) (hc0 : 0 ≤ c) (hc3 : c ≤ 3) (hx1 : -127 ≤ x) (hx2 : x ≤ 127) :
    -381 ≤ c * x ∧ c * x ≤ 381 := by
  have h : c = 0 ∨ c = 1 ∨ c = 2 ∨ c = 3 := by omega
  rcases h with rfl | rfl | rfl | rfl <;> omega

/-- A 2560-wide dot of such products fits in int32. -/
theorem vnni_sum_fits : 2560 * 381 = 975360 ∧ 975360 < 2 ^ 31 := by decide

/-- Broadcasting the scalar correction to eight lanes would apply it eight times; that equals one
application only when the correction is zero. -/
theorem eight_lane_correction_wrong (c : Int) (h : 8 * c = c) : c = 0 := by omega

-- § multi-token dispatch
/-- Template argument chosen for `n` tokens: n for 1..7, otherwise 8. -/
def ntDispatch (n : Nat) : Nat := if 1 ≤ n ∧ n ≤ 7 then n else 8

/-- For 1..8 tokens the selected template handles exactly `n` tokens. -/
theorem ntDispatch_exact (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ 8) : ntDispatch n = n := by
  unfold ntDispatch
  split <;> omega

/-- Token chunk sizes for the AVX2 path (at most 4 at a time), with fuel. -/
def chunks4 : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, r => if r = 0 then [] else min 4 r :: chunks4 f (r - min 4 r)

/-- With enough fuel the chunks sum to the token count. -/
theorem chunks4_sum : ∀ f r : Nat, r ≤ 4 * f → (chunks4 f r).sum = r
  | 0, r, h => by
      have : r = 0 := by omega
      subst this
      simp [chunks4]
  | f + 1, r, h => by
      by_cases hr : r = 0
      · subst hr
        simp [chunks4]
      · have ih := chunks4_sum f (r - min 4 r) (by omega)
        simp [chunks4, hr, ih] <;> omega

/-- Every AVX2 chunk is between 1 and 4 tokens. -/
theorem chunks4_bounded : ∀ f r : Nat, ∀ k ∈ chunks4 f r, 1 ≤ k ∧ k ≤ 4
  | 0, r, k, hk => by simp [chunks4] at hk
  | f + 1, r, k, hk => by
      by_cases hr : r = 0
      · subst hr
        simp [chunks4] at hk
      · simp only [chunks4, hr, if_false, List.mem_cons] at hk
        rcases hk with rfl | hk
        · omega
        · exact chunks4_bounded f _ k hk

/-- Processing tokens in chunks equals processing them all at once, token by token. -/
theorem chunked_map_eq {α β : Type} (f : α → β) (a b : List α) :
    (a ++ b).map f = a.map f ++ b.map f := List.map_append

end Ref.Q2RowDot

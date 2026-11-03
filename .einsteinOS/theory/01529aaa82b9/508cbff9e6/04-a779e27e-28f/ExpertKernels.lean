/-!
# MODULE: expert_kernels
Arithmetic contracts of the CPU Q2_0 expert kernel (`expert.cpp`).

## Source semantics
* The Q2_0 identity: `sum (c-1) * x = sum c*x - sum x`, which lets the kernel dot raw 2-bit codes
  and subtract the weight-independent correction (`hx`).
* A 64-weight block is two 32-element chunks, so a block dot splits additively.
* Activation quantization clamps codes to `[-127, 127]`.
* The multi-token entry dispatches on token count `1..8` to the template of the same width.
* Gate and up rows are `2r` and `2r+1`.

## Form template identifiers
* `row_dot` → `dotInt` (integer part); `hx` → `sumList`; `act_quant_q8_1` clamp → `clamp127`
* `s2_expert_vnni_multi` switch → `ntSel`
-/

namespace StrataExpertKernels

/-- Integer dot product of two lists (truncating to the shorter). -/
def dotInt : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dotInt as bs
  | _, _ => 0

/-- Sum of a list. -/
def sumList : List Int → Int
  | [] => 0
  | a :: as => a + sumList as

-- § Q2_0 identity

/-- Block dots split additively over the two activation chunks. -/
theorem dotInt_append (a b c d : List Int) (h : a.length = c.length) :
    dotInt (a ++ b) (c ++ d) = dotInt a c + dotInt b d := by
  induction a generalizing c with
  | nil =>
    cases c with
    | nil => simp [dotInt]
    | cons _ _ => simp at h
  | cons x xs ih =>
    cases c with
    | nil => simp at h
    | cons y ys =>
      have h' : xs.length = ys.length := by simpa using h
      simp [dotInt, ih ys h', Int.add_assoc]

/-- The Q2_0 identity: `sum (c-1)·x = sum c·x − sum x`. -/
theorem offset_dot (c x : List Int) (h : c.length = x.length) :
    dotInt (c.map (fun v => v - 1)) x = dotInt c x - sumList x := by
  induction c generalizing x with
  | nil =>
    cases x with
    | nil => simp [dotInt, sumList]
    | cons _ _ => simp at h
  | cons a as ih =>
    cases x with
    | nil => simp at h
    | cons b bs =>
      have h' : as.length = bs.length := by simpa using h
      simp only [List.map, dotInt, sumList]
      rw [ih bs h', Int.sub_mul]
      omega

-- § quantization

/-- Clamp of an activation code. -/
def clamp127 (q : Int) : Int := max (-127) (min 127 q)

/-- Quantized activation codes lie in `[-127, 127]`. -/
theorem clamp127_bounds (q : Int) : -127 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  unfold clamp127; omega

/-- Codes already in range are unchanged. -/
theorem clamp127_id (q : Int) (h1 : -127 ≤ q) (h2 : q ≤ 127) : clamp127 q = q := by
  unfold clamp127; omega

/-- Clamping is idempotent. -/
theorem clamp127_idem (q : Int) : clamp127 (clamp127 q) = clamp127 q := by
  unfold clamp127; omega

/-- A 32-wide chunk bound: 32 codes of magnitude ≤ 127 sum to at most 4064. -/
theorem chunk_sum_bound (n : Nat) (h : n ≤ 32) (s : Int) (hs : s ≤ 127 * (n : Int)) : s ≤ 4064 := by
  omega

-- § token dispatch

/-- Template width selected by the `switch (n_tokens)`; the default case is 8. -/
def ntSel : Nat → Nat
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 4
  | 5 => 5
  | 6 => 6
  | 7 => 7
  | _ => 8

/-- For `1 ≤ n ≤ 8` the selected template width equals the token count. -/
theorem ntSel_eq (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ 8) : ntSel n = n := by
  have h : n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 := by omega
  rcases h with h | h | h | h | h | h | h | h <;> subst h <;> rfl

/-- The selected width never exceeds `MAXT = 8`. -/
theorem ntSel_le (n : Nat) : ntSel n ≤ 8 := by
  unfold ntSel
  split <;> omega

-- § row addressing

/-- Row `r` of the intermediate produces gate row `2r` and up row `2r+1`, both below `2*FF`. -/
theorem gate_up_rows_lt (r : Nat) (h : r < 640) : 2 * r < 1280 ∧ 2 * r + 1 < 1280 := by omega

/-- Gate/up row split for `run_split`: the part ranges of `FF = 640` rows are inside `[0, 640]`. -/
theorem part_range_le (parts part : Nat) (hp : 0 < parts) (hq : part ≤ parts) :
    640 * part / parts ≤ 640 := by
  have h : 640 * part ≤ 640 * parts := Nat.mul_le_mul_left 640 hq
  have h2 : 640 * part / parts ≤ 640 * parts / parts := Nat.div_le_div_right h
  rw [Nat.mul_div_cancel 640 hp] at h2
  exact h2

end StrataExpertKernels

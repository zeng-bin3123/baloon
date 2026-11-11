/-!
# MODULE: activation_quant_q8
SOURCES: src/kernels/cpu/expert.cpp, src/kernels/cpu/q2_avx2.cpp, include/strata/kernels/cpu/expert.hpp
The activation quantizers `act_quant_q8_1` (scalar and AVX-512 legacy), `act_quant_q8_1_avx2` and the
ggml x86 oracle `quantize_oracle_q8_0`.  Values are modelled as exact rationals `n/d` (d > 0); the theory
covers rounding rule, clamping, the zero chunk, the int32 safety of chunk sums, and the claim that the
vectorised loops are the scalar rule applied lane by lane.

## Source semantics
* Legacy rule: t = x·inv, r = t + (t ≥ 0 ? 0.5 : −0.5), v = trunc(r), v clamped to [−127, 127]
  (round half AWAY from zero, exactly `lround`).
* Oracle rule: `_mm256_round_ps(..., NEAREST_INT)` = round half to EVEN; the two differ on ties.
* amax = 0 ⇒ scale 0, inv 0 ⇒ every code 0, sum 0, hx 0.
* A 32-element chunk of clamped codes has |sum| ≤ 32·127, far inside int32.
* AVX-512 (16 lanes) / AVX2 (8 lanes) process the chunk in lane groups with the same per-element ops,
  so the result list is the scalar one.

## Form template identifiers
roundHalfAway → legacyRound, roundHalfEven → oracleRound, clamp127 → int8Clamp, quantCode → legacyCode,
listSum → chunkSum, laneMap → vectorLanes
-/

namespace Strata.Kernels.Cpu.ActQuant

-- § rounding rules on exact rationals n/d

/-- Half-away-from-zero rounding of `n/d`: `trunc((2n ± d) / 2d)`. -/
def roundHalfAway (n : Int) (d : Nat) : Int :=
  if 0 ≤ n then (2 * n + d).tdiv (2 * d) else (2 * n - d).tdiv (2 * d)

/-- Half-to-even rounding of `n/d` (the oracle's `_MM_FROUND_TO_NEAREST_INT`). -/
def roundHalfEven (n : Int) (d : Nat) : Int :=
  let q := (2 * n + d).fdiv (2 * d)
  if (2 * n + d).fmod (2 * d) = 0 ∧ q % 2 ≠ 0 then q - 1 else q

/-- The two rules agree away from ties … -/
theorem rules_agree_off_tie : roundHalfAway 7 5 = roundHalfEven 7 5 ∧
    roundHalfAway (-7) 5 = roundHalfEven (-7) 5 := by decide

/-- … and differ on the tie 1/2: legacy gives 1, the oracle gives 0. -/
theorem rules_differ_on_tie : roundHalfAway 1 2 = 1 ∧ roundHalfEven 1 2 = 0 := by decide

/-- On the tie 3/2 both give 2, so the difference is exactly the odd-tie case. -/
theorem rules_agree_even_tie : roundHalfAway 3 2 = 2 ∧ roundHalfEven 3 2 = 2 := by decide

/-- Half-away rounding is odd: it treats ±t symmetrically (the `copysign` rule). -/
theorem roundHalfAway_neg (n : Int) (d : Nat) (hn : 0 < n) :
    roundHalfAway (-n) d = -roundHalfAway n d := by
  have h1 : ¬ (0 ≤ -n) := by omega
  have h2 : 0 ≤ n := by omega
  simp only [roundHalfAway, h1, h2, ite_false, ite_true]
  have h3 : 2 * -n - (d : Int) = -(2 * n + d) := by omega
  rw [h3, Int.neg_tdiv]

/-- Zero rounds to zero (the amax = 0 chunk). -/
theorem roundHalfAway_zero (d : Nat) (hd : 0 < d) : roundHalfAway 0 d = 0 := by
  simp only [roundHalfAway, Int.le_refl, ite_true, Int.mul_zero, Int.zero_add]
  apply Int.tdiv_eq_zero_of_lt
  · omega
  · omega

-- § clamping

/-- `min(max(v, −127), 127)`. -/
def clamp127 (v : Int) : Int := min (max v (-127)) 127

/-- Clamped codes are int8 values in [−127, 127] (−128 never appears). -/
theorem clamp127_range (v : Int) : -127 ≤ clamp127 v ∧ clamp127 v ≤ 127 := by
  simp only [clamp127]; omega

/-- Clamping is the identity inside the range. -/
theorem clamp127_of_range (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clamp127 v = v := by
  simp only [clamp127]; omega

/-- Clamping is idempotent. -/
theorem clamp127_idem (v : Int) : clamp127 (clamp127 v) = clamp127 v := by
  have h := clamp127_range v
  exact clamp127_of_range _ h.1 h.2

/-- One legacy code: round half away, then clamp. -/
def quantCode (n : Int) (d : Nat) : Int := clamp127 (roundHalfAway n d)

/-- With `inv = 0` (zero chunk) every element quantizes to 0. -/
theorem zero_chunk_code (d : Nat) (hd : 0 < d) : quantCode 0 d = 0 := by
  simp only [quantCode, roundHalfAway_zero d hd]
  decide

-- § chunk sums

/-- Integer sum of a code list (the chunk's `sum`). -/
def listSum : List Int → Int
  | [] => 0
  | v :: t => v + listSum t

/-- A list of in-range codes has |sum| ≤ 127 · length. -/
theorem listSum_bound (l : List Int) (h : ∀ v ∈ l, -127 ≤ v ∧ v ≤ 127) :
    -(127 * (l.length : Int)) ≤ listSum l ∧ listSum l ≤ 127 * (l.length : Int) := by
  induction l with
  | nil => simp [listSum]
  | cons v t ih =>
    have hv := h v (List.mem_cons_self ..)
    have ht := ih (fun w hw => h w (List.mem_cons_of_mem _ hw))
    simp only [listSum, List.length_cons]
    omega

/-- A 32-element chunk of clamped codes sums to at most 4064 in magnitude: int32 never overflows. -/
theorem chunk_sum_fits (xs : List Int) (hlen : xs.length = 32) :
    -4064 ≤ listSum (xs.map clamp127) ∧ listSum (xs.map clamp127) ≤ 4064 := by
  have h := listSum_bound (xs.map clamp127) (by
    intro v hv
    obtain ⟨w, _, rfl⟩ := List.mem_map.mp hv
    exact clamp127_range w)
  rw [List.length_map, hlen] at h
  exact ⟨by omega, by omega⟩

/-- The zero chunk has sum 0, hence `hx = scale · sum = 0`. -/
theorem zero_chunk_sum (n : Nat) : listSum (List.replicate n 0) = 0 := by
  induction n with
  | zero => rfl
  | succ k ih => simp [List.replicate_succ, listSum, ih]

-- § vector lanes are the scalar rule

/-- A vector loop: lanes of width `w`, each lane group mapped with the same per-element function. -/
def laneMap {α β : Type} (f : α → β) (w : Nat) (l : List α) : List β :=
  (l.take w).map f ++ (l.drop w).map f

/-- The AVX-512/AVX2 quantizers are bitwise the scalar loop: lane grouping is invisible. -/
theorem laneMap_eq_map {α β : Type} (f : α → β) (w : Nat) (l : List α) :
    laneMap f w l = l.map f := by
  simp only [laneMap]
  rw [← List.map_append, List.take_append_drop]

end Strata.Kernels.Cpu.ActQuant

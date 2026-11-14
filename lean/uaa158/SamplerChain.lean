/-!
# MODULE: sampler_chain
SOURCES: src/kernels/cuda/sampler.cu
Penalty, greedy argmax, history window, top-p bookkeeping, Philox uniform.
-/

namespace SamplerChain

-- § repeat/frequency/presence penalties

/-- Repeat stage: scale down non-positive logits, scale up positive ones. -/
def repeatStage (l r : Int) : Int :=
  if l ≤ 0 then l * r else l / r

/-- Full penalty applied to a token seen `cnt` times. -/
def penalize (logit : Int) (cnt : Nat) (r f pres : Int) : Int :=
  if cnt = 0 then logit
  else repeatStage logit r - (cnt : Int) * f - (if cnt > 0 then pres else 0)

/-- For r ≥ 1 the repeat stage never raises a logit. -/
theorem repeatStage_le (logit r : Int) (hr : 1 ≤ r) : repeatStage logit r ≤ logit := by
  simp only [repeatStage]
  split
  · rename_i h
    have : logit * (r - 1) ≤ 0 := Int.mul_nonpos_of_nonpos_of_nonneg h (by omega)
    rw [Int.mul_sub, Int.mul_one] at this; omega
  · rename_i h; exact Int.ediv_le_self r (by omega)

/-- Always-divide would raise negative logits. -/
theorem naive_divide_inverts : (-6 : Int) / 2 > -6 := by norm_num

/-- Presence penalty is boolean: same for cnt=1 and cnt=100. -/
theorem presence_boolean (logit r pres : Int) (c1 c2 : Nat) (h1 : 0 < c1) (h2 : 0 < c2) :
    penalize logit c1 r 0 pres = penalize logit c2 r 0 pres := by
  simp [penalize, Nat.pos_iff_ne_zero.mp h1, Nat.pos_iff_ne_zero.mp h2, h1, h2]

/-- Unseen tokens are unpenalized. -/
theorem penalize_unseen (logit r f pres : Int) : penalize logit 0 r f pres = logit :=
  if_pos rfl

/-- Non-negative penalties with r ≥ 1 never raise logits. -/
theorem penalize_le (logit : Int) (cnt : Nat) (r f pres : Int)
    (hr : 1 ≤ r) (hf : 0 ≤ f) (hp : 0 ≤ pres) : penalize logit cnt r f pres ≤ logit := by
  simp only [penalize]
  split
  · exact le_refl _
  · have h1 := repeatStage_le logit r hr
    have h2 : 0 ≤ (cnt : Int) * f := Int.mul_nonneg (by omega) hf
    split <;> omega

/-- Penalty order matters: scaling then penalising differs from penalising then scaling. -/
theorem penalty_after_temperature_differs :
    penalize (3 * 2) 1 1 0 1 ≠ penalize 3 1 1 0 1 * 2 := by norm_num [penalize, repeatStage]

-- § greedy argmax (temperature-0 bug: scaling by 0 gives all zeros → always token 0)

/-- Scan for the best (value, index), keeping lowest index on ties. -/
def amx : List Int → Nat → Nat → Int → Nat
  | [], _, bi, _ => bi
  | v :: vs, i, bi, bv => if v > bv then amx vs (i + 1) i v else amx vs (i + 1) bi bv

/-- argmax of a logit list (empty → 0). -/
def argmax : List Int → Nat
  | [] => 0
  | v :: vs => amx vs 1 0 v

/-- Positive scaling preserves argmax order. -/
theorem amx_scale (c : Int) (hc : 0 < c) : ∀ (l : List Int) (i bi : Nat) (bv : Int),
    amx (l.map (· * c)) i bi (bv * c) = amx l i bi bv
  | [], _, _, _ => rfl
  | v :: vs, i, bi, bv => by
    simp only [List.map, amx]
    have hiff : v * c > bv * c ↔ v > bv := Int.mul_lt_mul_right hc
    by_cases hv : v > bv
    · rw [if_pos (hiff.mpr hv), if_pos hv]; exact amx_scale c hc vs (i + 1) i v
    · rw [if_neg (fun h => hv (hiff.mp h)), if_neg hv]
      exact amx_scale c hc vs (i + 1) bi bv

/-- argmax is scale-invariant for positive c. -/
theorem argmax_scale (l : List Int) (c : Int) (hc : 0 < c) :
    argmax (l.map (· * c)) = argmax l := by
  cases l with
  | nil => rfl
  | cons v vs => exact amx_scale c hc vs 1 0 v

/-- If nothing beats the current best, it stays. -/
theorem amx_keep : ∀ (l : List Int) (i bi : Nat) (bv : Int),
    (∀ v, v ∈ l → v ≤ bv) → amx l i bi bv = bi
  | [], _, _, _, _ => rfl
  | v :: vs, i, bi, bv, h => by
    simp only [amx]
    rw [if_neg (by linarith [h v (List.mem_cons_self v vs)])]
    exact amx_keep vs (i + 1) bi bv (fun y hy => h y (List.mem_cons_of_mem v hy))

/-- Temperature-0 bug: all logits become 0 → always picks token 0. -/
theorem zero_temperature_bug (l : List Int) : argmax (l.map (· * 0)) = 0 := by
  cases l with
  | nil => rfl
  | cons v vs =>
    simp only [List.map, argmax, Int.mul_zero]
    apply amx_keep
    intro y hy
    simp only [List.mem_map] at hy
    obtain ⟨z, _, rfl⟩ := hy; simp

/-- Ties go to the lowest index. -/
theorem argmax_tie : argmax [5, 7, 7, 1] = 1 := rfl

/-- Sentinel (no candidate) → token 0. -/
def reportBest (wi nv : Nat) : Nat := if wi < nv then wi else 0

/-- reportBest always produces a valid vocabulary index. -/
theorem reportBest_valid (wi nv : Nat) (h : 0 < nv) : reportBest wi nv < nv := by
  simp only [reportBest]; split <;> omega

-- § history window (tail of length min(lastN, len))

/-- History tail window, clamped below by 0. -/
def window (lastN len : Int) : Int := (lastN.min len).max 0

/-- The window is exactly the tail of the history. -/
theorem window_tail (lastN len : Int) (hl : 0 ≤ len) :
    0 ≤ len - window lastN len ∧
    (len - window lastN len) + window lastN len = len ∧
    window lastN len ≤ len := by
  simp only [window]
  have h1 := Int.min_le_right lastN len
  have h2 : (lastN.min len).max 0 ≤ len :=
    Int.max_le.mpr ⟨hl, h1⟩
  omega

-- § top-k / top-p bookkeeping

/-- Valid top-k range: 1..64. -/
def topKOk (k : Int) : Bool := decide (1 ≤ k ∧ k ≤ 64)

/-- Survivors after top-p cut and min_keep clamp. -/
def keepAfterTopP (cut mk nk : Nat) : Nat :=
  if cut < mk then min mk nk else cut

/-- keepAfterTopP stays in bounds. -/
theorem keepAfterTopP_bounds (cut mk nk : Nat) (hc : cut ≤ nk) :
    keepAfterTopP cut mk nk ≤ nk ∧
    min mk nk ≤ keepAfterTopP cut mk nk := by
  simp only [keepAfterTopP]; split <;> omega

-- § Philox uniform: (c0 >> 8) * 2^-24 ∈ [0, 1)

/-- Top 24 bits of a 32-bit counter word (bit-shift equivalent). -/
def mantissa24 (c : Nat) : Nat := c / 256

/-- Top 24 bits of a 32-bit word are strictly < 2^24. -/
theorem mantissa24_lt (c : Nat) (h : c < 2 ^ 32) : mantissa24 c < 2 ^ 24 := by
  simp only [mantissa24]; omega

/-- Distinct tokens of a batch use distinct Philox counters. -/
theorem counters_distinct (base t1 t2 : Nat) (h : base + t1 = base + t2) : t1 = t2 := by
  exact Nat.add_left_cancel h

end SamplerChain

/-!
# MODULE: sampler_chain
Greedy argmax tie rule, penalty-window arithmetic and penalty semantics of the sampler.

## Source semantics
Chain order: penalties, top_k, top_p, temperature, pick. Greedy is the global argmax over penalised logits;
ties go to the lowest index (serial scan with strict `>`; parallel merge prefers the smaller index).
The penalty window is the TAIL of the history: the last min(penalty_last_n, history_len) entries.
Repeat penalty multiplies non-positive logits and divides positive ones; the presence term is a boolean.

## Form template identifiers
- `beats` ↔ `ov > bv || (ov == bv && oi < best)`; `windowLen`/`windowOff` ↔ `hlen`/`hrow += history_len - hlen`
- `applyPen` ↔ `apply_penalties`
-/

namespace Strata.SamplerChain

-- § greedy tie rule

/-- Candidate `(value, index)` beats the incumbent: larger value, or equal value and smaller index. -/
def beats (a b : Int × Nat) : Prop := a.1 > b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2)

/-- Beating is irreflexive. -/
theorem beats_irrefl (a : Int × Nat) : ¬ beats a a := by
  unfold beats; omega

/-- Beating is transitive, so the merge order is a strict order. -/
theorem beats_trans (a b c : Int × Nat) (h1 : beats a b) (h2 : beats b c) : beats a c := by
  unfold beats at *; omega

/-- Distinct indices are always comparable, so the argmax is unique and merge order is irrelevant. -/
theorem beats_total (a b : Int × Nat) (h : a.2 ≠ b.2) : beats a b ∨ beats b a := by
  unfold beats; omega

/-- On equal logits the lower index wins. -/
theorem beats_tie (v : Int) (i j : Nat) (h : i < j) : beats (v, i) (v, j) := by
  unfold beats; omega

-- § penalty window

/-- Window length: `min(penalty_last_n, history_len)`. -/
def windowLen (lastN histLen : Nat) : Nat := min lastN histLen

/-- Window start: skip the head so the window is the tail. -/
def windowOff (lastN histLen : Nat) : Nat := histLen - windowLen lastN histLen

/-- The window never exceeds the history. -/
theorem windowLen_le (lastN histLen : Nat) : windowLen lastN histLen ≤ histLen := by
  unfold windowLen; omega

/-- Offset plus length is exactly the history length, so the window ends at the newest entry. -/
theorem window_tail (lastN histLen : Nat) :
    windowOff lastN histLen + windowLen lastN histLen = histLen := by
  unfold windowOff windowLen; omega

/-- A zero `penalty_last_n` gives an empty window. -/
theorem window_empty (histLen : Nat) : windowLen 0 histLen = 0 := by
  unfold windowLen; omega

/-- Window reads stay in bounds: `off + j < histLen` for `j < len`. -/
theorem window_in_bounds (lastN histLen j : Nat) (h : j < windowLen lastN histLen) :
    windowOff lastN histLen + j < histLen := by
  unfold windowOff windowLen at *; omega

-- § penalty semantics (integer model of the float expression)

/-- Presence term: a boolean, `1` if the token appeared at all. -/
def presence (count : Nat) : Int := if count > 0 then 1 else 0

/-- Presence does not grow with the count beyond one. -/
theorem presence_le_one (count : Nat) : presence count ≤ 1 := by
  unfold presence; split <;> omega

/-- Presence is zero exactly for unseen tokens. -/
theorem presence_zero_iff (count : Nat) : presence count = 0 ↔ count = 0 := by
  unfold presence; split <;> omega

/-- Repeat step: multiply non-positive logits, divide positive ones (integer model). -/
def repeatStep (logit r : Int) : Int := if logit ≤ 0 then logit * r else logit / r

/-- Full penalty: repeat step then subtract frequency and presence terms; unseen tokens untouched. -/
def applyPen (logit : Int) (count : Nat) (r freq pres : Int) : Int :=
  if count = 0 then logit else repeatStep logit r - ((count : Int) * freq + presence count * pres)

/-- Unseen tokens keep their logit. -/
theorem applyPen_unseen (logit r freq pres : Int) : applyPen logit 0 r freq pres = logit := by
  simp [applyPen]

/-- A positive logit never increases under a penalty `r ≥ 1` with nonnegative additive terms. -/
theorem repeat_positive_le (logit r : Int) (hl : 0 < logit) (hr : 1 ≤ r) :
    repeatStep logit r ≤ logit := by
  unfold repeatStep
  have h1 : ¬ logit ≤ 0 := by omega
  simp only [h1, if_false]
  exact Int.ediv_le_self r (by omega)

/-- A non-positive logit never increases under `r ≥ 1` (the penalty is not inverted). -/
theorem repeat_nonpos_le (logit r : Int) (hl : logit ≤ 0) (hr : 1 ≤ r) :
    repeatStep logit r ≤ logit := by
  unfold repeatStep
  simp only [hl, if_true]
  have h : logit * r = logit + logit * (r - 1) := by
    rw [Int.mul_sub, Int.mul_one]; omega
  have h2 : logit * (r - 1) ≤ 0 :=
    Int.mul_nonpos_of_nonpos_of_nonneg hl (by omega)
  omega

-- § top-k bound

/-- The sampled path's effective k: capped at KMAX=64, zero means unsupported. -/
def effK (topK : Nat) : Nat := if topK < 64 then topK else 64

/-- `effK` never exceeds the keep-array size. -/
theorem effK_le (topK : Nat) : effK topK ≤ 64 := by
  unfold effK; split <;> omega

/-- The keep count after top_p is at least `min_keep` when possible and at most `n`. -/
def cutKeep (cut minKeep n : Nat) : Nat := if cut < minKeep then min minKeep n else cut

/-- Cut never exceeds `n` provided the incoming cut did not. -/
theorem cutKeep_le (cut minKeep n : Nat) (h : cut ≤ n) : cutKeep cut minKeep n ≤ n := by
  unfold cutKeep; split <;> omega

end Strata.SamplerChain

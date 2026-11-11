/-!
# MODULE: sampler_chain
SOURCES: src/kernels/cuda/sampler.cu
Model of the P2.S2 sampler: llama.cpp's repeat / frequency / presence penalty, the greedy
argmax (and the temperature-0 bug it fixes), the chain order `top_k → top_p → temperature →
penalties → pick`, the penalty history window, the top-p `min_keep` clamp and the Philox
uniform's range.

## Source semantics
* Penalty for a token seen `count > 0` times: `logit ≤ 0 ? logit·r : logit/r`, then
  `− count·f − [count>0]·presence` (presence is boolean).
* Greedy = argmax over penalised RAW logits, lowest index on ties; temperature must not be
  applied (scaling by `1/T = 0` made every logit 0 and always picked token 0).
* Survivors are chosen on raw logits; temperature scales them; penalties then see the scaled
  value — penalising before scaling is a different distribution.
* The history window is the TAIL `min(penalty_last_n, history_len)` entries.
* `philox_uniform = (c0 >> 8) · 2⁻²⁴ ∈ [0, 1)`.

## Form template identifiers
`apply_penalties` → `penalize`/`repeatStage`; serial greedy scan → `amx`/`argmax`;
`hrow += history_len - hlen` → `window`; top-p cut clamp → `keepAfterTopP`;
`philox_uniform` mantissa → `mantissa24`.
-/

namespace SamplerChain

-- § penalties

/-- The repeat stage only: multiply non-positive logits, divide positive ones. -/
def repeatStage (logit r : Int) : Int := if logit ≤ 0 then logit * r else logit / r

/-- Full `apply_penalties` for a token seen `count` times. -/
def penalize (logit : Int) (count : Nat) (r f pres : Int) : Int :=
  if count = 0 then logit
  else repeatStage logit r - (count : Int) * f - (if count > 0 then pres else 0)

/-- For `r ≥ 1` the repeat stage never RAISES a logit, on either sign.  Dividing
    unconditionally would raise every negative logit — the inversion the source avoids. -/
theorem repeatStage_le (logit r : Int) (hr : 1 ≤ r) : repeatStage logit r ≤ logit := by
  unfold repeatStage
  split
  · rename_i h
    have h1 : logit * (r - 1) ≤ 0 := Int.mul_nonpos_of_nonpos_of_nonneg h (by omega)
    rw [Int.mul_sub, Int.mul_one] at h1
    omega
  · rename_i h
    exact Int.ediv_le_self r (by omega)

/-- The naive "always divide" reading raises a negative logit (`-6 / 2 = -3 > -6`). -/
theorem naive_divide_inverts : (-6 : Int) / 2 > -6 := by decide

/-- The presence term is boolean: it is the same for a token seen once or a hundred times. -/
theorem presence_boolean (logit r pres : Int) (c1 c2 : Nat) (h1 : 0 < c1) (h2 : 0 < c2) :
    penalize logit c1 r 0 pres = penalize logit c2 r 0 pres := by
  have a1 : c1 ≠ 0 := by omega
  have a2 : c2 ≠ 0 := by omega
  simp [penalize, a1, a2, h1, h2]

/-- An unseen token is never penalised. -/
theorem penalize_unseen (logit r f pres : Int) : penalize logit 0 r f pres = logit := by
  simp [penalize]

/-- With non-negative frequency/presence penalties and `r ≥ 1`, penalties never raise a logit. -/
theorem penalize_le (logit : Int) (count : Nat) (r f pres : Int) (hr : 1 ≤ r) (hf : 0 ≤ f)
    (hp : 0 ≤ pres) : penalize logit count r f pres ≤ logit := by
  unfold penalize
  split
  · exact Int.le_refl _
  · have h1 := repeatStage_le logit r hr
    have h2 : 0 ≤ (count : Int) * f := Int.mul_nonneg (by omega) hf
    split <;> omega

/-- Order matters: penalising the temperature-scaled logit differs from scaling the penalised
    one (logit 3, scale 2, presence 1: `6 − 1 = 5` vs `(3 − 1)·2 = 4`). -/
theorem penalty_after_temperature_differs :
    penalize (3 * 2) 1 1 0 1 ≠ penalize 3 1 1 0 1 * 2 := by decide

-- § greedy argmax

/-- Serial scan state: remaining logits, next index, best index, best value; strict `>` keeps
    the lowest index on ties. -/
def amx : List Int → Nat → Nat → Int → Nat
  | [], _, bi, _ => bi
  | x :: xs, i, bi, bv => if x > bv then amx xs (i + 1) i x else amx xs (i + 1) bi bv

/-- Argmax of a non-empty logit row. -/
def argmax : List Int → Nat
  | [] => 0
  | x :: xs => amx xs 1 0 x

/-- Scaling by a positive constant cannot reorder anything, so it cannot move the argmax:
    temperature is irrelevant to greedy decoding. -/
theorem amx_scale (c : Int) (hc : 0 < c) : ∀ (l : List Int) (i bi : Nat) (bv : Int),
    amx (l.map (· * c)) i bi (bv * c) = amx l i bi bv
  | [], _, _, _ => rfl
  | x :: xs, i, bi, bv => by
    simp only [List.map, amx]
    have hiff : x * c > bv * c ↔ x > bv := Int.mul_lt_mul_right hc
    by_cases h : x > bv
    · rw [if_pos (hiff.mpr h), if_pos h]; exact amx_scale c hc xs (i + 1) i x
    · rw [if_neg (fun h' => h (hiff.mp h')), if_neg h]; exact amx_scale c hc xs (i + 1) bi bv

/-- `argmax` is invariant under positive scaling. -/
theorem argmax_scale (l : List Int) (c : Int) (hc : 0 < c) : argmax (l.map (· * c)) = argmax l := by
  cases l with
  | nil => rfl
  | cons x xs => exact amx_scale c hc xs 1 0 x

/-- If nothing later strictly exceeds the current best, the scan keeps it. -/
theorem amx_keep : ∀ (l : List Int) (i bi : Nat) (bv : Int), (∀ x, x ∈ l → x ≤ bv) →
    amx l i bi bv = bi
  | [], _, _, _, _ => rfl
  | x :: xs, i, bi, bv, h => by
    simp only [amx]
    rw [if_neg (by have := h x (by simp); omega)]
    exact amx_keep xs (i + 1) bi bv (fun y hy => h y (by simp [hy]))

/-- THE TEMPERATURE-0 BUG: multiplying by `inv_t = 0` makes every logit 0 and the argmax is
    token 0 whatever the model predicted. -/
theorem zero_temperature_bug (l : List Int) : argmax (l.map (· * 0)) = 0 := by
  cases l with
  | nil => rfl
  | cons x xs =>
    simp only [List.map, argmax, Int.mul_zero]
    exact amx_keep _ 1 0 0 (fun y hy => by
      simp only [List.mem_map] at hy
      obtain ⟨z, _, rfl⟩ := hy
      simp)

/-- Ties go to the lowest index (the block reduction's `(ov == bv && oi < bi)` rule). -/
theorem argmax_tie : argmax [5, 7, 7, 1] = 1 := by decide

/-- The block reduction's sentinel answer: `n_vocab` (no candidate) is reported as token 0. -/
def reportBest (wi nVocab : Nat) : Nat := if wi < nVocab then wi else 0

/-- The reported token is always a valid vocabulary index. -/
theorem reportBest_valid (wi nVocab : Nat) (h : 0 < nVocab) : reportBest wi nVocab < nVocab := by
  unfold reportBest; split <;> omega

-- § history window

/-- Window length `min(last_n, len)` clamped at 0 (Int inputs as in the source). -/
def window (lastN len : Int) : Int := max 0 (min lastN len)

/-- The window is the tail: it starts at `len − hlen ≥ 0` and ends exactly at `len`. -/
theorem window_tail (lastN len : Int) (hl : 0 ≤ len) :
    0 ≤ len - window lastN len ∧ (len - window lastN len) + window lastN len = len ∧
    window lastN len ≤ len := by
  unfold window
  have h1 := Int.min_le_right lastN len
  have h2 := Int.le_max_left 0 (min lastN len)
  have h3 : max 0 (min lastN len) ≤ len := Int.max_le.mpr ⟨hl, h1⟩
  omega

-- § top-k / top-p bookkeeping

/-- The sampled path supports `top_k ∈ 1..64` (`KMAX`). -/
def topKOk (k : Int) : Bool := 1 ≤ k && k ≤ 64

/-- Survivor count after the top-p cut and the `min_keep` clamp. -/
def keepAfterTopP (cut minKeep nKeep : Nat) : Nat :=
  if cut < minKeep then min minKeep nKeep else cut

/-- The clamp never keeps more than the top-k survivors and keeps at least `min_keep` of them
    whenever that many exist. -/
theorem keepAfterTopP_bounds (cut minKeep nKeep : Nat) (hc : cut ≤ nKeep) :
    keepAfterTopP cut minKeep nKeep ≤ nKeep ∧
    min minKeep nKeep ≤ keepAfterTopP cut minKeep nKeep := by
  unfold keepAfterTopP; split <;> omega

-- § Philox uniform

/-- `c0 >> 8` keeps 24 bits. -/
def mantissa24 (c : Nat) : Nat := c >>> 8

/-- For a 32-bit word the 24-bit mantissa is `< 2²⁴`, so `u = m · 2⁻²⁴ < 1` exactly: the
    inverse-CDF pick never sees `u = 1`. -/
theorem mantissa24_lt (c : Nat) (h : c < 2^32) : mantissa24 c < 2^24 := by
  unfold mantissa24; rw [Nat.shiftRight_eq_div_pow]; omega

/-- The per-token counter `p.counter + t` makes each token's draw a function of
    (seed, position) only: distinct tokens of a batch use distinct counters. -/
theorem counters_distinct (base t1 t2 : Nat) (h : base + t1 = base + t2) : t1 = t2 := by omega

end SamplerChain

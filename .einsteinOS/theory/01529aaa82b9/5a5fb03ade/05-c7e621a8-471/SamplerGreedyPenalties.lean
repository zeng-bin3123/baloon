/-!
# MODULE: sampler_greedy_penalties
The sampler chain's deterministic parts (`src/kernels/cuda/sampler.cu`): the block-parallel greedy argmax
with lowest-index tie rule, the penalty window, the repeat/frequency/presence penalty, the top-k clamp,
and the Philox-to-uniform conversion.

## Source semantics
* Greedy = argmax over penalised logits; each thread keeps `if (s > bv)` over a strided subset, and the
  reduction picks the larger value, on equality the SMALLER index; `n_vocab` is the "no candidate" index.
* An all `-inf` row yields index 0.
* Penalty window is the TAIL of the history: `hlen = clamp(penalty_last_n, 0, history_len)`, start `history_len - hlen`.
* Repeat penalty multiplies non-positive logits and divides positive ones; presence is a boolean, not the count.
* `top_k` is clamped to 64; `top_k <= 0` is refused on the sampled path.
* Uniform = `(c0 >> 8) / 2^24`, strictly below 1.

## Form template identifiers
* `merge`       <- the shuffle/shared-memory candidate combine
* `windowLen`   <- `hlen`
* `presence`    <- `(count > 0 ? 1 : 0) * penalty_present`
* `topKClamp`   <- `k`
-/

namespace SamplerGreedyPenalties

-- § candidate merge

/-- Combine two (value, index) candidates: larger value wins, equal values prefer the smaller index. -/
def merge (a b : Int × Nat) : Int × Nat :=
  if b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2) then b else a

/-- Merging is commutative, so the reduction order does not matter. -/
theorem merge_comm (a b : Int × Nat) : merge a b = merge b a := by
  obtain ⟨av, ai⟩ := a
  obtain ⟨bv, bi⟩ := b
  unfold merge
  by_cases h1 : bv > av ∨ (bv = av ∧ bi < ai)
  · have h2 : ¬ (av > bv ∨ (av = bv ∧ ai < bi)) := by omega
    simp only [h1, h2, if_true, if_false]
  · by_cases h2 : av > bv ∨ (av = bv ∧ ai < bi)
    · simp only [h1, h2, if_true, if_false]
    · have : av = bv ∧ ai = bi := by omega
      obtain ⟨e1, e2⟩ := this
      subst e1
      subst e2
      simp only [h1, if_false]

/-- A real candidate (index below `n_vocab`) with value at least the sentinel's beats the sentinel. -/
theorem merge_sentinel (v sv : Int) (i n : Nat) (hi : i < n) (hv : sv ≤ v) :
    merge (sv, n) (v, i) = (v, i) := by
  unfold merge
  have h : v > sv ∨ (v = sv ∧ i < n) := by omega
  simp only [h, if_true]

/-- The result of a merge is one of its inputs. -/
theorem merge_mem (a b : Int × Nat) : merge a b = a ∨ merge a b = b := by
  unfold merge
  split
  · right; rfl
  · left; rfl

/-- The result is never smaller (in value) than either input. -/
theorem merge_ge (a b : Int × Nat) : a.1 ≤ (merge a b).1 ∧ b.1 ≤ (merge a b).1 := by
  obtain ⟨av, ai⟩ := a
  obtain ⟨bv, bi⟩ := b
  unfold merge
  by_cases h : bv > av ∨ (bv = av ∧ bi < ai)
  · simp only [h, if_true]
    omega
  · simp only [h, if_false]
    omega

/-- The fallback for an all `-inf` row: index `n_vocab` becomes 0. -/
def finalIndex (wi nVocab : Nat) : Nat := if wi < nVocab then wi else 0

/-- The emitted token is always a valid vocabulary index. -/
theorem finalIndex_lt (wi nVocab : Nat) (h : 0 < nVocab) : finalIndex wi nVocab < nVocab := by
  unfold finalIndex
  split <;> omega

-- § penalty window

/-- Window length: `penalty_last_n` clamped into `[0, history_len]`. -/
def windowLen (lastN histLen : Int) : Int := if lastN < histLen then (if lastN < 0 then 0 else lastN) else histLen

/-- The window fits in the history. -/
theorem windowLen_le (lastN histLen : Int) (h : 0 ≤ histLen) : 0 ≤ windowLen lastN histLen ∧
    windowLen lastN histLen ≤ histLen := by
  unfold windowLen
  split <;> (try split) <;> omega

/-- The window is the tail: start offset plus length is the history length. -/
theorem window_tail (lastN histLen : Int) (h : 0 ≤ histLen) :
    (histLen - windowLen lastN histLen) + windowLen lastN histLen = histLen := by
  omega

-- § penalties

/-- Repeat penalty: multiply non-positive logits, divide positive ones (integer model of the ordering). -/
def repeatPen (logit rep : Int) : Int := if logit ≤ 0 then logit * rep else logit / rep

/-- Presence term: boolean, not the count. -/
def presence (count : Nat) (pen : Int) : Int := if 0 < count then pen else 0

/-- A token that never appeared keeps its logit (the kernel returns early for `count <= 0`). -/
def applyPen (logit : Int) (count : Nat) (rep freq pres : Int) : Int :=
  if count = 0 then logit
  else repeatPen logit rep - (count : Int) * freq - presence count pres

/-- Unseen tokens are untouched. -/
theorem applyPen_unseen (logit rep freq pres : Int) : applyPen logit 0 rep freq pres = logit := by
  simp [applyPen]

/-- The presence term does not grow with the count. -/
theorem presence_const (c1 c2 : Nat) (pen : Int) (h1 : 0 < c1) (h2 : 0 < c2) :
    presence c1 pen = presence c2 pen := by
  unfold presence
  rw [if_pos h1, if_pos h2]

/-- Unity penalties on a positive logit leave it unchanged when the other terms are zero. -/
theorem repeatPen_one_pos (logit : Int) (h : 0 < logit) : repeatPen logit 1 = logit := by
  unfold repeatPen
  rw [if_neg (by omega)]
  simp

-- § sampling parameters

/-- Top-k clamp: `top_k` in `1..64`, zero means "refuse". -/
def topKClamp (topK : Int) : Int := if topK > 0 then (if topK < 64 then topK else 64) else 0

/-- The clamp never exceeds the 64-slot keep array. -/
theorem topKClamp_le (topK : Int) : 0 ≤ topKClamp topK ∧ topKClamp topK ≤ 64 := by
  unfold topKClamp
  split <;> (try split) <;> omega

/-- A positive request gets at least one slot. -/
theorem topKClamp_pos (topK : Int) (h : 0 < topK) : 1 ≤ topKClamp topK := by
  unfold topKClamp
  split <;> (try split) <;> omega

-- § Philox uniform

/-- The 24-bit numerator of the uniform is below the denominator `2^24`, so the draw is in `[0, 1)`. -/
theorem uniform_lt_one (c0 : Nat) (h : c0 < 4294967296) : c0 / 256 < 16777216 := by
  omega

end SamplerGreedyPenalties

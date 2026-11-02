/-!
# MODULE: sampler
Sampler chain: penalties -> top_k -> top_p -> temperature -> pick, with a Philox counter-based uniform draw and a
block-wide greedy argmax whose tie rule is the lowest index.

## Source semantics
Greedy is the global argmax over penalised logits (temperature is NOT applied; temperature 0 would zero logits).
The penalty window is the tail of the history of length min(penalty_last_n, history_len).
Philox uniform uses the top 24 bits so the draw is in [0,1).

## Form template identifiers
sample_tokens -> sample_next_token; SamplerParams -> sampling_config
-/
namespace Strata.Sampler

/-- Candidate merge in the greedy reduction: larger value wins, tie goes to the smaller index. -/
def merge (a b : Int × Nat) : Int × Nat :=
  if b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2) then b else a

/-- The merged value is at least both inputs. -/
theorem merge_ge (a b : Int × Nat) : a.1 ≤ (merge a b).1 ∧ b.1 ≤ (merge a b).1 := by
  unfold merge; split <;> omega

/-- On equal values the merged index is the smaller one. -/
theorem merge_tie (a b : Int × Nat) (h : a.1 = b.1) :
    (merge a b).2 ≤ a.2 ∧ (merge a b).2 ≤ b.2 := by
  unfold merge; split <;> omega

/-- The merge returns one of its inputs. -/
theorem merge_mem (a b : Int × Nat) : merge a b = a ∨ merge a b = b := by
  unfold merge; split <;> simp

/-- Philox draw: the top 24 bits of a 32-bit word are below 2^24, so the uniform is below 1. -/
theorem uniform_lt_one (c0 : Nat) (h : c0 < 4294967296) : c0 / 256 < 16777216 := by omega

/-- Penalty window length. -/
def hlen (lastN histLen : Nat) : Nat := min lastN histLen

/-- Window is a tail: offset plus length equals the history length. -/
theorem window_tail (lastN histLen : Nat) : (histLen - hlen lastN histLen) + hlen lastN histLen = histLen := by
  unfold hlen; omega

/-- Window never exceeds the history. -/
theorem window_le (lastN histLen : Nat) : hlen lastN histLen ≤ histLen := by unfold hlen; omega

/-- Top-p cut after the min_keep clamp. -/
def clampCut (cut minKeep n : Nat) : Nat := if cut < minKeep then min minKeep n else cut

/-- The clamped cut never exceeds the survivor count. -/
theorem clampCut_le (cut minKeep n : Nat) (h : cut ≤ n) : clampCut cut minKeep n ≤ n := by
  unfold clampCut; split <;> omega

/-- The clamped cut honours min_keep when enough survivors exist. -/
theorem clampCut_ge (cut minKeep n : Nat) (h : minKeep ≤ n) : minKeep ≤ clampCut cut minKeep n := by
  unfold clampCut; split <;> omega

/-- Presence penalty is a boolean, not the count. -/
def presenceTerm (count : Nat) : Nat := if count > 0 then 1 else 0

/-- Presence term is at most one. -/
theorem presence_le_one (count : Nat) : presenceTerm count ≤ 1 := by unfold presenceTerm; split <;> omega

/-- Repeat penalty direction: non-positive logits multiply, positive divide (model over Int, penalty >= 1). -/
def repeatPen (logit p : Int) : Int := if logit ≤ 0 then logit * p else logit / p

/-- A positive logit never becomes larger under a penalty >= 1. -/
theorem repeat_pos (logit p : Int) (hl : 0 < logit) (hp : 1 ≤ p) : repeatPen logit p ≤ logit := by
  unfold repeatPen
  split
  · omega
  · exact Int.ediv_le_self _ hl.le

/-- A non-positive logit never becomes larger under a penalty >= 1. -/
theorem repeat_nonpos (logit p : Int) (hl : logit ≤ 0) (hp : 1 ≤ p) : repeatPen logit p ≤ logit := by
  unfold repeatPen
  split
  · nlinarith [mul_nonneg (neg_nonneg.mpr hl) (sub_nonneg.mpr hp)]
  · omega

end Strata.Sampler

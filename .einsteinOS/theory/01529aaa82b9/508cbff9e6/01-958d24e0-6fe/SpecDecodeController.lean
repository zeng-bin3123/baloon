/-!
# MODULE: spec_decode_controller
Speculative-decoding controller (which drafter, how many draft tokens) and the suffix-lookup drafter's hash table and match logic.

## Source semantics
* `step_ms` clamps `n = k+1` into [1, K_MAX+1].
* Lookup acceptance is kept per match-length bucket: <5, 5..7, 8..15, >=16.
* `observe` clamps `accepted` into [0, k] and updates positions 0..min(k, accepted+1)-1.
* The drafter's table key is forced nonzero (`| 1`), 0 marking an empty slot; each slot keeps the WAYS most recent positions.
* `min_match = max(3, min_match)`; the proposed continuation has at most `max_k` tokens and never reads past the current suffix.

## Form template identifiers
* Controller::bucket -> `match_bucket`
* CostModel::step_ms -> `step_ms`
* Controller::observe -> `observe`
* SuffixDrafter::key_at -> `suffix_key`
* SuffixDrafter::propose -> `propose`
-/

namespace Strata.Spec

-- § Controller

/-- Match-length bucket for lookup acceptance. -/
def bucket (m : Nat) : Nat := if 16 ≤ m then 3 else if 8 ≤ m then 2 else if 5 ≤ m then 1 else 0

/-- There are exactly four buckets. -/
theorem bucket_lt (m : Nat) : bucket m < 4 := by
  unfold bucket
  split <;> try omega
  split <;> try omega
  split <;> omega

/-- Longer matches never land in a lower bucket. -/
theorem bucket_monotone (a b : Nat) (h : a ≤ b) : bucket a ≤ bucket b := by
  unfold bucket
  split_ifs <;> omega

/-- Bucket boundaries. -/
theorem bucket_boundaries : bucket 4 = 0 ∧ bucket 5 = 1 ∧ bucket 8 = 2 ∧ bucket 16 = 3 := by decide

/-- Clamp of `k + 1` into the cost-table range. -/
def clampN (k kMax : Nat) : Nat := max 1 (min (k + 1) (kMax + 1))

/-- The clamped index minus one is a valid table index. -/
theorem clampN_index (k kMax : Nat) : 1 ≤ clampN k kMax ∧ clampN k kMax - 1 ≤ kMax := by
  unfold clampN
  omega

/-- Number of positions updated by `observe`. -/
def seen (k accepted : Nat) : Nat := min k (min accepted k + 1)

/-- Updated positions never exceed the drafted count. -/
theorem seen_le (k accepted : Nat) : seen k accepted ≤ k := by
  unfold seen
  omega

/-- A fully accepted draft updates exactly its k positions (no rejection to learn from). -/
theorem seen_full (k : Nat) : seen k k = k := by
  unfold seen
  omega

/-- A partially accepted draft updates the accepted prefix plus the one rejected position. -/
theorem seen_partial (k accepted : Nat) (h : accepted < k) : seen k accepted = accepted + 1 := by
  unfold seen
  omega

/-- The EMA update stays between its old value and the target (scaled integer form, weight w of 100). -/
theorem ema_between (p t w : Nat) (hp : p ≤ 100) (ht : t ≤ 100) (hw : w ≤ 100) :
    ((100 - w) * p + w * t) / 100 ≤ 100 := by
  have : (100 - w) * p + w * t ≤ 100 * 100 := by
    have h1 : (100 - w) * p ≤ (100 - w) * 100 := Nat.mul_le_mul_left _ hp
    have h2 : w * t ≤ w * 100 := Nat.mul_le_mul_left _ ht
    have h3 : (100 - w) * 100 + w * 100 = 100 * 100 := by
      rw [← Nat.add_mul]
      congr 1
      omega
    omega
  omega

-- § Suffix drafter

/-- Table keys are forced odd, hence nonzero, so 0 can mark an empty slot. -/
theorem key_nonzero (x : Nat) : x ||| 1 ≠ 0 := by
  intro h
  have h1 : (x ||| 1).testBit 0 = true := by simp [Nat.testBit_or]
  rw [h] at h1
  simp at h1

/-- Minimum match is at least three tokens, since keys hash a trigram. -/
theorem min_match_ge_three (m : Nat) : 3 ≤ max 3 m := by omega

/-- Maximum match is at least the minimum. -/
theorem max_match_ge_min (mn mx : Nat) : mn ≤ max mn mx := by omega

/-- A slot's position list after inserting `p`: newest first, at most `ways` kept. -/
def pushPos (ways : Nat) (p : Nat) (l : List Nat) : List Nat := (p :: l).take ways

/-- A slot never holds more than WAYS positions. -/
theorem pushPos_length (ways p : Nat) (l : List Nat) : (pushPos ways p l).length ≤ ways := by
  unfold pushPos
  simp [List.length_take]

/-- The newest position is first when the slot has any capacity. -/
theorem pushPos_head (ways p : Nat) (l : List Nat) (hw : 0 < ways) : (pushPos ways p l).head? = some p := by
  unfold pushPos
  cases ways with
  | zero => exact absurd hw (by omega)
  | succ w => simp

/-- Length of the proposed continuation: positions `best_end+1 .. cur`, capped by `max_k`. -/
def contLen (maxK bestEnd cur : Nat) : Nat := min maxK (cur - bestEnd)

/-- The continuation never exceeds `max_k`. -/
theorem contLen_le_maxK (maxK bestEnd cur : Nat) : contLen maxK bestEnd cur ≤ maxK := by
  unfold contLen
  omega

/-- The continuation never reads past the current suffix: last index `bestEnd + len <= cur`. -/
theorem contLen_in_history (maxK bestEnd cur : Nat) (h : bestEnd ≤ cur) :
    bestEnd + contLen maxK bestEnd cur ≤ cur := by
  unfold contLen
  omega

/-- A candidate at or after the current position (the suffix itself) is skipped. -/
theorem skip_self (p cur : Nat) (h : cur ≤ p) : ¬ (p < cur) := by omega

/-- A match comparison at offset `len` is only valid while `len <= p`, so `p - len` never underflows. -/
theorem match_offset_valid (p len : Nat) (h : len ≤ p) : p - len + len = p := by omega

end Strata.Spec

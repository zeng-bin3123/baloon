import Mathlib

/-!
# MODULE: suffix_hash_drafter
A speculative-decoding draft source. It keeps the token history, indexes every 3-token suffix in an
open-addressed hash table (each slot remembers the last `WAYS` end positions), and proposes a
continuation by finding the longest earlier match of the current suffix.

## Source semantics
* `min_match' = max 3 min_match`, `max_match' = max min_match max_match`.
* `key_at` is never 0 (low bit forced), so 0 marks an empty slot.
* Slot position lists are most-recent-first, capped at `WAYS`, and shifted on every insert.
* `propose` skips positions `≥ cur`, measures match length bounded by `max_match'` and by the
  start of history, keeps the strictly longest (ties keep the newer), requires `≥ min_match'`,
  and copies at most `max_k` tokens from `best_end+1` up to and including `cur`.

## Form template identifiers
* SuffixDrafter → Drafter, hist_ → history, table_ → slotTable, WAYS → ways,
  min_match_ → minMatch, max_match_ → maxMatch, propose → proposeContinuation,
  append → appendTokens, find_slot → lookupSlot.
-/

namespace Strata.Spec.SuffixDrafter

-- § constructor parameters

/-- Effective minimum match length of the constructor: `max 3 min_match`. -/
def effMin (minMatch : Nat) : Nat := max 3 minMatch

/-- Effective maximum match length of the constructor: `max min_match max_match`. -/
def effMax (minMatch maxMatch : Nat) : Nat := max minMatch maxMatch

/-- The effective minimum match length is always at least 3 (a key covers three tokens). -/
theorem effMin_ge_three (m : Nat) : 3 ≤ effMin m := by
  unfold effMin; omega

/-- The effective maximum is at least the raw minimum. -/
theorem effMax_ge_raw_min (m M : Nat) : m ≤ effMax m M := by
  unfold effMax; omega

/-- The effective maximum is at least the raw maximum. -/
theorem effMax_ge_raw_max (m M : Nat) : M ≤ effMax m M := by
  unfold effMax; omega

-- § hash key never zero

/-- The key finalizer forces the low bit, so it is never zero (0 means an empty slot). -/
theorem key_never_zero (x : Nat) : (x ||| 1) ≠ 0 := by
  intro h
  have h1 := congrArg (fun n => n.testBit 0) h
  simp [Nat.testBit_or] at h1

-- § slot way-shifting

/-- Insert an end position at the front of a slot's position list, truncating to `ways`. -/
def pushPos (ways e : Nat) (l : List Nat) : List Nat := (e :: l).take ways

/-- A slot never holds more than `ways` positions. -/
theorem pushPos_length_le (ways e : Nat) (l : List Nat) : (pushPos ways e l).length ≤ ways := by
  simp [pushPos, List.length_take]

/-- The count after insert is `min (n+1) ways`, matching `if (s->n < WAYS) ++s->n`. -/
theorem pushPos_length (ways e : Nat) (l : List Nat) :
    (pushPos ways e l).length = min ways (l.length + 1) := by
  simp [pushPos, List.length_take]

/-- With at least one way, the newest position sits at index 0. -/
theorem pushPos_head (ways e : Nat) (l : List Nat) (h : 1 ≤ ways) :
    (pushPos ways e l).head? = some e := by
  obtain ⟨w, rfl⟩ : ∃ w, ways = w + 1 := ⟨ways - 1, by omega⟩
  unfold pushPos
  rfl

-- § match length

/-- Match length loop: compare `h (p-len)` with `h (cur-len)` while `len ≤ p` and fuel remains.
Fuel stands for `max_match - len`. -/
def matchGo (h : Nat → Int) (p cur : Nat) : Nat → Nat → Nat
  | 0, len => len
  | fuel + 1, len =>
    if len ≤ p ∧ h (p - len) = h (cur - len) then matchGo h p cur fuel (len + 1) else len

/-- The loop never returns less than the starting length. -/
theorem matchGo_ge (h : Nat → Int) (p cur : Nat) :
    ∀ fuel len, len ≤ matchGo h p cur fuel len := by
  intro fuel
  induction fuel with
  | zero => intro len; simp [matchGo]
  | succ n ih =>
    intro len
    simp only [matchGo]
    split
    · have := ih (len + 1); omega
    · exact le_refl _

/-- The loop is bounded by starting length plus fuel, i.e. by `max_match`. -/
theorem matchGo_le (h : Nat → Int) (p cur : Nat) :
    ∀ fuel len, matchGo h p cur fuel len ≤ len + fuel := by
  intro fuel
  induction fuel with
  | zero => intro len; simp [matchGo]
  | succ n ih =>
    intro len
    simp only [matchGo]
    split
    · have := ih (len + 1); omega
    · omega

/-- Match length from scratch is at most `max_match`. -/
theorem matchLen_le_max (h : Nat → Int) (p cur maxM : Nat) :
    matchGo h p cur maxM 0 ≤ maxM := by
  have := matchGo_le h p cur maxM 0
  omega

-- § best candidate selection

/-- Fold over candidates (length, end), most recent first; strictly longer replaces. -/
def bestOf : List (Nat × Nat) → Nat × Nat → Nat × Nat
  | [], acc => acc
  | c :: cs, acc => bestOf cs (if c.1 > acc.1 then c else acc)

/-- The selected length never decreases below the initial accumulator's length. -/
theorem bestOf_ge (cs : List (Nat × Nat)) :
    ∀ acc : Nat × Nat, acc.1 ≤ (bestOf cs acc).1 := by
  induction cs with
  | nil => intro acc; simp [bestOf]
  | cons c cs ih =>
    intro acc
    simp only [bestOf]
    split
    · have := ih c; omega
    · exact ih acc

/-- Ties keep the accumulator (the newer, since candidates are scanned newest first). -/
theorem bestOf_tie_keeps_acc (c acc : Nat × Nat) (h : c.1 = acc.1) :
    bestOf [c] acc = acc := by
  simp [bestOf, h]

-- § continuation copy

/-- Tokens copied by `propose`: indices `best_end+1 ..` up to `cur`, at most `max_k` of them. -/
def continuation (hist : Nat → Int) (bestEnd cur maxK : Nat) : List Int :=
  (List.range (min maxK (cur - bestEnd))).map (fun i => hist (bestEnd + 1 + i))

/-- The number of drafted tokens is `min max_k (cur - best_end)`. -/
theorem continuation_length (hist : Nat → Int) (bestEnd cur maxK : Nat) :
    (continuation hist bestEnd cur maxK).length = min maxK (cur - bestEnd) := by
  simp [continuation]

/-- `propose` never returns more than `max_k` tokens. -/
theorem continuation_le_maxK (hist : Nat → Int) (bestEnd cur maxK : Nat) :
    (continuation hist bestEnd cur maxK).length ≤ maxK := by
  rw [continuation_length]; omega

/-- Every drafted index is a valid history index (at most `cur`). -/
theorem continuation_index_le_cur (bestEnd cur maxK i : Nat) (hb : bestEnd < cur)
    (hi : i < min maxK (cur - bestEnd)) : bestEnd + 1 + i ≤ cur := by
  omega

/-- A match whose end is strictly before `cur` always yields at least one token when `max_k ≥ 1`. -/
theorem continuation_nonempty (hist : Nat → Int) (bestEnd cur maxK : Nat)
    (hb : bestEnd < cur) (hk : 1 ≤ maxK) :
    0 < (continuation hist bestEnd cur maxK).length := by
  rw [continuation_length]; omega

-- § propose gating

/-- `propose` accepts only when the best length reaches the effective minimum. -/
def accepts (bestLen minMatch : Nat) : Bool := decide (minMatch ≤ bestLen)

/-- Acceptance implies the length is at least 3 whenever the minimum is `effMin`. -/
theorem accepts_len_ge_three (bestLen m : Nat) (h : accepts bestLen (effMin m) = true) :
    3 ≤ bestLen := by
  simp [accepts, effMin] at h
  omega

end Strata.Spec.SuffixDrafter

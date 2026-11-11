/-!
# MODULE: suffix_lookup_drafter
SOURCES: src/spec/suffix_drafter.cpp
A suffix-lookup drafter: hashes every 3-token window of the history into an open-addressing table that
remembers the last WAYS end positions, and proposes the continuation of the longest earlier match of the
current suffix.

## Source semantics
* Table capacity: the smallest power of two ≥ `2·capacity_tokens` (load factor ≤ 0.5), `mask = cap − 1`;
  linear probing from `key & mask`.
* `key_at` is OR-ed with 1 so it is never 0 (0 marks an empty slot).
* Each slot keeps the most recent WAYS positions, newest first; `n` saturates at WAYS.
* `propose` compares backwards from each candidate position `p < cur` while `len < max_match`,
  `len ≤ p` and tokens agree; picks the longest (newest on ties); refuses below `min_match`; drafts
  `hist[best_end+1 .. cur]` truncated to `max_k`.
* Constructor: `min_match_ = max(3, min_match)` but `max_match_ = max(min_match, max_match)` uses the RAW
  `min_match` — so a configuration like (1, 2) yields `max_match_ = 2 < min_match_ = 3` and the drafter can
  never propose (proved below).

## Form template identifiers
SuffixDrafter ctor capacity loop → growCap / tableCap, key_at `| 1` → keyNonzero,
Slot::pos shift → pushPos, propose's match loop → matchFrom / matchLen, propose's copy-out → draftOut,
min_match_ / max_match_ → effMin / effMax
-/

namespace Strata.SuffixDrafter

-- § table capacity

/-- `while (cap < target) cap <<= 1`, with fuel. -/
def growCap (target : Nat) : Nat → Nat → Nat
  | 0, cap => cap
  | fuel + 1, cap => if cap < target then growCap target fuel (2 * cap) else cap

/-- The constructor's table size: start at 1, grow past `2·capacity_tokens`. -/
def tableCap (capTokens : Nat) : Nat := growCap (2 * capTokens) (2 * capTokens) 1

/-- Powers of two. -/
def IsPow2 (n : Nat) : Prop := ∃ k, n = 2 ^ k

/-- Doubling preserves being a power of two. -/
theorem growCap_pow2 (t : Nat) : ∀ fuel cap, IsPow2 cap → IsPow2 (growCap t fuel cap) := by
  intro fuel
  induction fuel with
  | zero => intro cap h; exact h
  | succ f ih =>
    intro cap h
    obtain ⟨k, hk⟩ := h
    simp only [growCap]
    split
    · exact ih (2 * cap) ⟨k + 1, by rw [hk, Nat.pow_succ, Nat.mul_comm]⟩
    · exact ⟨k, hk⟩

/-- With enough fuel the loop reaches the target. -/
theorem growCap_ge (t : Nat) : ∀ fuel cap, 1 ≤ cap → t ≤ fuel + cap → t ≤ growCap t fuel cap := by
  intro fuel
  induction fuel with
  | zero => intro cap _ h; simp only [growCap]; omega
  | succ f ih =>
    intro cap h1 h2
    simp only [growCap]
    split
    · exact ih (2 * cap) (by omega) (by omega)
    · omega

/-- The loop never overshoots by a factor of two or more. -/
theorem growCap_lt (t : Nat) : ∀ fuel cap, cap < 2 * t → growCap t fuel cap < 2 * t := by
  intro fuel
  induction fuel with
  | zero => intro cap h; exact h
  | succ f ih =>
    intro cap h
    simp only [growCap]
    split
    · exact ih (2 * cap) (by omega)
    · exact h

/-- The table size is a power of two, so `& mask` is a valid modular index. -/
theorem tableCap_pow2 (c : Nat) : IsPow2 (tableCap c) :=
  growCap_pow2 _ _ 1 ⟨0, rfl⟩

/-- Load factor at most 0.5: the table holds at least twice the nominal capacity. -/
theorem tableCap_ge (c : Nat) : 2 * c ≤ tableCap c :=
  growCap_ge _ _ 1 (Nat.le_refl 1) (by omega)

/-- ...and is the SMALLEST such power of two (less than four times capacity), for capacity ≥ 1. -/
theorem tableCap_lt (c : Nat) (hc : 1 ≤ c) : tableCap c < 4 * c := by
  have := growCap_lt (2 * c) (2 * c) 1 (by omega)
  unfold tableCap
  omega

/-- The table is never empty, so `mask = cap − 1` is well-defined. -/
theorem tableCap_pos (c : Nat) : 0 < tableCap c := by
  obtain ⟨k, hk⟩ := tableCap_pow2 c
  rw [hk]
  exact Nat.pow_pos (by decide)

/-- Every probe index `i & mask` (= `i % cap`) is in range. -/
theorem probe_in_range (c i : Nat) : i % tableCap c < tableCap c :=
  Nat.mod_lt _ (tableCap_pos c)

/-- Capacity 0 still gives a one-slot table. -/
theorem tableCap_zero : tableCap 0 = 1 := by decide

/-- Example: 4,096 nominal tokens → 8,192 slots. -/
theorem tableCap_4096 : tableCap 4096 = 8192 := by decide

-- § hash keys

/-- `key | 1` is never 0, so a live key is never mistaken for an empty slot. -/
theorem keyNonzero (x : Nat) : x ||| 1 ≠ 0 := by
  intro h
  have h0 : (x ||| 1).testBit 0 = (0 : Nat).testBit 0 := by rw [h]
  simp at h0

-- § slot ring of recent positions

/-- `pos[w] = pos[w-1] … pos[0] = end`: newest first, oldest dropped past `ways`. -/
def pushPos (ways : Nat) (pos : List Nat) (e : Nat) : List Nat := (e :: pos).take ways

/-- A slot never holds more than WAYS positions. -/
theorem pushPos_length_le (ways : Nat) (pos : List Nat) (e : Nat) : (pushPos ways pos e).length ≤ ways := by
  unfold pushPos; simp only [List.length_take, List.length_cons]; omega

/-- The count saturates: `n ← min(n + 1, WAYS)`, exactly the C++ `if (n < WAYS) ++n`. -/
theorem pushPos_length (ways : Nat) (pos : List Nat) (e : Nat) :
    (pushPos ways pos e).length = min ways (pos.length + 1) := by
  unfold pushPos; simp

/-- The newest position is first, so the "most recent first" tie rule in `propose` holds. -/
theorem pushPos_head (ways : Nat) (pos : List Nat) (e : Nat) (hw : 0 < ways) :
    (pushPos ways pos e).head? = some e := by
  unfold pushPos
  obtain ⟨w, rfl⟩ : ∃ w, ways = w + 1 := ⟨ways - 1, by omega⟩
  simp

-- § match length

/-- The backward comparison loop: `while (len < max_match && len <= p && eq(len)) ++len`, with
`eq len` meaning `hist[p-len] == hist[cur-len]`. Fuel = `max_match`. -/
def matchFrom (eq : Nat → Bool) (p : Nat) : Nat → Nat → Nat
  | 0, len => len
  | fuel + 1, len => if len ≤ p ∧ eq len = true then matchFrom eq p fuel (len + 1) else len

/-- The match length of one candidate. -/
def matchLen (eq : Nat → Bool) (p maxM : Nat) : Nat := matchFrom eq p maxM 0

/-- The loop runs at most `fuel` more steps. -/
theorem matchFrom_le_fuel (eq : Nat → Bool) (p : Nat) :
    ∀ fuel len, matchFrom eq p fuel len ≤ len + fuel := by
  intro fuel
  induction fuel with
  | zero => intro len; simp [matchFrom]
  | succ f ih =>
    intro len
    simp only [matchFrom]
    split
    · have := ih (len + 1); omega
    · omega

/-- The loop never reads before the start of history: the result is at most `p + 1`. -/
theorem matchFrom_le_pos (eq : Nat → Bool) (p : Nat) :
    ∀ fuel len, len ≤ p + 1 → matchFrom eq p fuel len ≤ p + 1 := by
  intro fuel
  induction fuel with
  | zero => intro len h; simpa [matchFrom] using h
  | succ f ih =>
    intro len h
    simp only [matchFrom]
    split
    · rename_i hc; exact ih (len + 1) (by omega)
    · exact h

/-- Every counted position really matched. -/
theorem matchFrom_all_eq (eq : Nat → Bool) (p : Nat) :
    ∀ fuel len, (∀ j < len, eq j = true) → ∀ j < matchFrom eq p fuel len, eq j = true := by
  intro fuel
  induction fuel with
  | zero => intro len h j hj; exact h j (by simpa [matchFrom] using hj)
  | succ f ih =>
    intro len h j hj
    simp only [matchFrom] at hj
    split at hj
    · rename_i hc
      apply ih (len + 1) _ j hj
      intro i hi
      by_cases hil : i < len
      · exact h i hil
      · have : i = len := by omega
        subst this; exact hc.2
    · exact h j hj

/-- Match length is bounded by `max_match`. -/
theorem matchLen_le_max (eq : Nat → Bool) (p maxM : Nat) : matchLen eq p maxM ≤ maxM := by
  have := matchFrom_le_fuel eq p maxM 0
  unfold matchLen; omega

/-- Match length is bounded by the candidate's position + 1. -/
theorem matchLen_le_pos (eq : Nat → Bool) (p maxM : Nat) : matchLen eq p maxM ≤ p + 1 :=
  matchFrom_le_pos eq p maxM 0 (by omega)

/-- All positions inside the reported match agree. -/
theorem matchLen_sound (eq : Nat → Bool) (p maxM : Nat) : ∀ j < matchLen eq p maxM, eq j = true :=
  matchFrom_all_eq eq p maxM 0 (fun _ h => absurd h (Nat.not_lt_zero _))

-- § thresholds

/-- `min_match_ = max(3, min_match)`. -/
def effMin (minM : Nat) : Nat := max 3 minM

/-- `max_match_ = max(min_match, max_match)` — the RAW min_match. -/
def effMax (minM maxM : Nat) : Nat := max minM maxM

/-- When the configured min_match is ≥ 3 the thresholds are consistent. -/
theorem thresholds_consistent (minM maxM : Nat) (h : 3 ≤ minM) : effMin minM ≤ effMax minM maxM := by
  unfold effMin effMax
  simp only [Nat.max_def]
  split <;> split <;> omega

/-- Configuration finding: with `min_match = 1, max_match = 2` no match can ever reach `min_match_`,
so `propose` always returns 0 — a silent disable. -/
theorem config_1_2_never_proposes (eq : Nat → Bool) (p : Nat) :
    matchLen eq p (effMax 1 2) < effMin 1 := by
  have := matchLen_le_max eq p (effMax 1 2)
  have h1 : effMax 1 2 = 2 := by decide
  have h2 : effMin 1 = 3 := by decide
  omega

-- § drafted continuation

/-- `out[k++] = hist[q]` for `q` from `best_end + 1` up to the end, at most `max_k` tokens. -/
def draftOut (hist : List Int) (bestEnd maxK : Nat) : List Int := (hist.drop (bestEnd + 1)).take maxK

/-- Never more than `max_k` tokens are drafted. -/
theorem draftOut_le_maxK (hist : List Int) (bestEnd maxK : Nat) :
    (draftOut hist bestEnd maxK).length ≤ maxK := by
  unfold draftOut; simp only [List.length_take, List.length_drop]; omega

/-- Never reads past the current end of history (periodic overlap with the suffix is allowed). -/
theorem draftOut_le_history (hist : List Int) (bestEnd maxK : Nat) :
    (draftOut hist bestEnd maxK).length ≤ hist.length - (bestEnd + 1) := by
  unfold draftOut; simp only [List.length_take, List.length_drop]; omega

/-- Draft token i is exactly the history token after the match end. -/
theorem draftOut_get (hist : List Int) (bestEnd maxK i : Nat) (hi : i < maxK) :
    (draftOut hist bestEnd maxK)[i]? = hist[bestEnd + 1 + i]? := by
  unfold draftOut
  simp [List.getElem?_drop, hi]

end Strata.SuffixDrafter

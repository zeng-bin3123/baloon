/-!
# MODULE: suffix_drafter
SOURCES: src/spec/suffix_drafter.cpp
Prompt-lookup drafter: an open-addressed hash table keyed by the last three tokens remembers the most recent
positions where that trigram ended; `propose` extends the longest match and copies the continuation.

## Source semantics
* Keys are forced non-zero (`| 1`) because 0 marks an empty slot.
* The table has a power-of-two capacity of at least `2 * capacity_tokens` (load factor <= 0.5).
* Each slot keeps up to `WAYS` positions, newest first: appending shifts older ones down and drops the oldest.
* `propose` skips positions at or after the current suffix, requires a match of at least `min_match_`, caps the
  match at `max_match_`, and returns at most `max_k` tokens read from `best_end + 1 .. cur`.

## Form template identifiers
keyOf -> `key_at` (the `| 1` step)
shiftIn / nNext -> the `WAYS` shift in `SuffixDrafter::append`
matchAux -> the backwards `while (len < max_match_ && len <= p && ...)` loop in `propose`
contLen -> the `for (q = best_end + 1; q <= cur && k < max_k; ++q)` copy loop
-/

namespace Strata.Spec.Suffix

-- § constructor parameters

/-- `min_match_` after the constructor's `std::max(3, min_match)`. -/
def minMatch (m : Nat) : Nat := max 3 m

/-- `max_match_` after the constructor's `std::max(min_match, max_match)`. -/
def maxMatch (lo hi : Nat) : Nat := max lo hi

/-- A trigram is the shortest possible match: `min_match_ >= 3`. -/
theorem minMatch_ge_three (m : Nat) : 3 ≤ minMatch m := by
  unfold minMatch
  omega

/-- The maximum match is at least the requested minimum. -/
theorem maxMatch_ge_lo (lo hi : Nat) : lo ≤ maxMatch lo hi := by
  unfold maxMatch
  omega

/-- Table capacity: next power of two of `2 * capacity_tokens`. -/
def capFor (t : Nat) : Nat := Nat.nextPowerOfTwo (t * 2)

/-- Load factor at nominal capacity is at most one half. -/
theorem capFor_ge (t : Nat) : t * 2 ≤ capFor t := Nat.le_nextPowerOfTwo (t * 2)

-- § hashing

/-- A key is forced non-zero so that 0 can mark an empty slot. -/
def keyOf (x : Nat) : Nat := x ||| 1

/-- The forced-odd key is never the empty marker 0. -/
theorem keyOf_ne_zero (x : Nat) : keyOf x ≠ 0 := by
  unfold keyOf
  simp

/-- Linear probing stays inside the table when the capacity is positive. -/
theorem probe_lt (i cap : Nat) (h : 0 < cap) : (i + 1) % cap < cap :=
  Nat.mod_lt _ h

-- § append: the WAYS shift

/-- Newest-first position list after inserting `e`, truncated to `W` ways. -/
def shiftIn (l : List Nat) (e W : Nat) : List Nat := (e :: l).take W

/-- The way list never exceeds `WAYS` entries. -/
theorem shiftIn_length_le (l : List Nat) (e W : Nat) : (shiftIn l e W).length ≤ W := by
  unfold shiftIn
  exact List.length_take_le _ _

/-- The newest position is always in way 0. -/
theorem shiftIn_head (l : List Nat) (e w : Nat) : (shiftIn l e (w + 1)).head? = some e := by
  simp [shiftIn]

/-- The occupied-way count after an insert: `min(n + 1, WAYS)`. -/
def nNext (n W : Nat) : Nat := if n < W then n + 1 else n

/-- The count never exceeds `WAYS` (given it started within it). -/
theorem nNext_le (n W : Nat) (h : n ≤ W) : nNext n W ≤ W := by
  unfold nNext
  split <;> omega

/-- The count is non-decreasing. -/
theorem nNext_ge (n W : Nat) : n ≤ nNext n W := by
  unfold nNext
  split <;> omega

-- § propose: longest backward match

/-- Backward match length between positions `p` and `cur`, with loop fuel. -/
def matchAux (hist : Nat → Int) (maxm p cur : Nat) : Nat → Nat → Nat
  | 0, len => len
  | f + 1, len =>
    if len < maxm ∧ len ≤ p ∧ hist (p - len) = hist (cur - len)
    then matchAux hist maxm p cur f (len + 1) else len

/-- The match loop never exceeds `max_match_`, never reads before index 0, and never loses progress. -/
theorem matchAux_bounds (hist : Nat → Int) (maxm p cur : Nat) (f : Nat) :
    ∀ len, len ≤ maxm → len ≤ p + 1 →
      len ≤ matchAux hist maxm p cur f len ∧ matchAux hist maxm p cur f len ≤ maxm ∧
      matchAux hist maxm p cur f len ≤ p + 1 := by
  induction f with
  | zero =>
    intro len h1 h2
    simp only [matchAux]
    exact ⟨Nat.le_refl _, h1, h2⟩
  | succ f ih =>
    intro len h1 h2
    simp only [matchAux]
    split
    · rename_i hc
      obtain ⟨c1, c2, _⟩ := hc
      have := ih (len + 1) (by omega) (by omega)
      omega
    · exact ⟨Nat.le_refl _, h1, h2⟩

-- § propose: continuation copy

/-- Number of tokens copied from `best_end + 1 .. cur`, capped by `max_k`. -/
def contLen (bestEnd cur maxK : Nat) : Nat := min maxK (cur - bestEnd)

/-- The proposal never exceeds the caller's `max_k` (the output buffer size). -/
theorem contLen_le_maxK (bestEnd cur maxK : Nat) : contLen bestEnd cur maxK ≤ maxK := by
  unfold contLen
  omega

/-- The proposal never reads past the current suffix. -/
theorem contLen_le_span (bestEnd cur maxK : Nat) : contLen bestEnd cur maxK ≤ cur - bestEnd := by
  unfold contLen
  omega

/-- A genuine earlier match (`best_end < cur`) with room for a draft yields at least one token. -/
theorem contLen_pos (bestEnd cur maxK : Nat) (h : bestEnd < cur) (hk : 0 < maxK) :
    0 < contLen bestEnd cur maxK := by
  unfold contLen
  omega

/-- Final count returned by `propose`: zero on a guard failure, else the continuation length. -/
def proposeCount (n maxK bestLen minM bestEnd cur : Nat) : Nat :=
  if n < 4 ∨ maxK = 0 ∨ bestLen < minM then 0 else contLen bestEnd cur maxK

/-- `propose` never writes more than `max_k` tokens. -/
theorem proposeCount_le (n maxK bestLen minM bestEnd cur : Nat) :
    proposeCount n maxK bestLen minM bestEnd cur ≤ maxK := by
  unfold proposeCount
  split
  · omega
  · exact contLen_le_maxK _ _ _

end Strata.Spec.Suffix

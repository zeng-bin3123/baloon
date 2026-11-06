/-!
# MODULE: suffix_drafter
SOURCES: src/spec/suffix_drafter.cpp
A lookup (prompt-lookup / suffix) drafter: an open-addressing hash table keyed on the last three tokens stores up
to `WAYS` most-recent end positions per key; `propose` finds the longest match of the current suffix against
history and copies its continuation.

## Source semantics
* `min_match_ = max(3, min_match)`, `max_match_ = max(min_match, max_match)`.
* The table capacity is the smallest power of two that is at least `2 * capacity_tokens`.
* The key is a mixed hash with its low bit forced to 1, so `0` marks an empty slot.
* `append` shifts the way list right and inserts the new end position at the front; `n` grows up to `WAYS`.
* `propose` scans the ways from most recent, takes a strictly longer match only (ties keep the newer), requires
  length at least `min_match_`, then copies up to `max_k` tokens following the match, never past the current end.

## Form template identifiers
role mapping (source names -> abstract names):
  SuffixDrafter::append   -> history_append
  SuffixDrafter::propose  -> suffix_propose
  Slot::pos / WAYS        -> candidate_positions
  key_at                  -> trigram_key
  find_slot               -> probe_slot
  min_match_/max_match_   -> match_length_bounds
-/

namespace Strata.SuffixDrafter

-- § effective match bounds

/-- Effective minimum match length. -/
def minEff (minMatch : Nat) : Nat := max 3 minMatch

/-- Effective maximum match length. -/
def maxEff (minMatch maxMatch : Nat) : Nat := max minMatch maxMatch

/-- A match is never accepted below three tokens. -/
theorem minEff_ge_three (m : Nat) : 3 ≤ minEff m := by
  unfold minEff; omega

/-- The effective minimum is at least the requested one. -/
theorem minEff_ge_arg (m : Nat) : m ≤ minEff m := by
  unfold minEff; omega

/-- The effective maximum is at least both requested values. -/
theorem maxEff_ge (m x : Nat) : m ≤ maxEff m x ∧ x ≤ maxEff m x := by
  unfold maxEff; omega

/-- When the configured bounds are ordered and at least 3, they are used as given. -/
theorem bounds_ordered (m x : Nat) (h3 : 3 ≤ m) (hx : m ≤ x) :
    minEff m = m ∧ maxEff m x = x ∧ minEff m ≤ maxEff m x := by
  unfold minEff maxEff; omega

/-- If the effective maximum is below the effective minimum, no match can ever be proposed. -/
theorem no_proposal_when_cap_below_min (m x len : Nat) (h : maxEff m x < minEff m)
    (hl : len ≤ maxEff m x) : ¬ (minEff m ≤ len) := by
  omega

-- § table capacity

/-- Doubling loop `while (cap < m) cap <<= 1`, bounded by a fuel count. -/
def growCap (m : Nat) : Nat → Nat → Nat
  | 0, cap => cap
  | f + 1, cap => if cap < m then growCap m f (2 * cap) else cap

/-- The doubling loop reaches the target whenever the fuel suffices. -/
theorem growCap_ge (m : Nat) : ∀ f cap, m ≤ cap * 2 ^ f → m ≤ growCap m f cap := by
  intro f
  induction f with
  | zero =>
    intro cap h
    simp only [growCap]
    simpa using h
  | succ f ih =>
    intro cap h
    simp only [growCap]
    by_cases hc : cap < m
    · rw [if_pos hc]
      have e : cap * 2 ^ (f + 1) = 2 * cap * 2 ^ f := by
        rw [Nat.pow_succ, Nat.mul_comm (2 ^ f) 2, ← Nat.mul_assoc, Nat.mul_comm cap 2]
      apply ih
      calc m ≤ cap * 2 ^ (f + 1) := h
        _ = 2 * cap * 2 ^ f := e
    · rw [if_neg hc]
      omega

/-- The doubling loop stops immediately when the capacity already suffices. -/
theorem growCap_stop (m f cap : Nat) (h : m ≤ cap) : growCap m (f + 1) cap = cap := by
  simp only [growCap]
  rw [if_neg (by omega)]

-- § never-zero key

/-- Model of `x | 1`: set the lowest bit. -/
def setLow (x : Nat) : Nat := 2 * (x / 2) + 1

/-- A key is never 0, so 0 can mark an empty slot. -/
theorem setLow_ne_zero (x : Nat) : setLow x ≠ 0 := by
  unfold setLow; omega

/-- A key is odd. -/
theorem setLow_odd (x : Nat) : setLow x % 2 = 1 := by
  unfold setLow; omega

/-- Setting the low bit twice changes nothing. -/
theorem setLow_idem (x : Nat) : setLow (setLow x) = setLow x := by
  unfold setLow; omega

/-- Setting the low bit changes the value by at most one. -/
theorem setLow_close (x : Nat) : x ≤ setLow x ∧ setLow x ≤ x + 1 := by
  unfold setLow; omega

-- § way list

/-- Insert an end position at the front of the way list, dropping the oldest beyond `W`. -/
def pushWay (ways : List Nat) (p W : Nat) : List Nat := (p :: ways).take W

/-- The way list never exceeds `W` entries. -/
theorem pushWay_length (ways : List Nat) (p W : Nat) :
    (pushWay ways p W).length = min W (ways.length + 1) := by
  simp [pushWay]

/-- The newest position is first. -/
theorem pushWay_head (ways : List Nat) (p W : Nat) :
    (pushWay ways p (W + 1)).head? = some p := by
  simp [pushWay]

/-- The counter `n`: grows by one until it reaches `WAYS`. -/
def bumpN (n W : Nat) : Nat := if n < W then n + 1 else n

/-- The counter tracks the list length: it is `min (n + 1) W` while `n <= W`. -/
theorem bumpN_eq_min (n W : Nat) (h : n ≤ W) : bumpN n W = min (n + 1) W := by
  unfold bumpN
  split <;> omega

/-- The counter never exceeds `W` once it starts within `W`. -/
theorem bumpN_le (n W : Nat) (h : n ≤ W) : bumpN n W ≤ W := by
  unfold bumpN
  split <;> omega

-- § match length

/-- Backward match length: extend while within the history start and `same len` holds, up to `fuel` steps. -/
def matchLen (same : Nat → Bool) (p : Nat) : Nat → Nat → Nat
  | len, 0 => len
  | len, f + 1 => if len ≤ p ∧ same len = true then matchLen same p (len + 1) f else len

/-- The match is at most `start + fuel` long. -/
theorem matchLen_le_fuel (same : Nat → Bool) (p : Nat) (f : Nat) :
    ∀ len, matchLen same p len f ≤ len + f := by
  induction f with
  | zero => intro len; simp [matchLen]
  | succ f ih =>
    intro len
    simp only [matchLen]
    split
    · have := ih (len + 1)
      omega
    · omega

/-- The match never reads before the start of history: length at most `p + 1`. -/
theorem matchLen_le_pos (same : Nat → Bool) (p : Nat) (f : Nat) :
    ∀ len, len ≤ p + 1 → matchLen same p len f ≤ p + 1 := by
  induction f with
  | zero => intro len h; simp [matchLen]; omega
  | succ f ih =>
    intro len hl
    simp only [matchLen]
    split
    · next h =>
      have h1 : len ≤ p := h.1
      have := ih (len + 1) (by omega)
      exact this
    · omega

/-- The match is at least the starting length. -/
theorem matchLen_ge_start (same : Nat → Bool) (p : Nat) (f : Nat) :
    ∀ len, len ≤ matchLen same p len f := by
  induction f with
  | zero => intro len; simp [matchLen]
  | succ f ih =>
    intro len
    simp only [matchLen]
    split
    · have := ih (len + 1)
      omega
    · omega

-- § best candidate scan

/-- Scan step on (best length, best end): replace only by a strictly longer match. -/
def scanStep (s c : Nat × Nat) : Nat × Nat := if c.1 > s.1 then c else s

/-- The scan keeps the maximum length. -/
theorem scanStep_fst_ge (s c : Nat × Nat) : s.1 ≤ (scanStep s c).1 ∧ c.1 ≤ (scanStep s c).1 := by
  unfold scanStep
  split <;> omega

/-- Ties keep the incumbent (the more recent position, as ways are scanned newest first). -/
theorem scanStep_tie (s c : Nat × Nat) (h : c.1 = s.1) : scanStep s c = s := by
  unfold scanStep
  split
  · omega
  · rfl

/-- Scan over all candidates. -/
def scanAll (s : Nat × Nat) (cs : List (Nat × Nat)) : Nat × Nat := cs.foldl scanStep s

/-- The scanned best is at least the initial best. -/
theorem scanAll_ge_init (cs : List (Nat × Nat)) : ∀ s, s.1 ≤ (scanAll s cs).1 := by
  induction cs with
  | nil => intro s; simp [scanAll]
  | cons c cs ih =>
    intro s
    have h := ih (scanStep s c)
    have h2 := (scanStep_fst_ge s c).1
    simp only [scanAll, List.foldl_cons] at h ⊢
    omega

/-- The scanned best is at least every candidate's length. -/
theorem scanAll_ge_mem (cs : List (Nat × Nat)) : ∀ s, ∀ c ∈ cs, c.1 ≤ (scanAll s cs).1 := by
  induction cs with
  | nil => intro s c hc; simp at hc
  | cons c' cs ih =>
    intro s c hc
    rcases List.mem_cons.1 hc with h | h
    · subst h
      have h1 := scanAll_ge_init cs (scanStep s c)
      have h2 := (scanStep_fst_ge s c).2
      simp only [scanAll, List.foldl_cons] at h1 ⊢
      omega
    · have := ih (scanStep s c') c h
      simp only [scanAll, List.foldl_cons] at this ⊢
      exact this

-- § continuation copy

/-- Number of tokens copied after the match ending at `bestEnd`, with `cur` the current last index. -/
def contLen (cur bestEnd maxK : Nat) : Nat := min maxK (cur - bestEnd)

/-- The copy never exceeds the requested draft count. -/
theorem contLen_le_maxK (cur bestEnd maxK : Nat) : contLen cur bestEnd maxK ≤ maxK := by
  unfold contLen; omega

/-- The copy never reads past the current end of history. -/
theorem contLen_reads_in_history (cur bestEnd maxK : Nat) (h : bestEnd ≤ cur) :
    bestEnd + contLen cur bestEnd maxK ≤ cur := by
  unfold contLen; omega

/-- A match strictly before the current suffix yields a non-empty continuation when drafts are wanted. -/
theorem contLen_pos (cur bestEnd maxK : Nat) (h : bestEnd < cur) (hk : 1 ≤ maxK) :
    1 ≤ contLen cur bestEnd maxK := by
  unfold contLen; omega

end Strata.SuffixDrafter

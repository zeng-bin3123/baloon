/-!
# MODULE: suffix_drafter
Suffix-lookup drafter: a hash table keyed by the last three tokens remembers the most recent positions where each
trigram ended; `propose` extends the longest matching suffix and copies the continuation as a draft.

## Source semantics
* Table capacity is a power of two at least twice the nominal capacity, probing is `(i + 1) & mask`.
* A key is never zero (`mix(...) | 1`), so zero marks an empty slot.
* Each slot keeps its `WAYS` most recent end positions, newest first.
* The match length never exceeds `max_match` or the available history `p + 1`.
* The continuation copies at most `max_k` tokens and only reads history up to the current position.
* `min_match_ = max 3 min_match` but `max_match_ = max min_match max_match`, so a degenerate pair
  (`min_match < 3` and `max_match < 3`) can never match.

## Form template identifiers
* `Slot::pos` -> `pushWay`, `find_slot` probe -> `probeNext`, `key_at` -> `keyNonzero`
* `propose` match loop -> `matchAux`, continuation -> `contLen`
-/

namespace Strata.SuffixDrafter

-- § slot ways

/-- Insert a new end position at the front of a slot's way list and drop the oldest beyond `W` ways. -/
def pushWay (W : Nat) (l : List Nat) (x : Nat) : List Nat := (x :: l).take W

/-- A slot never holds more than `W` positions. -/
theorem pushWay_length_le (W : Nat) (l : List Nat) (x : Nat) : (pushWay W l x).length ≤ W := by
  unfold pushWay
  simp <;> omega

/-- The newest position is stored first. -/
theorem pushWay_head (W : Nat) (l : List Nat) (x : Nat) (hW : 0 < W) :
    (pushWay W l x).head? = some x := by
  unfold pushWay
  cases W with
  | zero => omega
  | succ w => simp

/-- The way counter `min (n + 1) W` matches the list length once the list is full or short. -/
theorem pushWay_length (W : Nat) (l : List Nat) (x : Nat) :
    (pushWay W l x).length = min (l.length + 1) W := by
  unfold pushWay
  simp [List.length_take] <;> omega

-- § table

/-- A key `mix(...) | 1` is never zero, so zero is a safe empty marker. -/
theorem keyNonzero (x : Nat) : (x ||| 1) ≠ 0 := by
  intro h
  first
    | (have h1 := Nat.or_eq_zero_iff.mp h; omega)
    | (have h1 : 1 ≤ x ||| 1 := Nat.right_le_or; omega)
    | simp at h

/-- A power-of-two table indexed by `& mask` stays in range. -/
theorem probe_in_range (k i : Nat) : i &&& (2 ^ k - 1) < 2 ^ k := by
  have hp : 0 < 2 ^ k := by
    first
      | exact Nat.two_pow_pos k
      | exact Nat.pos_pow_of_pos k (by decide)
  have h : 2 ^ k - 1 < 2 ^ k := by omega
  first
    | exact Nat.and_lt_two_pow i h
    | exact Nat.lt_of_le_of_lt Nat.and_le_right h

/-- The linear-probe successor `(i + 1) & mask` also stays in range. -/
theorem probeNext_in_range (k i : Nat) : (i + 1) &&& (2 ^ k - 1) < 2 ^ k :=
  probe_in_range k (i + 1)

-- § match loop

/-- The match loop: extend while within history and tokens agree, bounded by fuel `maxM - len`. -/
def matchAux (h : Nat → Nat) (p cur : Nat) : Nat → Nat → Nat
  | 0, len => len
  | fuel + 1, len =>
      if len ≤ p ∧ h (p - len) = h (cur - len) then matchAux h p cur fuel (len + 1) else len

/-- The loop never extends more than its fuel. -/
theorem matchAux_le_fuel (h : Nat → Nat) (p cur : Nat) :
    ∀ (fuel len : Nat), matchAux h p cur fuel len ≤ len + fuel
  | 0, len => by simp [matchAux]
  | fuel + 1, len => by
      have ih := matchAux_le_fuel h p cur fuel (len + 1)
      simp only [matchAux]
      split
      · omega
      · omega

/-- The loop never exceeds the available history `p + 1`. -/
theorem matchAux_le_hist (h : Nat → Nat) (p cur : Nat) :
    ∀ (fuel len : Nat), len ≤ p + 1 → matchAux h p cur fuel len ≤ p + 1
  | 0, len, hl => by simpa [matchAux] using hl
  | fuel + 1, len, hl => by
      simp only [matchAux]
      split
      · rename_i hc
        exact matchAux_le_hist h p cur fuel (len + 1) (by omega)
      · exact hl

/-- Starting from zero, a match is at most `max_match` long. -/
theorem match_le_max (h : Nat → Nat) (p cur maxM : Nat) :
    matchAux h p cur maxM 0 ≤ maxM := by
  have := matchAux_le_fuel h p cur maxM 0
  omega

-- § constructor clamps

/-- `min_match_` as built by the constructor. -/
def minMatch (a : Nat) : Nat := max 3 a

/-- `max_match_` as built by the constructor (it clamps against the raw `min_match`). -/
def maxMatch (a b : Nat) : Nat := max a b

/-- With a sane request (`min_match >= 3`) the effective bounds are ordered. -/
theorem bounds_ordered (a b : Nat) (ha : 3 ≤ a) : minMatch a ≤ maxMatch a b := by
  unfold minMatch maxMatch
  omega

/-- A degenerate request makes `max_match_` smaller than `min_match_`, so `propose` can never succeed. -/
theorem degenerate_never_matches (a b len : Nat) (ha : a < 3) (hb : b < 3)
    (hlen : len ≤ maxMatch a b) : len < minMatch a := by
  unfold minMatch maxMatch at *
  omega

-- § continuation

/-- Number of draft tokens copied: from `best_end + 1` up to `cur`, at most `max_k`. -/
def contLen (cur bestEnd maxK : Nat) : Nat := min maxK (cur - bestEnd)

/-- The draft never exceeds `max_k`. -/
theorem contLen_le_maxk (cur e k : Nat) : contLen cur e k ≤ k := by
  unfold contLen
  omega

/-- Every copied index `e + 1 + j` stays at or before the current position. -/
theorem cont_reads_in_history (cur e k j : Nat) (he : e < cur) (hj : j < contLen cur e k) :
    e + 1 + j ≤ cur := by
  unfold contLen at hj
  omega

end Strata.SuffixDrafter

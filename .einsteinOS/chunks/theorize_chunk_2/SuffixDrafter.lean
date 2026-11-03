/-!
# MODULE: suffix_drafter
Lookup drafter: hashes each 3-token suffix of the history into an open-addressed table that remembers the
WAYS most recent end positions; proposing finds the longest earlier match and copies its continuation.

## Source semantics
- min_match >= 3 and max_match >= min_match by construction.
- Slot probing is linear modulo the table size and stops after at most `size` probes.
- Inserting a position shifts older entries down and keeps at most WAYS of them.
- The proposal length is at most max_k and never reads beyond the current suffix end.

## Form template identifiers
min_match_ -> minMatch ; WAYS -> ways ; find_slot probe -> nextSlot ; Slot::n -> insertCount ;
continuation loop -> contLen
-/

namespace Strata.Suffix

-- § construction

/-- Effective minimum match length. -/
def minMatch (m : Nat) : Nat := max 3 m

/-- Effective maximum match length. -/
def maxMatch (mn mx : Nat) : Nat := max mn mx

/-- The minimum match is at least three. -/
theorem minMatch_ge (m : Nat) : 3 ≤ minMatch m := by unfold minMatch; omega

/-- The maximum match is at least the requested minimum. -/
theorem maxMatch_ge (mn mx : Nat) : mn ≤ maxMatch mn mx := by unfold maxMatch; omega

-- § probing

/-- Linear probe to the next slot. -/
def nextSlot (size i : Nat) : Nat := (i + 1) % size

/-- The next slot is a valid index of a nonempty table. -/
theorem nextSlot_lt (size i : Nat) (h : 0 < size) : nextSlot size i < size := by
  unfold nextSlot; exact Nat.mod_lt _ h

/-- Starting index of a probe sequence is valid. -/
theorem start_lt (size key : Nat) (h : 0 < size) : key % size < size := Nat.mod_lt _ h

-- § slot insertion

/-- Entry count after one more insertion, saturating at `ways`. -/
def insertCount (n ways : Nat) : Nat := if n < ways then n + 1 else n

/-- The count never exceeds `ways` once it starts within bounds. -/
theorem insertCount_le (n ways : Nat) (h : n ≤ ways) : insertCount n ways ≤ ways := by
  unfold insertCount; split <;> omega

/-- Insertion never lowers the count. -/
theorem insertCount_ge (n ways : Nat) : n ≤ insertCount n ways := by
  unfold insertCount; split <;> omega

/-- Shifting entries down: new position first, older ones follow, truncated to `ways`. -/
def pushFront (ways : Nat) (x : Nat) (l : List Nat) : List Nat := (x :: l).take ways

/-- The list stays within capacity. -/
theorem pushFront_length_le (ways x : Nat) (l : List Nat) : (pushFront ways x l).length ≤ ways := by
  unfold pushFront
  simp [List.length_take]

/-- With positive capacity the newest position is the head (most recent first). -/
theorem pushFront_head (ways x : Nat) (l : List Nat) (h : 0 < ways) : (pushFront ways x l).head? = some x := by
  unfold pushFront
  cases ways with
  | zero => omega
  | succ n => simp

-- § proposal

/-- Length of the copied continuation: positions bestEnd+1 .. cur, capped at maxK. -/
def contLen (bestEnd cur maxK : Nat) : Nat := min maxK (cur - bestEnd)

/-- Never more than max_k drafts. -/
theorem contLen_le (b c k : Nat) : contLen b c k ≤ k := by unfold contLen; omega

/-- The copied range stays within the history: bestEnd + len <= cur whenever bestEnd <= cur. -/
theorem contLen_in_range (b c k : Nat) (h : b ≤ c) : b + contLen b c k ≤ c := by
  unfold contLen; omega

end Strata.Suffix

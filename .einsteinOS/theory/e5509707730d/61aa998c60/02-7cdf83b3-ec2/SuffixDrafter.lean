/-!
# MODULE: suffix_drafter
SOURCES: src/spec/suffix_drafter.cpp
An n-gram suffix lookup drafter: an open-addressing hash table keyed by the last three tokens, each slot keeping
the most recent `WAYS` end positions; `propose` finds the longest earlier match and copies its continuation.

## Source semantics
* The table capacity is the first power of two `>= 2 * capacity_tokens`, so `key & mask == key % capacity`.
* A key is never 0 (`| 1`), so 0 can mean "empty slot".
* A slot keeps at most `WAYS` positions, newest first.
* A match is extended while indices stay valid; the copied continuation never reads past the current suffix.

## Form template identifiers
`cap <<= 1` loop -> grow; `mask_` -> mask; `Slot::pos/n` -> push/List; match loop -> mlLoop; copy count -> copyCount.
-/

namespace Strata.SuffixDrafter

-- § table capacity

/-- Double `c` until it reaches `target` (bounded by `fuel`). -/
def grow : Nat → Nat → Nat → Nat
  | 0, c, _ => c
  | f + 1, c, t => if c < t then grow f (2 * c) t else c

/-- With enough doublings available the capacity reaches the target. -/
theorem grow_ge (t : Nat) (f : Nat) : ∀ c, t ≤ c * 2 ^ f → t ≤ grow f c t := by
  induction f with
  | zero => intro c h; simpa [grow] using h
  | succ f ih =>
    intro c h
    simp only [grow]
    by_cases hc : c < t
    · rw [if_pos hc]
      apply ih
      have e : 2 * c * 2 ^ f = c * 2 ^ (f + 1) := by
        rw [Nat.pow_succ]
        ac_rfl
      rw [e]
      exact h
    · rw [if_neg hc]
      omega

/-- Starting from a power of two the capacity stays a power of two. -/
theorem grow_pow2 (t : Nat) (f : Nat) : ∀ k, ∃ j, grow f (2 ^ k) t = 2 ^ j := by
  induction f with
  | zero => intro k; exact ⟨k, rfl⟩
  | succ f ih =>
    intro k
    simp only [grow]
    by_cases hc : 2 ^ k < t
    · rw [if_pos hc]
      have e : 2 * 2 ^ k = 2 ^ (k + 1) := by rw [Nat.pow_succ, Nat.mul_comm]
      rw [e]
      exact ih (k + 1)
    · rw [if_neg hc]
      exact ⟨k, rfl⟩

/-- For a power-of-two capacity, masking equals the remainder, so probing `(i + 1) & mask` stays in the table. -/
theorem mask_is_mod (x k : Nat) : x &&& (2 ^ k - 1) = x % 2 ^ k := Nat.and_two_pow_sub_one_eq_mod x k

/-- The next probe index is always in range. -/
theorem probe_in_range (i cap : Nat) (hcap : 0 < cap) : (i + 1) % cap < cap := Nat.mod_lt _ hcap

-- § keys

/-- `key | 1` is never zero, so 0 is a safe "empty" marker. -/
theorem or_one_ne_zero (x : Nat) : x ||| 1 ≠ 0 := by
  intro h
  have h2 : (x ||| 1).testBit 0 = true := by simp [Nat.testBit_or]
  rw [h] at h2
  simp at h2

-- § slot positions

/-- Insert a position, keeping the `ways` newest. -/
def push (ways : Nat) (p : Nat) (l : List Nat) : List Nat := (p :: l).take ways

/-- A slot never holds more than `WAYS` positions. -/
theorem push_length_le (ways p : Nat) (l : List Nat) : (push ways p l).length ≤ ways := by
  unfold push
  simp [List.length_take]

/-- The new position is the most recent when the slot has any capacity. -/
theorem push_head (ways p : Nat) (l : List Nat) (h : 0 < ways) : (push ways p l).head? = some p := by
  unfold push
  cases ways with
  | zero => omega
  | succ w => rfl

/-- The counter `n = min(n + 1, WAYS)` equals the new list length when the slot was within capacity. -/
theorem push_length (ways p : Nat) (l : List Nat) (h : l.length ≤ ways) :
    (push ways p l).length = min (l.length + 1) ways := by
  unfold push
  simp [List.length_take]
  omega

-- § match extension and continuation

/-- Extend a match while `len ≤ p` and the tokens agree, up to `fuel` steps. -/
def mlLoop (h : Nat → Nat) (p cur : Nat) : Nat → Nat → Nat
  | 0, len => len
  | f + 1, len => if len ≤ p ∧ h (p - len) = h (cur - len) then mlLoop h p cur f (len + 1) else len

/-- The match grows by at most the fuel (`max_match`). -/
theorem mlLoop_le (h : Nat → Nat) (p cur : Nat) (f : Nat) : ∀ len, mlLoop h p cur f len ≤ len + f := by
  induction f with
  | zero => intro len; simp [mlLoop]
  | succ f ih =>
    intro len
    simp only [mlLoop]
    split
    · have := ih (len + 1)
      omega
    · omega

/-- The match never exceeds `p + 1` tokens, so `hist[p - len]` is always a valid index. -/
theorem mlLoop_le_p (h : Nat → Nat) (p cur : Nat) (f : Nat) : ∀ len, len ≤ p + 1 → mlLoop h p cur f len ≤ p + 1 := by
  induction f with
  | zero => intro len hl; simpa [mlLoop] using hl
  | succ f ih =>
    intro len hl
    simp only [mlLoop]
    split
    · rename_i hc
      obtain ⟨hlt, _⟩ := hc
      exact ih (len + 1) (by omega)
    · exact hl

/-- The match never shrinks. -/
theorem mlLoop_ge (h : Nat → Nat) (p cur : Nat) (f : Nat) : ∀ len, len ≤ mlLoop h p cur f len := by
  induction f with
  | zero => intro len; simp [mlLoop]
  | succ f ih =>
    intro len
    simp only [mlLoop]
    split
    · have := ih (len + 1)
      omega
    · omega

/-- Tokens copied by `propose`: from `best_end + 1` through `cur`, at most `max_k`. -/
def copyCount (bestEnd cur maxK : Nat) : Nat := min maxK (cur - bestEnd)

/-- The copy never exceeds `max_k`. -/
theorem copyCount_le (bestEnd cur maxK : Nat) : copyCount bestEnd cur maxK ≤ maxK := by unfold copyCount; omega

/-- The last copied index `best_end + count` is at most `cur`, a valid history index. -/
theorem copy_last_valid (bestEnd cur maxK : Nat) (h : bestEnd < cur) : bestEnd + copyCount bestEnd cur maxK ≤ cur := by
  unfold copyCount; omega

/-- A match strictly before the current suffix and `max_k ≥ 1` yields at least one draft token. -/
theorem copy_nonempty (bestEnd cur maxK : Nat) (h : bestEnd < cur) (hk : 1 ≤ maxK) : 1 ≤ copyCount bestEnd cur maxK := by
  unfold copyCount; omega

end Strata.SuffixDrafter

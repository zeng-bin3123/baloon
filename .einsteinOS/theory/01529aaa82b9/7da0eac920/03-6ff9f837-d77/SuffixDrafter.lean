/-!
# MODULE: suffix_drafter
Suffix-lookup drafter: a bounded open-addressing hash table maps the trigram ending at each history
position to its last `WAYS` positions; `propose` finds the longest earlier match of the current suffix
and copies its continuation as draft tokens.

## Source semantics
* The table capacity is the smallest power of two (by doubling) that is at least `2 * capacity_tokens`.
* A slot's position list is most-recent-first and holds at most `WAYS` entries.
* `min_match >= 3` and `max_match >= min_match`.
* A candidate position `p < cur`; the match length is bounded by `p` so history reads stay in bounds.
* The continuation read `q` runs over `best_end + 1 .. cur`, always inside the history, and yields at most `max_k` tokens.
* Keys are forced non-zero (zero marks an empty slot).

## Form template identifiers
* `grow`        <- `while (cap < capacity_tokens * 2) cap <<= 1`
* `insertPos`   <- the `pos[w] = pos[w-1]` shift and `pos[0] = end`
* `WAYS`        <- `SuffixDrafter::WAYS`
* `forceOdd`    <- `... | 1ull`
-/

namespace Strata.SuffixDrafter

-- § capacity

/-- Capacity doubling with a fuel bound. -/
def grow : Nat → Nat → Nat → Nat
  | 0, cap, _ => cap
  | f + 1, cap, t => if cap < t then grow f (cap * 2) t else cap

/-- Doubling reaches the target as soon as `cap * 2^f >= t`. -/
theorem grow_ge (t : Nat) : ∀ (f cap : Nat), t ≤ cap * 2 ^ f → t ≤ grow f cap t := by
  intro f
  induction f with
  | zero => intro cap h; simpa [grow] using h
  | succ f ih =>
    intro cap h
    simp only [grow]
    split
    · apply ih
      have : cap * 2 ^ (f + 1) = cap * 2 * 2 ^ f := by rw [Nat.pow_succ]; rw [Nat.mul_assoc, Nat.mul_comm (2 ^ f) 2, ← Nat.mul_assoc]
      omega
    · omega

/-- Starting from 1 with `t` fuel steps, the capacity is at least `t` (since `2^t >= t`). -/
theorem grow_from_one (t : Nat) : t ≤ grow t 1 t := by
  apply grow_ge
  have := Nat.lt_two_pow_self (n := t)
  omega

/-- Table index after probing stays below the capacity. -/
theorem probe_lt (i cap : Nat) (hc : 0 < cap) : (i + 1) % cap < cap := Nat.mod_lt _ hc

-- § slot lists

/-- Number of ways per slot. -/
def WAYS : Nat := 4

/-- Insert a position at the front, dropping the oldest beyond `WAYS`. -/
def insertPos (l : List Nat) (p : Nat) : List Nat := (p :: l).take WAYS

/-- The newest position is first. -/
theorem insertPos_head (l : List Nat) (p : Nat) : (insertPos l p).head? = some p := by
  simp [insertPos, WAYS]

/-- A slot never holds more than `WAYS` positions. -/
theorem insertPos_length_le (l : List Nat) (p : Nat) : (insertPos l p).length ≤ WAYS := by
  unfold insertPos
  exact List.length_take_le _ _

/-- The count field `n` follows `min(n + 1, WAYS)` when the list is below capacity. -/
theorem count_update (n : Nat) (h : n ≤ WAYS) : min (n + 1) WAYS ≤ WAYS := by omega

-- § keys and match parameters

/-- Force a key odd: arithmetic form of `x | 1`. -/
def forceOdd (x : Nat) : Nat := 2 * (x / 2) + 1

/-- A forced key is never zero (zero marks an empty slot). -/
theorem forceOdd_ne_zero (x : Nat) : forceOdd x ≠ 0 := by
  unfold forceOdd
  omega

/-- Constructor clamp: `min_match >= 3`. -/
theorem min_match_ge (m : Nat) : 3 ≤ max 3 m := by omega

/-- Constructor clamp: `max_match >= min_match`. -/
theorem max_match_ge (mn mx : Nat) : mn ≤ max mn mx := by omega

-- § propose

/-- A candidate is a strictly earlier position; reading back `len <= p` entries stays in bounds for the cursor too. -/
theorem match_read_in_bounds (p cur len : Nat) (hp : p < cur) (hl : len ≤ p) : len ≤ cur ∧ p - len ≤ p := by
  omega

/-- Continuation reads `q ∈ [best_end + 1, cur]` stay in the history of length `n` when `cur = n - 1`. -/
theorem continuation_in_bounds (n best_end q : Nat) (hn : 1 ≤ n) (hq : q ≤ n - 1) (hb : best_end < n - 1) :
    q < n := by omega

/-- The number of copied tokens is at most `max_k`, and at most the available continuation. -/
def copied (bestEnd cur maxK : Nat) : Nat := min maxK (cur - bestEnd)

/-- Copy count bound. -/
theorem copied_le (bestEnd cur maxK : Nat) : copied bestEnd cur maxK ≤ maxK := by
  unfold copied
  omega

/-- A found match has a continuation of at least one token when `best_end < cur` and `max_k >= 1`. -/
theorem copied_pos (bestEnd cur maxK : Nat) (h : bestEnd < cur) (hk : 1 ≤ maxK) : 1 ≤ copied bestEnd cur maxK := by
  unfold copied
  omega

/-- `propose` requires at least four tokens of history. -/
def canPropose (n maxK : Nat) : Prop := 4 ≤ n ∧ 0 < maxK

/-- With four tokens, the trigram key position `cur - 2` exists. -/
theorem key_positions (n : Nat) (h : 4 ≤ n) : 2 ≤ n - 1 := by omega

end Strata.SuffixDrafter

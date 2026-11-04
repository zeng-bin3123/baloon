/-!
# MODULE: suffix_drafter
SOURCES: src/spec/suffix_drafter.cpp
Lookup drafter for speculative decoding: an open-addressed hash table keyed by the last three
tokens stores the most recent `WAYS` end positions per key; `propose` finds the longest earlier
match of the current suffix and returns the tokens that followed it.

## Source semantics
* Keys are forced nonzero (`| 1`) so `0` can mean "empty slot".
* Each slot keeps its positions most-recent-first, at most `WAYS` of them.
* A candidate must end strictly before the current suffix (`p < cur`), otherwise it would match itself.
* The match length is capped by `max_match` and by the available history; a proposal needs
  `best_len >= min_match`, with `min_match >= 3`.
* The continuation never reads past `cur` and has at most `max_k` tokens.

## Form template identifiers
* `key_at` low bit            -> `setLow`
* `Slot::pos[WAYS]` insertion -> `insertPos`
* match loop                  -> `ml`
* continuation loop           -> `contLen`
-/

namespace Strata.Suffix

-- § key nonzero

/-- Setting the lowest bit: `x | 1` on naturals. -/
def setLow (x : Nat) : Nat := x / 2 * 2 + 1

/-- A key with the low bit forced on is never `0`, so `0` is a safe empty marker. -/
theorem setLow_ne_zero (x : Nat) : setLow x ≠ 0 := by
  unfold setLow
  omega

/-- The key is odd. -/
theorem setLow_odd (x : Nat) : setLow x % 2 = 1 := by
  unfold setLow
  omega

-- § probing

/-- Linear probing step `(i + 1) & mask` for a power-of-two table is `% cap`. -/
def nextProbe (i cap : Nat) : Nat := (i + 1) % cap

/-- A probe step stays inside the table. -/
theorem nextProbe_lt (i cap : Nat) (hc : 0 < cap) : nextProbe i cap < cap := by
  unfold nextProbe
  exact Nat.mod_lt _ hc

-- § slot way list

/-- Insert the newest end position at the front, keeping at most `W` entries. -/
def insertPos (W e : Nat) (l : List Nat) : List Nat := (e :: l).take W

/-- A slot never holds more than `W` positions. -/
theorem insertPos_length (W e : Nat) (l : List Nat) : (insertPos W e l).length ≤ W := by
  unfold insertPos
  simp [List.length_take]

/-- The newest position is first (most recent first). -/
theorem insertPos_head (W e : Nat) (l : List Nat) (hW : 0 < W) : (insertPos W e l).head? = some e := by
  cases W with
  | zero => exact absurd hW (by omega)
  | succ w => simp [insertPos]

/-- If positions are strictly decreasing and the new one is larger, the order is preserved. -/
theorem insertPos_sorted (W e : Nat) (l : List Nat) (hl : List.Pairwise (· > ·) l)
    (he : ∀ x ∈ l, x < e) : List.Pairwise (· > ·) (insertPos W e l) := by
  have h1 : List.Pairwise (· > ·) (e :: l) := List.pairwise_cons.mpr ⟨fun a ha => he a ha, hl⟩
  exact List.Pairwise.sublist (List.take_sublist _ _) h1

-- § suffix match length

/-- The match loop: counts `len` while `len <= p` and `h[p-len] = h[cur-len]`, at most `fuel` times. -/
def ml (h : Nat → Nat) (p cur : Nat) : Nat → Nat → Nat
  | 0, _ => 0
  | fuel + 1, len => if len ≤ p ∧ h (p - len) = h (cur - len) then 1 + ml h p cur fuel (len + 1) else 0

/-- The match length never exceeds `max_match`. -/
theorem ml_le_fuel (h : Nat → Nat) (p cur : Nat) : ∀ fuel len, ml h p cur fuel len ≤ fuel := by
  intro fuel
  induction fuel with
  | zero => intro len; simp [ml]
  | succ f ih =>
    intro len
    rw [ml]
    split_ifs with hc
    · have := ih (len + 1)
      omega
    · omega

/-- The match never reaches before the start of history: `len + matched <= p + 1`. -/
theorem ml_bound (h : Nat → Nat) (p cur : Nat) :
    ∀ fuel len, len ≤ p + 1 → ml h p cur fuel len + len ≤ p + 1 := by
  intro fuel
  induction fuel with
  | zero => intro len hl; simp [ml]; omega
  | succ f ih =>
    intro len hl
    rw [ml]
    split_ifs with hc
    · have h1 := hc.1
      have h2 := ih (len + 1) (by omega)
      omega
    · omega

/-- Matching a position against itself would give the full length: the reason `p >= cur` is skipped. -/
theorem ml_self (h : Nat → Nat) (cur : Nat) :
    ∀ fuel len, len ≤ cur + 1 → ml h cur cur fuel len = min fuel (cur + 1 - len) := by
  intro fuel
  induction fuel with
  | zero => intro len hl; simp [ml]
  | succ f ih =>
    intro len hl
    rw [ml]
    by_cases hc : len ≤ cur
    · have hc2 : len ≤ cur ∧ h (cur - len) = h (cur - len) := ⟨hc, rfl⟩
      rw [if_pos hc2, ih (len + 1) (by omega)]
      omega
    · have hc3 : ¬ (len ≤ cur ∧ h (cur - len) = h (cur - len)) := fun hh => hc hh.1
      rw [if_neg hc3]
      omega

/-- Constructor clamps: `min_match_ = max 3 min_match` is at least 3. -/
theorem minMatch_ge_three (m : Nat) : 3 ≤ max 3 m := by
  omega

/-- `max_match_ = max min_match max_match` is at least the requested minimum. -/
theorem maxMatch_ge_min (m M : Nat) : m ≤ max m M := by
  omega

/-- If the effective `max_match` is below the effective `min_match`, no proposal is ever made. -/
theorem no_proposal_if_small_max (h : Nat → Nat) (p cur maxm minm : Nat) (hlt : maxm < minm) :
    ml h p cur maxm 0 < minm := by
  have := ml_le_fuel h p cur maxm 0
  omega

-- § continuation

/-- Number of continuation tokens: from `e + 1` to `cur`, at most `max_k`. -/
def contLen (maxk cur e : Nat) : Nat := min maxk (cur - e)

/-- The continuation has at most `max_k` tokens. -/
theorem contLen_le_maxk (maxk cur e : Nat) : contLen maxk cur e ≤ maxk := by
  unfold contLen
  omega

/-- The continuation never reads past `cur`: its last index `e + len <= cur`. -/
theorem contLen_reads_in_history (maxk cur e : Nat) (he : e < cur) : e + contLen maxk cur e ≤ cur := by
  unfold contLen
  omega

end Strata.Suffix

/-!
# MODULE: suffix_drafter
Suffix-lookup drafter: hashes each trigram of the token history into an open-addressed table of recent end
positions, then proposes the continuation of the longest earlier match of the current suffix.

## Source semantics
* The table capacity is a power of two with load factor <= 0.5 at nominal capacity.
* Slot keys are `mix(..) | 1`, so a key is never 0 (0 marks an empty slot).
* Match length is capped by `max_match` and by `p + 1` (cannot read before the history start).
* The proposal reads `hist[best_end+1 .. cur]`, never past `cur`, and at most `max_k` tokens.
* Each slot keeps the most recent `WAYS` positions, newest first.

## Form template identifiers
SuffixDrafter::propose -> `matchLen`/`proposeCount`; table sizing loop -> `pow2Aux`; slot position shift ->
`insertFront`; probe step -> `probeNext`.
-/

namespace Strata.Spec.Suffix

-- § key is never zero

/-- OR-ing in bit 0 makes a key non-zero. -/
theorem key_ne_zero (x : Nat) : (x ||| 1) ≠ 0 := by
  intro h
  have h0 : (x ||| 1).testBit 0 = true := by simp [Nat.testBit_or]
  rw [h] at h0
  simp at h0

-- § table sizing

/-- The capacity-doubling loop `while (cap < target) cap <<= 1`, with explicit fuel. -/
def pow2Aux (cap target : Nat) : Nat → Nat
  | 0 => cap
  | f + 1 => if cap < target then pow2Aux (cap * 2) target f else cap

/-- With enough fuel the loop result reaches the target. -/
theorem pow2Aux_ge (target : Nat) : ∀ (fuel cap : Nat), 0 < cap → target ≤ cap * 2 ^ fuel →
    target ≤ pow2Aux cap target fuel := by
  intro fuel
  induction fuel with
  | zero =>
    intro cap hcap h
    simpa [pow2Aux] using h
  | succ f ih =>
    intro cap hcap h
    unfold pow2Aux
    split
    · rename_i hlt
      apply ih (cap * 2) (by omega)
      have e : cap * 2 ^ (f + 1) = cap * 2 * 2 ^ f := by
        rw [Nat.pow_succ]
        ac_rfl
      omega
    · rename_i hge
      omega

/-- The loop never shrinks the capacity. -/
theorem pow2Aux_le_self (target : Nat) : ∀ (fuel cap : Nat), cap ≤ pow2Aux cap target fuel := by
  intro fuel
  induction fuel with
  | zero => intro cap; simp [pow2Aux]
  | succ f ih =>
    intro cap
    unfold pow2Aux
    split
    · have := ih (cap * 2)
      omega
    · omega

-- § probing

/-- Linear probe step. -/
def probeNext (cap i : Nat) : Nat := (i + 1) % cap

/-- A probe never leaves the table. -/
theorem probeNext_lt (cap i : Nat) (h : 0 < cap) : probeNext cap i < cap := by
  unfold probeNext
  exact Nat.mod_lt _ h

-- § match length

/-- Match length: extend while below `maxM`, within the history, and equal. -/
def matchFrom (h : Nat → Nat) (p cur maxM len : Nat) : Nat → Nat
  | 0 => len
  | fuel + 1 =>
    if len < maxM ∧ len ≤ p ∧ h (p - len) = h (cur - len) then matchFrom h p cur maxM (len + 1) fuel
    else len

/-- The match is capped by `max_match`. -/
theorem matchFrom_le_max (h : Nat → Nat) (p cur maxM : Nat) :
    ∀ (fuel len : Nat), len ≤ maxM → matchFrom h p cur maxM len fuel ≤ maxM := by
  intro fuel
  induction fuel with
  | zero => intro len hl; simpa [matchFrom] using hl
  | succ f ih =>
    intro len hl
    unfold matchFrom
    split
    · rename_i hc
      have := hc.1
      exact ih (len + 1) (by omega)
    · exact hl

/-- The match never reads before the history start (`len <= p + 1`). -/
theorem matchFrom_le_hist (h : Nat → Nat) (p cur maxM : Nat) :
    ∀ (fuel len : Nat), len ≤ p + 1 → matchFrom h p cur maxM len fuel ≤ p + 1 := by
  intro fuel
  induction fuel with
  | zero => intro len hl; simpa [matchFrom] using hl
  | succ f ih =>
    intro len hl
    unfold matchFrom
    split
    · rename_i hc
      have := hc.2.1
      exact ih (len + 1) (by omega)
    · exact hl

-- § proposal

/-- Number of continuation tokens proposed from a match ending at `bestEnd`, with the suffix ending at `cur`. -/
def proposeCount (cur bestEnd maxK : Nat) : Nat := min maxK (cur - bestEnd)

/-- At most `max_k` tokens are proposed. -/
theorem proposeCount_le (cur bestEnd maxK : Nat) : proposeCount cur bestEnd maxK ≤ maxK := by
  unfold proposeCount
  omega

/-- Every token read, `hist[bestEnd + 1 + j]` for `j < count`, is at index `<= cur`. -/
theorem propose_reads_in_range (cur bestEnd maxK j : Nat) (hb : bestEnd ≤ cur)
    (hj : j < proposeCount cur bestEnd maxK) : bestEnd + 1 + j ≤ cur := by
  unfold proposeCount at hj
  omega

-- § ways

/-- Insert a position at the front of a slot's list, keeping at most `ways` entries. -/
def insertFront (pos : List Nat) (x ways : Nat) : List Nat := (x :: pos).take ways

/-- A slot never holds more than `ways` positions. -/
theorem insertFront_length (pos : List Nat) (x ways : Nat) : (insertFront pos x ways).length ≤ ways := by
  unfold insertFront
  exact List.length_take_le _ _

/-- The newest position is first. -/
theorem insertFront_head (pos : List Nat) (x w : Nat) : (insertFront pos x (w + 1)).head? = some x := by
  simp [insertFront]

end Strata.Spec.Suffix

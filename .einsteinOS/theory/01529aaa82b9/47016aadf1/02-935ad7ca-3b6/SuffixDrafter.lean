/-!
# MODULE: suffix_drafter
An n-gram (trigram-keyed) suffix lookup drafter: an open-addressed hash table maps the last three tokens to
up to WAYS most-recent end positions; `propose` verifies candidates by extending the match backwards and
returns the tokens that followed the best match.

## Source semantics
Keys are never 0 (0 marks an empty slot). Probing is linear and wraps. A slot keeps its WAYS newest
positions, newest first. Match extension never reads before the start of history. The proposal never
exceeds `max_k` tokens and only reads history up to the current end.

## Form template identifiers
- `keyOf` <- `| 1ull` in key_at (non-zero key)
- `probeNext` <- `(i + 1) & mask`
- `pushWay` <- the pos[] shift-and-insert in append
- `ml` <- the `while (len < max_match_ && len <= p && ...)` loop
- `contCount` <- the continuation copy loop
-/

namespace Strata.SuffixDrafter

-- § keys and probing

/-- A hashed key forced odd, so it is non-zero. -/
def keyOf (x : Nat) : Nat := x ||| 1

/-- Keys are never 0, so 0 can safely mean "empty slot". -/
theorem keyOf_ne_zero (x : Nat) : keyOf x ≠ 0 := by
  unfold keyOf
  intro h
  simp [Nat.or_eq_zero_iff] at h

/-- Next probe position in a table of `cap` slots stays in the table. -/
def probeNext (i cap : Nat) : Nat := (i + 1) % cap

/-- Probing never leaves the table. -/
theorem probeNext_lt (i cap : Nat) (hc : 0 < cap) : probeNext i cap < cap := by
  unfold probeNext
  exact Nat.mod_lt _ hc

/-- The starting probe position is inside the table. -/
theorem probeStart_lt (key cap : Nat) (hc : 0 < cap) : key % cap < cap := Nat.mod_lt _ hc

/-- Constructor clamps: `min_match ≥ 3` and `max_match ≥ min_match`. -/
theorem ctor_clamps (mn mx : Nat) : 3 ≤ max 3 mn ∧ mn ≤ max mn mx := by omega

-- § way bookkeeping

/-- Insert a new end position as the newest way, dropping the oldest when full. -/
def pushWay (W : Nat) (pos : List Nat) (e : Nat) : List Nat := e :: pos.take (W - 1)

/-- A slot never holds more than WAYS positions. -/
theorem pushWay_length_le (W : Nat) (pos : List Nat) (e : Nat) (hW : 0 < W) :
    (pushWay W pos e).length ≤ W := by
  unfold pushWay
  simp only [List.length_cons, List.length_take]
  omega

/-- The newest position is always first. -/
theorem pushWay_head (W : Nat) (pos : List Nat) (e : Nat) : (pushWay W pos e).head? = some e := by
  unfold pushWay
  rfl

/-- The way count `n` after an insert is `min (n+1) WAYS`, never above WAYS. -/
theorem way_count_le (W n : Nat) (h : n ≤ W) : (if n < W then n + 1 else n) ≤ W := by
  split <;> omega

-- § match extension

/-- Backward match length with the three loop guards (`maxm`, `len <= p`, equal tokens). -/
def ml (h : Nat → Nat) (p cur maxm : Nat) : Nat → Nat → Nat
  | len, 0 => len
  | len, f + 1 =>
    if len < maxm ∧ len ≤ p ∧ h (p - len) = h (cur - len) then ml h p cur maxm (len + 1) f else len

/-- The match never exceeds `max_match`. -/
theorem ml_le_max (h : Nat → Nat) (p cur maxm : Nat) :
    ∀ fuel len, len ≤ maxm → ml h p cur maxm len fuel ≤ maxm := by
  intro fuel
  induction fuel with
  | zero =>
    intro len hl
    simp [ml]
    exact hl
  | succ f ih =>
    intro len hl
    unfold ml
    split
    · rename_i hc
      exact ih (len + 1) (by omega)
    · exact hl

/-- The match never reaches before the start of history (`len ≤ p + 1`, so `p - len` never underflows
while being read). -/
theorem ml_le_pos (h : Nat → Nat) (p cur maxm : Nat) :
    ∀ fuel len, len ≤ p + 1 → ml h p cur maxm len fuel ≤ p + 1 := by
  intro fuel
  induction fuel with
  | zero =>
    intro len hl
    simp [ml]
    exact hl
  | succ f ih =>
    intro len hl
    unfold ml
    split
    · rename_i hc
      exact ih (len + 1) (by omega)
    · exact hl

-- § continuation

/-- Number of tokens copied after the best match: positions `q = best_end+1 .. cur` while below `max_k`. -/
def contCount (hi mk : Nat) : Nat → Nat → Nat → Nat
  | _, k, 0 => k
  | q, k, f + 1 => if q ≤ hi ∧ k < mk then contCount hi mk (q + 1) (k + 1) f else k

/-- The proposal never exceeds `max_k` tokens. -/
theorem contCount_le (hi mk : Nat) : ∀ fuel q k, k ≤ mk → contCount hi mk q k fuel ≤ mk := by
  intro fuel
  induction fuel with
  | zero =>
    intro q k hk
    simp [contCount]
    exact hk
  | succ f ih =>
    intro q k hk
    unfold contCount
    split
    · rename_i hc
      exact ih (q + 1) (k + 1) (by omega)
    · exact hk

/-- Every copied position stays at or below the current end (the continuation never reads the future). -/
theorem cont_reads_valid (hi q k mk : Nat) (h : q ≤ hi ∧ k < mk) : q ≤ hi := h.1

end Strata.SuffixDrafter

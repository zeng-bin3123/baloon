/-!
# MODULE: suffix_lookup_drafter
Suffix-lookup speculative drafter: a hash table from the last three tokens to the most recent positions
where they ended, and a proposal that copies the continuation of the longest earlier match.

## Source semantics
* Table capacity is the smallest power of two `≥ 2 * capacity_tokens`; probing uses `i & (cap - 1)`.
* Keys are `mix(...) | 1`, so key 0 means "empty slot".
* Each slot keeps up to `WAYS` positions, newest first: shift right, put the new end at index 0, `n = min(n+1, WAYS)`.
* `min_match = max(3, min_match)`, `max_match = max(min_match, max_match)`.
* `propose` needs `n ≥ 4`; candidates `p ≥ cur` are skipped; the match walk stops at `len ≤ p`; ties keep the
  newer candidate (strict `>`); the continuation copies `hist[best_end+1 .. cur]`, at most `max_k` tokens.

## Form template identifiers
SuffixDrafter::append -> index_tokens ; propose -> propose_continuation ; Slot.pos -> recent_positions ;
find_slot -> probe_slot ; key_at -> suffix_key ; WAYS -> ways ; mask_ -> probe_mask
-/

namespace Strata.Spec.SuffixDrafter

-- § key and table

/-- A key with the low bit forced is never the empty marker 0. -/
theorem key_ne_zero (x : UInt64) : x ||| 1 ≠ 0 := by
  bv_decide

/-- Doubling-until-large loop with fuel. -/
def grow (target : Nat) : Nat → Nat → Nat
  | 0, c => c
  | f + 1, c => if c < target then grow target f (2 * c) else c

/-- The table capacity is always a power of two (so `& mask` is a valid modulo). -/
theorem grow_pow2 (target : Nat) :
    ∀ (f c : Nat), (∃ k, c = 2 ^ k) → ∃ k, grow target f c = 2 ^ k := by
  intro f
  induction f with
  | zero => intro c h; simpa [grow] using h
  | succ f ih =>
      intro c h
      unfold grow
      by_cases hc : c < target
      · simp only [hc, if_true]
        obtain ⟨k, hk⟩ := h
        exact ih (2 * c) ⟨k + 1, by rw [hk, Nat.pow_succ]; omega⟩
      · simp only [hc, if_false]
        exact h

/-- The capacity starts at 1 = 2^0. -/
theorem grow_start : ∃ k, (1 : Nat) = 2 ^ k := ⟨0, rfl⟩

/-- Probe index masking stays inside the table. -/
theorem probe_index_lt (x k : Nat) : x &&& (2 ^ k - 1) < 2 ^ k := by
  have h : 2 ^ k - 1 < 2 ^ k := by
    have := Nat.two_pow_pos k
    omega
  exact Nat.and_lt_two_pow x h

/-- Next probe also stays inside the table. -/
theorem probe_next_lt (i k : Nat) : (i + 1) &&& (2 ^ k - 1) < 2 ^ k := probe_index_lt (i + 1) k

-- § per-slot position list

/-- Insert a new end position, newest first, keeping at most `W`. -/
def push (W new : Nat) (l : List Nat) : List Nat := (new :: l).take W

/-- The slot never holds more than `W` positions. -/
theorem push_length_le (W new : Nat) (l : List Nat) : (push W new l).length ≤ W := by
  unfold push
  simp [List.length_take]

/-- The count update `n = min(n+1, WAYS)` matches the real list length. -/
theorem push_length (W new : Nat) (l : List Nat) : (push W new l).length = min W (l.length + 1) := by
  unfold push
  simp [List.length_take]

/-- With room for at least one entry, the newest position is first. -/
theorem push_head (w new : Nat) (l : List Nat) : (push (w + 1) new l).head? = some new := by
  simp [push, List.take_succ_cons]

/-- Positions are inserted in increasing time, so each slot stays strictly newest-first. -/
theorem push_sorted (W new : Nat) (l : List Nat) (hs : l.Pairwise (· > ·)) (hlt : ∀ x ∈ l, x < new) :
    (push W new l).Pairwise (· > ·) := by
  unfold push
  have h : (new :: l).Pairwise (· > ·) := List.pairwise_cons.2 ⟨fun x hx => hlt x hx, hs⟩
  exact h.sublist (List.take_sublist _ _)

-- § parameters

/-- The effective minimum match is at least 3 (the key covers three tokens). -/
theorem min_match_ge (m : Nat) : 3 ≤ max 3 m := by omega

/-- The effective maximum match is at least the minimum. -/
theorem max_match_ge (m M : Nat) : m ≤ max m M := by omega

-- § propose

/-- The key of the current suffix needs three tokens, guaranteed by `n ≥ 4`. -/
theorem key_indices_valid (n : Nat) (h : 4 ≤ n) : 2 ≤ n - 1 := by omega

/-- A candidate before the current suffix leaves positive room for the continuation. -/
theorem candidate_before_cur (p cur : Nat) (h : ¬ p ≥ cur) : p < cur := by omega

/-- During the match walk, the earlier index `p - len` does not underflow. -/
theorem match_walk_index (p len : Nat) (h : len ≤ p) : p - len + len = p := by omega

/-- The current-suffix index `cur - len` is valid whenever `len ≤ p < cur`. -/
theorem match_walk_cur_index (p cur len : Nat) (h : len ≤ p) (hp : p < cur) : 0 < cur - len := by omega

/-- Number of tokens copied as the continuation. -/
def contCount (cur bestEnd maxK : Nat) : Nat := min maxK (cur - bestEnd)

/-- The proposal is at most `max_k` tokens. -/
theorem contCount_le (cur bestEnd maxK : Nat) : contCount cur bestEnd maxK ≤ maxK := by
  unfold contCount
  omega

/-- The last token read, `best_end + count`, is at most `cur`, an existing history index. -/
theorem contCount_in_history (cur bestEnd maxK : Nat) (h : bestEnd < cur) :
    bestEnd + contCount cur bestEnd maxK ≤ cur := by
  unfold contCount
  omega

/-- A non-empty continuation exists when a match was found before the end and `max_k > 0`. -/
theorem contCount_pos (cur bestEnd maxK : Nat) (h : bestEnd < cur) (hk : 0 < maxK) :
    0 < contCount cur bestEnd maxK := by
  unfold contCount
  omega

end Strata.Spec.SuffixDrafter

/-!
# MODULE: verify_window_schedule
SOURCES: src/core/verify.cpp
The speculative-decoding verifier: a window of up to `max_t` tokens is run through every layer as one
captured graph, optionally split into two token groups whose pre/post halves interleave, with the
host pool serving each (layer, group) ring; a separate commit graph keeps the accepted prefix.

## Source semantics
* Groups: `G = 2` iff the split is on and `T ≥ 2`; group A is `[0, ceil(T/2))`, B is `[ceil(T/2), T)`;
  with `G = 1` the only group is `[0, T)`.  Groups partition the window.
* Stream order: `pre(0,g)` for all g, then for each layer l and group g: `post(l,g)`, `pre(l+1,g)`.
  The k-th `pre` rings sequence `k+1`; `post(l,g)` waits for ring `l*G + g + 1`; the host loop's
  k-th iteration serves `(k / G, k % G)` with `want = k + 1`.
* GPU plan layout (int32 units): counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad to even |
  ptr(2cap) | ptr2(2cap) | start2(cap+1); `plan_i32 = ptr_off + 4cap + (cap+1) + 1`.
* Flag B only rises: `raise_flag` stores `max(cur, value)`.
* Commit row: `[n_keep, n_keep-1, pos0+0, …, pos0+n_keep-1, -1, …]`, `1 ≤ n_keep ≤ last_t`.
-/

namespace Ref.Verify

-- § token groups

/-- Number of token groups in a window of `T` tokens. -/
def groups (split : Bool) (n : Nat) : Nat := if split && decide (2 ≤ n) then 2 else 1

/-- `(tb, te)` of group `g`. -/
def bounds (G n g : Nat) : Nat × Nat :=
  if g = 0 then (0, if G = 2 then (n + 1) / 2 else n) else ((n + 1) / 2, n)

/-- One group covers the whole window. -/
theorem one_group (n : Nat) : bounds 1 T 0 = (0, T) := by simp only [bounds]

/-- Two groups are adjacent, cover `[0, T)`, and are both nonempty. -/
theorem two_groups (n : Nat) (hT : 2 ≤ T) :
    (bounds 2 T 0).1 = 0 ∧ (bounds 2 T 0).2 = (bounds 2 T 1).1 ∧ (bounds 2 T 1).2 = T ∧
    (bounds 2 T 0).1 < (bounds 2 T 0).2 ∧ (bounds 2 T 1).1 < (bounds 2 T 1).2 := by
  simp only [bounds]; exact (by omega)

/-- A split window is used only when there are at least two tokens. -/
theorem groups_two_iff (split : Bool) (n : Nat) : groups split T = 2 ↔ split = true ∧ 2 ≤ T := by
  simp only [groups]; cases split <;> by_cases h : 2 ≤ T <;> simp only [h]

-- § the stream schedule

/-- The two halves of a layer for one group. -/
inductive Ev where
  | pre (k p : Nat)
  | post (k p : Nat)
  deriving DecidableEq, Repr

/-- The order `record_window` enqueues for `n` layers and `G` groups. -/
def sched (G n : Nat) : List Ev :=
  (List.range G).map (Ev.pre 0) ++
  (List.range n).flatMap (fun l => (List.range G).flatMap
    (fun g => Ev.post k p :: if l + 1 < n then [Ev.pre (k + 1) p] else []))

/-- The ring number a `pre` produces / a `post` waits for. -/
def ringOf (G k p : Nat) : Nat := k * G + p + 1

/-- Ring numbers of the `pre` events in stream order. -/
def preRings (G : Nat) : List Ev → List Nat
  | [] => []
  | .pre k p :: es => ringOf G k p :: preRings G es
  | .post _ _ :: es => preRings G es

/-- Every `post` comes after its own `pre` (checked by a scan of the stream). -/
def postsAfterPres : List Ev → List (Nat × Nat) → Bool
  | [], _ => true
  | .pre k p :: es, seen => postsAfterPres es ((l, g) :: seen)
  | .post k p :: es, seen => seen.contains (l, g) && postsAfterPres es seen

/-- **The artifact's 48 layers, unsplit**: rings are published as 1, 2, …, 48 in stream order. -/
theorem rings_in_order_1 : preRings 1 (sched 1 48) = (List.range 48).map (· + 1) := by exact (by decide)

/-- **Split window**: the `pre` events still ring 1, 2, …, 96 in stream order. -/
theorem rings_in_order_2 : preRings 2 (sched 2 48) = (List.range 96).map (· + 1) := by exact (by decide)

/-- Every `post(l,g)` is enqueued after `pre(l,g)`, unsplit and split. -/
theorem posts_after_pres : postsAfterPres (sched 1 48) [] = true ∧ postsAfterPres (sched 2 48) [] = true := by exact (by decide)

/-- The host loop's iteration `k` serves `(k / G, k % G)` and waits for exactly that ring. -/
theorem host_ring (G k : Nat) : ringOf G (k / G) (k % G) = k + 1 := by
  simp only [ringOf]
  have := Nat.div_add_mod k G
  rw [Nat.mul_comm] at this
  exact (by omega)

-- § the GPU plan layout

/-- int32s before the 64-bit pointer arrays. -/
def i32Head (m : Nat) : Nat := 4 + (m + 1) + m + m

/-- `(i32 + 1) & ~1`: the pointer arrays start on an 8-byte boundary. -/
def ptrOff (m : Nat) : Nat := (i32Head m + 1) / 2 * 2

/-- int32 size of one plan slot. -/
def planI32 (m : Nat) : Nat := ptrOff m + 4 * m + (m + 1) + 1

/-- The plan's regions are ordered, the pointers are 8-byte aligned, and all fit in the slot. -/
theorem plan_layout (m : Nat) :
    i32Head m ≤ ptrOff m ∧ ptrOff m % 2 = 0 ∧
    ptrOff m + 2 * m ≤ ptrOff m + 2 * m ∧
    ptrOff m + 2 * m + 2 * cap = ptrOff m + 4 * m ∧
    ptrOff m + 4 * m + (cap + 1) < planI32 m := by
  simp only [planI32] ptrOff i32Head; exact (by omega)

/-- The device's offsets agree with the host sink's. -/
theorem plan_fields (m : Nat) :
    4 + (m + 1) + m + m = i32Head m ∧ ptrOff m + 4 * m + (cap + 1) + 1 = planI32 m := by
  simp only [i32Head] planI32; exact (by omega)

-- § flags and commit

/-- `raise_flag`: a compare-and-swap loop that only ever raises. -/
def raise (a b : Nat) : Nat := max a b

/-- A raise never lowers the flag and always reaches the requested value. -/
theorem raise_spec (a b : Nat) : a ≤ raise a b ∧ b ≤ raise a b := by simp only [raise]; exact (by omega)

/-- A late host function raising an older value cannot undo a newer one. -/
theorem raise_late (a b : Nat) (h : b ≤ a) : raise a b = a := by simp only [raise]; exact (by omega)

/-- Entry `2 + t` of the commit row. -/
def commitVal (n : Nat) (p : Int) (t : Nat) : Int := if t < n then p + t else -1

/-- How many committed cells a row of `m` entries keeps. -/
def keptCount (nk : Nat) : Nat → Nat
  | 0 => 0
  | m + 1 => keptCount nk m + if m < nk then 1 else 0

/-- Exactly `n_keep` cells are committed when `n_keep ≤ max_t`. -/
theorem keptCount_eq (nk m : Nat) (h : nk ≤ m) : keptCount nk m = nk := by
  induction m with
  | zero => simp only [keptCount]; exact (by omega)
  | succ m ih =>
    simp only [keptCount]
    by_cases hm : m < nk
    · have hk : nk = m + 1 := by exact (by omega)
      subst hk
      have aux : ∀ j, j ≤ m + 1 → keptCount (m + 1) j = j := by
        intro j hj
        induction j with
        | zero => rfl
        | succ j ihj => simp only [keptCount]; rw [ihj (by omega)]; simp only; exact (by omega)
      rw [aux m (by omega)]; simp only
    · rw [ih (by omega)]; simp only [hm]

/-- A kept cell is committed at its own position; a dropped one is `-1`. -/
theorem commitVal_spec (n : Nat) (p : Int) (t : Nat) :
    (t < n → commitVal n p t = pos0 + t) ∧ (n ≤ t → commitVal n p t = -1) := by
  simp only [commitVal]
  constructor
  · intro h; simp only [h]
  · intro h; simp only [show ¬ t < n by omega]

/-- `run` refuses a window running past the context. -/
theorem window_in_context (pos0 T maxCells t : Nat) (h : pos0 + T ≤ maxCells) (ht : t < T) :
    pos0 + t < maxCells := by exact (by omega)

end Ref.Verify


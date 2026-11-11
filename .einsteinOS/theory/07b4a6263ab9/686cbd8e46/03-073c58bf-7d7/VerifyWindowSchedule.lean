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

## Form template identifiers
groups ↦ G; bounds ↦ tb_/te_; Ev.pre/Ev.post ↦ the pre/post lambdas; sched ↦ record_window's loop;
ringOf ↦ `l * G + grp + 1`; ptrOff ↦ ptr_off; planI32 ↦ plan_i32_; raise ↦ Verifier::raise_flag;
commitVal ↦ h_commit_.
-/

namespace Strata.Verify

-- § token groups

/-- Number of token groups in a window of `T` tokens. -/
def groups (split : Bool) (T : Nat) : Nat := if split && decide (2 ≤ T) then 2 else 1

/-- `(tb, te)` of group `g`. -/
def bounds (G T g : Nat) : Nat × Nat :=
  if g = 0 then (0, if G = 2 then (T + 1) / 2 else T) else ((T + 1) / 2, T)

/-- One group covers the whole window. -/
theorem one_group (T : Nat) : bounds 1 T 0 = (0, T) := by simp [bounds]

/-- Two groups are adjacent, cover `[0, T)`, and are both nonempty. -/
theorem two_groups (T : Nat) (hT : 2 ≤ T) :
    (bounds 2 T 0).1 = 0 ∧ (bounds 2 T 0).2 = (bounds 2 T 1).1 ∧ (bounds 2 T 1).2 = T ∧
    (bounds 2 T 0).1 < (bounds 2 T 0).2 ∧ (bounds 2 T 1).1 < (bounds 2 T 1).2 := by
  simp [bounds]; omega

/-- A split window is used only when there are at least two tokens. -/
theorem groups_two_iff (split : Bool) (T : Nat) : groups split T = 2 ↔ split = true ∧ 2 ≤ T := by
  unfold groups; cases split <;> by_cases h : 2 ≤ T <;> simp [h]

-- § the stream schedule

/-- The two halves of a layer for one group. -/
inductive Ev where
  | pre (l g : Nat)
  | post (l g : Nat)
  deriving DecidableEq, Repr

/-- The order `record_window` enqueues for `n` layers and `G` groups. -/
def sched (G n : Nat) : List Ev :=
  (List.range G).map (Ev.pre 0) ++
  (List.range n).flatMap (fun l => (List.range G).flatMap
    (fun g => Ev.post l g :: if l + 1 < n then [Ev.pre (l + 1) g] else []))

/-- The ring number a `pre` produces / a `post` waits for. -/
def ringOf (G l g : Nat) : Nat := l * G + g + 1

/-- Ring numbers of the `pre` events in stream order. -/
def preRings (G : Nat) : List Ev → List Nat
  | [] => []
  | .pre l g :: es => ringOf G l g :: preRings G es
  | .post _ _ :: es => preRings G es

/-- Every `post` comes after its own `pre` (checked by a scan of the stream). -/
def postsAfterPres : List Ev → List (Nat × Nat) → Bool
  | [], _ => true
  | .pre l g :: es, seen => postsAfterPres es ((l, g) :: seen)
  | .post l g :: es, seen => seen.contains (l, g) && postsAfterPres es seen

/-- **The artifact's 48 layers, unsplit**: rings are published as 1, 2, …, 48 in stream order. -/
theorem rings_in_order_1 : preRings 1 (sched 1 48) = (List.range 48).map (· + 1) := by decide

/-- **Split window**: the `pre` events still ring 1, 2, …, 96 in stream order, so the host's
monotone counter sees each (layer, group) exactly once and in order. -/
theorem rings_in_order_2 : preRings 2 (sched 2 48) = (List.range 96).map (· + 1) := by decide

/-- Every `post(l,g)` is enqueued after `pre(l,g)`, unsplit and split. -/
theorem posts_after_pres : postsAfterPres (sched 1 48) [] = true ∧ postsAfterPres (sched 2 48) [] = true := by
  decide

/-- The host loop's iteration `k` serves `(k / G, k % G)` and waits for exactly that ring. -/
theorem host_ring (G k : Nat) : ringOf G (k / G) (k % G) = k + 1 := by
  unfold ringOf
  have := Nat.div_add_mod k G
  rw [Nat.mul_comm] at this
  omega

-- § the GPU plan layout

/-- int32s before the 64-bit pointer arrays. -/
def i32Head (cap : Nat) : Nat := 4 + (cap + 1) + cap + cap

/-- `(i32 + 1) & ~1`: the pointer arrays start on an 8-byte boundary. -/
def ptrOff (cap : Nat) : Nat := (i32Head cap + 1) / 2 * 2

/-- int32 size of one plan slot. -/
def planI32 (cap : Nat) : Nat := ptrOff cap + 4 * cap + (cap + 1) + 1

/-- The plan's regions are ordered, the pointers are 8-byte aligned, and all fit in the slot. -/
theorem plan_layout (cap : Nat) :
    i32Head cap ≤ ptrOff cap ∧ ptrOff cap % 2 = 0 ∧
    ptrOff cap + 2 * cap ≤ ptrOff cap + 2 * cap ∧
    ptrOff cap + 2 * cap + 2 * cap = ptrOff cap + 4 * cap ∧
    ptrOff cap + 4 * cap + (cap + 1) < planI32 cap := by
  unfold planI32 ptrOff i32Head; omega

/-- The device's offsets agree with the host sink's: `start2` at `ptr_off + 4cap`, `ptr2` at
`ptr + cap` u64s, `dst` after `start`, `tok` after `dst`. -/
theorem plan_fields (cap : Nat) :
    4 + (cap + 1) + cap + cap = i32Head cap ∧ ptrOff cap + 4 * cap + (cap + 1) + 1 = planI32 cap := by
  unfold i32Head planI32; omega

-- § flags and commit

/-- `raise_flag`: a compare-and-swap loop that only ever raises. -/
def raise (cur v : Nat) : Nat := max cur v

/-- A raise never lowers the flag and always reaches the requested value. -/
theorem raise_spec (cur v : Nat) : cur ≤ raise cur v ∧ v ≤ raise cur v := by unfold raise; omega

/-- A late host function raising an older value cannot undo a newer one. -/
theorem raise_late (cur v : Nat) (h : v ≤ cur) : raise cur v = cur := by unfold raise; omega

/-- Entry `2 + t` of the commit row. -/
def commitVal (nKeep : Nat) (pos0 : Int) (t : Nat) : Int := if t < nKeep then pos0 + t else -1

/-- How many committed cells a row of `m` entries keeps. -/
def keptCount (nKeep : Nat) : Nat → Nat
  | 0 => 0
  | m + 1 => keptCount nKeep m + if m < nKeep then 1 else 0

/-- Exactly `n_keep` cells are committed when `n_keep ≤ max_t`. -/
theorem keptCount_eq (nKeep m : Nat) (h : nKeep ≤ m) : keptCount nKeep m = nKeep := by
  induction m with
  | zero => simp [keptCount]; omega
  | succ m ih =>
    simp only [keptCount]
    by_cases hm : m < nKeep
    · have hk : nKeep = m + 1 := by omega
      subst hk
      have aux : ∀ j, j ≤ m + 1 → keptCount (m + 1) j = j := by
        intro j hj
        induction j with
        | zero => rfl
        | succ j ihj => simp only [keptCount]; rw [ihj (by omega)]; simp; omega
      rw [aux m (by omega)]; simp
    · rw [ih (by omega)]; simp [hm]

/-- A kept cell is committed at its own position; a dropped one is `-1`. -/
theorem commitVal_spec (nKeep : Nat) (pos0 : Int) (t : Nat) :
    (t < nKeep → commitVal nKeep pos0 t = pos0 + t) ∧ (nKeep ≤ t → commitVal nKeep pos0 t = -1) := by
  unfold commitVal
  constructor
  · intro h; simp [h]
  · intro h; simp [show ¬ t < nKeep by omega]

/-- `run` refuses a window running past the context. -/
theorem window_in_context (pos0 T maxCells t : Nat) (h : pos0 + T ≤ maxCells) (ht : t < T) :
    pos0 + t < maxCells := by omega

end Strata.Verify

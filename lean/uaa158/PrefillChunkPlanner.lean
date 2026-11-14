/-!
# MODULE: prefill_chunk_planner
SOURCES: src/prefill/prefill.cpp
Chunking, 256-aligned bump allocation, expert grouping, 8-slot staging ring, PLE window.
-/

namespace Ref.Prefill.Planner

-- § chunking

/-- Sum of chunk sizes. -/
def chunkTotal : List Nat → Nat
  | [] => 0
  | a :: as => a + chunkTotal as

/-- Chunk sizes for `m` remaining tokens with chunk `T`. -/
def chunkSizes (T : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, m => if m = 0 then [] else min T m :: chunkSizes T f (m - min T m)

/-- Chunks sum to the prompt length. -/
theorem chunkSizes_total {T : Nat} (hT : 0 < T) :
    ∀ fuel n, n ≤ fuel → chunkTotal (chunkSizes T fuel n) = n := by
  intro fuel
  induction fuel with
  | zero =>
    intro n hn
    simp only [chunkSizes, chunkTotal]
    omega
  | succ fuel ih =>
    intro n hn
    by_cases hn0 : n = 0
    · simp [chunkSizes, hn0, chunkTotal]
    · simp only [chunkSizes, hn0, ↓reduceIte, chunkTotal]
      have key : chunkTotal (chunkSizes T fuel (n - min T n)) = n - min T n :=
        ih _ (by omega)
      omega

/-- Each chunk is nonempty and at most T. -/
theorem chunkSizes_bounds {T : Nat} (hT : 0 < T) :
    ∀ fuel n, ∀ c ∈ chunkSizes T fuel n, 0 < c ∧ c ≤ T := by
  intro fuel
  induction fuel with
  | zero => intro n c hc; simp only [chunkSizes] at hc; exact absurd hc (List.not_mem_nil _)
  | succ fuel ih =>
    intro n c hc
    by_cases hn0 : n = 0
    · simp [chunkSizes, hn0] at hc
    · simp only [chunkSizes, hn0, ↓reduceIte, List.mem_cons] at hc
      rcases hc with rfl | hc
      · constructor <;> omega
      · exact ih _ c hc

-- § device allocation (each allocation is 256-aligned with 256-byte guard)

/-- 256-aligned size with 256-byte guard: commuted form `256 * ⌈(n+256)/256⌉`. -/
def allocBytes (n : Nat) : Nat := 256 * ((n + 256 + 255) / 256)

/-- allocBytes is 256-aligned. -/
theorem allocBytes_aligned (n : Nat) : allocBytes n % 256 = 0 := by
  show 256 * ((n + 256 + 255) / 256) % 256 = 0
  exact Nat.mul_mod_right 256 _

/-- allocBytes provides at least 256 guard bytes. -/
theorem allocBytes_guard (n : Nat) : n + 256 ≤ allocBytes n := by
  show n + 256 ≤ 256 * ((n + 256 + 255) / 256)
  omega

/-- allocBytes wastes less than 512 bytes. -/
theorem allocBytes_tight (n : Nat) : allocBytes n < n + 512 := by
  show 256 * ((n + 256 + 255) / 256) < n + 512
  omega

/-- Bump allocator over a capacity-bounded region. -/
def bump : Nat → Nat → List Nat → Option (List Nat)
  | _, _, [] => some []
  | cap, off, (sz :: rest) =>
    if off + sz ≤ cap
    then (bump cap (off + sz) rest).map (fun rs => off :: rs)
    else none

/-- Successful bump keeps all allocations in [used, cap]. -/
theorem bump_fits : ∀ (bs : List Nat) (cap used : Nat) (offs : List Nat),
    used ≤ cap → bump cap used bs = some offs → used + chunkTotal bs ≤ cap
  | [], _, _, _, h0, _ => by simp [chunkTotal]; exact h0
  | sz :: rest, cap, used, offs, _, h => by
    simp only [bump] at h
    split at h
    · rename_i hle
      cases hr : bump cap (used + sz) rest with
      | none => simp [hr] at h
      | some r =>
        have key := bump_fits rest cap (used + sz) r hle hr
        simp only [chunkTotal]; omega
    · cases h

/-- A region of size ≥ chunkTotal always admits the bump. -/
theorem bump_succeeds : ∀ (bs : List Nat) (cap used : Nat),
    used + chunkTotal bs ≤ cap → (bump cap used bs).isSome
  | [], _, _, _ => by simp [bump]
  | sz :: rest, cap, used, h => by
    simp only [chunkTotal] at h
    have hle : used + sz ≤ cap := by omega
    have ih := bump_succeeds rest cap (used + sz) (by omega)
    simp only [bump, hle, ↓reduceIte]
    cases hr : bump cap (used + sz) rest with
    | none => simp [hr] at ih
    | some r => simp

-- § grouping by expert

/-- Start offset of expert `k` (exclusive prefix sum). -/
def offAt (ids : List Nat) : Nat → Nat
  | 0 => 0
  | k + 1 => offAt ids k + ids.count k

/-- offAt satisfies the prefix-sum recurrence. -/
theorem offAt_succ (ids : List Nat) (k : Nat) :
    offAt ids (k + 1) = offAt ids k + ids.count k := Eq.refl _

/-- The rank of pair i among same-expert pairs is strictly less than the expert's total. -/
theorem rank_lt (ids : List Nat) (i : Nat) (hi : i < ids.length) :
    (ids.take i).count ids[i] < ids.count ids[i] := by
  have split : ids = ids.take i ++ ids[i] :: ids.drop (i + 1) := by
    rw [← List.drop_eq_getElem_cons hi]
    exact (List.take_append_drop i ids).symm
  have h := congrArg (List.count ids[i]) split
  rw [List.count_append, List.count_cons_self] at h
  omega

/-- Slot assigned to pair i: band start + its rank within the band. -/
def slotOf (ids : List Nat) (i : Nat) (hlt : i < ids.length) : Nat :=
  offAt ids ids[i] + (ids.take i).count ids[i]

/-- Every slot lies in [off[e], off[e+1]). -/
theorem slotOf_in_band (ids : List Nat) (i : Nat) (hi : i < ids.length) :
    offAt ids ids[i] ≤ slotOf ids i hi ∧ slotOf ids i hi < offAt ids (ids[i] + 1) := by
  have hr := rank_lt ids i hi
  simp only [slotOf, offAt_succ]; omega

-- § staging ring (8 slots, round-robin, lookahead 7)

/-- Staging slot of the j-th blob: round-robin over 8 slots. -/
def stageSlot (j : Nat) : Nat := j % 8

/-- A staging slot is reused only 8 blobs later. -/
theorem stage_reuse_distance {j j' : Nat} (hlt : j < j') (h : stageSlot j = stageSlot j') :
    j + 8 ≤ j' := by
  simp only [stageSlot] at h; omega

/-- Within a 7-blob lookahead window, no two blobs share a slot. -/
theorem lookahead_disjoint {j j' : Nat} (hlt : j < j') (hw : j' ≤ j + 7) :
    stageSlot j ≠ stageSlot j' := by
  intro heq; have := stage_reuse_distance hlt heq; omega

/-- The dequantization double buffer alternates. -/
theorem dq_alternates (j : Nat) : j % 2 ≠ (j + 1) % 2 := by
  intro h; omega

-- § PLE predecessor window (prev carries last two tokens)

/-- Slide the two-token window: (p0, p1) → (p1, tok). -/
def pleShift : Int × Int → Int → Int × Int :=
  fun w tok => (w.2, tok)

/-- Window shift distributes over list concatenation. -/
theorem pleShift_chunks (w : Int × Int) (xs ys : List Int) :
    ys.foldl pleShift (xs.foldl pleShift w) = (xs ++ ys).foldl pleShift w :=
  (List.foldl_append pleShift w xs ys).symm

/-- After any list ending in a, b, the window holds (a, b). -/
theorem pleShift_last_two (w : Int × Int) (xs : List Int) (a b : Int) :
    (xs ++ [a, b]).foldl pleShift w = (a, b) := by
  rw [List.foldl_append]; simp [pleShift]

end Ref.Prefill.Planner

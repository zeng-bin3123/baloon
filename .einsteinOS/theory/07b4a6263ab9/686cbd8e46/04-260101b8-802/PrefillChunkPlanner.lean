/-!
# MODULE: prefill_chunk_planner
SOURCES: src/prefill/prefill.cpp
The host-side plan of the batched prompt path: splitting the prompt into chunks, the device-buffer bump
allocator (owned, borrowed, or count-only for `bytes_needed`), grouping the routed (token, k) pairs by
expert with a counting sort, the host→device staging ring, and the PLE predecessor window across chunks.

## Source semantics
* Chunks start at `c0 = 0, T, 2T, …` and have size `min(T, n - c0)`; they cover the prompt exactly.
* Every allocation takes `((n·size + 256 + 255) & ~255)` bytes: 256-aligned with at least a 256-byte guard.
  A borrowed region refuses an allocation that would overflow; `bytes_needed` runs the same sequence in
  count-only mode, so a region of that size always satisfies `init`.
* Grouping: `cnt[e]` counts routed pairs per expert, `off` is its exclusive prefix sum, and pair `i`
  goes to slot `off[e] + (earlier pairs with the same expert)`, which lies inside expert `e`'s band.
* The staging ring has 8 slots used round-robin with a lookahead of 7; a slot is reused only 8 blobs
  later, after its previous blob was dequantized.
* `prev` carries the last two tokens from chunk to chunk and into `ss.ple_prev`.

## Form template identifiers
chunk loop → chunkSizes / chunkTotal · Alloc::take rounding → allocBytes · borrowed bump → bump ·
cnt/off/fill → offAt / slotOf · stage_next → stageSlot · prev[] → pleShift
-/

namespace Strata.Prefill.Planner

-- § chunking

/-- Sum of a list of sizes. -/
def chunkTotal : List Nat → Nat
  | [] => 0
  | x :: xs => x + chunkTotal xs

/-- The chunk sizes for `n` remaining tokens with chunk `T` (`fuel` bounds the recursion). -/
def chunkSizes (T : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | fuel + 1, n => if n = 0 then [] else min T n :: chunkSizes T fuel (n - min T n)

/-- The chunks cover the prompt exactly. -/
theorem chunkSizes_total {T : Nat} (hT : 0 < T) :
    ∀ fuel n, n ≤ fuel → chunkTotal (chunkSizes T fuel n) = n := by
  intro fuel
  induction fuel with
  | zero => intro n h; simp [chunkSizes, chunkTotal]; omega
  | succ fuel ih =>
    intro n h
    by_cases hn : n = 0
    · simp [chunkSizes, hn, chunkTotal]
    · simp only [chunkSizes, hn, if_false, chunkTotal]
      have := ih (n - min T n) (by omega)
      omega

/-- No chunk exceeds the configured size and none is empty. -/
theorem chunkSizes_bounds {T : Nat} (hT : 0 < T) :
    ∀ fuel n, ∀ c ∈ chunkSizes T fuel n, 0 < c ∧ c ≤ T := by
  intro fuel
  induction fuel with
  | zero => intro n c h; simp [chunkSizes] at h
  | succ fuel ih =>
    intro n c h
    by_cases hn : n = 0
    · simp [chunkSizes, hn] at h
    · simp only [chunkSizes, hn, if_false, List.mem_cons] at h
      rcases h with h | h
      · subst h; omega
      · exact ih _ c h

-- § device allocation

/-- `Alloc::take`'s footprint for `x` payload bytes. -/
def allocBytes (x : Nat) : Nat := (x + 256 + 255) / 256 * 256

/-- Every buffer is 256-aligned. -/
theorem allocBytes_aligned (x : Nat) : allocBytes x % 256 = 0 := by
  unfold allocBytes
  omega

/-- Every buffer carries at least 256 guard bytes past its payload. -/
theorem allocBytes_guard (x : Nat) : x + 256 ≤ allocBytes x := by
  unfold allocBytes
  omega

/-- The footprint wastes less than 512 bytes. -/
theorem allocBytes_tight (x : Nat) : allocBytes x < x + 512 := by
  unfold allocBytes
  omega

/-- Bump allocation in a borrowed region: offsets of each buffer, or refusal. -/
def bump : Nat → Nat → List Nat → Option (List Nat)
  | _, _, [] => some []
  | cap, used, b :: bs => if used + b ≤ cap then (bump cap (used + b) bs).map (used :: ·) else none

/-- A successful bump, started inside the region, keeps the whole sequence inside it. -/
theorem bump_fits : ∀ (bs : List Nat) (cap used : Nat) (offs : List Nat),
    used ≤ cap → bump cap used bs = some offs → used + chunkTotal bs ≤ cap
  | [], _, _, _, h0, _ => by simp [chunkTotal]; exact h0
  | b :: bs, cap, used, offs, _, h => by
    simp only [bump] at h
    split at h
    · rename_i hle
      cases hr : bump cap (used + b) bs with
      | none => rw [hr] at h; cases h
      | some r =>
        have := bump_fits bs cap (used + b) r hle hr
        simp only [chunkTotal]
        omega
    · cases h

/-- A region at least as large as the counted sequence always satisfies it: sizing the borrowed
region from `bytes_needed` cannot make `init` fail. -/
theorem bump_succeeds : ∀ (bs : List Nat) (cap used : Nat),
    used + chunkTotal bs ≤ cap → (bump cap used bs).isSome
  | [], _, _, _ => by simp [bump]
  | b :: bs, cap, used, h => by
    simp only [chunkTotal] at h
    have hle : used + b ≤ cap := by omega
    have ih := bump_succeeds bs cap (used + b) (by omega)
    simp only [bump, hle, if_true]
    cases hr : bump cap (used + b) bs with
    | none => rw [hr] at ih; cases ih
    | some r => simp

-- § grouping by expert

/-- `off[e]`: the first slot of expert `e` (exclusive prefix sum of the counts). -/
def offAt (ids : List Nat) : Nat → Nat
  | 0 => 0
  | e + 1 => offAt ids e + ids.count e

/-- Expert `e`'s band ends where expert `e + 1`'s begins. -/
theorem offAt_succ (ids : List Nat) (e : Nat) : offAt ids (e + 1) = offAt ids e + ids.count e := rfl

/-- The rank of pair `i` among the pairs routed to the same expert is below that expert's count. -/
theorem rank_lt (ids : List Nat) (i : Nat) (hi : i < ids.length) :
    (ids.take i).count ids[i] < ids.count ids[i] := by
  have hsplit : ids = ids.take i ++ ids[i] :: ids.drop (i + 1) := by
    rw [← List.drop_eq_getElem_cons hi]
    exact (List.take_append_drop i ids).symm
  have h := congrArg (List.count ids[i]) hsplit
  rw [List.count_append, List.count_cons_self] at h
  omega

/-- `slot_host[i] = fill[e]++`: the band start plus the rank. -/
def slotOf (ids : List Nat) (i : Nat) (hi : i < ids.length) : Nat :=
  offAt ids ids[i] + (ids.take i).count ids[i]

/-- Every pair lands inside its expert's band `[off[e], off[e+1])`, so each expert's GEMM reads
exactly its own rows of `Xs`. -/
theorem slotOf_in_band (ids : List Nat) (i : Nat) (hi : i < ids.length) :
    offAt ids ids[i] ≤ slotOf ids i hi ∧ slotOf ids i hi < offAt ids (ids[i] + 1) := by
  have h := rank_lt ids i hi
  simp only [slotOf, offAt_succ]
  omega

-- § staging ring

/-- The staging slot of the `j`-th streamed blob (8 slots, round-robin). -/
def stageSlot (j : Nat) : Nat := j % 8

/-- A staging slot is reused only 8 blobs later, beyond the lookahead of 7. -/
theorem stage_reuse_distance {j j' : Nat} (hlt : j < j') (h : stageSlot j = stageSlot j') : j + 8 ≤ j' := by
  unfold stageSlot at h
  omega

/-- Blobs staged within one lookahead window never share a slot. -/
theorem lookahead_disjoint {j j' : Nat} (hlt : j < j') (hw : j' ≤ j + 7) : stageSlot j ≠ stageSlot j' := by
  intro h
  have := stage_reuse_distance hlt h
  omega

/-- The dequantization double buffer alternates between two buffers. -/
theorem dq_alternates (j : Nat) : j % 2 ≠ (j + 1) % 2 := by omega

-- § PLE window across chunks

/-- `prev[0] = prev[1]; prev[1] = tok`. -/
def pleShift (w : Int × Int) (tok : Int) : Int × Int := (w.2, tok)

/-- Chunking does not change the window: shifting through two chunks is shifting through their
concatenation. -/
theorem pleShift_chunks (w : Int × Int) (xs ys : List Int) :
    ys.foldl pleShift (xs.foldl pleShift w) = (xs ++ ys).foldl pleShift w := by
  rw [List.foldl_append]

/-- At the end of the prompt the window holds its last two tokens, handed to the token path. -/
theorem pleShift_last_two (w : Int × Int) (xs : List Int) (a b : Int) :
    (xs ++ [a, b]).foldl pleShift w = (a, b) := by
  simp [List.foldl_append, pleShift]

end Strata.Prefill.Planner

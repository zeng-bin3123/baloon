/-!
# MODULE: prefill_chunk_pipeline
SOURCES: src/prefill/prefill.cpp
The host side of chunked prefill: the bump allocator (`Alloc::take`) and the claim that `bytes_needed`
sizes a borrowed region exactly; the chunk loop over the prompt; the PLE previous-token window; the
counting-sort grouping of (token, k) pairs by expert; and the 8-slot host→device staging ring whose
lookahead of STAGE−1 never reuses a slot still in flight.

## Source semantics
* `take(n)` reserves `((n·sizeof(T) + 256 + 255) & ~255)` bytes: a 256-aligned size with ≥ 256 bytes of
  slack.  From a region it fails iff `used + bytes > cap`; count-only mode only sums.
* `bytes_needed` replays `init`'s take sequence in count-only mode, so `init` from a region of that size
  never fails.
* Chunks: c0 = 0, T, 2T, …; size min(T, n − c0); they cover the prompt.
* `prev` after the chunk loop holds the last two tokens.
* Grouping: cnt[e] = #pairs routed to e, off = exclusive prefix sum; off[NE] = T·K when all ids < NE.
* Staging slot of the j-th streamed expert is j mod 8; with lookahead 7, two experts in flight together
  always use different slots.

## Form template identifiers
alignTake → takeBytes, runAlloc → bumpAllocate, total → countOnlyBytes, chunkSizes → promptChunks,
shiftPrev → pleWindow, offOf → expertOffset, stageSlot → stagingSlot
-/

namespace Strata.Prefill.Pipeline

-- § the bump allocator

/-- Bytes reserved for a raw request of x bytes: `(x + 511) & ~255`. -/
def alignTake (x : Nat) : Nat := (x + 511) / 256 * 256

/-- Every reservation is 256-byte aligned, so every bump pointer stays aligned. -/
theorem alignTake_aligned (x : Nat) : alignTake x % 256 = 0 := by
  unfold alignTake; omega

/-- A reservation always covers the request plus at least 256 bytes of slack. -/
theorem alignTake_covers (x : Nat) : x + 256 ≤ alignTake x ∧ alignTake x ≤ x + 511 := by
  unfold alignTake; omega

/-- `x & ~255` on the byte count is rounding down to a multiple of 256. -/
theorem mask_example : (1000 + 511) / 256 * 256 = 1280 := by decide

/-- Sequential takes from a region of capacity `cap`; `none` is the first failure. -/
def runAlloc (cap : Nat) : Nat → List Nat → Option Nat
  | used, [] => some used
  | used, x :: t => if used + alignTake x > cap then none else runAlloc cap (used + alignTake x) t

/-- Count-only mode: the sum of all reservations. -/
def total : List Nat → Nat
  | [] => 0
  | x :: t => alignTake x + total t

/-- A successful allocation used exactly the count-only total and stayed within capacity. -/
theorem runAlloc_some (cap used : Nat) (xs : List Nat) (u : Nat) (h : runAlloc cap used xs = some u) :
    u = used + total xs ∧ u ≤ cap ∨ (xs = [] ∧ u = used) := by
  induction xs generalizing used with
  | nil =>
    simp only [runAlloc, Option.some.injEq] at h
    right; exact ⟨rfl, h.symm⟩
  | cons x t ih =>
    simp only [runAlloc] at h
    split at h
    · contradiction
    · rename_i hc
      left
      rcases ih _ h with ⟨h1, h2⟩ | ⟨h1, h2⟩
      · simp only [total]; omega
      · subst h1; simp only [total]; omega

/-- `bytes_needed` is sufficient: from a region at least the counted size, `init` never fails. -/
theorem runAlloc_fits (cap used : Nat) (xs : List Nat) (h : used + total xs ≤ cap) :
    runAlloc cap used xs = some (used + total xs) := by
  induction xs generalizing used with
  | nil => simp [runAlloc, total]
  | cons x t ih =>
    simp only [runAlloc, total] at h ⊢
    have hnot : ¬ (used + alignTake x > cap) := by omega
    simp only [hnot, ite_false]
    rw [ih (used + alignTake x) (by omega)]
    congr 1; omega

/-- The exact statement of `bytes_needed`: the counted size (slack aside) is enough for `init`. -/
theorem bytes_needed_exact (xs : List Nat) (slack : Nat) :
    runAlloc (total xs + slack) 0 xs = some (total xs) := by
  have := runAlloc_fits (total xs + slack) 0 xs (by omega)
  simpa using this

-- § chunking the prompt

/-- Chunk sizes min(T, n − c0) of the loop `for (c0 = 0; c0 < n; c0 += T)`. -/
def chunkSizes (T : Nat) (hT : 0 < T) : Nat → List Nat
  | n => if _h : n = 0 then [] else min T n :: chunkSizes T hT (n - min T n)
termination_by n => n
decreasing_by omega

/-- The chunks cover the whole prompt. -/
theorem chunkSizes_sum (T : Nat) (hT : 0 < T) (n : Nat) : (chunkSizes T hT n).foldr (· + ·) 0 = n := by
  induction n using Nat.strongRecOn with
  | ind n ih =>
    rw [chunkSizes]
    split
    · simp_all
    · simp only [List.foldr_cons]
      rw [ih (n - min T n) (by omega)]
      omega

/-- Every chunk is non-empty and at most T tokens (fits the chunk buffers). -/
theorem chunkSizes_bounds (T : Nat) (hT : 0 < T) (n c : Nat) (hc : c ∈ chunkSizes T hT n) :
    0 < c ∧ c ≤ T := by
  induction n using Nat.strongRecOn with
  | ind n ih =>
    rw [chunkSizes] at hc
    split at hc
    · simp at hc
    · rcases List.mem_cons.mp hc with h' | h'
      · omega
      · exact ih (n - min T n) (by omega) h'

-- § the PLE previous-token window

/-- `prev[0] = prev[1]; prev[1] = tok`. -/
def shiftPrev (p : Int × Int) (tok : Int) : Int × Int := (p.2, tok)

/-- After any prompt ending in a, b the window holds (a, b), whatever came before. -/
theorem window_last_two (p : Int × Int) (xs : List Int) (a b : Int) :
    (xs ++ [a, b]).foldl shiftPrev p = (a, b) := by
  rw [List.foldl_append]
  rfl

-- § grouping (token, k) pairs by expert

/-- `off[e]`: number of routed ids below e (exclusive prefix of the counts). -/
def offOf (ids : List Nat) (e : Nat) : Nat := (ids.filter (fun i => decide (i < e))).length

/-- `off[e+1] = off[e] + cnt[e]`. -/
theorem off_succ (ids : List Nat) (e : Nat) : offOf ids (e + 1) = offOf ids e + ids.count e := by
  induction ids with
  | nil => simp [offOf]
  | cons a t ih =>
    simp only [offOf, List.filter_cons, List.count_cons] at ih ⊢
    by_cases h1 : a < e
    · have h2 : a < e + 1 := by omega
      have h3 : ¬ (a == e) = true := by simp; omega
      simp [h1, h2, h3, ih]; omega
    · by_cases h4 : a = e
      · subst h4
        simp [ih]
        omega
      · have h2 : ¬ a < e + 1 := by omega
        have h3 : ¬ (a == e) = true := by simp; omega
        simp [h1, h2, h3, ih]

/-- `off[0] = 0`. -/
theorem off_zero (ids : List Nat) : offOf ids 0 = 0 := by simp [offOf]

/-- With every id in range, `off[NE]` is the number of pairs: every slot of the gathered buffer is used. -/
theorem off_last (ids : List Nat) (NE : Nat) (h : ∀ i ∈ ids, i < NE) : offOf ids NE = ids.length := by
  unfold offOf
  rw [List.filter_eq_self.mpr (by intro i hi; simpa using h i hi)]

/-- Expert ranges are ordered: off is monotone in e. -/
theorem off_mono (ids : List Nat) (e : Nat) : offOf ids e ≤ offOf ids (e + 1) := by
  rw [off_succ]; omega

-- § the staging ring

def STAGE : Nat := 8

/-- Slot of the j-th streamed expert. -/
def stageSlot (j : Nat) : Nat := j % STAGE

/-- Two experts in flight together (at most STAGE−1 = lookahead apart) use different slots. -/
theorem stage_slots_distinct (a b : Nat) (hab : a < b) (hclose : b ≤ a + (STAGE - 1)) :
    stageSlot a ≠ stageSlot b := by
  simp only [stageSlot, STAGE] at *
  omega

/-- The dequantized ring alternates: consecutive experts use different FP16 buffers. -/
theorem dq_alternates (j : Nat) : j % 2 ≠ (j + 1) % 2 := by omega

end Strata.Prefill.Pipeline

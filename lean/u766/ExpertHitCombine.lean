/-!
# MODULE: expert_hit_combine
SOURCES: include/ref/core/expert_source.hpp, src/core/expert_source.cpp
The two-phase GPU hit path (`expert_hit_run`): `Launch` decides once which router indices are
hits, zeroes `hit_out` and computes the hits into it; the CPU pool zeroes the hit rows of `out`;
`Combine` adds `hit_out` into `parts` exactly once, only after a `Launch` that had work.

## Source semantics
* `hit_out` is ZEROED each launch, so a row that was a hit last layer and a miss now contributes 0.
* The CPU writes zero into hit rows (the buffer is reused, so skipping is not enough).
* `Combine` without a pending `Launch` adds nothing; `hit_pending` is cleared, so a second
  `Combine` cannot add the same buffer twice.
* Net effect per row: the final part is the GPU value for a hit and the CPU value for a miss.
* Each pending `Combine` increments exactly one of `hit_ready` / `hit_late`.

## Form template identifiers
`HitPhase::Launch` → `launch`; `HitPhase::Combine` → `combineHits`; `hit_pending` → `pending`;
`hit_out` → `hitOut`; `parts_out` → `parts`; `is_hit` → `hit`; `hit_ready/hit_late` → `ready/late`.
-/

namespace Ref.ExpertHitCombine

-- § State

/-- The per-session hit-path state, rows indexed by router index. -/
structure HState where
  parts : Nat → Int
  hitOut : Nat → Int
  pending : Bool
  ready : Nat
  late : Nat

/-- `Launch`: zero `hit_out`, write the hits' GPU values, and arm `Combine` iff there were hits. -/
def launch (s : HState) (hit : Nat → Bool) (gpu : Nat → Int) (nHits : Nat) : HState :=
  { s with hitOut := fun i => if hit i then gpu i else 0, pending := decide (0 < nHits) }

/-- The CPU pool's output: zero for a hit row, its own value for a miss row. -/
def cpuOut (hit : Nat → Bool) (cpu : Nat → Int) : Nat → Int :=
  fun i => if hit i then 0 else cpu i

/-- `Combine`: `parts += hit_out` once, and record whether the GPU finished in time. -/
def combineHits (s : HState) (doneInTime : Bool) : HState :=
  if s.pending then
    { s with parts := fun i => s.parts i + s.hitOut i, pending := false,
             ready := if doneInTime then s.ready + 1 else s.ready,
             late := if doneInTime then s.late else s.late + 1 }
  else s

-- § Protocol theorems

/-- A `Combine` with no `Launch` in front of it adds nothing. -/
theorem combine_without_launch {s : HState} (h : s.pending = false) (b : Bool) :
    combineHits s b = s := by
  simp [combineHits, h]

/-- A second `Combine` is a no-op: the hit buffer is never added twice. -/
theorem combine_twice (s : HState) (b1 b2 : Bool) :
    combineHits (combineHits s b1) b2 = combineHits s b1 := by
  by_cases h : s.pending = true
  · have hp : (combineHits s b1).pending = false := by simp [combineHits, h]
    exact combine_without_launch hp b2
  · have hf : s.pending = false := by simpa using h
    rw [combine_without_launch hf b1, combine_without_launch hf b2]

/-- A launch with no hits schedules no combine. -/
theorem launch_no_hits (s : HState) (hit : Nat → Bool) (gpu : Nat → Int) :
    (launch s hit gpu 0).pending = false := by
  simp [launch]

/-- After a launch, a miss row of `hit_out` is zero - a stale hit cannot leak in. -/
theorem stale_row_zero (s : HState) (hit : Nat → Bool) (gpu : Nat → Int) (n : Nat) {i : Nat}
    (hi : hit i = false) : (launch s hit gpu n).hitOut i = 0 := by
  simp [launch, hi]

/-- THE SPLIT IS CORRECT: with the CPU's rows in `parts`, every row ends as the GPU value for a hit
and the CPU value for a miss - each expert counted exactly once. -/
theorem final_row (s : HState) (hit : Nat → Bool) (gpu cpu : Nat → Int) {n : Nat} (hn : 0 < n)
    (b : Bool) (i : Nat) :
    (combineHits (launch { s with parts := cpuOut hit cpu } hit gpu n) b).parts i =
      if hit i then gpu i else cpu i := by
  have hp : decide (0 < n) = true := by simp [hn]
  simp only [combineHits, launch, hp, if_true, cpuOut]
  cases hit i <;> simp

/-- Each pending combine is classified exactly once as ready or late. -/
theorem ready_late_count {s : HState} (h : s.pending = true) (b : Bool) :
    (combineHits s b).ready + (combineHits s b).late = s.ready + s.late + 1 := by
  cases b <;> simp [combineHits, h] <;> omega

-- § Hit-path readiness

/-- The ten buffers `hits_ready` requires, as presence flags. -/
def hitsReady (flags : List Bool) : Bool := flags.all id

/-- All or none: one missing buffer disables the whole hit path. -/
theorem hitsReady_missing {flags : List Bool} (h : false ∈ flags) : hitsReady flags = false := by
  simp only [hitsReady, List.all_eq_false]
  exact ⟨false, h, by simp⟩

end Ref.ExpertHitCombine

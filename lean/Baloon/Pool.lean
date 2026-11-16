-- Baloon.Pool: worker-pool row-range splitting, task decomposition and multi-token
-- piece-loop termination invariants.
-- Independent implementation proving the same mathematical facts as Ref.Pool.

namespace Baloon.Pool

/-- Row start of task part out of parts over total rows. -/
def rowStart (total parts part : Nat) : Nat := total * part / parts

theorem rowStart_zero (total parts : Nat) : rowStart total parts 0 = 0 := by
  simp [rowStart]

theorem rowStart_last (total parts : Nat) (hp : 0 < parts) : rowStart total parts parts = total := by
  unfold rowStart; exact Nat.mul_div_cancel total hp

theorem rowStart_mono (total parts part : Nat) : rowStart total parts part ≤ rowStart total parts (part + 1) := by
  unfold rowStart; exact Nat.div_le_div_right (Nat.mul_le_mul_left total (Nat.le_succ part))

theorem rowStart_le_total (total parts part : Nat) (hp : 0 < parts) (hpart : part ≤ parts) :
    rowStart total parts part ≤ total := by
  have h2 : rowStart total parts part ≤ rowStart total parts parts := by
    unfold rowStart; exact Nat.div_le_div_right (Nat.mul_le_mul_left total hpart)
  rw [rowStart_last total parts hp] at h2; exact h2

theorem task_expert_lt (i parts n : Nat) (h : i < parts * n) : i / parts < n :=
  Nat.div_lt_of_lt_mul h

theorem task_part_lt (i parts : Nat) (hp : 0 < parts) : i % parts < parts :=
  Nat.mod_lt _ hp

theorem task_decode (i parts : Nat) : parts * (i / parts) + i % parts = i :=
  Nat.div_add_mod i parts

/-- End of the current piece: stay inside the expert (per rows) and inside the task range. -/
def pieceEnd (per r0 g1 r : Nat) : Nat := min per (r0 + (g1 - r))

theorem global_row_decode (r per : Nat) : per * (r / per) + r % per = r :=
  Nat.div_add_mod r per

theorem local_start_lt (r per : Nat) (hper : 0 < per) : r % per < per :=
  Nat.mod_lt _ hper

theorem piece_progress (per r0 g1 r : Nat) (h0 : r0 < per) (hr : r < g1) :
    r0 < pieceEnd per r0 g1 r := by
  unfold pieceEnd; omega

theorem piece_in_expert (per r0 g1 r : Nat) : pieceEnd per r0 g1 r ≤ per := by
  unfold pieceEnd; omega

theorem piece_no_overshoot (per r0 g1 r : Nat) (hr : r ≤ g1) :
    pieceEnd per r0 g1 r - r0 ≤ g1 - r := by
  unfold pieceEnd; omega

-- ── worker pool sizing and drain ─────────────────────────────────────────────

/-- Worker count: requested when non-zero, else min(1, available_cores). -/
def poolWorkers (requested cores : Nat) : Nat := if 0 < requested then requested else max 1 cores

theorem poolWorkers_pos (requested cores : Nat) : 0 < poolWorkers requested cores := by
  unfold poolWorkers; split_ifs <;> omega

theorem poolWorkers_request (requested cores : Nat) (h : 0 < requested) :
    poolWorkers requested cores = requested := by
  unfold poolWorkers; rw [if_pos h]; omega

/-- Core list after the optional reservation of core 0 for the host. -/
def reserveHost (skipFirst : Bool) (cores : List Nat) : List Nat :=
  if skipFirst then cores.tail else cores

theorem reserveHost_length (cores : List Nat) (h : cores ≠ []) :
    (reserveHost true cores).length + 1 = cores.length := by
  cases cores with
  | nil => exact absurd rfl h
  | cons a t => simp [reserveHost]

theorem reserveHost_false (cores : List Nat) : reserveHost false cores = cores := by
  simp [reserveHost]

/-- With host_works the drain uses workers plus the host thread. -/
def drainThreads (nw : Nat) (hostWorks : Bool) : Nat := nw + if hostWorks then 1 else 0

theorem multi_tasks_pos (nw : Nat) (hostWorks : Bool) (h : 1 ≤ nw) :
    0 < 3 * drainThreads nw hostWorks := by
  unfold drainThreads; split <;> omega

/-- Split phases are used only up to capacity. -/
def useSplit (n kMax : Nat) (oracle : Bool) : Bool := decide (n ≤ kMax) && !oracle

theorem oracle_never_split (n kMax : Nat) : useSplit n kMax true = false := by simp [useSplit]

end Baloon.Pool

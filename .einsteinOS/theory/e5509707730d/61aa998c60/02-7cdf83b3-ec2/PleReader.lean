/-!
# MODULE: ple_reader
SOURCES: src/ngram/ple_reader.cpp
Asynchronous direct-I/O reader for the 90-byte PLE table rows: set-associative row cache, page-aligned read jobs
(one or two pages), page de-duplication, free-slot accounting, ticket counting and latency-ring bookkeeping.

## Source semantics
* Cache: `sets = rows / 8`, 8 ways per set, round-robin replacement; slot and data indices stay inside the arrays.
* A row at byte `at` is fetched with a page-aligned read of 1 or 2 pages (PAGE = 4096) containing all 90 bytes.
* Jobs for rows on the same page are merged; the merged length is the max and still covers every row.
* `free_slots + in_flight = max_inflight` at all times; a ticket completes when its pending count reaches 0.

## Form template identifiers
ROW_BYTES -> 90; PAGE -> 4096; WAYS -> 8; EMPTY -> 0xFFFFFFFF; `Job::length` -> jobLength; `pump/finish` -> pump/finish.
-/

namespace Strata.PleReader

-- § row cache

/-- Number of sets for a cache of `rows` rows. -/
def sets (rows : Nat) : Nat := rows / 8

/-- Capacity never exceeds the requested rows. -/
theorem capacity_le (rows : Nat) : sets rows * 8 ≤ rows := by unfold sets; omega

/-- A set index is in range when there is at least one set. -/
theorem set_index_lt (h s : Nat) (hs : 0 < s) : h % s < s := Nat.mod_lt _ hs

/-- Flat way index `s*8 + w` stays inside `sets * 8`. -/
theorem way_index_lt (s w n : Nat) (hs : s < n) (hw : w < 8) : s * 8 + w < n * 8 := by omega

/-- Data byte offset of a cache slot stays inside the data array (`ROW_BYTES = 90`). -/
theorem data_index_lt (s w n : Nat) (hs : s < n) (hw : w < 8) : (s * 8 + w) * 90 + 90 ≤ n * 8 * 90 := by omega

/-- Round-robin pointer update. -/
def nextWay (w : Nat) : Nat := (w + 1) % 8

/-- The pointer is always a valid way. -/
theorem nextWay_lt (w : Nat) : nextWay w < 8 := by unfold nextWay; omega

/-- Eight inserts return the pointer to its start, so each way is replaced once per cycle. -/
theorem nextWay_cycle (w : Nat) : (w + 8) % 8 = w % 8 := by omega

/-- A key table: way -> optional row. -/
abbrev Keys := Nat → Option Nat

/-- Store `row` at way `w`. -/
def insertAt (keys : Keys) (w row : Nat) : Keys := fun i => if i = w then some row else keys i

/-- After an insert the row is found at that way. -/
theorem insertAt_found (keys : Keys) (w row : Nat) : insertAt keys w row w = some row := by
  simp [insertAt]

/-- Other ways are untouched. -/
theorem insertAt_other (keys : Keys) (w row i : Nat) (h : i ≠ w) : insertAt keys w row i = keys i := by
  simp [insertAt, h]

/-- Real row ids (table of at most 320,001,536 rows) never collide with the EMPTY marker. -/
theorem row_ne_empty (n_rows row : Nat) (hn : n_rows ≤ 320001536) (hr : row < n_rows) : row ≠ 4294967295 := by
  omega

-- § page-aligned jobs

/-- First page-aligned offset of a row starting at byte `at`. -/
def jobFirst (at_ : Nat) : Nat := at_ / 4096 * 4096

/-- Read length: from the first page up to the page holding the row's last byte, plus one page. -/
def jobLength (at_ : Nat) : Nat := (at_ + 89) / 4096 * 4096 - jobFirst at_ + 4096

/-- The read start is page aligned and not after the row. -/
theorem jobFirst_props (at_ : Nat) : jobFirst at_ % 4096 = 0 ∧ jobFirst at_ ≤ at_ := by
  unfold jobFirst
  constructor <;> omega

/-- A row needs one page or two (it may straddle a boundary). -/
theorem jobLength_cases (at_ : Nat) : jobLength at_ = 4096 ∨ jobLength at_ = 8192 := by
  unfold jobLength jobFirst
  omega

/-- The read covers all 90 bytes of the row: `in_page + 90 <= length`. -/
theorem job_covers_row (at_ : Nat) : (at_ - jobFirst at_) + 90 ≤ jobLength at_ := by
  unfold jobLength jobFirst
  omega

/-- The in-page offset is below one page. -/
theorem in_page_lt (at_ : Nat) : at_ - jobFirst at_ < 4096 := by
  unfold jobFirst
  omega

/-- Read buffers are two pages per slot, which holds the longest job. -/
theorem slot_holds_job (at_ : Nat) : jobLength at_ ≤ 2 * 4096 := by
  rcases jobLength_cases at_ with h | h <;> omega

/-- Slot `s < max` addresses `[s*8192, s*8192 + 8192)` inside the `max * 8192` slab. -/
theorem slot_buf_in_slab (s mx : Nat) (h : s < mx) : s * 8192 + 8192 ≤ mx * 8192 := by omega

/-- De-duplication keeps the maximum length, which still covers every merged row. -/
theorem merged_covers (inPage len len' : Nat) (h : inPage + 90 ≤ len) : inPage + 90 ≤ max len len' := by omega

/-- A row inside the table lies inside its byte range (`rows * 90`). -/
theorem row_in_table (row rows : Nat) (h : row < rows) : row * 90 + 90 ≤ rows * 90 := by omega

-- § slots, queue and tickets

/-- State of the free-slot accounting: (free, in flight). -/
structure Slots where
  free : Nat
  inflight : Nat

/-- One `pump` step: submit a queued job if both a job and a free slot exist. -/
def pump (s : Slots) (queued : Nat) : Slots × Nat :=
  if 0 < queued ∧ 0 < s.free then (⟨s.free - 1, s.inflight + 1⟩, queued - 1) else (s, queued)

/-- One completion: the slot returns to the free list. -/
def finish (s : Slots) : Slots := ⟨s.free + 1, s.inflight - 1⟩

/-- `pump` conserves the slot total. -/
theorem pump_conserves (s : Slots) (q mx : Nat) (h : s.free + s.inflight = mx) :
    (pump s q).1.free + (pump s q).1.inflight = mx := by
  unfold pump
  split
  · rename_i hc
    obtain ⟨_, h2⟩ := hc
    simp only
    omega
  · exact h

/-- `finish` conserves the slot total when something was in flight. -/
theorem finish_conserves (s : Slots) (mx : Nat) (h : s.free + s.inflight = mx) (hi : 0 < s.inflight) :
    (finish s).free + (finish s).inflight = mx := by
  unfold finish
  simp only
  omega

/-- A completion decrements the pending count, saturating at 0. -/
def tick (pending : Nat) : Nat := if 0 < pending then pending - 1 else 0

/-- After as many completions as jobs, the ticket is done. -/
theorem tick_iter_done (jobs : Nat) : ∀ m, jobs ≤ m → Nat.iterate tick m jobs = 0 := by
  intro m
  induction m generalizing jobs with
  | zero =>
    intro h
    have : jobs = 0 := by omega
    subst this
    rfl
  | succ m ih =>
    intro h
    show Nat.iterate tick m (tick jobs) = 0
    by_cases hj : 0 < jobs
    · have : tick jobs = jobs - 1 := by simp [tick, hj]
      rw [this]
      exact ih (jobs - 1) (by omega)
    · have : jobs = 0 := by omega
      subst this
      have h0 : tick 0 = 0 := by simp [tick]
      rw [h0]
      exact ih 0 (by omega)

-- § latency statistics

/-- Percentile index `min(n-1, round(q (n-1)))` is a valid sample index when `n > 0`. -/
theorem percentile_index_lt (n x : Nat) (h : 0 < n) : min (n - 1) x < n := by omega

/-- The latency ring position is always inside the 65536-entry ring. -/
theorem ring_pos_lt (pos : Nat) : pos % 65536 < 65536 := by omega

/-- Fault injection: a completion is held iff it is younger than the delay; the two cases are complementary. -/
theorem delay_split (age delay : Nat) : age < delay ∨ delay ≤ age := by omega

end Strata.PleReader

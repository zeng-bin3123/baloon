/-!
# MODULE: ple_reader_paging
SOURCES: src/ngram/ple_reader.cpp
The asynchronous 4 KiB-page reader for the PLE row table: row-to-page job geometry with dedup, a set-associative row
cache, a bounded in-flight slot pool, and a ticket counter that never yields 0.

## Source semantics
* A row of 90 bytes at file offset `at` is read with one 4 KiB-aligned job of length 4096, or 8192 when the row straddles
  a page boundary; the row's bytes always lie inside the job.
* The row cache has `rows / 8` sets of 8 ways; a set index is always below the set count and a way index below 8.
* The sentinel EMPTY = 0xFFFFFFFF can never equal a real row index (the table has 320,001,536 rows).
* free slots + in-flight slots = max_inflight at all times (pump and finish preserve it).
* Ticket ids wrap past 2^32 but skip 0.
* Single-token `issue` is refused while another is pending, and `collect` without `issue` fails.

## Form template identifiers
PAGE -> PAGE ; ROW_BYTES -> ROW ; Job::length -> jobLength ; Use::in_page -> inPage ; RowCache::sets -> sets ;
EMPTY -> EMPTY ; free_slots / inflight -> (free, inflight) ; next_ticket -> bumpTicket
-/

namespace Strata.PleReader

/-- The read granule (4 KiB, the direct-I/O alignment). -/
def PAGE : Nat := 4096
/-- Bytes per PLE row: 5 blocks of 18 bytes. -/
def ROW : Nat := 90

-- § job geometry

/-- Absolute offset of a row. -/
def rowAt (tableOff row : Nat) : Nat := tableOff + row * 90
/-- First page start of the job. -/
def firstPage (at_ : Nat) : Nat := at_ / 4096 * 4096
/-- Job length: from the first page through the page holding the row's last byte. -/
def jobLength (at_ : Nat) : Nat := (at_ + 89) / 4096 * 4096 - firstPage at_ + 4096
/-- Byte offset of the row inside the job buffer. -/
def inPage (at_ : Nat) : Nat := at_ - firstPage at_

/-- The job starts on a page boundary. -/
theorem firstPage_aligned (at_ : Nat) : firstPage at_ % 4096 = 0 := by
  unfold firstPage; omega

/-- The job starts at or before the row. -/
theorem firstPage_le (at_ : Nat) : firstPage at_ ≤ at_ := by
  unfold firstPage; omega

/-- The job length is a whole number of pages. -/
theorem jobLength_aligned (at_ : Nat) : jobLength at_ % 4096 = 0 := by
  unfold jobLength firstPage; omega

/-- A job is one or two pages. -/
theorem jobLength_cases (at_ : Nat) : jobLength at_ = 4096 ∨ jobLength at_ = 8192 := by
  unfold jobLength firstPage; omega

/-- The row offset inside the buffer is within the first page. -/
theorem inPage_lt (at_ : Nat) : inPage at_ < 4096 := by
  unfold inPage firstPage; omega

/-- THE ROW'S BYTES LIE INSIDE THE JOB: `in_page + 90 <= length`, so the copy cannot overrun the read buffer. -/
theorem row_inside_job (at_ : Nat) : inPage at_ + 90 ≤ jobLength at_ := by
  unfold inPage jobLength firstPage; omega

/-- A row that does not straddle a page needs one page. -/
theorem single_page_when_fits (at_ : Nat) (h : inPage at_ + 90 ≤ 4096) : jobLength at_ = 4096 := by
  unfold inPage jobLength firstPage at *; omega

/-- A straddling row needs two pages. -/
theorem two_pages_when_straddling (at_ : Nat) (h : 4096 < inPage at_ + 90) : jobLength at_ = 8192 := by
  unfold inPage jobLength firstPage at *; omega

/-- Two rows in the same page share a job: equal first-page keys. -/
theorem same_page_same_job (a b : Nat) (h : a / 4096 = b / 4096) : firstPage a = firstPage b := by
  unfold firstPage; rw [h]

-- § row cache

/-- Number of sets for `rows` cache rows (8 ways). -/
def sets (rows : Nat) : Nat := rows / 8

/-- Capacity never exceeds the requested rows. -/
theorem capacity_le (rows : Nat) : sets rows * 8 ≤ rows := by
  unfold sets; omega

/-- A set index is below the set count (non-empty cache). -/
theorem set_index_lt (mixed rows : Nat) (h : 0 < sets rows) : mixed % sets rows < sets rows :=
  Nat.mod_lt _ h

/-- Every `(set, way)` slot lies inside the key array. -/
theorem slot_in_array (s w nSets : Nat) (hs : s < nSets) (hw : w < 8) : s * 8 + w < nSets * 8 := by
  omega

/-- The round-robin way pointer stays below 8. -/
theorem next_way_lt (w : Nat) : (w + 1) % 8 < 8 := by omega

/-- The empty-key sentinel. -/
def EMPTY : Nat := 4294967295

/-- No real row index equals the sentinel (the table has fewer than 2^32 - 1 rows). -/
theorem row_ne_empty (row : Nat) (h : row < 320001536) : row ≠ EMPTY := by
  unfold EMPTY; omega

-- § in-flight slots

/-- Slot accounting: `(free, inflight)`. -/
structure Slots where
  free : Nat
  inflight : Nat

/-- Submitting a job moves one slot from free to in-flight. -/
def Slots.pump (s : Slots) : Slots :=
  if 0 < s.free then { free := s.free - 1, inflight := s.inflight + 1 } else s

/-- Finishing a job moves one slot back. -/
def Slots.finish (s : Slots) : Slots :=
  if 0 < s.inflight then { free := s.free + 1, inflight := s.inflight - 1 } else s

/-- `pump` preserves free + inflight. -/
theorem pump_conserves (s : Slots) : s.pump.free + s.pump.inflight = s.free + s.inflight := by
  unfold Slots.pump
  split <;> simp <;> omega

/-- `finish` preserves free + inflight. -/
theorem finish_conserves (s : Slots) : s.finish.free + s.finish.inflight = s.free + s.inflight := by
  unfold Slots.finish
  split <;> simp <;> omega

/-- A job is only submitted when a slot is free. -/
theorem pump_requires_free (s : Slots) (h : s.free = 0) : s.pump = s := by
  unfold Slots.pump
  rw [if_neg (by omega)]

-- § tickets

/-- Next ticket id: increment mod 2^32, skipping 0. -/
def bumpTicket (t : Nat) : Nat := if (t + 1) % 4294967296 = 0 then 1 else (t + 1) % 4294967296

/-- A ticket id is never 0. -/
theorem bumpTicket_ne_zero (t : Nat) : bumpTicket t ≠ 0 := by
  unfold bumpTicket
  split <;> omega

/-- A ticket id fits 32 bits. -/
theorem bumpTicket_lt (t : Nat) : bumpTicket t < 4294967296 := by
  unfold bumpTicket
  split <;> omega

-- § single-token issue/collect protocol (PleTable, Direct mode)

/-- `issue`: refused when a token is already pending. Returns `(accepted, pending')`. -/
def issueB (pending : Bool) : Bool × Bool := if pending then (false, pending) else (true, true)

/-- `collect`: fails without a pending issue. Returns `(ok, pending')`. -/
def collectB (pending : Bool) : Bool × Bool := if pending then (true, false) else (false, false)

/-- Two issues in a row: the second is refused. -/
theorem double_issue_refused : (issueB (issueB false).2).1 = false := by decide

/-- `collect` without `issue` fails. -/
theorem collect_without_issue : (collectB false).1 = false := by decide

/-- `issue` then `collect` succeeds and leaves nothing pending. -/
theorem issue_collect_ok : (collectB (issueB false).2) = (true, false) := by decide

/-- The reader refuses a table that extends past the file. -/
def tableFits (tableOff nRows fileSize : Nat) : Bool := decide (tableOff + nRows * 90 ≤ fileSize)

/-- The artifact's PLE table is 28,800,138,240 bytes. -/
theorem ple_table_bytes : 320001536 * 90 = 28800138240 := by decide

end Strata.PleReader

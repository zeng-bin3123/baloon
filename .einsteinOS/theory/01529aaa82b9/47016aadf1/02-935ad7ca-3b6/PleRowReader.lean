/-!
# MODULE: ple_row_reader
Asynchronous direct-I/O reader for the PLE table: row-to-page job geometry, a set-associative row cache,
in-flight slot accounting, latency ring and ticket ids.

## Source semantics
A 90-byte row lives in one 4 KiB page or straddles two, so a job reads 1 or 2 pages and the row lies
inside the read. Slots move between free and in-flight without loss. The ticket id is never 0.

## Form template identifiers
- `PAGE`, `ROW` <- PAGE, ROW_BYTES (4096 assumed, 90)
- `jobFirst`, `jobLength`, `inPage` <- `first`, `length`, `in_page` in PleReader::issue
- `Slots`, `pumpStep`, `finishStep` <- free_slots / inflight / queue in Impl
- `nextTicket` <- the `next_ticket` increment that skips 0
-/

namespace Strata.PleRowReader

def PAGE : Nat := 4096
def ROW : Nat := 90

-- § page jobs

/-- Aligned file offset of the first page touched by a row at byte offset `at`. -/
def jobFirst (a : Nat) : Nat := a / 4096 * 4096

/-- Read length: one page, or two when the row straddles a page boundary. -/
def jobLength (a : Nat) : Nat := (a + 89) / 4096 * 4096 - a / 4096 * 4096 + 4096

/-- Offset of the row inside the read buffer. -/
def inPage (a : Nat) : Nat := a - a / 4096 * 4096

/-- A job reads exactly one or two pages. -/
theorem jobLength_cases (a : Nat) : jobLength a = 4096 ∨ jobLength a = 8192 := by
  unfold jobLength
  omega

/-- The whole row lies inside the bytes the job reads. -/
theorem row_inside_job (a : Nat) : inPage a + 90 ≤ jobLength a := by
  unfold inPage jobLength
  omega

/-- The row starts inside the first page. -/
theorem inPage_lt (a : Nat) : inPage a < 4096 := by
  unfold inPage
  omega

/-- The job offset is page aligned and not after the row. -/
theorem jobFirst_aligned (a : Nat) : jobFirst a % 4096 = 0 ∧ jobFirst a ≤ a := by
  unfold jobFirst
  omega

/-- The row offset is the job offset plus the in-page offset. -/
theorem job_offset_decompose (a : Nat) : jobFirst a + inPage a = a := by
  unfold jobFirst inPage
  omega

/-- A row index below `n_rows` yields a byte range inside the table. -/
theorem row_in_table (row nRows : Nat) (h : row < nRows) : row * 90 + 90 ≤ nRows * 90 := by omega

-- § cache

/-- The set index of a row stays inside the cache. -/
theorem set_index_lt (h sets : Nat) (hs : 0 < sets) : h % sets < sets := Nat.mod_lt _ hs

/-- Round-robin replacement pointer stays inside the 8 ways. -/
theorem way_next_lt (w : Nat) : (w + 1) % 8 < 8 := by omega

/-- Cache capacity `sets * WAYS` never exceeds the requested row count. -/
theorem cache_capacity_le (rows : Nat) : rows / 8 * 8 ≤ rows := by omega

-- § slot accounting

structure Slots where
  free : Nat
  inflight : Nat
  queue : Nat

/-- Submit one queued job into a free slot. -/
def pumpStep (s : Slots) : Slots :=
  if 0 < s.queue ∧ 0 < s.free then
    { free := s.free - 1, inflight := s.inflight + 1, queue := s.queue - 1 }
  else s

/-- Complete one in-flight job and release its slot. -/
def finishStep (s : Slots) : Slots :=
  if 0 < s.inflight then { free := s.free + 1, inflight := s.inflight - 1, queue := s.queue } else s

/-- Submitting never loses or duplicates a slot. -/
theorem pump_conserves (s : Slots) (M : Nat) (h : s.free + s.inflight = M) :
    (pumpStep s).free + (pumpStep s).inflight = M := by
  unfold pumpStep
  repeat' split
  all_goals first | omega | (dsimp only; omega)

/-- Completing never loses or duplicates a slot. -/
theorem finish_conserves (s : Slots) (M : Nat) (h : s.free + s.inflight = M) :
    (finishStep s).free + (finishStep s).inflight = M := by
  unfold finishStep
  repeat' split
  all_goals first | omega | (dsimp only; omega)

/-- Submitting never grows the queue. -/
theorem pump_queue_le (s : Slots) : (pumpStep s).queue ≤ s.queue := by
  unfold pumpStep
  repeat' split
  all_goals first | omega | (dsimp only; omega)

-- § ring buffers and tickets

/-- The latency ring index stays inside the 65536-entry ring. -/
theorem ring_index_lt (pos : Nat) : pos % 65536 < 65536 := by omega

/-- The percentile index `min (size-1) k` is a valid index when samples exist. -/
theorem percentile_index_lt (size k : Nat) (h : 0 < size) : min (size - 1) k < size := by omega

/-- The successor ticket id (32-bit wrap, skipping 0). -/
def nextTicket (t : Nat) : Nat :=
  if (t + 1) % 4294967296 = 0 then 1 else (t + 1) % 4294967296

/-- Ticket ids are never 0, even across wraparound. -/
theorem nextTicket_ne_zero (t : Nat) : nextTicket t ≠ 0 := by
  unfold nextTicket
  split
  · omega
  · assumption

end Strata.PleRowReader

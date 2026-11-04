/-!
# MODULE: ple_reader
Direct-I/O reader for the PLE n-gram table: rows of 90 bytes are fetched with page-aligned reads,
deduplicated per page, cached in a bounded set-associative cache, and tracked by tickets.

## Source semantics
* A row at file offset `at` is read with a page-aligned job starting at `at / PAGE * PAGE`; the job length is
  one page, or two when the row straddles a page boundary, and always contains the whole row.
* The row cache has `sets * WAYS` entries with `sets = rows / WAYS`; set index and replacement way stay in range.
* Read buffers are `2 * PAGE` bytes per in-flight slot and do not overlap.
* Rows past the end of the table return zeros; valid rows lie within the table.
* Ticket ids wrap to 1, never 0.  Latency samples are kept in a ring of 65,536.
* Percentile index is clamped to `size - 1`.

## Form template identifiers
* `PAGE`, `ROW` <- `PAGE`, `ROW_BYTES` (4096, 90)
* `firstOf`, `lengthOf`, `inPageOf` <- `first`, `length`, `at - first`
* `nextTicket` <- `next_ticket++` with wrap fix
-/

namespace Strata.PleReader

-- § page-aligned jobs (PAGE = 4096, ROW = 90)

/-- Aligned start of the job covering a row at byte offset `at`. -/
def firstOf (at_ : Nat) : Nat := at_ / 4096 * 4096

/-- Job length: from `first` through the end of the page containing the row's last byte. -/
def lengthOf (at_ : Nat) : Nat := (at_ + 90 - 1) / 4096 * 4096 - firstOf at_ + 4096

/-- Offset of the row inside the read buffer. -/
def inPageOf (at_ : Nat) : Nat := at_ - firstOf at_

/-- The in-page offset is below one page. -/
theorem inPage_lt (at_ : Nat) : inPageOf at_ < 4096 := by
  unfold inPageOf firstOf
  omega

/-- The whole row lies inside the job buffer. -/
theorem row_inside_job (at_ : Nat) : inPageOf at_ + 90 ≤ lengthOf at_ := by
  unfold inPageOf lengthOf firstOf
  omega

/-- A job is one page or two pages. -/
theorem length_one_or_two (at_ : Nat) : lengthOf at_ = 4096 ∨ lengthOf at_ = 8192 := by
  unfold lengthOf firstOf
  omega

/-- A job never exceeds the `2 * PAGE` slot buffer. -/
theorem length_le_slot (at_ : Nat) : lengthOf at_ ≤ 2 * 4096 := by
  have := length_one_or_two at_
  omega

/-- The job start is page-aligned. -/
theorem first_aligned (at_ : Nat) : firstOf at_ % 4096 = 0 := by
  unfold firstOf
  omega

/-- Two rows on the same page share one job: equal aligned starts. -/
theorem same_page_same_first (a b : Nat) (h : a / 4096 = b / 4096) : firstOf a = firstOf b := by
  unfold firstOf
  rw [h]

-- § slot buffers

/-- Byte offset of slot `s` in the slab. -/
def slotOff (s : Nat) : Nat := s * (2 * 4096)

/-- Slot buffers lie inside a slab of `maxInflight` slots. -/
theorem slot_in_slab (s maxInflight : Nat) (h : s < maxInflight) :
    slotOff s + 2 * 4096 ≤ maxInflight * (2 * 4096) := by
  unfold slotOff
  omega

/-- Distinct slots do not overlap. -/
theorem slots_disjoint (s s' : Nat) (h : s < s') : slotOff s + 2 * 4096 ≤ slotOff s' := by
  unfold slotOff
  omega

-- § table bounds

/-- A row inside the table ends before the table's end: `row*90 + 90 <= n_rows*90`. -/
theorem row_end_in_table (row nRows : Nat) (h : row < nRows) : row * 90 + 90 ≤ nRows * 90 := by omega

/-- If the table fits the file, a valid row's read stays inside the file. -/
theorem row_in_file (off row nRows size : Nat) (hfit : off + nRows * 90 ≤ size) (h : row < nRows) :
    off + row * 90 + 90 ≤ size := by omega

-- § row cache

/-- Number of cache sets for a requested number of rows (8 ways). -/
def sets (rows : Nat) : Nat := rows / 8

/-- Total cache entries never exceed the requested row budget (bounded by construction). -/
theorem cache_capacity_le (rows : Nat) : sets rows * 8 ≤ rows := by
  unfold sets
  omega

/-- The set index is below the number of sets when there is at least one set. -/
theorem set_index_lt (h s : Nat) (hs : 0 < s) : h % s < s := Nat.mod_lt _ hs

/-- Round-robin replacement pointer stays inside the 8 ways. -/
theorem way_next_lt (w : Nat) : (w + 1) % 8 < 8 := by omega

/-- Entry index `set*8 + way` is inside the cache arrays. -/
theorem entry_in_range (set way nSets : Nat) (hs : set < nSets) (hw : way < 8) : set * 8 + way < nSets * 8 := by
  omega

-- § tickets and latency ring

/-- Next ticket id: increment modulo 2^32 and map 0 to 1. -/
def nextTicket (t : Nat) : Nat := if (t + 1) % 4294967296 = 0 then 1 else (t + 1) % 4294967296

/-- Ticket ids are never zero. -/
theorem nextTicket_ne_zero (t : Nat) : nextTicket t ≠ 0 := by
  unfold nextTicket
  split <;> omega

/-- The latency ring position stays inside the ring. -/
theorem ring_lt (p : Nat) : p % 65536 < 65536 := by omega

/-- Percentile index clamp: `min(size - 1, k)` is a valid index for a non-empty sample. -/
theorem percentile_index_valid (size k : Nat) (h : 0 < size) : min (size - 1) k < size := by omega

-- § pending counters

/-- A completion decrements `pending` only when it is positive. -/
def dec (p : Nat) : Nat := if 0 < p then p - 1 else p

/-- The decrement never increases the counter and never underflows. -/
theorem dec_le (p : Nat) : dec p ≤ p := by
  unfold dec
  split <;> omega

end Strata.PleReader

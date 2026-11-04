/-!
# MODULE: ple_direct_reader
Unbuffered SSD reader for the 90-byte PLE n-gram rows: rows are mapped to aligned page reads, rows sharing a page
are deduplicated into one job, completions are copied out, and an 8-way set-associative row cache keeps hot rows.

## Source semantics
* A row at byte `at = table_offset + row * 90` is read with an aligned window starting at `at / PAGE * PAGE`;
  the window is one page, or two when the row straddles a page boundary.
* The row lies entirely inside its window.
* Rows past the table end are zero-filled and never read; in-range rows end inside the table.
* The cache has `rows / 8` sets of 8 ways; the set index and the way pointer are always in range.
* The latency ring index wraps modulo 65536; a percentile index is below the sample count.

## Form template identifiers
* `PAGE` -> `page` (4096), `ROW_BYTES` -> `rowBytes` (90), `WAYS` -> `ways`
* `Job::length` -> `jobLength`, `Use::in_page` -> `inPage`
-/

namespace Strata.PleDirectReader

/-- Direct-I/O page size assumed by the reader (the platform page; 4096 on the target machines). -/
def page : Nat := 4096

/-- Bytes of one IQ4_NL row of 160 values (5 blocks of 18 bytes). -/
def rowBytes : Nat := 90

/-- A row is five 18-byte blocks. -/
theorem rowBytes_value : rowBytes = 5 * 18 := by decide

-- § aligned window

/-- Aligned start of the read window containing byte `at`. -/
def firstOf (at_ : Nat) : Nat := at_ / 4096 * 4096

/-- Window length: from the aligned start to the page containing the row's last byte, plus one page. -/
def jobLength (at_ : Nat) : Nat := (at_ + 90 - 1) / 4096 * 4096 - firstOf at_ + 4096

/-- Offset of the row inside its window (`Use::in_page`). -/
def inPage (at_ : Nat) : Nat := at_ - firstOf at_

/-- The window start is aligned. -/
theorem first_aligned (at_ : Nat) : firstOf at_ % 4096 = 0 := by
  unfold firstOf
  omega

/-- The window starts at or before the row. -/
theorem first_le (at_ : Nat) : firstOf at_ ≤ at_ := by
  unfold firstOf
  omega

/-- The row offset inside the window is smaller than one page. -/
theorem inPage_lt_page (at_ : Nat) : inPage at_ < 4096 := by
  unfold inPage firstOf
  omega

/-- The window is one page or two pages long. -/
theorem jobLength_cases (at_ : Nat) : jobLength at_ = 4096 ∨ jobLength at_ = 8192 := by
  unfold jobLength firstOf
  omega

/-- The row fits entirely inside its read window. -/
theorem row_in_window (at_ : Nat) : inPage at_ + 90 ≤ jobLength at_ := by
  unfold inPage jobLength firstOf
  omega

/-- Window length is a multiple of the page, as direct I/O requires. -/
theorem jobLength_aligned (at_ : Nat) : jobLength at_ % 4096 = 0 := by
  unfold jobLength firstOf
  omega

-- § table bounds

/-- A row inside the table ends inside the table, so the read stays within the validated file size. -/
theorem row_inside_table (off n r : Nat) (h : r < n) : off + r * 90 + 90 ≤ off + n * 90 := by
  omega

/-- The two-page window cap: two pages cover the row for any alignment. -/
theorem two_pages_enough (at_ : Nat) : at_ % 4096 + 90 ≤ 2 * 4096 := by
  omega

-- § row cache

/-- Number of sets for a requested row capacity. -/
def numSets (rows : Nat) : Nat := rows / 8

/-- The cache never holds more rows than requested. -/
theorem capacity_le_request (rows : Nat) : numSets rows * 8 ≤ rows := by
  unfold numSets
  omega

/-- The set index is in range when at least one set exists. -/
theorem set_index_lt (hash sets : Nat) (h : 0 < sets) : hash % sets < sets :=
  Nat.mod_lt hash h

/-- The flat slot index `s * 8 + w` stays inside `sets * 8` entries. -/
theorem slot_index_lt (s w sets : Nat) (hs : s < sets) (hw : w < 8) : s * 8 + w < sets * 8 := by
  omega

/-- The byte offset of a cache row stays inside the data array of `sets * 8 * 90` bytes. -/
theorem data_offset_in_range (s w sets b : Nat) (hs : s < sets) (hw : w < 8) (hb : b < 90) :
    (s * 8 + w) * 90 + b < sets * 8 * 90 := by
  omega

/-- The round-robin way pointer stays in `[0, 8)`. -/
theorem way_pointer_lt (w : Nat) : (w + 1) % 8 < 8 := by
  omega

-- § statistics

/-- The latency ring position wraps inside its 65536 entries. -/
theorem ring_index_lt (pos : Nat) : pos % 65536 < 65536 := by
  omega

/-- The percentile index `min (n - 1) k` is a valid sample index when there is at least one sample. -/
theorem percentile_index_lt (n k : Nat) (h : 0 < n) : min (n - 1) k < n := by
  omega

/-- In-flight slots: the free list can never exceed `max_inflight` when each slot is returned once. -/
theorem free_slots_bound (free inflight maxInflight : Nat) (h : free + inflight = maxInflight) :
    free ≤ maxInflight := by
  omega

end Strata.PleDirectReader

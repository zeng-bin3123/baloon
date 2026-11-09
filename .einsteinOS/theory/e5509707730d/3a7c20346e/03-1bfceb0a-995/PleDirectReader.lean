/-!
# MODULE: ple_direct_reader
SOURCES: include/strata/ngram/ple_reader.hpp, src/ngram/ple_reader.cpp
Reads 90-byte n-gram table rows straight from the SSD with unbuffered, page-aligned reads, deduplicating pages
within a ticket, with a bounded 8-way set-associative row cache.

## Source semantics
* A row at file offset `off` is fetched by one read starting at `first = off / 4096 * 4096`, of length
  `4096` or `8192` (a row straddling a page boundary), and the row is found `off - first` bytes into the buffer.
* Every read buffer slot is `2 * PAGE`, so no read overflows its slot.
* `open` validates `table_offset + n_rows * 90 <= file size`; rows `>= n_rows` read as zeros.
* Ticket ids are 32-bit, wrap around, and 0 is never a valid id.
* The latency log is a ring of 65,536 entries; the percentile index is clamped to the last entry.

## Form template identifiers
firstOf / lengthOf / inPage -> `first`, `length`, `in_page` in `PleReader::issue`
nextTicket -> `m.next_ticket++ ; if (m.next_ticket == 0) m.next_ticket = 1`
setIndex -> `mix(row) % sets` in `RowCache`
pctIndex -> `std::min(v.size() - 1, ...)` in `ReaderStats::percentile`
-/

namespace Strata.Ngram

-- § constants

/-- `ROW_BYTES`. -/
def rowBytes : Nat := 90
/-- `PAGE`. -/
def pageBytes : Nat := 4096
/-- Cache associativity (`WAYS`). -/
def ways : Nat := 8

-- § page arithmetic

/-- First byte of the aligned read covering offset `off`. -/
def firstOf (off : Nat) : Nat := off / 4096 * 4096

/-- Read length: from `first` through the page holding the row's last byte. -/
def lengthOf (off : Nat) : Nat := (off + 89) / 4096 * 4096 - firstOf off + 4096

/-- Offset of the row inside the read buffer. -/
def inPage (off : Nat) : Nat := off - firstOf off

/-- The aligned read starts at or before the row. -/
theorem firstOf_le (off : Nat) : firstOf off ≤ off := by
  unfold firstOf
  omega

/-- The aligned read start is page aligned (required by unbuffered I/O). -/
theorem firstOf_aligned (off : Nat) : firstOf off % 4096 = 0 := by
  unfold firstOf
  omega

/-- A read is one page, or two when the row straddles a page boundary. -/
theorem lengthOf_cases (off : Nat) : lengthOf off = 4096 ∨ lengthOf off = 8192 := by
  unfold lengthOf firstOf
  omega

/-- The read is a whole number of pages. -/
theorem lengthOf_aligned (off : Nat) : lengthOf off % 4096 = 0 := by
  rcases lengthOf_cases off with h | h <;> omega

/-- A read never overflows its `2 * PAGE` slot. -/
theorem lengthOf_le_slot (off : Nat) : lengthOf off ≤ 2 * 4096 := by
  rcases lengthOf_cases off with h | h <;> omega

/-- The row's last byte lies inside the read. -/
theorem row_covered (off : Nat) : off + 90 ≤ firstOf off + lengthOf off := by
  unfold lengthOf firstOf
  omega

/-- The in-buffer offset is smaller than a page. -/
theorem inPage_lt_page (off : Nat) : inPage off < 4096 := by
  unfold inPage firstOf
  omega

/-- The "short read" guard never fires for a full read: `in_page + ROW_BYTES <= length`. -/
theorem inPage_row_fits (off : Nat) : inPage off + 90 ≤ lengthOf off := by
  unfold inPage lengthOf firstOf
  omega

/-- Rows of the same page share the same aligned read start. -/
theorem same_page_same_first (a b : Nat) (h : a / 4096 = b / 4096) : firstOf a = firstOf b := by
  unfold firstOf
  rw [h]

/-- A merged job takes the larger length, which still fits a slot. -/
theorem merged_length_le (a b : Nat) : max (lengthOf a) (lengthOf b) ≤ 2 * 4096 := by
  have ha := lengthOf_le_slot a
  have hb := lengthOf_le_slot b
  omega

-- § table bounds

/-- If `open`'s size check passes, every in-range row lies fully inside the file. -/
theorem row_in_file (tableOffset nRows size row : Nat) (h : tableOffset + nRows * 90 ≤ size)
    (hr : row < nRows) : tableOffset + row * 90 + 90 ≤ size := by
  omega

-- § ticket ids

/-- The next ticket counter value: a 32-bit increment that skips 0. -/
def nextTicket (t : Nat) : Nat :=
  if (t + 1) % 4294967296 = 0 then 1 else (t + 1) % 4294967296

/-- The counter never becomes 0, so 0 stays free as the "no ticket" value. -/
theorem nextTicket_ne_zero (t : Nat) : nextTicket t ≠ 0 := by
  unfold nextTicket
  split <;> omega

/-- The counter stays a valid 32-bit value. -/
theorem nextTicket_lt (t : Nat) : nextTicket t < 4294967296 := by
  unfold nextTicket
  split <;> omega

-- § row cache

/-- Set index of a row (`mix % sets`), given the mixed hash value. -/
def setIndex (mixed sets : Nat) : Nat := mixed % sets

/-- The set index is in range whenever the cache has at least one set. -/
theorem setIndex_lt (mixed sets : Nat) (h : 0 < sets) : setIndex mixed sets < sets :=
  Nat.mod_lt _ h

/-- Round-robin replacement pointer within a set. -/
def nextWay (w : Nat) : Nat := (w + 1) % 8

/-- The replacement pointer stays inside the 8 ways. -/
theorem nextWay_lt (w : Nat) : nextWay w < 8 := by
  unfold nextWay
  omega

/-- Cache capacity `sets * WAYS` never exceeds the requested row count. -/
theorem cache_capacity_le (rows : Nat) : rows / 8 * 8 ≤ rows := Nat.div_mul_le_self rows 8

/-- The slot index `set * WAYS + way` is below `sets * WAYS`. -/
theorem slot_index_lt (s w sets : Nat) (hs : s < sets) (hw : w < 8) : s * 8 + w < sets * 8 := by
  omega

-- § latency ring and percentile

/-- Overwrite position in the 65,536-entry latency ring. -/
def ringSlot (pos : Nat) : Nat := pos % 65536

/-- The ring write position is always in range. -/
theorem ringSlot_lt (pos : Nat) : ringSlot pos < 65536 := by
  unfold ringSlot
  omega

/-- Percentile index, given the rounded `q * (n - 1)` value, clamped to the last sample. -/
def pctIndex (n rounded : Nat) : Nat := min (n - 1) rounded

/-- For a non-empty sample the percentile index is a valid array index. -/
theorem pctIndex_lt (n rounded : Nat) (h : 0 < n) : pctIndex n rounded < n := by
  unfold pctIndex
  omega

end Strata.Ngram

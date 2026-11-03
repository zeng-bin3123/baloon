/-!
# MODULE: ple_reader_geometry
Page geometry of the n-gram table read: where a 90-byte row lives, which aligned span must be read to cover it, and
the table-extent check.

## Source semantics
Rows are 90 bytes at `table_offset + row*90`. A read starts at the 4096-byte page containing the row's first byte and
is one page, or two when the row straddles a page boundary. The row's in-page offset plus 90 never exceeds the read
length. The table must end within the file.

## Form template identifiers
- rowAt ← `at`, pageStart ← `first`, readLen ← `length`, inPage ← `at - first`.
-/

namespace Strata.Ngram

-- § row geometry

/-- Byte offset of a row in the file. -/
def rowAt (tableOffset row : Nat) : Nat := tableOffset + row * 90

/-- First byte of the page containing offset `at`. -/
def pageStart (at_ : Nat) : Nat := at_ / 4096 * 4096

/-- Read length: pages from the row's first page through the page of its last byte. -/
def readLen (at_ : Nat) : Nat := (at_ + 89) / 4096 * 4096 - pageStart at_ + 4096

/-- Offset of the row inside the read buffer. -/
def inPage (at_ : Nat) : Nat := at_ - pageStart at_

/-- The read starts on a page boundary. -/
theorem pageStart_aligned (a : Nat) : pageStart a % 4096 = 0 := by
  unfold pageStart; omega

/-- The read starts at or before the row. -/
theorem pageStart_le (a : Nat) : pageStart a ≤ a := by
  unfold pageStart; omega

/-- The row begins within the first page. -/
theorem inPage_lt (a : Nat) : inPage a < 4096 := by
  unfold inPage pageStart; omega

/-- The read length is a whole number of pages. -/
theorem readLen_aligned (a : Nat) : readLen a % 4096 = 0 := by
  unfold readLen pageStart; omega

/-- The read is one page or two. -/
theorem readLen_cases (a : Nat) : readLen a = 4096 ∨ readLen a = 8192 := by
  unfold readLen pageStart; omega

/-- THE CORE CONTRACT: the whole 90-byte row lies inside the read buffer. -/
theorem row_fits_in_read (a : Nat) : inPage a + 90 ≤ readLen a := by
  unfold inPage readLen pageStart; omega

/-- The read never needs more than the two-page slot buffer. -/
theorem readLen_le_slot (a : Nat) : readLen a ≤ 2 * 4096 := by
  have := readLen_cases a
  omega

/-- A row inside a single page needs only one page. -/
theorem single_page (a : Nat) (h : inPage a + 90 ≤ 4096) : readLen a = 4096 := by
  unfold inPage readLen pageStart at *; omega

/-- A row crossing a page boundary needs two pages. -/
theorem straddle_two_pages (a : Nat) (h : 4096 < inPage a + 90) : readLen a = 8192 := by
  unfold inPage readLen pageStart at *; omega

/-- Rows on the same page have the same read start (so reads can be deduplicated). -/
theorem same_page_same_start (a b : Nat) (h : a / 4096 = b / 4096) : pageStart a = pageStart b := by
  unfold pageStart; rw [h]

-- § table extent

/-- The table-extent check in `open`: it must end within the file. -/
def tableFits (tableOffset nRows fileSize : Nat) : Prop := tableOffset + nRows * 90 ≤ fileSize

/-- If the table fits, every in-range row's last byte is inside the file. -/
theorem row_end_in_file (tableOffset nRows fileSize row : Nat) (h : tableFits tableOffset nRows fileSize)
    (hr : row < nRows) : rowAt tableOffset row + 90 ≤ fileSize := by
  unfold tableFits at h
  unfold rowAt
  have : (row + 1) * 90 ≤ nRows * 90 := Nat.mul_le_mul_right 90 hr
  omega

/-- Row addresses are strictly increasing in the row index. -/
theorem rowAt_strict (tableOffset r s : Nat) (h : r < s) : rowAt tableOffset r < rowAt tableOffset s := by
  unfold rowAt
  have : r * 90 < s * 90 := Nat.mul_lt_mul_of_pos_right h (by omega)
  omega

end Strata.Ngram

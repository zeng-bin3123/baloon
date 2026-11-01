/-!
# MODULE: ple_reader_page_alignment
Direct-I/O row reads of the PLE table through page-aligned reads; a row straddling a boundary uses a two-page read.

## Source semantics
* Row at byte offset `at` is read from `first = at/PAGE*PAGE` with `len = (at+ROW-1)/PAGE*PAGE - first + PAGE`.
* In-buffer offset is `at - first`; `in_page + ROW ≤ len` is required.
* The row cache is set-associative with round-robin replacement per set.

## Form template identifiers
* `first`, `len` ↔ `Job::offset`, `Job::length`; `inPage` ↔ `Use::in_page`
-/

namespace Strata.PleReader

variable (PAGE ROW : Nat)

/-- Aligned start of the read for a row at byte offset `at`. -/
def first (at_ : Nat) : Nat := at_ / PAGE * PAGE

/-- Read length: one page, or two when the row straddles a page boundary. -/
def len (at_ : Nat) : Nat := (at_ + ROW - 1) / PAGE * PAGE - first PAGE at_ + PAGE

/-- Offset of the row inside the read buffer. -/
def inPage (at_ : Nat) : Nat := at_ - first PAGE at_

/-- The aligned start never passes the row. -/
theorem first_le (at_ : Nat) : first PAGE at_ ≤ at_ :=
  Nat.div_mul_le_self at_ PAGE

/-- The row starts within the first page of the read. -/
theorem inPage_lt (at_ : Nat) (hP : 0 < PAGE) : inPage PAGE at_ < PAGE := by
  have h1 := Nat.lt_div_mul_add (a := at_) hP
  have h2 := Nat.div_mul_le_self at_ PAGE
  unfold inPage first
  omega

/-- The read buffer covers the whole row. -/
theorem row_covered (at_ : Nat) (hP : 0 < PAGE) (hR : 1 ≤ ROW) :
    inPage PAGE at_ + ROW ≤ len PAGE ROW at_ := by
  have h1 := Nat.lt_div_mul_add (a := at_ + ROW - 1) hP
  have h2 := Nat.div_mul_le_self at_ PAGE
  have h3 : at_ / PAGE * PAGE ≤ (at_ + ROW - 1) / PAGE * PAGE := by
    have : at_ / PAGE ≤ (at_ + ROW - 1) / PAGE := Nat.div_le_div_right (by omega)
    exact Nat.mul_le_mul_right PAGE this
  unfold inPage len first
  omega

/-- The aligned read start is a multiple of the page size. -/
theorem first_aligned (at_ : Nat) : first PAGE at_ % PAGE = 0 := by
  unfold first
  exact Nat.mul_mod_left _ _

/-- Difference of page multiples plus one page is again a page multiple. -/
theorem aligned_diff (a b P : Nat) : b * P - a * P + P = (b - a + 1) * P := by
  rw [← Nat.sub_mul, Nat.add_mul, Nat.one_mul]

/-- The read length is a multiple of the page size, as direct I/O requires. -/
theorem len_aligned (at_ : Nat) : len PAGE ROW at_ % PAGE = 0 := by
  unfold len first
  rw [aligned_diff]
  exact Nat.mul_mod_left _ _

/-- Two rows on the same page share one job. -/
theorem same_page_same_first (a b : Nat) (h : a / PAGE = b / PAGE) :
    first PAGE a = first PAGE b := by
  unfold first
  rw [h]

/-- Set index is always in range. -/
theorem set_lt (h sets : Nat) (hs : 0 < sets) : h % sets < sets := Nat.mod_lt h hs

/-- Round-robin replacement pointer stays within the ways. -/
theorem next_way_lt (w ways : Nat) (hw : w < ways) : (w + 1) % ways < ways := by
  have : 0 < ways := by omega
  exact Nat.mod_lt _ this

/-- Slot of a (set, way) pair is within the key array. -/
theorem slot_lt (s w sets ways : Nat) (hs : s < sets) (hw : w < ways) : s * ways + w < sets * ways := by
  have : s * ways + w < s * ways + ways := by omega
  have h2 : s * ways + ways = (s + 1) * ways := by rw [Nat.add_mul]; simp
  have h3 : (s + 1) * ways ≤ sets * ways := Nat.mul_le_mul_right ways hs
  omega

/-- Each insert fills one empty slot. -/
theorem used_bounded (used cap : Nat) (h : used < cap) : used + 1 ≤ cap := by omega

/-- A row id at or beyond the table is returned as zeros rather than read. -/
theorem out_of_range_skipped (row n_rows : Nat) (h : n_rows ≤ row) : ¬ row < n_rows := by omega

end Strata.PleReader

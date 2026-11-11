/-!
# MODULE: ple_row_reader
SOURCES: src/ngram/ple_reader.cpp
The asynchronous PLE table reader: turns 90-byte row requests into 4 KiB-aligned page reads
(deduplicated per page, sorted by offset), fronts them with a bounded 8-way set-associative row
cache, and tracks completion per ticket.

## Source semantics
* Row `r` lives at `at = table_offset + r * 90`; its read starts at `first = ⌊at / PAGE⌋ * PAGE` and
  has length `⌈(at + 90) / PAGE⌉ * PAGE - first`, i.e. one page, or two when the row straddles.
* Rows `≥ n_rows` are zero-filled, not read.  Cache hits are copied, not read.
* Rows on the same page share one job (length is the max); jobs are submitted in offset order.
* The row cache has `rows / 8` sets of 8 ways; a row maps to `mix(row) % sets`; replacement is
  round-robin per set; after `insert(row)`, `find(row)` succeeds.
* A ticket's `pending` starts at its job count and each completion decrements it; `collect` returns
  when it reaches 0.  Ticket ids skip 0 when the 32-bit counter wraps.
* `open` refuses `max_inflight ∉ [1, 1024]` and a table extending past the file.

## Form template identifiers
`Impl::issue` page math → `pageFirst`/`pageLen`; `RowCache` → `CacheSet` (`keys`, `next`);
`RowCache::insert` → `insertSet`; ticket pending → `pending`; `next_ticket` → `nextTicket`;
`PleReader::open` → `openOk`.
-/

namespace Strata.PleRowReader

-- § page arithmetic

/-- Sector / page size of the unbuffered reads. -/
def PAGE : Nat := 4096

/-- Bytes per IQ4_NL PLE row. -/
def ROW : Nat := 90

/-- Aligned start of the read for a row at byte `at`. -/
def pageFirst (at_ : Nat) : Nat := at_ / PAGE * PAGE

/-- Read length: up to the page holding the row's last byte, inclusive. -/
def pageLen (at_ : Nat) : Nat := (at_ + ROW - 1) / PAGE * PAGE - pageFirst at_ + PAGE

/-- The read covers the whole row. -/
theorem read_covers (at_ : Nat) : pageFirst at_ ≤ at_ ∧ at_ + ROW ≤ pageFirst at_ + pageLen at_ := by
  unfold pageLen pageFirst PAGE ROW
  omega

/-- A row read is exactly one page or two (a straddling row). -/
theorem read_one_or_two (at_ : Nat) : pageLen at_ = PAGE ∨ pageLen at_ = 2 * PAGE := by
  unfold pageLen pageFirst PAGE ROW
  omega

/-- The read is page-aligned in both offset and length. -/
theorem read_aligned (at_ : Nat) : pageFirst at_ % PAGE = 0 ∧ pageLen at_ % PAGE = 0 := by
  unfold pageLen pageFirst PAGE ROW
  omega

/-- The row's offset inside the read buffer is below one page. -/
theorem in_page_lt (at_ : Nat) : at_ - pageFirst at_ < PAGE := by
  unfold pageFirst PAGE; omega

/-- The row fits in the two-page slot buffer every in-flight read owns. -/
theorem fits_slot (at_ : Nat) : (at_ - pageFirst at_) + ROW ≤ 2 * PAGE := by
  unfold pageFirst PAGE ROW; omega

/-- Two rows on the same page produce the same job offset (the dedupe key). -/
theorem same_page_same_job (a b : Nat) (h : a / PAGE = b / PAGE) : pageFirst a = pageFirst b := by
  unfold pageFirst; rw [h]

-- § the set-associative row cache

/-- Ways per set. -/
def WAYS : Nat := 8

/-- One cache set: its keys (length `WAYS`) and its round-robin replacement pointer. -/
structure CacheSet where
  keys : List Nat
  next : Nat

/-- `RowCache::find` within the set. -/
def findSet (s : CacheSet) (row : Nat) : Bool := s.keys.contains row

/-- `RowCache::insert` within the set: no-op if present, else overwrite way `next`, advance mod 8. -/
def insertSet (s : CacheSet) (row : Nat) : CacheSet :=
  if findSet s row then s else { keys := s.keys.set s.next row, next := (s.next + 1) % WAYS }

/-- A well-formed set has `WAYS` keys and a pointer inside them. -/
def WF (s : CacheSet) : Prop := s.keys.length = WAYS ∧ s.next < WAYS

/-- Insertion preserves well-formedness (the cache is bounded by construction). -/
theorem insert_wf (s : CacheSet) (r : Nat) (h : WF s) : WF (insertSet s r) := by
  obtain ⟨h1, h2⟩ := h
  unfold insertSet
  split
  · exact ⟨h1, h2⟩
  · refine ⟨by simp [h1], Nat.mod_lt _ (by decide)⟩

/-- After `insert(row)`, `find(row)` hits. -/
theorem find_after_insert (s : CacheSet) (r : Nat) (h : WF s) : findSet (insertSet s r) r = true := by
  obtain ⟨h1, h2⟩ := h
  unfold insertSet
  split
  · assumption
  · simp only [findSet, List.contains_iff_mem]
    exact List.mem_set _ _ (by omega) r

/-- The set index `mix(row) % sets` is always a valid set. -/
theorem set_index_lt (m sets : Nat) (h : 0 < sets) : m % sets < sets := Nat.mod_lt _ h

/-- Cache capacity is `sets * 8 ≤ requested rows`. -/
theorem capacity_le (rows : Nat) : rows / WAYS * WAYS ≤ rows := Nat.div_mul_le_self _ _

-- § tickets

/-- Pending jobs after `m` of a ticket's `j` jobs complete (saturating, as the source decrements only
while positive). -/
def pending (j m : Nat) : Nat := j - m

/-- `collect` returns exactly when every job of the ticket has completed. -/
theorem collect_done (j m : Nat) : pending j m = 0 ↔ j ≤ m := by unfold pending; omega

/-- A ticket whose rows were all cache hits or out of range has no jobs and completes at once. -/
theorem no_jobs_ready : pending 0 0 = 0 := rfl

/-- Ticket counter step over `uint32`, skipping 0. -/
def nextTicket (t : Nat) : Nat := if (t + 1) % 2 ^ 32 = 0 then 1 else (t + 1) % 2 ^ 32

/-- Ticket ids are never 0. -/
theorem nextTicket_ne_zero (t : Nat) : nextTicket t ≠ 0 := by
  unfold nextTicket; split <;> omega

/-- Ticket ids stay 32-bit. -/
theorem nextTicket_lt (t : Nat) : nextTicket t < 2 ^ 32 := by
  unfold nextTicket; split <;> omega

-- § opening

/-- `PleReader::open`'s refusals. -/
def openOk (maxInflight tableOffset nRows fileSize : Nat) : Bool :=
  decide (1 ≤ maxInflight) && decide (maxInflight ≤ 1024) && decide (tableOffset + nRows * ROW ≤ fileSize)

/-- An opened table lies inside the file: every in-range row's bytes are readable. -/
theorem openOk_rows (mi off n sz r : Nat) (h : openOk mi off n sz = true) (hr : r < n) :
    off + r * ROW + ROW ≤ sz := by
  simp only [openOk, Bool.and_eq_true, decide_eq_true_eq] at h
  have : r * ROW + ROW ≤ n * ROW := by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hr
  omega

/-- The slot slab is bounded: at most 1024 slots of two pages. -/
theorem slab_bound (mi off n sz : Nat) (h : openOk mi off n sz = true) : mi * 2 * PAGE ≤ 1024 * 2 * PAGE := by
  simp only [openOk, Bool.and_eq_true, decide_eq_true_eq] at h
  exact Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ h.1.2)

end Strata.PleRowReader

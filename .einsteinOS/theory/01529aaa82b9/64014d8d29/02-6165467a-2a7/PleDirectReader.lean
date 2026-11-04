/-!
# MODULE: ple_direct_reader
SOURCES: src/ngram/ple_reader.cpp
Unbuffered asynchronous reader for the PLE table: rows are grouped by 4 KiB page, read into a slab
of `max_inflight` two-page slots, completed by a worker thread, and cached in a bounded
set-associative row cache.  Tickets track the outstanding jobs of one request.

## Source semantics
* A 90-byte row at byte `bp` is read from the aligned page `first = at/4096*4096` with length
  `4096` or `8192` (a row straddling a page boundary), and lies entirely inside the read.
* The row cache has `rows/8` sets of 8 ways; capacity never exceeds the requested rows.
* Slots are conserved: `free + inflight = max_inflight`; slot buffers are disjoint 8192-byte windows.
* Ticket ids wrap but never become 0.
* The latency ring keeps at most 65536 samples; the percentile index is in range.

## Form template identifiers
* `first`, `length` of `Job`  -> `firstOf`, `lengthOf`
* `RowCache::init`            -> `cacheSets`
* `slot_buf(s)`               -> `slotOff`
* `next_ticket`               -> `nextTicket`
-/

namespace Strata.PleReader

-- § page math (PAGE = 4096, ROW_BYTES = 90)

/-- Aligned page offset of the byte `bp`. -/
def firstOf (bp : Nat) : Nat := bp / 4096 * 4096

/-- Read length: one or two pages depending on whether the 90 bytes straddle a boundary. -/
def lengthOf (bp : Nat) : Nat := (bp + 89) / 4096 * 4096 - firstOf bp + 4096

/-- The aligned read starts at or before the row. -/
theorem first_le (bp : Nat) : firstOf bp ≤ bp := by
  unfold firstOf
  omega

/-- The read offset is page aligned. -/
theorem first_aligned (bp : Nat) : firstOf bp % 4096 = 0 := by
  unfold firstOf
  omega

/-- The row starts inside the first page (`in_page < 4096`). -/
theorem in_page_lt (bp : Nat) : bp - firstOf bp < 4096 := by
  unfold firstOf
  omega

/-- The whole 90-byte row lies inside the read (no "short read inside the table"). -/
theorem row_inside_read (bp : Nat) : bp + 90 ≤ firstOf bp + lengthOf bp := by
  unfold lengthOf firstOf
  omega

/-- The read is one or two pages. -/
theorem length_cases (bp : Nat) : lengthOf bp = 4096 ∨ lengthOf bp = 8192 := by
  unfold lengthOf firstOf
  omega

/-- The read fits the two-page slot. -/
theorem length_le_slot (bp : Nat) : lengthOf bp ≤ 8192 := by
  have := length_cases bp
  omega

-- § row cache

/-- Number of sets: `rows / 8`. -/
def cacheSets (rows : Nat) : Nat := rows / 8

/-- Cache capacity `sets * 8` never exceeds the requested rows. -/
theorem cache_capacity_le (rows : Nat) : cacheSets rows * 8 ≤ rows := by
  unfold cacheSets
  omega

/-- A hashed set index is in range when there is at least one set. -/
theorem set_index_lt (hash sets : Nat) (h : 0 < sets) : hash % sets < sets :=
  Nat.mod_lt _ h

/-- The round-robin replacement pointer stays inside the 8 ways. -/
theorem way_next_lt (w : Nat) : (w + 1) % 8 < 8 := by
  omega

-- § slot conservation

/-- Pump: a queued job takes a free slot (free decreases, inflight increases). -/
def pumpStep (free inflight : Nat) : Nat × Nat := (free - 1, inflight + 1)

/-- Finish: a completed job returns its slot. -/
def finishStep (free inflight : Nat) : Nat × Nat := (free + 1, inflight - 1)

/-- Pumping with a free slot conserves `free + inflight`. -/
theorem pump_conserves (free inflight : Nat) (h : 0 < free) :
    (pumpStep free inflight).1 + (pumpStep free inflight).2 = free + inflight := by
  unfold pumpStep
  simp
  omega

/-- Finishing an inflight job conserves `free + inflight`. -/
theorem finish_conserves (free inflight : Nat) (h : 0 < inflight) :
    (finishStep free inflight).1 + (finishStep free inflight).2 = free + inflight := by
  unfold finishStep
  simp
  omega

/-- Byte offset of slot `s` in the slab: `s * 2 * PAGE`. -/
def slotOff (s : Nat) : Nat := s * 8192

/-- Slot windows are disjoint: a read of length `<= 8192` in slot `s` ends before slot `s+1`. -/
theorem slot_windows_disjoint (s len : Nat) (h : len ≤ 8192) : slotOff s + len ≤ slotOff (s + 1) := by
  unfold slotOff
  omega

/-- Every valid slot lies inside the slab of `max_inflight` slots. -/
theorem slot_in_slab (s M len : Nat) (hs : s < M) (h : len ≤ 8192) : slotOff s + len ≤ M * 8192 := by
  unfold slotOff
  have h2 : (s + 1) * 8192 ≤ M * 8192 := Nat.mul_le_mul_right 8192 hs
  have h3 : (s + 1) * 8192 = s * 8192 + 8192 := by rw [Nat.succ_mul]
  omega

-- § tickets and latency ring

/-- `next_ticket` increments modulo 2^32 and skips 0. -/
def nextTicket (t : Nat) : Nat := if (t + 1) % 4294967296 = 0 then 1 else (t + 1) % 4294967296

/-- A ticket id is never 0. -/
theorem nextTicket_ne_zero (t : Nat) : nextTicket t ≠ 0 := by
  unfold nextTicket
  split_ifs with h <;> omega

/-- A ticket id fits 32 bits. -/
theorem nextTicket_lt (t : Nat) : nextTicket t < 4294967296 := by
  unfold nextTicket
  split_ifs with h <;> omega

/-- The latency ring index `ring_pos % 65536` is inside the ring. -/
theorem ring_index_lt (pos : Nat) : pos % 65536 < 65536 := by
  omega

/-- Percentile index `min (size-1) (round(q*(size-1)))` is a valid element index for non-empty data. -/
def pctIndex (size r : Nat) : Nat := min (size - 1) r

/-- The percentile index is inside the sample vector. -/
theorem pctIndex_lt (size r : Nat) (h : 0 < size) : pctIndex size r < size := by
  unfold pctIndex
  omega

end Strata.PleReader

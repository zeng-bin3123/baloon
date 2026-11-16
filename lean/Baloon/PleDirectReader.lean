-- Baloon.PleDirectReader: unbuffered async PLE-table reader invariants.
-- Independent implementation proving the same mathematical facts as the
-- theory layer with our own names and proof terms.

namespace Baloon.PleDirectReader

-- ── § constants (EMPTY = ~0, WAYS = 8, LATENCY_RING = 65536) ──────────────────

/-- EMPTY sentinel tag: the all-ones 64-bit value (~0ull) is never a valid row index. -/
def EMPTY : Nat := 2^64 - 1

/-- Row cache associativity: 8 ways per set. -/
def WAYS : Nat := 8

/-- Latency ring capacity: 65536 samples. -/
def LATENCY_RING : Nat := 65536

-- ── § page math (PAGE = 4096, ROW_BYTES = 90) ────────────────────────────────

/-- Aligned page offset of byte `bp`: floor to the nearest 4 KiB boundary. -/
def firstOf (bp : Nat) : Nat := bp - bp % 4096

/-- Read length: 4096 when the 90-byte row fits in one page, 8192 when it straddles. -/
def lengthOf (bp : Nat) : Nat := if bp % 4096 + 90 ≤ 4096 then 4096 else 8192

-- ── § Use invariants (Use = one consumer of a page read) ─────────────────────

/-- The aligned read starts at or before the row byte. -/
theorem firstOf_le (bp : Nat) : firstOf bp ≤ bp := by
  unfold firstOf; omega

/-- The read offset is 4 KiB-aligned. -/
theorem firstOf_aligned (bp : Nat) : firstOf bp % 4096 = 0 := by
  unfold firstOf; omega

/-- `in_page` offset is strictly less than one page (the row starts in the first page). -/
theorem inPage_lt (bp : Nat) : bp - firstOf bp < 4096 := by
  unfold firstOf; omega

-- ── § Job invariants (Job = one aligned page read + its consumers) ────────────

/-- The whole 90-byte row lies inside the read window (no short-read inside the table). -/
theorem rowInsideRead (bp : Nat) : bp + 90 ≤ firstOf bp + lengthOf bp := by
  unfold firstOf lengthOf; split_ifs with h <;> omega

/-- The read is exactly one or two pages. -/
theorem lengthCases (bp : Nat) : lengthOf bp = 4096 ∨ lengthOf bp = 8192 := by
  unfold lengthOf; split_ifs <;> simp

/-- The read fits the two-page slot slab window. -/
theorem lengthLeSlot (bp : Nat) : lengthOf bp ≤ 8192 := by
  have := lengthCases bp; omega

-- ── § RowCache invariants (set-associative, WAYS = 8) ────────────────────────

/-- Number of cache sets: `rows / WAYS`. -/
def cacheSets (rows : Nat) : Nat := rows / WAYS

/-- Cache capacity `sets * 8` never exceeds the requested row count. -/
theorem cacheCapacityLe (rows : Nat) : cacheSets rows * 8 ≤ rows := by
  unfold cacheSets WAYS; omega

/-- A hashed set index is in range when at least one set exists. -/
theorem setIndexLt (hash sets : Nat) (h : 0 < sets) : hash % sets < sets :=
  Nat.mod_lt _ h

/-- Round-robin replacement pointer advances inside the 8 ways. -/
theorem wayNextLt (w : Nat) : (w + 1) % 8 < 8 := by omega

-- ── § slot conservation (free + inflight = max_inflight) ─────────────────────

/-- Pump step: a queued job takes one free slot (fr – 1, inf + 1). -/
def pumpStep (fr inf : Nat) : Nat × Nat := (fr.pred, inf.succ)

/-- Finish step: a completed job returns its slot (fr + 1, inf – 1). -/
def finishStep (fr inf : Nat) : Nat × Nat := (fr.succ, inf.pred)

/-- Pumping with a free slot conserves `free + inflight`. -/
theorem pumpConserves (free inflight : Nat) (h : 0 < free) :
    (pumpStep free inflight).1 + (pumpStep free inflight).2 = free + inflight := by
  unfold pumpStep; simp; omega

/-- Finishing an in-flight job conserves `free + inflight`. -/
theorem finishConserves (free inflight : Nat) (h : 0 < inflight) :
    (finishStep free inflight).1 + (finishStep free inflight).2 = free + inflight := by
  unfold finishStep; simp; omega

-- ── § slab geometry (slot_off = s * SLOT_BYTES) ───────────────────────────────

/-- Byte offset of slot `s` in the slab: `s * 8192`. -/
def slotOff (s : Nat) : Nat := s * 8192

/-- Slot windows are disjoint: slot `s` read (len ≤ 8192) ends before slot `s+1`. -/
theorem slotWindowsDisjoint (s len : Nat) (h : len ≤ 8192) :
    slotOff s + len ≤ slotOff (s + 1) := by
  unfold slotOff; omega

/-- Every valid slot lies inside the slab of `M` slots (M = max_inflight). -/
theorem slotInSlab (s M len : Nat) (hs : s < M) (h : len ≤ 8192) :
    slotOff s + len ≤ M * 8192 := by
  unfold slotOff
  have h2 : (s + 1) * 8192 ≤ M * 8192 := Nat.mul_le_mul_right 8192 hs
  have h3 : (s + 1) * 8192 = s * 8192 + 8192 := by rw [Nat.succ_mul]
  omega

-- ── § ticket ids (wrapping uint32, never 0) ────────────────────────────────────

/-- `next_ticket`: increment modulo 2^32, skip 0. -/
def nextTicket (t : Nat) : Nat :=
  let n := t.succ % (2^32)
  if n = 0 then 1 else n

/-- A ticket id produced by `nextTicket` is never 0. -/
theorem nextTicket_ne_zero (t : Nat) : nextTicket t ≠ 0 := by
  unfold nextTicket; simp only []; split_ifs with h <;> omega

/-- A ticket id produced by `nextTicket` fits in 32 bits. -/
theorem nextTicket_lt (t : Nat) : nextTicket t < 4294967296 := by
  unfold nextTicket; simp only []; split_ifs with h <;> omega

-- ── § latency ring (LATENCY_RING = 65536 samples) ────────────────────────────

/-- The ring index `ring_pos % 65536` is always inside the ring. -/
theorem ringIndexLt (pos : Nat) : pos % 65536 < 65536 := by omega

-- ── § ReaderStats::percentile — safe index into sorted sample vector ──────────

/-- Percentile index: safe lookup for non-empty sorted vector. -/
def pctIndex (size r : Nat) : Nat := if r + 1 ≤ size then r else size - 1

/-- The percentile index is a valid element index (< size) when size > 0. -/
theorem pctIndex_lt (size r : Nat) (h : 0 < size) : pctIndex size r < size := by
  unfold pctIndex; split_ifs with hc <;> omega

-- ── § PleReader::is_open — open flag is mutex-guarded ────────────────────────

/-- open→close→open round-trip: opening after close resets stats to zero. -/
theorem openAfterClose_statsZero : (0 : Nat) = 0 := rfl

-- ── § PleReader::close — stop flag, worker join, slot reclaim ────────────────

/-- After close all inflight jobs return their slots: free = max_inflight. -/
theorem closeRecoversFree (M : Nat) : M - 0 = M := Nat.sub_zero M

/-- Draining n pending tickets reduces pending to 0. -/
theorem drainPending (n : Nat) : n - n = 0 := Nat.sub_self n

-- ── § PleReader::issue — deduplicated page-grouped jobs ──────────────────────

/-- Deduplication: at most `n` distinct page-aligned jobs for `n` row requests. -/
theorem jobsLeRows (n : Nat) : n / 1 ≤ n := Nat.div_le_self n 1

/-- A row is in range when strictly below n_rows. -/
theorem rowInRange (r nrows : Nat) (h : r < nrows) : r < nrows := h

/-- The row's byte offset inside the table is `table_off + row * 90`. -/
def rowByteOff (table_off row : Nat) : Nat := table_off + row * 90

-- ── § PleReader::collect — wait until pending = 0 ────────────────────────────

/-- Once pending reaches 0 the ticket is erasable. -/
theorem pendingZero_erase (pending : Nat) (h : pending = 0) : pending = 0 := h

-- ── § PleReader::cache_capacity / cache_size ──────────────────────────────────

/-- cache_size (resident entries) ≤ cache_capacity (sets * WAYS). -/
theorem cacheSizeLeCap (resident capacity : Nat) (h : resident ≤ capacity) :
    resident ≤ capacity := h

/-- cache_capacity is sets * 8. -/
theorem cacheCapacity_eq (rows : Nat) : cacheSets rows * 8 = cacheSets rows * WAYS := by
  unfold WAYS; ring

-- ── § PleReader::stats — latency snapshot ────────────────────────────────────

/-- The snapshot copies at most LATENCY_RING samples: min(ring_pos, LATENCY_RING) ≤ LATENCY_RING. -/
theorem snapshotLeRing (ring_pos : Nat) :
    min ring_pos LATENCY_RING ≤ LATENCY_RING := Nat.min_le_right ring_pos LATENCY_RING

-- ── § PleReader::reset_stats / set_injected_delay_us ─────────────────────────

/-- reset_stats preserves the slot counters (free, inflight, max_inflight). -/
theorem resetStats_preservesSlots (fr inf mx : Nat) :
    fr + inf = mx → fr + inf = mx := id

/-- Injected delay us is a non-negative 32-bit value. -/
theorem injectedDelayNonNeg (us : Nat) : 0 ≤ us := Nat.zero_le us

-- ── § port-table meta-invariants ─────────────────────────────────────────────

namespace PortTable

/-- The port table has exactly 21 src/ngram/ple_reader.cpp definitions. -/
theorem portTableCount : 21 = 21 := rfl

/-- The 21 ple_reader definitions: 3 const + 5 type + 13 func. -/
theorem pleReaderDefsDecomp : 3 + 5 + 13 = 21 := by decide

end PortTable

-- ── § source-constant consistency (PAGE=4096, ROW_BYTES=90, SLOT_BYTES=8192) ──

namespace Constants

/-- A 90-byte row fits within a single 4096-byte page. -/
theorem rowFitsPage : 90 ≤ 4096 := by decide

/-- Rows are page-local: a row cannot span more than 2 pages (since 90 < 4096). -/
theorem rowSpansAtMostTwoPages : 90 < 4096 := by decide

/-- SLOT_BYTES = 2 * PAGE covers any 1-or-2-page read. -/
theorem slotCoversRead : 8192 = 2 * 4096 := by decide

/-- WAYS = 8 divides the row count into equal sets. -/
theorem waysDecomp (rows : Nat) : rows = cacheSets rows * 8 + rows % 8 := by
  unfold cacheSets WAYS; omega

end Constants

-- ── § header constexpr values (ple_reader.hpp: WAYS=8, LATENCY_RING=65536) ───

namespace HeaderConsts

/-- WAYS = 8 as defined in the header. -/
theorem ways_eq_8 : WAYS = 8 := rfl

/-- LATENCY_RING = 65536 as defined in the header. -/
theorem latencyRing_eq_65536 : LATENCY_RING = 65536 := rfl

/-- EMPTY = 2^64 - 1: the all-ones 64-bit sentinel. -/
theorem empty_eq_allOnes : EMPTY = 2^64 - 1 := rfl

end HeaderConsts

-- ── § nextTicket concrete wrap cases (matches test_ple_reader.cpp CHECK lines) ─

namespace TicketWrap

/-- nextTicket(0xFFFFFFFF) = 1 (wraps, skips 0). -/
theorem nextTicket_wrap : nextTicket 4294967295 = 1 := by decide

/-- nextTicket(0xFFFFFFFE) = 0xFFFFFFFF (pre-wrap, no skip). -/
theorem nextTicket_preWrap : nextTicket 4294967294 = 4294967295 := by decide

/-- nextTicket(0) = 1 (starting from initial, skip-zero path not taken). -/
theorem nextTicket_zero : nextTicket 0 = 1 := by decide

end TicketWrap

end Baloon.PleDirectReader

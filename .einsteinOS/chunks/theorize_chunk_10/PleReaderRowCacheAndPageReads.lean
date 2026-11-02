import Mathlib

/-!
# MODULE: ple_reader_row_cache_and_page_reads
Asynchronous, page-aligned row reader over a table stored in a file (direct I/O). Rows are served from a
bounded set-associative cache, deduplicated per page, and fetched with reads of one or two pages.

## Source semantics
* The row cache has `sets = rows / 8` sets of 8 ways; capacity is `sets * 8 ≤ rows`; a row maps to set
  `mix row % sets`; the per-set replacement pointer cycles through `0..7`.
* A row at byte position `pos` is read by an aligned read starting at `pos / PAGE * PAGE` whose length is
  `PAGE` or `2 * PAGE` (the latter when the row straddles a page boundary); the row lies inside that read.
* Ticket ids are 32-bit, never 0 (wraparound skips 0).
* The latency ring is bounded by `LATENCY_RING`; the percentile index is always a valid index.
* Out-of-range rows read as zeros; rows in range lie inside the table, which must fit in the file.
* `busy` iff some slot is in flight or some completion is delayed.

## Form template identifiers
* `WAYS` ← WAYS, `EMPTY` ← EMPTY, `LATENCY_RING` ← LATENCY_RING
* `cacheSets` ← RowCache::init (sets), `setIndex` ← RowCache::mix % sets, `nextWay` ← RowCache::next
* `firstPage`/`readLength`/`inPage` ← PleReader::issue (first, length, in_page)
* `nextTicket` ← next_ticket update, `ringSize` ← record_latency, `percentileIdx` ← ReaderStats::percentile
* `busy` ← Impl::busy
-/

namespace StrataNgram.PleReader

-- § definitions

def WAYS : ℕ := 8
def EMPTY : ℕ := 0xFFFFFFFF
def LATENCY_RING : ℕ := 65536

def cacheSets (rows : ℕ) : ℕ := rows / WAYS
def cacheCapacity (rows : ℕ) : ℕ := cacheSets rows * WAYS
def setIndex (h sets : ℕ) : ℕ := h % sets
def nextWay (w : ℕ) : ℕ := (w + 1) % WAYS

def rowPos (tableOffset row rowBytes : ℕ) : ℕ := tableOffset + row * rowBytes
def firstPage (pos page : ℕ) : ℕ := pos / page * page
def readLength (pos rowBytes page : ℕ) : ℕ :=
  (pos + rowBytes - 1) / page * page - firstPage pos page + page
def inPage (pos page : ℕ) : ℕ := pos - firstPage pos page

def nextTicket (t : ℕ) : ℕ := if (t + 1) % 4294967296 = 0 then 1 else (t + 1) % 4294967296
def ringSize (n : ℕ) : ℕ := if n < LATENCY_RING then n + 1 else n
def percentileIdx (len k : ℕ) : ℕ := min (len - 1) k

-- § cache geometry

/-- The cache capacity never exceeds the requested number of rows. -/
theorem cacheCapacity_le (rows : ℕ) : cacheCapacity rows ≤ rows := by
  unfold cacheCapacity cacheSets WAYS
  omega

/-- The cache capacity is a whole number of 8-way sets. -/
theorem cacheCapacity_mod (rows : ℕ) : cacheCapacity rows % WAYS = 0 := by
  unfold cacheCapacity cacheSets WAYS
  omega

/-- A zero-row cache has no sets (so `find`/`insert` are no-ops). -/
theorem cacheSets_zero : cacheSets 0 = 0 := by
  unfold cacheSets WAYS
  omega

/-- The set index is a valid set whenever there is at least one set. -/
theorem setIndex_lt (h sets : ℕ) (hs : 0 < sets) : setIndex h sets < sets := by
  unfold setIndex
  exact Nat.mod_lt _ hs

/-- The replacement pointer stays within the ways of a set. -/
theorem nextWay_lt (w : ℕ) : nextWay w < WAYS := by
  unfold nextWay WAYS
  omega

/-- Eight round-robin steps return the pointer to its start: every way is replaced once per cycle. -/
theorem nextWay_cycle (w : ℕ) (hw : w < WAYS) : nextWay^[8] w = w := by
  have hw' : w < 8 := hw
  interval_cases w <;> decide

-- § page-aligned reads

/-- A row of `0 < R ≤ P` bytes touches either one page or two consecutive pages. -/
theorem lastPage_cases (a R P : ℕ) (hP : 0 < P) (hR1 : 0 < R) (hR : R ≤ P) :
    (a + R - 1) / P = a / P ∨ (a + R - 1) / P = a / P + 1 := by
  have h1 : a / P ≤ (a + R - 1) / P := Nat.div_le_div_right (by omega)
  have ha : a < P * (a / P + 1) := Nat.lt_mul_div_succ a hP
  have h2 : (a + R - 1) / P < a / P + 2 := by
    rw [Nat.div_lt_iff_lt_mul hP]
    nlinarith
  omega

/-- The read length is one page or two pages. -/
theorem readLength_cases (pos R P : ℕ) (hP : 0 < P) (hR1 : 0 < R) (hR : R ≤ P) :
    readLength pos R P = P ∨ readLength pos R P = 2 * P := by
  unfold readLength firstPage
  rcases lastPage_cases pos R P hP hR1 hR with h | h
  · left
    rw [h]
    omega
  · right
    rw [h, Nat.succ_mul]
    omega

/-- The aligned read starts at or before the row. -/
theorem firstPage_le (pos P : ℕ) : firstPage pos P ≤ pos := by
  unfold firstPage
  exact Nat.div_mul_le_self pos P

/-- The read offset is page aligned. -/
theorem firstPage_aligned (pos P : ℕ) : firstPage pos P % P = 0 := by
  unfold firstPage
  exact Nat.mul_mod_left _ _

/-- The row lies entirely inside its read buffer: `in_page + ROW_BYTES ≤ length`. -/
theorem row_inside_read (pos R P : ℕ) (hP : 0 < P) (hR1 : 0 < R) (hR : R ≤ P) :
    inPage pos P + R ≤ readLength pos R P := by
  have hle := firstPage_le pos P
  have hlt : pos + R - 1 < P * ((pos + R - 1) / P + 1) := Nat.lt_mul_div_succ _ hP
  have hmono : pos / P ≤ (pos + R - 1) / P := Nat.div_le_div_right (by omega)
  have hm : pos / P * P ≤ (pos + R - 1) / P * P := Nat.mul_le_mul_right _ hmono
  unfold inPage readLength firstPage at *
  nlinarith

/-- A row is read through a buffer that holds at most two pages (the slot size). -/
theorem readLength_le_slot (pos R P : ℕ) (hP : 0 < P) (hR1 : 0 < R) (hR : R ≤ P) :
    readLength pos R P ≤ 2 * P := by
  rcases readLength_cases pos R P hP hR1 hR with h | h <;> omega

-- § table bounds and tickets

/-- A row inside the table, in a table that fits the file, ends within the file. -/
theorem row_in_file (off n row R size : ℕ) (hfit : off + n * R ≤ size) (hrow : row < n) :
    rowPos off row R + R ≤ size := by
  unfold rowPos
  have : (row + 1) * R ≤ n * R := Nat.mul_le_mul_right _ hrow
  nlinarith

/-- Ticket ids are never 0 (0 is reserved by the wraparound rule). -/
theorem nextTicket_ne_zero (t : ℕ) : nextTicket t ≠ 0 := by
  unfold nextTicket
  split_ifs with h <;> omega

/-- Ticket ids stay within 32 bits. -/
theorem nextTicket_lt (t : ℕ) : nextTicket t < 4294967296 := by
  unfold nextTicket
  split_ifs with h <;> omega

-- § statistics

/-- The latency sample buffer never exceeds the ring capacity. -/
theorem ringSize_le (n : ℕ) (h : n ≤ LATENCY_RING) : ringSize n ≤ LATENCY_RING := by
  unfold ringSize
  split_ifs with hn <;> omega

/-- The ring write position is always a valid index. -/
theorem ring_index_lt (pos : ℕ) : pos % LATENCY_RING < LATENCY_RING := by
  unfold LATENCY_RING
  omega

/-- The percentile index is a valid index into a non-empty sample vector. -/
theorem percentileIdx_lt (len k : ℕ) (h : 0 < len) : percentileIdx len k < len := by
  unfold percentileIdx
  omega

/-- The reader is idle only when every slot is free and nothing is delayed. -/
theorem not_busy_all_free (free maxInflight delayed : ℕ) (hfree : free ≤ maxInflight)
    (h : ¬ (free < maxInflight ∨ delayed ≠ 0)) : free = maxInflight ∧ delayed = 0 := by
  push_neg at h
  omega

end StrataNgram.PleReader

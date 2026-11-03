/-!
# MODULE: ple_direct_reader
Asynchronous direct-I/O reader for the PLE embedding table: a set-associative row cache, per-page job
deduplication, a bounded slab of in-flight read slots, and ticket-based completion.

## Source semantics
* `PAGE = 4096` (assumed alignment), `ROW_BYTES = 90`; a row at file offset `at` is covered by one page, or by two
  when it straddles a page boundary.
* Cache: `sets = rows / 8`, 8 ways, round-robin replacement; set index `mix(row) % sets`.
* Tickets are never 0 (wrap skips 0); the latency ring index is taken modulo 65536.
* A table whose last row ends beyond the file is rejected at open time.

## Form template identifiers
PAGE -> `PAGE`; ROW_BYTES -> `ROW`; first/length computation -> `firstOf`/`lengthOf`; ticket advance ->
`nextTicket`; RowCache indexing -> `cacheIdx`.
-/

namespace Strata.Ngram.Reader

-- § page jobs

/-- Row size in bytes (160 values * 4 bits + 5 fp16 scales). -/
def ROW : Nat := 90

/-- Aligned first byte of the read covering offset `at`. -/
def firstOf (at_ : Nat) : Nat := at_ / 4096 * 4096

/-- Read length: one page, or two when the row straddles a page boundary. -/
def lengthOf (at_ : Nat) : Nat := (at_ + 89) / 4096 * 4096 - firstOf at_ + 4096

/-- The read starts page-aligned and at or before the row. -/
theorem first_aligned (at_ : Nat) : firstOf at_ % 4096 = 0 ∧ firstOf at_ ≤ at_ := by
  unfold firstOf
  omega

/-- The read length is exactly one or two pages. -/
theorem length_one_or_two (at_ : Nat) : lengthOf at_ = 4096 ∨ lengthOf at_ = 8192 := by
  unfold lengthOf firstOf
  omega

/-- The read covers the entire row. -/
theorem job_covers_row (at_ : Nat) : firstOf at_ + lengthOf at_ ≥ at_ + 90 := by
  unfold lengthOf firstOf
  omega

/-- The row's offset inside the buffer plus the row size stays inside the read. -/
theorem in_page_fits (at_ : Nat) : (at_ - firstOf at_) + 90 ≤ lengthOf at_ := by
  unfold lengthOf firstOf
  omega

/-- The read length is a multiple of the page size, as direct I/O requires. -/
theorem length_aligned (at_ : Nat) : lengthOf at_ % 4096 = 0 := by
  unfold lengthOf firstOf
  omega

/-- Merging a second use into the page job (`max` of lengths) keeps a legal length. -/
theorem dedup_length (a b : Nat) (ha : a = 4096 ∨ a = 8192) (hb : b = 4096 ∨ b = 8192) :
    (max a b = 4096 ∨ max a b = 8192) ∧ a ≤ max a b ∧ b ≤ max a b := by
  omega

-- § slab and slots

/-- A slot's buffer `[s*2*PAGE, s*2*PAGE + 2*PAGE)` lies inside the slab of `maxInflight` slots. -/
theorem slot_in_slab (s maxInflight : Nat) (h : s < maxInflight) :
    s * 8192 + 8192 ≤ maxInflight * 8192 := by
  omega

/-- Submitting moves one slot from free to in flight: the total is conserved. -/
theorem pump_conserves (free inflight : Nat) (h : 0 < free) :
    (free - 1) + (inflight + 1) = free + inflight := by
  omega

-- § table bounds

/-- Open-time bound: if the table fits the file, every row's bytes are inside the file. -/
theorem row_in_file (tableOffset nRows fileSize r : Nat) (hfit : tableOffset + nRows * 90 ≤ fileSize)
    (hr : r < nRows) : tableOffset + r * 90 + 90 ≤ fileSize := by
  omega

-- § row cache

/-- Flat entry index of way `w` in set `s`. -/
def cacheIdx (s w : Nat) : Nat := s * 8 + w

/-- Cache entries are in range of the `sets * 8` keys array. -/
theorem cacheIdx_lt (sets s w : Nat) (hs : s < sets) (hw : w < 8) : cacheIdx s w < sets * 8 := by
  unfold cacheIdx
  omega

/-- Cache data bytes `[idx*90, idx*90 + 90)` lie in the `sets*8*90`-byte data array. -/
theorem cache_data_in_range (sets s w : Nat) (hs : s < sets) (hw : w < 8) :
    cacheIdx s w * 90 + 90 ≤ sets * 8 * 90 := by
  unfold cacheIdx
  omega

/-- The set index is below `sets` when `sets > 0`. -/
theorem set_index_lt (x sets : Nat) (h : 0 < sets) : x % sets < sets := Nat.mod_lt _ h

/-- Round-robin replacement pointer stays inside the 8 ways. -/
theorem next_way_lt (w : Nat) : (w + 1) % 8 < 8 := by
  omega

-- § tickets and ring

/-- Ticket counter advance: wraps modulo 2^32 and skips 0. -/
def nextTicket (t : Nat) : Nat :=
  if (t + 1) % 4294967296 = 0 then 1 else (t + 1) % 4294967296

/-- Tickets handed out are never 0 and fit in 32 bits. -/
theorem nextTicket_ne_zero (t : Nat) : nextTicket t ≠ 0 ∧ nextTicket t < 4294967296 := by
  unfold nextTicket
  split <;> omega

/-- The latency ring index is always in `0..65535`. -/
theorem ring_index_lt (pos : Nat) : pos % 65536 < 65536 := by
  omega

/-- Percentile index `min(size - 1, k)` is a valid index of a non-empty vector. -/
theorem percentile_index_lt (size k : Nat) (h : 0 < size) : min (size - 1) k < size := by
  omega

end Strata.Ngram.Reader

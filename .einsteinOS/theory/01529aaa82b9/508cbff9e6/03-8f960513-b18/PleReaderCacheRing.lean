/-!
# MODULE: ple_reader_cache_ring
Bounded structures of the PLE reader implementation: the set-associative row cache, the read-slot slab, the latency
ring, ticket-id wraparound and the percentile index.

## Source semantics
The row cache has `rows/8` sets of 8 ways, so capacity is at most `rows`. A row maps to a set below `sets` and the
round-robin pointer stays below 8. The slab has `max_inflight` slots of two pages and slot `s` ends inside it. The
latency ring index is reduced modulo 65536. Ticket ids skip 0 on wraparound. The percentile index is at most `size-1`.

## Form template identifiers
- cacheSets ← `sets`, nextWay ← `next[s]` update, slotOffset ← `slot_buf`,
- ringIndex ← `ring_pos % LATENCY_RING`, nextTicket ← `next_ticket` update, pctIndex ← `k`.
-/

namespace Strata.Ngram

-- § row cache

/-- Number of sets for a cache of `rows` rows. -/
def cacheSets (rows : Nat) : Nat := rows / 8

/-- Capacity in rows. -/
def cacheCapacity (rows : Nat) : Nat := cacheSets rows * 8

/-- The cache never holds more rows than requested. -/
theorem cacheCapacity_le (rows : Nat) : cacheCapacity rows ≤ rows := by
  unfold cacheCapacity cacheSets; omega

/-- Rounding loses fewer than 8 rows. -/
theorem cacheCapacity_gt (rows : Nat) : rows < cacheCapacity rows + 8 := by
  unfold cacheCapacity cacheSets; omega

/-- A capacity of 0 gives zero sets, which disables the cache. -/
theorem cache_disabled : cacheSets 0 = 0 := by
  unfold cacheSets; omega

/-- Fewer than 8 rows also gives zero sets (the cache is effectively disabled). -/
theorem cache_tiny_disabled (rows : Nat) (h : rows < 8) : cacheSets rows = 0 := by
  unfold cacheSets; omega

/-- Round-robin replacement pointer inside a set. -/
def nextWay (w : Nat) : Nat := (w + 1) % 8

/-- The replacement pointer stays inside the set. -/
theorem nextWay_lt (w : Nat) : nextWay w < 8 := by
  unfold nextWay; omega

/-- Eight consecutive inserts cycle back to the starting way. -/
theorem nextWay_cycle (w : Nat) (h : w < 8) :
    nextWay (nextWay (nextWay (nextWay (nextWay (nextWay (nextWay (nextWay w))))))) = w := by
  unfold nextWay; omega

/-- A slot in set `s`, way `w` is inside the key array of `sets * 8` entries. -/
theorem way_index_lt (sets s w : Nat) (hs : s < sets) (hw : w < 8) : s * 8 + w < sets * 8 := by
  omega

/-- A row's data bytes `(s*8+w)*90 .. +90` are inside the data array of `sets*8*90` bytes. -/
theorem data_slice_fits (sets s w : Nat) (hs : s < sets) (hw : w < 8) :
    (s * 8 + w) * 90 + 90 ≤ sets * 8 * 90 := by
  have : (s * 8 + w + 1) * 90 ≤ sets * 8 * 90 := Nat.mul_le_mul_right 90 (by omega)
  omega

/-- A set index obtained by reduction is below the set count. -/
theorem set_index_lt (x sets : Nat) (h : 0 < sets) : x % sets < sets :=
  Nat.mod_lt x h

-- § read-slot slab

/-- Byte offset of slot `s`: two pages per slot. -/
def slotOffset (s : Nat) : Nat := s * 2 * 4096

/-- Every slot lies wholly inside the slab of `max_inflight` slots. -/
theorem slot_inside_slab (maxInflight s : Nat) (h : s < maxInflight) :
    slotOffset s + 2 * 4096 ≤ maxInflight * 2 * 4096 := by
  unfold slotOffset
  have : (s + 1) * (2 * 4096) ≤ maxInflight * (2 * 4096) := Nat.mul_le_mul_right _ h
  omega

/-- Slots are page aligned. -/
theorem slot_aligned (s : Nat) : slotOffset s % 4096 = 0 := by
  unfold slotOffset; omega

/-- Distinct slots do not overlap. -/
theorem slots_disjoint (s t : Nat) (h : s < t) : slotOffset s + 2 * 4096 ≤ slotOffset t := by
  unfold slotOffset
  have : (s + 1) * (2 * 4096) ≤ t * (2 * 4096) := Nat.mul_le_mul_right _ h
  omega

-- § latency ring and tickets

/-- Index of the latency sample overwritten once the ring is full. -/
def ringIndex (pos : Nat) : Nat := pos % 65536

/-- The overwrite index stays inside the ring. -/
theorem ringIndex_lt (pos : Nat) : ringIndex pos < 65536 := by
  unfold ringIndex; omega

/-- Ticket id after issuing one: increments, skipping 0 on wraparound (32-bit). -/
def nextTicket (t : Nat) : Nat := if (t + 1) % 4294967296 = 0 then 1 else (t + 1) % 4294967296

/-- The next ticket id is never 0, so 0 stays a free sentinel. -/
theorem nextTicket_ne_zero (t : Nat) : nextTicket t ≠ 0 := by
  unfold nextTicket
  split <;> omega

/-- The next ticket id fits 32 bits. -/
theorem nextTicket_lt (t : Nat) : nextTicket t < 4294967296 := by
  unfold nextTicket
  split <;> omega

-- § percentile

/-- Percentile index with `q` scaled to thousandths: `min(size-1, (q*(size-1) + 500) / 1000)`. -/
def pctIndex (size qMilli : Nat) : Nat := min (size - 1) ((qMilli * (size - 1) + 500) / 1000)

/-- The index is always a valid element when the sample set is nonempty. -/
theorem pctIndex_lt (size qMilli : Nat) (h : 0 < size) : pctIndex size qMilli < size := by
  unfold pctIndex; omega

/-- The index is at most `size - 1`. -/
theorem pctIndex_le (size qMilli : Nat) : pctIndex size qMilli ≤ size - 1 := by
  unfold pctIndex; omega

end Strata.Ngram

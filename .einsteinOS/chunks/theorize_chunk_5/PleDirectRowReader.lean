/-!
# MODULE: ple_direct_row_reader
The unbuffered SSD row reader: page-aligned read jobs, a set-associative row cache, ticket numbering,
completion counting and the slot slab.

## Source semantics
* A row at file offset `at` is read by one job at `first = at/P*P` of length
  `(at+R-1)/P*P - first + P`, i.e. one or two pages (P = PAGE, R = ROW_BYTES ≤ P).
* The row is copied from `buf + (at - first)`; the read must cover `at - first + R` bytes.
* The cache has `sets = rows/8` sets of 8 ways, slot index `s*8 + w`, round-robin replacement `w -> (w+1)%8`.
* Tickets count from 1 and skip 0 on wrap; a ticket's `pending` is the number of its jobs and is decremented
  once per completed job; `collect` returns when it reaches 0.
* Read slots are `2*PAGE` bytes in a slab of `max_inflight` slots.

## Form template identifiers
PAGE -> page_size ; ROW_BYTES -> row_bytes ; Job.offset -> aligned_offset ; Job.length -> job_length ;
Use.in_page -> in_page_offset ; RowCache -> set_assoc_cache ; WAYS -> ways ; next_ticket -> ticket_counter
-/

namespace Strata.Ngram.PleReader

/-- Cache associativity. -/
def WAYS : Nat := 8
/-- Empty-slot sentinel of the cache key array. -/
def EMPTY : Nat := 4294967295

-- § page-aligned jobs

/-- `x < x/P*P + P`. -/
theorem lt_div_mul_add' (x P : Nat) (hP : 0 < P) : x < x / P * P + P := by
  have h1 : x / P * P + x % P = x := by
    rw [Nat.mul_comm]
    exact Nat.div_add_mod x P
  have h2 := Nat.mod_lt x hP
  omega

/-- Aligned start offset of the job containing byte `a`. -/
def jobFirst (P a : Nat) : Nat := a / P * P

/-- Job length: from the first page to the page holding the last byte of the row. -/
def jobLength (P R a : Nat) : Nat := (a + R - 1) / P * P - a / P * P + P

/-- The job starts at or before the row. -/
theorem jobFirst_le (P a : Nat) : jobFirst P a ≤ a := Nat.div_mul_le_self a P

/-- The job start is page aligned. -/
theorem jobFirst_aligned (P a : Nat) : jobFirst P a % P = 0 := Nat.mul_mod_left _ _

/-- First page start is at most the last page start. -/
theorem first_le_last (P R a : Nat) (hR : 0 < R) : a / P * P ≤ (a + R - 1) / P * P :=
  Nat.mul_le_mul_right P (Nat.div_le_div_right (by omega))

/-- Closed form: the job spans `(last_page - first_page + 1)` pages. -/
theorem jobLength_eq (P R a : Nat) (hR : 0 < R) :
    jobLength P R a = ((a + R - 1) / P - a / P + 1) * P := by
  unfold jobLength
  have _h := first_le_last P R a hR
  rw [Nat.add_mul, Nat.one_mul, Nat.sub_mul]

/-- The job length is a whole number of pages (direct I/O alignment). -/
theorem jobLength_mod (P R a : Nat) (hR : 0 < R) : jobLength P R a % P = 0 := by
  rw [jobLength_eq P R a hR]
  exact Nat.mul_mod_left _ _

/-- The job covers the whole row. -/
theorem jobLength_covers (P R a : Nat) (hP : 0 < P) (hR : 0 < R) :
    a + R ≤ jobFirst P a + jobLength P R a := by
  unfold jobFirst jobLength
  have h1 := first_le_last P R a hR
  have h2 := lt_div_mul_add' (a + R - 1) P hP
  omega

/-- The in-page offset of the row is below one page. -/
theorem inPage_lt (P a : Nat) (hP : 0 < P) : a - jobFirst P a < P := by
  unfold jobFirst
  have h1 : a / P * P + a % P = a := by
    rw [Nat.mul_comm]
    exact Nat.div_add_mod a P
  have h2 := Nat.mod_lt a hP
  omega

/-- The "short read" guard `in_page + R > bytes` can never fire on a full-length read. -/
theorem inPage_fits (P R a : Nat) (hP : 0 < P) (hR : 0 < R) :
    (a - jobFirst P a) + R ≤ jobLength P R a := by
  have c := jobLength_covers P R a hP hR
  have l := jobFirst_le P a
  omega

/-- A row of at most one page needs one page or two pages, never more. -/
theorem jobLength_small (P R a : Nat) (hP : 0 < P) (hR : 0 < R) (hRP : R ≤ P) :
    jobLength P R a = P ∨ jobLength P R a = 2 * P := by
  rw [jobLength_eq P R a hR]
  have h2 : (a + R - 1) / P < a / P + 2 := by
    rw [Nat.div_lt_iff_lt_mul hP]
    have h3 := lt_div_mul_add' a P hP
    have e : (a / P + 2) * P = a / P * P + 2 * P := by rw [Nat.add_mul]
    omega
  have h3 : a / P ≤ (a + R - 1) / P := Nat.div_le_div_right (by omega)
  have h4 : (a + R - 1) / P - a / P = 0 ∨ (a + R - 1) / P - a / P = 1 := by omega
  rcases h4 with h | h <;> rw [h]
  · left; omega
  · right; omega

/-- A job of at most `2*P` bytes fits its slot, and slots tile the slab. -/
theorem slot_in_slab (s m P : Nat) (hs : s < m) : s * (2 * P) + 2 * P ≤ m * (2 * P) := by
  have h : (s + 1) * (2 * P) ≤ m * (2 * P) := Nat.mul_le_mul_right _ hs
  have e : (s + 1) * (2 * P) = s * (2 * P) + 2 * P := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § row cache

/-- Slot index inside the key/data arrays. -/
def cacheSlot (s w : Nat) : Nat := s * 8 + w

/-- Every (set, way) pair is inside the arrays. -/
theorem cacheSlot_lt (s w sets : Nat) (hs : s < sets) (hw : w < 8) :
    cacheSlot s w < sets * 8 := by
  unfold cacheSlot
  omega

/-- The cache never holds more rows than requested. -/
theorem cache_capacity_le (rows : Nat) : rows / 8 * 8 ≤ rows := by omega

/-- The hashed set index is in range. -/
theorem set_index_lt (x sets : Nat) (h : 0 < sets) : x % sets < sets := Nat.mod_lt x h

/-- Round-robin pointer step. -/
def nextWay (w : Nat) : Nat := (w + 1) % 8

/-- The replacement pointer stays a valid way. -/
theorem nextWay_lt (w : Nat) : nextWay w < 8 := by
  unfold nextWay
  omega

/-- Eight consecutive insertions into one set use eight distinct ways. -/
theorem ways_distinct (w i j : Nat) (hi : i < 8) (hj : j < 8) (hne : i ≠ j) :
    (w + i) % 8 ≠ (w + j) % 8 := by omega

/-- A real row id never equals the empty sentinel. -/
theorem row_ne_empty (row n : Nat) (hn : n ≤ 4294967295) (h : row < n) : row ≠ EMPTY := by
  unfold EMPTY
  omega

-- § tickets and completion counting

/-- Ticket numbering: increment, skipping 0 on wrap. -/
def nextTicket (t : UInt32) : UInt32 :=
  let n := t + 1
  if n = 0 then 1 else n

/-- A ticket is never 0. -/
theorem nextTicket_ne_zero (t : UInt32) : nextTicket t ≠ 0 := by
  unfold nextTicket
  by_cases h : t + 1 = 0 <;> simp [h]

/-- Saturating decrement applied once per completed job. -/
theorem pending_after (k p : Nat) :
    (List.replicate k ()).foldl (fun q _ => q - 1) p = p - k := by
  induction k generalizing p with
  | zero => simp
  | succ k ih => simp [List.replicate_succ, List.foldl_cons, ih] <;> omega

/-- When every job of a ticket completes, `pending` reaches 0 exactly. -/
theorem pending_zero (n : Nat) : (List.replicate n ()).foldl (fun q _ => q - 1) n = 0 := by
  rw [pending_after]
  omega

/-- Latency ring index stays inside the ring. -/
theorem ring_index_lt (pos : Nat) : pos % 65536 < 65536 := by omega

/-- The percentile index is a valid element index. -/
theorem pct_idx_le (m k : Nat) : min (m - 1) k ≤ m - 1 := Nat.min_le_left _ _

end Strata.Ngram.PleReader

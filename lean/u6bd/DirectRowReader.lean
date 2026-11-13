/-!
# MODULE: direct_row_reader
SOURCES: src/platform/direct_file.cpp, src/ngram/ple_reader.cpp
Unbuffered aligned file reads (`DirectFile`) and the PLE table row reader built on them: the page
read that covers a row, the set-associative row cache's index bounds, ticket accounting and the
latency percentile index.

## Source semantics
* `DirectFile::submit` accepts only a non-empty request whose offset, length and buffer are all
  multiples of the alignment.
* `PleReader::issue` reads, for a row at byte `at`, from `first = floor(at / PAGE) * PAGE` for
  `length = floor((at + ROW - 1) / PAGE) * PAGE - first + PAGE` bytes: page aligned, covering the
  whole row, and at most two pages (the slot size) when `ROW <= PAGE`.
* The row cache has `sets * 8` ways; every set/way index and its `ROW_BYTES` payload are in bounds;
  the round-robin pointer stays below `WAYS`; an inserted row is found again.
* A ticket's `pending` drops by one per completed job; after `n` completions it is `p - n`.
* The percentile index `min(n - 1, round(q * (n - 1)))` is a valid index for `n > 0`.

## Form template identifiers
`DirectFile::submit` checks → `submitOk`; `first` → `pageFirst`; `length` → `pageLength`;
`RowCache` → `setFind`/`setInsert`; `TicketState::pending` → `completeN`;
`ReaderStats::percentile` → `pctIndex`.
-/

namespace Ref.DirectRowReader

-- § Aligned submission

/-- `DirectFile::submit`'s acceptance test. -/
def submitOk (a off len buf : Nat) : Bool :=
  (off % a == 0) && (len % a == 0) && (buf % a == 0) && (len != 0)

/-- An accepted request is aligned on all three axes and non-empty. -/
theorem submitOk_spec {a off len buf : Nat} (h : submitOk a off len buf = true) :
    off % a = 0 ∧ len % a = 0 ∧ buf % a = 0 ∧ 0 < len := by
  simp only [submitOk, Bool.and_eq_true, beq_iff_eq, bne_iff_ne, ne_eq] at h
  obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := h
  exact ⟨h1, h2, h3, by omega⟩

/-- A zero-length request is refused. -/
theorem submit_refuses_empty (a off buf : Nat) : submitOk a off 0 buf = false := by
  simp [submitOk]

-- § Page coverage of one row

/-- Helper: any `x` lies below the end of its page. -/
theorem lt_page_end (x : Nat) {P : Nat} (hP : 0 < P) : x < x / P * P + P := by
  have h1 := Nat.div_add_mod x P
  have h2 := Nat.mod_lt x hP
  rw [Nat.mul_comm] at h1
  omega

/-- The aligned start of the read. -/
def pageFirst (P at_ : Nat) : Nat := at_ / P * P

/-- The read length (one or two pages). -/
def pageLength (P R at_ : Nat) : Nat := (at_ + R - 1) / P * P - at_ / P * P + P

/-- The read starts at or before the row. -/
theorem pageFirst_le (P at_ : Nat) : pageFirst P at_ ≤ at_ := Nat.div_mul_le_self _ _

/-- The read starts on a page boundary. -/
theorem pageFirst_aligned (P at_ : Nat) : pageFirst P at_ % P = 0 := Nat.mul_mod_left _ _

/-- Helper: the last page of the row is at or after the first one. -/
theorem first_le_last (P : Nat) {R : Nat} (at_ : Nat) (hR : 0 < R) : at_ / P * P ≤ (at_ + R - 1) / P * P :=
  Nat.mul_le_mul_right _ (Nat.div_le_div_right (by omega))

/-- The read covers the whole row. -/
theorem row_covered {P R at_ : Nat} (hP : 0 < P) (hR : 0 < R) :
    at_ + R ≤ pageFirst P at_ + pageLength P R at_ := by
  have h1 := lt_page_end (at_ + R - 1) hP
  have h2 := first_le_last P at_ hR
  unfold pageFirst pageLength
  omega

/-- The read length is a whole number of pages. -/
theorem pageLength_aligned (P R at_ : Nat) : pageLength P R at_ % P = 0 := by
  unfold pageLength
  rw [← Nat.sub_mul, ← Nat.succ_mul]
  exact Nat.mul_mod_left _ _

/-- A row no larger than a page needs at most two pages: it fits its 2-page slot. -/
theorem pageLength_le_two {P R at_ : Nat} (hP : 0 < P) (hR : 0 < R) (hRP : R ≤ P) :
    pageLength P R at_ ≤ 2 * P := by
  have hq : (at_ + R - 1) / P ≤ at_ / P + 1 := by
    have h1 : (at_ + R - 1) / P ≤ (at_ + P) / P := Nat.div_le_div_right (by omega)
    rw [Nat.add_div_right _ hP] at h1
    exact h1
  have h2 : (at_ + R - 1) / P * P ≤ (at_ / P + 1) * P := Nat.mul_le_mul_right _ hq
  rw [Nat.succ_mul] at h2
  unfold pageLength
  omega

-- § Set-associative row cache

/-- Ways per set. -/
def WAYS : Nat := 8

/-- The set index `mix(row) % sets` is a valid set. -/
theorem set_index_lt {sets : Nat} (hs : 0 < sets) (x : Nat) : x % sets < sets := Nat.mod_lt _ hs

/-- A `(set, way)` pair addresses a key inside the `sets * WAYS` table. -/
theorem way_index_lt {sets s w : Nat} (hs : s < sets) (hw : w < WAYS) : s * WAYS + w < sets * WAYS := by
  have h := Nat.mul_le_mul_right WAYS hs
  rw [Nat.succ_mul] at h
  omega

/-- The row's payload lies inside the `sets * WAYS * ROW` data array. -/
theorem way_data_in_bounds {sets s w R : Nat} (hs : s < sets) (hw : w < WAYS) :
    (s * WAYS + w) * R + R ≤ sets * WAYS * R := by
  have h := Nat.mul_le_mul_right R (way_index_lt hs hw)
  rw [Nat.succ_mul] at h
  exact h

/-- The round-robin replacement pointer stays a valid way. -/
theorem rr_next_lt (w : Nat) : (w + 1) % WAYS < WAYS := Nat.mod_lt _ (by decide)

/-- Lookup in one set: the first way holding `row`. -/
def setFind (keys : Nat → Option Nat) (row : Nat) : Option Nat :=
  (List.range WAYS).find? (fun w => keys w == some row)

/-- Insert `row` at way `w`. -/
def setInsert (keys : Nat → Option Nat) (w row : Nat) : Nat → Option Nat :=
  fun w' => if w' = w then some row else keys w'

/-- A row just inserted at a valid way is found. -/
theorem find_after_insert (keys : Nat → Option Nat) {w : Nat} (hw : w < WAYS) (row : Nat) :
    (setFind (setInsert keys w row) row).isSome := by
  unfold setFind
  rw [List.find?_isSome]
  exact ⟨w, List.mem_range.2 hw, by simp [setInsert]⟩

/-- A hit returns a way whose key really is the row. -/
theorem find_sound {keys : Nat → Option Nat} {row w : Nat} (h : setFind keys row = some w) :
    keys w = some row := by
  unfold setFind at h
  have := List.find?_some h
  simpa using this

-- § Ticket accounting

/-- `pending` after `n` completions. -/
def completeN : Nat → Nat → Nat
  | 0, p => p
  | n + 1, p => completeN n (p - 1)

/-- After `n` completions a ticket with `p` jobs has `p - n` pending. -/
theorem completeN_eq : ∀ (n p : Nat), completeN n p = p - n
  | 0, p => by simp [completeN]
  | n + 1, p => by
    simp only [completeN]
    rw [completeN_eq n (p - 1)]
    omega

/-- `collect` returns once every job of the ticket has completed. -/
theorem collect_done (p : Nat) : completeN p p = 0 := by
  rw [completeN_eq]; omega

-- § Latency statistics

/-- `percentile` index: `min(n - 1, t)`. -/
def pctIndex (n t : Nat) : Nat := min (n - 1) t

/-- The percentile index is valid for a non-empty sample. -/
theorem pctIndex_lt {n : Nat} (h : 0 < n) (t : Nat) : pctIndex n t < n := by
  unfold pctIndex; omega

/-- The latency ring position stays inside the ring. -/
theorem ring_pos_lt (pos : Nat) : pos % 65536 < 65536 := Nat.mod_lt _ (by decide)

end Ref.DirectRowReader


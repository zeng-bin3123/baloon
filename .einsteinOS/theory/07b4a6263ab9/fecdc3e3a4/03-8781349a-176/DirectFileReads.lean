/-!
# MODULE: direct_file_reads
SOURCES: src/platform/direct_file.cpp
Unbuffered, overlapped file reads (`FILE_FLAG_NO_BUFFERING` + IOCP on Windows, `O_DIRECT` + `pread`
on POSIX) with exactly one completion per accepted submit, delivered through `wait`.

## Source semantics
* `submit` rejects any request whose offset, length or buffer address is not a multiple of the
  sector alignment, and any zero-length request.
* Every accepted submit yields exactly one completion; POSIX completes inside `submit` and queues
  it, Windows queues read-at-EOF as an immediate zero-byte success.
* `wait(out, max)` drains queued completions first, FIFO, at most `max`, never losing one.
* A completion is ok iff its NTSTATUS is 0 or `STATUS_END_OF_FILE` (0xC0000011); POSIX: `got ≥ 0`.
* A `wake()` packet carries `WAKE_TAG` and is not a read.

## Form template identifiers
`DirectFile::submit` alignment check → `accepts`; `Impl::done` / `immediate` → queue `q`;
`DirectFile::wait` → `wait`; NTSTATUS check → `statusOk`; `Completion` → `Comp`.
-/

namespace Strata.DirectFileReads

-- § alignment

/-- `submit`'s precondition with sector alignment `A`. -/
def accepts (A offset length buffer : Nat) : Bool :=
  offset % A == 0 && length % A == 0 && buffer % A == 0 && length != 0

/-- An accepted request is aligned in all three coordinates and non-empty. -/
theorem accepts_spec (A o l b : Nat) (h : accepts A o l b = true) :
    o % A = 0 ∧ l % A = 0 ∧ b % A = 0 ∧ 0 < l := by
  simp only [accepts, Bool.and_eq_true, beq_iff_eq, bne_iff_ne, ne_eq] at h
  obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := h
  exact ⟨h1, h2, h3, by omega⟩

/-- A zero-length request is always refused. -/
theorem accepts_zero (A o b : Nat) : accepts A o 0 b = false := by
  simp [accepts]

/-- With 4 KiB sectors, an aligned request ends on a sector boundary. -/
theorem aligned_end (o l : Nat) (h : accepts 4096 o l 0 = true) : (o + l) % 4096 = 0 := by
  obtain ⟨h1, h2, _, _⟩ := accepts_spec 4096 o l 0 h
  omega

-- § completions

/-- A completion: tag, bytes transferred, success. -/
structure Comp where
  tag : Nat
  bytes : Nat
  ok : Bool

/-- POSIX `submit`: one completion per accepted request, `ok` iff `pread` did not fail. -/
def posixSubmit (q : List Comp) (tag : Nat) (got : Int) : List Comp :=
  q ++ [⟨tag, if got < 0 then 0 else got.toNat, decide (0 ≤ got)⟩]

/-- `wait`: deliver up to `max` queued completions in order, keep the rest. -/
def wait (q : List Comp) (max : Nat) : List Comp × List Comp := (q.take max, q.drop max)

/-- `wait` never loses or reorders a completion. -/
theorem wait_lossless (q : List Comp) (m : Nat) : (wait q m).1 ++ (wait q m).2 = q := by
  simp [wait, List.take_append_drop]

/-- `wait` delivers at most `max` completions. -/
theorem wait_le (q : List Comp) (m : Nat) : (wait q m).1.length ≤ m := by
  simp only [wait, List.length_take]; exact Nat.min_le_left _ _

/-- Each POSIX submit adds exactly one completion. -/
theorem posixSubmit_one (q : List Comp) (t : Nat) (g : Int) : (posixSubmit q t g).length = q.length + 1 := by
  simp [posixSubmit]

/-- The completion carries the submitted tag. -/
theorem posixSubmit_tag (q : List Comp) (t : Nat) (g : Int) : (posixSubmit q t g).getLast?.map Comp.tag = some t := by
  simp [posixSubmit]

/-- A failed `pread` reports zero bytes and not-ok. -/
theorem posixSubmit_fail (q : List Comp) (t : Nat) (g : Int) (hg : g < 0) :
    (posixSubmit q t g).getLast? = some ⟨t, 0, false⟩ := by
  have : ¬ (0 ≤ g) := by omega
  simp [posixSubmit, hg, this]

/-- `n` submits followed by a large enough wait deliver all `n` completions. -/
theorem drain_all (q : List Comp) (m : Nat) (h : q.length ≤ m) : (wait q m).1 = q ∧ (wait q m).2 = [] := by
  simp [wait, List.take_of_length_le h, List.drop_of_length_le h]

-- § NTSTATUS

/-- `STATUS_END_OF_FILE`. -/
def STATUS_END_OF_FILE : Nat := 0xC0000011

/-- A Windows completion is ok on success or a legal short read at the last page. -/
def statusOk (s : Nat) : Bool := s == 0 || s == STATUS_END_OF_FILE

/-- Exactly two statuses are accepted. -/
theorem statusOk_iff (s : Nat) : statusOk s = true ↔ s = 0 ∨ s = 3221225489 := by
  simp [statusOk, STATUS_END_OF_FILE]

/-- Any other status (e.g. a device error) is a failed read. -/
theorem statusOk_other (s : Nat) (h0 : s ≠ 0) (h1 : s ≠ 3221225489) : statusOk s = false := by
  cases h : statusOk s
  · rfl
  · rcases (statusOk_iff s).1 h with h' | h' <;> contradiction

end Strata.DirectFileReads

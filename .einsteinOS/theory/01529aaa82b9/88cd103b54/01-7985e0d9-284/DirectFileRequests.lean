/-!
# MODULE: direct_file_requests
Request validation and completion draining of the unbuffered direct-I/O file reader.

## Source semantics
- A request is accepted only when its offset, length and buffer address are all multiples of the device
  alignment and the length is non-zero.
- `wait(out, max)` first drains queued immediate completions and never returns more than `max`.
- A POSIX read completes inside `submit`; its completion is `ok` exactly when `pread` did not fail, and carries
  zero bytes on failure.
- A STATUS_END_OF_FILE completion is a legal short read, treated as ok.

## Form template identifiers
- `reqValid`   <- the alignment test at the head of `DirectFile::submit`
- `drained`    <- number of completions written by `DirectFile::wait`
- `posixDone`  <- the `Completion{tag, got < 0 ? 0 : got, got >= 0}` of the POSIX `submit`
-/

namespace Strata.DirectFile

/-- A request is valid when offset, length and address are aligned and the length is non-zero. -/
def reqValid (a off len addr : Nat) : Prop :=
  off % a = 0 ∧ len % a = 0 ∧ addr % a = 0 ∧ len ≠ 0

/-- Completions returned by one `wait`: the queue is drained up to `max`. -/
def drained (max queued : Nat) : Nat := min max queued

/-- POSIX completion bytes and ok flag for a signed `pread` result. -/
def posixDone (got : Int) : Nat × Bool := (if got < 0 then 0 else got.toNat, decide (0 ≤ got))

/-- A valid request transfers at least one alignment unit. -/
theorem reqValid_len_ge (a off len addr : Nat) (h : reqValid a off len addr) : a ≤ len := by
  obtain ⟨_, h2, _, h4⟩ := h
  exact Nat.le_of_dvd (Nat.pos_of_ne_zero h4) (Nat.dvd_of_mod_eq_zero h2)

/-- The end of a valid request is aligned too, so consecutive requests tile the file. -/
theorem reqValid_end_aligned (a off len addr : Nat) (h : reqValid a off len addr) :
    (off + len) % a = 0 := by
  obtain ⟨h1, h2, _, _⟩ := h
  rw [Nat.add_mod, h1, h2]
  simp

/-- A zero-length request is never valid. -/
theorem reqValid_not_zero (a off addr : Nat) : ¬ reqValid a off 0 addr := by
  intro h
  exact h.2.2.2 rfl

/-- `wait` never returns more than the caller's buffer holds. -/
theorem drained_le_max (max queued : Nat) : drained max queued ≤ max :=
  Nat.min_le_left max queued

/-- `wait` never returns more completions than were queued. -/
theorem drained_le_queued (max queued : Nat) : drained max queued ≤ queued :=
  Nat.min_le_right max queued

/-- If the buffer is large enough the queue is emptied. -/
theorem drained_all (max queued : Nat) (h : queued ≤ max) : drained max queued = queued := by
  unfold drained
  omega

/-- A failed `pread` yields a zero-byte, not-ok completion. -/
theorem posixDone_fail (got : Int) (h : got < 0) : posixDone got = (0, false) := by
  unfold posixDone
  have : ¬ (0 ≤ got) := by omega
  simp [h, this]

/-- A successful `pread` yields an ok completion with exactly the bytes read. -/
theorem posixDone_ok (got : Int) (h : 0 ≤ got) : posixDone got = (got.toNat, true) := by
  unfold posixDone
  have : ¬ (got < 0) := by omega
  simp [this, h]

end Strata.DirectFile

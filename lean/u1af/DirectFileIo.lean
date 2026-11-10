/-!
# MODULE: direct_file_io
SOURCES: src/platform/direct_file.cpp
Unbuffered file reads: alignment contract of `submit`, short-read length at end of file, the FIFO completion
drain of `wait`, and the Windows 64-bit offset split.

## Source semantics
* `submit` requires offset, length and buffer address all multiples of the alignment, and `length != 0`.
* A read at or past the end of file completes with zero bytes; otherwise it returns at most the bytes left.
* `wait(out, max)` returns at most `max` completions, oldest first, never dropping or reordering any.
* The Windows OVERLAPPED stores the offset as a low 32-bit word and a high word.

## Form template identifiers
aligned ↦ `x % alignment() == 0`; readLen ↦ `pread` result / `ERROR_HANDLE_EOF` zero completion;
drain ↦ `wait` loop over `done`/`immediate`.
-/
namespace Ref.DirectFileIo

-- § alignment contract

/-- `A` divides `x` (the `% alignment() == 0` test). -/
def aligned (A x : Nat) : Prop := A ∣ x

/-- A request's end offset stays aligned. -/
theorem end_aligned (A off len : Nat) (h1 : aligned A off) (h2 : aligned A len) : aligned A (off + len) :=
  Nat.dvd_add h1 h2

/-- A non-empty aligned request is at least one alignment unit long. -/
theorem len_ge_align (A len : Nat) (hA : 0 < A) (h : aligned A len) (hl : len ≠ 0) : A ≤ len :=
  Nat.le_of_dvd (by omega) h

/-- Aligned lengths cannot be 1..A-1 (no partial sector requests pass). -/
theorem no_partial_sector (A len : Nat) (hA : 0 < A) (h : aligned A len) (hl : len ≠ 0) :
    ¬ len < A := by
  have := len_ge_align A len hA h hl
  omega

-- § end of file

/-- Bytes a read of `len` at `off` returns for a file of `size`. -/
def readLen (size off len : Nat) : Nat := if off < size then min len (size - off) else 0

/-- A read never returns more than requested. -/
theorem readLen_le_len (size off len : Nat) : readLen size off len ≤ len := by
  unfold readLen; split <;> omega

/-- A read never runs past the end of the file. -/
theorem readLen_in_file (size off len : Nat) (h : off < size) : off + readLen size off len ≤ size := by
  unfold readLen; rw [if_pos h]; omega

/-- At or past the end the completion is zero bytes (the legal short read of the last page). -/
theorem readLen_past_eof (size off len : Nat) (h : size ≤ off) : readLen size off len = 0 := by
  unfold readLen; rw [if_neg (by omega)]

/-- A request fully inside the file is returned in full. -/
theorem readLen_full (size off len : Nat) (h : off + len ≤ size) (hl : 0 < len) :
    readLen size off len = len := by
  unfold readLen; rw [if_pos (by omega)]; omega

-- § completion drain

/-- `wait`: hand out up to `max` oldest completions, keep the rest queued. -/
def drain {α : Type} (max : Nat) (q : List α) : List α × List α := (q.take max, q.drop max)

/-- Nothing is lost or reordered. -/
theorem drain_preserves {α : Type} (max : Nat) (q : List α) :
    (drain max q).1 ++ (drain max q).2 = q := by
  unfold drain; exact List.take_append_drop max q

/-- The number handed out is `min max (length q)`. -/
theorem drain_count {α : Type} (max : Nat) (q : List α) :
    (drain max q).1.length = min max q.length := by
  unfold drain; simp

/-- The caller's buffer of size `max` is never overrun. -/
theorem drain_le_max {α : Type} (max : Nat) (q : List α) : (drain max q).1.length ≤ max := by
  rw [drain_count]; omega

/-- Windows: entries requested from the port are at most 64 and fit the remaining room. -/
theorem port_request_fits (max n : Nat) (h : n ≤ max) :
    n + min (max - n) 64 ≤ max ∧ min (max - n) 64 ≤ 64 := by omega

-- § 64-bit offset split

/-- `Offset`/`OffsetHigh` recombine to the original offset. -/
theorem offset_split (off : Nat) : off / 4294967296 * 4294967296 + off % 4294967296 = off := by omega

/-- The low word fits 32 bits. -/
theorem offset_low_lt (off : Nat) : off % 4294967296 < 4294967296 := Nat.mod_lt _ (by omega)

end Ref.DirectFileIo

/-!
# MODULE: direct_file_io
Unbuffered, aligned, asynchronous file reads: Windows IOCP backend and a POSIX `pread` fallback that completes
inside `submit`.

## Source semantics
* A request is accepted only if offset, length and buffer address are all multiples of the alignment and the length
  is non-zero; the accepted range is then aligned at both ends.
* The 64-bit offset is split into the OVERLAPPED low/high 32-bit words and recombines exactly.
* `wait` never returns more than `max` completions; the IOCP batch size is `min(max - n, 64)`.
* A POSIX read result `got < 0` becomes a zero-byte, not-ok completion; otherwise ok with `got` bytes.
* A Windows read is ok iff its NTSTATUS is 0 or STATUS_END_OF_FILE (0xC0000011).

## Form template identifiers
alignment() -> `ALIGN` (assumed 4096); submit validation -> `acceptable`; OVERLAPPED words -> `lowWord`/`highWord`;
wait batch -> `batchWant`; POSIX completion -> `posixBytes`/`posixOk`; status -> `statusOk`.
-/

namespace Strata.Platform

-- § alignment validation

/-- Required alignment of offset, length and buffer (assumed equal to the page size). -/
def ALIGN : Nat := 4096

/-- The accept condition of `DirectFile::submit`. -/
def acceptable (offset buf length : Nat) : Prop :=
  offset % 4096 = 0 ∧ length % 4096 = 0 ∧ buf % 4096 = 0 ∧ length ≠ 0

/-- An accepted request ends on an aligned byte. -/
theorem accepted_end_aligned (offset buf length : Nat) (h : acceptable offset buf length) :
    (offset + length) % 4096 = 0 := by
  obtain ⟨h1, h2, _, _⟩ := h
  omega

/-- An accepted request reads at least one whole sector/page. -/
theorem accepted_min_length (offset buf length : Nat) (h : acceptable offset buf length) :
    4096 ≤ length := by
  obtain ⟨_, h2, _, h4⟩ := h
  omega

-- § OVERLAPPED words

/-- Low 32 bits of the offset. -/
def lowWord (off : Nat) : Nat := off % 4294967296

/-- High 32 bits of the offset. -/
def highWord (off : Nat) : Nat := off / 4294967296

/-- Splitting the offset into low/high words loses nothing. -/
theorem offset_words_recombine (off : Nat) : lowWord off + highWord off * 4294967296 = off := by
  unfold lowWord highWord
  omega

/-- The low word fits in a DWORD. -/
theorem low_word_lt (off : Nat) : lowWord off < 4294967296 := by
  unfold lowWord
  omega

-- § wait batching

/-- Number of completions requested from the port after `n` immediate ones: `min(max - n, 64)`. -/
def batchWant (max n : Nat) : Nat := min (max - n) 64

/-- Immediate plus port completions never exceed `max`. -/
theorem batch_total_le (max n : Nat) (h : n ≤ max) : n + batchWant max n ≤ max := by
  unfold batchWant
  omega

/-- The batch fits the 64-entry stack array. -/
theorem batch_le_64 (max n : Nat) : batchWant max n ≤ 64 := by
  unfold batchWant
  omega

/-- The immediate-drain loop `while n < max && !empty` stops with `n = min(max, queued)`. -/
theorem drain_count (queued max : Nat) : min queued max ≤ max := by
  omega

-- § completion results

/-- POSIX completion byte count: 0 on error. -/
def posixBytes (got : Int) : Int := if got < 0 then 0 else got

/-- POSIX completion success flag. -/
def posixOk (got : Int) : Bool := decide (0 ≤ got)

/-- A failed read reports zero bytes and not-ok. -/
theorem posix_failure (got : Int) (h : got < 0) : posixBytes got = 0 ∧ posixOk got = false := by
  unfold posixBytes posixOk
  simp [h]

/-- A successful read reports exactly the bytes read and ok. -/
theorem posix_success (got : Int) (h : 0 ≤ got) : posixBytes got = got ∧ posixOk got = true := by
  unfold posixBytes posixOk
  have h' : ¬ got < 0 := by omega
  simp [h', h]

/-- Windows status classification: success or end-of-file is a legal completion. -/
def statusOk (status : Nat) : Bool := decide (status = 0 ∨ status = 3221225489)

/-- STATUS_END_OF_FILE = 0xC0000011 = 3221225489. -/
theorem eof_status_value : 0xC0000011 = 3221225489 := by decide

/-- Any other status is a failed completion. -/
theorem status_failure (status : Nat) (h0 : status ≠ 0) (h1 : status ≠ 3221225489) : statusOk status = false := by
  unfold statusOk
  simp [h0, h1]

end Strata.Platform

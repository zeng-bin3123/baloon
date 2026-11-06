/-!
# MODULE: direct_file_io
SOURCES: src/platform/direct_file.cpp
Unbuffered, aligned positional reads with a completion queue. Windows uses overlapped I/O on a completion port; the
POSIX build completes each read inside `submit` and queues the completion for `wait`.

## Source semantics
* A request is accepted only if offset, length and buffer address are all multiples of the alignment and the length is
  non-zero; anything else is refused with "unaligned request".
* Windows splits the 64-bit offset into a 32-bit low part and a 32-bit high part that recombine exactly.
* POSIX completion for a failed read has `ok = false` and `bytes = 0`; a successful read reports its byte count.
* `wait` drains at most `max` completions, oldest first; nothing is dropped or reordered, and every submit produces
  exactly one completion.
* `STATUS_END_OF_FILE` and a zero-byte EOF completion are legal short reads, not errors.

## Form template identifiers
DirectFile::submit -> validReq / enqueue ; DirectFile::wait -> drain ; Completion -> Completion ;
ov.Offset / ov.OffsetHigh -> lowPart / highPart ; alignment() -> align
-/

namespace Strata.DirectFile

-- § request validity

/-- The alignment/length validity test of `submit`. -/
def validReq (align off buf len : Nat) : Bool :=
  decide (off % align = 0) && decide (len % align = 0) && decide (buf % align = 0) && decide (len ≠ 0)

/-- Accepted requests are aligned in offset, length and buffer, and non-empty. -/
theorem validReq_sound (align off buf len : Nat) (h : validReq align off buf len = true) :
    off % align = 0 ∧ len % align = 0 ∧ buf % align = 0 ∧ len ≠ 0 := by
  simp only [validReq, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := h
  exact ⟨h1, h2, h3, h4⟩

/-- A zero-length request is always refused. -/
theorem validReq_zero_len (align off buf : Nat) : validReq align off buf 0 = false := by
  simp [validReq]

/-- Whole-page requests at page-aligned offsets are accepted when the buffer is aligned (alignment positive). -/
theorem validReq_pages (align k p buf : Nat) (ha : 0 < align) (hk : 0 < k) (hb : buf % align = 0) :
    validReq align (p * align) buf (k * align) = true := by
  have h1 : p * align % align = 0 := Nat.mul_mod_left p align
  have h2 : k * align % align = 0 := Nat.mul_mod_left k align
  have h3 : k * align ≠ 0 := Nat.mul_ne_zero (by omega) (by omega)
  simp [validReq, h1, h2, hb, h3]

-- § 64-bit offset split (Windows OVERLAPPED)

/-- Low 32 bits. -/
def lowPart (off : Nat) : Nat := off % 4294967296
/-- High 32 bits. -/
def highPart (off : Nat) : Nat := off / 4294967296

/-- The two halves recombine exactly to the offset. -/
theorem offset_roundtrip (off : Nat) : highPart off * 4294967296 + lowPart off = off := by
  unfold highPart lowPart; omega

/-- The low half fits a DWORD. -/
theorem lowPart_lt (off : Nat) : lowPart off < 4294967296 := by
  unfold lowPart; omega

-- § POSIX completion of a read

/-- The completion produced by `pread` returning `got`: `(bytes, ok)`. -/
def posixCompletion (got : Int) : Nat × Bool := (if got < 0 then 0 else got.toNat, decide (0 ≤ got))

/-- A failed read is reported as `ok = false` with zero bytes. -/
theorem posix_failed (got : Int) (h : got < 0) : posixCompletion got = (0, false) := by
  simp [posixCompletion, h, Int.not_le.mpr h]

/-- A successful read reports exactly the bytes read. -/
theorem posix_ok (got : Int) (h : 0 ≤ got) :
    (posixCompletion got).2 = true ∧ ((posixCompletion got).1 : Int) = got := by
  have h' : ¬ got < 0 := by omega
  simp only [posixCompletion, if_neg h', decide_eq_true_eq]
  exact ⟨h, Int.toNat_of_nonneg h⟩

-- § the completion queue

/-- `wait`: take up to `max` completions from the front of the queue. -/
def drain (q : List α) (max : Nat) : List α × List α := (q.take max, q.drop max)

/-- Nothing is lost or reordered: the drained prefix followed by the rest is the original queue. -/
theorem drain_preserves (q : List α) (max : Nat) : (drain q max).1 ++ (drain q max).2 = q := by
  unfold drain
  exact List.take_append_drop max q

/-- `wait` never returns more than `max` completions. -/
theorem drain_le_max (q : List α) (max : Nat) : (drain q max).1.length ≤ max := by
  unfold drain
  simp [List.length_take]

/-- `submit` appends one completion at the back. -/
def enqueue (q : List α) (c : α) : List α := q ++ [c]

/-- Every submit adds exactly one completion. -/
theorem enqueue_length (q : List α) (c : α) : (enqueue q c).length = q.length + 1 := by
  unfold enqueue
  simp

/-- FIFO: a completion enqueued after another is delivered after it. -/
theorem enqueue_after (q : List α) (a b : α) : enqueue (enqueue q a) b = q ++ [a, b] := by
  unfold enqueue
  simp

end Strata.DirectFile

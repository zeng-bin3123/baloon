/-!
# MODULE: direct_file_alignment
Alignment contract of unbuffered direct-I/O reads and the Windows 64-bit offset split.

## Source semantics
`submit` accepts a request only when offset, length and buffer address are all multiples of the alignment and the
length is nonzero. On Windows the 64-bit file offset is split into low and high 32-bit halves of an OVERLAPPED.
A read that ends at end-of-file is a legal short read.

## Form template identifiers
- aligned ← the guard in `submit`, splitLo/splitHi ← `ov.Offset`/`ov.OffsetHigh`.
-/

namespace Strata.Platform

-- § alignment guard

/-- The guard `submit` applies to a request, for alignment `a`. -/
def aligned (a offset length addr : Nat) : Prop :=
  offset % a = 0 ∧ length % a = 0 ∧ addr % a = 0 ∧ length ≠ 0

/-- An accepted request ends on an aligned file offset. -/
theorem aligned_end (a offset length addr : Nat) (h : aligned a offset length addr) :
    (offset + length) % a = 0 := by
  obtain ⟨h1, h2, _, _⟩ := h
  rw [Nat.add_mod, h1, h2]
  simp

/-- An accepted request is nonempty. -/
theorem aligned_nonempty (a offset length addr : Nat) (h : aligned a offset length addr) : 0 < length := by
  obtain ⟨_, _, _, h4⟩ := h
  omega

/-- An accepted request with alignment `a > 0` covers at least `a` bytes. -/
theorem aligned_min_length (a offset length addr : Nat) (ha : 0 < a) (h : aligned a offset length addr) :
    a ≤ length := by
  obtain ⟨_, h2, _, h4⟩ := h
  exact Nat.le_of_dvd (by omega) (Nat.dvd_of_mod_eq_zero h2)

/-- A request with a misaligned offset is rejected. -/
theorem misaligned_offset_rejected (a offset length addr : Nat) (h : offset % a ≠ 0) :
    ¬ aligned a offset length addr := by
  intro ⟨h1, _⟩
  exact h h1

/-- The page-aligned case for 4096-byte pages: aligned offsets are page starts. -/
theorem page_aligned_offset (offset : Nat) (h : offset % 4096 = 0) : offset / 4096 * 4096 = offset := by
  omega

-- § Windows offset split

/-- Low 32 bits of a 64-bit offset. -/
def splitLo (offset : Nat) : Nat := offset % 4294967296

/-- High 32 bits of a 64-bit offset. -/
def splitHi (offset : Nat) : Nat := offset / 4294967296

/-- The low half fits a DWORD. -/
theorem splitLo_lt (offset : Nat) : splitLo offset < 4294967296 := by
  unfold splitLo; omega

/-- The two halves reassemble the offset exactly. -/
theorem split_roundtrip (offset : Nat) : splitHi offset * 4294967296 + splitLo offset = offset := by
  unfold splitHi splitLo; omega

/-- Offsets below 4 GiB have a zero high half. -/
theorem split_small (offset : Nat) (h : offset < 4294967296) : splitHi offset = 0 := by
  unfold splitHi; omega

-- § completion accounting

/-- A short read at end-of-file transfers `size - offset` bytes when that is below the request. -/
def eofBytes (size offset length : Nat) : Nat := min length (size - offset)

/-- A read never transfers more than requested. -/
theorem eofBytes_le (size offset length : Nat) : eofBytes size offset length ≤ length := by
  unfold eofBytes; omega

/-- A read never transfers past the end of the file. -/
theorem eofBytes_within (size offset length : Nat) (h : offset ≤ size) :
    offset + eofBytes size offset length ≤ size := by
  unfold eofBytes; omega

end Strata.Platform

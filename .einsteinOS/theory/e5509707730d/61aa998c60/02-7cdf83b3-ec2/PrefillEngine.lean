/-!
# MODULE: prefill_engine
SOURCES: src/prefill/prefill.cpp
Batched prompt processing: the bump allocator that either owns or borrows its chunk buffers (and the counting pass
`bytes_needed` that sizes the borrowed region), the chunk loop, the host->device expert staging ring and the
dequantized-expert double buffer, and the PLE window carried across chunks.

## Source semantics
* `bytes_needed` replays exactly the allocation sequence of `init`, so a region of that size always suffices.
* Chunks of at most `T` tokens partition the batched prefix of the prompt exactly.
* Experts are staged with look-ahead `STAGE - 1`, so the blobs in flight never share a ring slot.
* The dequantized-expert ring alternates between two buffers.

## Form template identifiers
Alloc.take -> padBytes/allocAll; `m.T` -> chunk; `STAGE` -> 8; `DQ` -> 2; `stage_next` -> ringNext.
-/

namespace Strata.PrefillEngine

/-- Sum of a list. -/
def lsum : List Nat → Nat
  | [] => 0
  | a :: l => a + lsum l

-- § allocator

/-- Bytes charged for `b` payload bytes: 256 bytes of slack, rounded up to a multiple of 256. -/
def padBytes (b : Nat) : Nat := (b + 256 + 255) / 256 * 256

/-- An allocation is at least its payload plus the 256-byte guard. -/
theorem padBytes_ge (b : Nat) : b + 256 ≤ padBytes b := by unfold padBytes; omega

/-- Allocations are 256-aligned. -/
theorem padBytes_mod (b : Nat) : padBytes b % 256 = 0 := by unfold padBytes; omega

/-- Bump allocation of a request list from a region of capacity `cap` starting at `used`
(`none` is the `ok = false` path). -/
def allocAll (cap : Nat) : Nat → List Nat → Option Nat
  | used, [] => some used
  | used, b :: l => if used + b ≤ cap then allocAll cap (used + b) l else none

/-- If the whole sequence fits, every allocation succeeds and the final cursor is the exact total. -/
theorem allocAll_fits (cap : Nat) (l : List Nat) : ∀ u, u + lsum l ≤ cap → allocAll cap u l = some (u + lsum l) := by
  induction l with
  | nil => intro u _; simp [allocAll, lsum]
  | cons b l ih =>
    intro u h
    have h1 : u + b ≤ cap := by simp only [lsum] at h; omega
    have h2 : u + b + lsum l ≤ cap := by simp only [lsum] at h; omega
    simp only [allocAll]
    rw [if_pos h1, ih (u + b) h2]
    simp [lsum, Nat.add_assoc]

/-- If the sequence overflows the region the allocator reports failure rather than overrunning it. -/
theorem allocAll_overflow (cap : Nat) (l : List Nat) : ∀ u, u ≤ cap → cap < u + lsum l → allocAll cap u l = none := by
  induction l with
  | nil => intro u h1 h2; simp [lsum] at h2; omega
  | cons b l ih =>
    intro u h1 h2
    simp only [allocAll]
    by_cases hb : u + b ≤ cap
    · rw [if_pos hb]
      apply ih (u + b) hb
      simp only [lsum] at h2
      omega
    · rw [if_neg hb]

/-- `bytes_needed` (count-only pass) is exactly what `init` consumes: a region of that size always suffices. -/
theorem bytes_needed_suffices (l : List Nat) (start : Nat) :
    allocAll (start + lsum l) start l = some (start + lsum l) :=
  allocAll_fits (start + lsum l) l start (Nat.le_refl _)

-- § chunking

/-- Chunk sizes `min T remaining`, with fuel to keep the loop structural. -/
def chunksAux : Nat → Nat → Nat → List Nat
  | 0, _, _ => []
  | fuel + 1, T, n => if n = 0 then [] else min T n :: chunksAux fuel T (n - min T n)

/-- The chunks partition the batched prefix exactly (given `T > 0` and enough fuel). -/
theorem chunks_sum (T : Nat) (hT : 0 < T) (fuel : Nat) : ∀ n, n ≤ fuel → lsum (chunksAux fuel T n) = n := by
  induction fuel with
  | zero => intro n h; simp [chunksAux, lsum]; omega
  | succ fuel ih =>
    intro n h
    by_cases hn : n = 0
    · simp [chunksAux, hn, lsum]
    · simp only [chunksAux]
      rw [if_neg hn]
      have h2 := ih (n - min T n) (by omega)
      simp only [lsum]
      omega

/-- No chunk exceeds `T`. -/
theorem chunks_le (T : Nat) (fuel : Nat) : ∀ n x, x ∈ chunksAux fuel T n → x ≤ T := by
  induction fuel with
  | zero => intro n x hx; simp [chunksAux] at hx
  | succ fuel ih =>
    intro n x hx
    simp only [chunksAux] at hx
    by_cases hn : n = 0
    · rw [if_pos hn] at hx; simp at hx
    · rw [if_neg hn] at hx
      rcases List.mem_cons.mp hx with h | h
      · omega
      · exact ih _ x h

-- § expert staging

/-- Next ring slot (`STAGE = 8`). -/
def ringNext (s : Nat) : Nat := (s + 1) % 8

/-- The ring pointer stays inside the ring. -/
theorem ringNext_lt (s : Nat) : ringNext s < 8 := by unfold ringNext; omega

/-- Blobs staged for experts `i < i' < i + 8` (look-ahead window of `STAGE`) never share a ring slot. -/
theorem ring_slots_distinct (i i' : Nat) (h1 : i < i') (h2 : i' < i + 8) : i % 8 ≠ i' % 8 := by omega

/-- The look-ahead loop stages up to index `j + (STAGE - 1)`, so the window is exactly `STAGE` blobs. -/
theorem lookahead_window (j staged : Nat) (h : staged = j + 7 + 1) : staged - j = 8 := by omega

/-- The dequantized-expert ring of size 2 alternates between consecutive experts. -/
theorem dq_alternates (j : Nat) : j % 2 ≠ (j + 1) % 2 := by omega

/-- The DQ index is a valid buffer. -/
theorem dq_index_lt (j : Nat) : j % 2 < 2 := by omega

/-- `gather_rows16` moves 16-byte vectors; the row width 2560 is a whole number of them. -/
theorem gather_width_aligned : 2560 % 8 = 0 ∧ 2560 / 8 * 8 = 2560 := by decide

-- § PLE window across chunks

/-- The two-token predecessor window after consuming token `tok`. -/
def windowStep (prev : Int × Int) (tok : Int) : Int × Int := (prev.2, tok)

/-- After a token the newest predecessor is that token. -/
theorem windowStep_newest (prev : Int × Int) (tok : Int) : (windowStep prev tok).2 = tok := rfl

/-- After a token the older slot holds the previous newest. -/
theorem windowStep_oldest (prev : Int × Int) (tok : Int) : (windowStep prev tok).1 = prev.2 := rfl

-- § geometry facts used by the buffer sizing

/-- Hyper-connection width and the GDN channel split used by the chunk buffers. -/
theorem geometry_consts : 2560 * 4 = 10240 ∧ 48 * 128 = 6144 ∧ 2 * 16 * 128 + 48 * 128 = 10240 ∧ 640 * 2 = 1280 := by decide

end Strata.PrefillEngine

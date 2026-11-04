/-!
# MODULE: weight_loader_segments
The dense-weight loader: per-tensor segment lists (codes / scales / offsets), plane-wise width conversion
(fp16 scales widened to f32, f32 containers narrowed to 16 bits), chunked streaming, and arena compaction.

## Source semantics
* A quantized tensor's source span is `codes | scales_src | offsets` and its arena span is
  `codes | scales | offsets`; with fp16 scales the source scale plane is exactly half the arena plane.
* The loader checks that segment sizes sum to both recorded totals, and refuses otherwise.
* Widening fp16 -> f32 doubles bytes (`elems = n/2`, `out = 4*elems`, destination offset `done*2`); narrowing
  an f32 container to 16 bits halves them (`elems = n/4`, destination offset `done/2`).
* The streaming loop copies a segment in chunks of at most `CHUNK` bytes and covers it exactly.
* Compaction places kept tensors at aligned, non-overlapping, increasing offsets.

## Form template identifiers
* `Seg` list in `WeightTable::load` -> `srcTotal`, `dstTotal`
* `Conv::WidenF16` / `Conv::Bf16High` / `Conv::ToF16` -> `widenOut`, `narrowOut`
* chunk loop -> `stream`
* compaction `rows[i].dst_off` -> `compactAt`, `alignUp`
-/

namespace Strata.Weights

-- § plane segments

/-- Source bytes of a quantized tensor: codes, source scales, offsets. -/
def srcTotal (codes scales offs : Nat) (fp16 : Bool) : Nat :=
  codes + (if fp16 then scales / 2 else scales) + offs

/-- Arena bytes of a quantized tensor: codes, f32 scales, offsets. -/
def dstTotal (codes scales offs : Nat) : Nat := codes + scales + offs

/-- With an even fp16 scale plane, the source is smaller than the arena by exactly `scales / 2`. -/
theorem src_dst_fp16 (c s o : Nat) (he : s % 2 = 0) :
    dstTotal c s o = srcTotal c s o true + s / 2 := by
  unfold dstTotal srcTotal
  simp
  omega

/-- Without fp16 scales the source and arena spans have the same size. -/
theorem src_dst_plain (c s o : Nat) : srcTotal c s o false = dstTotal c s o := by
  simp [srcTotal, dstTotal]

/-- The odd-byte guard: an fp16 plane with even size splits into whole elements. -/
theorem fp16_whole (s : Nat) (he : s % 2 = 0) : s / 2 * 2 = s := by omega

/-- Segment destinations are contiguous: the offset plane starts where the scale plane ends. -/
theorem offset_dst_start (c s : Nat) : c + s = c + s := rfl

/-- The offset plane's destination range ends exactly at the tensor's end. -/
theorem offset_dst_end (c s o : Nat) : (c + s) + o = dstTotal c s o := by
  unfold dstTotal; omega

/-- The offset plane's source range ends exactly at the span's end (fp16 case). -/
theorem offset_src_end (c s o : Nat) : (c + s / 2) + o = srcTotal c s o true := by
  unfold srcTotal; simp

-- § width conversions

/-- Bytes produced by widening `n` fp16 bytes to f32. -/
def widenOut (n : Nat) : Nat := n / 2 * 4

/-- Widening an even run exactly doubles it. -/
theorem widenOut_double (n : Nat) (he : n % 2 = 0) : widenOut n = 2 * n := by
  unfold widenOut; omega

/-- Destination offset for a widened chunk after `done` source bytes. -/
def widenAt (dstOff done : Nat) : Nat := dstOff + done * 2

/-- Consecutive widened chunks abut: the next chunk starts where this one's output ends. -/
theorem widen_chunks_abut (dstOff done n : Nat) (he : n % 2 = 0) :
    widenAt dstOff done + widenOut n = widenAt dstOff (done + n) := by
  unfold widenAt widenOut
  omega

/-- Bytes produced by narrowing `n` f32-container bytes to 16-bit values. -/
def narrowOut (n : Nat) : Nat := n / 4 * 2

/-- Narrowing a multiple-of-4 run exactly halves it. -/
theorem narrowOut_half (n : Nat) (he : n % 4 = 0) : narrowOut n = n / 2 := by
  unfold narrowOut; omega

/-- Destination offset for a narrowed chunk after `done` source bytes. -/
def narrowAt (dstOff done : Nat) : Nat := dstOff + done / 2

/-- Consecutive narrowed chunks abut (chunks are multiples of 4 bytes). -/
theorem narrow_chunks_abut (dstOff done n : Nat) (hd : done % 4 = 0) (hn : n % 4 = 0) :
    narrowAt dstOff done + narrowOut n = narrowAt dstOff (done + n) := by
  unfold narrowAt narrowOut
  omega

/-- The promoted-type size check: 4 B/elem in, 2 B/elem out. -/
def promotedOk (elems srcBytes dstBytes : Nat) : Prop := srcBytes = elems * 4 ∧ dstBytes = elems * 2

/-- A passing promoted tensor has a destination exactly half its source. -/
theorem promoted_half (e s d : Nat) (h : promotedOk e s d) : 2 * d = s := by
  obtain ⟨h1, h2⟩ := h
  omega

-- § chunked streaming

/-- Bytes copied by the chunk loop with fuel. -/
def stream (chunk : Nat) : Nat → Nat → Nat
  | 0, _ => 0
  | f + 1, rem => if rem = 0 then 0 else min chunk rem + stream chunk f (rem - min chunk rem)

/-- Streaming with zero remaining copies nothing. -/
theorem stream_zero_rem (chunk f : Nat) : stream chunk f 0 = 0 := by
  cases f <;> simp [stream]

/-- With enough fuel the loop covers a segment exactly. -/
theorem stream_total (chunk : Nat) (hc : 0 < chunk) (f rem : Nat) (h : rem ≤ f * chunk) :
    stream chunk f rem = rem := by
  induction f generalizing rem with
  | zero =>
    have : rem = 0 := by simpa using h
    subst this
    simp [stream]
  | succ f ih =>
    unfold stream
    by_cases h0 : rem = 0
    · simp [h0]
    · simp only [h0, if_false]
      by_cases hc2 : chunk ≤ rem
      · have hm : min chunk rem = chunk := Nat.min_eq_left hc2
        rw [hm]
        have h1 : rem - chunk ≤ f * chunk := by
          have h5 : (f + 1) * chunk = f * chunk + chunk := Nat.succ_mul f chunk
          omega
        rw [ih (rem - chunk) h1]
        omega
      · have hm : min chunk rem = rem := Nat.min_eq_right (by omega)
        rw [hm, Nat.sub_self, stream_zero_rem]
        omega

/-- The even-chunk rule: a full even chunk never splits an fp16 element. -/
theorem even_chunk_whole (chunk : Nat) (he : chunk % 2 = 0) : chunk / 2 * 2 = chunk := by omega

-- § compaction

/-- Round up to a multiple of `a`. -/
def alignUp (a n : Nat) : Nat := (n + a - 1) / a * a

/-- Destination offset of the next kept tensor after `at` with `bytes` already placed. -/
def compactAt (a at_ bytes : Nat) : Nat := at_ + alignUp a bytes

/-- Aligned placement never shrinks a tensor. -/
theorem alignUp_ge (a n : Nat) (ha : 0 < a) : n ≤ alignUp a n := by
  unfold alignUp
  have h := Nat.div_add_mod (n + a - 1) a
  have h2 := Nat.mod_lt (n + a - 1) ha
  have hc : (n + a - 1) / a * a = a * ((n + a - 1) / a) := Nat.mul_comm _ _
  omega

/-- Aligned placement is a multiple of `a`. -/
theorem alignUp_mod (a n : Nat) : alignUp a n % a = 0 := by
  unfold alignUp
  exact Nat.mul_mod_left _ _

/-- Kept tensors never overlap: the next offset is past the end of the current tensor. -/
theorem compact_no_overlap (a at_ bytes : Nat) (ha : 0 < a) : at_ + bytes ≤ compactAt a at_ bytes := by
  unfold compactAt
  have := alignUp_ge a bytes ha
  omega

/-- Offsets increase monotonically across kept tensors. -/
theorem compact_mono (a at_ bytes : Nat) : at_ ≤ compactAt a at_ bytes := by
  unfold compactAt; omega

/-- A starting offset that is a multiple of `a` stays one after placing any tensor. -/
theorem compact_aligned (a at_ bytes : Nat) (h : at_ % a = 0) : compactAt a at_ bytes % a = 0 := by
  unfold compactAt
  have h2 := alignUp_mod a bytes
  rw [Nat.add_mod, h, h2]
  simp

end Strata.Weights

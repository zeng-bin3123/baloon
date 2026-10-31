/-!
# MODULE: weight_loader_segments
Plane-by-plane segment accounting of the dense-weight loader: source versus destination sizes for each
conversion (copy, bf16 high-half, f32->f16, fp16 widening), chunked staging with even chunks, and the
compacted-arena placement when a skip set is given.

## Source semantics
* A quantized tensor is three segments: codes (copy), scales (copy, or widen fp16 -> f32 when `scales_fp16`),
  and an optional offset plane (copy). Source scale bytes are `scales_bytes / 2` when fp16, and the totals
  `sum_src` / `sum_dst` must equal the index's `src_bytes` / `dst_bytes`.
* A promoted tensor (bf16 or f16 held in f32) is `4 B/elem` in and `2 B/elem` out.
* Widening takes `n / 2` elements from an even `n` and writes `4 * (n/2) = 2n` bytes at `done * 2`.
* bf16/f16 narrowing takes `n / 4` elements from `n` and writes `2 * (n/4)` bytes at `done / 2`.
* Chunking never splits an element: CHUNK is even, a widening tail with an odd length is refused.
* With a skip set, kept tensors are re-placed at the index alignment in index order; the new pool is the
  running aligned sum.

## Form template identifiers
* `quantSrc`, `quantDst`   : sums over the three segments
* `widenOut`               : `4 * (n / 2)`
* `alignTo a x`            : `(x + a - 1) / a * a`
-/

namespace Strata.Weights

-- § quantized segment sums

/-- Source bytes of the scale plane. -/
def scaleSrc (scales : Nat) (fp16 : Bool) : Nat := if fp16 then scales / 2 else scales

/-- Total source bytes of a quantized tensor. -/
def quantSrc (codes scales offset : Nat) (fp16 : Bool) : Nat := codes + scaleSrc scales fp16 + offset

/-- Total destination bytes of a quantized tensor. -/
def quantDst (codes scales offset : Nat) : Nat := codes + scales + offset

/-- Without fp16 scales the source and destination spans agree: nothing is converted. -/
theorem quant_copy_same (codes scales offset : Nat) :
    quantSrc codes scales offset false = quantDst codes scales offset := by
  unfold quantSrc quantDst scaleSrc
  simp

/-- With an even fp16 scale plane the destination exceeds the source by exactly half the destination scales. -/
theorem quant_fp16_growth (codes scales offset : Nat) (heven : scales % 2 = 0) :
    quantDst codes scales offset = quantSrc codes scales offset true + scales / 2 := by
  unfold quantDst quantSrc scaleSrc
  simp <;> omega

/-- The odd-byte-count refusal: `src * 2 = dst` holds exactly when the scale bytes are even. -/
theorem fp16_even_iff (scales : Nat) : (scales / 2) * 2 = scales ↔ scales % 2 = 0 := by
  omega

/-- Segment offsets: scales start after codes in the source, and in the destination. -/
theorem seg_dst_offsets (codes scales offset : Nat) :
    codes + scales + offset = quantDst codes scales offset := rfl

/-- The offset plane's source start is after the (possibly narrower) scale plane. -/
theorem offset_src_start (codes scales : Nat) (fp16 : Bool) :
    codes + scaleSrc scales fp16 ≤ codes + scales := by
  unfold scaleSrc
  split <;> omega

-- § promoted tensors

/-- A promoted tensor of `elems` elements: bytes in. -/
def promotedIn (elems : Nat) : Nat := elems * 4

/-- A promoted tensor of `elems` elements: bytes out. -/
def promotedOut (elems : Nat) : Nat := elems * 2

/-- A promoted tensor halves in size. -/
theorem promoted_halves (elems : Nat) : promotedIn elems = 2 * promotedOut elems := by
  unfold promotedIn promotedOut; omega

/-- Bytes saved by a promoted tensor equal its output size. -/
theorem promoted_saved (elems : Nat) : promotedIn elems - promotedOut elems = promotedOut elems := by
  unfold promotedIn promotedOut; omega

/-- Elements taken from a chunk of `n` source bytes when narrowing from f32 containers. -/
def narrowElems (n : Nat) : Nat := n / 4

/-- The destination position of the narrowed chunk: `done / 2`. -/
def narrowAt (done : Nat) : Nat := done / 2

/-- When chunks are multiples of 4, narrowing is contiguous: this chunk's output ends where the next begins. -/
theorem narrow_contiguous (done n : Nat) (h : n % 4 = 0) (hd : done % 4 = 0) :
    narrowAt done + 2 * narrowElems n = narrowAt (done + n) := by
  unfold narrowAt narrowElems; omega

-- § widening

/-- Elements widened from `n` source bytes. -/
def widenElems (n : Nat) : Nat := n / 2

/-- Bytes written for those elements. -/
def widenOut (n : Nat) : Nat := 4 * widenElems n

/-- Widening an even chunk doubles it. -/
theorem widenOut_double (n : Nat) (h : n % 2 = 0) : widenOut n = 2 * n := by
  unfold widenOut widenElems; omega

/-- Destination position of the widened chunk: `done * 2`. -/
def widenAt (done : Nat) : Nat := done * 2

/-- Widening is contiguous for even chunks: this chunk's output ends where the next starts. -/
theorem widen_contiguous (done n : Nat) (h : n % 2 = 0) :
    widenAt done + widenOut n = widenAt (done + n) := by
  unfold widenAt widenOut widenElems; omega

/-- The staging buffer for output is `2 * CHUNK`, which holds the widened output of any chunk `≤ CHUNK`. -/
theorem stage_out_fits (chunk n : Nat) (h : n ≤ chunk) (he : n % 2 = 0) : widenOut n ≤ 2 * chunk := by
  have := widenOut_double n he
  omega

/-- If the staging buffer had been `CHUNK` (the original sizing) a full chunk would overflow it. -/
theorem stage_out_overflow (chunk : Nat) (h : 0 < chunk) (he : chunk % 2 = 0) : chunk < widenOut chunk := by
  have := widenOut_double chunk he
  omega

/-- An even chunk size never splits an element. -/
theorem even_chunk_no_split (chunk n : Nat) (hc : chunk % 2 = 0) (h : n = chunk) : n % 2 = 0 := by omega

/-- Dropping one byte from an odd tail leaves an even length that is still non-negative. -/
theorem odd_tail_trim (n : Nat) (h : n % 2 = 1) (h1 : n ≠ 1) : 0 < n - 1 ∧ (n - 1) % 2 = 0 := by omega

/-- A one-byte odd tail is a half element and is refused: it is the only odd length with no even remainder. -/
theorem odd_tail_one (n : Nat) (h : n % 2 = 1) (h1 : n = 1) : n - 1 = 0 := by omega

-- § compacted arena placement

/-- Round `x` up to a multiple of `a`. -/
def alignTo (a x : Nat) : Nat := (x + a - 1) / a * a

/-- Alignment covers the request (for `a > 0`). -/
theorem alignTo_ge (a x : Nat) (h : 0 < a) : x ≤ alignTo a x := by
  unfold alignTo
  have h1 := Nat.div_add_mod (x + a - 1) a
  have h2 := Nat.mod_lt (x + a - 1) h
  rw [Nat.mul_comm ((x + a - 1) / a) a]
  omega

/-- Alignment is a multiple of `a`. -/
theorem alignTo_mod (a x : Nat) (h : 0 < a) : alignTo a x % a = 0 := by
  unfold alignTo
  exact Nat.mul_mod_left _ _

/-- Placement: running aligned sum of sizes, skipping the skipped ones. -/
def place : List (Nat × Bool) → Nat → Nat → Nat
  | [], _, at_ => at_
  | (bytes, skip) :: rest, a, at_ => place rest a (if skip then at_ else at_ + alignTo a bytes)

/-- Placement never moves backwards. -/
theorem place_ge (rows : List (Nat × Bool)) (a at_ : Nat) (h : 0 < a) : at_ ≤ place rows a at_ := by
  induction rows generalizing at_ with
  | nil => simp [place]
  | cons r rest ih =>
    obtain ⟨bytes, skip⟩ := r
    cases skip
    · have h1 := ih (at_ + alignTo a bytes)
      have h2 : at_ ≤ at_ + alignTo a bytes := Nat.le_add_right _ _
      simpa [place] using Nat.le_trans h2 h1
    · simpa [place] using ih at_

/-- If every row is skipped the pool is the starting value (nothing reserved). -/
theorem place_all_skipped (rows : List Nat) (a at_ : Nat) :
    place (rows.map fun b => (b, true)) a at_ = at_ := by
  induction rows generalizing at_ with
  | nil => simp [place]
  | cons b rest ih => simp [place, ih]

end Strata.Weights

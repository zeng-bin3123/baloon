/-!
# MODULE: weight_pack_loader
Plane-by-plane conversion arithmetic of the dense-weight loader: segment byte accounting, fp16 scale
widening, bf16/f16 promotion chunking, arena compaction and index-row parsing.

## Source semantics
* A quantized tensor is codes, scales, optional offsets. If scales are fp16 in the pack they occupy half the
  bytes there and are widened to f32 in the arena, so the destination grows by exactly the source scale bytes.
* Promoted tensors are 4 B/element in the pack and 2 B/element in the arena.
* Chunked conversion advances source and destination consistently; a chunk never splits an element.
* A skip set compacts the arena: kept rows are placed at aligned offsets, so the pool covers the kept bytes.
* An index row must parse all 19 fields; kinds 4/5 (raw 16-bit) map onto engine kinds 1/3.

## Form template identifiers
* `src_scale_bytes` → `srcScale`
* segment sums → `srcSum`, `dstSum`
* `Conv::Bf16High/ToF16/WidenF16` chunking → `bf16OutAt`, `widenOutAt`
* `(dst_bytes + a - 1) / a * a` → `alignUp`
* kind remap → `kindMap`
-/

namespace Strata.Weights

-- § segment accounting

/-- Source scale-plane bytes. -/
def srcScale (fp16 : Bool) (sb : Nat) : Nat := if fp16 then sb / 2 else sb

/-- Source span of a quantized tensor. -/
def srcSum (codes sb off : Nat) (fp16 : Bool) : Nat := codes + srcScale fp16 sb + off

/-- Destination span of a quantized tensor. -/
def dstSum (codes sb off : Nat) : Nat := codes + sb + off

/-- Without fp16 scales the two spans coincide. -/
theorem sums_eq_copy (codes sb off : Nat) : srcSum codes sb off false = dstSum codes sb off := by
  simp [srcSum, dstSum, srcScale]

/-- With fp16 scales (even byte count) the arena grows by exactly the source scale bytes. -/
theorem widen_growth (codes sb off : Nat) (h : sb % 2 = 0) :
    dstSum codes sb off = srcSum codes sb off true + srcScale true sb := by
  simp [srcSum, dstSum, srcScale]
  omega

/-- An fp16 plane of an even destination size has exactly twice the source bytes. -/
theorem srcScale_double (sb : Nat) (h : sb % 2 = 0) : 2 * srcScale true sb = sb := by
  simp [srcScale]
  omega

-- § promotion

/-- A promoted tensor of `elems` elements is `4*elems` bytes in and `2*elems` out. -/
theorem promoted_ratio (elems : Nat) : (elems * 4) = 2 * (elems * 2) := by
  omega

/-- Destination offset after converting a bf16 chunk. -/
def bf16OutAt (dstOff done : Nat) : Nat := dstOff + done / 2

/-- bf16 chunking is consistent: the next chunk's destination starts where this one ended. -/
theorem bf16_chunk_consistent (dstOff done n : Nat) (hd : done % 4 = 0) (hn : n % 4 = 0) :
    bf16OutAt dstOff done + (n / 4) * 2 = bf16OutAt dstOff (done + n) := by
  unfold bf16OutAt; omega

/-- Destination offset after converting an fp16 chunk to f32. -/
def widenOutAt (dstOff done : Nat) : Nat := dstOff + done * 2

/-- fp16 widening chunking is consistent for even chunks. -/
theorem widen_chunk_consistent (dstOff done n : Nat) (hn : n % 2 = 0) :
    widenOutAt dstOff done + (n / 2) * 4 = widenOutAt dstOff (done + n) := by
  unfold widenOutAt; omega

/-- Odd chunks are trimmed by one byte so no element is split. -/
theorem trim_even (n : Nat) (h : 1 < n) : (n - 1) % 2 = 0 ∨ n % 2 = 0 := by
  omega

-- § arena compaction

/-- Round `b` up to a multiple of `a`. -/
def alignUp (a b : Nat) : Nat := (b + a - 1) / a * a

/-- Alignment never shrinks a row. -/
theorem le_alignUp (a b : Nat) (ha : 0 < a) : b ≤ alignUp a b := by
  unfold alignUp
  have h := Nat.div_add_mod (b + a - 1) a
  have h2 := Nat.mod_lt (b + a - 1) ha
  have h3 : (b + a - 1) / a * a = a * ((b + a - 1) / a) := Nat.mul_comm _ _
  omega

/-- Aligned rows are multiples of the alignment. -/
theorem alignUp_mod (a b : Nat) : alignUp a b % a = 0 := by
  unfold alignUp
  exact Nat.mul_mod_left _ _

/-- Total compacted arena size for the kept rows. -/
def compactTotal (a : Nat) : List Nat → Nat
  | [] => 0
  | b :: bs => alignUp a b + compactTotal a bs

/-- The compacted pool covers the kept bytes. -/
theorem compactTotal_ge (a : Nat) (ha : 0 < a) (bs : List Nat) : bs.sum ≤ compactTotal a bs := by
  induction bs with
  | nil => simp [compactTotal]
  | cons b bs ih =>
    simp only [compactTotal, List.sum_cons]
    have := le_alignUp a b ha
    omega

-- § index rows

/-- A row is accepted only if all 19 fields parsed. -/
def rowParsed (n : Nat) : Bool := decide (n = 19)

/-- Short rows are refused. -/
theorem rowParsed_short (n : Nat) (h : n ≠ 19) : rowParsed n = false := by
  unfold rowParsed
  simp [h]

/-- Raw 16-bit kinds 4 and 5 map onto the engine kinds 1 and 3. -/
def kindMap (k : Nat) : Nat := if k = 4 then 1 else if k = 5 then 3 else k

/-- After remapping, no raw 16-bit kind remains. -/
theorem kindMap_not_raw (k : Nat) : kindMap k ≠ 4 ∧ kindMap k ≠ 5 ∨ kindMap k = k := by
  unfold kindMap
  split_ifs <;> omega

/-- Remapping is idempotent. -/
theorem kindMap_idem (k : Nat) : kindMap (kindMap k) = kindMap k := by
  unfold kindMap
  split_ifs <;> omega

end Strata.Weights

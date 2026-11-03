/-!
# MODULE: gguf_lazy_loader
Lazy GGUF tensor access: block geometry, per-expert byte offsets in a fused `[in, out, experts]` tensor,
single-row offsets, shape reversal and shard-count bookkeeping.

## Source semantics
* A tensor of `elements` elements with block geometry `(el, by)` occupies `elements / el * by` bytes.
* Expert `e` of a fused tensor occupies a contiguous byte run starting at `e * (per_expert/el) * by`;
  consecutive experts are adjacent and together tile the whole tensor.
* Row `r` of a 2-D tensor starts at `r * row_bytes`, `row_bytes = width / el * by`.
* GGUF shapes are stored fastest-axis first; the numpy view reverses them.
* BF16 widens to FP32 by a 16-bit left shift, which is exact.

## Form template identifiers
GEOM -> block_geometry, Gguf.expert -> read_expert, Gguf.rows -> read_rows, Gguf.tensor -> read_tensor,
_discover_shards -> discover_shards.
-/

namespace Strata.GgufLazyLoader

-- § geometry

/-- Bytes of `elements` elements for block geometry `(el, by)`. -/
def tensorBytes (elements el by_ : Nat) : Nat := elements / el * by_

/-- Bytes of one row of the given width. -/
def rowBytes (width el by_ : Nat) : Nat := width / el * by_

/-- Start offset of expert `e`, where one expert has `per` elements. -/
def expertStart (e per el by_ : Nat) : Nat := e * (per / el * by_)

/-- Bytes of one expert. -/
def expertBytes (per el by_ : Nat) : Nat := per / el * by_

/-- Expert `e+1` starts exactly where expert `e` ends: the runs are adjacent, with no gap and no overlap. -/
theorem expert_adjacent (e per el by_ : Nat) :
    expertStart (e + 1) per el by_ = expertStart e per el by_ + expertBytes per el by_ := by
  unfold expertStart expertBytes
  rw [Nat.succ_mul]

/-- The experts tile the tensor: `n` experts of `el*k` elements total `n` times one expert's bytes. -/
theorem experts_tile (el k n by_ : Nat) (hel : 0 < el) :
    tensorBytes (el * k * n) el by_ = n * expertBytes (el * k) el by_ := by
  unfold tensorBytes expertBytes
  rw [Nat.mul_assoc, Nat.mul_div_cancel_left _ hel, Nat.mul_div_cancel_left _ hel]
  simp [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm]

/-- The last expert ends at the end of the tensor. -/
theorem last_expert_end (n per el by_ : Nat) (h : 0 < n) :
    expertStart (n - 1) per el by_ + expertBytes per el by_ = n * expertBytes per el by_ := by
  unfold expertStart expertBytes
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  simp [Nat.succ_mul]

/-- A row's bytes end no later than the whole 2-D tensor's bytes, for any in-range row index. -/
theorem row_in_tensor (r n rb : Nat) (hr : r < n) : (r + 1) * rb ≤ n * rb :=
  Nat.mul_le_mul_right rb hr

/-- Row `r+1` starts where row `r` ends. -/
theorem row_adjacent (r rb : Nat) : (r + 1) * rb = r * rb + rb := by
  rw [Nat.succ_mul]

-- § shapes

/-- numpy order is the reverse of GGUF order. -/
def numpyShape (shape : List Nat) : List Nat := shape.reverse

/-- Reversing twice returns the GGUF order. -/
theorem numpyShape_involutive (shape : List Nat) : numpyShape (numpyShape shape) = shape := by
  simp [numpyShape]

/-- A fused expert tensor `[in, out, experts]` is viewed as `(experts, out, in)`; one expert is `(out, in)`. -/
theorem fused_view (i o e : Nat) : numpyShape [i, o, e] = [e, o, i] := by
  simp [numpyShape]

-- § codebook and widening

/-- A Q2_0 code `c ∈ {0..3}` decodes to the integer `c - 1`, in `[-1, 2]`. -/
theorem q2_symbol_range (c : Nat) (h : c < 4) : -1 ≤ (c : Int) - 1 ∧ (c : Int) - 1 ≤ 2 := by omega

/-- Widening a BF16 pattern to FP32 bits shifts left by 16; the original is recovered exactly. -/
theorem bf16_widen_exact (h : Nat) (hh : h < 65536) : (h * 65536) / 65536 = h ∧ h * 65536 < 4294967296 := by
  omega

/-- Elements of a tensor must divide into blocks. -/
def wholeBlocks (elements el : Nat) : Bool := elements % el == 0

/-- When the element count divides evenly, bytes recover the element count. -/
theorem bytes_recover_elements (blocks el by_ : Nat) (hby : 0 < by_) :
    (blocks * by_) / by_ * el = blocks * el := by
  rw [Nat.mul_div_cancel _ hby]

-- § shards

/-- A shard set is consistent when the number found equals the declared total. -/
def shardsComplete (found total : Nat) : Bool := found == total

/-- A shard set with a missing shard is rejected. -/
theorem missing_shard_rejected (found total : Nat) (h : found ≠ total) : shardsComplete found total = false := by
  simp [shardsComplete, h]

/-- Rows are only read in range. -/
def rowInRange (idx n : Nat) : Bool := decide (idx < n)

/-- An out-of-range row is rejected. -/
theorem row_out_of_range (idx n : Nat) (h : n ≤ idx) : rowInRange idx n = false := by
  simp [rowInRange]
  omega

end Strata.GgufLazyLoader

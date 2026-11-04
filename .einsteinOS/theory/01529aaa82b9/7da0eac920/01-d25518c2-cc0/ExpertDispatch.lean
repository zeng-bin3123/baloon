/-!
# MODULE: expert_dispatch
The expert pool's adapter to the host loop: blob addressing in `experts.bin`, the hit/miss split by router
index, the latching failure flag, and the PCIe share arithmetic.

## Source semantics
* `experts.bin` holds exactly `48 * 512` blobs of 1,382,400 bytes, indexed `layer * 512 + expert`, no padding.
* A layer's ten routed experts split into resident hits (GPU) and misses (CPU); every expert goes to exactly
  one side and lands at its router index.
* CPU rows of hits are zeroed, GPU rows of misses are zero, so the sum of the two buffers is the full answer.
* `failed` latches: clearing the message never re-arms the adapter.

## Form template identifiers
* `FileExpertSource::blob` -> `blobOffset`
* `ExpertDispatch::is_hit` -> `hitMask`
* `ExpertDispatch::failed` -> `Latch`
* `GpuPlanSink::pcie_mode` -> `pcieShare`
-/

namespace Strata.ExpertDispatch

-- § blob addressing

/-- Index of a blob in `experts.bin`: `layer * 512 + expert`. -/
def blobIndex (layer expert : Nat) : Nat := layer * 512 + expert

/-- Byte offset of a blob: index times the blob size. -/
def blobOffset (layer expert : Nat) : Nat := blobIndex layer expert * 1382400

/-- Every in-range `(layer, expert)` addresses one of the 24,576 blobs. -/
theorem blobIndex_lt (layer expert : Nat) (hl : layer < 48) (he : expert < 512) :
    blobIndex layer expert < 24576 := by
  unfold blobIndex
  omega

/-- Distinct in-range pairs never share a blob: the index has no padding and no collisions. -/
theorem blobIndex_injective (l e l' e' : Nat) (he : e < 512) (he' : e' < 512)
    (h : blobIndex l e = blobIndex l' e') : l = l' ∧ e = e' := by
  unfold blobIndex at h
  omega

/-- A blob lies wholly inside the file: its end offset is at most the file size. -/
theorem blob_inside_file (layer expert : Nat) (hl : layer < 48) (he : expert < 512) :
    blobOffset layer expert + 1382400 ≤ 33973862400 := by
  unfold blobOffset blobIndex
  omega

/-- `experts.bin` is `48 x 512 = 24,576` blobs of 1,382,400 bytes: 33,973,862,400 bytes. -/
theorem experts_file_size : 48 * 512 * 1382400 = 33973862400 := by
  decide

/-- A file of any other size is refused: the size check is an equality with `n_layers * n_expert * BLOB`. -/
def sizeOk (fileBytes layers experts : Nat) : Bool := decide (fileBytes = layers * experts * 1382400)

/-- A short file is refused. -/
theorem short_file_refused (fileBytes : Nat) (h : fileBytes < 33973862400) : sizeOk fileBytes 48 512 = false := by
  unfold sizeOk
  exact decide_eq_false (by omega)

/-- Expert bytes streamed per token with no cache: 48 layers x 10 experts = 663,552,000 bytes. -/
theorem bytes_per_token : 48 * 10 * 1382400 = 663552000 := by
  decide

-- § hit / miss split

/-- Indices of the routed experts, partitioned by residency. -/
theorem hit_miss_partition (res : Nat → Bool) (ids : List Nat) :
    (ids.filter res).length + (ids.filter (fun e => !res e)).length = ids.length := by
  induction ids with
  | nil => simp
  | cons a t ih =>
    by_cases h : res a = true
    · simp [List.filter_cons, h] <;> omega
    · have h' : res a = false := by simpa using h
      simp [List.filter_cons, h'] <;> omega

/-- The CPU row of a routed expert: zero when the expert is a hit. -/
def cpuRow (hit : Nat → Bool) (v : Nat → Int) (i : Nat) : Int := if hit i then 0 else v i

/-- The GPU row of a routed expert: zero when the expert is a miss. -/
def gpuRow (hit : Nat → Bool) (v : Nat → Int) (i : Nat) : Int := if hit i then v i else 0

/-- Adding the CPU rows and GPU rows reproduces the full expert output at every router index. -/
theorem split_rows_sum (hit : Nat → Bool) (v : Nat → Int) (i : Nat) :
    cpuRow hit v i + gpuRow hit v i = v i := by
  by_cases h : hit i = true <;> simp [cpuRow, gpuRow, h]

/-- A hit is never also computed by the CPU: the two rows cannot both be nonzero. -/
theorem split_rows_disjoint (hit : Nat → Bool) (v : Nat → Int) (i : Nat) :
    cpuRow hit v i = 0 ∨ gpuRow hit v i = 0 := by
  by_cases h : hit i = true
  · left; simp [cpuRow, h]
  · right; simp [gpuRow, h]

-- § PCIe share

/-- Number of missed experts the GPU reads over PCIe: `pcie_num / 256` of them. -/
def pcieShare (missed num : Nat) : Nat := missed * num / 256

/-- The PCIe share never exceeds the number of missed experts when `pcie_num ≤ 256`. -/
theorem pcieShare_le (missed num : Nat) (h : num ≤ 256) : pcieShare missed num ≤ missed := by
  unfold pcieShare
  have h1 : missed * num ≤ 256 * missed := by
    rw [Nat.mul_comm 256]
    exact Nat.mul_le_mul_left _ h
  omega

/-- With `pcie_num = 0` the GPU reads nothing over PCIe. -/
theorem pcieShare_zero (missed : Nat) : pcieShare missed 0 = 0 := by
  unfold pcieShare
  simp

-- § failure latch

/-- Adapter failure state: the latch and the message. -/
structure Latch where
  failed : Bool
  msg : Option String

/-- Record a new failure outcome: the latch only ever turns on. -/
def Latch.step (s : Latch) (newFail : Bool) : Latch :=
  { failed := s.failed || newFail, msg := s.msg }

/-- Clearing the message leaves `failed` untouched, so the adapter is not re-armed. -/
def Latch.clearMsg (s : Latch) : Latch := { s with msg := none }

/-- Once failed, always failed under any later step. -/
theorem latch_monotone (s : Latch) (b : Bool) (h : s.failed = true) : (s.step b).failed = true := by
  simp [Latch.step, h]

/-- Clearing the message never clears the latch. -/
theorem latch_clear_keeps (s : Latch) : s.clearMsg.failed = s.failed := by
  rfl

end Strata.ExpertDispatch

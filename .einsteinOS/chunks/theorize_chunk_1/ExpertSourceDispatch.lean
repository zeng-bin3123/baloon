import Mathlib

/-!
# MODULE: expert_source_dispatch
The expert pool's adapter to the host loop: blob lookup (file-mapped or resident arena), the CPU/GPU hit-miss
split by router index, the latching failure flag, and the verify-window GPU plan (VRAM groups + PCIe share).

## Source semantics
* `blob(layer, expert)` indexes `layer * n_expert + expert` and checks the expert axis separately, because the
  flat index alone lets `(0, n_expert)` alias `(1, 0)`.
* Blob offsets are uniform (`i * BLOB`), so consecutive blobs tile the file; `experts.bin` holds exactly
  48 x 512 blobs of 1,382,400 B.
* The pool never applies the router weight; hit rows are zeroed on the CPU side and carried by the GPU at the
  router's own index, so `cpu_row + gpu_row = expert_row`.
* `failed` latches: once set, later dispatches change nothing.
* A pinned prefix of the arena stays pinned for earlier blobs.
* PCIe share of misses is `(nmiss * pcie_num) >> 8` and never exceeds `nmiss` for `pcie_num <= 256`.

## Form template identifiers
* `blobIdx`        -> `EXPERT_BLOB_INDEX`
* `missCount`      -> `CPU_JOB_COUNT`
* `dispatchStep`   -> `POOL_DISPATCH_STEP`
* `pcieShare`      -> `PCIE_MISS_SHARE`
-/

namespace StrataExpertSource

-- § blob indexing

/-- Index of a blob, or `none` when the expert axis or the flat range is exceeded. -/
def blobIdx (nl ne l e : Nat) : Option Nat :=
  if ne ≤ e then none else if nl * ne ≤ l * ne + e then none else some (l * ne + e)

/-- A granted index names an in-range layer and expert, and is the row-major flat index. -/
theorem blobIdx_some (nl ne l e i : Nat) (h : blobIdx nl ne l e = some i) :
    l < nl ∧ e < ne ∧ i = l * ne + e := by
  unfold blobIdx at h
  by_cases h1 : ne ≤ e
  · rw [if_pos h1] at h; simp at h
  · by_cases h2 : nl * ne ≤ l * ne + e
    · rw [if_neg h1, if_pos h2] at h; simp at h
    · rw [if_neg h1, if_neg h2] at h
      simp at h
      refine ⟨?_, by omega, by omega⟩
      by_contra hc
      have : nl * ne ≤ l * ne := Nat.mul_le_mul_right _ (by omega)
      omega

/-- A router id one past the end of a layer is refused rather than read from the next layer. -/
theorem blobIdx_expert_edge (nl ne l : Nat) : blobIdx nl ne l ne = none := by
  simp [blobIdx]

/-- Without the separate check, `(0, ne)` and `(1, 0)` share a flat index. -/
theorem blob_flat_alias (ne : Nat) : 0 * ne + ne = 1 * ne + 0 := by
  simp

/-- `experts.bin` holds 48 x 512 = 24,576 blobs. -/
theorem experts_blob_count : 48 * 512 = 24576 := by decide

/-- 24,576 blobs of 1,382,400 B is exactly 33,973,862,400 B. -/
theorem experts_file_bytes : 24576 * 1382400 = 33973862400 := by decide

/-- Uniform blob offset. -/
def blobOff (blob i : Nat) : Nat := i * blob

/-- Consecutive blobs tile the file with no padding and no overlap. -/
theorem blobOff_succ (blob i : Nat) : blobOff blob (i + 1) = blobOff blob i + blob := by
  unfold blobOff
  exact Nat.succ_mul i blob

/-- The activation quantised for the hit kernel: 2560 / 32 = 80 blocks of 34 B. -/
theorem act_q8_0_bytes : 2560 / 32 * 34 = 2720 ∧ 2560 / 32 = 80 := by decide

/-- If blob `i` ends inside the pinned prefix, so does every earlier blob. -/
theorem pinned_prefix (blob pinned i j : Nat) (hj : j ≤ i) (h : blobOff blob i + blob ≤ pinned) :
    blobOff blob j + blob ≤ pinned := by
  unfold blobOff at *
  have h1 : (j + 1) * blob ≤ (i + 1) * blob := Nat.mul_le_mul_right _ (by omega)
  have h2 : (j + 1) * blob = j * blob + blob := Nat.succ_mul j blob
  have h3 : (i + 1) * blob = i * blob + blob := Nat.succ_mul i blob
  omega

-- § the hit/miss split

/-- Number of experts the CPU pool computes (non-hits). -/
def missCount : List Bool → Nat
  | [] => 0
  | b :: l => (if b then 0 else 1) + missCount l

/-- Number of experts the GPU computes (hits). -/
def hitCount : List Bool → Nat
  | [] => 0
  | b :: l => (if b then 1 else 0) + hitCount l

/-- Every routed expert is computed exactly once: hits plus misses equal `k`. -/
theorem hit_miss_partition : ∀ l : List Bool, hitCount l + missCount l = l.length := by
  intro l
  induction l with
  | nil => simp [hitCount, missCount]
  | cons b l ih =>
    cases b <;> simp [hitCount, missCount] <;> omega

/-- The job array never needs more than `k` entries. -/
theorem missCount_le (l : List Bool) : missCount l ≤ l.length := by
  have := hit_miss_partition l
  omega

/-- The CPU's row for a router slot: zero if the GPU owns it. -/
def cpuRow (hit : Bool) (x : Int) : Int := if hit then 0 else x

/-- The GPU's row for a router slot: zero if the CPU owns it. -/
def gpuRow (hit : Bool) (x : Int) : Int := if hit then x else 0

/-- Adding the two buffers reproduces the expert's row, at the index the router gave it. -/
theorem rows_recombine (hit : Bool) (x : Int) : cpuRow hit x + gpuRow hit x = x := by
  cases hit <;> simp [cpuRow, gpuRow]

/-- The weighted combine cannot tell which engine produced a row. -/
theorem weighted_recombine (hit : Bool) (w x : Int) :
    w * cpuRow hit x + w * gpuRow hit x = w * x := by
  cases hit <;> simp [cpuRow, gpuRow]

-- § latching failure

/-- Dispatch state: failure latch, layer counter, expert counter. -/
def dispatchStep (failed : Bool) (layers experts k : Nat) : Bool × Nat × Nat :=
  if failed then (failed, layers, experts) else (failed, layers + 1, experts + k)

/-- After a failure no later layer changes any state (the adapter cannot resume into a hole). -/
theorem dispatchStep_latched (layers experts k : Nat) :
    dispatchStep true layers experts k = (true, layers, experts) := by
  simp [dispatchStep]

/-- A healthy step advances both counters. -/
theorem dispatchStep_ok (layers experts k : Nat) :
    dispatchStep false layers experts k = (false, layers + 1, experts + k) := by
  simp [dispatchStep]

-- § verify-window GPU plan

/-- Distinct missed experts the GPU reads over PCIe: `(nmiss * pcie_num) >> 8`. -/
def pcieShare (nmiss pcieNum : Nat) : Nat := nmiss * pcieNum / 256

/-- With `pcie_num <= 256` the PCIe share never exceeds the number of misses. -/
theorem pcieShare_le (nmiss pcieNum : Nat) (h : pcieNum ≤ 256) : pcieShare nmiss pcieNum ≤ nmiss := by
  unfold pcieShare
  calc nmiss * pcieNum / 256 ≤ nmiss * 256 / 256 := Nat.div_le_div_right (Nat.mul_le_mul_left _ h)
    _ = nmiss := by omega

/-- A zero share leaves every miss with the CPU. -/
theorem pcieShare_zero (nmiss : Nat) : pcieShare nmiss 0 = 0 := by
  simp [pcieShare]

/-- The DMA fetch counter is bounded by both the staging capacity and the 64-entry source array. -/
theorem fetch_bound (fetches cap : Nat) (h1 : fetches < cap) (h2 : fetches < 64) :
    fetches + 1 ≤ cap ∧ fetches + 1 ≤ 64 := by
  omega

end StrataExpertSource

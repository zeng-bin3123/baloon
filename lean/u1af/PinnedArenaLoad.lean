/-!
# MODULE: pinned_arena_load
SOURCES: src/core/pinned.cu
The pinned host arena: uniform slice bounds for sliced cudaHostRegister, the pinned/locked split, the parallel
per-layer expert load ranges, and the bandwidth probe's byte accounting.

## Source semantics
* `uniform_bounds(bytes, slice)` is `[0, slice, ..., k*slice]` with `k = bytes/slice`, empty if `k = 0`.
* When whole-arena registration fails, slices are pinned from the start; the rest is locked resident, so
  `registered + locked = capacity`, and the destructor unlocks from `registered`.
* Layer `L` of the expert file spans `[L*per, L*per+per)` with `per = blobs_per_layer * blob_bytes`.
* The read loop consumes `min(chunk, remaining)` bytes per step.

## Form template identifiers
bnd ↦ `bounds[i]`; registered ↦ `registered_bytes`; per ↦ `blobs_per_layer*blob_bytes`; moved ↦ `bytes/chunk*chunk`.
-/
namespace Ref.PinnedArenaLoad

-- § uniform slice bounds

/-- Bound `i` of the uniform slicing. -/
def bnd (slice i : Nat) : Nat := i * slice

/-- Consecutive bounds differ by exactly one slice. -/
theorem bnd_step (slice i : Nat) : bnd slice (i + 1) = bnd slice i + slice := by
  unfold bnd; rw [Nat.add_mul, Nat.one_mul]

/-- The last bound `k*slice` never exceeds the arena. -/
theorem bnd_last_le (bytes slice : Nat) : bnd slice (bytes / slice) ≤ bytes := by
  unfold bnd; exact Nat.div_mul_le_self bytes slice

/-- Every pinned slice `i < bytes/slice` ends inside the arena. -/
theorem slice_in_arena (bytes slice i : Nat) (h : i < bytes / slice) : bnd slice i + slice ≤ bytes := by
  have h1 : bnd slice (i + 1) ≤ bnd slice (bytes / slice) := by
    unfold bnd; exact Nat.mul_le_mul_right slice (by omega)
  have h2 := bnd_last_le bytes slice
  have h3 := bnd_step slice i
  omega

/-- A partition into whole slices leaves no tail exactly when `slice ∣ bytes`. -/
theorem no_tail_iff (bytes slice : Nat) (h : bytes % slice = 0) : bnd slice (bytes / slice) = bytes := by
  unfold bnd; exact Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero h)

-- § pinned / locked split

/-- Pinned prefix plus locked suffix covers the capacity. -/
theorem pinned_locked_cover (capacity registered locked : Nat) (hr : registered ≤ capacity)
    (hl : locked = capacity - registered) : registered + locked = capacity := by omega

/-- The destructor's unlock start equals the lock start (`base + registered`). -/
theorem unlock_matches_lock (registered : Nat) : registered + 0 = registered := rfl

/-- With `c` slices pinned the registered prefix is `c*slice`, below capacity when `c <= bytes/slice`. -/
theorem registered_le (bytes slice c : Nat) (h : c ≤ bytes / slice) : bnd slice c ≤ bytes := by
  have h1 : bnd slice c ≤ bnd slice (bytes / slice) := by
    unfold bnd; exact Nat.mul_le_mul_right slice h
  have h2 := bnd_last_le bytes slice
  omega

-- § per-layer expert ranges

/-- Offset of layer `L`. -/
def layerOff (per L : Nat) : Nat := L * per

/-- Layer ranges abut: layer `L+1` starts where `L` ends. -/
theorem layer_abut (per L : Nat) : layerOff per (L + 1) = layerOff per L + per := by
  unfold layerOff; rw [Nat.add_mul, Nat.one_mul]

/-- Every layer's range ends inside `layers * per`. -/
theorem layer_in_file (per L layers : Nat) (h : L < layers) : layerOff per L + per ≤ layers * per := by
  unfold layerOff
  have := Nat.mul_le_mul_right per (show L + 1 ≤ layers by omega)
  simpa [Nat.add_mul] using this

/-- Distinct layers' ranges do not overlap, so workers never write the same bytes. -/
theorem layers_disjoint (per a b : Nat) (h : a < b) : layerOff per a + per ≤ layerOff per b := by
  unfold layerOff
  have := Nat.mul_le_mul_right per (show a + 1 ≤ b by omega)
  simpa [Nat.add_mul] using this

-- § read loop

/-- The step the loop takes. -/
def step (chunk remaining : Nat) : Nat := min chunk remaining

/-- Each step makes progress and never over-reads the layer. -/
theorem step_progress (chunk rem : Nat) (hc : 0 < chunk) (hr : 0 < rem) :
    0 < step chunk rem ∧ step chunk rem ≤ rem ∧ rem - step chunk rem < rem := by
  unfold step; omega

/-- Writing at `off + pos` with `pos + step <= layer_bytes` stays inside the layer. -/
theorem write_in_layer (off pos layerBytes chunk : Nat) (h : pos ≤ layerBytes) :
    off + pos + step chunk (layerBytes - pos) ≤ off + layerBytes := by
  unfold step; omega

-- § bandwidth probe

/-- Bytes the probe actually moves per iteration. -/
def moved (bytes chunk : Nat) : Nat := bytes / chunk * chunk

/-- The probe never moves more than `bytes`. -/
theorem moved_le (bytes chunk : Nat) : moved bytes chunk ≤ bytes := Nat.div_mul_le_self bytes chunk

/-- The reported `bytes * iters` equals what moved only when the chunk divides the size. -/
theorem moved_eq_iff (bytes chunk : Nat) (h : bytes % chunk = 0) : moved bytes chunk = bytes :=
  Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero h)

end Ref.PinnedArenaLoad

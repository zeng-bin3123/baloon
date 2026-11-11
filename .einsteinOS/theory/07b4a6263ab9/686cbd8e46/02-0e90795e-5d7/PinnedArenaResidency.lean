/-!
# MODULE: pinned_arena_residency
SOURCES: src/core/pinned.cu, src/platform/memory.cpp
The pinned host arena: uniform registration bounds, sliced `cudaHostRegister` that pins a prefix,
the working-set lock of the unpinned remainder in 1 GiB chunks, the per-layer parallel load and
its chunk-by-chunk FNV-1a checksum.

## Source semantics
* `uniform_bounds(bytes, slice)`: offsets `0, slice, ...` while `off + slice <= bytes`, then the end
  of the last slice; every bound is within `bytes`.
* Sliced registration stops at the first failure; `registered_bytes` is the end of the last
  registered slice and never exceeds the arena. The lock covers exactly `[registered, bytes)`.
* `pinned(l, e)` is `blob_offset + blob_bytes <= pinned_bytes`, monotone in `pinned_bytes`.
* `lock_resident` (Windows) locks in chunks of at most 1 GiB whose sizes sum to the region.
* `fnv1a64` composes over chunks, so hashing a layer chunk by chunk equals hashing it whole.
* Layer `L` of the uniform load starts at `L * blobs_per_layer * blob_bytes`; layers tile the file.

## Form template identifiers
`uniform_bounds` → `uniformBounds`; sliced `cudaHostRegister` loop → `registerPrefix`;
`ArenaExpertSource::pinned` → `pinnedBlob`; `lock_resident` chunk loop → `chunks`;
`fnv1a64` → `fnv1a64`; `load_experts` offsets → `layerOff`.
-/

namespace Strata.PinnedArenaResidency

-- § Uniform registration bounds

/-- `uniform_bounds(bytes, slice)`. -/
def uniformBounds (bytes slice : Nat) : List Nat :=
  if slice = 0 then [] else
  if bytes / slice = 0 then [] else (List.range (bytes / slice + 1)).map (· * slice)

/-- Every registration bound lies within the arena. -/
theorem uniformBounds_le {bytes slice x : Nat} (h : x ∈ uniformBounds bytes slice) : x ≤ bytes := by
  unfold uniformBounds at h
  split at h
  · cases h
  · split at h
    · cases h
    · simp only [List.mem_map, List.mem_range] at h
      obtain ⟨i, hi, rfl⟩ := h
      have h1 : i * slice ≤ bytes / slice * slice := Nat.mul_le_mul_right _ (by omega)
      have h2 := Nat.div_mul_le_self bytes slice
      omega

/-- Consecutive bounds are exactly one slice apart. -/
theorem uniformBounds_step (i slice : Nat) : (i + 1) * slice = i * slice + slice := Nat.succ_mul i slice

-- § Sliced registration prefix

/-- Register `(off, len)` slices in order until the first failure; returns `registered_bytes`. -/
def registerPrefix (ok : Nat → Bool) : List (Nat × Nat) → Nat → Nat
  | [], acc => acc
  | (off, n) :: t, acc => if ok off then registerPrefix ok t (off + n) else acc

/-- `registered_bytes` never exceeds the arena when every slice ends inside it. -/
theorem registerPrefix_le (ok : Nat → Bool) (total : Nat) :
    ∀ (sl : List (Nat × Nat)) (acc : Nat), acc ≤ total → (∀ p ∈ sl, p.1 + p.2 ≤ total) →
      registerPrefix ok sl acc ≤ total
  | [], acc, hacc, _ => hacc
  | (off, n) :: t, acc, hacc, hall => by
    simp only [registerPrefix]
    split
    · exact registerPrefix_le ok total t (off + n) (hall (off, n) (by simp))
        (fun p hp => hall p (by simp [hp]))
    · exact hacc

/-- The working-set lock covers exactly the unregistered tail: pinned + locked = arena. -/
theorem lock_partition {registered bytes : Nat} (h : registered ≤ bytes) :
    registered + (bytes - registered) = bytes := by omega

/-- `ArenaExpertSource::pinned`: the whole blob lies in the registered prefix. -/
def pinnedBlob (off len pinnedBytes : Nat) : Bool := decide (off + len ≤ pinnedBytes)

/-- Registering more never un-pins a blob. -/
theorem pinned_mono {off len p p' : Nat} (h : pinnedBlob off len p = true) (hp : p ≤ p') :
    pinnedBlob off len p' = true := by
  simp only [pinnedBlob, decide_eq_true_eq] at h ⊢
  omega

-- § Chunked locking (`lock_resident`)

/-- The chunk sizes `min c r` the lock loop issues, with fuel bounding the iterations. -/
def chunks (c : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, r => if r = 0 then [] else min c r :: chunks c f (r - min c r)

/-- The chunks sum to the region: every byte is locked once. -/
theorem chunks_sum {c : Nat} (hc : 0 < c) : ∀ (f r : Nat), r ≤ f → (chunks c f r).sum = r
  | 0, r, h => by
    have : r = 0 := by omega
    subst this
    rfl
  | f + 1, r, h => by
    unfold chunks
    split
    · simp; omega
    · have ih := chunks_sum hc f (r - min c r) (by omega)
      simp only [List.sum_cons, ih]
      omega

/-- Every chunk is non-empty and at most the chunk size (`c = 1 GiB > 0` in the source). -/
theorem chunks_bounded {c : Nat} (hc : 0 < c) : ∀ (f r x : Nat), x ∈ chunks c f r → 0 < x ∧ x ≤ c
  | 0, r, x, h => by simp [chunks] at h
  | f + 1, r, x, h => by
    unfold chunks at h
    split at h
    · simp at h
    · rename_i hr
      simp only [List.mem_cons] at h
      rcases h with rfl | h
      · constructor <;> omega
      · exact chunks_bounded hc f _ x h

-- § FNV-1a checksum

/-- The FNV-1a 64-bit prime. -/
def FNV_PRIME : UInt64 := 1099511628211

/-- The FNV-1a 64-bit offset basis (the per-layer seed). -/
def FNV_OFFSET : UInt64 := 1469598103934665603

/-- One FNV-1a step. -/
def fnvStep (h : UInt64) (b : UInt8) : UInt64 := (h ^^^ b.toUInt64) * FNV_PRIME

/-- `fnv1a64(p, n, seed)`. -/
def fnv1a64 (bs : List UInt8) (seed : UInt64) : UInt64 := bs.foldl fnvStep seed

/-- Hashing nothing returns the seed. -/
theorem fnv_nil (seed : UInt64) : fnv1a64 [] seed = seed := rfl

/-- Hashing a concatenation continues from the first part's hash. -/
theorem fnv_append (a b : List UInt8) (seed : UInt64) :
    fnv1a64 (a ++ b) seed = fnv1a64 b (fnv1a64 a seed) := by
  simp [fnv1a64, List.foldl_append]

/-- The loader's chunk-by-chunk hash equals the hash of the whole layer. -/
theorem fnv_chunked : ∀ (cs : List (List UInt8)) (seed : UInt64),
    fnv1a64 cs.flatten seed = cs.foldl (fun h c => fnv1a64 c h) seed
  | [], seed => rfl
  | c :: t, seed => by
    simp only [List.flatten_cons, List.foldl_cons]
    rw [fnv_append]
    exact fnv_chunked t (fnv1a64 c seed)

-- § Uniform per-layer load offsets

/-- `load_experts`: layer `L` starts at `L * blobs_per_layer * blob_bytes`. -/
def layerOff (bpl bb L : Nat) : Nat := L * bpl * bb

/-- Layers are contiguous: one layer ends exactly where the next begins. -/
theorem layerOff_contig (bpl bb L : Nat) : layerOff bpl bb L + bpl * bb = layerOff bpl bb (L + 1) := by
  simp only [layerOff, Nat.succ_mul, Nat.add_mul]

/-- Every layer's range ends inside the file of `layers` layers. -/
theorem layer_in_file {bpl bb L layers : Nat} (h : L < layers) :
    layerOff bpl bb L + bpl * bb ≤ layers * bpl * bb := by
  rw [layerOff_contig]
  exact Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ h)

end Strata.PinnedArenaResidency

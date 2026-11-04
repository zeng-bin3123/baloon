/-!
# MODULE: pinned_arena_load
The pinned host arena and the parallel expert load (`pinned.cu`): registration slice bounds, the FNV-1a layer
checksum computed chunk by chunk, the chunked copy loop, uniform layer offsets, and pinned/locked accounting.

## Source semantics
* `uniform_bounds(bytes, slice)` is `{0, s, 2s, ..., q*s}` with `q = bytes / slice` when `q > 0`, else empty;
  every bound is `<= bytes` and consecutive bounds differ by `slice`.
* Hashing a layer chunk by chunk with the previous hash as the seed equals hashing the concatenation.
* The copy loop moves `min(chunk, remaining)` per iteration; the chunk sizes sum to the layer's bytes.
* `load_experts` places layer `L` at `L * blobs_per_layer * blob_bytes`; consecutive layers abut.
* When only a prefix is registered, the locked remainder plus the registered prefix covers the arena.

## Form template identifiers
* `uniform_bounds` -> `uniformBounds`; `fnv1a64` -> `fnv1a`; copy loop in `load_experts_ranges` -> `copyChunks`
* `layer_off[L]` in `load_experts` -> `layerOff`; `registered_bytes` / `locked_bytes` -> `registered` / `locked`
-/

namespace PinnedArenaLoad

-- § registration slice bounds

/-- Slice boundaries: multiples of `slice` up to `bytes / slice`, or empty when no full slice fits. -/
def uniformBounds (bytes slice : Nat) : List Nat :=
  if slice = 0 ∨ bytes / slice = 0 then [] else (List.range (bytes / slice + 1)).map (· * slice)

/-- With a full slice available the bounds have `bytes / slice + 1` entries. -/
theorem uniformBounds_length (bytes slice : Nat) (h : ¬ (slice = 0 ∨ bytes / slice = 0)) :
    (uniformBounds bytes slice).length = bytes / slice + 1 := by
  unfold uniformBounds
  rw [if_neg h]
  simp

/-- Every bound lies inside the arena. -/
theorem uniformBounds_le (bytes slice x : Nat) (h : x ∈ uniformBounds bytes slice) : x ≤ bytes := by
  unfold uniformBounds at h
  split at h
  · simp at h
  · simp only [List.mem_map, List.mem_range] at h
    obtain ⟨k, hk, hx⟩ := h
    subst hx
    have h1 : k * slice ≤ (bytes / slice) * slice :=
      Nat.mul_le_mul_right _ (Nat.le_of_lt_succ hk)
    exact Nat.le_trans h1 (Nat.div_mul_le_self bytes slice)

/-- Consecutive bounds differ by exactly one slice. -/
theorem bound_step (k slice : Nat) : (k + 1) * slice = k * slice + slice := by
  rw [Nat.add_mul, Nat.one_mul]

-- § FNV-1a checksum

/-- FNV-1a over a byte list with an explicit seed (the running hash). -/
def fnv1a (seed : UInt64) (xs : List UInt8) : UInt64 :=
  xs.foldl (fun h b => (h ^^^ b.toUInt64) * 1099511628211) seed

/-- Hashing a concatenation is hashing the first part, then the second with the first's hash as seed. -/
theorem fnv1a_append (seed : UInt64) (xs ys : List UInt8) :
    fnv1a seed (xs ++ ys) = fnv1a (fnv1a seed xs) ys := by
  simp [fnv1a, List.foldl_append]

/-- Hashing chunk by chunk (as the copy loop does) equals hashing the whole layer in one go. -/
theorem fnv1a_chunks : ∀ (seed : UInt64) (cs : List (List UInt8)),
    cs.foldl fnv1a seed = fnv1a seed cs.flatten
  | seed, [] => by simp [fnv1a]
  | seed, c :: cs => by
    rw [List.foldl_cons, fnv1a_chunks (fnv1a seed c) cs, List.flatten_cons, fnv1a_append]

-- § the chunked copy loop

/-- Chunk sizes of the copy loop: `min chunk remaining`, with fuel for structural recursion. -/
def copyChunks (chunk : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, r => if r = 0 then [] else min chunk r :: copyChunks chunk f (r - min chunk r)

/-- The copied bytes add up to the layer's byte count (given enough fuel and a positive chunk). -/
theorem copyChunks_sum (chunk : Nat) (hc : 0 < chunk) :
    ∀ (f r : Nat), r ≤ f → (copyChunks chunk f r).sum = r
  | 0, r, h => by
    have : r = 0 := by omega
    subst this
    simp [copyChunks]
  | f + 1, r, h => by
    unfold copyChunks
    by_cases hr : r = 0
    · simp [hr]
    · rw [if_neg hr]
      have h1 : 0 < min chunk r := by omega
      have h2 : r - min chunk r ≤ f := by omega
      rw [List.sum_cons, copyChunks_sum chunk hc f (r - min chunk r) h2]
      omega

/-- No copy exceeds the chunk buffer, so the staging buffer of `chunk` bytes never overflows. -/
theorem copyChunks_le (chunk : Nat) :
    ∀ (f r c : Nat), c ∈ copyChunks chunk f r → c ≤ chunk
  | 0, r, c, h => by simp [copyChunks] at h
  | f + 1, r, c, h => by
    unfold copyChunks at h
    by_cases hr : r = 0
    · simp [hr] at h
    · rw [if_neg hr, List.mem_cons] at h
      rcases h with h | h
      · subst h; omega
      · exact copyChunks_le chunk f _ c h

-- § uniform layer offsets

/-- Offset of layer `L`: `L * blobs_per_layer * blob_bytes`. -/
def layerOff (blobsPerLayer blobBytes L : Nat) : Nat := L * (blobsPerLayer * blobBytes)

/-- Consecutive layers abut: layer `L+1` begins where layer `L`'s bytes end. -/
theorem layerOff_abut (b w L : Nat) : layerOff b w (L + 1) = layerOff b w L + b * w := by
  unfold layerOff
  rw [Nat.add_mul, Nat.one_mul]

/-- The last layer ends at `layers * blobs_per_layer * blob_bytes`. -/
theorem layerOff_end (b w L layers : Nat) (h : L < layers) :
    layerOff b w L + b * w ≤ layers * (b * w) := by
  rw [← layerOff_abut]
  exact Nat.mul_le_mul_right _ h

-- § pinned and locked accounting

/-- Locked remainder after a registered prefix. -/
def locked (bytes registered : Nat) : Nat := bytes - registered

/-- Registered prefix plus locked remainder covers the arena exactly. -/
theorem registered_plus_locked (bytes registered : Nat) (h : registered ≤ bytes) :
    registered + locked bytes registered = bytes := by
  unfold locked; omega

end PinnedArenaLoad

import Mathlib

/-!
# MODULE: pinned_arena_expert_load
The pinned host arena and the parallel expert load: FNV-1a checksum chained across chunks, chunked reads that
tile a layer range exactly, contiguous per-layer ranges, sliced `cudaHostRegister` bookkeeping, and the
bandwidth probe's byte accounting.

## Source semantics
* The per-layer hash is chained chunk by chunk (`h = fnv1a64(chunk, h)`), which equals hashing the layer whole.
* A layer is read in chunks of `min(chunk, remaining)`; the chunk sizes sum to the layer's bytes.
* Layer `L` starts at `L * blobs_per_layer * blob_bytes`; consecutive layers are contiguous.
* Uniform slices: slice `i` fits iff `i < bytes / slice`.
* Registered prefix + locked tail cover the arena exactly.
* The arena is one blob longer than the file, so a max-blob copy from any expert start stays inside.
* `stream_bandwidth` reports `bytes * iters`, but each pass copies only `(bytes / chunk) * chunk`.

## Form template identifiers
* `fnvStep/fnv`      -> `FNV1A64_CHAIN`
* `chunkList`        -> `READ_CHUNK_PLAN`
* `layerOff`         -> `LAYER_RANGE_OFFSET`
* `passBytes`        -> `STREAM_PASS_BYTES`
-/

namespace StrataPinned

-- § FNV-1a checksum

/-- FNV-1a 64-bit prime. -/
def fnvPrime : Nat := 1099511628211

/-- One FNV-1a step on 64-bit state. -/
def fnvStep (h b : Nat) : Nat := ((h ^^^ b) * fnvPrime) % 2 ^ 64

/-- Fold a byte list into a running hash. -/
def fnv (bytes : List Nat) (seed : Nat) : Nat := bytes.foldl fnvStep seed

/-- Chunk-chained hashing equals hashing the concatenation, so the chunk size cannot change the checksum. -/
theorem fnv_append (a b : List Nat) (seed : Nat) : fnv (a ++ b) seed = fnv b (fnv a seed) := by
  simp [fnv, List.foldl_append]

/-- Hashing nothing leaves the seed unchanged. -/
theorem fnv_nil (seed : Nat) : fnv [] seed = seed := rfl

-- § chunked reads

/-- Chunk sizes for reading `rem` bytes with at most `c` per read (fuel-bounded). -/
def chunkList (c : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, r => if r = 0 then [] else min c r :: chunkList c f (r - min c r)

/-- With positive chunk size and enough fuel the chunks cover the range exactly. -/
theorem chunkList_sum (c : Nat) (hc : 0 < c) :
    ∀ (f r : Nat), r ≤ f → (chunkList c f r).sum = r := by
  intro f
  induction f with
  | zero =>
    intro r hr
    have h0 : r = 0 := by omega
    subst h0
    simp [chunkList]
  | succ f ih =>
    intro r hr
    by_cases h0 : r = 0
    · subst h0; simp [chunkList]
    · simp only [chunkList, if_neg h0, List.sum_cons]
      have h := ih (r - min c r) (by omega)
      rw [h]
      omega

/-- Every chunk is non-empty and at most `c`. -/
theorem chunkList_bounds (c : Nat) (hc : 0 < c) :
    ∀ (f r : Nat), ∀ n ∈ chunkList c f r, 0 < n ∧ n ≤ c := by
  intro f
  induction f with
  | zero => intro r n hn; simp [chunkList] at hn
  | succ f ih =>
    intro r n hn
    by_cases h0 : r = 0
    · subst h0; simp [chunkList] at hn
    · simp only [chunkList, if_neg h0, List.mem_cons] at hn
      rcases hn with hn | hn
      · subst hn; omega
      · exact ih _ n hn

-- § layer ranges

/-- Start of layer `L` in a uniform file. -/
def layerOff (blobsPerLayer blob L : Nat) : Nat := L * blobsPerLayer * blob

/-- Consecutive layers are contiguous. -/
theorem layerOff_succ (b blob L : Nat) : layerOff b blob (L + 1) = layerOff b blob L + b * blob := by
  unfold layerOff
  rw [Nat.succ_mul, Nat.add_mul]

/-- 48 layers of 512 blobs of 1,382,400 B total the expert file size. -/
theorem layers_total : layerOff 512 1382400 48 = 33973862400 := by decide

-- § uniform registration slices

/-- Slice `i` of size `s` lies fully within `bytes` iff `i < bytes / s`. -/
theorem slice_fits_iff (bytes s i : Nat) (hs : 0 < s) : (i + 1) * s ≤ bytes ↔ i < bytes / s := by
  rw [Nat.lt_iff_add_one_le, Nat.le_div_iff_mul_le hs]

/-- Registered prefix plus locked tail equals the arena. -/
theorem registered_plus_tail (bytes reg : Nat) (h : reg ≤ bytes) : reg + (bytes - reg) = bytes := by
  omega

/-- A max-blob copy from any expert start stays inside the arena that is one blob longer than the file. -/
theorem arena_slack (off b mx want : Nat) (h : off + b ≤ want) : off + mx ≤ want + mx := by
  omega

-- § bandwidth probe accounting

/-- Bytes the probe actually copies per pass. -/
def passBytes (bytes chunk : Nat) : Nat := bytes / chunk * chunk

/-- The reported `bytes` is never less than what a pass copies. -/
theorem passBytes_le (bytes chunk : Nat) : passBytes bytes chunk ≤ bytes := by
  unfold passBytes
  exact Nat.div_mul_le_self bytes chunk

/-- The two agree exactly when the chunk divides the size. -/
theorem passBytes_eq_of_dvd (bytes chunk : Nat) (h : bytes % chunk = 0) : passBytes bytes chunk = bytes := by
  unfold passBytes
  have := Nat.div_add_mod bytes chunk
  rw [h] at this
  rw [Nat.mul_comm]
  omega

end StrataPinned

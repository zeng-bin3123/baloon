/-!
# MODULE: prefill_chunk_driver
Chunked prefill driver: bump allocation of the device region, chunk iteration, host-side counting sort of
(token, expert) pairs, and the host->device expert staging ring with a dequantization ring.

## Source semantics
* Each allocation is rounded as `(n*sizeof + 256 + 255) & ~255`: 256-aligned and at least 256 bytes above the payload.
* In borrowed-region mode an allocation fails exactly when it would exceed the capacity; the count-only mode used by
  `bytes_needed` accumulates the same sizes, so a region of that size always fits the sequence.
* Chunks are `min(T, n - c0)` tokens starting at `c0 < n`, so they never run past `n`.
* MoE grouping: `off[e+1] = off[e] + cnt[e]`; token-expert pair `i = t*K + k` of expert `e` gets slot
  `off[e] + rank` with `rank < cnt[e]`; slot ranges of different experts are disjoint; `src[slot] = i / K`.
* Staging: at most `STAGE` blobs are live (`staged <= j + STAGE - 1`), so their slots `idx % STAGE` are distinct;
  consecutive experts alternate between the `DQ = 2` dequant buffers.
* Layer pattern: QSA at layers `l % 4 = 3` (12 of 48), GDN elsewhere (36).

## Form template identifiers
Alloc::take -> `alignedBytes`/`takeSeq`; chunk loop -> `chunkLen`; counting sort -> `offAt`; ring -> `% 8`, `% 2`;
GDN state floats -> `gdnFloats`.
-/

namespace Strata.Prefill.Driver

-- § allocation

/-- Bytes charged for an allocation of `b` payload bytes. -/
def alignedBytes (b : Nat) : Nat := (b + 256 + 255) / 256 * 256

/-- Charged size is 256-aligned. -/
theorem alignedBytes_aligned (b : Nat) : alignedBytes b % 256 = 0 := by
  unfold alignedBytes
  omega

/-- Charged size covers the payload plus the 256-byte slack. -/
theorem alignedBytes_ge (b : Nat) : b + 256 ≤ alignedBytes b := by
  unfold alignedBytes
  omega

/-- Charged size is less than payload + 512. -/
theorem alignedBytes_lt (b : Nat) : alignedBytes b < b + 512 := by
  unfold alignedBytes
  omega

/-- Sequence of bump allocations inside a region of capacity `cap`; `none` when one does not fit. -/
def takeSeq (cap : Nat) : List Nat → Nat → Option Nat
  | [], used => some used
  | b :: bs, used => if used + b ≤ cap then takeSeq cap bs (used + b) else none

/-- A successful sequence consumes exactly the sum of the sizes and never exceeds the capacity. -/
theorem takeSeq_spec (cap : Nat) : ∀ (bs : List Nat) (used u : Nat), takeSeq cap bs used = some u →
    u = used + bs.sum ∧ (used ≤ cap → u ≤ cap) := by
  intro bs
  induction bs with
  | nil =>
    intro used u h
    simp only [takeSeq] at h
    cases h
    simp
  | cons b bs ih =>
    intro used u h
    simp only [takeSeq] at h
    split at h
    · rename_i hc
      obtain ⟨h1, h2⟩ := ih (used + b) u h
      refine ⟨?_, fun _ => h2 (by omega)⟩
      rw [List.sum_cons]
      omega
    · cases h

/-- A region of exactly the counted size fits the whole sequence. -/
theorem takeSeq_fits (bs : List Nat) (used : Nat) : takeSeq (used + bs.sum) bs used = some (used + bs.sum) := by
  induction bs generalizing used with
  | nil => simp [takeSeq]
  | cons b bs ih =>
    have h1 : used + b ≤ used + (b + bs.sum) := by omega
    have e : used + b + bs.sum = used + (b + bs.sum) := by omega
    have ih' := ih (used + b)
    rw [e] at ih'
    simp only [takeSeq, List.sum_cons]
    simp [h1, ih']

-- § chunk loop

/-- Length of the chunk starting at `c0`. -/
def chunkLen (T n c0 : Nat) : Nat := min T (n - c0)

/-- Every chunk is non-empty, no longer than `T`, and ends at or before `n`. -/
theorem chunk_bounds (T n c0 : Nat) (hT : 0 < T) (hc : c0 < n) :
    0 < chunkLen T n c0 ∧ chunkLen T n c0 ≤ T ∧ c0 + chunkLen T n c0 ≤ n := by
  unfold chunkLen
  omega

-- § counting sort

/-- Prefix-sum offsets: `off[0] = 0`, `off[e+1] = off[e] + cnt[e]`. -/
def offAt (cnt : Nat → Nat) : Nat → Nat
  | 0 => 0
  | e + 1 => offAt cnt e + cnt e

/-- Offsets are monotone. -/
theorem offAt_mono (cnt : Nat → Nat) (e d : Nat) : offAt cnt e ≤ offAt cnt (e + d) := by
  induction d with
  | zero => simp
  | succ d ih =>
    have h : offAt cnt (e + (d + 1)) = offAt cnt (e + d) + cnt (e + d) := rfl
    omega

/-- A slot `off[e] + r` with `r < cnt[e]` lies in expert `e`'s range. -/
theorem slot_in_range (cnt : Nat → Nat) (e r : Nat) (hr : r < cnt e) :
    offAt cnt e ≤ offAt cnt e + r ∧ offAt cnt e + r < offAt cnt (e + 1) := by
  have h : offAt cnt (e + 1) = offAt cnt e + cnt e := rfl
  omega

/-- Ranges of different experts are disjoint. -/
theorem ranges_disjoint (cnt : Nat → Nat) (e f : Nat) (h : e < f) : offAt cnt (e + 1) ≤ offAt cnt f := by
  obtain ⟨d, hd⟩ := Nat.exists_eq_add_of_le h
  rw [hd]
  exact offAt_mono cnt (e + 1) d

/-- If the counts total `T*K` over 512 experts, every slot is below `T*K`. -/
theorem slot_lt_total (cnt : Nat → Nat) (T e r : Nat) (he : e < 512) (hr : r < cnt e)
    (hsum : offAt cnt 512 = T * 10) : offAt cnt e + r < T * 10 := by
  have h1 := (slot_in_range cnt e r hr).2
  have h2 := offAt_mono cnt (e + 1) (512 - (e + 1))
  have e2 : e + 1 + (512 - (e + 1)) = 512 := by omega
  rw [e2] at h2
  omega

/-- The source token of pair `i = t*K + k` (K = 10) is `t`. -/
theorem src_of_pair (t k : Nat) (hk : k < 10) : (t * 10 + k) / 10 = t := by
  omega

-- § staging ring

/-- Blobs staged in the window `[j, j + 7]` occupy distinct ring slots (`idx % 8`). -/
theorem ring_distinct (j a b : Nat) (ha : j ≤ a) (ha' : a ≤ j + 7) (hb : j ≤ b) (hb' : b ≤ j + 7)
    (h : a % 8 = b % 8) : a = b := by
  omega

/-- Consecutive experts use different dequantization buffers (`DQ = 2`). -/
theorem dq_alternates (j : Nat) : j % 2 ≠ (j + 1) % 2 := by
  omega

-- § geometry coherence

/-- Buffer dimensions used by `init` agree with each other. -/
theorem buffer_dims :
    2560 * 4 = 10240 ∧ 48 * 128 = 6144 ∧ 24 * 256 = 6144 ∧ 24 * 512 = 12288 ∧ 2 * 640 = 1280 ∧
      16 * 160 = 2560 ∧ 1280 * 640 * 4 = 1280 * 2560 ∧ 2560 * 160 * 4 = 2560 * 640 := by decide

/-- GDN state floats per layer: recurrence plus conv history. -/
def gdnFloats : Nat := 128 * 48 * 128 + 10240 * 3

/-- GDN state per layer is 3,268,608 bytes, matching the memory planner's per-layer term. -/
theorem gdn_state_bytes : gdnFloats * 4 = 3268608 := by decide

/-- 12 QSA layers (`l % 4 = 3`) and 36 GDN layers among 48. -/
theorem layer_pattern :
    ((List.range 48).filter (fun l => l % 4 = 3)).length = 12 ∧
      ((List.range 48).filter (fun l => l % 4 ≠ 3)).length = 36 := by decide

/-- PLE row index `t*16 + h` is inside the `T*16` batch rows. -/
theorem ple_rows_range (t T h : Nat) (ht : t < T) (hh : h < 16) : t * 16 + h < T * 16 := by
  omega

end Strata.Prefill.Driver

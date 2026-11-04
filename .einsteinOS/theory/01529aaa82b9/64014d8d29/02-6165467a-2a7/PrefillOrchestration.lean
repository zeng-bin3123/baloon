/-!
# MODULE: prefill_orchestration
SOURCES: src/prefill/prefill.cpp
Batched prompt processing: the prompt is cut into chunks of `T` tokens; per chunk every layer runs
a hyper-connection read, a GDN / QSA / MoE half, and a write.  MoE experts are grouped by id on the
host, staged through a host->device ring and dequantized through a two-slot ring.  Buffers come from
a bump allocator over a borrowed region (or `cudaMalloc`).

## Source semantics
* Chunks of size `min T (n - c0)` partition `n` positions exactly.
* The bump allocator rounds each request to 256 bytes (+256 slack) and fails exactly when the
  region would overflow.
* A staging ring of 8 slots with lookahead 7 never aliases two live blobs.
* Grouping tokens by expert yields cursors strictly inside each expert's block.

## Form template identifiers
* `m.T` (chunk size)          -> `T`
* `Alloc::take`               -> `roundBytes`, `allocAll`
* `STAGE`, `DQ`               -> ring sizes 8 and 2
* `gdn_floats`                -> `gdnFloats`
-/

namespace Strata.Prefill

-- § chunk partition

/-- Chunk lengths produced by the `for (c0 = 0; c0 < n; c0 += T)` loop (fuel-bounded). -/
def chunksAux : Nat → Nat → Nat → List Nat
  | 0, _, _ => []
  | fuel + 1, T, n => if n = 0 then [] else min T n :: chunksAux fuel T (n - min T n)

/-- The chunk list of an `n`-token prompt with chunk size `T`. -/
def chunks (T n : Nat) : List Nat := chunksAux n T n

/-- Chunk lengths sum to the number of positions processed. -/
theorem sum_chunksAux (T : Nat) (hT : 0 < T) :
    ∀ fuel n, n ≤ fuel → (chunksAux fuel T n).sum = n := by
  intro fuel
  induction fuel with
  | zero =>
    intro n h
    have hn : n = 0 := by omega
    subst hn
    simp [chunksAux]
  | succ f ih =>
    intro n h
    by_cases hn : n = 0
    · subst hn
      simp [chunksAux]
    · have hm : 0 < min T n := by omega
      have h2 := ih (n - min T n) (by omega)
      rw [chunksAux, if_neg hn, List.sum_cons, h2]
      omega

/-- The full chunk list covers the prompt exactly. -/
theorem sum_chunks (T n : Nat) (hT : 0 < T) : (chunks T n).sum = n :=
  sum_chunksAux T hT n n (Nat.le_refl n)

/-- Every chunk is non-empty and at most `T` tokens (so it fits the chunk buffers). -/
theorem chunksAux_bounds (T : Nat) (hT : 0 < T) :
    ∀ fuel n, ∀ x ∈ chunksAux fuel T n, 0 < x ∧ x ≤ T := by
  intro fuel
  induction fuel with
  | zero =>
    intro n x hx
    simp [chunksAux] at hx
  | succ f ih =>
    intro n x hx
    by_cases hn : n = 0
    · subst hn
      simp [chunksAux] at hx
    · rw [chunksAux, if_neg hn] at hx
      rcases List.mem_cons.mp hx with h | hx'
      · rw [h]
        omega
      · exact ih _ x hx'

-- § bump allocator

/-- Bytes charged for a request of `b` bytes: `(b + 256 + 255) & ~255`. -/
def roundBytes (b : Nat) : Nat := (b + 256 + 255) / 256 * 256

/-- Every allocation is 256-byte aligned. -/
theorem roundBytes_aligned (b : Nat) : roundBytes b % 256 = 0 := by
  unfold roundBytes
  omega

/-- Every allocation leaves at least the 256 byte slack. -/
theorem roundBytes_ge (b : Nat) : b + 256 ≤ roundBytes b := by
  unfold roundBytes
  omega

/-- Rounding wastes less than one alignment unit. -/
theorem roundBytes_lt (b : Nat) : roundBytes b < b + 256 + 256 := by
  unfold roundBytes
  omega

/-- Sequential bump allocation of `sizes` into a region of capacity `cap`. -/
def allocAll (cap : Nat) : List Nat → Nat → Option Nat
  | [], used => some used
  | b :: bs, used => if used + b ≤ cap then allocAll cap bs (used + b) else none

/-- A successful allocation sequence never exceeds the region. -/
theorem allocAll_le (cap : Nat) : ∀ (bs : List Nat) (used u : Nat),
    used ≤ cap → allocAll cap bs used = some u → u ≤ cap := by
  intro bs
  induction bs with
  | nil =>
    intro used u hu h
    simp [allocAll] at h
    omega
  | cons b bs ih =>
    intro used u hu h
    by_cases hb : used + b ≤ cap
    · rw [allocAll, if_pos hb] at h
      exact ih _ u hb h
    · rw [allocAll, if_neg hb] at h
      simp at h

/-- A single request fits exactly when the region has room. -/
theorem alloc_single_iff (cap used b : Nat) :
    (allocAll cap [b] used).isSome = true ↔ used + b ≤ cap := by
  by_cases hb : used + b ≤ cap
  · simp [allocAll, hb]
  · simp [allocAll, hb]

-- § staging rings

/-- Two staged blobs within one lookahead window (< 8 apart) never share a staging slot. -/
theorem stage_slots_distinct (i k : Nat) (hlt : i < k) (hwin : k < i + 8) : i % 8 ≠ k % 8 := by
  omega

/-- Lookahead 7 is strictly below the ring size 8, so the window fits the ring. -/
theorem lookahead_lt_stage : (7 : Nat) < 8 := by decide

/-- Two experts in flight on the 2-slot dequantized ring never share a slot. -/
theorem dq_slots_distinct (j k : Nat) (hlt : j < k) (hwin : k < j + 2) : j % 2 ≠ k % 2 := by
  omega

-- § expert grouping (counting sort by expert id)

/-- A token-expert pair's destination row lies strictly inside its expert's block. -/
theorem group_slot_in_block (off cnt c : Nat) (h : c < cnt) : off + c < off + cnt := by
  omega

/-- The next block starts where this block ends (`off[e+1] = off[e] + cnt[e]`). -/
theorem group_blocks_disjoint (off cnt c off' : Nat) (h : c < cnt) (h' : off' = off + cnt) :
    off + c < off' := by
  omega

-- § GDN state layout

/-- Floats of GDN state per layer: `128*48*128` recurrent plus `10240*3` conv history. -/
def gdnFloats : Nat := 128 * 48 * 128 + 10240 * 3

/-- The per-layer GDN state is 817,152 floats. -/
theorem gdnFloats_value : gdnFloats = 817152 := by decide

/-- The conv history starts after the recurrent state inside one layer's block. -/
theorem gdn_conv_offset (idx : Nat) :
    idx * gdnFloats + 128 * 48 * 128 < (idx + 1) * gdnFloats := by
  unfold gdnFloats
  omega

end Strata.Prefill

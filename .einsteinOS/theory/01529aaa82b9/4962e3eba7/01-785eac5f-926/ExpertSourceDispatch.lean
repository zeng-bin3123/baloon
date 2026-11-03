import Mathlib

/-!
# MODULE: expert_source_dispatch
The expert pool's adapter to the host loop: file/arena blob addressing, the per-layer hit/miss split by router
index, the GPU/PCIe share of verify-window misses, and the sticky failure latch (`ExpertDispatch`,
`FileExpertSource`, `ArenaExpertSource`, `expert_pool_dispatch*`, `expert_hit_run`).

## Source semantics
* `experts.bin` holds exactly `48 * 512 = 24,576` blobs of `1,382,400` bytes; blob index is
  `layer * n_expert + expert` with no padding, and `expert < n_expert` is checked separately from the flat index.
* A layer routes `k` experts; each router index is a hit (GPU computes it, CPU row zeroed) or a miss (a CPU job).
  Hits and misses partition the router indices, and `parts[j] = cpu[j] + gpu[j]` so the router weight, applied
  once by `moe_combine`, sees each expert exactly once.
* In a verify window `pcie_num/256` of the distinct missed experts (the last ones in routing order) go to the GPU.
* `failed` latches: once set it is never re-armed by clearing the message.

## Form template identifiers
* `blobIdx`, `blobOk`   <- `FileExpertSource::blob`, `ArenaExpertSource::blob`
* `latch`               <- `ExpertDispatch::failed`
* `pcieShare`           <- `m = (nmiss * d.pcie_num) >> 8` in `expert_pool_dispatch_multi`
* `hit`/`cpu`/`gpu` rows <- `is_hit`, the pool's `out`, `hit_out`
-/

namespace Strata.ExpertSource

-- § geometry of the pack

/-- Bytes of one expert blob in the canonical pack. -/
def blobBytes : Nat := 1382400

/-- The pack holds `48 x 512` blobs of `1,382,400` B: exactly the size of `experts.bin`. -/
theorem pack_size : 48 * 512 * blobBytes = 33973862400 := by
  unfold blobBytes
  norm_num

/-- `48 x 512` is `24,576` blobs. -/
theorem pack_blobs : 48 * 512 = 24576 := by
  norm_num

-- § addressing

/-- Flat blob index: `layer * n_expert + expert`, no padding. -/
def blobIdx (ne l e : Nat) : Nat := l * ne + e

/-- A valid `(layer, expert)` pair addresses a blob inside the file. -/
theorem blobIdx_lt (nl ne l e : Nat) (hl : l < nl) (he : e < ne) : blobIdx ne l e < nl * ne := by
  unfold blobIdx
  have h1 : (l + 1) * ne ≤ nl * ne := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hl)
  have h2 : (l + 1) * ne = l * ne + ne := by ring
  omega

/-- Why `expert >= n_expert` needs its own check: the flat index of `(l, ne + e')` is `(l + 1, e')`'s. -/
theorem blobIdx_unchecked_alias (ne l e' : Nat) : blobIdx ne l (ne + e') = blobIdx ne (l + 1) e' := by
  unfold blobIdx
  ring

/-- Distinct valid pairs address distinct blobs. -/
theorem blobIdx_inj (ne l l' e e' : Nat) (hne : 0 < ne) (he : e < ne) (he' : e' < ne)
    (h : blobIdx ne l e = blobIdx ne l' e') : l = l' ∧ e = e' := by
  unfold blobIdx at h
  have hm : (l * ne + e) % ne = e := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right]
    exact Nat.mod_eq_of_lt he
  have hm' : (l' * ne + e') % ne = e' := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right]
    exact Nat.mod_eq_of_lt he'
  have hee : e = e' := by
    calc e = (l * ne + e) % ne := hm.symm
      _ = (l' * ne + e') % ne := by rw [h]
      _ = e' := hm'
  have hll : l * ne = l' * ne := by omega
  exact ⟨Nat.eq_of_mul_eq_mul_right hne hll, hee⟩

-- § the hit/miss split

/-- Hits and misses partition the router indices: every index is exactly one of the two. -/
theorem split_partition {α : Type} (p : α → Bool) (l : List α) :
    (l.filter p).length + (l.filter (fun x => !p x)).length = l.length := by
  induction l with
  | nil => simp
  | cons a t ih =>
    cases hp : p a <;> simp [List.filter_cons, hp] <;> omega

/-- The CPU job count never exceeds the number of routed experts. -/
theorem jobs_le_k {α : Type} (p : α → Bool) (l : List α) : (l.filter p).length ≤ l.length := by
  exact List.length_filter_le p l

/-- A hit's row is zero on the CPU side and the expert on the GPU side (and the reverse for a miss), so the sum
is the expert's output once: the router weight is applied exactly once per expert. -/
theorem split_combine (k : Nat) (w e : Fin k → ℤ) (hit : Fin k → Bool) :
    ∑ j, w j * ((if hit j then 0 else e j) + (if hit j then e j else 0)) = ∑ j, w j * e j := by
  refine Finset.sum_congr rfl ?_
  intro j _
  cases hh : hit j <;> simp [hh]

-- § the GPU/PCIe share of a verify window

/-- Distinct missed experts the GPU reads over PCIe: `nmiss * pcie_num / 256`. -/
def pcieShare (nmiss pcieNum : Nat) : Nat := nmiss * pcieNum / 256

/-- With `pcie_num <= 256` the PCIe share never exceeds the number of misses. -/
theorem pcieShare_le (nmiss pcieNum : Nat) (h : pcieNum ≤ 256) : pcieShare nmiss pcieNum ≤ nmiss := by
  unfold pcieShare
  have h1 : nmiss * pcieNum ≤ nmiss * 256 := Nat.mul_le_mul_left nmiss h
  omega

/-- Choosing the last `m` of `nmiss` miss ranks selects exactly `m` experts. -/
theorem pcie_select_card (nmiss m : Nat) (h : m ≤ nmiss) : (Finset.Ico (nmiss - m) nmiss).card = m := by
  rw [Nat.card_Ico]
  omega

-- § the failure latch

/-- `failed` after a step: set when the step fails, never cleared. -/
def latch (failed ok : Bool) : Bool := failed || !ok

/-- Once failed, always failed: clearing the message cannot re-arm the adapter. -/
theorem latch_sticky (ok : Bool) : latch true ok = true := by
  simp [latch]

/-- The latch is monotone over any step. -/
theorem latch_mono (f ok : Bool) (h : f = true) : latch f ok = true := by
  simp [latch, h]

/-- A failing step sets the latch from any state. -/
theorem latch_sets (f : Bool) : latch f false = true := by
  simp [latch]

-- § pinned arena

/-- A blob is DMA-able only if its whole extent lies inside the registered range. -/
def pinnedAt (off bytes pinnedBytes : Nat) : Prop := off + bytes ≤ pinnedBytes

/-- A pinned blob's start is inside the registered range. -/
theorem pinnedAt_start (off bytes pinnedBytes : Nat) (h : pinnedAt off bytes pinnedBytes) :
    off ≤ pinnedBytes := by
  unfold pinnedAt at h
  omega

end Strata.ExpertSource

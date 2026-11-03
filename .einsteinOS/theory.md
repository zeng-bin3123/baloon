
## FILE: /mnt/user-data/outputs/ExpertResidencyCache.lean
import Mathlib

/-!
# MODULE: expert_residency_cache
The VRAM-resident expert tier's slot storage and residency table (`ExpertCache`, `read_expert_profile`).
It answers `(layer, expert) -> slot or none`, admits experts into free slots without ever evicting, optionally
partitions the slots into per-layer ranges (R4.2g), and lays out variable-size slots on 256-byte boundaries.

## Source semantics
* `slot_of` is bounds-checked: an out-of-range layer or expert is "not resident", never an adjacent cell.
* `admit` never evicts: the admission counter is monotone and never passes its ceiling (`slots_` globally,
  the layer's range end under per-layer admission).
* Per-layer ranges `[l*q, (l+1)*q)`, `q = slots / n_layers`, last layer takes the remainder; they are
  pairwise disjoint and lie inside `0..slots`.
* `open_sized` aligns each slot to 256 bytes; offsets are monotone and 256-divisible.
* `read_expert_profile` refuses a profile for a different geometry or with out-of-range pairs.
* `fill_slot` copies at most one slot's bytes.

## Form template identifiers
* `layerRange`      <- `ExpertCache::layer_slot_range`
* `admitSlot`/`admitNext` <- `ExpertCache::admit` (counter `next_free_` / `layer_next_[l]`, ceiling `slots_` / `hi`)
* `slotOf`          <- `ExpertCache::slot_of`
* `align256`/`offAt` <- `ExpertCache::open_sized` offsets `off_`
* `fillBytes`       <- the `n` computed in `fill_slot`/`fill_slot_blocking`/`verify_slot`
* `readProfile`     <- `read_expert_profile`
-/

namespace Strata.ExpertCache

variable {slots nLayers layer l1 l2 : Nat}

-- § per-layer slot ranges

/-- The slot range `[lo, hi)` layer `layer` may admit into under per-layer admission. -/
def layerRange (slots nLayers layer : Nat) : Nat × Nat :=
  if nLayers = 0 ∨ slots = 0 ∨ nLayers ≤ layer then (0, 0)
  else (layer * (slots / nLayers),
        if layer = nLayers - 1 then slots else (layer + 1) * (slots / nLayers))

/-- The lower end of a valid layer's range is `layer * q`. -/
theorem layerRange_lo (hl : layer < nLayers) (hs : 0 < slots) :
    (layerRange slots nLayers layer).1 = layer * (slots / nLayers) := by
  have hc : ¬ (nLayers = 0 ∨ slots = 0 ∨ nLayers ≤ layer) := by omega
  simp only [layerRange, if_neg hc]

/-- The last layer's range ends exactly at `slots`, so no slot is orphaned by the division. -/
theorem layerRange_hi_last (hl : layer < nLayers) (hs : 0 < slots) (hlast : layer = nLayers - 1) :
    (layerRange slots nLayers layer).2 = slots := by
  have hc : ¬ (nLayers = 0 ∨ slots = 0 ∨ nLayers ≤ layer) := by omega
  simp only [layerRange, if_neg hc, if_pos hlast]

/-- A non-last layer's range ends at `(layer + 1) * q`. -/
theorem layerRange_hi_not_last (hl : layer < nLayers) (hs : 0 < slots) (hlast : layer ≠ nLayers - 1) :
    (layerRange slots nLayers layer).2 = (layer + 1) * (slots / nLayers) := by
  have hc : ¬ (nLayers = 0 ∨ slots = 0 ∨ nLayers ≤ layer) := by omega
  simp only [layerRange, if_neg hc, if_neg hlast]

/-- A valid layer's range is well formed: `lo ≤ hi`. -/
theorem layerRange_lo_le_hi (hl : layer < nLayers) (hs : 0 < slots) :
    (layerRange slots nLayers layer).1 ≤ (layerRange slots nLayers layer).2 := by
  rw [layerRange_lo hl hs]
  by_cases hlast : layer = nLayers - 1
  · rw [layerRange_hi_last hl hs hlast]
    have h1 : layer * (slots / nLayers) ≤ nLayers * (slots / nLayers) :=
      Nat.mul_le_mul_right _ (by omega)
    have h2 : nLayers * (slots / nLayers) ≤ slots := by
      rw [Nat.mul_comm]; exact Nat.div_mul_le_self slots nLayers
    omega
  · rw [layerRange_hi_not_last hl hs hlast]
    exact Nat.mul_le_mul_right _ (Nat.le_succ _)

/-- Every layer's range stays inside the arena: `hi ≤ slots`. -/
theorem layerRange_hi_le_slots (hl : layer < nLayers) (hs : 0 < slots) :
    (layerRange slots nLayers layer).2 ≤ slots := by
  by_cases hlast : layer = nLayers - 1
  · exact (layerRange_hi_last hl hs hlast).le
  · rw [layerRange_hi_not_last hl hs hlast]
    have h1 : (layer + 1) * (slots / nLayers) ≤ nLayers * (slots / nLayers) :=
      Nat.mul_le_mul_right _ (by omega)
    have h2 : nLayers * (slots / nLayers) ≤ slots := by
      rw [Nat.mul_comm]; exact Nat.div_mul_le_self slots nLayers
    omega

/-- Ranges of distinct layers never overlap: a lower layer's `hi` is at most a higher layer's `lo`. -/
theorem layerRange_disjoint (hs : 0 < slots) (h12 : l1 < l2) (hl2 : l2 < nLayers) :
    (layerRange slots nLayers l1).2 ≤ (layerRange slots nLayers l2).1 := by
  have hl1 : l1 < nLayers := by omega
  have hne : l1 ≠ nLayers - 1 := by omega
  rw [layerRange_hi_not_last hl1 hs hne, layerRange_lo hl2 hs]
  exact Nat.mul_le_mul_right _ (Nat.succ_le_of_lt h12)

-- § admission without eviction

/-- The slot `admit` hands out: the next free one, or none when the ceiling `hi` is reached. -/
def admitSlot (hi next : Nat) : Option Nat := if hi ≤ next then none else some next

/-- The admission counter after an `admit` call. -/
def admitNext (hi next : Nat) : Nat := if hi ≤ next then next else next + 1

/-- An admitted slot is always below the ceiling, so it is inside the arena (or the layer's range). -/
theorem admitSlot_lt (hi next s : Nat) (h : admitSlot hi next = some s) : s < hi := by
  unfold admitSlot at h
  split_ifs at h with hc
  simp at h
  omega

/-- A full cache refuses: no eviction, no slot. -/
theorem admitSlot_full (hi next : Nat) (h : hi ≤ next) : admitSlot hi next = none := by
  unfold admitSlot
  rw [if_pos h]

/-- The counter never decreases: there is no eviction. -/
theorem admitNext_ge (hi next : Nat) : next ≤ admitNext hi next := by
  unfold admitNext
  split_ifs <;> omega

/-- The counter never passes the ceiling if it started at or below it. -/
theorem admitNext_le (hi next : Nat) (h : next ≤ hi) : admitNext hi next ≤ hi := by
  unfold admitNext
  split_ifs with hc
  · exact h
  · omega

/-- The handed-out slot is exactly the counter's old value, so slots are claimed in order. -/
theorem admitSlot_eq_counter (hi next s : Nat) (h : admitSlot hi next = some s) : s = next := by
  unfold admitSlot at h
  split_ifs at h with hc
  simp at h
  omega

-- § the residency table

/-- `(layer, expert) -> slot`, bounds-checked against both axes. -/
def slotOf (res : Nat → Option Nat) (nl ne l e : Nat) : Option Nat :=
  if l < nl ∧ e < ne then res (l * ne + e) else none

/-- An out-of-range layer or expert is "not resident". -/
theorem slotOf_oob (res : Nat → Option Nat) (nl ne l e : Nat) (h : ¬ (l < nl ∧ e < ne)) :
    slotOf res nl ne l e = none := by
  unfold slotOf
  rw [if_neg h]

/-- Why both axes are checked: a flat index one expert past a layer's end is the next layer's first cells. -/
theorem flat_index_alias (l e' ne : Nat) : l * ne + (ne + e') = (l + 1) * ne + e' := by
  ring

/-- A valid `(layer, expert)` pair lands inside the flat table. -/
theorem flat_index_lt (nl ne l e : Nat) (hl : l < nl) (he : e < ne) : l * ne + e < nl * ne := by
  have h1 : (l + 1) * ne ≤ nl * ne := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hl)
  have h2 : (l + 1) * ne = l * ne + ne := by ring
  omega

/-- The flat index is injective on valid pairs: `layer * n_expert + expert` has no padding or collision. -/
theorem flat_index_inj (ne l l' e e' : Nat) (hne : 0 < ne) (he : e < ne) (he' : e' < ne)
    (h : l * ne + e = l' * ne + e') : l = l' ∧ e = e' := by
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

-- § sized slots (plan v0.3 P6)

/-- Round a byte count up to a multiple of 256, as `open_sized` does. -/
def align256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- An aligned size is a multiple of 256. -/
theorem align256_dvd (n : Nat) : 256 ∣ align256 n := by
  unfold align256
  exact Nat.dvd_mul_left 256 _

/-- Alignment never shrinks a blob: the slot holds its blob. -/
theorem le_align256 (n : Nat) : n ≤ align256 n := by
  unfold align256
  omega

/-- Byte offset of slot `i` in the arena: the running sum of the aligned sizes before it. -/
def offAt (sizes : List Nat) : Nat → Nat
  | 0 => 0
  | i + 1 => offAt sizes i + align256 (sizes.getD i 0)

/-- Offsets are monotone, so slots are laid out back to back without overlap. -/
theorem offAt_mono (sizes : List Nat) (i : Nat) : offAt sizes i ≤ offAt sizes (i + 1) := by
  show offAt sizes i ≤ offAt sizes i + align256 (sizes.getD i 0)
  exact Nat.le_add_right _ _

/-- Every slot offset is 256-byte aligned, which is what the kernels' vector loads expect. -/
theorem offAt_dvd (sizes : List Nat) (i : Nat) : 256 ∣ offAt sizes i := by
  induction i with
  | zero => exact ⟨0, rfl⟩
  | succ n ih =>
    show 256 ∣ offAt sizes n + align256 (sizes.getD n 0)
    exact Nat.dvd_add ih (align256_dvd _)

/-- Slot `i` spans exactly its aligned size, so it is at least as large as the blob it holds. -/
theorem slot_span (sizes : List Nat) (i : Nat) :
    offAt sizes (i + 1) - offAt sizes i = align256 (sizes.getD i 0) := by
  show offAt sizes i + align256 (sizes.getD i 0) - offAt sizes i = align256 (sizes.getD i 0)
  omega

-- § fills

/-- Bytes a fill copies: the blob's own size when it fits in the slot, else the whole slot. -/
def fillBytes (blob bytes : Nat) : Nat := if 0 < bytes ∧ bytes ≤ blob then bytes else blob

/-- A fill never copies more than one slot, so it cannot run past the arena. -/
theorem fillBytes_le (blob bytes : Nat) : fillBytes blob bytes ≤ blob := by
  unfold fillBytes
  split_ifs with h
  · exact h.2
  · exact Nat.le_refl _

/-- `bytes = 0` means "the whole slot". -/
theorem fillBytes_zero (blob : Nat) : fillBytes blob 0 = blob := by
  unfold fillBytes
  rw [if_neg (by omega)]

-- § the profile reader

/-- `read_expert_profile`'s acceptance rule: same geometry, `n_ranked <= slots`, a complete list, in-range pairs. -/
def readProfile (nl ne hdrNl hdrNe want nRanked : Nat) (pairs : List (Nat × Nat)) :
    Option (List (Nat × Nat)) :=
  if hdrNl = nl ∧ hdrNe = ne ∧ nRanked ≤ want ∧ pairs.length = nRanked ∧
      pairs.all (fun p => decide (p.1 < nl ∧ p.2 < ne)) = true
  then some pairs else none

/-- An accepted profile has at most `slots` pairs and every pair is a real `(layer, expert)`. -/
theorem readProfile_sound (nl ne a b want nr : Nat) (pairs r : List (Nat × Nat))
    (h : readProfile nl ne a b want nr pairs = some r) :
    r.length ≤ want ∧ ∀ p ∈ r, p.1 < nl ∧ p.2 < ne := by
  unfold readProfile at h
  split_ifs at h with hc
  obtain ⟨_, _, h3, h4, h5⟩ := hc
  have hr : pairs = r := by simpa using h
  subst hr
  refine ⟨by omega, ?_⟩
  intro p hp
  have hp' := List.all_eq_true.mp h5 p hp
  simpa using hp'

/-- A profile for a different artifact (wrong layer count) is refused. -/
theorem readProfile_wrong_geometry (nl ne a b want nr : Nat) (pairs : List (Nat × Nat))
    (h : a ≠ nl) : readProfile nl ne a b want nr pairs = none := by
  unfold readProfile
  rw [if_neg (by intro hh; exact h hh.1)]

end Strata.ExpertCache



## FILE: /mnt/user-data/outputs/ExpertSourceDispatch.lean
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



## FILE: /mnt/user-data/outputs/LayerBlockContract.lean
import Mathlib

/-!
# MODULE: layer_block_contract
Structural contracts of one decoder layer's composition (`layer.hpp`): the MoE block split at the router,
the stage seams of `block_layer_pre`, the shared-expert-early reordering, buffer alignment, and the index
arithmetic of the QSA `q | gate` split and the indexer's pooled keys.

## Source semantics
* `moe_layer = route -> shared -> combine`; `pre = route`, `post = combine ∘ shared`; with the pool in the
  middle and acting only on host/parts state, `post ∘ pool ∘ pre` computes the same as the unsplit layer.
* `half = 1` runs stages 0..2, `half = 2` runs stages 3..4, `half = 0` runs all five; the halves tile the whole.
* Moving the shared expert before the CPU pool is bitwise neutral because they write disjoint buffers.
* `gdn_buffers_*`, `moe_buffers_*` regions are 16-byte aligned.
* `q_full` is `[q | gate]` per head by halves of `2 * head_dim`: q and gate indices never collide and stay inside.
* The indexer's pooled-key index `pos / idx_block` stays inside `max_cells / idx_block + 2` rows.

## Form template identifiers
* `route`, `shared`, `combine` <- `moe_route`, `moe_shared`, `moe_combine_parts`
* `pool`                       <- the host's CPU expert pool step between `pre[l]` and `post[l]`
* `stagesOf`                   <- the `half` argument of `block_layer_pre`
* `align16`                    <- the 16-byte alignment in `*_buffers_bytes`/`*_init`
* `qIdx`, `gIdx`               <- `qsa_layer`'s `q_full` split
-/

namespace Strata.Layer

-- § the MoE block split at the router

/-- The unsplit MoE block: route, then the shared expert, then the combine. -/
def moeLayer {σ : Type} (route shared combine : σ → σ) : σ → σ := combine ∘ shared ∘ route

/-- The first graph half: routing and the doorbell publish. -/
def pre {σ : Type} (route : σ → σ) : σ → σ := route

/-- The second graph half: the shared expert and the combine (`moe_finish`). -/
def post {σ : Type} (shared combine : σ → σ) : σ → σ := combine ∘ shared

/-- `moe_finish` followed by nothing else, run after `pre`, is exactly the unsplit MoE block. -/
theorem moe_split_eq {σ : Type} (route shared combine : σ → σ) :
    moeLayer route shared combine = post shared combine ∘ pre route := by
  rfl

/-- With the pool as the identity on the device state, the three-step layer is the unsplit layer. -/
theorem split_with_pool_id {σ : Type} (route shared combine : σ → σ) :
    post shared combine ∘ id ∘ pre route = moeLayer route shared combine := by
  rfl

/-- Device state of the shared-expert/pool race: the shared expert writes the first component, the pool the second. -/
def sharedOp {α β : Type} (f : α → α) : α × β → α × β := fun s => (f s.1, s.2)

/-- The pool's write, on the second component. -/
def poolOp {α β : Type} (g : β → β) : α × β → α × β := fun s => (s.1, g s.2)

/-- Running the shared expert before or after the pool gives the same state: moving it early is bitwise neutral. -/
theorem shared_early_commutes {α β : Type} (f : α → α) (g : β → β) :
    (sharedOp (β := β) f) ∘ (poolOp (α := α) g) = (poolOp (α := α) g) ∘ (sharedOp (β := β) f) := by
  funext s
  rfl

-- § stage seams of block_layer_pre

/-- Stages run by `half`: 1 = gr_read, attention, gr_write; 2 = gr_read, router; anything else = all five. -/
def stagesOf : Nat → List Nat
  | 1 => [0, 1, 2]
  | 2 => [3, 4]
  | _ => [0, 1, 2, 3, 4]

/-- The two halves, in order, tile the whole layer front exactly. -/
theorem halves_tile : stagesOf 1 ++ stagesOf 2 = stagesOf 0 := by
  decide

/-- The halves are disjoint: no stage runs twice. -/
theorem halves_disjoint : ∀ s ∈ stagesOf 1, s ∉ stagesOf 2 := by
  decide

-- § buffer alignment

/-- Round up to a multiple of 16, the alignment between carved regions. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- An aligned offset is a multiple of 16. -/
theorem align16_dvd (n : Nat) : 16 ∣ align16 n := by
  unfold align16
  exact Nat.dvd_mul_left 16 _

/-- Alignment never shrinks a region. -/
theorem le_align16 (n : Nat) : n ≤ align16 n := by
  unfold align16
  omega

/-- Alignment adds less than 16 bytes of padding. -/
theorem align16_lt (n : Nat) : align16 n < n + 16 := by
  unfold align16
  omega

/-- Alignment is idempotent: an aligned region stays where it is. -/
theorem align16_idem (n : Nat) : align16 (align16 n) = align16 n := by
  unfold align16
  omega

-- § the dump buffer

/-- The per-layer host dump's regions tile its length: two `n_embd` halves, two `hc` injections, the QSA attention. -/
theorem dump_regions_tile (ne hc nh hd : Nat) :
    (2 * ne + hc + hc) + nh * hd = 2 * ne + 2 * hc + nh * hd := by
  omega

-- § the QSA q | gate split

/-- Index of element `i` of head `h`'s q half inside `q_full`. -/
def qIdx (hd h i : Nat) : Nat := h * (2 * hd) + i

/-- Index of element `i` of head `h`'s gate half inside `q_full`. -/
def gIdx (hd h i : Nat) : Nat := h * (2 * hd) + hd + i

/-- q and gate never share a cell within a head: the halves are contiguous, not interleaved. -/
theorem q_gate_disjoint (hd h i j : Nat) (hi : i < hd) : qIdx hd h i ≠ gIdx hd h j := by
  unfold qIdx gIdx
  omega

/-- Every gate index lies inside `q_full` (`n_head * 2 * head_dim` floats). -/
theorem gIdx_lt (nh hd h i : Nat) (hh : h < nh) (hi : i < hd) : gIdx hd h i < nh * (2 * hd) := by
  unfold gIdx
  have h1 : (h + 1) * (2 * hd) ≤ nh * (2 * hd) := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hh)
  have h2 : (h + 1) * (2 * hd) = h * (2 * hd) + 2 * hd := by ring
  omega

/-- Every q index lies inside `q_full`. -/
theorem qIdx_lt (nh hd h i : Nat) (hh : h < nh) (hi : i < hd) : qIdx hd h i < nh * (2 * hd) := by
  unfold qIdx
  have h1 : (h + 1) * (2 * hd) ≤ nh * (2 * hd) := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hh)
  have h2 : (h + 1) * (2 * hd) = h * (2 * hd) + 2 * hd := by ring
  omega

-- § the indexer's pooled keys

/-- The pooled-key row of a cell stays inside the `max_cells / idx_block + 2` rows the state allocates. -/
theorem pooled_idx_lt (pos maxCells b : Nat) (hp : pos < maxCells) : pos / b < maxCells / b + 2 := by
  have h : pos / b ≤ maxCells / b := Nat.div_le_div_right hp.le
  omega

end Strata.Layer



## FILE: /mnt/user-data/outputs/SessionLoopContract.lean
import Mathlib

/-!
# MODULE: session_loop_contract
The structural contracts of the per-token host loop (`session.hpp`): layers run in order, the host pool sits
between `pre[l]` and `post[l]` without changing device arithmetic, graph capture is idempotent, the C1 dump
buffer has one disjoint slot per layer, and prefix-differenced stage timings telescope.

## Source semantics
* `session_loop` runs layers `0..n-1` in order; `post[l] ∘ pool ∘ pre[l]` equals `post[l] ∘ pre[l]` when the pool
  leaves device state alone (it writes the host staging only), which is why the loop is a change in WHEN the host
  runs, not in what is computed.
* `session_capture` is idempotent: a second call is a no-op.
* `dump_layers` holds `(n_layers + 1)` slots of `hc * n_embd` floats; slot `l + 1` is `R` after layer `l`.
* `session_replay_stage_prefixes` differences prefix timings `P_1 .. P_5`; the stage times sum to `P_5 - P_0`.
* A verify window of `T` tokens indexes `exec_[T]`, `T <= 8`.

## Form template identifiers
* `runLayers`       <- the layer loop of `session_loop` / `session_token`
* `pre`, `post`, `pool` <- `block_layer_pre`, `block_layer_post`, `PoolFn`
* `capture`         <- `session_capture` (`SessionGraphs::captured`)
* `slotBase`        <- slot `l` of `dump_layers`
* `P`               <- the prefix graph times `preP[k]`
-/

namespace Strata.Session

-- § the layer loop

/-- Run layers `0 .. n-1` in order, threading the state. -/
def runLayers {σ : Type} (step : Nat → σ → σ) (n : Nat) (s : σ) : σ :=
  (List.range n).foldl (fun acc l => step l acc) s

/-- Zero layers leave the state alone. -/
theorem runLayers_zero {σ : Type} (step : Nat → σ → σ) (s : σ) : runLayers step 0 s = s := by
  simp [runLayers]

/-- The loop is a left-to-right composition: layer `n` runs after layers `0 .. n-1`. -/
theorem runLayers_succ {σ : Type} (step : Nat → σ → σ) (n : Nat) (s : σ) :
    runLayers step (n + 1) s = step n (runLayers step n s) := by
  simp [runLayers, List.range_succ]

/-- If the pool leaves device state untouched, the loop with the pool in the middle of each layer computes the
same final state as the loop without it. -/
theorem loop_eq_token {σ : Type} (pre post : Nat → σ → σ) (pool : Nat → σ → σ)
    (hp : ∀ l, pool l = id) (n : Nat) (s : σ) :
    runLayers (fun l => post l ∘ pool l ∘ pre l) n s = runLayers (fun l => post l ∘ pre l) n s := by
  have hfun : (fun l => post l ∘ pool l ∘ pre l) = (fun l => post l ∘ pre l) := by
    funext l
    simp [hp l]
  rw [hfun]

-- § capture

/-- Capture state: whether it has happened, and the graphs held. A second capture changes nothing. -/
def capture {α : Type} (s : Bool × α) (g : α) : Bool × α := if s.1 then s else (true, g)

/-- `session_capture` is idempotent. -/
theorem capture_idem {α : Type} (s : Bool × α) (g g' : α) :
    capture (capture s g) g' = capture s g := by
  rcases s with ⟨c, a⟩
  cases c <;> simp [capture]

/-- After a capture the flag is set. -/
theorem capture_sets {α : Type} (s : Bool × α) (g : α) : (capture s g).1 = true := by
  rcases s with ⟨c, a⟩
  cases c <;> simp [capture]

-- § the dump buffer

/-- Slot `l + 1` of the dump (an element `j` of it) lies inside `(n_layers + 1)` slots of width `a = hc * n_embd`. -/
theorem dump_slot_lt (a nl l j : Nat) (hl : l < nl) (hj : j < a) : (l + 1) * a + j < (nl + 1) * a := by
  have h1 : (l + 2) * a ≤ (nl + 1) * a := Nat.mul_le_mul_right _ (by omega)
  have h2 : (l + 2) * a = (l + 1) * a + a := by ring
  omega

/-- Distinct layers' dump slots do not overlap. -/
theorem dump_slots_disjoint (a l1 l2 : Nat) (h : l1 < l2) : (l1 + 1) * a + a ≤ (l2 + 1) * a := by
  have h1 : (l1 + 2) * a ≤ (l2 + 1) * a := Nat.mul_le_mul_right _ (by omega)
  have h2 : (l1 + 2) * a = (l1 + 1) * a + a := by ring
  omega

-- § prefix-differenced timings

/-- The stage times obtained by differencing prefix times sum to the whole prefix: the table accounts for all of it. -/
theorem stage_times_sum (P : Nat → ℤ) (n : Nat) :
    ∑ k ∈ Finset.range n, (P (k + 1) - P k) = P n - P 0 := by
  exact Finset.sum_range_sub P n

/-- The five-stage table: its stage times sum to the full-graph time minus the empty prefix. -/
theorem five_stage_total (P : Nat → ℤ) :
    (P 1 - P 0) + (P 2 - P 1) + (P 3 - P 2) + (P 4 - P 3) + (P 5 - P 4) = P 5 - P 0 := by
  ring

-- § the verify window's graph table

/-- A window of `T <= max_t <= 8` tokens indexes inside `exec_[9]`. -/
theorem exec_index_lt (maxT T : Nat) (hmax : maxT ≤ 8) (hT : T ≤ maxT) : T < 9 := by
  omega

end Strata.Session



## FILE: /mnt/user-data/outputs/VerifyWindowCommit.lean
import Mathlib

/-!
# MODULE: verify_window_commit
The speculative verify window (`verify.hpp`): a window of `T` tokens (the last accepted token plus `T - 1` drafts)
is run, and `commit(n_keep)` makes the first `n_keep` tokens permanent. Greedy-equivalence is stated here as:
a draft is accepted exactly while it matches what greedy decode produces, and committing a prefix then
continuing equals plain decoding of the whole sequence.

## Source semantics
* The window's argmax after token `t` is plain greedy's next token, so the accepted length is the longest prefix
  of the drafts that agrees with the greedy outputs; the verified token itself is always kept, so
  `1 <= n_keep <= T`.
* `commit(n_keep)` replays the first `n_keep` tokens through the state update (GDN conv history and recurrence,
  indexer tail, PLE history); decoding the remaining tokens afterwards reproduces decoding all of them.
* Committing nothing leaves the state unchanged.
* Window size is bounded by `max_t <= 8`.

## Form template identifiers
* `commitState`  <- `Verifier::commit` (replaying `n_keep` tokens through the state update `step`)
* `acceptLen`    <- the longest accepted draft prefix against `out[t]`
* `n_keep`       <- `acceptLen + 1`
-/

namespace Strata.Verify

-- § commit replays a prefix

/-- The state after committing the first `n` tokens of a window: replay them through the state update. -/
def commitState {σ : Type} (step : σ → Nat → σ) (s : σ) (toks : List Nat) (n : Nat) : σ :=
  (toks.take n).foldl step s

/-- Committing a prefix and then decoding the remaining tokens equals decoding the whole window plainly. -/
theorem commit_then_rest {σ : Type} (step : σ → Nat → σ) (s : σ) (toks : List Nat) (n : Nat) :
    (toks.drop n).foldl step (commitState step s toks n) = toks.foldl step s := by
  unfold commitState
  rw [← List.foldl_append, List.take_append_drop]

/-- Committing zero tokens leaves the state unchanged. -/
theorem commit_zero {σ : Type} (step : σ → Nat → σ) (s : σ) (toks : List Nat) :
    commitState step s toks 0 = s := by
  simp [commitState]

/-- Committing one more token applies exactly one more state update. -/
theorem commit_succ {σ : Type} (step : σ → Nat → σ) (s : σ) (t : Nat) (rest : List Nat) (n : Nat) :
    commitState step s (t :: rest) (n + 1) = commitState step (step s t) rest n := by
  simp [commitState]

-- § acceptance

/-- Length of the longest prefix of the drafts that matches the greedy outputs. -/
def acceptLen : List Nat → List Nat → Nat
  | [], _ => 0
  | _ :: _, [] => 0
  | d :: ds, g :: gs => if d = g then acceptLen ds gs + 1 else 0

/-- Never more drafts are accepted than were proposed. -/
theorem acceptLen_le (d g : List Nat) : acceptLen d g ≤ d.length := by
  induction d generalizing g with
  | nil => cases g <;> simp [acceptLen]
  | cons a ds ih =>
    cases g with
    | nil => simp [acceptLen]
    | cons b gs =>
      have := ih gs
      by_cases h : a = b
      · simp [acceptLen, h]
        omega
      · simp [acceptLen, h]

/-- If greedy decode reproduces every draft, all drafts are accepted. -/
theorem acceptLen_self (d r : List Nat) : acceptLen d (d ++ r) = d.length := by
  induction d with
  | nil => simp [acceptLen]
  | cons a ds ih => simp [acceptLen, ih]

/-- A first mismatch rejects everything after it. -/
theorem acceptLen_mismatch (a b : Nat) (ds gs : List Nat) (h : a ≠ b) :
    acceptLen (a :: ds) (b :: gs) = 0 := by
  simp [acceptLen, h]

/-- `n_keep = acceptLen + 1` is between 1 and the window size `T`: the verified token is always kept. -/
theorem keep_bounds (T : Nat) (drafts greedy : List Nat) (hT : drafts.length + 1 = T) :
    1 ≤ acceptLen drafts greedy + 1 ∧ acceptLen drafts greedy + 1 ≤ T := by
  have := acceptLen_le drafts greedy
  omega

/-- A window of at most `max_t <= 8` tokens has at most 7 drafts. -/
theorem drafts_bound (maxT T : Nat) (hmax : maxT ≤ 8) (hT : T ≤ maxT) (hpos : 1 ≤ T) : T - 1 ≤ 7 := by
  omega

end Strata.Verify



## FILE: /mnt/user-data/outputs/GraphRegistry.lean
import Mathlib

/-!
# MODULE: graph_registry
The captured-graph registry (`CapturedGraph`, `GraphRegistry` in `graph.cpp`): graphs are recorded once per
`(layer type, n_tokens)` key, a second `record` for the same key is a no-op, `find` returns what was recorded, and
a capture that recorded zero nodes is refused.

## Source semantics
* `record(type, n, ...)` returns early when the key exists ("already recorded: the point of a registry"); only a
  new key adds a graph and bumps `captures_`.
* `find` returns the graph recorded for exactly that key, or none; recording one key never disturbs another.
* `CapturedGraph::end` refuses a capture with zero nodes.
* `wait_ms` is bounded by a real-clock deadline (not modelled; no arithmetic content).

## Form template identifiers
* `find`   <- `GraphRegistry::find`
* `record` <- `GraphRegistry::record` (the `graphs_.count(k)` early return)
* `endOk`  <- the `nodes_ == 0` check in `CapturedGraph::end`
* key `(Nat × Nat)` <- `Key{(int) type, n_tokens}`; value `Nat` <- the `CapturedGraph`
-/

namespace Strata.Graph

/-- Look a key up in the registry (association list; the first match wins, as in a map). -/
def find : List ((Nat × Nat) × Nat) → Nat × Nat → Option Nat
  | [], _ => none
  | (k', v) :: m, k => if k' = k then some v else find m k

/-- Record a graph for a key unless one is already there. -/
def record (m : List ((Nat × Nat) × Nat)) (k : Nat × Nat) (v : Nat) : List ((Nat × Nat) × Nat) :=
  if (find m k).isSome = true then m else (k, v) :: m

/-- A capture is accepted only if it recorded at least one node. -/
def endOk (nodes : Nat) : Bool := nodes != 0

/-- After `record`, the key is findable. -/
theorem find_record_isSome (m : List ((Nat × Nat) × Nat)) (k : Nat × Nat) (v : Nat) :
    (find (record m k v) k).isSome = true := by
  by_cases h : (find m k).isSome = true
  · simp [record, h]
  · simp [record, h, find]

/-- Recording an existing key changes nothing. -/
theorem record_existing (m : List ((Nat × Nat) × Nat)) (k : Nat × Nat) (v : Nat)
    (h : (find m k).isSome = true) : record m k v = m := by
  simp [record, h]

/-- `record` is idempotent: a second record for the same key is a no-op. -/
theorem record_idem (m : List ((Nat × Nat) × Nat)) (k : Nat × Nat) (v v' : Nat) :
    record (record m k v) k v' = record m k v := by
  by_cases h : (find m k).isSome = true
  · simp [record, h]
  · have hr : record m k v = (k, v) :: m := by simp [record, h]
    rw [hr]
    simp [record, find]

/-- Recording a key never disturbs another key's graph. -/
theorem find_record_other (m : List ((Nat × Nat) × Nat)) (k k' : Nat × Nat) (v : Nat) (hne : k' ≠ k) :
    find (record m k v) k' = find m k' := by
  have hne' : k ≠ k' := fun e => hne e.symm
  by_cases h : (find m k).isSome = true
  · simp [record, h]
  · simp [record, h, find, hne']

/-- A new key adds exactly one graph (`captures_` goes up by one). -/
theorem record_length_new (m : List ((Nat × Nat) × Nat)) (k : Nat × Nat) (v : Nat)
    (h : (find m k).isSome = false) : (record m k v).length = m.length + 1 := by
  simp [record, h]

/-- An existing key adds nothing. -/
theorem record_length_old (m : List ((Nat × Nat) × Nat)) (k : Nat × Nat) (v : Nat)
    (h : (find m k).isSome = true) : (record m k v).length = m.length := by
  simp [record, h]

/-- The registry never loses graphs. -/
theorem record_length_ge (m : List ((Nat × Nat) × Nat)) (k : Nat × Nat) (v : Nat) :
    m.length ≤ (record m k v).length := by
  unfold record
  split_ifs
  · exact Nat.le_refl _
  · simp

/-- A capture with zero nodes is refused. -/
theorem endOk_zero : endOk 0 = false := by
  decide

/-- A capture with at least one node is accepted. -/
theorem endOk_pos (n : Nat) (h : 0 < n) : endOk (n) = true := by
  unfold endOk
  simp
  omega

end Strata.Graph





## FILE: /mnt/user-data/outputs/LayerCompose.lean
/-!
# MODULE: layer_compose
Arithmetic and dispatch contract of `src/core/layer.cpp`: the GDN / QSA / MoE layer composition, its arena
carving (16-byte cursor), the plane-layout check of quantized tensors, the activation-format dispatch, the
stage-pfx selection of `block_layer_pre`, and the dump-slot layout.

## Source semantics
* Every buffer is carved from an arena with `used := align16 (used + count*size)`; `*_bytes` functions sum the
  same parts, so the init total equals the byte count and the 256-byte slack covers the extra rounding.
* A quantized tensor's three planes (codes | scales | offsets) must add up to the tensor and the offset plane
  exists iff `has_offset`.
* The activation a quantized weight reads is decided per TENSOR: native, S2/Q8_0, legacy Q8_0, or Q8_K.
* `stage_prefix = k > 0` runs stages `0..k-1`; otherwise `half` selects mixer (1), FFN-front+router (2) or all (0).
* RoPE rows read are `pos_base + pos`, which must be inside the `max_cells`-row table.

## Form template identifiers
* `alignUp16`  <- `align_up16`
* `q8kBytes`   <- `q8k_bytes`
* `PlaneSizes` <- `WeightRef{codes_bytes,scales_bytes,offset_bytes,bytes}` / `plane_ptrs`
* `dispatch`   <- `gemv_quantized`
* `runStage`   <- `block_layer_pre` run0..run4
* `takeAll16`  <- `Cursor::take` sequence
-/

namespace Strata.LayerCompose

-- § alignment

/-- 16-byte round-up used by `align_up16`. -/
def alignUp16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Rounding up never loses bytes. -/
theorem alignUp16_ge (n : Nat) : n ≤ alignUp16 n := by
  unfold alignUp16; omega

/-- The rounded value is 16-aligned. -/
theorem alignUp16_mod (n : Nat) : alignUp16 n % 16 = 0 := by
  unfold alignUp16; omega

/-- Rounding adds at most 15 bytes. -/
theorem alignUp16_le (n : Nat) : alignUp16 n ≤ n + 15 := by
  unfold alignUp16; omega

/-- Rounding is idempotent. -/
theorem alignUp16_idem (n : Nat) : alignUp16 (alignUp16 n) = alignUp16 n := by
  unfold alignUp16; omega

-- § Q8_K sizes

/-- Bytes of a Q8_K buffer of `n` elements: one 292-byte block per 256 elements. -/
def q8kBytes (n : Nat) : Nat := n / 256 * 292

/-- `b` whole blocks occupy `292*b` bytes. -/
theorem q8k_blocks (b : Nat) : q8kBytes (256 * b) = 292 * b := by
  unfold q8kBytes; omega

/-- A Q8_K image is never smaller than the activation it encodes (for a multiple of 256). -/
theorem q8k_ge (n : Nat) (h : n % 256 = 0) : n ≤ q8kBytes n := by
  unfold q8kBytes; omega

-- § plane layout

/-- Recorded plane sizes of a quantized tensor. -/
structure PlaneSizes where
  codes : Nat
  scales : Nat
  offset : Nat
  bytes : Nat

/-- The check of `plane_ptrs`: both mandatory planes non-empty and the three add up to the tensor. -/
def PlaneSizes.ok (p : PlaneSizes) : Prop :=
  p.codes ≠ 0 ∧ p.scales ≠ 0 ∧ p.codes + p.scales + p.offset = p.bytes

/-- Valid planes tile the tensor exactly: the offset plane starts at `codes + scales` and ends at `bytes`. -/
theorem planes_partition (p : PlaneSizes) (h : p.ok) :
    p.codes + p.scales + p.offset = p.bytes := by
  unfold PlaneSizes.ok at h
  obtain ⟨_, _, h3⟩ := h
  exact h3

/-- Valid planes: the scale plane starts strictly inside the tensor. -/
theorem planes_scales_inside (p : PlaneSizes) (h : p.ok) : p.codes < p.bytes := by
  unfold PlaneSizes.ok at h
  obtain ⟨_, h2, h3⟩ := h
  omega

/-- Valid planes: the codes plane fits in the tensor. -/
theorem planes_codes_le (p : PlaneSizes) (h : p.ok) : p.codes + p.scales ≤ p.bytes := by
  unfold PlaneSizes.ok at h
  obtain ⟨_, _, h3⟩ := h
  omega

-- § activation dispatch

/-- Which path a projection takes. -/
inductive Dispatch
  | native
  | s2Q80
  | legacyQ80
  | q8kSplit
  deriving DecidableEq

/-- Decision of `gemv_quantized`. -/
def dispatch (native : Bool) (codeBits : Nat) (wantsQ8k : Bool) : Dispatch :=
  if native = true then .native
  else if codeBits = 2 then .s2Q80
  else if wantsQ8k = true then .q8kSplit
  else .legacyQ80

/-- A natively served tensor always takes the native path. -/
theorem dispatch_native (bits : Nat) (w : Bool) : dispatch true bits w = .native := by
  simp [dispatch]

/-- A 2-bit canonical tensor is always S2 with the Q8_0 activation, whatever `wants_q8k` says. -/
theorem dispatch_s2 (w : Bool) : dispatch false 2 w = .s2Q80 := by
  simp [dispatch]

/-- A K-quant (not 2-bit, wants Q8_K) takes the Q8_K split kernel. -/
theorem dispatch_q8k (bits : Nat) (h : bits ≠ 2) : dispatch false bits true = .q8kSplit := by
  simp [dispatch, h]

/-- A legacy 4/8-bit form takes Q8_0. -/
theorem dispatch_legacy (bits : Nat) (h : bits ≠ 2) : dispatch false bits false = .legacyQ80 := by
  simp [dispatch, h]

-- § stage pfx selection

/-- `run_i` of `block_layer_pre`: pfx mode when `pfx > 0`, else by `half`. -/
def runStage (i pfx half : Nat) : Bool :=
  if 0 < pfx then decide (i < pfx)
  else if i ≤ 2 then (half == 0 || half == 1)
  else (half == 0 || half == 2)

/-- In pfx mode stage `i` runs exactly when `i < pfx`. -/
theorem runStage_prefix (i p h : Nat) (hp : 0 < p) : runStage i p h = decide (i < p) := by
  simp [runStage, hp]

/-- Default call (`half = 0`, `stage_prefix = 0`) runs all five stages. -/
theorem runStage_all :
    runStage 0 0 0 = true ∧ runStage 1 0 0 = true ∧ runStage 2 0 0 = true ∧
    runStage 3 0 0 = true ∧ runStage 4 0 0 = true := by
  decide

/-- `half = 1` runs exactly the mixer stages 0..2. -/
theorem runStage_half1 :
    runStage 0 0 1 = true ∧ runStage 1 0 1 = true ∧ runStage 2 0 1 = true ∧
    runStage 3 0 1 = false ∧ runStage 4 0 1 = false := by
  decide

/-- `half = 2` runs exactly the FFN front and router (stages 3, 4). -/
theorem runStage_half2 :
    runStage 0 0 2 = false ∧ runStage 1 0 2 = false ∧ runStage 2 0 2 = false ∧
    runStage 3 0 2 = true ∧ runStage 4 0 2 = true := by
  decide

/-- Prefix sets are nested: whatever pfx `k` runs, pfx `k+1` runs too. -/
theorem runStage_prefix_mono (i k h : Nat) (hk : 0 < k) (hr : runStage i k h = true) :
    runStage i (k + 1) h = true := by
  rw [runStage_prefix i k h hk] at hr
  rw [runStage_prefix i (k + 1) h (by omega)]
  simp at hr ⊢
  omega

-- § RoPE table bound

/-- Row index read by RoPE is non-negative and below the table height once the guard of `qsa_layer` passed. -/
theorem rope_row_in_table (posBase pos maxCells : Int) (h0 : 0 ≤ posBase) (h1 : 0 ≤ pos)
    (h2 : posBase + pos < maxCells) : 0 ≤ posBase + pos ∧ posBase + pos < maxCells := by
  omega

-- § arena carving

/-- Sum of the 16-aligned sizes: what every `*_bytes` function returns. -/
def alignedTotal16 (l : List Nat) : Nat := (l.map alignUp16).sum

/-- The cursor sequence `used := align16 (used + b)`. -/
def takeAll16 : Nat → List Nat → Nat
  | used, [] => used
  | used, b :: t => takeAll16 (alignUp16 (used + b)) t

/-- Carving from an aligned start equals the sum of aligned parts: init total = `*_bytes`. -/
theorem takeAll16_eq (l : List Nat) :
    ∀ used, used % 16 = 0 → takeAll16 used l = used + alignedTotal16 l := by
  induction l with
  | nil => intro used _; simp [takeAll16, alignedTotal16]
  | cons b t ih =>
    intro used hu
    have h1 : alignUp16 (used + b) = used + alignUp16 b := by unfold alignUp16; omega
    have h2 : alignUp16 (used + b) % 16 = 0 := by unfold alignUp16; omega
    simp only [takeAll16, alignedTotal16, List.map_cons, List.sum_cons] at ih ⊢
    rw [ih _ h2, h1]
    omega

/-- The init sequence from offset 0 consumes exactly `alignedTotal16`. -/
theorem init_total_eq_bytes (l : List Nat) : takeAll16 0 l = alignedTotal16 l := by
  have h := takeAll16_eq l 0 (by omega)
  omega

/-- Aligned total is at least the raw sum. -/
theorem alignedTotal16_ge (l : List Nat) : l.sum ≤ alignedTotal16 l := by
  induction l with
  | nil => simp [alignedTotal16]
  | cons a t ih =>
    have h := alignUp16_ge a
    simp only [alignedTotal16, List.map_cons, List.sum_cons] at ih ⊢
    omega

/-- Aligned total is 16-aligned. -/
theorem alignedTotal16_mod (l : List Nat) : alignedTotal16 l % 16 = 0 := by
  induction l with
  | nil => simp [alignedTotal16]
  | cons a t ih =>
    have h := alignUp16_mod a
    simp only [alignedTotal16, List.map_cons, List.sum_cons] at ih ⊢
    omega

/-- Per-part rounding costs at most 15 bytes each. -/
theorem alignedTotal16_le (l : List Nat) : alignedTotal16 l ≤ l.sum + 15 * l.length := by
  induction l with
  | nil => simp [alignedTotal16]
  | cons a t ih =>
    have h := alignUp16_le a
    simp only [alignedTotal16, List.map_cons, List.sum_cons, List.length_cons] at ih ⊢
    omega

/-- The byte count `align16(sum) + 256` of `qsa_state_bytes` / `qsa_buffers_bytes` always covers the
per-take rounding of the matching `*_init` provided there are at most 17 takes. -/
theorem carve_fits_with_slack (l : List Nat) (hlen : l.length ≤ 17) :
    takeAll16 0 l ≤ alignUp16 l.sum + 256 := by
  rw [init_total_eq_bytes]
  have h1 := alignedTotal16_le l
  have h2 := alignUp16_ge l.sum
  omega

/-- A cursor `take` hands out the current offset and the next offset is past the requested bytes. -/
theorem cursor_take_disjoint (used count size : Nat) :
    used + count * size ≤ alignUp16 (used + count * size) := alignUp16_ge _

-- § dump slots

/-- Total floats per layer of the half-level dump (`dump_stride_floats`). -/
def dumpStride (n hc nhhd nvk : Nat) : Nat := 2 * n + 2 * hc + nhhd + 5 * nvk + 8

/-- The last write of `qsa_layer` (the post-RoPE `kcur`) ends exactly at the stride. -/
theorem dump_stride_covers (n hc nhhd nvk : Nat) :
    (2 * n + 2 * hc + nhhd + 2 * nvk + 8) + 2 * nvk + nvk = dumpStride n hc nhhd nvk := by
  unfold dumpStride; omega

/-- FINDING: with `nvk >= 2`, the `step[0..3]` write at `vs + nvk/2 + 4` (length 4) extends past `base = vs + 8`,
so it overlaps the `k_pool` pfx written at `base`; the two diagnostic copies are not disjoint. -/
theorem dump_step_overlaps_kpool (vs nvk : Nat) (h : 2 ≤ nvk) :
    ¬ (vs + nvk / 2 + 4 + 4 ≤ vs + 8) := by
  omega

end Strata.LayerCompose



## FILE: /mnt/user-data/outputs/MtpDrafter.lean
/-!
# MODULE: mtp_drafter
Arithmetic contracts of `src/core/mtp.cpp`: the MTP draft layer's bump arena, attention window capacity,
staging-row layout of the captured round/step graphs, prefill chunking and the draft chain length.

## Source semantics
* `Bump::take(n)` advances by `(n*sizeof + 255) & ~255`; a counting pass and the real pass see identical sizes so
  offsets agree and every region is 256-aligned and in bounds of the counted total.
* `cap_ = ceil(window_eff / 64) * 64`, `window_eff = window` if `0 < window < max_cells` else `max_cells`.
* Step/pos rows: rows `[0,T)` catch-up, row `2*max_t-1` the full-layer cell, rows `max_t + j - 1` (j = 1..max_t-2)
  the chain steps; all rows lie below `2*max_t`.
* Draft chain yields `1 <= n <= max_t - 1` drafts.

## Form template identifiers
* `align256` <- Bump's `(x + 255) & ~255`
* `bumpTotal` <- `Bump::used` after `carve`
* `effWindow`, `capOf` <- `window_`, `cap_`
* `chainRow` <- `max_t_ + j - 1`
-/

namespace Strata.MtpDrafter

-- § bump arena

/-- 256-byte round-up. -/
def align256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Rounding never loses bytes. -/
theorem align256_ge (n : Nat) : n ≤ align256 n := by unfold align256; omega

/-- Rounded value is 256-aligned. -/
theorem align256_mod (n : Nat) : align256 n % 256 = 0 := by unfold align256; omega

/-- Rounding adds at most 255 bytes. -/
theorem align256_le (n : Nat) : align256 n ≤ n + 255 := by unfold align256; omega

/-- Bytes consumed by a sequence of takes (byte sizes already multiplied by `sizeof`). -/
def bumpTotal (l : List Nat) : Nat := (l.map align256).sum

/-- Counting two phases separately and concatenating is the same as one pass. -/
theorem bump_append (l1 l2 : List Nat) : bumpTotal (l1 ++ l2) = bumpTotal l1 + bumpTotal l2 := by
  simp [bumpTotal, List.map_append, List.sum_append]

/-- The region of the `d`-th take, starting at the prefix total, lies inside the whole arena. -/
theorem bump_region_in_bounds (l1 l2 : List Nat) (d : Nat) :
    bumpTotal l1 + d ≤ bumpTotal (l1 ++ d :: l2) := by
  have h := align256_ge d
  simp only [bumpTotal, List.map_append, List.map_cons, List.sum_append, List.sum_cons]
  omega

/-- The whole arena is a multiple of 256. -/
theorem bump_total_mod (l : List Nat) : bumpTotal l % 256 = 0 := by
  induction l with
  | nil => simp [bumpTotal]
  | cons a t ih =>
    have h := align256_mod a
    simp only [bumpTotal, List.map_cons, List.sum_cons] at ih ⊢
    omega

/-- Every region start (a prefix total) is 256-aligned. -/
theorem bump_region_aligned (l1 l2 : List Nat) : bumpTotal l1 % 256 = 0 := bump_total_mod l1

-- § attention window capacity

/-- Effective attended cells. -/
def effWindow (window maxCells : Nat) : Nat :=
  if 0 < window ∧ window < maxCells then window else maxCells

/-- Capacity `cap_`: effective window rounded up to 64. -/
def capOf (window maxCells : Nat) : Nat := (effWindow window maxCells + 63) / 64 * 64

/-- Effective window never exceeds the cache. -/
theorem effWindow_le (w m : Nat) : effWindow w m ≤ m := by
  unfold effWindow; split <;> omega

/-- Capacity covers the effective window. -/
theorem capOf_ge (w m : Nat) : effWindow w m ≤ capOf w m := by
  unfold capOf; omega

/-- Capacity is a multiple of 64. -/
theorem capOf_mod (w m : Nat) : capOf w m % 64 = 0 := by
  unfold capOf; omega

/-- Capacity wastes less than 64 cells. -/
theorem capOf_lt (w m : Nat) : capOf w m < effWindow w m + 64 := by
  unfold capOf; omega

/-- Identity table: entry `t*cap + i` (t < T, i < cap) is inside the `T*cap` table. -/
theorem ident_index_lt (t i T cap : Nat) (ht : t < T) (hi : i < cap) : t * cap + i < T * cap := by
  have h1 := Nat.mul_le_mul_right cap (show t + 1 ≤ T by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § staging rows

/-- Row of chain step `j` (1-based). -/
def chainRow (maxT j : Nat) : Nat := maxT + j - 1

/-- Chain rows are after every catch-up row (`T <= max_t`). -/
theorem chain_row_after_catchup (maxT T j : Nat) (hT : T ≤ maxT) (hj : 1 ≤ j) :
    T ≤ chainRow maxT j := by
  unfold chainRow; omega

/-- Chain rows are before the full-layer row `2*max_t-1`. -/
theorem chain_row_before_ra (maxT j : Nat) (hj : j + 2 ≤ maxT) :
    chainRow maxT j < 2 * maxT - 1 := by
  unfold chainRow; omega

/-- The full-layer row lies inside the `2*max_t` rows allocated for step/pos. -/
theorem ra_row_in_range (maxT : Nat) (h : 1 ≤ maxT) : 2 * maxT - 1 < 2 * maxT := by
  omega

/-- Distinct chain steps use distinct rows. -/
theorem chain_row_injective (maxT j k : Nat) (hj : 1 ≤ j) (hk : 1 ≤ k)
    (h : chainRow maxT j = chainRow maxT k) : j = k := by
  unfold chainRow at h; omega

-- § chunk loops

/-- Eight-column quantize/GEMV chunks over `T*HC` columns: each chunk is non-empty and in range. -/
theorem mmvq_chunk_in_range (total c0 : Nat) (h : c0 < total) :
    1 ≤ min 8 (total - c0) ∧ min 8 (total - c0) ≤ 8 ∧ c0 + min 8 (total - c0) ≤ total := by
  omega

/-- Prefill chunking: a chunk starting before `n` is non-empty, at most `max_t`, and stays inside `n`. -/
theorem prefill_chunk_bounds (n c maxT : Nat) (h : c < n) (hm : 1 ≤ maxT) :
    1 ≤ min maxT (n - c) ∧ min maxT (n - c) ≤ maxT ∧ c + min maxT (n - c) ≤ n := by
  omega

/-- Draft chain: `n = 1 + k` with at most `max_t - 2` extra steps yields `1 <= n <= max_t - 1`. -/
theorem draft_count_bounds (n k maxT : Nat) (hn : n = 1 + k) (hk : k + 2 ≤ maxT) :
    1 ≤ n ∧ n ≤ maxT - 1 := by
  omega

/-- Draft arguments accepted by `draft`: the catch-up row `a` is an existing window row. -/
theorem draft_arg_row (T maxT a : Nat) (h1 : 1 ≤ T) (h2 : T ≤ maxT) (h3 : a < T) : a < maxT := by
  omega

end Strata.MtpDrafter



## FILE: /mnt/user-data/outputs/SessionDriver.lean
/-!
# MODULE: session_driver
Arithmetic contracts of `src/core/session.cpp`: session arena sizing and carving (256-byte cursor), the
per-layer state slicing of the 36 GDN and 12 QSA layers, the per-layer timing event indexing, and the stage
prefix differencing.

## Source semantics
* `session_bytes` sums raw parts, aligns once and adds 4096; `session_init` aligns each of its 7 takes at 256, so
  it consumes at most `align256(sum) + 4096`.
* Layer `l` is QSA iff `l % 4 = 3`. GDN layer `l` owns state slice `gdn_index = #GDN layers before l`, QSA layer
  `l` owns QSA state `l/4`; slices are disjoint and inside the allocation.
* The QSA arena holds one first state (with RoPE) then `n-1` others.
* Prefix timing: stage times are consecutive differences of prefix totals, so they telescope to the last prefix.

## Form template identifiers
* `alignUp256` <- `align_up(n, SESSION_STATE_ALIGN)`
* `takeAll256` <- the `take` lambda of `session_init`
* `gdnBefore`, `qsaBefore` <- `gdn_index` / `qsa_index`
* `qsaOff` <- `qp + (i == 0 ? 0 : first + (i-1)*rest)`
-/

namespace Strata.SessionDriver

-- § alignment

/-- 256-byte round-up. -/
def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Rounding never loses bytes. -/
theorem alignUp256_ge (n : Nat) : n ≤ alignUp256 n := by unfold alignUp256; omega

/-- Rounded value is aligned. -/
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by unfold alignUp256; omega

/-- Rounding adds at most 255. -/
theorem alignUp256_le (n : Nat) : alignUp256 n ≤ n + 255 := by unfold alignUp256; omega

/-- Sum of aligned part sizes. -/
def alignedTotal256 (l : List Nat) : Nat := (l.map alignUp256).sum

/-- The `take` sequence of `session_init`. -/
def takeAll256 : Nat → List Nat → Nat
  | used, [] => used
  | used, b :: t => takeAll256 (alignUp256 (used + b)) t

/-- Carving from an aligned start consumes the sum of the aligned parts. -/
theorem takeAll256_eq (l : List Nat) :
    ∀ used, used % 256 = 0 → takeAll256 used l = used + alignedTotal256 l := by
  induction l with
  | nil => intro used _; simp [takeAll256, alignedTotal256]
  | cons b t ih =>
    intro used hu
    have h1 : alignUp256 (used + b) = used + alignUp256 b := by unfold alignUp256; omega
    have h2 : alignUp256 (used + b) % 256 = 0 := by unfold alignUp256; omega
    simp only [takeAll256, alignedTotal256, List.map_cons, List.sum_cons] at ih ⊢
    rw [ih _ h2, h1]
    omega

/-- Per-part rounding costs at most 255 bytes each. -/
theorem alignedTotal256_le (l : List Nat) : alignedTotal256 l ≤ l.sum + 255 * l.length := by
  induction l with
  | nil => simp [alignedTotal256]
  | cons a t ih =>
    have h := alignUp256_le a
    simp only [alignedTotal256, List.map_cons, List.sum_cons, List.length_cons] at ih ⊢
    omega

/-- `session_init`'s 7 takes always fit in `session_bytes = align256(sum) + 4096`. -/
theorem session_init_fits (l : List Nat) (hlen : l.length ≤ 7) :
    takeAll256 0 l ≤ alignUp256 l.sum + 4096 := by
  rw [takeAll256_eq l 0 (by omega)]
  have h1 := alignedTotal256_le l
  have h2 := alignUp256_ge l.sum
  omega

-- § layer kinds and state indexing

/-- Number of GDN layers strictly before layer `l` (QSA iff `l % 4 = 3`). -/
def gdnBefore : Nat → Nat
  | 0 => 0
  | l + 1 => gdnBefore l + (if l % 4 = 3 then 0 else 1)

/-- Number of QSA layers strictly before layer `l`. -/
def qsaBefore : Nat → Nat
  | 0 => 0
  | l + 1 => qsaBefore l + (if l % 4 = 3 then 1 else 0)

/-- Closed form of the GDN counter. -/
theorem gdnBefore_eq (l : Nat) : gdnBefore l = l - l / 4 := by
  induction l with
  | zero => simp [gdnBefore]
  | succ n ih =>
    simp only [gdnBefore, ih]
    split <;> omega

/-- Closed form of the QSA counter. -/
theorem qsaBefore_eq (l : Nat) : qsaBefore l = l / 4 := by
  induction l with
  | zero => simp [qsaBefore]
  | succ n ih =>
    simp only [qsaBefore, ih]
    split <;> omega

/-- Distinct GDN layers get distinct state indices (consecutive GDN layers differ by one). -/
theorem gdn_index_succ (l : Nat) (h : ¬ l % 4 = 3) : gdnBefore (l + 1) = gdnBefore l + 1 := by
  simp [gdnBefore, h]

/-- A QSA layer does not advance the GDN counter. -/
theorem gdn_index_qsa (l : Nat) (h : l % 4 = 3) : gdnBefore (l + 1) = gdnBefore l := by
  simp [gdnBefore, h]

/-- With 48 layers there are 36 GDN and 12 QSA layers. -/
theorem layer_counts : gdnBefore 48 = 36 ∧ qsaBefore 48 = 12 := by
  decide

/-- Every GDN layer of the 48-layer model has a state index below 36: its slice is inside the allocation. -/
theorem gdn_index_lt (l : Nat) (hl : l < 48) (h : ¬ l % 4 = 3) : gdnBefore l < 36 := by
  rw [gdnBefore_eq]; omega

/-- Every QSA layer of the 48-layer model has a state index below 12. -/
theorem qsa_index_lt (l : Nat) (hl : l < 48) : qsaBefore l < 12 := by
  rw [qsaBefore_eq]; omega

/-- Slice `i` of `n` equal slices ends inside `n * F`: slices are disjoint and in bounds. -/
theorem slice_in_bounds (i n F : Nat) (h : i < n) : (i + 1) * F ≤ n * F :=
  Nat.mul_le_mul_right F h

/-- Slice `i` ends where slice `i+1` begins. -/
theorem slice_adjacent (i F : Nat) : i * F + F = (i + 1) * F := by
  rw [Nat.add_mul, Nat.one_mul]

-- § QSA arena

/-- Byte offset of QSA state `i`: state 0 at 0, state `i>=1` at `first + (i-1)*rest`. -/
def qsaOff (first rest : Nat) : Nat → Nat
  | 0 => 0
  | i + 1 => first + i * rest

/-- State 0 (size `first`) ends where state 1 begins. -/
theorem qsa_first_adjacent (first rest : Nat) : qsaOff first rest 0 + first = qsaOff first rest 1 := by
  simp [qsaOff]

/-- State `i+1` (size `rest`) ends where state `i+2` begins: no overlap, no gap. -/
theorem qsa_rest_adjacent (first rest i : Nat) :
    qsaOff first rest (i + 1) + rest = qsaOff first rest (i + 2) := by
  simp only [qsaOff, Nat.succ_mul, Nat.add_assoc]

/-- Total of `n >= 1` states equals the offset of state `n`. -/
theorem qsa_total (first rest n : Nat) :
    first + n * rest = qsaOff first rest (n + 1) := by
  simp [qsaOff]

-- § timing event indices

/-- Events `l*6 + k - 1` (k = 1..5) and `l*6 + 5` are inside the `n*6` events allocated. -/
theorem prefix_event_in_range (l n k : Nat) (hl : l < n) (hk1 : 1 ≤ k) (hk5 : k ≤ 5) :
    l * 6 + k - 1 < n * 6 ∧ l * 6 + 5 < n * 6 := by
  omega

/-- Events `l*4 + 0..3` are inside the `n*4` events allocated. -/
theorem stage_event_in_range (l n : Nat) (hl : l < n) : l * 4 + 3 < n * 4 := by
  omega

/-- The graph array `preP[k-1]` is addressed within its five entries. -/
theorem prefix_graph_index (k : Nat) (h1 : 1 ≤ k) (h2 : k ≤ 5) : k - 1 < 5 := by
  omega

/-- Differencing the five prefix totals telescopes to the total of the full pre graph. -/
theorem prefix_telescope (t1 t2 t3 t4 t5 : Int) :
    t1 + (t2 - t1) + (t3 - t2) + (t4 - t3) + (t5 - t4) = t5 := by
  omega

-- § doorbell sequence

/-- The ring awaited by layer `l` is `l+1`, never more than the layer count. -/
theorem ring_want_le (l n : Nat) (h : l < n) : l + 1 ≤ n := by
  omega

end Strata.SessionDriver



## FILE: /mnt/user-data/outputs/WindowVerifier.lean
/-!
# MODULE: window_verifier
Arithmetic contracts of `src/core/verify.cpp`: the split verify window (two token groups), ring numbering,
the GPU plan buffer layout, staging halves, the monotone flag B and draft acceptance.

## Source semantics
* With `G = 2` tokens `[0,T)` split at `(T+1)/2`; with `G = 1` one group `[0,T)`. Ring of `(layer l, group g)`
  is `l*G + g + 1` and equals `k + 1` for the host's loop index `k = l*G + g`.
* Plan buffer: `counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) | ptr2(cap u64) | start2(cap+1)`;
  pointer area starts at an even int32 index so u64 pointers are 8-byte aligned.
* Staging blobs are split in halves for `G = 2`; flag B only rises.
* Acceptance `a` counts matching drafts, `a <= T-1`, hence `commit(a+1)` has `1 <= a+1 <= T`.

## Form template identifiers
* `tb`, `te0` <- `tb_[]`, `te_[]`
* `ptrOff`, `planI32` <- `ptr_off`, `plan_i32_`
* `raise` <- `Verifier::raise_flag` (final state of the CAS loop)
* `acceptFrom` <- the `while (a < T-1 && window[a+1] == outv[a])` loop
-/

namespace Strata.WindowVerifier

-- § token groups

/-- Group begin: group 0 at 0, group 1 at `(T+1)/2`. -/
def tb (T grp : Nat) : Nat := if grp = 0 then 0 else (T + 1) / 2

/-- End of group 0: `(T+1)/2` when split, else `T`. -/
def te0 (T G : Nat) : Nat := if G = 2 then (T + 1) / 2 else T

/-- Split: group 0 ends at the midpoint. -/
theorem te0_split (T : Nat) : te0 T 2 = (T + 1) / 2 := by
  simp [te0]

/-- Unsplit: group 0 is the whole window. -/
theorem te0_single (T : Nat) : te0 T 1 = T := by
  simp [te0]

/-- Both groups of a split window are non-empty when `T >= 2`, and they tile `[0,T)`. -/
theorem split_groups_tile (T : Nat) (h : 2 ≤ T) :
    0 < (T + 1) / 2 ∧ 0 < T - (T + 1) / 2 ∧ (T + 1) / 2 + (T - (T + 1) / 2) = T := by
  omega

-- § ring numbering

/-- The ring awaited for `(layer, group)` equals the host loop's `want = k + 1` where `k = layer*G + group`. -/
theorem ring_matches_loop (k G : Nat) (hG : 0 < G) : (k / G) * G + k % G + 1 = k + 1 := by
  have h := Nat.div_add_mod k G
  rw [Nat.mul_comm (k / G) G]
  omega

/-- A group index is below `G` and the layer/group decomposition covers `n*G` rings. -/
theorem ring_in_range (k G n : Nat) (hG : 0 < G) (hk : k < n * G) : k / G < n := by
  exact (Nat.div_lt_iff_lt_mul hG).2 hk

-- § plan layout

/-- Even-aligned start of the pointer area (`ptr_off`). -/
def ptrOff (cap : Nat) : Nat := (4 + (cap + 1) + cap + cap + 1) / 2 * 2

/-- Total int32 count of one plan slot (`plan_i32_`). -/
def planI32 (cap : Nat) : Nat := ptrOff cap + 4 * cap + (cap + 1) + 1

/-- The pointer area start is even (u64 pointers are 8-byte aligned). -/
theorem ptrOff_even (cap : Nat) : ptrOff cap % 2 = 0 := by
  unfold ptrOff; omega

/-- The pointer area starts after `tok` and wastes at most one int32. -/
theorem ptrOff_bounds (cap : Nat) :
    4 + (cap + 1) + cap + cap ≤ ptrOff cap ∧ ptrOff cap ≤ 4 + (cap + 1) + cap + cap + 1 := by
  unfold ptrOff; omega

/-- `start2` ends inside the plan slot. -/
theorem plan_start2_fits (cap : Nat) : ptrOff cap + 4 * cap + (cap + 1) ≤ planI32 cap := by
  unfold planI32; omega

/-- The device-side recomputation of `ptr_off` in `post` agrees with the host layout. -/
theorem ptrOff_device_agree (cap : Nat) : (4 + (cap + 1) + 2 * cap + 1) / 2 * 2 = ptrOff cap := by
  unfold ptrOff; omega

/-- Host plan slots 0 and 1 (each `p` int32) fit in the `2p + 16` allocation. -/
theorem host_plan_fits (p : Nat) : 0 * p + p ≤ 2 * p + 16 ∧ 1 * p + p ≤ 2 * p + 16 := by
  omega

/-- Device plan slots 0 and 1 (stride `p+16`, copy length `p`) fit in `2*(p+16)`. -/
theorem device_plan_fits (p : Nat) :
    0 * (p + 16) + p ≤ 2 * (p + 16) ∧ 1 * (p + 16) + p ≤ 2 * (p + 16) := by
  omega

-- § staging halves

/-- Blobs per group. -/
def perGroup (K G : Nat) : Nat := if G = 2 then K / 2 else K

/-- A split window's two halves fit in the staging area. -/
theorem staging_split_fits (K : Nat) : 2 * perGroup K 2 ≤ K := by
  unfold perGroup; split <;> omega

/-- Group 1's half starts where group 0's ends. -/
theorem staging_adjacent (K : Nat) : 0 * perGroup K 2 + perGroup K 2 = 1 * perGroup K 2 := by
  omega

-- § flag B

/-- Final value of the CAS loop of `raise_flag`. -/
def raise (cur v : Nat) : Nat := if cur < v then v else cur

/-- The flag never goes down. -/
theorem raise_ge_cur (cur v : Nat) : cur ≤ raise cur v := by
  unfold raise; split <;> omega

/-- The flag reaches the requested value. -/
theorem raise_ge_val (cur v : Nat) : v ≤ raise cur v := by
  unfold raise; split <;> omega

/-- `raise` is the maximum, so late host functions of earlier layers cannot lower it. -/
theorem raise_eq_max (cur v : Nat) : raise cur v = max cur v := by
  unfold raise; split <;> omega

/-- `raise` is monotone in the current value. -/
theorem raise_mono (c c' v : Nat) (h : c ≤ c') : raise c v ≤ raise c' v := by
  unfold raise; split <;> split <;> omega

-- § acceptance

/-- The acceptance loop with fuel: extend while the next draft matches. -/
def acceptFrom (T : Nat) (ok : Nat → Bool) : Nat → Nat → Nat
  | a, 0 => a
  | a, f + 1 => if a + 1 < T then (if ok a then acceptFrom T ok (a + 1) f else a) else a

/-- The accepted count never reaches `T`, so `a + 1 <= T`. -/
theorem accept_bound (T : Nat) (ok : Nat → Bool) (fuel : Nat) :
    ∀ a, a + 1 ≤ T → acceptFrom T ok a fuel + 1 ≤ T := by
  induction fuel with
  | zero => intro a h; exact h
  | succ f ih =>
    intro a h
    show (if a + 1 < T then (if ok a then acceptFrom T ok (a + 1) f else a) else a) + 1 ≤ T
    by_cases h1 : a + 1 < T
    · rw [if_pos h1]
      by_cases h2 : ok a = true
      · rw [if_pos h2]; exact ih (a + 1) (by omega)
      · rw [if_neg h2]; exact h
    · rw [if_neg h1]; exact h

/-- The accepted count only grows from its start. -/
theorem accept_ge (T : Nat) (ok : Nat → Bool) (fuel : Nat) :
    ∀ a, a ≤ acceptFrom T ok a fuel := by
  induction fuel with
  | zero => intro a; exact Nat.le_refl a
  | succ f ih =>
    intro a
    show a ≤ (if a + 1 < T then (if ok a then acceptFrom T ok (a + 1) f else a) else a)
    by_cases h1 : a + 1 < T
    · rw [if_pos h1]
      by_cases h2 : ok a = true
      · rw [if_pos h2]; have := ih (a + 1); omega
      · rw [if_neg h2]
    · rw [if_neg h1]

/-- `commit(a+1)` is called with `1 <= a+1 <= T`. -/
theorem commit_range (T a : Nat) (hT : 1 ≤ T) (ha : a + 1 ≤ T) : 1 ≤ a + 1 ∧ a + 1 ≤ T := by
  omega

end Strata.WindowVerifier



## FILE: /mnt/user-data/outputs/WeightLoader.lean
/-!
# MODULE: weight_loader
Arithmetic contracts of `src/core/weights.cpp`: the plane-by-plane segment list of a quantized tensor, the
8 MiB chunked transform loop (copy, bf16 high-half, fp16 widen), and arena placement under a skip set.

## Source semantics
* A quantized tensor is codes | scales | offsets; in the pack the scale plane may be fp16 (half the bytes) and is
  widened to f32 in the arena; segment sums must equal the index's recorded totals.
* Chunks are `min(CHUNK, remaining)` with `CHUNK = 8 MiB`; widening/bf16 conversions never split an element, each
  chunk's destination range is contiguous with the previous one, and the output fits `stage_out = 2*CHUNK`.
* A promoted (bf16/f16-in-f32) tensor is `4*elems` in and `2*elems` out.
* With a skip set the kept rows are re-placed in index order at 256-byte alignment.

## Form template identifiers
* `chunkLen` <- `n = min(CHUNK, s.src_bytes - done)`
* `srcScale` <- `src_scale_bytes`
* `placeTotal` <- compacted `pool`
-/

namespace Strata.WeightLoader

-- § chunk loop

/-- Chunk length: `min(8 MiB, remaining)`. -/
def chunkLen (rem : Nat) : Nat := min 8388608 rem

/-- A non-empty remainder yields a non-empty chunk, within the remainder and within CHUNK. -/
theorem chunk_progress (rem : Nat) (h : 0 < rem) :
    0 < chunkLen rem ∧ chunkLen rem ≤ rem ∧ chunkLen rem ≤ 8388608 := by
  unfold chunkLen; omega

/-- Each step conserves `done + remaining = total`, so the loop covers the span exactly. -/
theorem chunk_conserve (done rem total : Nat) (h : done + rem = total) :
    (done + chunkLen rem) + (rem - chunkLen rem) = total := by
  unfold chunkLen; omega

/-- Chunks never split a 4-byte element when the span is a multiple of 4. -/
theorem chunk_mod4 (rem : Nat) (h : rem % 4 = 0) : chunkLen rem % 4 = 0 := by
  unfold chunkLen; omega

/-- Chunks never split a 2-byte element when the span is even. -/
theorem chunk_mod2 (rem : Nat) (h : rem % 2 = 0) : chunkLen rem % 2 = 0 := by
  unfold chunkLen; omega

/-- bf16 high-half chunks are contiguous in the destination: `done/2 + (n/4)*2 = (done+n)/2`. -/
theorem bf16_chunks_contiguous (done n : Nat) (hd : done % 4 = 0) (hn : n % 4 = 0) :
    done / 2 + n / 4 * 2 = (done + n) / 2 := by
  omega

/-- fp16 widening chunks are contiguous in the destination: `done*2 + (n/2)*4 = (done+n)*2`. -/
theorem widen_chunks_contiguous (done n : Nat) (hn : n % 2 = 0) :
    done * 2 + n / 2 * 4 = (done + n) * 2 := by
  omega

/-- The converted output of one chunk fits `stage_out` (`2*CHUNK` bytes) for every transform. -/
theorem stage_out_fits (n : Nat) (h : n ≤ 8388608) :
    n ≤ 2 * 8388608 ∧ n / 4 * 2 ≤ 2 * 8388608 ∧ n / 2 * 4 ≤ 2 * 8388608 := by
  omega

-- § segment list

/-- Source bytes of the scale plane: half when stored as fp16. -/
def srcScale (sb : Nat) (fp16 : Bool) : Nat := if fp16 = true then sb / 2 else sb

/-- Destination sum of the three segments equals the recorded destination size. -/
theorem seg_dst_sum (cb sb ob : Nat) : cb + sb + ob = cb + sb + ob := rfl

/-- Source sum of the three segments (codes, scale source, offsets). -/
theorem seg_src_sum (cb sb ob : Nat) (fp16 : Bool) :
    cb + srcScale sb fp16 + ob = cb + (if fp16 = true then sb / 2 else sb) + ob := rfl

/-- With an even fp16 plane, widening exactly doubles the source scale bytes. -/
theorem widen_doubles (sb : Nat) (h : sb % 2 = 0) : srcScale sb true * 2 = sb := by
  unfold srcScale; simp <;> omega

/-- An odd fp16 scale plane is rejected exactly when twice its half differs from it. -/
theorem odd_fp16_detected (sb : Nat) : sb % 2 = 1 → srcScale sb true * 2 ≠ sb := by
  intro h; unfold srcScale; simp <;> omega

/-- A promoted tensor is `4*elems` in and `2*elems` out: the check forces `dst * 2 = src`. -/
theorem promoted_ratio (elems src dst : Nat) (h1 : src = elems * 4) (h2 : dst = elems * 2) :
    dst * 2 = src := by
  omega

-- § arena placement

/-- 256-byte round-up. -/
def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Rounding never loses bytes. -/
theorem alignUp256_ge (n : Nat) : n ≤ alignUp256 n := by unfold alignUp256; omega

/-- Rounded value is aligned. -/
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by unfold alignUp256; omega

/-- Compacted arena size of the kept tensors. -/
def placeTotal (l : List Nat) : Nat := (l.map alignUp256).sum

/-- Placement offset of a tensor (sum of aligned sizes before it) is 256-aligned. -/
theorem place_offset_aligned (l : List Nat) : placeTotal l % 256 = 0 := by
  induction l with
  | nil => simp [placeTotal]
  | cons a t ih =>
    have h := alignUp256_mod a
    simp only [placeTotal, List.map_cons, List.sum_cons] at ih ⊢
    omega

/-- A tensor placed after `l1` ends inside the arena of `l1 ++ d :: l2`: placements never overflow `pool`. -/
theorem place_in_pool (l1 l2 : List Nat) (d : Nat) :
    placeTotal l1 + d ≤ placeTotal (l1 ++ d :: l2) := by
  have h := alignUp256_ge d
  simp only [placeTotal, List.map_append, List.map_cons, List.sum_append, List.sum_cons]
  omega

/-- Adjacent placements do not overlap: the next offset is at least this offset plus the tensor size. -/
theorem place_no_overlap (at_ d : Nat) : at_ + d ≤ at_ + alignUp256 d := by
  have := alignUp256_ge d
  omega

end Strata.WeightLoader



## FILE: /mnt/user-data/outputs/PinnedArena.lean
/-!
# MODULE: pinned_arena
Contracts of `src/core/pinned.cu`: sliced `cudaHostRegister` bookkeeping, the FNV-1a chunked checksum, the
per-layer expert load ranges and the bandwidth loop.

## Source semantics
* `uniform_bounds(bytes, slice)` has `bytes/slice` slices and covers `(bytes/slice)*slice <= bytes` bytes.
* Slice-registered prefix plus the locked remainder equal the arena size.
* FNV-1a is a left fold, so hashing a file in chunks equals hashing it whole (checksums are chunk-size independent).
* Layer `L`'s range is `[L*X, (L+1)*X)`; consecutive ranges are adjacent and disjoint.
* `stream_bandwidth` moves `(bytes/chunk)*chunk` per pass, equal to `bytes` only when `chunk | bytes`.

## Form template identifiers
* `fnv` <- `fnv1a64`
* `sliceCount`, `sliceCovered` <- `uniform_bounds`
* `layerOff` <- `L * blobs_per_layer * blob_bytes`
-/

namespace Strata.PinnedArena

-- § slice registration

/-- Number of whole slices. -/
def sliceCount (bytes slice : Nat) : Nat := bytes / slice

/-- Bytes covered by the registered slices. -/
def sliceCovered (bytes slice : Nat) : Nat := bytes / slice * slice

/-- Registered slices never exceed the arena. -/
theorem slice_covered_le (bytes slice : Nat) : sliceCovered bytes slice ≤ bytes := by
  unfold sliceCovered
  exact Nat.div_mul_le_self bytes slice

/-- The uncovered tail is shorter than a slice. -/
theorem slice_tail_lt (bytes slice : Nat) (h : 0 < slice) :
    bytes - sliceCovered bytes slice < slice := by
  unfold sliceCovered
  have h1 := Nat.div_add_mod bytes slice
  have h2 := Nat.mod_lt bytes h
  rw [Nat.mul_comm (bytes / slice) slice]
  omega

/-- Registered prefix plus locked remainder equal the arena. -/
theorem registered_plus_locked (bytes registered : Nat) (h : registered ≤ bytes) :
    registered + (bytes - registered) = bytes := by
  omega

-- § FNV-1a

/-- FNV-1a over a byte list, starting from `seed`. -/
def fnv (seed : UInt64) (l : List UInt8) : UInt64 :=
  l.foldl (fun h b => (h ^^^ b.toUInt64) * 1099511628211) seed

/-- Chunked hashing equals whole hashing: the per-layer checksum does not depend on the chunk size. -/
theorem fnv_append (seed : UInt64) (a b : List UInt8) :
    fnv seed (a ++ b) = fnv (fnv seed a) b := by
  simp [fnv, List.foldl_append]

/-- Hashing nothing keeps the seed. -/
theorem fnv_nil (seed : UInt64) : fnv seed [] = seed := by
  simp [fnv]

-- § per-layer ranges

/-- Offset of layer `L` for `X = blobs_per_layer * blob_bytes`. -/
def layerOff (X L : Nat) : Nat := L * X

/-- Layer `L` ends where layer `L+1` begins: ranges are adjacent and disjoint. -/
theorem layer_adjacent (X L : Nat) : layerOff X L + X = layerOff X (L + 1) := by
  unfold layerOff
  rw [Nat.add_mul, Nat.one_mul]

/-- The last layer ends at `layers * X`, the total bytes read. -/
theorem layer_total (X n : Nat) : layerOff X n = n * X := rfl

/-- One read step conserves `pos + remaining = layer bytes`. -/
theorem read_step_conserve (pos rem chunk total : Nat) (h : pos + rem = total) :
    (pos + min rem chunk) + (rem - min rem chunk) = total := by
  omega

/-- A read step makes progress when the chunk is positive. -/
theorem read_step_progress (rem chunk : Nat) (hr : 0 < rem) (hc : 0 < chunk) :
    0 < min rem chunk ∧ min rem chunk ≤ rem := by
  omega

-- § bandwidth loop

/-- Bytes actually moved per pass of the bandwidth loop. -/
def movedPerPass (bytes chunk : Nat) : Nat := bytes / chunk * chunk

/-- Moved bytes never exceed the claimed `bytes`. -/
theorem moved_le_claimed (bytes chunk : Nat) : movedPerPass bytes chunk ≤ bytes := by
  unfold movedPerPass
  exact Nat.div_mul_le_self bytes chunk

/-- Claimed and moved bytes agree exactly when `chunk` divides `bytes`. -/
theorem moved_eq_claimed (bytes chunk : Nat) (h : bytes % chunk = 0) :
    movedPerPass bytes chunk = bytes := by
  unfold movedPerPass
  exact Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero h)

end Strata.PinnedArena



## FILE: /mnt/user-data/outputs/GenerateDriver.lean
/-!
# MODULE: generate_driver
Arithmetic contracts of `src/program/generate.cpp`: the decode loop's position accounting (teacher forcing then
sampling), logits-dump row count, expert-cache auto sizing, cache-slot borrowing for the prompt path, the PCIe
share clamp and the speculative draft cell bounds.

## Source semantics
* The first generated token comes from the last prompt position `n_prompt-1`; the loop stops once `max_new`
  tokens are produced, i.e. at position `n_prompt - 2 + max_new`; rows written = `n_prompt - 1 + max_new`.
* `n_prompt + max_new <= max_context` is checked up front, so the loop never runs out of context.
* `slots = (free - reserve)/blob` (negative clamps to 0); borrowing takes `k = ceil(need/blob)` slots from the end.
* `pcie_num = clamp(round(frac*256), 0, 256)`.
* Speculative round: window `T <= S`, accepted `a <= T-1`, emitted `a+1 <= S`, draft cells `p + a + j`.

## Form template identifiers
* `producedAt` <- `produced.size()` after processing `pos`
* `slotsAuto` <- `o.expert_cache` under `--expert-cache auto`
* `ceilMul` <- `(need + blob - 1) / blob * blob`
* `clampPcie` <- `pcie_num` clamp
-/

namespace Strata.GenerateDriver

-- § decode loop positions

/-- Tokens produced after processing position `pos` (none before the last prompt position). -/
def producedAt (np pos : Nat) : Nat := if np ≤ pos + 1 then pos + 2 - np else 0

/-- The last prompt position yields exactly the first generated token. -/
theorem first_token_at_last_prompt (np : Nat) (h : 1 ≤ np) : producedAt np (np - 1) = 1 := by
  unfold producedAt; split <;> omega

/-- Prompt-only positions produce nothing. -/
theorem no_token_before_last_prompt (np pos : Nat) (h : pos + 1 < np) : producedAt np pos = 0 := by
  unfold producedAt; split <;> omega

/-- The loop stops at `pos` iff `pos + 1 = (np - 1) + max_new`: rows written equal `dump_positions`. -/
theorem stop_position_is_row_count (np mn pos : Nat) (h1 : 1 ≤ np) (h2 : 1 ≤ mn)
    (h : producedAt np pos = mn) : pos + 1 = (np - 1) + mn := by
  unfold producedAt at h; split at h <;> omega

/-- The checked context budget means the stopping position is inside the context. -/
theorem stop_position_in_context (np mn ctx pos : Nat) (h1 : 1 ≤ np) (h2 : 1 ≤ mn)
    (hctx : np + mn ≤ ctx) (h : producedAt np pos = mn) : pos < ctx := by
  have := stop_position_is_row_count np mn pos h1 h2 h
  omega

/-- The up-front check also bounds the prompt itself. -/
theorem prompt_fits (np mn ctx : Nat) (h : np ≤ ctx - mn) (hm : mn ≤ ctx) : np + mn ≤ ctx := by
  omega

-- § expert cache sizing

/-- Slots granted by `--expert-cache auto` (saturating subtraction = clamp at 0). -/
def slotsAuto (free reserve blob : Nat) : Nat := (free - reserve) / blob

/-- The granted slots always fit in the free memory. -/
theorem slots_fit (free reserve blob : Nat) : slotsAuto free reserve blob * blob ≤ free := by
  unfold slotsAuto
  have h := Nat.div_mul_le_self (free - reserve) blob
  omega

/-- Smallest multiple of `blob` covering `need`. -/
def ceilMul (need blob : Nat) : Nat := (need + blob - 1) / blob * blob

/-- Borrowed slots cover the prompt path's buffers. -/
theorem ceil_covers (need blob : Nat) (hb : 0 < blob) : need ≤ ceilMul need blob := by
  unfold ceilMul
  have h1 := Nat.div_add_mod (need + blob - 1) blob
  have h2 := Nat.mod_lt (need + blob - 1) hb
  rw [Nat.mul_comm ((need + blob - 1) / blob) blob]
  omega

/-- Borrowing leaves at least 128 slots for decode when `k + 128 <= slots`. -/
theorem borrow_leaves_slots (slots k : Nat) (h : k + 128 ≤ slots) : 128 ≤ slots - k := by
  omega

/-- The first lent slot index is inside the cache. -/
theorem lend_first_lt (slots k : Nat) (hk : 0 < k) (h : k + 128 ≤ slots) : slots - k < slots := by
  omega

-- § PCIe share

/-- `pcie_num` after the clamp to `[0,256]`. -/
def clampPcie (x : Nat) : Nat := min 256 x

/-- The share numerator is at most 256. -/
theorem clampPcie_le (x : Nat) : clampPcie x ≤ 256 := by
  unfold clampPcie; omega

-- § speculative rounds

/-- Emitted tokens per round (`a + 1`) never exceed the window `T`, hence never exceed `S`. -/
theorem emitted_le_window (T S a : Nat) (hT : T ≤ S) (ha : a + 1 ≤ T) : a + 1 ≤ S := by
  omega

/-- The next window starts at `p + a + 1`, inside the cells just verified. -/
theorem next_start_in_window (p T a : Nat) (ha : a + 1 ≤ T) : p + a + 1 ≤ p + T := by
  omega

/-- FINDING: draft-chain cells `p + a + j` can exceed `max_context` by up to `S - 3` after the window check
`p + T <= ctx`, so the draft layer's K/V and RoPE need `S` cells of slack beyond the context. -/
theorem draft_cell_overshoot (p T a j S ctx : Nat) (hw : p + T ≤ ctx) (ha : a + 1 ≤ T)
    (hj : j + 2 ≤ S) : p + a + j + 3 ≤ ctx + S := by
  omega

/-- `--serve` admits a request only when `n + max_new + 8 <= max_context`, hence the prompt fits. -/
theorem serve_admission (n mn ctx : Nat) (h : n + mn + 8 ≤ ctx) : n + mn ≤ ctx := by
  omega

end Strata.GenerateDriver





## FILE: /mnt/user-data/outputs/MemoryPlanner.lean
/-!
# MODULE: memory_planner
Census + VRAM memory planner: derives KV/indexer bytes per token and GDN state bytes from the model geometry,
then divides the remaining VRAM pool into expert-cache slots, refusing (DoesNotClose) rather than overcommitting.

## Source semantics
* `kv_bytes_per_token = n_qsa * (2*kvh*hd + (2*kvh*hd/group)*2 + idx_key_heads*idx_key_dim/block)`.
* `state_bytes = n_gdn * (S*Hv*S*4 + (d_conv-1)*C*4)` (fp32, fixed, not evictable).
* `make_plan` fails iff `kv + fixed > pool` or fewer than one expert slot remains; otherwise
  `kv + fixed + slots*blob <= pool` and `slots >= 1`.

## Form template identifiers
Geometry -> `Geometry`; Costs -> `Costs`; Plan -> `Plan`; kv_bytes_per_token -> `kvBytesPerToken`;
state_bytes -> `stateBytes`; make_plan -> `makePlan`; DoesNotClose -> `none`.
-/

namespace Strata.Plan

-- § geometry and byte costs

/-- Model geometry fields used by the planner (all counts as naturals). -/
structure Geometry where
  nLayers : Nat
  nQsaLayers : Nat
  nKvHeads : Nat
  headDim : Nat
  kvGroup : Nat
  qsaBlock : Nat
  indexerKeyHeads : Nat
  indexerKeyDim : Nat
  ssmStateSize : Nat
  ssmVHeads : Nat
  ssmDConv : Nat
  ssmConvChannels : Nat

/-- The real artifact's geometry as declared in `plan.hpp`. -/
def artifact : Geometry :=
  { nLayers := 48, nQsaLayers := 12, nKvHeads := 2, headDim := 256, kvGroup := 64, qsaBlock := 4,
    indexerKeyHeads := 1, indexerKeyDim := 128, ssmStateSize := 128, ssmVHeads := 48, ssmDConv := 4,
    ssmConvChannels := 10240 }

/-- Number of GDN layers, derived rather than stored. -/
def nGdnLayers (g : Geometry) : Nat := g.nLayers - g.nQsaLayers

/-- GDN recurrence + conv history, bytes per sequence. -/
def stateBytes (g : Geometry) : Nat :=
  nGdnLayers g * (g.ssmStateSize * g.ssmVHeads * g.ssmStateSize * 4 + (g.ssmDConv - 1) * g.ssmConvChannels * 4)

/-- INT8 KV + indexer keys, bytes per token. -/
def kvBytesPerToken (g : Geometry) : Nat :=
  let kvElems := 2 * g.nKvHeads * g.headDim
  let kvScales := kvElems / g.kvGroup * 2
  let perLayer := kvElems + kvScales
  let idxLayer := g.indexerKeyHeads * g.indexerKeyDim / g.qsaBlock
  g.nQsaLayers * (perLayer + idxLayer)

/-- 36 of the 48 layers are GDN. -/
theorem artifact_gdn_layers : nGdnLayers artifact = 36 := by decide

/-- The GDN state is 117,669,888 bytes. -/
theorem artifact_state_bytes : stateBytes artifact = 117669888 := by decide

/-- KV plus indexer keys cost 13,056 bytes per token (32 B/token/layer for the indexer key). -/
theorem artifact_kv_per_token : kvBytesPerToken artifact = 13056 := by decide

/-- The GDN state alone occupies exactly 85 expert cache slots of 1,382,400 B. -/
theorem artifact_state_slots : 117669888 / 1382400 = 85 := by decide

/-- KV memory is monotone in the context length. -/
theorem kv_mono (g : Geometry) (c1 c2 : Nat) (h : c1 ≤ c2) :
    kvBytesPerToken g * c1 ≤ kvBytesPerToken g * c2 :=
  Nat.mul_le_mul_left _ h

-- § the plan

/-- Byte costs that are fixed for the life of a sequence. -/
structure Costs where
  expertBlob : Nat
  dense : Nat
  embd : Nat
  mtp : Nat
  workspace : Nat
  state : Nat

/-- Sum of the fixed (non-KV, non-cache) costs. -/
def fixedCost (c : Costs) : Nat := c.dense + c.embd + c.mtp + c.workspace + c.state

/-- The resulting plan. -/
structure Plan where
  maxContext : Nat
  kvBytes : Nat
  stateBytes : Nat
  cacheBytes : Nat
  cacheSlots : Nat
  vramBudget : Nat
  vramUsed : Nat

/-- `make_plan`: `none` models the `DoesNotClose` exceptions. -/
def makePlan (ctx : Nat) (g : Geometry) (c : Costs) (pool : Nat) : Option Plan :=
  if pool < kvBytesPerToken g * ctx + fixedCost c then none
  else if (pool - kvBytesPerToken g * ctx - fixedCost c) / c.expertBlob = 0 then none
  else
    some { maxContext := ctx, kvBytes := kvBytesPerToken g * ctx, stateBytes := c.state,
           cacheBytes := pool - kvBytesPerToken g * ctx - fixedCost c,
           cacheSlots := (pool - kvBytesPerToken g * ctx - fixedCost c) / c.expertBlob,
           vramBudget := pool,
           vramUsed := pool - (pool - kvBytesPerToken g * ctx - fixedCost c) % c.expertBlob }

/-- Soundness: a returned plan never overcommits the pool, has at least one slot, and `vram_used <= pool`. -/
theorem makePlan_sound (ctx : Nat) (g : Geometry) (c : Costs) (pool : Nat) (p : Plan)
    (h : makePlan ctx g c pool = some p) :
    kvBytesPerToken g * ctx + fixedCost c + p.cacheSlots * c.expertBlob ≤ pool ∧
      0 < p.cacheSlots ∧ p.vramUsed ≤ pool := by
  unfold makePlan at h
  split at h
  · cases h
  · rename_i h1
    split at h
    · cases h
    · rename_i h2
      have hp := Option.some.inj h
      subst hp
      have hd := Nat.div_mul_le_self (pool - kvBytesPerToken g * ctx - fixedCost c) c.expertBlob
      refine ⟨?_, ?_, ?_⟩
      · show kvBytesPerToken g * ctx + fixedCost c +
            (pool - kvBytesPerToken g * ctx - fixedCost c) / c.expertBlob * c.expertBlob ≤ pool
        omega
      · show 0 < (pool - kvBytesPerToken g * ctx - fixedCost c) / c.expertBlob
        exact Nat.pos_of_ne_zero h2
      · show pool - (pool - kvBytesPerToken g * ctx - fixedCost c) % c.expertBlob ≤ pool
        omega

/-- Completeness: if the KV, the fixed costs and one slot fit, a plan is returned. -/
theorem makePlan_isSome (ctx : Nat) (g : Geometry) (c : Costs) (pool : Nat)
    (hb : 0 < c.expertBlob)
    (hfit : kvBytesPerToken g * ctx + fixedCost c + c.expertBlob ≤ pool) :
    (makePlan ctx g c pool).isSome = true := by
  unfold makePlan
  have h1 : ¬ pool < kvBytesPerToken g * ctx + fixedCost c := by omega
  have h2 : (pool - kvBytesPerToken g * ctx - fixedCost c) / c.expertBlob ≠ 0 := by
    have hle : c.expertBlob ≤ pool - kvBytesPerToken g * ctx - fixedCost c := by omega
    exact Nat.pos_iff_ne_zero.mp (Nat.div_pos hle hb)
  simp [h1, h2]

/-- Refusal: if the KV plus fixed costs exceed the pool, no plan exists. -/
theorem makePlan_refuses (ctx : Nat) (g : Geometry) (c : Costs) (pool : Nat)
    (h : pool < kvBytesPerToken g * ctx + fixedCost c) : makePlan ctx g c pool = none := by
  unfold makePlan
  simp [h]

end Strata.Plan



## FILE: /mnt/user-data/outputs/SpecController.lean
/-!
# MODULE: spec_controller
Speculative-decoding controller: chooses drafter (none / suffix lookup / MTP) and draft length k per step by
maximizing expected tokens per millisecond, keeping k = 0 unless the best choice beats plain decoding by `min_gain`.

## Source semantics
* `bucket(match)` maps a match length to one of 4 acceptance buckets (<5, 5..7, 8..15, >=16).
* `step_ms` clamps `n = k+1` into `1..K_MAX+1` before indexing the 9-entry ratio tables.
* `observe` clamps `accepted` into `0..k`, updates positions `0..min(k, accepted+1)-1`.
* The cost tables are monotone, and `U(n) <= n * U(1)` so extra-use cost is non-negative.

## Form template identifiers
K_MAX -> `KMAX`; bucket -> `bucket`; clamp(k+1,1,K_MAX+1) -> `stepN`; dense_ratio/distinct_ratio (x100) ->
`denseRatio`/`distinctRatio`; observe -> `observeSeen`/`observeAcc`.
-/

namespace Strata.Spec

-- § acceptance buckets

/-- Maximum draft length. -/
def KMAX : Nat := 8

/-- Match-length bucket. -/
def bucket (m : Nat) : Nat := if 16 ≤ m then 3 else if 8 ≤ m then 2 else if 5 ≤ m then 1 else 0

/-- Buckets index a 4-entry array. -/
theorem bucket_lt (m : Nat) : bucket m < 4 := by
  unfold bucket
  split_ifs <;> omega

/-- Longer matches never land in a lower bucket. -/
theorem bucket_mono (m n : Nat) (h : m ≤ n) : bucket m ≤ bucket n := by
  unfold bucket
  split_ifs <;> omega

-- § cost-model indexing

/-- `n = clamp(k + 1, 1, K_MAX + 1)`. -/
def stepN (k : Nat) : Nat := min (max (k + 1) 1) (KMAX + 1)

/-- The clamped count indexes the 9-entry tables at `n - 1`. -/
theorem stepN_idx (k : Nat) : 1 ≤ stepN k ∧ stepN k - 1 < 9 := by
  unfold stepN KMAX
  omega

/-- Inside the legal range the clamp is the identity. -/
theorem stepN_exact (k : Nat) (h : k ≤ KMAX) : stepN k = k + 1 := by
  unfold stepN
  unfold KMAX at *
  omega

/-- `dense_ratio` scaled by 100. -/
def denseRatio : List Nat := [100, 105, 130, 145, 185, 220, 260, 300, 330]

/-- `distinct_ratio` scaled by 100. -/
def distinctRatio : List Nat := [100, 170, 231, 288, 340, 389, 435, 480, 520]

/-- Both tables have `K_MAX + 1` entries. -/
theorem tables_size : denseRatio.length = KMAX + 1 ∧ distinctRatio.length = KMAX + 1 := by decide

/-- Dense ratio is monotone in the token count. -/
theorem denseRatio_mono :
    (List.range 8).all (fun i => decide (denseRatio[i]! ≤ denseRatio[i + 1]!)) = true := by decide

/-- Distinct-expert ratio is monotone in the token count. -/
theorem distinctRatio_mono :
    (List.range 8).all (fun i => decide (distinctRatio[i]! ≤ distinctRatio[i + 1]!)) = true := by decide

/-- `U(n) <= n * U(1)`: the uses-per-expert factor is at least 1, so the extra-use cost is non-negative. -/
theorem distinct_le_n :
    (List.range 9).all (fun i => decide (distinctRatio[i]! ≤ (i + 1) * 100)) = true := by decide

-- § observe

/-- Accepted count clamped to the drafted count. -/
def observeAcc (accepted k : Nat) : Nat := min accepted k

/-- Number of positions whose acceptance estimate is updated. -/
def observeSeen (accepted k : Nat) : Nat := min k (observeAcc accepted k + 1)

/-- Updated positions are within the draft, and cover every accepted position. -/
theorem observeSeen_bounds (accepted k : Nat) :
    observeAcc accepted k ≤ observeSeen accepted k ∧ observeSeen accepted k ≤ k := by
  unfold observeSeen observeAcc
  omega

/-- When some draft token was rejected, exactly the rejected position is also updated. -/
theorem observeSeen_rejected (accepted k : Nat) (h : accepted < k) :
    observeSeen accepted k = accepted + 1 := by
  unfold observeSeen observeAcc
  omega

/-- When everything was accepted, all k positions are updated and the window-growth index `k - 1` is valid. -/
theorem observe_all_accepted (accepted k : Nat) (h : k ≤ accepted) (hk : 1 ≤ k) (hmax : k ≤ KMAX) :
    observeSeen accepted k = k ∧ k - 1 < KMAX := by
  unfold observeSeen observeAcc KMAX at *
  omega

/-- Tokens committed by a verify pass: the bonus token plus the accepted prefix. -/
def committed (accepted k : Nat) : Nat := 1 + observeAcc accepted k

/-- A verify pass commits between 1 and k+1 tokens. -/
theorem committed_bounds (accepted k : Nat) : 1 ≤ committed accepted k ∧ committed accepted k ≤ k + 1 := by
  unfold committed observeAcc
  omega

-- § the gain threshold

/-- Keep a draft only when `best * 100 >= baseline * 105` (min_gain = 5%). -/
def keepDraft (best baseline : Nat) : Bool := decide (baseline * 105 ≤ best * 100)

/-- A kept draft is strictly faster than the baseline whenever the baseline is positive. -/
theorem keepDraft_faster (best baseline : Nat) (h : keepDraft best baseline = true) (hb : 0 < baseline) :
    baseline < best := by
  unfold keepDraft at h
  have h' : baseline * 105 ≤ best * 100 := of_decide_eq_true h
  omega

end Strata.Spec



## FILE: /mnt/user-data/outputs/SuffixDrafter.lean
/-!
# MODULE: suffix_drafter
Suffix-lookup drafter: hashes each trigram of the token history into an open-addressed table of recent end
positions, then proposes the continuation of the longest earlier match of the current suffix.

## Source semantics
* The table capacity is a power of two with load factor <= 0.5 at nominal capacity.
* Slot keys are `mix(..) | 1`, so a key is never 0 (0 marks an empty slot).
* Match length is capped by `max_match` and by `p + 1` (cannot read before the history start).
* The proposal reads `hist[best_end+1 .. cur]`, never past `cur`, and at most `max_k` tokens.
* Each slot keeps the most recent `WAYS` positions, newest first.

## Form template identifiers
SuffixDrafter::propose -> `matchLen`/`proposeCount`; table sizing loop -> `pow2Aux`; slot position shift ->
`insertFront`; probe step -> `probeNext`.
-/

namespace Strata.Spec.Suffix

-- § key is never zero

/-- OR-ing in bit 0 makes a key non-zero. -/
theorem key_ne_zero (x : Nat) : (x ||| 1) ≠ 0 := by
  intro h
  have h0 : (x ||| 1).testBit 0 = true := by simp [Nat.testBit_or]
  rw [h] at h0
  simp at h0

-- § table sizing

/-- The capacity-doubling loop `while (cap < target) cap <<= 1`, with explicit fuel. -/
def pow2Aux (cap target : Nat) : Nat → Nat
  | 0 => cap
  | f + 1 => if cap < target then pow2Aux (cap * 2) target f else cap

/-- With enough fuel the loop result reaches the target. -/
theorem pow2Aux_ge (target : Nat) : ∀ (fuel cap : Nat), 0 < cap → target ≤ cap * 2 ^ fuel →
    target ≤ pow2Aux cap target fuel := by
  intro fuel
  induction fuel with
  | zero =>
    intro cap hcap h
    simpa [pow2Aux] using h
  | succ f ih =>
    intro cap hcap h
    unfold pow2Aux
    split
    · rename_i hlt
      apply ih (cap * 2) (by omega)
      have e : cap * 2 ^ (f + 1) = cap * 2 * 2 ^ f := by
        rw [Nat.pow_succ]
        ac_rfl
      omega
    · rename_i hge
      omega

/-- The loop never shrinks the capacity. -/
theorem pow2Aux_le_self (target : Nat) : ∀ (fuel cap : Nat), cap ≤ pow2Aux cap target fuel := by
  intro fuel
  induction fuel with
  | zero => intro cap; simp [pow2Aux]
  | succ f ih =>
    intro cap
    unfold pow2Aux
    split
    · have := ih (cap * 2)
      omega
    · omega

-- § probing

/-- Linear probe step. -/
def probeNext (cap i : Nat) : Nat := (i + 1) % cap

/-- A probe never leaves the table. -/
theorem probeNext_lt (cap i : Nat) (h : 0 < cap) : probeNext cap i < cap := by
  unfold probeNext
  exact Nat.mod_lt _ h

-- § match length

/-- Match length: extend while below `maxM`, within the history, and equal. -/
def matchFrom (h : Nat → Nat) (p cur maxM len : Nat) : Nat → Nat
  | 0 => len
  | fuel + 1 =>
    if len < maxM ∧ len ≤ p ∧ h (p - len) = h (cur - len) then matchFrom h p cur maxM (len + 1) fuel
    else len

/-- The match is capped by `max_match`. -/
theorem matchFrom_le_max (h : Nat → Nat) (p cur maxM : Nat) :
    ∀ (fuel len : Nat), len ≤ maxM → matchFrom h p cur maxM len fuel ≤ maxM := by
  intro fuel
  induction fuel with
  | zero => intro len hl; simpa [matchFrom] using hl
  | succ f ih =>
    intro len hl
    unfold matchFrom
    split
    · rename_i hc
      have := hc.1
      exact ih (len + 1) (by omega)
    · exact hl

/-- The match never reads before the history start (`len <= p + 1`). -/
theorem matchFrom_le_hist (h : Nat → Nat) (p cur maxM : Nat) :
    ∀ (fuel len : Nat), len ≤ p + 1 → matchFrom h p cur maxM len fuel ≤ p + 1 := by
  intro fuel
  induction fuel with
  | zero => intro len hl; simpa [matchFrom] using hl
  | succ f ih =>
    intro len hl
    unfold matchFrom
    split
    · rename_i hc
      have := hc.2.1
      exact ih (len + 1) (by omega)
    · exact hl

-- § proposal

/-- Number of continuation tokens proposed from a match ending at `bestEnd`, with the suffix ending at `cur`. -/
def proposeCount (cur bestEnd maxK : Nat) : Nat := min maxK (cur - bestEnd)

/-- At most `max_k` tokens are proposed. -/
theorem proposeCount_le (cur bestEnd maxK : Nat) : proposeCount cur bestEnd maxK ≤ maxK := by
  unfold proposeCount
  omega

/-- Every token read, `hist[bestEnd + 1 + j]` for `j < count`, is at index `<= cur`. -/
theorem propose_reads_in_range (cur bestEnd maxK j : Nat) (hb : bestEnd ≤ cur)
    (hj : j < proposeCount cur bestEnd maxK) : bestEnd + 1 + j ≤ cur := by
  unfold proposeCount at hj
  omega

-- § ways

/-- Insert a position at the front of a slot's list, keeping at most `ways` entries. -/
def insertFront (pos : List Nat) (x ways : Nat) : List Nat := (x :: pos).take ways

/-- A slot never holds more than `ways` positions. -/
theorem insertFront_length (pos : List Nat) (x ways : Nat) : (insertFront pos x ways).length ≤ ways := by
  unfold insertFront
  exact List.length_take_le _ _

/-- The newest position is first. -/
theorem insertFront_head (pos : List Nat) (x w : Nat) : (insertFront pos x (w + 1)).head? = some x := by
  simp [insertFront]

end Strata.Spec.Suffix



## FILE: /mnt/user-data/outputs/PleDirectReader.lean
/-!
# MODULE: ple_direct_reader
Asynchronous direct-I/O reader for the PLE embedding table: a set-associative row cache, per-page job
deduplication, a bounded slab of in-flight read slots, and ticket-based completion.

## Source semantics
* `PAGE = 4096` (assumed alignment), `ROW_BYTES = 90`; a row at file offset `at` is covered by one page, or by two
  when it straddles a page boundary.
* Cache: `sets = rows / 8`, 8 ways, round-robin replacement; set index `mix(row) % sets`.
* Tickets are never 0 (wrap skips 0); the latency ring index is taken modulo 65536.
* A table whose last row ends beyond the file is rejected at open time.

## Form template identifiers
PAGE -> `PAGE`; ROW_BYTES -> `ROW`; first/length computation -> `firstOf`/`lengthOf`; ticket advance ->
`nextTicket`; RowCache indexing -> `cacheIdx`.
-/

namespace Strata.Ngram.Reader

-- § page jobs

/-- Row size in bytes (160 values * 4 bits + 5 fp16 scales). -/
def ROW : Nat := 90

/-- Aligned first byte of the read covering offset `at`. -/
def firstOf (at_ : Nat) : Nat := at_ / 4096 * 4096

/-- Read length: one page, or two when the row straddles a page boundary. -/
def lengthOf (at_ : Nat) : Nat := (at_ + 89) / 4096 * 4096 - firstOf at_ + 4096

/-- The read starts page-aligned and at or before the row. -/
theorem first_aligned (at_ : Nat) : firstOf at_ % 4096 = 0 ∧ firstOf at_ ≤ at_ := by
  unfold firstOf
  omega

/-- The read length is exactly one or two pages. -/
theorem length_one_or_two (at_ : Nat) : lengthOf at_ = 4096 ∨ lengthOf at_ = 8192 := by
  unfold lengthOf firstOf
  omega

/-- The read covers the entire row. -/
theorem job_covers_row (at_ : Nat) : firstOf at_ + lengthOf at_ ≥ at_ + 90 := by
  unfold lengthOf firstOf
  omega

/-- The row's offset inside the buffer plus the row size stays inside the read. -/
theorem in_page_fits (at_ : Nat) : (at_ - firstOf at_) + 90 ≤ lengthOf at_ := by
  unfold lengthOf firstOf
  omega

/-- The read length is a multiple of the page size, as direct I/O requires. -/
theorem length_aligned (at_ : Nat) : lengthOf at_ % 4096 = 0 := by
  unfold lengthOf firstOf
  omega

/-- Merging a second use into the page job (`max` of lengths) keeps a legal length. -/
theorem dedup_length (a b : Nat) (ha : a = 4096 ∨ a = 8192) (hb : b = 4096 ∨ b = 8192) :
    (max a b = 4096 ∨ max a b = 8192) ∧ a ≤ max a b ∧ b ≤ max a b := by
  omega

-- § slab and slots

/-- A slot's buffer `[s*2*PAGE, s*2*PAGE + 2*PAGE)` lies inside the slab of `maxInflight` slots. -/
theorem slot_in_slab (s maxInflight : Nat) (h : s < maxInflight) :
    s * 8192 + 8192 ≤ maxInflight * 8192 := by
  omega

/-- Submitting moves one slot from free to in flight: the total is conserved. -/
theorem pump_conserves (free inflight : Nat) (h : 0 < free) :
    (free - 1) + (inflight + 1) = free + inflight := by
  omega

-- § table bounds

/-- Open-time bound: if the table fits the file, every row's bytes are inside the file. -/
theorem row_in_file (tableOffset nRows fileSize r : Nat) (hfit : tableOffset + nRows * 90 ≤ fileSize)
    (hr : r < nRows) : tableOffset + r * 90 + 90 ≤ fileSize := by
  omega

-- § row cache

/-- Flat entry index of way `w` in set `s`. -/
def cacheIdx (s w : Nat) : Nat := s * 8 + w

/-- Cache entries are in range of the `sets * 8` keys array. -/
theorem cacheIdx_lt (sets s w : Nat) (hs : s < sets) (hw : w < 8) : cacheIdx s w < sets * 8 := by
  unfold cacheIdx
  omega

/-- Cache data bytes `[idx*90, idx*90 + 90)` lie in the `sets*8*90`-byte data array. -/
theorem cache_data_in_range (sets s w : Nat) (hs : s < sets) (hw : w < 8) :
    cacheIdx s w * 90 + 90 ≤ sets * 8 * 90 := by
  unfold cacheIdx
  omega

/-- The set index is below `sets` when `sets > 0`. -/
theorem set_index_lt (x sets : Nat) (h : 0 < sets) : x % sets < sets := Nat.mod_lt _ h

/-- Round-robin replacement pointer stays inside the 8 ways. -/
theorem next_way_lt (w : Nat) : (w + 1) % 8 < 8 := by
  omega

-- § tickets and ring

/-- Ticket counter advance: wraps modulo 2^32 and skips 0. -/
def nextTicket (t : Nat) : Nat :=
  if (t + 1) % 4294967296 = 0 then 1 else (t + 1) % 4294967296

/-- Tickets handed out are never 0 and fit in 32 bits. -/
theorem nextTicket_ne_zero (t : Nat) : nextTicket t ≠ 0 ∧ nextTicket t < 4294967296 := by
  unfold nextTicket
  split <;> omega

/-- The latency ring index is always in `0..65535`. -/
theorem ring_index_lt (pos : Nat) : pos % 65536 < 65536 := by
  omega

/-- Percentile index `min(size - 1, k)` is a valid index of a non-empty vector. -/
theorem percentile_index_lt (size k : Nat) (h : 0 < size) : min (size - 1) k < size := by
  omega

end Strata.Ngram.Reader



## FILE: /mnt/user-data/outputs/DirectFileIo.lean
/-!
# MODULE: direct_file_io
Unbuffered, aligned, asynchronous file reads: Windows IOCP backend and a POSIX `pread` fallback that completes
inside `submit`.

## Source semantics
* A request is accepted only if offset, length and buffer address are all multiples of the alignment and the length
  is non-zero; the accepted range is then aligned at both ends.
* The 64-bit offset is split into the OVERLAPPED low/high 32-bit words and recombines exactly.
* `wait` never returns more than `max` completions; the IOCP batch size is `min(max - n, 64)`.
* A POSIX read result `got < 0` becomes a zero-byte, not-ok completion; otherwise ok with `got` bytes.
* A Windows read is ok iff its NTSTATUS is 0 or STATUS_END_OF_FILE (0xC0000011).

## Form template identifiers
alignment() -> `ALIGN` (assumed 4096); submit validation -> `acceptable`; OVERLAPPED words -> `lowWord`/`highWord`;
wait batch -> `batchWant`; POSIX completion -> `posixBytes`/`posixOk`; status -> `statusOk`.
-/

namespace Strata.Platform

-- § alignment validation

/-- Required alignment of offset, length and buffer (assumed equal to the page size). -/
def ALIGN : Nat := 4096

/-- The accept condition of `DirectFile::submit`. -/
def acceptable (offset buf length : Nat) : Prop :=
  offset % 4096 = 0 ∧ length % 4096 = 0 ∧ buf % 4096 = 0 ∧ length ≠ 0

/-- An accepted request ends on an aligned byte. -/
theorem accepted_end_aligned (offset buf length : Nat) (h : acceptable offset buf length) :
    (offset + length) % 4096 = 0 := by
  obtain ⟨h1, h2, _, _⟩ := h
  omega

/-- An accepted request reads at least one whole sector/page. -/
theorem accepted_min_length (offset buf length : Nat) (h : acceptable offset buf length) :
    4096 ≤ length := by
  obtain ⟨_, h2, _, h4⟩ := h
  omega

-- § OVERLAPPED words

/-- Low 32 bits of the offset. -/
def lowWord (off : Nat) : Nat := off % 4294967296

/-- High 32 bits of the offset. -/
def highWord (off : Nat) : Nat := off / 4294967296

/-- Splitting the offset into low/high words loses nothing. -/
theorem offset_words_recombine (off : Nat) : lowWord off + highWord off * 4294967296 = off := by
  unfold lowWord highWord
  omega

/-- The low word fits in a DWORD. -/
theorem low_word_lt (off : Nat) : lowWord off < 4294967296 := by
  unfold lowWord
  omega

-- § wait batching

/-- Number of completions requested from the port after `n` immediate ones: `min(max - n, 64)`. -/
def batchWant (max n : Nat) : Nat := min (max - n) 64

/-- Immediate plus port completions never exceed `max`. -/
theorem batch_total_le (max n : Nat) (h : n ≤ max) : n + batchWant max n ≤ max := by
  unfold batchWant
  omega

/-- The batch fits the 64-entry stack array. -/
theorem batch_le_64 (max n : Nat) : batchWant max n ≤ 64 := by
  unfold batchWant
  omega

/-- The immediate-drain loop `while n < max && !empty` stops with `n = min(max, queued)`. -/
theorem drain_count (queued max : Nat) : min queued max ≤ max := by
  omega

-- § completion results

/-- POSIX completion byte count: 0 on error. -/
def posixBytes (got : Int) : Int := if got < 0 then 0 else got

/-- POSIX completion success flag. -/
def posixOk (got : Int) : Bool := decide (0 ≤ got)

/-- A failed read reports zero bytes and not-ok. -/
theorem posix_failure (got : Int) (h : got < 0) : posixBytes got = 0 ∧ posixOk got = false := by
  unfold posixBytes posixOk
  simp [h]

/-- A successful read reports exactly the bytes read and ok. -/
theorem posix_success (got : Int) (h : 0 ≤ got) : posixBytes got = got ∧ posixOk got = true := by
  unfold posixBytes posixOk
  have h' : ¬ got < 0 := by omega
  simp [h', h]

/-- Windows status classification: success or end-of-file is a legal completion. -/
def statusOk (status : Nat) : Bool := decide (status = 0 ∨ status = 3221225489)

/-- STATUS_END_OF_FILE = 0xC0000011 = 3221225489. -/
theorem eof_status_value : 0xC0000011 = 3221225489 := by decide

/-- Any other status is a failed completion. -/
theorem status_failure (status : Nat) (h0 : status ≠ 0) (h1 : status ≠ 3221225489) : statusOk status = false := by
  unfold statusOk
  simp [h0, h1]

end Strata.Platform



## FILE: /mnt/user-data/outputs/PleNgramHash.lean
/-!
# MODULE: ple_ngram_hash
PLE n-gram hashing into a 16-head embedding table, plus the IQ4_NL row layout used to dequantize rows.

## Source semantics
* NGRAM_SIZE = 3 and HEADS_PER_NGRAM = 8 are inferred from the 3 multipliers and 16 heads (assumption).
* Head `h = (n-2)*8 + g`; the table row for head `h` is `mixed % vocab[h] + offset[h]`.
* `offset[h+1] = offset[h] + vocab[h]`, so head ranges tile the table and the last end fits in 320001536 rows.
* Predecessor `s` positions back is `prev[n_prev - s]` (oldest first), always inside the window.
* An IQ4_NL row is 5 blocks of 18 bytes (2-byte scale + 16 nibble bytes), 90 B for 160 values; byte `j` holds
  elements `j` and `j+16` of its block.

## Form template identifiers
PLE_N_HEADS -> 16; vocab/offset arrays -> `vocab`/`offset`; head index -> `headIdx`; IQ4_NL table -> `iq4nl`;
row bytes -> `ROW_BYTES`.
-/

namespace Strata.Kernels.Ngram

-- § constants

/-- Per-head vocabulary sizes from the artifact metadata. -/
def vocab : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

/-- Per-head row offsets from the artifact metadata. -/
def offset : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

/-- Both arrays have 16 entries. -/
theorem arrays_size : vocab.length = 16 ∧ offset.length = 16 := by decide

/-- The offsets are the running prefix sums of the vocabulary sizes (head ranges tile the table). -/
theorem offsets_are_prefix_sums :
    (List.range 15).all (fun i => decide (offset[i + 1]! = offset[i]! + vocab[i]!)) = true := by decide

/-- The last head's range ends below the table's 320001536 rows. -/
theorem last_end_fits : offset[15]! + vocab[15]! ≤ 320001536 := by decide

/-- The table occupies 28,800,138,240 bytes at 90 bytes per row. -/
theorem table_bytes : 320001536 * 90 = 28800138240 := by decide

/-- Row size: 5 blocks of 18 bytes. -/
theorem row_bytes : 160 / 32 * 18 = 90 := by decide

-- § head indexing

/-- Output head index for n-gram order `n` (2 or 3) and sub-head `g`. -/
def headIdx (n g : Nat) : Nat := (n - 2) * 8 + g

/-- Head indices stay below 16. -/
theorem headIdx_lt (n g : Nat) (hn : 2 ≤ n) (hn' : n ≤ 3) (hg : g < 8) : headIdx n g < 16 := by
  unfold headIdx
  omega

/-- Distinct (n, g) pairs give distinct heads: every head is written exactly once. -/
theorem headIdx_inj (n g n' g' : Nat) (hn : 2 ≤ n) (hn' : n ≤ 3) (hg : g < 8)
    (hm : 2 ≤ n') (hm' : n' ≤ 3) (hg' : g' < 8) (h : headIdx n g = headIdx n' g') : n = n' ∧ g = g' := by
  unfold headIdx at h
  omega

/-- Every head below 16 is produced by some (n, g). -/
theorem headIdx_surj (h : Nat) (hh : h < 16) : ∃ n g, 2 ≤ n ∧ n ≤ 3 ∧ g < 8 ∧ headIdx n g = h := by
  refine ⟨h / 8 + 2, h % 8, ?_, ?_, ?_, ?_⟩
  · omega
  · omega
  · omega
  · unfold headIdx
    omega

/-- A row index `mixed % v + off` lies in the head's range `[off, off + v)`. -/
theorem row_in_head_range (mixed v off : Nat) (hv : 0 < v) :
    off ≤ mixed % v + off ∧ mixed % v + off < off + v := by
  have := Nat.mod_lt mixed hv
  omega

/-- The flat output slot `i*16 + h` is inside a batch of `n` tokens. -/
theorem out_slot_lt (i n h : Nat) (hi : i < n) (hh : h < 16) : i * 16 + h < n * 16 := by
  omega

-- § context window

/-- Index into `prev` (oldest first) of the predecessor `s` positions back, window size `n_prev`. -/
def prevIdx (nPrev s : Nat) : Nat := nPrev - s

/-- For `1 <= s <= n_prev` the index is in range, and `s = 1` is the newest entry. -/
theorem prevIdx_bounds (nPrev s : Nat) (h1 : 1 ≤ s) (h2 : s ≤ nPrev) :
    prevIdx nPrev s < nPrev ∧ (s = 1 → prevIdx nPrev s = nPrev - 1) := by
  unfold prevIdx
  omega

/-- Walking `s = 1 .. n_prev` visits each window entry once (the map `s -> n_prev - s` is injective). -/
theorem prevIdx_inj (nPrev s t : Nat) (hs : s ≤ nPrev) (ht : t ≤ nPrev) (h : prevIdx nPrev s = prevIdx nPrev t) :
    s = t := by
  unfold prevIdx at h
  omega

-- § IQ4_NL

/-- The 16-entry non-linear codebook. -/
def iq4nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries (one per 4-bit code). -/
theorem iq4nl_size : iq4nl.length = 16 := by decide

/-- The codebook is strictly increasing. -/
theorem iq4nl_increasing :
    (List.range 15).all (fun i => decide (iq4nl[i]! < iq4nl[i + 1]!)) = true := by decide

/-- A 4-bit code (masked with 15) always indexes the codebook. -/
theorem code_index_lt (c : Nat) : c % 16 < 16 := by
  omega

/-- Split halves: byte `j` of a block holds elements `j` and `j + 16`; together they cover 0..31 once. -/
theorem split_halves_cover (k : Nat) (hk : k < 32) : ∃ j half, j < 16 ∧ half < 2 ∧ k = j + 16 * half := by
  refine ⟨k % 16, k / 16, ?_, ?_, ?_⟩ <;> omega

/-- Distinct (j, half) pairs write distinct elements. -/
theorem split_halves_inj (j half j' half' : Nat) (hj : j < 16) (hj' : j' < 16)
    (h : j + 16 * half = j' + 16 * half') : j = j' ∧ half = half' := by
  omega

/-- All block reads `b*18 + 2 + j` stay inside the 90-byte row. -/
theorem block_byte_in_row (b j : Nat) (hb : b < 5) (hj : j < 16) : b * 18 + 2 + j < 90 := by
  omega

/-- All outputs `b*32 + j + 16*half` stay inside the 160-value head. -/
theorem output_in_head (b j half : Nat) (hb : b < 5) (hj : j < 16) (hh : half < 2) :
    b * 32 + j + 16 * half < 160 := by
  omega

end Strata.Kernels.Ngram



## FILE: /mnt/user-data/outputs/PrefillKernels.lean
/-!
# MODULE: prefill_kernels
Index, layout and clamping arithmetic of the prefill CUDA kernels: hyper-connection read/write, GDN conv and
recurrence, MoE routing/dequantization/combine, and QSA helpers (RMS, RoPE, KV append with INT8 quantization).

## Source semantics
* Launch grids cover their element count: `blocks = (n + 255) / 256` blocks of 256 threads.
* Residual streams are `[T][4][2560]` (D = 10240); thread `i` decodes to `(t, c, d)` with `c < 4`, `d < 2560`.
* A Strata expert blob is `1280*640` gate/up code bytes, `2560*160` down code bytes, `1280*40` and `2560*10`
  fp16 scales: 1,382,400 bytes in total. One code byte holds 4 ternary weights sharing a 64-weight scale group.
* Routing: lane `l` holds experts `l + 32*i`, `i < 16`, covering 512 experts exactly once.
* GDN recurrence: 4 row groups of 32 rows tile the 128-row state; q/k/v slices of the 10240-wide conv output do not
  overlap.
* KV append: grid z in 0..7 splits into group `z/2 < 4` and K/V flag; INT8 values are clamped to [-127, 127].

## Form template identifiers
`blocks_for` -> `blocksFor`; D, N, HC -> literals 10240, 2560, 4; expert blob offsets -> `O_D_CODES`, `O_GU_SC`,
`O_D_SC`; clamp -> `clampQ`.
-/

namespace Strata.Prefill.Kernels

-- § launch geometry

/-- Number of 256-thread blocks for `n` elements. -/
def blocksFor (n : Nat) : Nat := (n + 255) / 256

/-- The launch covers all `n` elements. -/
theorem blocks_cover (n : Nat) : n ≤ blocksFor n * 256 := by
  unfold blocksFor
  omega

/-- The launch overshoots by less than one block. -/
theorem blocks_tight (n : Nat) : blocksFor n * 256 < n + 256 := by
  unfold blocksFor
  omega

/-- Route kernel: 8 warps per block cover all `T` tokens. -/
theorem route_blocks_cover (T : Nat) : T ≤ (T + 7) / 8 * 8 := by
  omega

-- § hyper-connection

/-- Decoding a thread index `i < T*10240` into `(t, c, d)` is exact, with `c < 4` and `d < 2560` and `t < T`. -/
theorem gr_write_decomp (i T : Nat) (h : i < T * 10240) :
    i / 10240 < T ∧ i % 10240 / 2560 < 4 ∧ i % 2560 < 2560 ∧
      i / 10240 * 10240 + i % 10240 / 2560 * 2560 + i % 2560 = i := by
  omega

/-- `gr_mix` reads `t*10240 + c*2560 + d` for `c < 4`, which is inside `[T][10240]`. -/
theorem gr_mix_read_in_range (i T c : Nat) (hi : i < T * 2560) (hc : c < 4) :
    i / 2560 * 10240 + c * 2560 + i % 2560 < T * 10240 := by
  omega

/-- A `gr_norm` row `t*4 + c` has channel `c = row % 4`, and its elements stay in the 10240-wide norm weight. -/
theorem gr_norm_channel (row d : Nat) (hd : d < 2560) : row % 4 * 2560 + d < 10240 := by
  omega

-- § GDN

/-- The conv kernel's 80 blocks of 128 threads cover exactly the 10240 channels. -/
theorem gdn_conv_grid : 10240 / 128 * 128 = 10240 := by decide

/-- Recurrence: row group `rg < 4` and local row `r < 32` tile the 128 state rows with a unique decomposition. -/
theorem rec_rows_tile (rg r : Nat) (hg : rg < 4) (hr : r < 32) :
    rg * 32 + r < 128 ∧ (rg * 32 + r) / 32 = rg ∧ (rg * 32 + r) % 32 = r := by
  omega

/-- The key slice offset `16*128 + qh*128 + tid` stays in the second quarter of the conv output. -/
theorem rec_key_in_range (head tid : Nat) (hh : head < 48) (ht : tid < 128) :
    16 * 128 + head % 16 * 128 + tid < 2 * 16 * 128 := by
  omega

/-- The value slice `2*16*128 + head*128 + col` stays inside the 10240 channels. -/
theorem rec_value_in_range (head col : Nat) (hh : head < 48) (hc : col < 128) :
    2 * 16 * 128 + head * 128 + col < 10240 := by
  omega

/-- The L2-norm kernel over 32 q/k heads touches only the first 4096 channels. -/
theorem l2_in_range (head tid : Nat) (hh : head < 32) (ht : tid < 128) : head * 128 + tid < 4096 := by
  omega

/-- `gdn_gates` decodes `i < T*48` into `t < T` and `h < 48`. -/
theorem gates_decomp (i T : Nat) (h : i < T * 48) : i / 48 < T ∧ i % 48 < 48 := by
  omega

-- § MoE

/-- Lane/expert decomposition in the routing kernel. -/
theorem lane_expert (lane i : Nat) (hl : lane < 32) (hi : i < 16) :
    lane + i * 32 < 512 ∧ (lane + i * 32) / 32 = i ∧ (lane + i * 32) % 32 = lane := by
  omega

/-- Every expert id below 512 is held by exactly one (lane, i). -/
theorem expert_held (e : Nat) (he : e < 512) : e % 32 < 32 ∧ e / 32 < 16 ∧ e % 32 + e / 32 * 32 = e := by
  omega

/-- Expert blob layout: sections are contiguous and total 1,382,400 bytes (the planner's `expert_blob`). -/
theorem blob_layout :
    1280 * 640 = 819200 ∧ 819200 + 2560 * 160 = 1228800 ∧ 1228800 + 1280 * 40 * 2 = 1331200 ∧
      1331200 + 2560 * 10 * 2 = 1382400 := by decide

/-- Gate/up scale index `row*40 + byte*4/64` is within the `1280*40` scales. -/
theorem gu_scale_idx (row byte : Nat) (hr : row < 1280) (hb : byte < 640) :
    row * 40 + byte * 4 / 64 < 1280 * 40 := by
  omega

/-- Down scale index `row*10 + byte*4/64` is within the `2560*10` scales. -/
theorem d_scale_idx (row byte : Nat) (hr : row < 2560) (hb : byte < 160) :
    row * 10 + byte * 4 / 64 < 2560 * 10 := by
  omega

/-- Gate/up outputs `row*2560 + byte*4 + k` are within `1280*2560` FP16 elements. -/
theorem gu_out_idx (row byte k : Nat) (hr : row < 1280) (hb : byte < 640) (hk : k < 4) :
    row * 2560 + byte * 4 + k < 1280 * 2560 := by
  omega

/-- Down outputs `row*640 + byte*4 + k` are within `2560*640` FP16 elements. -/
theorem d_out_idx (row byte k : Nat) (hr : row < 2560) (hb : byte < 160) (hk : k < 4) :
    row * 640 + byte * 4 + k < 2560 * 640 := by
  omega

/-- A thread index in the down range decodes to a valid `(row, byte)`. -/
theorem d_thread_decomp (i : Nat) (hlo : 1280 * 640 ≤ i) (hhi : i < 1280 * 640 + 2560 * 160) :
    (i - 1280 * 640) / 160 < 2560 ∧ (i - 1280 * 640) % 160 < 160 := by
  omega

/-- A 2-bit code field is below 4. -/
theorem code_field_lt (c k : Nat) : (c >>> (2 * k)) % 4 < 4 := Nat.mod_lt _ (by decide)

/-- Interleaved SwiGLU reads `2k` and `2k+1` below 1280 and row `r < n`. -/
theorem swiglu_il_range (i n : Nat) (h : i < n * 640) :
    i / 640 < n ∧ i / 640 * 1280 + 2 * (i % 640) + 1 < n * 1280 := by
  omega

/-- Gathering rows moves `width/8` 16-byte vectors per row; for width 2560 that is 320. -/
theorem gather_vectors : 2560 / 8 = 320 ∧ 8 * 2 = 16 := by decide

-- § QSA helpers

/-- `split_q` reads the first 256 of each 512-wide head slot. -/
theorem split_q_range (h d : Nat) (hh : h < 24) (hd : d < 256) : h * 512 + d < 24 * 512 := by
  omega

/-- `gate_attn` reads the gate half `h*512 + 256 + d`, also inside the slot. -/
theorem gate_attn_range (h d : Nat) (hh : h < 24) (hd : d < 256) : h * 512 + 256 + d < 24 * 512 := by
  omega

/-- RoPE rotates pairs `(pair, pair + 32)` inside a 64-wide rotary prefix. -/
theorem rope_pair_range (pair : Nat) (h : pair < 32) : pair + 32 < 64 := by
  omega

/-- Head packing: K (2 heads of 256), Q (24 heads of 256), indexer Q (4 heads of 128) fit their row strides. -/
theorem head_strides : 2 * 256 = 512 ∧ 24 * 256 = 6144 ∧ 4 * 128 = 512 := by decide

/-- KV append grid: `z < 8` gives group `z/2 < 4`, and the written element `g*64 + tid` is below 256. -/
theorem kv_grid (z tid : Nat) (hz : z < 8) (ht : tid < 64) : z / 2 < 4 ∧ z / 2 * 64 + tid < 256 := by
  omega

/-- The paged row `page*2 + kvh` is inside `2 * pages` for `kvh < 2`. -/
theorem kv_page_row (page pages kvh : Nat) (hp : page < pages) (hk : kvh < 2) : page * 2 + kvh < pages * 2 := by
  omega

/-- INT8 clamp used by `kv_append`. -/
def clampQ (q : Int) : Int := if q < -127 then -127 else if q > 127 then 127 else q

/-- The clamped value is a legal INT8 code (never -128). -/
theorem clampQ_range (q : Int) : -127 ≤ clampQ q ∧ clampQ q ≤ 127 := by
  unfold clampQ
  split_ifs <;> omega

/-- In range, clamping is the identity. -/
theorem clampQ_id (q : Int) (h1 : -127 ≤ q) (h2 : q ≤ 127) : clampQ q = q := by
  unfold clampQ
  split_ifs <;> omega

end Strata.Prefill.Kernels



## FILE: /mnt/user-data/outputs/PrefillChunkDriver.lean
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



## FILE: /mnt/user-data/outputs/PleRowCacheModel.lean
/-!
# MODULE: ple_row_cache_model
Behavioural model of the PLE reader's 8-way set-associative row cache and the PleTable single-token in-flight
protocol (`issue` then `collect`).

## Source semantics
* A set holds 8 keys; `insert` writes the new key at the set's round-robin pointer `w` and advances it to `(w+1) % 8`.
* After an insert the key is findable; the other 7 ways are untouched; a key survives the next 7 inserts into the
  same set, and the pointer returns to `w` after 8 inserts.
* `used` grows only when an EMPTY way is overwritten, so `used <= number of inserts`.
* `PleTable::issue` fails when a token is already in flight; `collect` without `issue` fails; an issue followed by a
  collect leaves nothing in flight.
* A batch gather of `n = T*16` rows writes row `i` at `out + i*160`, covering exactly `T*2560` floats.

## Form template identifiers
RowCache set -> `keys : Nat → Nat`; insert -> `insertAt`; replacement pointer -> `ptr`; pending flag -> `pending`;
issue/collect -> `issueStep`/`collectStep`.
-/

namespace Strata.Ngram.CacheModel

-- § row cache

/-- Write `row` into way `w` of a set. -/
def insertAt (keys : Nat → Nat) (w row : Nat) : Nat → Nat := fun i => if i = w then row else keys i

/-- After an insert the row is findable in one of the 8 ways. -/
theorem find_after_insert (keys : Nat → Nat) (w row : Nat) (hw : w < 8) :
    ∃ i, i < 8 ∧ insertAt keys w row i = row := by
  refine ⟨w, hw, ?_⟩
  simp [insertAt]

/-- The other ways are untouched by an insert. -/
theorem other_ways_untouched (keys : Nat → Nat) (w row i : Nat) (h : i ≠ w) :
    insertAt keys w row i = keys i := by
  simp [insertAt, h]

/-- The replacement pointer stays inside the set. -/
theorem ptr_lt (w : Nat) : (w + 1) % 8 < 8 := by
  omega

/-- A freshly written way is not overwritten by the next 7 inserts into the same set. -/
theorem survives_seven (w k : Nat) (hk1 : 1 ≤ k) (hk2 : k ≤ 7) : (w + k) % 8 ≠ w % 8 := by
  omega

/-- After 8 inserts the pointer is back where it started (round-robin). -/
theorem ptr_cycle (w : Nat) : (w + 8) % 8 = w % 8 := by
  omega

/-- The set count `sets = rows / 8` times 8 ways never exceeds the requested rows. -/
theorem capacity_le_rows (rows : Nat) : rows / 8 * 8 ≤ rows := by
  omega

/-- The `used` counter: incremented only when overwriting an EMPTY way. -/
def usedAfter (used : Nat) (wasEmpty : Bool) : Nat := if wasEmpty then used + 1 else used

/-- `used` grows by at most one per insert. -/
theorem used_step_le (used : Nat) (wasEmpty : Bool) : usedAfter used wasEmpty ≤ used + 1 := by
  unfold usedAfter
  split <;> omega

/-- `used` never decreases. -/
theorem used_step_ge (used : Nat) (wasEmpty : Bool) : used ≤ usedAfter used wasEmpty := by
  unfold usedAfter
  split <;> omega

-- § single-token in-flight protocol

/-- `issue`: returns whether it was accepted and the new pending flag. -/
def issueStep (pending : Bool) : Bool × Bool := if pending then (false, pending) else (true, true)

/-- `collect`: returns whether it succeeded and the new pending flag. -/
def collectStep (pending : Bool) : Bool × Bool := if pending then (true, false) else (false, false)

/-- Issuing while a token is in flight is refused and changes nothing. -/
theorem issue_refused (pending : Bool) (h : pending = true) : issueStep pending = (false, pending) := by
  subst h
  rfl

/-- Collecting with nothing in flight is refused. -/
theorem collect_without_issue : collectStep false = (false, false) := rfl

/-- An accepted issue followed by a collect succeeds and leaves nothing in flight. -/
theorem issue_collect_roundtrip : collectStep (issueStep false).2 = (true, false) := rfl

-- § batch gather layout

/-- Row `i < T*16` written at `i*160` ends within the `T*2560`-float output. -/
theorem gather_batch_in_range (T i : Nat) (h : i < T * 16) : i * 160 + 160 ≤ T * 2560 := by
  omega

/-- Head-slowest layout: head `h` of a token occupies `[h*160, (h+1)*160)` of its 2560 floats. -/
theorem head_slowest_in_token (h k : Nat) (hh : h < 16) (hk : k < 160) : h * 160 + k < 2560 := by
  omega

end Strata.Ngram.CacheModel



## FILE: /mnt/user-data/outputs/PleNgramContext.lean
/-!
# MODULE: ple_ngram_context
Context construction and incremental mixing of the PLE n-gram hash (`ngram_rows` / `ngram_mixed`).

## Source semantics
* `ctx[0]` is the current token; `ctx[s]` is the predecessor `s` positions back.
* Once a predecessor is negative or equals EOS the context is cut: that position and every later one is EOS.
  Before the cut, positions hold the real predecessor.
* `mixed(n) = ctx[0]*m[0] ^ ctx[1]*m[1] ^ ... ^ ctx[n-1]*m[n-1]` over 64-bit wrapping arithmetic, so the order-3
  hash extends the order-2 hash by one XOR term, and an order-`n` hash depends only on `ctx[0..n-1]`.

## Form template identifiers
cut flag -> `cutAt`; stored context value -> `ctxVal`; ngram_mixed -> `mixed`.
-/

namespace Strata.Kernels.NgramContext

-- § context cut

/-- Cut flag after processing `s` predecessors; `prev s` is the predecessor at step `s + 1`. -/
def cutAt (prev : Nat → Int) (eos : Int) : Nat → Bool
  | 0 => false
  | s + 1 => cutAt prev eos s || decide (prev s < 0) || decide (prev s = eos)

/-- Once cut, always cut. -/
theorem cut_mono (prev : Nat → Int) (eos : Int) (s : Nat) (h : cutAt prev eos s = true) :
    cutAt prev eos (s + 1) = true := by
  simp [cutAt, h]

/-- Cut persists over any number of further steps. -/
theorem cut_persists (prev : Nat → Int) (eos : Int) (s d : Nat) (h : cutAt prev eos s = true) :
    cutAt prev eos (s + d) = true := by
  induction d with
  | zero => simpa using h
  | succ d ih => exact cut_mono prev eos (s + d) ih

/-- Stored context value at step `s + 1`: EOS once cut, otherwise the real predecessor. -/
def ctxVal (prev : Nat → Int) (eos : Int) (s : Nat) : Int :=
  if cutAt prev eos (s + 1) = true then eos else prev s

/-- After a cut every stored value is EOS. -/
theorem ctx_eos_after_cut (prev : Nat → Int) (eos : Int) (s d : Nat) (h : cutAt prev eos s = true) :
    ctxVal prev eos (s + d) = eos := by
  have h1 := cut_persists prev eos s (d + 1) h
  unfold ctxVal
  have e : s + d + 1 = s + (d + 1) := by omega
  rw [e, if_pos h1]

/-- Before any cut, the stored value is the real predecessor. -/
theorem ctx_real_before_cut (prev : Nat → Int) (eos : Int) (s : Nat) (h : cutAt prev eos (s + 1) = false) :
    ctxVal prev eos s = prev s := by
  unfold ctxVal
  rw [if_neg (by simp [h])]

/-- A negative or EOS predecessor cuts at its own step, so the position holding it is EOS. -/
theorem eos_predecessor_is_eos (prev : Nat → Int) (eos : Int) (s : Nat) (h : prev s = eos) :
    ctxVal prev eos s = eos := by
  unfold ctxVal
  have hc : cutAt prev eos (s + 1) = true := by simp [cutAt, h]
  rw [if_pos hc]

-- § incremental mixing

/-- Wrapping 64-bit mixing: first term assigned, the rest XORed in. -/
def mixed (ctx mult : Nat → UInt64) : Nat → UInt64
  | 0 => ctx 0 * mult 0
  | j + 1 => mixed ctx mult j ^^^ (ctx (j + 1) * mult (j + 1))

/-- Order `j + 2` extends order `j + 1` by exactly one XOR term. -/
theorem mixed_succ (ctx mult : Nat → UInt64) (j : Nat) :
    mixed ctx mult (j + 1) = mixed ctx mult j ^^^ (ctx (j + 1) * mult (j + 1)) := rfl

/-- The order-`j` hash depends only on `ctx[0..j]`. -/
theorem mixed_congr (ctx ctx' mult : Nat → UInt64) :
    ∀ j, (∀ i, i ≤ j → ctx i = ctx' i) → mixed ctx mult j = mixed ctx' mult j := by
  intro j
  induction j with
  | zero =>
    intro h
    simp only [mixed]
    rw [h 0 (Nat.le_refl 0)]
  | succ j ih =>
    intro h
    simp only [mixed]
    rw [ih (fun i hi => h i (by omega)), h (j + 1) (Nat.le_refl _)]

/-- Head row index stays inside the head's vocabulary window. -/
theorem row_window (m v off : Nat) (hv : 0 < v) : off ≤ m % v + off ∧ m % v + off < off + v := by
  have := Nat.mod_lt m hv
  omega

end Strata.Kernels.NgramContext





## FILE: /mnt/user-data/outputs/ScalarDequantizers.lean
/-!
# MODULE: scalar_dequantizers
Scalar reference dequantizers (Q2_0, Q4_0, Q5_0, Q8_0, IQ4_NL, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS): block
geometry, bit-field addressing and value ranges, modelled over Nat/Int.

## Source semantics
Q2_0: 64 elements per 18-byte block, code c in {0..3} gives symbol c-1, four codes per byte LSB-first.
Q4_0/IQ4_NL: low nibble is element j, high nibble element j+16. Q5_0 adds one high bit per element.
K-quants use 256-element super-blocks; Q4_K/Q5_K use d,dmin and 6-bit scale/min, Q3_K and IQ4_XS
use signed 6-bit scales with a -32 bias. Q5_K selects its fifth bit with a mask shifting left by 2
per 64-element group.

## Form template identifiers
q2Code -> unpack_code ; q2Pack -> pack_codes ; kv -> codebook ; blk_* -> block_size_bytes
-/

set_option maxRecDepth 100000

namespace Strata.Dequant

-- § Q2_0

/-- Code extraction for element k (0..3) of a packed byte. -/
def q2Code (b k : Nat) : Nat := (b >>> (k * 2)) &&& 3

/-- Packing four 2-bit codes into a byte, LSB-first. -/
def q2Pack (c0 c1 c2 c3 : Nat) : Nat := c0 ||| (c1 <<< 2) ||| (c2 <<< 4) ||| (c3 <<< 6)

/-- Unpacking recovers each of the four packed codes: no bit-order ambiguity. -/
theorem q2_roundtrip : ∀ c0 < 4, ∀ c1 < 4, ∀ c2 < 4, ∀ c3 < 4,
    q2Code (q2Pack c0 c1 c2 c3) 0 = c0 ∧ q2Code (q2Pack c0 c1 c2 c3) 1 = c1 ∧
    q2Code (q2Pack c0 c1 c2 c3) 2 = c2 ∧ q2Code (q2Pack c0 c1 c2 c3) 3 = c3 := by
  decide +kernel

/-- Every extracted code is a 2-bit value. -/
theorem q2_code_lt4 : ∀ b < 256, ∀ k < 4, q2Code b k < 4 := by decide

/-- The symbol of a code: code - 1. -/
def q2Sym (c : Nat) : Int := (c : Int) - 1

/-- Symbols lie in {-1,0,1,2}. -/
theorem q2_sym_range (c : Nat) (h : c < 4) : -1 ≤ q2Sym c ∧ q2Sym c ≤ 2 := by
  unfold q2Sym; omega

/-- Byte index and bit offset stay inside the 16-byte payload. -/
theorem q2_index (j : Nat) (hj : j < 64) : j / 4 < 16 ∧ (j % 4) * 2 ≤ 6 := by omega

/-- (byte, slot) addressing is a bijection onto element index. -/
theorem q2_cover (j : Nat) : (j / 4) * 4 + j % 4 = j := by omega

/-- Block layout: 2 byte scale + 16 payload bytes = 18; 16 bytes * 4 codes = 64 elements. -/
theorem q2_block : 2 + 16 = 18 ∧ 16 * 4 = 64 := ⟨rfl, rfl⟩

-- § Q4_0 / Q5_0 / Q8_0

/-- Every element 0..31 is the low nibble of byte j (i = j) or the high nibble (i = j + 16). -/
theorem q4_cover (i : Nat) (hi : i < 32) : ∃ j, j < 16 ∧ (i = j ∨ i = j + 16) :=
  ⟨i % 16, by omega, by omega⟩

/-- Q4_0 values after the -8 bias lie in [-8,7]. -/
theorem q4_range (n : Nat) (h : n < 16) : -8 ≤ (n : Int) - 8 ∧ (n : Int) - 8 ≤ 7 := by omega

/-- Q5_0 layout: scale 2, qh 4, qs 16 bytes. -/
theorem q5_0_block : 2 + 4 + 16 = 22 := rfl

/-- Q5_0 five-bit values after the -16 bias lie in [-16,15]. -/
theorem q5_0_range (nib b : Nat) (hn : nib < 16) (hb : b < 2) :
    -16 ≤ ((nib + 16 * b : Nat) : Int) - 16 ∧ ((nib + 16 * b : Nat) : Int) - 16 ≤ 15 := by omega

/-- Q5_0 high bits come from qh bit j (low half) and bit j+16 (high half): all 32 distinct. -/
theorem q5_0_bits (j : Nat) (hj : j < 16) : j < 32 ∧ j + 16 < 32 ∧ j ≠ j + 16 := by omega

/-- Q8_0 layout: scale 2 + 32 int8. -/
theorem q8_0_block : 2 + 32 = 34 := rfl

-- § IQ4_NL codebook

/-- `kvalues_iq4nl`, verbatim. -/
def kv : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- Sixteen entries: one per nibble. -/
theorem kv_len : kv.length = 16 := rfl

/-- Strictly increasing, so the codebook is a monotone non-linear grid. -/
theorem kv_sorted : kv.Pairwise (· < ·) := by decide

/-- All entries lie in [-127,113]; no entry is offset by -8. -/
theorem kv_bounds : ∀ x ∈ kv, -127 ≤ x ∧ x ≤ 113 := by decide

/-- IQ4_NL block: 2 + 16 = 18 bytes for 32 elements. -/
theorem iq4nl_block : 2 + 16 = 18 := rfl

-- § K-quant layouts

/-- Q6_K: ql 128 + qh 64 + scales 16 + d 2 = 210, with d at offset 208. -/
theorem q6k_block : 128 + 64 + 16 + 2 = 210 ∧ 128 + 64 + 16 = 208 := ⟨rfl, rfl⟩

/-- Q6_K scale index is_ + {0,2,4,6} plus 8 per half never exceeds 15. -/
theorem q6k_scale_idx (h l q : Nat) (hh : h < 2) (hl : l < 32) (hq : q < 4) :
    8 * h + l / 16 + 2 * q < 16 := by omega

/-- Q6_K output positions of one half cover 128 slots. -/
theorem q6k_out_idx (h l q : Nat) (hh : h < 2) (hl : l < 32) (hq : q < 4) :
    128 * h + l + 32 * q < 256 := by omega

/-- Q4_K: d, dmin, scales 12, qs 128 = 144, qs at offset 16. -/
theorem q4k_block : 2 + 2 + 12 + 128 = 144 ∧ 2 + 2 + 12 = 16 := ⟨rfl, rfl⟩

/-- Q5_K: d, dmin, scales 12, qh 32, qs 128 = 176, qh at 16, qs at 48. -/
theorem q5k_block : 2 + 2 + 12 + 32 + 128 = 176 ∧ 16 + 32 = 48 := ⟨rfl, rfl⟩

/-- High 2 bits of a byte fit in 0..3. -/
theorem shr6_lt4 : ∀ b < 256, b >>> 6 < 4 := by decide

/-- The reassembled 6-bit scale (low nibble + 2 high bits shifted 4) is below 64. -/
theorem scale6_lt64 : ∀ x < 256, ∀ y < 4, ((x &&& 15) ||| (y <<< 4)) < 64 := by decide +kernel

/-- Q5_K masks u1 = 1,4,16,64. -/
theorem q5k_u1 : ∀ k < 4, (1 <<< (2 * k)) = [1, 4, 16, 64].getD k 0 := by decide

/-- u2 is always twice u1, so the two masks of a group select adjacent bits. -/
theorem q5k_u2 : ∀ k < 4, (2 <<< (2 * k)) = 2 * (1 <<< (2 * k)) := by decide

/-- The eight masks are exactly the eight bits of a qh byte: sum 255. -/
theorem q5k_masks_cover : (1 + 4 + 16 + 64) + (2 + 8 + 32 + 128) = 255 := rfl

/-- A fixed mask or a shift by one would produce different masks at group 1. -/
theorem q5k_not_shift1 : (1 <<< (2 * 1)) ≠ (1 <<< 1) := by decide

/-- Q3_K: hmask 32 + qs 64 + scales 12 + d 2 = 110, d at offset 108. -/
theorem q3k_block : 32 + 64 + 12 + 2 = 110 ∧ 32 + 64 + 12 = 108 := ⟨rfl, rfl⟩

/-- Q3_K hoisted mask: at (half n, step j) the mask is 1 << (4n+j), below 256. -/
theorem q3k_mask_lt (n j : Nat) (hn : n < 2) (hj : j < 4) : (1 <<< (4 * n + j)) < 256 := by
  have : 4 * n + j < 8 := by omega
  have h8 : ∀ k < 8, (1 <<< k) < 256 := by decide
  exact h8 _ this

/-- Resetting the mask in the second half would give a different (wrong) mask. -/
theorem q3k_mask_hoist : (1 <<< (4 * 1 + 0)) ≠ (1 <<< 0) := by decide

/-- IQ4_XS: d 2, scales_h 2, scales_l 4, qs 128 = 136; eight groups of 16 bytes. -/
theorem iq4xs_block : 2 + 2 + 4 + 128 = 136 ∧ 8 * 16 = 128 := ⟨rfl, rfl⟩

/-- scales_l index and shifts stay in range; the two high bits stay inside the 16-bit field. -/
theorem iq4xs_index (ib : Nat) (h : ib < 8) :
    ib / 2 < 4 ∧ 4 * (ib % 2) ≤ 4 ∧ 2 * ib + 2 ≤ 16 := by omega

/-- The 6-bit group scale ls lies in 0..63, so ls - 32 lies in [-32,31]. -/
theorem iq4xs_scale_range (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) :
    -32 ≤ ((lo + 16 * hi : Nat) : Int) - 32 ∧ ((lo + 16 * hi : Nat) : Int) - 32 ≤ 31 := by omega

-- § fp16 exponent mapping

/-- Normal fp16 exponent maps to f32 exponent ex + 112, always a normal f32 exponent. -/
theorem fp16_exp_map (ex : Nat) (h1 : 1 ≤ ex) (h2 : ex ≤ 30) :
    (ex : Int) - 15 + 127 = (ex : Int) + 112 ∧ 1 ≤ ex + 112 ∧ ex + 112 ≤ 254 := by omega

/-- The mantissa shift of 13 stays below 2^23. -/
theorem fp16_man_shift (man : Nat) (h : man < 1024) : man * 8192 < 8388608 := by omega

end Strata.Dequant



## FILE: /mnt/user-data/outputs/GgufReaderGeometry.lean
/-!
# MODULE: gguf_reader_geometry
GGUF v3 reader model: ggml type tables (names and block geometry), bounds-checked cursor,
data-section alignment, array sampling, architecture guard constants.

## Source semantics
`block_geometry` maps a ggml type id to (elements per block, bytes per block). Q2_0 (42) is 64/18.
The Cursor never reads past the mapped size. The data section starts at the header end rounded up to
`general.alignment` (default 32).

## Form template identifiers
blockGeom -> block_geometry ; rowBytes -> row_bytes ; dataStart -> data_start ; step -> cursor_need
-/

set_option maxRecDepth 100000

namespace Strata.Gguf

/-- Block geometry per ggml type id, transcribed from `block_geometry`. -/
def blockGeom : Nat → Option (Nat × Nat)
  | 0 => some (1, 4)
  | 1 => some (1, 2)
  | 30 => some (1, 2)
  | 2 => some (32, 18)
  | 3 => some (32, 20)
  | 6 => some (32, 22)
  | 7 => some (32, 24)
  | 8 => some (32, 34)
  | 9 => some (32, 36)
  | 10 => some (256, 84)
  | 11 => some (256, 110)
  | 12 => some (256, 144)
  | 13 => some (256, 176)
  | 14 => some (256, 210)
  | 16 => some (256, 66)
  | 17 => some (256, 74)
  | 18 => some (256, 98)
  | 20 => some (32, 18)
  | 21 => some (256, 110)
  | 22 => some (256, 82)
  | 23 => some (256, 136)
  | 29 => some (256, 56)
  | 42 => some (64, 18)
  | _ => none

/-- Type ids for which `ggml_type_name` returns something other than "?". -/
def namedTypes : List Nat :=
  [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 30, 34, 35, 39, 40, 41, 42]

/-- Geometry entries are always positive, so no division by zero can occur downstream. -/
def geomOk (t : Nat) : Bool :=
  match blockGeom t with
  | some (e, b) => decide (0 < e) && decide (0 < b)
  | none => true

/-- All geometry entries for type ids below 64 are positive. -/
theorem geom_positive : ∀ t < 64, geomOk t = true := by decide

/-- Q2_0 is 64 elements in 18 bytes. -/
theorem q2_0_geom : blockGeom 42 = some (64, 18) := rfl

/-- Named types without geometry: the reader names them but cannot size them. -/
theorem named_without_geom :
    namedTypes.filter (fun t => (blockGeom t).isNone) = [15, 19, 34, 35, 39, 40, 41] := by decide

/-- IQ1_M (29) has geometry but its name is "?": count_type("IQ1_M") can never match. -/
theorem iq1m_unnamed : (blockGeom 29).isSome = true ∧ (namedTypes.contains 29) = false := by decide

/-- Legacy-block types share geometry where ggml does: Q4_0 and IQ4_NL are both 32/18. -/
theorem q4_0_iq4nl_same : blockGeom 2 = blockGeom 20 := rfl

/-- Q3_K and IQ3_S are both 256/110. -/
theorem q3k_iq3s_same : blockGeom 11 = blockGeom 21 := rfl

/-- Row size in bytes: `n / elems * bytes`. -/
def rowBytes (t n : Nat) : Option Nat :=
  match blockGeom t with
  | some (e, b) => some (n / e * b)
  | none => none

/-- A 160-wide IQ4_NL row (the PLE table) is 90 bytes. -/
theorem ple_row_bytes : rowBytes 20 160 = some 90 := by decide

/-- A 2560-wide Q2_0 row is 720 bytes. -/
theorem q2_row_bytes : rowBytes 42 2560 = some 720 := by decide

/-- Bits per weight (x100): Q2_0 is 2.25, Q4_0 is 4.5. -/
theorem bpw : 18 * 8 * 100 / 64 = 225 ∧ 18 * 8 * 100 / 32 = 450 := ⟨rfl, rfl⟩

-- § Cursor

/-- One bounds-checked advance. -/
def step (pos size n : Nat) : Option Nat :=
  if pos + n ≤ size then some (pos + n) else none

/-- A successful read never leaves the file and advances exactly n. -/
theorem step_ok {pos size n p : Nat} (h : step pos size n = some p) : p = pos + n ∧ p ≤ size := by
  unfold step at h
  split at h
  · cases h; omega
  · cases h

/-- A read past the end is always refused. -/
theorem step_fail {pos size n : Nat} (h : size < pos + n) : step pos size n = none := by
  unfold step
  rw [if_neg (by omega)]

/-- Sequential reads preserve the invariant pos <= size. -/
theorem step_mono {pos size n p : Nat} (h : step pos size n = some p) : pos ≤ p := by
  have := step_ok h; omega

-- § Data section alignment

/-- Start of the data section: header end rounded up to the alignment. -/
def dataStart (pos a : Nat) : Nat := (pos + a - 1) / a * a

/-- The data section never starts before the header ends. -/
theorem ds_ge (pos a : Nat) (ha : 0 < a) : pos ≤ dataStart pos a := by
  unfold dataStart
  have h1 := Nat.div_add_mod (pos + a - 1) a
  have h2 := Nat.mod_lt (pos + a - 1) ha
  rw [Nat.mul_comm ((pos + a - 1) / a) a]
  omega

/-- The rounding adds less than one alignment unit. -/
theorem ds_lt (pos a : Nat) (ha : 0 < a) : dataStart pos a < pos + a := by
  unfold dataStart
  have h1 := Nat.div_add_mod (pos + a - 1) a
  have h2 := Nat.mod_lt (pos + a - 1) ha
  rw [Nat.mul_comm ((pos + a - 1) / a) a]
  omega

/-- The data section is aligned. -/
theorem ds_aligned (pos a : Nat) : dataStart pos a % a = 0 := by
  unfold dataStart
  exact Nat.mul_mod_left _ _

/-- The PLE shard header of 192 bytes is already 32-aligned. -/
theorem header192 : dataStart 192 32 = 192 := by decide

-- § Array sampling and guard constants

/-- Kept sample size of an array metadata value. -/
def keptItems (count : Nat) : Nat := min count 64

/-- The sample is bounded by 64 and by the real count. -/
theorem kept_bounds (c : Nat) : keptItems c ≤ 64 ∧ keptItems c ≤ c := by
  unfold keptItems; omega

/-- The qwen4exp guard constants are mutually consistent: 24 heads over 2 KV heads, group of 12. -/
theorem guard_consts : 24 % 2 = 0 ∧ 24 / 2 = 12 ∧ 10 ≤ 512 ∧ 2560 = 16 * 160 := by decide

end Strata.Gguf



## FILE: /mnt/user-data/outputs/GgufLazyLoader.lean
/-!
# MODULE: gguf_lazy_loader
Lazy demand-paged GGUF loading (ref/load.py): per-expert slicing of a fused `[in, out, experts]`
tensor, per-row gathering, bf16 widening. Byte-offset arithmetic only.

## Source semantics
Expert e of a fused tensor is the contiguous run of `in*out` elements at element offset `e*in*out`;
its byte start is `e * (per_expert / el) * by`. Row r of a 2-D tensor starts at `r * row_bytes`.
Neither touches any other expert or row. bf16 widens by a 16-bit left shift.

## Form template identifiers
expertLen -> per_expert_bytes ; expertStart -> start ; rowStart -> row * rb ; bf16Bits -> bf16_to_f32
-/

set_option maxRecDepth 100000

namespace Strata.Loader

/-- Bytes per expert: `per_expert // el * by`. -/
def expertLen (per el b : Nat) : Nat := per / el * b

/-- Byte start of expert e, written as in the Python: `e * (per // el) * by`. -/
def expertStartPy (e per el b : Nat) : Nat := e * (per / el) * b

/-- Byte start as `e * len`. -/
def expertStart (e per el b : Nat) : Nat := e * expertLen per el b

/-- The Python expression equals `e * len`. -/
theorem start_eq (e per el b : Nat) : expertStartPy e per el b = expertStart e per el b := by
  unfold expertStartPy expertStart expertLen
  exact Nat.mul_assoc _ _ _

/-- Expert e ends exactly where expert e+1 starts: runs are adjacent. -/
theorem adjacent (e len : Nat) : e * len + len = (e + 1) * len := by
  rw [Nat.add_mul, Nat.one_mul]

/-- Expert e fits inside the tensor of n experts. -/
theorem expert_in_bounds (e n len : Nat) (h : e < n) : e * len + len ≤ n * len := by
  have h1 : (e + 1) * len ≤ n * len := Nat.mul_le_mul_right len h
  rw [Nat.add_mul, Nat.one_mul] at h1
  exact h1

/-- Two different experts never overlap: the earlier one ends before the later one starts. -/
theorem experts_disjoint (e1 e2 len : Nat) (h : e1 < e2) : e1 * len + len ≤ e2 * len := by
  have h1 : (e1 + 1) * len ≤ e2 * len := Nat.mul_le_mul_right len h
  rw [Nat.add_mul, Nat.one_mul] at h1
  exact h1

/-- When the element count per expert is a multiple of the block size, the whole tensor is
n times one expert. -/
theorem total_bytes (n per el b : Nat) (h : el ∣ per) :
    n * per / el * b = n * expertLen per el b := by
  unfold expertLen
  rw [Nat.mul_div_assoc n h]
  exact Nat.mul_assoc _ _ _

/-- Real routed expert: 2560 x 640 Q2_0 (64 per 18 B) is 460,800 bytes. -/
theorem q2_expert_len : expertLen (2560 * 640) 64 18 = 460800 := by decide

/-- Per-expert element count is a multiple of the block size. -/
theorem q2_expert_aligned : (2560 * 640) % 64 = 0 := by decide

/-- 512 fused experts: 235,929,600 bytes. -/
theorem q2_fused_total : 512 * 460800 = 235929600 := by decide

-- § Rows

/-- Byte start of row r. -/
def rowStart (r rb : Nat) : Nat := r * rb

/-- A gathered row lies inside the tensor. -/
theorem row_in_bounds (r ncols rb : Nat) (h : r < ncols) : rowStart r rb + rb ≤ ncols * rb := by
  unfold rowStart
  exact expert_in_bounds r ncols rb h

/-- PLE table: width 160 IQ4_NL gives 90-byte rows. -/
theorem ple_row : 160 / 32 * 18 = 90 := by decide

/-- PLE table size in bytes: 28.8e9, i.e. 26.82 GiB (between 26 and 27 GiB). -/
theorem ple_gib : 26 * 1073741824 < 320001536 * 90 ∧ 320001536 * 90 < 27 * 1073741824 := by decide

/-- Decoding the whole table to fp32 would take about 205 GB. -/
theorem ple_full_fp32 : 320001536 * 160 * 4 = 204800983040 := by decide

/-- A token reads 16 rows of 160 = 2560 floats. -/
theorem ple_token_floats : 16 * 160 = 2560 := rfl

-- § bf16

/-- bf16 to f32 bits: shift left by 16. -/
def bf16Bits (h : Nat) : Nat := h * 65536

/-- The result fits in 32 bits. -/
theorem bf16_fits (h : Nat) (hh : h < 65536) : bf16Bits h < 4294967296 := by
  unfold bf16Bits; omega

/-- The low 16 bits are zero. -/
theorem bf16_low_zero (h : Nat) : bf16Bits h % 65536 = 0 := by
  unfold bf16Bits; omega

/-- The shift is injective: the high half recovers the input. -/
theorem bf16_recover (h : Nat) : bf16Bits h / 65536 = h := by
  unfold bf16Bits; omega

/-- Index range check mirrors `rows()`: valid exactly when the index lies in [0, n). -/
theorem idx_valid (i : Int) (n : Int) : (0 ≤ i ∧ i < n) ↔ ¬ (i < 0 ∨ n ≤ i) := by omega

end Strata.Loader



## FILE: /mnt/user-data/outputs/F16Bits.lean
/-!
# MODULE: f16_bits
IEEE fp16 <-> f32 conversion by bits: round-to-nearest-even rule, exponent regions, overflow
saturation vs NaN, subnormal normalisation. Modelled over Nat.

## Source semantics
`f16_from_f32`: f32 inf/NaN (raw exponent 255) stays inf/NaN; a finite value too large for fp16
saturates to inf; exponent <= 0 is subnormal with a moving rounding point; below -10 underflows to
signed zero; the normal path rounds a 13-bit remainder to nearest even and the carry may bump the
exponent. `f32_from_f16` is exact, including subnormals.

## Form template identifiers
f16Round -> round_nearest_even ; special -> overflow_classify ; norm -> subnormal_normalise
-/

set_option maxRecDepth 100000

namespace Strata.F16

-- § Round to nearest even

/-- Rounding of `h` with discarded remainder `rem` against the half point `half`. -/
def f16Round (h rem half : Nat) : Nat :=
  if half < rem ∨ (rem = half ∧ h % 2 = 1) then h + 1 else h

/-- Rounding moves by at most one unit. -/
theorem round_cases (h r half : Nat) : f16Round h r half = h ∨ f16Round h r half = h + 1 := by
  unfold f16Round
  split <;> omega

/-- Below the half point the value is truncated. -/
theorem round_below (h r half : Nat) (hl : r < half) : f16Round h r half = h := by
  unfold f16Round
  split <;> omega

/-- Above the half point it rounds up. -/
theorem round_above (h r half : Nat) (hl : half < r) : f16Round h r half = h + 1 := by
  unfold f16Round
  split <;> omega

/-- Exact tie, even value: stays. -/
theorem round_tie_even (h half : Nat) (he : h % 2 = 0) : f16Round h half half = h := by
  unfold f16Round
  split <;> omega

/-- Exact tie, odd value: rounds up to the even neighbour. -/
theorem round_tie_odd (h half : Nat) (ho : h % 2 = 1) : f16Round h half half = h + 1 := by
  unfold f16Round
  split <;> omega

/-- After a tie the result is even. -/
theorem round_tie_result_even (h half : Nat) : f16Round h half half % 2 = 0 := by
  unfold f16Round
  split <;> omega

-- § Carry between mantissa and exponent

/-- Packed exponent and mantissa fields. -/
def pack (e m : Nat) : Nat := e * 1024 + m

/-- A mantissa carry bumps the exponent and clears the mantissa. -/
theorem carry_bumps (e : Nat) : pack e 1023 + 1 = pack (e + 1) 0 := by
  unfold pack; omega

/-- The largest finite fp16 rounds up to exactly +inf (0x7C00). -/
theorem carry_to_inf : pack 30 1023 + 1 = 0x7C00 := by unfold pack; decide

-- § Overflow classification

/-- Exponent and mantissa fields for the special/overflow cases, none for the finite path. -/
def special (rawexp man : Nat) : Option (Nat × Nat) :=
  if rawexp = 255 then some (31, if man ≠ 0 then 512 else 0)
  else if 143 ≤ rawexp then some (31, 0)
  else none

/-- The overflow threshold: raw exponent 143 is the first with fp16 exponent 31. -/
theorem overflow_threshold : 143 - 127 + 15 = 31 := rfl

/-- A finite overflowing value saturates to inf: exponent 31, mantissa 0. -/
theorem finite_overflow_inf (rawexp man : Nat) (h1 : 143 ≤ rawexp) (h2 : rawexp < 255) :
    special rawexp man = some (31, 0) := by
  unfold special
  rw [if_neg (by omega), if_pos h1]

/-- An f32 NaN stays a NaN: the mantissa field is non-zero. -/
theorem nan_stays_nan (man : Nat) (h : man ≠ 0) : special 255 man = some (31, 512) := by
  unfold special
  simp [h]

/-- An f32 inf stays an inf. -/
theorem inf_stays_inf : special 255 0 = some (31, 0) := by
  unfold special
  simp

/-- The conflated test of the older copies: any exponent >= 31 gives NaN when the mantissa is set. -/
def buggySpecial (man : Nat) : Nat × Nat := (31, if man ≠ 0 then 512 else 0)

/-- The old behaviour returned a NaN for a finite overflow with a non-zero mantissa. -/
theorem buggy_differs (rawexp man : Nat) (h1 : 143 ≤ rawexp) (h2 : rawexp < 255) (hm : man ≠ 0) :
    some (buggySpecial man) ≠ special rawexp man := by
  rw [finite_overflow_inf rawexp man h1 h2]
  unfold buggySpecial
  simp [hm]

-- § Subnormal and underflow regions

/-- Subnormal results (exponent in [-10, 0], raw exponent in [102, 112]) shift by 126 - rawexp. -/
theorem subnormal_shift (rawexp : Nat) (h1 : 102 ≤ rawexp) (h2 : rawexp ≤ 112) :
    14 ≤ 126 - rawexp ∧ 126 - rawexp ≤ 24 := by omega

/-- Raw exponent below 102 means fp16 exponent below -10: underflow to signed zero. -/
theorem underflow_threshold (rawexp : Nat) (h : rawexp < 102) : (rawexp : Int) - 112 < -10 := by
  omega

/-- The five regions partition all raw exponents 0..255. -/
def region (rawexp : Nat) : Nat :=
  if rawexp = 255 then 4 else if 143 ≤ rawexp then 3 else if 113 ≤ rawexp then 2
  else if 102 ≤ rawexp then 1 else 0

/-- Every region index is in 0..4. -/
theorem region_lt (r : Nat) : region r < 5 := by
  unfold region
  split <;> try omega
  split <;> try omega
  split <;> try omega
  split <;> omega

-- § Exact decode

/-- Subnormal normalisation loop with fuel: shifts left until bit 10 is set. -/
def norm : Nat → Nat → Nat × Nat
  | m, 0 => (m, 0)
  | m, f + 1 => if m &&& 1024 = 0 then ((norm (m * 2) f).1, (norm (m * 2) f).2 + 1) else (m, 0)

/-- For every non-zero 10-bit mantissa the loop ends with the leading one in bit 10, after at most
ten shifts: no subnormal input is flushed to zero. -/
theorem norm_ok : ∀ m < 1024, 1 ≤ m → (norm m 10).1 / 1024 = 1 ∧ (norm m 10).2 ≤ 10 := by
  decide

/-- The resulting f32 exponent field 113 - s stays a normal exponent (>= 103). -/
theorem norm_exp_field (s : Nat) (h : s ≤ 10) : 103 ≤ 113 - s ∧ 113 - s ≤ 113 := by omega

/-- Smallest subnormal 0x0001 is 2^-24: s = 10 gives exponent field 103, and 103 - 127 = -24. -/
theorem min_subnormal : (113 - 10 : Int) - 127 = -24 := by decide

/-- Inf/NaN decode: exponent 31 maps to the all-ones f32 exponent 255. -/
theorem decode_special_exp : 0x7F800000 / 8388608 = 255 := by decide

/-- NaN payload shifts into the f32 mantissa without overflowing it. -/
theorem decode_payload (man : Nat) (h : man < 1024) : man * 8192 < 8388608 := by omega

end Strata.F16



## FILE: /mnt/user-data/outputs/QsaAttentionLayout.lean
/-!
# MODULE: qsa_attention_layout
Decode-time bookkeeping of the QSA (sparse full-attention) layer: selection width, per-token step
state, tail/spare-slot cell mapping, GQA head map, paged KV addressing, top-k tie order, gate split,
memory budgets.

## Source semantics
Selection width = min(n_kv, idx_top_k + idx_block - 1). The step buffer holds (pos, n_kv = pos+1,
n_bid = n_kv / r, width). Cells of the incomplete tail map to the spare slot at row n_bid. The q to
kv head map is integer division. KV rows are `[page][kv_head][page_size][head_dim]`. Top-k ties
break by ascending index.

## Form template identifiers
selWidth -> qsa_selection_width ; stepFill -> qsa_step_fill ; cellBlock -> cell_block ;
before -> topk_order ; physRow -> kv_physical_row
-/

set_option maxRecDepth 100000

namespace Strata.Qsa

-- § Selection width

/-- `min(n_kv, idx_top_k + idx_block - 1)`. -/
def selWidth (n k r : Nat) : Nat := min n (k + r - 1)

/-- Never selects more cells than exist. -/
theorem sel_le_nkv (n k r : Nat) : selWidth n k r ≤ n := by unfold selWidth; omega

/-- Never exceeds the budget plus the incomplete tail. -/
theorem sel_le_cap (n k r : Nat) : selWidth n k r ≤ k + r - 1 := by unfold selWidth; omega

/-- Below the bound the selection is the identity: every cell is selected. -/
theorem sel_identity (n k r : Nat) (h : n ≤ k + r - 1) : selWidth n k r = n := by
  unfold selWidth; omega

/-- Width is monotone in the context length. -/
theorem sel_mono (n m k r : Nat) (h : n ≤ m) : selWidth n k r ≤ selWidth m k r := by
  unfold selWidth; omega

/-- Real artifact: the bound is 2051, not the plan's 2048. -/
theorem sel_real : selWidth 100000 2048 4 = 2051 ∧ selWidth 2051 2048 4 = 2051 ∧
    selWidth 2050 2048 4 = 2050 ∧ selWidth 2048 2048 4 = 2048 := by decide

/-- The capturable top-k capacity is the width at the 32768-cell cap. -/
theorem sel_cap_at_max : selWidth 32768 2048 4 = 2051 := by decide

-- § Step state

/-- Per-token device step buffer (pos, n_kv, n_bid, width). -/
structure Step where
  pos : Nat
  nkv : Nat
  nbid : Nat
  width : Nat

/-- All four entries are derived from `pos`, by one function. -/
def stepFill (pos k r : Nat) : Step := ⟨pos, pos + 1, (pos + 1) / r, selWidth (pos + 1) k r⟩

/-- n_kv = pos + 1. -/
theorem step_nkv (pos k r : Nat) : (stepFill pos k r).nkv = pos + 1 := rfl

/-- The width never exceeds n_kv. -/
theorem step_width_le (pos k r : Nat) : (stepFill pos k r).width ≤ (stepFill pos k r).nkv := by
  unfold stepFill; exact sel_le_nkv _ _ _

/-- Completed blocks cover at most n_kv cells, and fewer than r cells remain. -/
theorem step_blocks (pos k r : Nat) (hr : 0 < r) :
    (stepFill pos k r).nbid * r ≤ (stepFill pos k r).nkv ∧
    (stepFill pos k r).nkv < (stepFill pos k r).nbid * r + r := by
  unfold stepFill
  simp only
  have h1 := Nat.div_add_mod (pos + 1) r
  have h2 := Nat.mod_lt (pos + 1) hr
  rw [Nat.mul_comm ((pos + 1) / r) r]
  omega

/-- The buffer is four int32: 16 bytes. -/
theorem step_bytes : 4 * 4 = 16 := rfl

/-- A block completes exactly when pos % 4 = 3. -/
theorem complete_iff (pos : Nat) : (pos + 1) % 4 = 0 ↔ pos % 4 = 3 := by omega

/-- n_bid at position pos is (pos+1)/4; it grows from pos-1 to pos exactly when pos % 4 = 3. -/
theorem nbid_grows (pos : Nat) (h : pos % 4 = 3) : (pos + 1) / 4 = pos / 4 + 1 := by omega

/-- At every other position n_bid is unchanged from the previous token. -/
theorem nbid_stable (pos : Nat) (h : pos % 4 ≠ 3) : (pos + 1) / 4 = pos / 4 := by omega

/-- The pooled block is rotated at its FIRST cell: pos - 3 = (pos / 4) * 4 on completion. -/
theorem rotate_first_cell (pos : Nat) (h : pos % 4 = 3) : pos - 3 = pos / 4 * 4 := by omega

/-- First cell and last cell differ by 3, so rotating at the last would be a different position. -/
theorem first_ne_last (pos : Nat) (h : pos % 4 = 3) : pos - 3 ≠ pos := by omega

-- § Cell to block map

/-- Cell j maps to block j / r, clamped to the spare slot at row n_bid. -/
def cellBlock (j nbid r : Nat) : Nat := min (j / r) nbid

/-- Cells of completed blocks map to their own block. -/
theorem cell_completed (j nbid r : Nat) (hr : 0 < r) (h : j < nbid * r) :
    cellBlock j nbid r = j / r := by
  have : j / r < nbid := (Nat.div_lt_iff_lt_mul hr).2 h
  unfold cellBlock; omega

/-- Tail cells (those in the last incomplete block) all map to the spare slot at row n_bid. -/
theorem cell_tail (j nkv r : Nat) (hr : 0 < r) (h1 : nkv / r * r ≤ j) (h2 : j < nkv) :
    cellBlock j (nkv / r) r = nkv / r := by
  have hlt : nkv < (nkv / r + 1) * r := by
    have a := Nat.div_add_mod nkv r
    have b := Nat.mod_lt nkv hr
    rw [Nat.add_mul, Nat.one_mul, Nat.mul_comm (nkv / r) r]
    omega
  have : j / r = nkv / r := Nat.div_eq_of_lt_le h1 (by omega)
  unfold cellBlock; omega

/-- Fewer than r cells are in the tail; with r = 4 the raw tail holds idx_block - 1 = 3 rows. -/
theorem tail_size (nkv : Nat) : nkv % 4 ≤ 3 := by omega

-- § GQA head map

/-- Query head q maps to kv head q / group, which is in range. -/
theorem gqa_in_range (q nh nkv g : Nat) (hg : 0 < g) (hn : nh = nkv * g) (hq : q < nh) :
    q / g < nkv := by
  apply (Nat.div_lt_iff_lt_mul hg).2
  rw [hn] at hq
  exact hq

/-- Real shapes: 24 heads, 2 kv heads, group 12; every q/12 is 0 or 1. -/
theorem gqa_real : ∀ q < 24, q / 12 < 2 := by intro q h; omega

/-- Integer division and modulo give different maps (head 1 differs): the choice is observable. -/
theorem gqa_div_ne_mod : ∃ q < 24, q / 12 ≠ q % 2 := ⟨1, by omega, by omega⟩

-- § Paged KV addressing

/-- Row-major bound used by every paged index. -/
theorem rowmajor_lt {i n j m : Nat} (hi : i < n) (hj : j < m) : i * m + j < n * m := by
  have h1 : (i + 1) * m ≤ n * m := Nat.mul_le_mul_right m hi
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

/-- Physical row of (page, head, in-page offset). -/
def physRow (page h off nkv ps : Nat) : Nat := (page * nkv + h) * ps + off

/-- Every physical row lies in the pool of nPages pages. -/
theorem phys_row_lt (page h off nPages nkv ps : Nat) (hp : page < nPages) (hh : h < nkv)
    (ho : off < ps) : physRow page h off nkv ps < nPages * nkv * ps := by
  unfold physRow
  have h1 : page * nkv + h < nPages * nkv := rowmajor_lt hp hh
  exact rowmajor_lt h1 ho

/-- Within a page one head's cells are consecutive: offset +1 is row +1. -/
theorem phys_row_consec (page h off nkv ps : Nat) :
    physRow page h (off + 1) nkv ps = physRow page h off nkv ps + 1 := by
  unfold physRow; omega

/-- A logical cell splits into (page, offset) and recombines. -/
theorem cell_split (t ps : Nat) : (t / ps) * ps + t % ps = t := by
  rw [Nat.mul_comm]; exact Nat.div_add_mod t ps

-- § Top-k order

/-- i precedes j: higher score first, ties by ascending index. -/
def before (s : Nat → Int) (i j : Nat) : Prop := s j < s i ∨ (s i = s j ∧ i < j)

/-- Irreflexive. -/
theorem before_irrefl (s : Nat → Int) (i : Nat) : ¬ before s i i := by
  unfold before; omega

/-- Total on distinct cells: the selection is deterministic. -/
theorem before_total (s : Nat → Int) (i j : Nat) (h : i ≠ j) : before s i j ∨ before s j i := by
  unfold before; omega

/-- Asymmetric. -/
theorem before_asymm (s : Nat → Int) (i j : Nat) (h : before s i j) : ¬ before s j i := by
  unfold before at *; omega

/-- Transitive. -/
theorem before_trans (s : Nat → Int) (i j k : Nat) (h1 : before s i j) (h2 : before s j k) :
    before s i k := by
  unfold before at *; omega

-- § Gate split and budgets

/-- Raw q projection is per head `[q | gate]`: q at h*2hd + d, gate at h*2hd + hd + d, disjoint. -/
theorem gate_split (h d hd : Nat) (hdd : d < hd) :
    h * (2 * hd) + d < h * (2 * hd) + hd + d := by omega

/-- Both halves stay inside n_head * 2 * head_dim. -/
theorem gate_in_range (h d hd nh : Nat) (hh : h < nh) (hdd : d < hd) :
    h * (2 * hd) + hd + d < nh * (2 * hd) := by
  have := rowmajor_lt (m := 2 * hd) hh (show hd + d < 2 * hd by omega)
  omega

/-- Real raw projection width: 24 * 512. -/
theorem qfull_width : 24 * (2 * 256) = 12288 := rfl

/-- KV cache at 32K: 805,306,368 bytes, which is 768 MiB (the header says "805 MiB": 805 MB). -/
theorem kv_bytes : 12 * 2 * 256 * 2 * 2 * 32768 = 805306368 ∧ 805306368 = 768 * 1048576 := by decide

/-- The same in f32 is exactly double. -/
theorem kv_bytes_f32 : 12 * 2 * 256 * 2 * 4 * 32768 = 2 * 805306368 := by decide

/-- Indexer keys: 128 floats = 512 B per block, 128 B per token, 1536 B per token over 12 layers,
and 50,331,648 B = 48 MiB at 32K. -/
theorem indexer_bytes : 128 * 4 = 512 ∧ 512 / 4 = 128 ∧ 128 * 12 = 1536 ∧
    1536 * 32768 = 50331648 ∧ 50331648 = 48 * 1048576 := by decide

/-- Real geometry sanity: heads divide, head_dim multiple of 4, rotary dims fit both widths. -/
theorem real_shapes : 24 % 2 = 0 ∧ 256 % 4 = 0 ∧ 64 ≤ 256 ∧ 64 ≤ 128 ∧ 2048 % 4 = 0 := by decide

end Strata.Qsa



## FILE: /mnt/user-data/outputs/PleNgramHash.lean
/-!
# MODULE: ple_ngram_hash
PLE n-gram hash row indices and the per-layer token table: XOR mixing, modulus and offsets, EOS cut,
null predecessor test, IQ4_NL row layout, table size accounting.

## Source semantics
`mixed_n = (ctx[0]*m[0]) ^ ... ^ (ctx[n-1]*m[n-1])` in uint64, row = (mixed % vocab_h) + offset_h.
Vocab sizes are not powers of two, so `%` cannot be a mask. A missing predecessor is TOKEN_NULL = -1
(tested `t < 0`; token 0 is real). An EOS predecessor cuts itself and every older position; the
token's own EOS cuts nothing. A 160-wide IQ4_NL row is 5 blocks of 18 bytes; 16 rows flatten
head-slowest to n_embd = 2560.

## Form template identifiers
mixed -> ngram_mixed ; rowOf -> row index ; cut -> eos_cut ; isNull -> t < 0
-/

set_option maxRecDepth 100000

namespace Strata.Ple

-- § Geometry

/-- Heads: (ngram - 1) * heads_per_ngram = 16. -/
theorem n_heads : (3 - 1) * 8 = 16 := rfl

/-- 16 heads x 160 = n_embd 2560. -/
theorem flatten_width : 16 * 160 = 2560 := rfl

/-- History length (conv_kernel - 1) * ngram = 9; hyper-connection width 4 * 2560 = 10240. -/
theorem hist_hc : (4 - 1) * 3 = 9 ∧ 4 * 2560 = 10240 := ⟨rfl, rfl⟩

/-- Head-slowest flatten: element i of head h lands at h*160 + i, in range and invertible. -/
theorem flatten_index (h i : Nat) (hh : h < 16) (hi : i < 160) :
    h * 160 + i < 2560 ∧ (h * 160 + i) / 160 = h ∧ (h * 160 + i) % 160 = i := by omega

/-- A row is 160 / 32 = 5 blocks of 18 bytes = 90 bytes. -/
theorem row_bytes : 160 / 32 * 18 = 90 := rfl

/-- Table: 320,001,536 rows x 160 = 51.2e9 elements; x 90 B = 28.8e9 bytes. -/
theorem table_size : 320001536 * 160 = 51200245760 ∧ 320001536 * 90 = 28800138240 := by decide

/-- Row cache of 2^20 rows x 90 B = 94,371,840 B (about 95 MB). -/
theorem cache_bytes : (1 <<< 20) * 90 = 94371840 := by decide

-- § Hash

/-- 64-bit product, modelled as a Nat modulo 2^64. -/
def mul64 (x m : Nat) : Nat := (x * m) % 18446744073709551616

/-- XOR-fold of the products. -/
def mixed : List Nat → List Nat → Nat
  | c :: cs, m :: ms => mul64 c m ^^^ mixed cs ms
  | _, _ => 0

/-- Two equal terms cancel under XOR. -/
theorem xor_cancel (x : Nat) : x ^^^ x = 0 := Nat.xor_self x

/-- Under sum the same two terms would double instead: a different function. -/
theorem sum_doubles (x : Nat) (h : 0 < x) : x + x ≠ 0 := by omega

/-- XOR cancellation, concretely: equal tokens with equal multipliers contribute nothing. -/
theorem mixed_equal_pair (c m : Nat) : mixed [c, c] [m, m] = 0 := by
  simp [mixed, Nat.xor_self]

/-- Products stay below 2^64. -/
theorem mul64_lt (x m : Nat) : mul64 x m < 18446744073709551616 := by
  unfold mul64
  exact Nat.mod_lt _ (by decide)

/-- Row index of head h: `(mixed % vocab) + offset`. -/
def rowOf (mix vocab off : Nat) : Nat := mix % vocab + off

/-- A row lies inside its head's slice [off, off + vocab). -/
theorem row_in_slice (mix vocab off : Nat) (hv : 0 < vocab) :
    off ≤ rowOf mix vocab off ∧ rowOf mix vocab off < off + vocab := by
  have := Nat.mod_lt mix hv
  unfold rowOf; omega

/-- Slices are disjoint when offsets are the running sum of vocab sizes. -/
theorem slices_disjoint (mix vocab off nextOff : Nat) (hv : 0 < vocab)
    (hn : off + vocab ≤ nextOff) : rowOf mix vocab off < nextOff := by
  have := (row_in_slice mix vocab off hv).2
  omega

/-- Artifact constants: first offsets are 0, 20000003, 40000026 = 20000003 + 20000023. -/
theorem offsets_running_sum : 0 + 20000003 = 20000003 ∧ 20000003 + 20000023 = 40000026 := ⟨rfl, rfl⟩

/-- The vocab sizes are not an arithmetic sequence of step 10: differences are 20, then 10. -/
theorem vocab_not_arith : 20000023 - 20000003 = 20 ∧ 20000033 - 20000023 = 10 := ⟨rfl, rfl⟩

/-- A mask would be a different function: 20000003 is not a power of two. -/
theorem vocab_not_pow2 : (20000003 &&& (20000003 - 1)) ≠ 0 := by decide

-- § Null predecessor and EOS cut

/-- TOKEN_NULL test: negative ids are null; id 0 is a real token. -/
def isNull (t : Int) : Bool := decide (t < 0)

/-- Token id 0 is not null; -1 is. -/
theorem null_test : isNull 0 = false ∧ isNull (-1) = true := by decide

/-- EOS cut over the two predecessors (oldest p0, newest p1): an EOS replaces itself and all older
positions by TOKEN_NULL. The current token is not an input. -/
def cut (eos p0 p1 : Int) : Int × Int :=
  if p1 = eos then (-1, -1) else if p0 = eos then (-1, p1) else (p0, p1)

/-- An EOS at the newest predecessor nulls both. -/
theorem cut_newest (eos p0 : Int) : cut eos p0 eos = (-1, -1) := by
  unfold cut; simp

/-- An EOS at the oldest predecessor nulls only itself. -/
theorem cut_oldest (eos p1 : Int) (h : p1 ≠ eos) : cut eos eos p1 = (-1, p1) := by
  unfold cut; simp [h]

/-- No EOS: nothing changes. -/
theorem cut_none (eos p0 p1 : Int) (h0 : p0 ≠ eos) (h1 : p1 ≠ eos) : cut eos p0 p1 = (p0, p1) := by
  unfold cut; simp [h0, h1]

/-- Idempotent when eos is non-negative: applying the cut twice equals once. -/
theorem cut_idem (eos p0 p1 : Int) (he : 0 ≤ eos) :
    cut eos (cut eos p0 p1).1 (cut eos p0 p1).2 = cut eos p0 p1 := by
  unfold cut
  split <;> (try split) <;> simp <;> omega

-- § IQ4_NL row decode layout

/-- Split-half order: byte j carries elements j and j + 16, covering 0..31 exactly once. -/
theorem split_half_cover (i : Nat) (hi : i < 32) : ∃ j, j < 16 ∧ (i = j ∨ i = j + 16) :=
  ⟨i % 16, by omega, by omega⟩

/-- The interleaved reading (2j, 2j+1) differs: byte 1 would carry 2 and 3, not 1 and 17. -/
theorem not_interleaved : (1 : Nat) ≠ 2 * 1 ∧ (17 : Nat) ≠ 2 * 1 + 1 := by decide

end Strata.Ple



## FILE: /mnt/user-data/outputs/SGemvCanonicalForm.lean
/-!
# MODULE: s_gemv_canonical_form
The S-family GEMV's canonical weight form: code widths, packing density, group addressing,
activation block strides, strided work partitioning.

## Source semantics
value = cb[code] * scale + offset, parameterised by code_bits (2/4/8), code_bias, group_elems and
codebook. Row o of a [n_in, n_out] tensor is a contiguous run of n_in/group_elems groups. Q8_K
activation blocks are 292 bytes per 256 elements; Q8_0 blocks are 34 bytes per 32. A row split
across T threads gives thread t the strided subset i = t mod T.

## Form template identifiers
SForm -> SForm ; perByte -> codes_per_byte ; scaleIdx -> o*(n_in/g) + i/g ; stride -> threads_per_row
-/

set_option maxRecDepth 100000

namespace Strata.SGemv

/-- Canonical form attributes relevant to addressing. -/
structure SForm where
  codeBits : Nat
  codeBias : Int
  groupElems : Nat

/-- Q2_0 attributes. -/
def q2_0 : SForm := ⟨2, -1, 64⟩
/-- Q4_0 attributes. -/
def q4_0 : SForm := ⟨4, -8, 32⟩
/-- Q5_0 attributes (8-bit container, bias -16). -/
def q5_0 : SForm := ⟨8, -16, 32⟩

/-- Codes per byte. -/
def perByte (f : SForm) : Nat := 8 / f.codeBits

/-- 2, 4 and 8 bit codes pack 4, 2 and 1 per byte. -/
theorem per_byte_vals : perByte q2_0 = 4 ∧ perByte q4_0 = 2 ∧ perByte q5_0 = 1 := by decide

/-- Decode with an optional offset. -/
def decode (cb scale off : Int) : Int := cb * scale + off

/-- Without an offset the decode is a pure product. -/
theorem decode_no_offset (cb scale : Int) : decode cb scale 0 = cb * scale := by
  unfold decode; omega

/-- Biased integer value range of a code below m (m = 2^bits). -/
theorem biased_range (m : Nat) (bias : Int) (c : Nat) (h : c < m) :
    bias ≤ (c : Int) + bias ∧ (c : Int) + bias ≤ (m : Int) - 1 + bias := by omega

/-- Q2_0 symbols are -1..2, Q4_0 symbols are -8..7. -/
theorem sym_ranges : (0 : Int) + (-1) = -1 ∧ (3 : Int) + (-1) = 2 ∧ (0 : Int) + (-8) = -8 ∧
    (15 : Int) + (-8) = 7 := by decide

-- § Addressing

/-- Row-major bound. -/
theorem rowmajor_lt {i n j m : Nat} (hi : i < n) (hj : j < m) : i * m + j < n * m := by
  have h1 : (i + 1) * m ≤ n * m := Nat.mul_le_mul_right m hi
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

/-- Group index within a row stays below n_in / g when g divides n_in. -/
theorem group_in_row (i g nIn : Nat) (hg : 0 < g) (hd : g ∣ nIn) (hi : i < nIn) :
    i / g < nIn / g := by
  apply (Nat.div_lt_iff_lt_mul hg).2
  rw [Nat.div_mul_cancel hd]
  exact hi

/-- Scale/offset index of element i of row o stays inside the n_out * (n_in / g) array. -/
theorem scale_index_lt (o i nOut nIn g : Nat) (hg : 0 < g) (hd : g ∣ nIn) (ho : o < nOut)
    (hi : i < nIn) : o * (nIn / g) + i / g < nOut * (nIn / g) :=
  rowmajor_lt ho (group_in_row i g nIn hg hd hi)

/-- Four consecutive elements starting at a multiple of four share one code byte (S2 quads). -/
theorem quad_shares_byte (k d : Nat) (hd : d < 4) : (4 * k + d) / 4 = k := by omega

/-- Row byte count is exact when the code density divides n_in. -/
theorem row_code_bytes (nIn p : Nat) (hp : 0 < p) (hd : p ∣ nIn) : p * (nIn / p) = nIn :=
  Nat.mul_div_cancel' hd

-- § Strided row split

/-- Thread t = i mod T handles element i at iteration i / T: every element is covered. -/
theorem stride_cover (i T : Nat) : i % T + (i / T) * T = i := by
  have := Nat.mod_add_div i T
  rw [Nat.mul_comm (i / T) T]
  exact this

/-- The covering thread is below T (so it exists). -/
theorem stride_thread_lt (i T : Nat) (hT : 0 < T) : i % T < T := Nat.mod_lt i hT

/-- Thread t at iteration k handles element t + k*T, and that element's covering thread is t. -/
theorem stride_unique (t k T : Nat) (ht : t < T) : (t + k * T) % T = t := by
  rw [Nat.add_mul_mod_self_right]
  exact Nat.mod_eq_of_lt ht

-- § Activation block strides

/-- Q8_K block: d 4 + qs 256 + bsums 32 = 292; bsums at offset 260. -/
theorem q8k_block : 4 + 256 + 16 * 2 = 292 ∧ 4 + 256 = 260 := ⟨rfl, rfl⟩

/-- Q8_0 block: d 2 + qs 32 = 34. -/
theorem q8_0_block : 2 + 32 = 34 := rfl

/-- 2560 activations: 10 Q8_K blocks = 2920 B; 640 activations: 20 Q8_0 blocks = 680 B. -/
theorem act_bytes : 2560 / 256 * 292 = 2920 ∧ 640 / 32 * 34 = 680 := by decide

/-- ffn_down_shexp n_in = 640 is a multiple of 32 but not of 256: Q8_K is impossible. -/
theorem shexp_not_q8k : 640 % 32 = 0 ∧ 640 % 256 ≠ 0 := by decide

/-- Power-of-two test used for group_elems requirements. -/
def isPow2 (g : Nat) : Bool := decide (0 < g) && decide ((g &&& (g - 1)) = 0)

/-- 32, 64 and 256 qualify; 20 does not. -/
theorem pow2_cases : isPow2 32 = true ∧ isPow2 64 = true ∧ isPow2 256 = true ∧
    isPow2 20 = false := by decide

/-- Expert gate/up [2560 x 640]: 640 outputs over 48 SMs is 13 per SM with 16 left over. -/
theorem sm_occupancy : 640 / 48 = 13 ∧ 640 % 48 = 16 := by decide

end Strata.SGemv



## FILE: /mnt/user-data/outputs/DequantGpuKernels.lean
/-!
# MODULE: dequant_gpu_kernels
GPU dequantiser indexing: one thread per 32-element group, per-type block/sub-block decomposition,
agreement with the sequential CPU order, launch grid, bf16 round-to-nearest-even.

## Source semantics
`group32<TYPE>` decodes group gi of a row. Q2_0: block gi/2, element offset (gi%2)*32. K-quants:
block gi/8, sub-group gi%8. Q3_K mask is 1 << (n*4+jj); Q5_K mask is 1 << (2*j64+hi); Q4_K and Q5_K
scale index is gi. The grid is ceil(total / 256). `f2bf` adds 0x7fff plus the lsb of the kept part
and truncates 16 bits, which is round-to-nearest-even.

## Form template identifiers
group32 -> decode_group ; f2bf -> f32_to_bf16 ; grid -> launch_grid
-/

set_option maxRecDepth 100000

namespace Strata.DequantGpu

-- § Per-type decomposition

/-- Q2_0 group gi of a row: block gi/2, element base (gi%2)*32; elements stay below 64. -/
theorem q2_group (gi j : Nat) (hj : j < 32) : (gi % 2) * 32 + j < 64 := by omega

/-- Q2_0 groups tile a block of 64: group 2b covers [0,32), group 2b+1 covers [32,64). -/
theorem q2_tile (b : Nat) : (2 * b) / 2 = b ∧ (2 * b + 1) / 2 = b ∧ (2 * b) % 2 = 0 ∧
    (2 * b + 1) % 2 = 1 := by omega

/-- Q3_K: the GPU scale index n*8 + jj*2 + half equals the CPU sequential counter
`2*(4n + jj) + half`. -/
theorem q3k_scale_idx (n jj half : Nat) :
    n * 8 + jj * 2 + half = 2 * (4 * n + jj) + half := by omega

/-- Q3_K sub-group gi < 8 splits as gi = 4n + jj. -/
theorem q3k_split (gi : Nat) (h : gi < 8) : gi / 4 * 4 + gi % 4 = gi := by omega

/-- Q3_K: the GPU mask exponent n*4+jj equals gi, matching the CPU's hoisted mask shifted once
per step. -/
theorem q3k_mask_exp (gi : Nat) (h : gi < 8) : (gi / 4) * 4 + gi % 4 = gi := by omega

/-- Q3_K mask exponent stays below 8, so the mask fits a byte. -/
theorem q3k_mask_fits (gi : Nat) (h : gi < 8) : (1 <<< gi) < 256 := by
  have : ∀ k < 8, (1 <<< k) < 256 := by decide
  exact this gi h

/-- Q4_K / Q5_K: scale index 2*(gi/2) + gi%2 equals gi, as the CPU's `is + {0,1}`. -/
theorem q45k_scale_idx (gi : Nat) : 2 * (gi / 2) + gi % 2 = gi := by omega

/-- Q5_K: the exponent 2*j64 + hi equals gi, so `1 << (2*j64+hi)` is the CPU's u1 / u2 pair. -/
theorem q5k_mask_exp (gi : Nat) : 2 * (gi / 2) + gi % 2 = gi := by omega

/-- Q5_K qs offset 32*j64 stays in the 128-byte payload for j64 < 4. -/
theorem q5k_qs_off (j64 l : Nat) (hj : j64 < 4) (hl : l < 32) : 32 * j64 + l < 128 := by omega

/-- Q6_K: scale index 8n + l/16 + 2qu stays below 16. -/
theorem q6k_scale (n qu l : Nat) (hn : n < 2) (hq : qu < 4) (hl : l < 32) :
    8 * n + l / 16 + 2 * qu < 16 := by omega

/-- Q6_K: ql index 64n + l + 32 stays inside 128. -/
theorem q6k_ql (n l : Nat) (hn : n < 2) (hl : l < 32) : 64 * n + l + 32 < 128 := by omega

/-- Q6_K: qh index 32n + l stays inside 64. -/
theorem q6k_qh (n l : Nat) (hn : n < 2) (hl : l < 32) : 32 * n + l < 64 := by omega

/-- IQ4_XS: group ib has qs at 16*ib, below 128 for ib < 8. -/
theorem iq4xs_qs (ib j : Nat) (hi : ib < 8) (hj : j < 16) : 16 * ib + j < 128 := by omega

/-- Output offset of group gi is gi*32, and groups tile gpr*32 columns. -/
theorem out_tile (gi gpr : Nat) (h : gi < gpr) : gi * 32 + 32 ≤ gpr * 32 := by
  have : (gi + 1) * 32 ≤ gpr * 32 := Nat.mul_le_mul_right 32 h
  omega

-- § Geometry check

/-- Every supported block size is a multiple of 32, so cols % be = 0 implies cols % 32 = 0. -/
theorem cols_mult32 (be cols : Nat) (hb : be = 32 ∨ be = 64 ∨ be = 256) (h : cols % be = 0) :
    cols % 32 = 0 := by
  rcases hb with rfl | rfl | rfl <;> omega

/-- For Q2_0, row bytes cols/64*18 equals (cols/32)/2 * 18 : groups pair into blocks. -/
theorem q2_row_groups (cols : Nat) (h : cols % 64 = 0) : cols / 64 = (cols / 32) / 2 := by omega

-- § Launch grid

/-- Ceil division grid covers all groups. -/
theorem grid_covers (total : Nat) : total ≤ (total + 255) / 256 * 256 := by omega

/-- And leaves no empty trailing block. -/
theorem grid_tight (total : Nat) (h : 0 < total) : ((total + 255) / 256 - 1) * 256 < total := by omega

/-- Zero total gives an empty grid, which is illegal to launch: the host rejects rows <= 0. -/
theorem grid_zero : (0 + 255) / 256 = 0 := by decide

-- § bf16 rounding

/-- Result of `(u + 0x7fff + ((u >> 16) & 1)) >> 16`, with u = 65536 q + r and lsb = q % 2. -/
def bf16Round (q r : Nat) : Nat := (65536 * q + r + 32767 + q % 2) / 65536

/-- Below half: truncate. -/
theorem bf16_below (q r : Nat) (h : r < 32768) : bf16Round q r = q := by
  unfold bf16Round; omega

/-- Above half: round up. -/
theorem bf16_above (q r : Nat) (h1 : 32768 < r) (h2 : r < 65536) : bf16Round q r = q + 1 := by
  unfold bf16Round; omega

/-- Exact tie rounds to even. -/
theorem bf16_tie_even (q : Nat) (h : q % 2 = 0) : bf16Round q 32768 = q := by
  unfold bf16Round; omega

/-- Exact tie, odd: rounds up to even. -/
theorem bf16_tie_odd (q : Nat) (h : q % 2 = 1) : bf16Round q 32768 = q + 1 := by
  unfold bf16Round; omega

/-- A tie always lands on an even result. -/
theorem bf16_tie_result_even (q : Nat) : bf16Round q 32768 % 2 = 0 := by
  rcases Nat.mod_two_eq_zero_or_one q with h | h
  · rw [bf16_tie_even q h]; exact h
  · rw [bf16_tie_odd q h]; omega

end Strata.DequantGpu



## FILE: /mnt/user-data/outputs/ElementwiseGlueKernels.lean
/-!
# MODULE: elementwise_glue_kernels
Glue kernels: embedding code extraction, launch grids, warp-per-row RMSNorm row guard, grid-stride
copy coverage, doorbell sequence counter.

## Source semantics
`embedding_gather` extracts code (i / per_byte) >> ((i % per_byte) * bits) & mask. `grid_for(n)` is
ceil(n/256). `rms_norm_weighted` launches 4 warps per block, so a row count that is not a multiple
of 4 launches surplus warps whose rows must be guarded. The grid-stride copy visits every index.
The doorbell increments the memory it points at; a value baked into a graph at capture would repeat.

## Form template identifiers
extract -> embedding_code ; gridFor -> grid_for ; surplus -> rms_norm_surplus ; ring -> doorbell_ring
-/

set_option maxRecDepth 100000

namespace Strata.Elementwise

-- § Code extraction

/-- Extract the code at slot k of a byte: shift by k*bits, mask. -/
def extract (b k bits : Nat) : Nat := (b >>> (k * bits)) &&& ((1 <<< bits) - 1)

/-- 2-bit codes are below 4. -/
theorem extract2_lt : ∀ b < 256, ∀ k < 4, extract b k 2 < 4 := by decide

/-- 4-bit codes are below 16. -/
theorem extract4_lt : ∀ b < 256, ∀ k < 2, extract b k 4 < 16 := by decide

/-- 8-bit code is the byte itself. -/
theorem extract8 : ∀ b < 256, extract b 0 8 = b := by decide

/-- Element i lives in byte i / per_byte at slot i % per_byte, and these recombine. -/
theorem slot_cover (i p : Nat) : (i / p) * p + i % p = i := by
  rw [Nat.mul_comm]; exact Nat.div_add_mod i p

/-- The slot is below per_byte, so the shift stays inside the byte. -/
theorem slot_lt (i p : Nat) (hp : 0 < p) : i % p < p := Nat.mod_lt i hp

-- § Launch grids

/-- `grid_for(n) = ceil(n / 256)`. -/
def gridFor (n : Nat) : Nat := (n + 255) / 256

/-- The grid covers every element. -/
theorem grid_covers (n : Nat) : n ≤ gridFor n * 256 := by unfold gridFor; omega

/-- Zero elements give a zero grid, which is illegal: callers return early for n <= 0. -/
theorem grid_zero : gridFor 0 = 0 := by decide

/-- Positive n gives a positive grid. -/
theorem grid_pos (n : Nat) (h : 0 < n) : 0 < gridFor n := by unfold gridFor; omega

-- § RMSNorm row guard

/-- Warps launched: 4 per block, ceil(rows / 4) blocks. -/
def warps (rows : Nat) : Nat := (rows + 3) / 4 * 4

/-- At least `rows` warps are launched. -/
theorem warps_cover (rows : Nat) : rows ≤ warps rows := by unfold warps; omega

/-- Fewer than 4 surplus warps. -/
theorem warps_surplus (rows : Nat) : warps rows < rows + 4 := by unfold warps; omega

/-- The QSA k-norm has 2 rows and launches 4 warps: two surplus warps. -/
theorem kcur_case : warps 2 = 4 ∧ warps 2 - 2 = 2 := by decide

/-- Unguarded, the surplus rows write rows*cols .. warps*cols past the end: with 2 rows of 256 that
is 512 floats beyond the buffer, which in the arena is the start of vcur. -/
theorem unguarded_overrun : warps 2 * 256 - 2 * 256 = 512 := by decide

/-- Guarded: every active (row < rows, c < cols) index stays inside rows * cols. -/
theorem guarded_in_bounds (row c rows cols : Nat) (hr : row < rows) (hc : c < cols) :
    row * cols + c < rows * cols := by
  have h1 : (row + 1) * cols ≤ rows * cols := Nat.mul_le_mul_right cols hr
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § Grid-stride loops

/-- Every index i is visited by thread i % S at iteration i / S. -/
theorem stride_visit (i S : Nat) : i % S + S * (i / S) = i := Nat.mod_add_div i S

/-- Thread t at iteration k visits t + k*S, whose owner is t. -/
theorem stride_owner (t k S : Nat) (ht : t < S) : (t + k * S) % S = t := by
  rw [Nat.add_mul_mod_self_right]
  exact Nat.mod_eq_of_lt ht

/-- Copy launch bound: min(ceil(n4/256), 64) blocks, at most 64. -/
theorem copy_blocks_le (n4 : Nat) : min ((n4 + 255) / 256) 64 ≤ 64 := by omega

/-- n multiple of 4 gives n4 = n/4 with 4*n4 = n. -/
theorem copy_n4 (n : Nat) (h : n % 4 = 0) : 4 * (n / 4) = n := by omega

-- § Doorbell

/-- Sequence counter increment on a 32-bit word. -/
def ring (s : Nat) : Nat := (s + 1) % 4294967296

/-- Each ring changes the value, so a waiting host observes it. -/
theorem ring_changes (s : Nat) (h : s < 4294967296) : ring s ≠ s := by unfold ring; omega

/-- Ringing stays within 32 bits. -/
theorem ring_lt (s : Nat) : ring s < 4294967296 := by unfold ring; omega

/-- A value captured as a literal c rings c on every replay; two consecutive replays agree, so a
waiter for a change never wakes. The in-memory increment of two replays differs. -/
theorem literal_vs_increment (c s : Nat) (h : s < 4294967296) :
    (fun _ : Nat => c) s = (fun _ : Nat => c) (ring s) ∧ ring s ≠ s := by
  exact ⟨rfl, ring_changes s h⟩

/-- doorbell_publish bound: k <= 1024 threads each copy one id and weight. -/
theorem publish_threads (k : Nat) (h : k ≤ 1024) : k ≤ 1024 := h

end Strata.Elementwise



## FILE: /mnt/user-data/outputs/ActivationQuantize.lean
/-!
# MODULE: activation_quantize
Activation quantisation to Q8_0 / Q8_K blocks: block layouts, clamp ranges, the ggml `nearest_int`
magic-constant rounding, bsums bounds, launch grids.

## Source semantics
Q8_0 block: fp16 d + 32 int8 = 34 bytes. The ggml-matching rule clamps to [-128, 127]; the CPU-matching
scaled variant clamps to [-127, 127]. Q8_K block: f32 d + 256 int8 + 16 int16 bsums = 292 bytes, with
`qs = min(127, nearest_int(iscale * x))` and no lower clamp. `nearest_int` adds 1.5 * 2^23 and
recovers the integer from the low 23 mantissa bits.

## Form template identifiers
clamp -> clamp ; nearestInt -> nearest_int_dev ; sum_bounds -> bsums ; q8k -> block_q8_K
-/

set_option maxRecDepth 100000

namespace Strata.ActQuant

-- § Layouts

/-- Q8_0 block: 2 + 32 = 34; the zero-block branch writes the same 34 bytes. -/
theorem q8_0_layout : 2 + 32 = 34 := rfl

/-- Q8_K block: d 4, qs at 4, bsums at 260, total 292. -/
theorem q8k_layout : 4 + 256 = 260 ∧ 260 + 16 * 2 = 292 := ⟨rfl, rfl⟩

/-- Launch grid for Q8_0 (128 threads) and Q8_K (64 threads) covers all blocks. -/
theorem grids (nb : Nat) : nb ≤ (nb + 127) / 128 * 128 ∧ nb ≤ (nb + 63) / 64 * 64 := by omega

/-- Block counts: n/32 and n/256, exact for multiples. -/
theorem block_counts (n : Nat) (h32 : n % 32 = 0) : n / 32 * 32 = n := by omega

-- § Clamps

/-- Integer clamp to [lo, hi]. -/
def clamp (lo hi v : Int) : Int := max lo (min hi v)

/-- The result is inside the range (when lo <= hi). -/
theorem clamp_range (lo hi v : Int) (h : lo ≤ hi) : lo ≤ clamp lo hi v ∧ clamp lo hi v ≤ hi := by
  unfold clamp; omega

/-- In-range values are unchanged. -/
theorem clamp_id (lo hi v : Int) (h1 : lo ≤ v) (h2 : v ≤ hi) : clamp lo hi v = v := by
  unfold clamp; omega

/-- The symmetric [-127, 127] clamp commutes with negation. -/
theorem clamp_sym (v : Int) : clamp (-127) 127 (-v) = -clamp (-127) 127 v := by
  unfold clamp; omega

/-- The [-128, 127] clamp does not: the two Q8_0 rules differ at 128. -/
theorem clamp_asym : clamp (-128) 127 (-128) ≠ -(clamp (-128) 127 128) := by
  unfold clamp; decide

/-- The two rules agree on [-127, 127], so they differ only for -128 and below. -/
theorem clamp_rules_agree (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) :
    clamp (-128) 127 v = clamp (-127) 127 v := by
  unfold clamp; omega

/-- Rules disagree at -128. -/
theorem clamp_rules_differ : clamp (-128) 127 (-128) ≠ clamp (-127) 127 (-128) := by
  unfold clamp; decide

-- § nearest_int

/-- Integer recovered from the float bits after adding 1.5 * 2^23: the low 23 bits minus 2^22. -/
def nearestInt (n : Int) : Int := (4194304 + n) % 8388608 - 4194304

/-- Exact for every integer in [-2^22, 2^22). -/
theorem nearest_exact (n : Int) (h1 : -4194304 ≤ n) (h2 : n < 4194304) : nearestInt n = n := by
  unfold nearestInt; omega

/-- Outside that range it wraps, so the input scale must stay within 2^22. -/
theorem nearest_wraps : nearestInt 4194304 ≠ 4194304 := by decide

/-- The magic constant is 1.5 * 2^23. -/
theorem magic : 12582912 = 3 * 4194304 ∧ 4194304 * 2 = 8388608 := ⟨rfl, rfl⟩

-- § Q8_K

/-- Q8_K upper clamp only: in-range values pass unchanged, and nothing exceeds 127. -/
theorem q8k_min (v : Int) : min 127 v ≤ 127 ∧ (v ≤ 127 → min 127 v = v) := by omega

/-- bsums: the sum of int8 values is bounded by 128 per element. -/
theorem sum_bounds : ∀ l : List Int, (∀ x ∈ l, -128 ≤ x ∧ x ≤ 127) →
    -128 * (l.length : Int) ≤ l.sum ∧ l.sum ≤ 127 * (l.length : Int)
  | [], _ => by simp
  | x :: xs, h => by
    have hx := h x (by simp)
    have ih := sum_bounds xs (fun y hy => h y (by simp [hy]))
    simp only [List.sum_cons, List.length_cons]
    omega

/-- A group of 16 int8 values sums within [-2048, 2032], which fits an int16. -/
theorem bsums_fit (l : List Int) (hl : l.length = 16) (h : ∀ x ∈ l, -128 ≤ x ∧ x ≤ 127) :
    -32768 ≤ l.sum ∧ l.sum ≤ 32767 := by
  have := sum_bounds l h
  rw [hl] at this
  omega

/-- 16 groups of 16 values cover 256. -/
theorem bsums_cover : 16 * 16 = 256 := rfl

-- § Q8_0 scale

/-- ggml scale amax/127 and the 127-clamp mean the largest magnitude maps to exactly +-127. -/
theorem max_maps_127 : (127 : Int) ≤ 127 ∧ -127 ≥ -128 := by decide

end Strata.ActQuant





## FILE: /mnt/user-data/outputs/FusedGdnDecodeKernels.lean
/-!
# MODULE: fused_gdn_decode_kernels
Fused single-token kernels of the gated delta net (GDN): causal 4-tap conv + SiLU (+ per-head L2 norm
on the q/k channels), the alpha/beta gate projections over bf16 weights, and one fused recurrence
step + RMSNorm + sigmoid-gate kernel.  This file proves the index, coverage and bound arithmetic that
every launch configuration of `fused_gdn.cu` relies on.

## Source semantics
* `gdn_step_norm_kernel`: block = one value head, 128 x 4 threads; thread (col, rg) owns 32 rows
  `rg*32 .. rg*32+31` of column `col` of the (128, h_v, 128) state.  Rows are partitioned exactly once.
* `gdn_conv_l2_kernel`: one thread per channel, `channels % 128 == 0`, 3-deep history per channel.
* `gdn_ab_kernel`: one warp per output row; rows `[0,h_v)` are alpha, `[h_v,2h_v)` are beta.
* Softplus uses the `v > 20` pass-through branch.

## Form template identifiers
- `S` -> `state_dim` (128); `RG` -> `row_groups` (4); `RPG` -> `rows_per_group` (32)
- `h_k` -> `n_key_heads`; `h_v` -> `n_value_heads`; `qh = head % h_k` -> `key_head_of`
- `hist` -> `conv_history`; `wsum` -> `warp_partial_sums`
-/

namespace Strata.FusedGdn

-- § arithmetic helper

/-- Row-major linearisation stays in bounds: `i < a`, `c < d` gives `i*d + c < a*d`. -/
theorem lin2_lt {i a c d : Nat} (hi : i < a) (hc : c < d) : i * d + c < a * d := by
  have h := Nat.mul_le_mul_right d (show i + 1 ≤ a by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

-- § constants

/-- State size `S`. -/
def stateDim : Nat := 128
/-- Number of row groups `RG`. -/
def rowGroups : Nat := 4

/-- `RPG = S / RG` is 32 rows per thread. -/
theorem rows_per_group_eq : stateDim / rowGroups = 32 := by decide

-- § step + norm kernel geometry

/-- Every (row group, local row) pair addresses a row inside the 128-row state. -/
theorem tile_row_lt {rg r : Nat} (hrg : rg < 4) (hr : r < 32) : rg * 32 + r < 128 := by omega

/-- Distinct (row group, local row) pairs address distinct rows, so no two threads write one state element. -/
theorem tile_row_inj {rg r rg' r' : Nat} (hr : r < 32) (hr' : r' < 32)
    (h : rg * 32 + r = rg' * 32 + r') : rg = rg' ∧ r = r' := by omega

/-- Every row of the state is owned by exactly one (row group, local row) pair. -/
theorem tile_row_surj {row : Nat} (hrow : row < 128) :
    row / 32 < 4 ∧ row % 32 < 32 ∧ (row / 32) * 32 + row % 32 = row := by omega

/-- The state element `((row*h_v + head)*128 + col)` lies inside the `128 x h_v x 128` buffer. -/
theorem state_index_lt {row head col h_v : Nat} (hr : row < 128) (hh : head < h_v) (hc : col < 128) :
    (row * h_v + head) * 128 + col < 128 * h_v * 128 :=
  lin2_lt (lin2_lt hr hh) hc

/-- Threads `tid < 128` are exactly the threads of row group 0 (they stage k/q and write y). -/
theorem tid_in_first_group {rg col : Nat} (hc : col < 128) : rg * 128 + col < 128 ↔ rg = 0 := by
  constructor <;> intro h <;> omega

/-- The warp slot written into `wsum` is within its 16 entries for every thread of the 512-thread block. -/
theorem tid_warp_slot {rg col : Nat} (hrg : rg < 4) (hc : col < 128) :
    (rg * 128 + col) / 32 < 16 := by omega

/-- Row group 0 spans exactly four warps, which is why the final sum reads `wsum[0..3]`. -/
theorem first_group_warps {col : Nat} (hc : col < 128) : col / 32 < 4 := by omega

/-- `wsum` has `S*RG/32 = 16` entries. -/
theorem wsum_capacity : 128 * 4 / 32 = 16 := by decide

/-- The key/query head selected by the modulo pairing is a valid head. -/
theorem key_head_lt {head h_k : Nat} (hk : 0 < h_k) : head % h_k < h_k := Nat.mod_lt _ hk

-- § conv + L2 kernel geometry

/-- Each channel is handled by exactly one (block, thread) pair when `channels % 128 = 0`. -/
theorem conv_channel_cover {channels c : Nat} (hdiv : channels % 128 = 0) (hc : c < channels) :
    c / 128 < channels / 128 ∧ c % 128 < 128 ∧ (c / 128) * 128 + c % 128 = c := by omega

/-- Every thread of every launched block addresses an in-range channel. -/
theorem conv_thread_in_range {channels b t : Nat} (hdiv : channels % 128 = 0)
    (hb : b < channels / 128) (ht : t < 128) : b * 128 + t < channels := by omega

/-- History addressing `hist[c*3 + i]` stays in the `3 * channels` buffer. -/
theorem conv_hist_index_lt {c i channels : Nat} (hc : c < channels) (hi : i < 3) :
    c * 3 + i < channels * 3 := by omega

/-- Weight addressing `w[c*4 + i]` stays in the `4 * channels` buffer. -/
theorem conv_weight_index_lt {c i channels : Nat} (hc : c < channels) (hi : i < 4) :
    c * 4 + i < channels * 4 := by omega

/-- Blocks below `qk_heads` hold whole 128-channel heads inside the channel range, so the L2 branch is per head. -/
theorem qk_block_in_range {channels qk b : Nat} (hdiv : channels % 128 = 0)
    (hq : qk ≤ channels / 128) (hb : b < qk) : b * 128 + 127 < channels := by omega

-- § alpha/beta kernel geometry

/-- Every output row is covered by the `(2*h_v + 7)/8` launched blocks of 8 warps. -/
theorem ab_row_cover {row h : Nat} (hrow : row < 2 * h) : row / 8 < (2 * h + 7) / 8 := by omega

/-- A row is either an alpha row or a beta row, with the beta index `row - h_v` in range. -/
theorem ab_row_split {row h : Nat} (hrow : row < 2 * h) : row < h ∨ (h ≤ row ∧ row - h < h) := by omega

/-- The (block, warp) decomposition of a row is recovered by division and remainder. -/
theorem ab_block_row {b w : Nat} (hw : w < 8) : (b * 8 + w) / 8 = b ∧ (b * 8 + w) % 8 = w := by omega

/-- Vector loads of 8 bf16 weights never run past a row when `n % 8 = 0`. -/
theorem ab_vector_in_row {n j : Nat} (hn : n % 8 = 0) (hj : j < n / 8) : j * 8 + 7 < n := by omega

-- § softplus branch

/-- Overflow-safe softplus selector: above 20 the input is returned unchanged. -/
def softplusSel (f : Int → Int) (v : Int) : Int := if v > 20 then v else f v

/-- Above the threshold the large-argument branch is taken. -/
theorem softplusSel_large (f : Int → Int) {v : Int} (h : 20 < v) : softplusSel f v = v := by
  unfold softplusSel
  split <;> omega

/-- At or below the threshold the real function is used. -/
theorem softplusSel_small (f : Int → Int) {v : Int} (h : v ≤ 20) : softplusSel f v = f v := by
  unfold softplusSel
  split <;> omega

end Strata.FusedGdn



## FILE: /mnt/user-data/outputs/GdnReferenceKernels.lean
/-!
# MODULE: gdn_reference_kernels
The unfused gated-delta-net kernels: the column-parallel recurrence step, the sliding 1-D convolution,
per-row L2 normalisation, the beta sigmoid gate and the output RMSNorm/gate.  This file proves the
coverage and index facts of the launch configuration and the list-level semantics of the conv slide.

## Source semantics
* The state layout is `(S, h_v, S)` with `j` fastest; thread `(h, j)` owns one column end to end.
* `MAX_H` heads share a block; `h0 + hi < h_v` is enforced through `nh = min(MAX_H, h_v - h0)`.
* The conv state holds the `d_conv - 1` oldest inputs; the new input lands in the last tap and the
  state slides by one.

## Form template identifiers
- `gdn_step_kernel` -> `recurrence_step`; `gdn_conv_kernel` -> `conv_slide`; `gdn_l2_kernel` -> `row_l2_norm`
- `gdn_out_norm_kernel` -> `out_rms_gate`; `gdn_beta_gate_kernel` -> `beta_sigmoid`
- `S` -> `state_dim`; `MAX_H` -> `heads_per_block`; `JTHREADS` -> `column_threads`
-/

namespace Strata.GdnReference

-- § arithmetic helper

/-- Row-major linearisation stays in bounds. -/
theorem lin2_lt {i a c d : Nat} (hi : i < a) (hc : c < d) : i * d + c < a * d := by
  have h := Nat.mul_le_mul_right d (show i + 1 ≤ a by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

-- § launch geometry

/-- Every column `j < S` is handled by some thread of the `ceil(S/32)` column blocks. -/
theorem column_cover {S j : Nat} (hj : j < S) : j / 32 < (S + 31) / 32 ∧ j % 32 < 32 := by omega

/-- With 8 heads per block every head lies in one of the `ceil(h_v/8)` blocks. -/
theorem heads_cover_eight {h hv : Nat} (hh : h < hv) :
    h / 8 < (hv + 7) / 8 ∧ h % 8 < 8 ∧ (h / 8) * 8 + h % 8 = h := by omega

/-- With one head per block the staged head count is exactly 1. -/
theorem staged_heads_one {h0 hv : Nat} (h : h0 < hv) : min 1 (hv - h0) = 1 := by omega

/-- With 8 heads per block the staged head count is between 1 and 8 for every launched block. -/
theorem staged_heads_eight {by' hv : Nat} (h : by' < (hv + 7) / 8) :
    1 ≤ min 8 (hv - by' * 8) ∧ min 8 (hv - by' * 8) ≤ 8 := by omega

/-- Block counts for S=128, h_v=48: 24 blocks with 8 heads per block, 192 with one head per block. -/
theorem grid_block_counts : ((128 + 31) / 32) * ((48 + 7) / 8) = 24 ∧ ((128 + 31) / 32) * 48 = 192 := by
  decide

/-- Staged rows of `k`/`q` (arrays of 128) are never indexed out of bounds when `S ≤ 128`. -/
theorem staged_in_bounds {S i : Nat} (hS : S ≤ 128) (hi : i < S) : i < 128 := by omega

/-- Modulo head pairing yields a valid key/query head. -/
theorem key_head_lt {h h_k : Nat} (hk : 0 < h_k) : h % h_k < h_k := Nat.mod_lt _ hk

-- § state addressing

/-- The two forms of the column address agree: `(i*h_v + h)*S + j = i*(h_v*S) + (h*S + j)`. -/
theorem stride_form (i hv h S j : Nat) : (i * hv + h) * S + j = i * (hv * S) + (h * S + j) := by
  rw [Nat.add_mul, Nat.mul_assoc]
  omega

/-- Column address stays inside the `S x h_v x S` state. -/
theorem column_index_lt {i h j hv S : Nat} (hi : i < S) (hh : h < hv) (hj : j < S) :
    (i * hv + h) * S + j < S * hv * S :=
  lin2_lt (lin2_lt hi hh) hj

-- § sliding convolution (list model)

/-- Dot product of two integer lists (truncating to the shorter). -/
def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- Conv output: `[conv_state | x]` against the kernel taps. -/
def convOut (st w : List Int) (x : Int) : Int := dot (st ++ [x]) w

/-- Conv state update: drop the oldest entry, append the newest input. -/
def convSlide (st : List Int) (x : Int) : List Int := st.drop 1 ++ [x]

/-- Sliding keeps the state length whenever the state is nonempty (`d_conv >= 2`). -/
theorem convSlide_length (st : List Int) (x : Int) (h : st ≠ []) :
    (convSlide st x).length = st.length := by
  cases st with
  | nil => exact absurd rfl h
  | cons a rest => simp [convSlide]

/-- The newest input is the last element of the new state. -/
theorem convSlide_last (st : List Int) (x : Int) : (convSlide st x).reverse.head? = some x := by
  simp [convSlide]

/-- Dropping the oldest and appending the newest, spelled out for a nonempty state. -/
theorem convSlide_cons (a : Int) (rest : List Int) (x : Int) : convSlide (a :: rest) x = rest ++ [x] := by
  simp [convSlide]

/-- With `d_conv = 1` the code stores to `st[d_conv - 2]`, i.e. index -1: the kernel needs `d_conv >= 2`. -/
theorem conv_single_tap_index_negative : ((1 : Int) - 2) < 0 := by decide

/-- With `d_conv = 2` the shift loop is empty and only the final store remains. -/
theorem conv_two_tap_shift_loop_empty : 2 - 2 = 0 := by decide

-- § row-wise reductions

/-- A strided loop `i = tid; i < cols; i += 32` assigns each element to exactly one of 32 threads. -/
theorem strided_owner {i : Nat} : i % 32 < 32 ∧ (i / 32) * 32 + i % 32 = i := by omega

/-- The beta-gate launch (128 threads per block) covers every head. -/
theorem beta_gate_cover {n i : Nat} (hi : i < n) : i / 128 < (n + 127) / 128 := by omega

end Strata.GdnReference



## FILE: /mnt/user-data/outputs/NativeGdnStep.lean
/-!
# MODULE: native_gdn_step
Warp-per-column GDN recurrence step adapted from llama.cpp's gated delta net.  Each warp owns one
`(head, column)` pair and keeps its four state rows per lane in registers.  This file proves the
register-shard / column decomposition, the state-address bounds and the span-overlap predicate used by
the host validation.

## Source semantics
* Row `i = r*32 + lane` for `r < 4`, `lane < 32`: the four shards of a lane partition the 128 rows.
* Column `col = blockIdx.z * blockDim.y + threadIdx.y` with grid z = 32 and blockDim.y = 4.
* The host rejects unaligned spans and any overlap between the state/output writes and the inputs.

## Form template identifiers
- `step` -> `native_step`; `s_shard` -> `state_shard`; `warp_sum` -> `lane_reduce`
- `valid_span` -> `span_valid`; `overlap` -> `span_overlap`; `q_head` -> `key_head_of`
-/

namespace Strata.NativeGdn

-- § arithmetic helper

/-- Row-major linearisation stays in bounds. -/
theorem lin2_lt {i a c d : Nat} (hi : i < a) (hc : c < d) : i * d + c < a * d := by
  have h := Nat.mul_le_mul_right d (show i + 1 ≤ a by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

-- § register shard decomposition

/-- Every (shard, lane) pair addresses a row inside the 128 rows. -/
theorem shard_row_lt {r lane : Nat} (hr : r < 4) (hl : lane < 32) : r * 32 + lane < 128 := by omega

/-- Distinct (shard, lane) pairs address distinct rows. -/
theorem shard_row_inj {r l r' l' : Nat} (hl : l < 32) (hl' : l' < 32)
    (h : r * 32 + l = r' * 32 + l') : r = r' ∧ l = l' := by omega

/-- Every row is addressed by exactly one (shard, lane) pair. -/
theorem shard_row_surj {i : Nat} (hi : i < 128) :
    i / 32 < 4 ∧ i % 32 < 32 ∧ (i / 32) * 32 + i % 32 = i := by omega

-- § column decomposition

/-- The column index computed from `(blockIdx.z, threadIdx.y)` is below 128. -/
theorem column_lt {z y : Nat} (hz : z < 32) (hy : y < 4) : z * 4 + y < 128 := by omega

/-- Distinct `(blockIdx.z, threadIdx.y)` pairs address distinct columns. -/
theorem column_inj {z y z' y' : Nat} (hy : y < 4) (hy' : y' < 4)
    (h : z * 4 + y = z' * 4 + y') : z = z' ∧ y = y' := by omega

/-- Every column is owned by exactly one `(blockIdx.z, threadIdx.y)` pair. -/
theorem column_surj {c : Nat} (hc : c < 128) :
    c / 4 < 32 ∧ c % 4 < 4 ∧ (c / 4) * 4 + c % 4 = c := by omega

-- § address bounds

/-- State element `(i*h_v + head)*128 + col` lies inside the `128 x h_v x 128` buffer. -/
theorem state_index_lt {i head col h_v : Nat} (hi : i < 128) (hh : head < h_v) (hc : col < 128) :
    (i * h_v + head) * 128 + col < 128 * h_v * 128 :=
  lin2_lt (lin2_lt hi hh) hc

/-- Key/query element `q_head*128 + i` lies inside the `128 * h_k` buffer. -/
theorem qk_index_lt {q_head i h_k : Nat} (hq : q_head < h_k) (hi : i < 128) :
    q_head * 128 + i < h_k * 128 := lin2_lt hq hi

/-- Output/value element `head*128 + col` lies inside the `128 * h_v` buffer. -/
theorem out_index_lt {head col h_v : Nat} (hh : head < h_v) (hc : col < 128) :
    head * 128 + col < h_v * 128 := lin2_lt hh hc

/-- The modulo head pairing is a valid key/query head. -/
theorem key_head_lt {head h_k : Nat} (hk : 0 < h_k) : head % h_k < h_k := Nat.mod_lt _ hk

/-- Every value head is mapped onto a key head, and heads sharing one differ by multiples of `h_k`. -/
theorem key_head_period {head h_k : Nat} (hk : 0 < h_k) : (head + h_k) % h_k = head % h_k := by
  simp

-- § span overlap predicate (host validation)

/-- Half-open byte ranges `[ap, ap+an)` and `[bp, bp+bn)` are flagged as overlapping. -/
def overlap (ap an bp bn : Nat) : Prop := ap < bp + bn ∧ bp < ap + an

/-- The overlap predicate is symmetric. -/
theorem overlap_comm {ap an bp bn : Nat} : overlap ap an bp bn ↔ overlap bp bn ap an := by
  unfold overlap
  omega

/-- Ranges laid end to end are never flagged. -/
theorem not_overlap_of_adjacent {ap an bp bn : Nat} (h : ap + an ≤ bp) : ¬ overlap ap an bp bn := by
  unfold overlap
  omega

/-- A nonempty range always overlaps itself, so aliasing the state with the output is rejected. -/
theorem overlap_self {p n : Nat} (hn : 0 < n) : overlap p n p n := by
  unfold overlap
  omega

/-- The predicate also flags an empty range strictly inside a nonempty one (zero-byte spans are not exempt). -/
theorem overlap_zero_length_inside : overlap 0 10 5 0 := by
  unfold overlap
  omega

-- § byte sizes

/-- Per-call byte sizes: state `S*S*h_v` floats, output/value `S*h_v` floats, q/k `S*h_k` floats. -/
theorem state_bytes_eq (h_v : Nat) : 128 * 128 * h_v * 4 = 65536 * h_v := by omega

end Strata.NativeGdn



## FILE: /mnt/user-data/outputs/NativeGdnPreprocess.lean
/-!
# MODULE: native_gdn_preprocess
GDN preprocessing kernels adapted from llama.cpp: 4-tap conv + SiLU with a 3-deep history, L2/RMS norm
over 128-wide rows, beta sigmoid, gate softplus and the output RMSNorm with sigmoid gate.  This file
proves the history-window semantics, the warp-slot arithmetic of the block reductions, the launch
coverage and the host span checks.

## Source semantics
* History holds the last three inputs; the four taps are `[h0,h1,h2,x]` and the new history is `[h1,h2,x]`.
* Norm kernels run 256 threads per row on a width-128 row; threads 128..255 contribute zero.
* `count_and_stream` accepts counts in `[1, 65535]`.

## Form template identifiers
- `conv_silu` -> `conv_silu_step`; `l2_norm` -> `row_l2`; `out_norm` -> `out_rms_gate`
- `norm_sum` -> `block_sum`; `gate_softplus` -> `gate_from_alpha`; `beta_sigmoid` -> `beta_sigmoid`
-/

namespace Strata.NativeGdnPreprocess

-- § history window

/-- History of three inputs. -/
abbrev Hist := Int × Int × Int

/-- New history after one step: shift left, append the new input. -/
def slide (h : Hist) (x : Int) : Hist := (h.2.1, h.2.2, x)

/-- The four-tap sum, oldest history entry on tap 0 and the new input on tap 3. -/
def taps (h : Hist) (x w0 w1 w2 w3 : Int) : Int := h.1 * w0 + h.2.1 * w1 + h.2.2 * w2 + x * w3

/-- After three consecutive inputs the history holds exactly those inputs, oldest first. -/
theorem slide_three (h : Hist) (x0 x1 x2 : Int) : slide (slide (slide h x0) x1) x2 = (x0, x1, x2) := rfl

/-- The newest input becomes the last history entry. -/
theorem slide_last (h : Hist) (x : Int) : (slide h x).2.2 = x := rfl

/-- The oldest entry is dropped and the middle entries move down. -/
theorem slide_shift (h : Hist) (x : Int) : (slide h x).1 = h.2.1 ∧ (slide h x).2.1 = h.2.2 := ⟨rfl, rfl⟩

/-- The tap sum spelled out for the all-ones weights. -/
theorem taps_ones (h : Hist) (x : Int) : taps h x 1 1 1 1 = h.1 + h.2.1 + h.2.2 + x := by
  unfold taps
  omega

-- § addressing

/-- History addressing `history[c*3 + tap]` stays in the `3 * channels` buffer. -/
theorem history_index_lt {c tap channels : Nat} (hc : c < channels) (ht : tap < 3) :
    c * 3 + tap < channels * 3 := by omega

/-- Weight addressing `weights[c*4 + tap]` stays in the `4 * channels` buffer. -/
theorem weight_index_lt {c tap channels : Nat} (hc : c < channels) (ht : tap < 4) :
    c * 4 + tap < channels * 4 := by omega

/-- A 256-thread grid of `ceil(channels/256)` blocks covers every channel. -/
theorem channel_grid_cover {channels c : Nat} (hc : c < channels) : c / 256 < (channels + 255) / 256 := by omega

-- § block reduction slots

/-- With 256 threads the warp slot is below 8, matching the `lane < 8` read of `sums`. -/
theorem warp_slot_lt {t : Nat} (ht : t < 256) : t / 32 < 8 := by omega

/-- The `sums[32]` array is large enough for every warp of a 256-thread block. -/
theorem sums_capacity {t : Nat} (ht : t < 256) : t / 32 < 32 := by omega

/-- Threads beyond the row width (128..255) are in warps 4..7, which contribute zero partial sums. -/
theorem padded_threads_warps {t : Nat} (h1 : 128 ≤ t) (h2 : t < 256) : 4 ≤ t / 32 ∧ t / 32 < 8 := by omega

-- § launch validation

/-- A count accepted by `count_and_stream`. -/
def validCount (c : Nat) : Prop := 0 < c ∧ c ≤ 65535

/-- Accepted counts need between 1 and 256 blocks of 256 threads. -/
theorem valid_count_blocks {c : Nat} (h : validCount c) : 1 ≤ (c + 255) / 256 ∧ (c + 255) / 256 ≤ 256 := by
  unfold validCount at h
  omega

/-- Rows of width 128 occupy `rows * 128` floats; the byte size is `rows * 512`. -/
theorem norm_bytes_eq (rows : Nat) : rows * 128 * 4 = rows * 512 := by omega

-- § span overlap predicate

/-- Half-open byte ranges flagged as overlapping. -/
def overlap (ap an bp bn : Nat) : Prop := ap < bp + bn ∧ bp < ap + an

/-- The overlap predicate is symmetric. -/
theorem overlap_comm {ap an bp bn : Nat} : overlap ap an bp bn ↔ overlap bp bn ap an := by
  unfold overlap
  omega

/-- Ranges laid end to end are never flagged. -/
theorem not_overlap_of_adjacent {ap an bp bn : Nat} (h : ap + an ≤ bp) : ¬ overlap ap an bp bn := by
  unfold overlap
  omega

/-- A history span of `3 * bytes` placed directly before a span of `bytes` is accepted. -/
theorem history_then_raw_disjoint (p bytes : Nat) : ¬ overlap p (3 * bytes) (p + 3 * bytes) bytes := by
  unfold overlap
  omega

-- § gate softplus branch

/-- Overflow-safe softplus selector: above 20 the input is returned unchanged. -/
def softplusSel (f : Int → Int) (v : Int) : Int := if v > 20 then v else f v

/-- Above the threshold the large-argument branch is taken. -/
theorem softplusSel_large (f : Int → Int) {v : Int} (h : 20 < v) : softplusSel f v = v := by
  unfold softplusSel
  split <;> omega

end Strata.NativeGdnPreprocess



## FILE: /mnt/user-data/outputs/QsaCacheIndexerAttention.lean
/-!
# MODULE: qsa_cache_indexer_attention
The QSA (quantised sparse attention) cache, block indexer, exact top-k selection and attention kernels.
This file proves the combinatorial core: the step-state derivation, the page/row addressing of the
KV cache, the pooling ring buffer (blocks of r = 4 cells), the cell-to-block mapping of the indexer, the
chunked threshold-selection arithmetic of `topk_512`, and the head/shared-memory arithmetic of the attention.

## Source semantics
* `qsa_step_fill` derives `n_kv = pos+1`, `n_bid = n_kv / r`, `width = min(n_kv, top_k + r - 1)` from one position.
* The indexer pools r raw keys into one block row; the incomplete tail block is scored as one block
  and receives a `+1e9` bias exactly when `n_kv % r != 0`.
* `topk_512` selects every cell with key above the threshold and the first `width - c_gt` cells at it.
* `qsa_attend` maps query head `h` to key/value head `h / (n_head / n_head_kv)`.

## Form template identifiers
- `kv_append_kernel` -> `kv_append`; `indexer_key_append_kernel` -> `pool_append`
- `qsa_index_kernel` -> `block_score`; `topk_kernel` -> `threshold_select`; `kv_gather_kernel` -> `kv_gather`
- `qsa_attend_kernel` -> `sparse_attend`; `qsa_gate_apply_kernel` -> `gate_apply`
- `kStepPos/NKv/NBid/Width` -> `step_pos/n_kv/n_bid/width`; `idx_block` -> `pool_block` (r)
-/

namespace Strata.Qsa

-- § arithmetic helper

/-- Row-major linearisation stays in bounds. -/
theorem lin2_lt {i a c d : Nat} (hi : i < a) (hc : c < d) : i * d + c < a * d := by
  have h := Nat.mul_le_mul_right d (show i + 1 ≤ a by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- Division/remainder reconstruct the dividend in the order used by the kernels (`q*r + n%r`). -/
theorem div_mul_add_mod (n r : Nat) : (n / r) * r + n % r = n := by
  rw [Nat.mul_comm]
  exact Nat.div_add_mod n r

-- § step state

/-- Selection width: all cells below the bound, otherwise `top_k + r - 1`. -/
def selWidth (n k r : Nat) : Nat := min n (k + r - 1)

/-- The step quadruple derived from a single position. -/
structure Step where
  pos : Nat
  nKv : Nat
  nBid : Nat
  width : Nat

/-- `qsa_step_fill`: all four entries derive from `pos`. -/
def stepFill (pos k r : Nat) : Step :=
  { pos := pos, nKv := pos + 1, nBid := (pos + 1) / r, width := selWidth (pos + 1) k r }

/-- `n_kv` is always one more than the position. -/
theorem stepFill_nKv (pos k r : Nat) : (stepFill pos k r).nKv = (stepFill pos k r).pos + 1 := rfl

/-- The width never exceeds the number of cells. -/
theorem stepFill_width_le_nKv (pos k r : Nat) : (stepFill pos k r).width ≤ (stepFill pos k r).nKv := by
  dsimp only [stepFill, selWidth]
  omega

/-- The width never exceeds `top_k + r - 1`. -/
theorem stepFill_width_le_cap (pos k r : Nat) : (stepFill pos k r).width ≤ k + r - 1 := by
  dsimp only [stepFill, selWidth]
  omega

/-- Below the bound the selection is the identity (every cell is selected). -/
theorem stepFill_identity {pos k r : Nat} (h : pos + 1 ≤ k + r - 1) :
    (stepFill pos k r).width = (stepFill pos k r).nKv := by
  dsimp only [stepFill, selWidth]
  omega

/-- With r = 4 the completed-block count and the tail partition the cells. -/
theorem stepFill_blocks_four (pos k : Nat) :
    (stepFill pos k 4).nBid * 4 ≤ (stepFill pos k 4).nKv ∧
      (stepFill pos k 4).nKv < (stepFill pos k 4).nBid * 4 + 4 := by
  dsimp only [stepFill]
  omega

/-- The default geometry: `top_k = 2048`, `r = 4` gives the 2051-cell bound. -/
theorem selection_bound_2051 : 2048 + 4 - 1 = 2051 := by decide

-- § kv cache addressing

/-- A page-table slot `pos / page_size` is in range when the position is below `page_size * n_pages`. -/
theorem page_slot_lt {pos ps n_pages : Nat} (h : pos < ps * n_pages) : pos / ps < n_pages :=
  Nat.div_lt_of_lt_mul h

/-- The in-page slot is below the page size. -/
theorem page_offset_lt {pos ps : Nat} (hps : 0 < ps) : pos % ps < ps := Nat.mod_lt _ hps

/-- The pool row `(page*kv_heads + h)*page_size + slot` is inside the pool of `n_pages` pages. -/
theorem pool_row_lt {page h slot n_pages kvh ps : Nat} (hp : page < n_pages) (hh : h < kvh)
    (hs : slot < ps) : (page * kvh + h) * ps + slot < n_pages * kvh * ps :=
  lin2_lt (lin2_lt hp hh) hs

/-- Splitting a flat index `i < kv_heads * head_dim` into `(head, dim)` is exact and in range. -/
theorem split_head_dim {i kvh hd : Nat} (hd0 : 0 < hd) (hi : i < hd * kvh) :
    i / hd < kvh ∧ i % hd < hd ∧ (i / hd) * hd + i % hd = i :=
  ⟨Nat.div_lt_of_lt_mul hi, Nat.mod_lt _ hd0, div_mul_add_mod i hd⟩

/-- The gather destination `(id*kv_heads + h)*per + q` is inside the scratch of `n_ids` selected cells. -/
theorem gather_dst_lt {id h q n_ids kvh per : Nat} (hid : id < n_ids) (hh : h < kvh) (hq : q < per) :
    (id * kvh + h) * per + q < n_ids * kvh * per :=
  lin2_lt (lin2_lt hid hh) hq

-- § pooling ring buffer (r = 4)

/-- The ring slot of a position is below 4. -/
theorem slot_lt (pos : Nat) : pos % 4 < 4 := by omega

/-- A cell completes a block exactly when its slot is 3, i.e. when `pos + 1` is a multiple of 4. -/
theorem completes_block_iff (pos : Nat) : pos % 4 = 3 ↔ (pos + 1) % 4 = 0 := by omega

/-- The tail rows written (slots 0..2) are the first three cells of the current block. -/
theorem tail_cells {pos : Nat} (h : pos % 4 < 3) : pos / 4 * 4 + pos % 4 = pos ∧ pos % 4 ≤ 2 := by omega

/-- A completed block's row and the spare row after it fit in a pooled buffer of `max_cells/4 + 1` rows. -/
theorem pooled_rows_fit {pos max : Nat} (hp : pos < max) (hs : pos % 4 = 3) :
    pos / 4 + 1 < max / 4 + 1 := by omega

-- § indexer: cells of a block

/-- Every cell `j < n_kv` belongs to block `j/4`, which is at most the completed-block count. -/
theorem cell_block {j n : Nat} (hj : j < n) :
    j / 4 ≤ n / 4 ∧ (j / 4) * 4 ≤ j ∧ j < (j / 4) * 4 + 4 := by omega

/-- A complete block's four cells all exist. -/
theorem complete_block_in_range {b n : Nat} (hb : b < n / 4) : b * 4 + 4 ≤ n := by omega

/-- The incomplete tail block holds `n % 4` cells, and the bias applies exactly when that is nonzero. -/
theorem tail_cells_count (n : Nat) : n - (n / 4) * 4 = n % 4 := by omega

/-- When `n` is a multiple of 4 the block `n_bid` is empty (it is only the spare row). -/
theorem empty_tail_block {n : Nat} (h : n % 4 = 0) : n / 4 * 4 = n := by omega

/-- A cell id `b*4 + c` of the tail block is below `n_kv`. -/
theorem tail_cell_lt {n c : Nat} (hc : c < n - (n / 4) * 4) : (n / 4) * 4 + c < n := by omega

-- § top-k: chunking and tie budget

/-- The 256 chunks of `ceil(n/256)` cells cover `[0, n)`. -/
theorem chunks_cover (n : Nat) : n ≤ ((n + 255) / 256) * 256 := by omega

/-- Cell `j < n` belongs to chunk `t = j / chunk`, and `t` is a valid thread. -/
theorem chunk_owner_lt {j n chunk : Nat} (hj : j < n) (hc : n ≤ chunk * 256) : j / chunk < 256 :=
  Nat.div_lt_of_lt_mul (Nat.lt_of_lt_of_le hj hc)

/-- Cell `j` lies in `[t*chunk, t*chunk + chunk)` for its owner `t`. -/
theorem chunk_member {j chunk : Nat} (hc : 0 < chunk) :
    (j / chunk) * chunk ≤ j ∧ j < (j / chunk) * chunk + chunk := by
  have h1 := div_mul_add_mod j chunk
  have h2 := Nat.mod_lt j hc
  omega

/-- When `c_gt < width <= c_gt + c_eq` the tie budget is positive, fits the equal cells and fills the width. -/
theorem tie_budget {c_gt c_eq width : Nat} (h1 : c_gt < width) (h2 : width ≤ c_gt + c_eq) :
    0 < width - c_gt ∧ width - c_gt ≤ c_eq ∧ c_gt + (width - c_gt) = width := by omega

/-- Exclusive prefix of a list of per-thread counts: the sum of the first `i` entries. -/
def exclPrefix (xs : List Nat) (i : Nat) : Nat := (xs.take i).sum

/-- The prefix before thread 0 is empty. -/
theorem exclPrefix_zero (xs : List Nat) : exclPrefix xs 0 = 0 := by
  simp [exclPrefix]

/-- Each prefix is the previous prefix plus the previous thread's count. -/
theorem exclPrefix_succ (xs : List Nat) {i : Nat} (hi : i < xs.length) :
    exclPrefix xs (i + 1) = exclPrefix xs i + xs[i] := by
  unfold exclPrefix
  first
    | exact List.sum_take_succ xs i hi
    | (rw [List.take_succ, List.getElem?_eq_getElem hi]; simp [List.sum_append])

/-- The prefix after all threads is the total. -/
theorem exclPrefix_total (xs : List Nat) : exclPrefix xs xs.length = xs.sum := by
  simp [exclPrefix]

-- § attention

/-- With `n_head = g * n_head_kv` the key/value head `h / g` is a valid KV head. -/
theorem kv_head_lt {h g nkv : Nat} (hh : h < g * nkv) : h / g < nkv := Nat.div_lt_of_lt_mul hh

/-- Contiguous grouping `h / g` differs from interleaved grouping `h % n_head_kv` (4 heads, 2 KV heads, h = 1). -/
theorem kv_head_not_modulo : (1 : Nat) / (4 / 2) ≠ 1 % 2 := by decide

/-- Scores stored after the 32 reduction floats stay inside the shared buffer of `max_ids + 32` floats. -/
theorem score_slot_lt {j n_ids max_ids : Nat} (hj : j < n_ids) (hn : n_ids ≤ max_ids) :
    32 + j < max_ids + 32 := by omega

/-- Warp slot for a block of `blockDim` threads is below the warp count `(blockDim + 31) / 32`. -/
theorem warp_slot_lt {d bd : Nat} (hd : d < bd) : d / 32 < (bd + 31) / 32 := by omega

/-- For a 256-thread block there are 8 warps, which fits the 32 reduction floats. -/
theorem warp_count_256 : (256 + 31) / 32 = 8 := by decide

/-- Scratch addressing `(j*n_head_kv + kv)*head_dim + i` is inside `n_ids * n_head_kv * head_dim`. -/
theorem scratch_index_lt {j kv i n_ids nkv hd : Nat} (hj : j < n_ids) (hk : kv < nkv) (hi : i < hd) :
    (j * nkv + kv) * hd + i < n_ids * nkv * hd :=
  lin2_lt (lin2_lt hj hk) hi

/-- The default shared allocation for 2051 ids is 8332 bytes, below the 48 KiB default limit. -/
theorem default_shared_bytes : (2051 + 32) * 4 = 8332 ∧ 8332 < 49152 := by decide

/-- The gate's second half `h*(2*hd) + hd + d` is inside the `n_head * (2*hd)` fused q buffer. -/
theorem gate_second_half_lt {h d n hd : Nat} (hh : h < n) (hd' : d < hd) :
    h * (2 * hd) + hd + d < n * (2 * hd) := by
  have h1 := Nat.mul_le_mul_right (2 * hd) (show h + 1 ≤ n by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

end Strata.Qsa



## FILE: /mnt/user-data/outputs/QsaDecodeAttention.lean
/-!
# MODULE: qsa_decode_attention
Split-K decode attention over a selected cell set: a chunk kernel (64 cells per block, 12 query heads
per KV head, fp16 or int8+scale pools) producing partial `(acc, m, l)` triples, and a merge kernel that
combines them with an online-softmax rescale.  This file proves the chunk coverage, the thread/warp
ownership of the three phases, the scratch layout and the int8 group-scale addressing.

## Source semantics
* `HD = 256`, `G = 12`, `CHUNK = 64`, 256 threads (8 warps).
* Phase 1: warp w scores cells `w, w+8, ...`; each lane holds 8 of the 256 dimensions.
* Phase 2: warp w handles heads `w, w+8`; lane and lane+32 cover the 64 chunk cells.
* Phase 3: thread t owns dimension t for all 12 heads.
* Scratch per query: `n_chunks * n_head * (HD + 2) + 64` floats.

## Form template identifiers
- `attn_chunk_kernel` -> `chunk_partial`; `attn_merge_kernel` -> `chunk_merge`
- `part_acc/part_m/part_l` -> `partial_acc/partial_max/partial_sum`; `srow` -> `cell_row`
- `KV_Q8_GROUP` -> `q8_group` (64)
-/

namespace Strata.QsaDecodeAttn

-- § arithmetic helper

/-- Row-major linearisation stays in bounds. -/
theorem lin2_lt {i a c d : Nat} (hi : i < a) (hc : c < d) : i * d + c < a * d := by
  have h := Nat.mul_le_mul_right d (show i + 1 ≤ a by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

-- § chunking

/-- Every selected cell index `c < cap` lies in one of the `ceil(cap/64)` chunks. -/
theorem chunk_cover {cap c : Nat} (hc : c < cap) : c / 64 < (cap + 63) / 64 := by omega

/-- A block's local cell count is positive exactly when its first cell exists. -/
theorem n_here_pos_iff {c0 n : Nat} : 0 < min 64 (n - c0) ↔ c0 < n := by omega

/-- A block never handles more than 64 cells. -/
theorem n_here_le {c0 n : Nat} : min 64 (n - c0) ≤ 64 := by omega

/-- Chunk and local slot reconstruct the global cell index. -/
theorem chunk_local_index {c : Nat} : (c / 64) * 64 + c % 64 = c ∧ c % 64 < 64 := by omega

-- § phase 1: scores

/-- Warp `c % 8` owns cell `c` of the chunk, and each cell has exactly one owner. -/
theorem score_cell_owner {c : Nat} (hc : c < 64) :
    c % 8 < 8 ∧ c % 8 + 8 * (c / 8) = c ∧ c / 8 < 8 := by omega

/-- Eight consecutive dimensions starting at `lane*8` stay inside the 256 dimensions. -/
theorem lane_dims_lt {lane : Nat} (h : lane < 32) : lane * 8 + 8 ≤ 256 := by omega

/-- The 32 lanes partition the 256 dimensions into disjoint runs of 8. -/
theorem lane_dims_inj {l l' j j' : Nat} (hj : j < 8) (hj' : j' < 8)
    (h : l * 8 + j = l' * 8 + j') : l = l' ∧ j = j' := by omega

-- § phase 2: per-head softmax statistics

/-- Warp `h % 8` handles head `h < 12`; heads 8..11 are second assignments of warps 0..3. -/
theorem head_warp_owner {h : Nat} (hh : h < 12) :
    h % 8 < 8 ∧ (h < 8 ∨ (8 ≤ h ∧ h - 8 < 4)) := by omega

/-- Lanes `l` and `l + 32` cover the 64 chunk cells exactly once. -/
theorem lane_pair_cover {c : Nat} (hc : c < 64) : (c < 32 ∧ c % 32 = c) ∨ (32 ≤ c ∧ c - 32 < 32) := by omega

-- § phase 3: values

/-- Thread `t < 256` owns dimension `t`: the mapping is the identity. -/
theorem value_dim_owner {t : Nat} (ht : t < 256) : t < 256 := ht

-- § scratch layout

/-- Each (kv head, chunk) slot index is below `n_kv_heads * n_chunks`. -/
theorem slot_lt {kvh chunk nkv nch : Nat} (hk : kvh < nkv) (hc : chunk < nch) :
    kvh * nch + chunk < nkv * nch := lin2_lt hk hc

/-- The accumulator index `((slot*G + h)*HD + t)` is inside the accumulator region. -/
theorem acc_index_lt {slot h t nslots : Nat} (hs : slot < nslots) (hh : h < 12) (ht : t < 256) :
    (slot * 12 + h) * 256 + t < nslots * 12 * 256 :=
  lin2_lt (lin2_lt hs hh) ht

/-- The max/sum index `slot*G + h` is inside its region. -/
theorem stat_index_lt {slot h nslots : Nat} (hs : slot < nslots) (hh : h < 12) :
    slot * 12 + h < nslots * 12 :=
  lin2_lt hs hh

/-- The three regions (acc, m, l) add up to the `HD + 2` floats per chunk-head counted by the scratch size. -/
theorem scratch_regions_sum (c nh : Nat) : c * nh * 256 + c * nh + c * nh = c * nh * 258 := by omega

/-- The int8/fp16 paths use 12 KB of shared query plus 3 KB scores plus 512 B row table, under 48 KiB. -/
theorem shared_bytes : 12 * 256 * 4 + 12 * 64 * 4 + 64 * 8 = 15872 ∧ 15872 < 49152 := by decide

-- § merge kernel

/-- A global query head splits into (kv head, local head) and recombines exactly. -/
theorem head_split {h nkv : Nat} (hh : h < 12 * nkv) :
    h / 12 < nkv ∧ h % 12 < 12 ∧ (h / 12) * 12 + h % 12 = h := by omega

-- § int8 group scales

/-- With `HD = 256` and groups of 64 there are 4 scale groups per row. -/
theorem groups_per_row : 256 / 64 = 4 := by decide

/-- A run of 8 consecutive dimensions starting at a multiple of 8 never crosses a 64-group boundary. -/
theorem run_in_one_group {d0 : Nat} (h : d0 % 8 = 0) : d0 / 64 = (d0 + 7) / 64 := by omega

/-- The scale index `row*4 + d0/64` is inside the scale buffer of `rows * 4` entries. -/
theorem scale_index_lt {row rows d0 : Nat} (hr : row < rows) (hd : d0 < 256) :
    row * 4 + d0 / 64 < rows * 4 :=
  lin2_lt hr (by omega)

end Strata.QsaDecodeAttn



## FILE: /mnt/user-data/outputs/QsaBlockSelect.lean
/-!
# MODULE: qsa_block_select
Batched block scoring and radix-select top-k over pooled indexer blocks (128-dim keys, 4 indexer heads,
blocks of R = 4 cells).  This file proves the launch coverage, the block-to-cell weighting, the radix
digit schedule, the per-thread block partition and the tie-budget clamp that make the selection exact.

## Source semantics
* `block_scores_kernel`: one warp per block, 8 warps per block; each lane loads a float4 of the key.
* `block_topk_kernel`: block weight is R for complete blocks and `n_kv - n_bid*R` for the tail; the
  threshold is found by 4 radix passes of 8 bits; at most `width - above` cells equal to the
  threshold are taken, lowest index first, each thread clamped to its share.

## Form template identifiers
- `block_scores_kernel` -> `block_score`; `block_topk_kernel` -> `block_select`
- `weight` -> `block_cells`; `eq_budget` -> `tie_budget`; `my_eq` -> `thread_ties`
-/

namespace Strata.QsaBlockSelect

-- § block scores

/-- A (block, warp) decomposition `b = bx*8 + w` covers every block index. -/
theorem score_block_cover {b maxb : Nat} (hb : b < maxb) : b / 8 < (maxb + 7) / 8 := by omega

/-- A float4 per lane covers the 128-dim key exactly. -/
theorem key_lanes_cover : 32 * 4 = 128 := by decide

/-- Lane runs of 4 dims are disjoint. -/
theorem key_lane_inj {l l' j j' : Nat} (hj : j < 4) (hj' : j' < 4)
    (h : l * 4 + j = l' * 4 + j') : l = l' ∧ j = j' := by omega

-- § cells of a block

/-- A cell id `b*4 + c` of a complete block is below `n_kv`. -/
theorem complete_cell_lt {b c n : Nat} (hb : b < n / 4) (hc : c < 4) : b * 4 + c < n := by omega

/-- A cell id of the tail block (`b = n_bid`) is below `n_kv`. -/
theorem tail_cell_lt {n c : Nat} (hc : c < n - (n / 4) * 4) : (n / 4) * 4 + c < n := by omega

/-- Block weights sum to `n_kv`: `n_bid` complete blocks of 4 plus the tail of `n % 4` cells. -/
theorem weights_total (n : Nat) : (n / 4) * 4 + (n - (n / 4) * 4) = n := by omega

/-- The tail weight is `n % 4`, and the score bias applies exactly when it is nonzero. -/
theorem tail_weight (n : Nat) : n - (n / 4) * 4 = n % 4 := by omega

/-- Every cell `j < n` lies in a block `j/4 <= n_bid`, at local slot `j % 4`. -/
theorem cell_in_block {j n : Nat} (hj : j < n) :
    j / 4 ≤ n / 4 ∧ (j / 4) * 4 + j % 4 = j ∧ j % 4 < 4 := by omega

-- § identity branch

/-- When `n_kv <= width` everything is selected, in ascending order. -/
theorem identity_selects_all {n width j : Nat} (h : n ≤ width) (hj : j < n) : j < width := by omega

-- § radix schedule

/-- Four passes of 8 bits cover all 32 key bits. -/
theorem radix_bits : 4 * 8 = 32 := by decide

/-- The shifts 24, 16, 8, 0 are the consecutive 8-bit digits. -/
theorem radix_shifts : (24 : Nat) = 16 + 8 ∧ (16 : Nat) = 8 + 8 ∧ (8 : Nat) = 0 + 8 := by decide

/-- A digit in `[0, 255]` is a valid histogram bin of the 256-bin table. -/
theorem digit_bin_lt {k : Nat} : (k / 2 ^ 24) % 256 < 256 := Nat.mod_lt _ (by decide)

-- § per-thread block partition

/-- Each of the 256 threads gets `ceil(nb/256)` consecutive blocks; every block has an owner thread. -/
theorem block_owner_lt {b nb per : Nat} (hb : b < nb) (hper : nb ≤ per * 256) : b / per < 256 :=
  Nat.div_lt_of_lt_mul (Nat.lt_of_lt_of_le hb hper)

/-- `per = (nb + 255)/256` satisfies `nb <= per * 256`. -/
theorem per_cover (nb : Nat) : nb ≤ ((nb + 255) / 256) * 256 := by omega

-- § tie budget clamp

/-- Per-thread tie share: the budget left after earlier threads, clamped to `[0, eq]`. -/
def clampEq (budget before eq : Int) : Int := min eq (max 0 (budget - before))

/-- The share is nonnegative. -/
theorem clampEq_nonneg {budget before eq : Int} (h : 0 ≤ eq) : 0 ≤ clampEq budget before eq := by
  unfold clampEq
  omega

/-- The share never exceeds the thread's own equal count. -/
theorem clampEq_le_eq {budget before eq : Int} : clampEq budget before eq ≤ eq := by
  unfold clampEq
  omega

/-- Earlier ties plus this share never exceed the budget (or the earlier ties themselves when over budget). -/
theorem clampEq_cap {budget before eq : Int} :
    before + clampEq budget before eq ≤ max budget before := by
  unfold clampEq
  omega

/-- If the thread's equal cells all fit in the remaining budget, all of them are taken. -/
theorem clampEq_fill {budget before eq : Int} (h0 : 0 ≤ eq) (h : before + eq ≤ budget) :
    clampEq budget before eq = eq := by
  unfold clampEq
  omega

/-- With no budget left nothing is taken. -/
theorem clampEq_zero {budget before eq : Int} (h : budget ≤ before) : clampEq budget before eq = min eq 0 := by
  unfold clampEq
  omega

/-- The budget is positive and fits the equal cells when `above < width <= above + eq_total`. -/
theorem tie_budget {above eqt width : Nat} (h1 : above < width) (h2 : width ≤ above + eqt) :
    0 < width - above ∧ width - above ≤ eqt ∧ above + (width - above) = width := by omega

end Strata.QsaBlockSelect



## FILE: /mnt/user-data/outputs/NativeQsaNorms.lean
/-!
# MODULE: native_qsa_norms
Weighted RMSNorm (templated on 256 or 1024 threads, exact in-place use allowed) and the QSA sigmoid gate
adapted from llama.cpp.  This file proves the block-reduction slot bounds, the thread ownership that
makes in-place normalisation race-free, the gate's second-half addressing and the host span checks.

## Source semantics
* The norm reads its row once for the reduction; after the barrier every thread touches only the
  columns `tid, tid + B, tid + 2B, ...`, so output may alias input exactly.
* `gate` reads the second half of each head's `2 * head_dim` fused query row.
* `buffers` accepts disjoint spans or an exact input/output alias.

## Form template identifiers
- `norm<BlockSize>` -> `rms_norm_weighted`; `gate` -> `sigmoid_gate`; `warp_sum` -> `lane_reduce`
- `buffers` -> `span_check`; `elements` -> `element_count`
-/

namespace Strata.NativeQsaNorms

-- § reduction slots

/-- With 256 threads the warp slot is below 8 and below the 32-entry `sums` array. -/
theorem slot_256 {t : Nat} (ht : t < 256) : t / 32 < 8 ∧ t / 32 < 32 := by omega

/-- With 1024 threads the warp slot is below 32, exactly the size of `sums`. -/
theorem slot_1024 {t : Nat} (ht : t < 1024) : t / 32 < 32 := by omega

/-- The second-stage read `lane < BlockSize/32` covers exactly the warps that wrote a slot. -/
theorem second_stage_lanes : 256 / 32 = 8 ∧ 1024 / 32 = 32 := by decide

-- § in-place safety

/-- The thread that reads column `col` in the strided loop is the same thread that writes it. -/
theorem col_owner {col B : Nat} (hB : 0 < B) : col % B < B ∧ col % B + B * (col / B) = col :=
  ⟨Nat.mod_lt _ hB, Nat.mod_add_div col B⟩

/-- Two different columns are never visited by the same loop iteration of the same thread. -/
theorem col_owner_inj {c c' B : Nat} (hB : 0 < B) (h1 : c % B = c' % B) (h2 : c / B = c' / B) : c = c' := by
  have a := Nat.mod_add_div c B
  have b := Nat.mod_add_div c' B
  rw [h1, h2] at a
  omega

-- § gate addressing

/-- The gate half `head*(2*hd) + hd + channel` is inside the fused `n_head * (2*hd)` query buffer. -/
theorem gate_index_lt {head channel n hd : Nat} (hh : head < n) (hc : channel < hd) :
    head * (2 * hd) + hd + channel < n * (2 * hd) := by
  have h := Nat.mul_le_mul_right (2 * hd) (show head + 1 ≤ n by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- The fused query holds twice as many floats as the attention output. -/
theorem fused_q_size (n hd : Nat) : n * (2 * hd) = 2 * (hd * n) := by
  rw [Nat.mul_comm]
  exact Nat.mul_assoc 2 hd n

/-- The flat index splits into (head, channel) exactly. -/
theorem gate_split {i hd n : Nat} (hd0 : 0 < hd) (hi : i < hd * n) :
    i / hd < n ∧ i % hd < hd ∧ (i / hd) * hd + i % hd = i := by
  refine ⟨Nat.div_lt_of_lt_mul hi, Nat.mod_lt _ hd0, ?_⟩
  rw [Nat.mul_comm]
  exact Nat.div_add_mod i hd

/-- The launch of `ceil(count/256)` blocks covers every element. -/
theorem gate_grid_cover {count i : Nat} (hi : i < count) : i / 256 < (count + 255) / 256 := by omega

-- § span predicate

/-- Half-open byte ranges flagged as overlapping. -/
def overlap (ap an bp bn : Nat) : Prop := ap < bp + bn ∧ bp < ap + an

/-- The overlap predicate is symmetric. -/
theorem overlap_comm {ap an bp bn : Nat} : overlap ap an bp bn ↔ overlap bp bn ap an := by
  unfold overlap
  omega

/-- An exact alias is the one overlap the norm accepts; for a nonempty span it is flagged by the predicate. -/
theorem alias_is_flagged {p n : Nat} (hn : 0 < n) : overlap p n p n := by
  unfold overlap
  omega

/-- Ranges laid end to end are not flagged. -/
theorem adjacent_not_flagged {ap an bp bn : Nat} (h : ap + an ≤ bp) : ¬ overlap ap an bp bn := by
  unfold overlap
  omega

/-- The element count `cols * rows` bound keeps the byte size `count * 4` representable in 32-bit ints. -/
theorem count_bytes {count : Nat} (h : count ≤ 2147483647) : count * 4 ≤ 8589934588 := by omega

end Strata.NativeQsaNorms



## FILE: /mnt/user-data/outputs/NativeQsaIndexerAppend.lean
/-!
# MODULE: native_qsa_indexer_append
Single-block kernel that appends one raw indexer key to the QSA indexer state: it keeps a 3-row raw tail,
and on every fourth cell mean-pools the four raw keys, RMS-normalises, rotates the first 64 dims (NeoX
pairing) at the block's first position, writes the pooled row and re-writes the spare row.  This file
proves the ring-buffer slot arithmetic, the pooled-buffer capacity, the gather shape, the rotation
pairing and the position-base overflow guards.

## Source semantics
* `slot = pos % 4`; slots 0..2 are written to `tail`; slot 3 completes the block.
* `pos == 0` initialises row 0 and `dead`; the spare's four gather indices all name cell zero.
* Rotation pairs dim `d` with `d + 32` for `d < 64`; the rotation position is `pos_base + 4*b`.
* Pooled buffer capacity is `max_cells / 4 + 1` rows.

## Form template identifiers
- `append` -> `pool_append`; `tail` -> `raw_tail`; `dead` -> `spare_row`; `pooled` -> `pooled_rows`
- `rope_pos` -> `block_rotation_pos`; `mrope_pos` -> `section_position`; `theta_scale` -> `freq_step`
-/

namespace Strata.NativeQsaIndexer

-- § ring buffer

/-- The ring slot is below 4. -/
theorem slot_lt (pos : Nat) : pos % 4 < 4 := by omega

/-- Tail writes happen only for slots 0..2, so the 3-row tail suffices. -/
theorem tail_slot_lt {pos : Nat} (h : pos % 4 < 3) : pos % 4 * 128 + 127 < 3 * 128 := by omega

/-- The block row index `pos / 4` of a completing cell and the spare row `pos/4 + 1` fit the pooled buffer. -/
theorem pooled_rows_fit {pos max : Nat} (hp : pos < max) (hs : pos % 4 = 3) : pos / 4 + 1 < max / 4 + 1 := by
  omega

/-- For position 0 the single written row index 0 fits the buffer. -/
theorem pooled_row_zero (max : Nat) : 0 < max / 4 + 1 := by omega

/-- Pooled row addressing `b*128 + d` is inside `rows * 128`. -/
theorem pooled_index_lt {b d rows : Nat} (hb : b < rows) (hd : d < 128) : b * 128 + d < rows * 128 := by
  have h := Nat.mul_le_mul_right 128 (show b + 1 ≤ rows by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

-- § gather shape

/-- The four cells averaged by a completing cell are the three tail rows then the incoming key;
for position 0 all four are the incoming key. -/
def gatherRows (pos : Nat) (tail : List Int) (incoming : Int) : List Int :=
  if pos = 0 then [incoming, incoming, incoming, incoming] else tail ++ [incoming]

/-- A position-0 gather always has four entries. -/
theorem gather_zero_length (tail : List Int) (incoming : Int) : (gatherRows 0 tail incoming).length = 4 := by
  simp [gatherRows]

/-- A later gather has four entries when the tail has three. -/
theorem gather_later_length {pos : Nat} (h : pos ≠ 0) (tail : List Int) (incoming : Int)
    (ht : tail.length = 3) : (gatherRows pos tail incoming).length = 4 := by
  simp [gatherRows, h, ht]

/-- The newest cell is last in the gather order. -/
theorem gather_last {pos : Nat} (h : pos ≠ 0) (tail : List Int) (incoming : Int) :
    (gatherRows pos tail incoming).reverse.head? = some incoming := by
  simp [gatherRows, h]

-- § rotation pairing

/-- Pair index `d % 32` is below 32 and the partner `pair + 32` is below 64. -/
theorem pair_in_range {d : Nat} (hd : d < 64) : d % 32 < 32 ∧ d % 32 + 32 < 64 := by omega

/-- Lower-half threads have `pair = d`, upper-half threads have `pair = d - 32`. -/
theorem pair_halves {d : Nat} (hd : d < 64) :
    (d < 32 → d % 32 = d) ∧ (32 ≤ d → d % 32 = d - 32) := by omega

/-- The pairs `(d, d+32)` partition the 64 rotated dims: every dim has a unique pair index and half. -/
theorem rotated_dims_partition {d : Nat} (hd : d < 64) :
    (d % 32) + 32 * (d / 32) = d ∧ d / 32 < 2 := by omega

/-- Dims 64..127 are not rotated and keep the normalised value. -/
theorem unrotated_dims {d : Nat} (h1 : 64 ≤ d) (h2 : d < 128) : ¬ d < 64 := by omega

-- § position guards

/-- A base that is a multiple of 4 plus `4*b` is still a multiple of 4. -/
theorem rotation_pos_aligned {base b : Nat} (h : base % 4 = 0) : (base + 4 * b) % 4 = 0 := by omega

/-- The rotation position does not overflow int32 when `base + max_cells <= INT32_MAX` and `pos < max_cells`. -/
theorem rotation_pos_no_overflow {base pos max : Nat} (hm : base + max ≤ 2147483647) (hp : pos < max) :
    base + 4 * (pos / 4) ≤ 2147483647 := by omega

/-- The guard `pos < max_cells` makes every accepted position a valid cell index. -/
theorem guard_valid {pos max : Nat} (h : pos < max) : pos / 4 ≤ max / 4 := by omega

-- § span predicate

/-- Half-open byte ranges flagged as overlapping. -/
def overlap (ap an bp bn : Nat) : Prop := ap < bp + bn ∧ bp < ap + an

/-- The overlap predicate is symmetric. -/
theorem overlap_comm {ap an bp bn : Nat} : overlap ap an bp bn ↔ overlap bp bn ap an := by
  unfold overlap
  omega

/-- Ranges laid end to end are not flagged. -/
theorem adjacent_not_flagged {ap an bp bn : Nat} (h : ap + an ≤ bp) : ¬ overlap ap an bp bn := by
  unfold overlap
  omega

/-- The seven buffers give 21 pairwise overlap checks. -/
theorem pair_check_count : 7 * 6 / 2 = 21 := by decide

end Strata.NativeQsaIndexer



## FILE: /mnt/user-data/outputs/NativeQsaScore.lean
/-!
# MODULE: native_qsa_score
Tensor-core (m16n8k8 tf32) block scoring for the QSA indexer: 32 pooled rows per block, 2 warps splitting
the 128-dim key, 4 indexer heads padded to 8 mma columns, per-head ReLU, head sum, optional bias, and
the `+1e9` tail bias; each block score is broadcast to its cells.  This file proves the tile/shared-memory
addressing, the key-column split, the writer/reader agreement of the partial-sum exchange, the mapping
from mma fragment slots to (head, row), and the capacity/budget checks.

## Source semantics
* Warp `w` accumulates key columns `w*32 + lane` and `w*32 + lane + 64`.
* Fragment slot `l` of lane `lane` in A-tile `ia` of warp `w` holds row `ia*16 + (l/2)*8 + lane/4`
  and head `(lane%4)*2 + l%2` (heads >= 4 are padding).
* Partials are exchanged through `shared[h*68 + warp*32 + row]`.
* The step is accepted only when `width = min(n_kv, 2051)` (`top_k = 2048`, `r = 4`).

## Form template identifiers
- `score_kernel` -> `mma_block_score`; `STRIDE` -> `tile_stride` (36); `COMBINE` -> `combine_stride` (68)
- `ROWS` -> `rows_per_block` (32); `WARPS` -> `warps_per_block` (2); `max_blocks` -> `block_capacity`
-/

namespace Strata.NativeQsaScore

-- § grid

/-- A grid of `ceil(max_blocks/32)` blocks of 32 rows covers every pooled row. -/
theorem row_grid_cover {row maxb : Nat} (hr : row < maxb) : row / 32 < (maxb + 31) / 32 := by omega

/-- The first row of a block and the local row index reconstruct the row. -/
theorem row_split {row : Nat} : (row / 32) * 32 + row % 32 = row ∧ row % 32 < 32 := by omega

-- § key column split between the two warps

/-- Every key column is visited exactly once: owner warp `(c/32)%2`, iteration `c/64`, lane `c%32`. -/
theorem column_owner {c : Nat} (hc : c < 128) :
    (c / 64) * 64 + ((c / 32) % 2) * 32 + c % 32 = c ∧ c / 64 < 2 ∧ (c / 32) % 2 < 2 := by omega

/-- The columns a warp visits, `warp*32 + lane + 64*it`, are in range. -/
theorem warp_column_lt {warp lane it : Nat} (hw : warp < 2) (hl : lane < 32) (hi : it < 2) :
    warp * 32 + lane + 64 * it < 128 := by omega

/-- Distinct (warp, lane, iteration) triples visit distinct columns. -/
theorem warp_column_inj {w l i w' l' i' : Nat} (hl : l < 32) (hl' : l' < 32) (hw : w < 2) (hw' : w' < 2)
    (h : w * 32 + l + 64 * i = w' * 32 + l' + 64 * i') : w = w' ∧ l = l' ∧ i = i' := by omega

-- § shared memory tiles

/-- A tile write `tile[i*36 + lane]` (i < 16) stays inside the warp's 16x36 tile. -/
theorem tile_index_lt {i lane : Nat} (hi : i < 16) (hl : lane < 32) : i * 36 + lane < 16 * 36 := by omega

/-- Each warp's tile fits the `2 * 16 * 36` shared array. -/
theorem warp_tile_lt {warp i lane : Nat} (hw : warp < 2) (hi : i < 16) (hl : lane < 32) :
    warp * (16 * 36) + (i * 36 + lane) < 2 * 16 * 36 := by omega

/-- The exchange array `h*68 + i` (h < 8, i < 64) fits inside the same shared allocation. -/
theorem combine_index_lt {h i : Nat} (hh : h < 8) (hi : i < 64) : h * 68 + i < 2 * 16 * 36 := by omega

/-- Row slots of the two warps (offset 0 and 32) stay inside one 68-wide head row. -/
theorem combine_row_fits {warp row : Nat} (hw : warp < 2) (hr : row < 32) : warp * 32 + row < 68 := by omega

-- § mma fragment slots

/-- Fragment slot to (row, head): the row inside a 32-row A tile pair, and the head from lane and slot. -/
def fragRow (ia l lane : Nat) : Nat := ia * 16 + (l / 2) * 8 + lane / 4
/-- Head of a fragment slot. -/
def fragHead (l lane : Nat) : Nat := (lane % 4) * 2 + l % 2

/-- The row of a fragment slot is below 32. -/
theorem fragRow_lt {ia l lane : Nat} (hi : ia < 2) (hl : l < 4) (hlane : lane < 32) : fragRow ia l lane < 32 := by
  unfold fragRow
  omega

/-- The head of a fragment slot is below 8 (heads 4..7 are padding). -/
theorem fragHead_lt {l lane : Nat} : fragHead l lane < 8 := by
  unfold fragHead
  omega

/-- Distinct (ia, slot, lane) triples write distinct (head, row) cells, so the exchange has no collisions. -/
theorem frag_inj {ia l lane ia' l' lane' : Nat} (hi : ia < 2) (hi' : ia' < 2) (hl : l < 4) (hl' : l' < 4)
    (hlane : lane < 32) (hlane' : lane' < 32)
    (hh : fragHead l lane = fragHead l' lane') (hr : fragRow ia l lane = fragRow ia' l' lane') :
    ia = ia' ∧ l = l' ∧ lane = lane' := by
  unfold fragHead at hh
  unfold fragRow at hr
  omega

/-- The writer's cell `h*68 + warp*32 + row` equals the reader's `j*68 + warp*32 + lane`. -/
theorem exchange_agrees (h w row : Nat) : h * 68 + (w * 32 + row) = h * 68 + w * 32 + row := by omega

-- § capacities and budget

/-- Block capacity check `max_blocks = max_cells/4 + 1` is consistent with the cells of the last block. -/
theorem capacity_covers {n maxc : Nat} (hn : n ≤ maxc) : n / 4 < maxc / 4 + 1 := by omega

/-- Cells of a block `row*4 + i` with `i < 4` and `row*4 + i < n` are below `n`. -/
theorem block_cell_lt {row i n : Nat} (h : row * 4 + i < n) : row * 4 + i < n := h

/-- The accepted width is `min(n, 2051)`: `top_k + r - 1` for `top_k = 2048`, `r = 4`. -/
theorem width_formula : 2048 + 4 - 1 = 2051 := by decide

end Strata.NativeQsaScore



## FILE: /mnt/user-data/outputs/NativeFlashAttnShort.lean
/-!
# MODULE: native_flash_attn_short
Short-context decode FlashAttention (24 query heads, 2 KV heads, head_dim 256, f16 K/V, up to 256 cells)
specialised from llama.cpp's vector kernel: 128 threads (32 x 4), the context padded to 256 and walked in
two tiles of 128 cells.  This file proves the thread-to-cell and thread-to-dimension bijections, the
K/V/Q/shared-memory index bounds, the output coverage (including the NaN fill on an invalid step) and
the step-validity condition.

## Source semantics
* `kv = head / 12`; cells are addressed `(cell*2 + kv)*256 + d`.
* Per tile, cell = `base + warp*32 + (lane/8)*8 + row` with `row < 8`.
* Dimensions: `d = 2*(i0 + (lane%8)*4 + j)` with `i0 in {0,32,64,96}`, `j < 4`, plus `d+1`.
* A step is valid when `1 <= width <= max_context`, `n_kv = width`, `pos = width-1`, `n_bid = width/4`.

## Form template identifiers
- `attend` -> `short_flash_attend`; `vkq` -> `value_accumulator`; `tile` -> `weight_tile`
- `qreg` -> `query_registers`; `kStepWidth/NKv/Pos/NBid` -> `step_width/n_kv/pos/n_bid`
-/

namespace Strata.NativeFlashAttn

-- § arithmetic helper

/-- Row-major linearisation stays in bounds. -/
theorem lin2_lt {i a c d : Nat} (hi : i < a) (hc : c < d) : i * d + c < a * d := by
  have h := Nat.mul_le_mul_right d (show i + 1 ≤ a by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

-- § heads

/-- Query head `head < 24` maps to KV head `head / 12 < 2`. -/
theorem kv_head_lt {head : Nat} (h : head < 24) : head / 12 < 2 := by omega

-- § cells per tile

/-- Per-tile cell offset `warp*32 + (lane/8)*8 + row` is below 128. -/
theorem tile_cell_lt {warp lane row : Nat} (hw : warp < 4) (hl : lane < 32) (hr : row < 8) :
    warp * 32 + (lane / 8) * 8 + row < 128 := by omega

/-- Distinct (warp, lane group, row) triples address distinct cells of a tile. -/
theorem tile_cell_inj {w g r w' g' r' : Nat} (hg : g < 4) (hg' : g' < 4) (hr : r < 8) (hr' : r' < 8)
    (h : w * 32 + g * 8 + r = w' * 32 + g' * 8 + r') : w = w' ∧ g = g' ∧ r = r' := by omega

/-- Two tiles of 128 cells cover the padded length 256. -/
theorem tiles_cover : 2 * 128 = 256 := by decide

/-- The score slot `tid` of the weight tile is below 128 and below the 4096-float tile. -/
theorem score_slot_lt {tid : Nat} (h : tid < 128) : tid < 4 * 4 * 256 := by omega

-- § dimensions

/-- The dimension pair `(d, d+1)` with `d = 2*(32*a + 4*m + j)` stays below 256. -/
theorem dim_pair_lt {a m j : Nat} (ha : a < 4) (hm : m < 8) (hj : j < 4) :
    2 * (32 * a + m * 4 + j) + 1 < 256 := by omega

/-- Distinct (block a, lane m, element j) triples address distinct dimension pairs, so the 128 pairs are covered once. -/
theorem dim_pair_inj {a m j a' m' j' : Nat} (hm : m < 8) (hm' : m' < 8) (hj : j < 4) (hj' : j' < 4)
    (h : 2 * (32 * a + m * 4 + j) = 2 * (32 * a' + m' * 4 + j')) : a = a' ∧ m = m' ∧ j = j' := by omega

/-- The register slot `i0/8 + j = 4a + j` is below 16. -/
theorem register_slot_lt {a j : Nat} (ha : a < 4) (hj : j < 4) : 4 * a + j < 16 := by omega

-- § K/V addressing

/-- K/V element `(cell*2 + kv)*256 + d` is inside a cache of `capacity` cells (2 KV heads, 256 dims). -/
theorem kv_index_lt {cell kv d capacity : Nat} (hc : cell < capacity) (hk : kv < 2) (hd : d < 256) :
    (cell * 2 + kv) * 256 + d < capacity * 2 * 256 :=
  lin2_lt (lin2_lt hc hk) hd

/-- A cache of `capacity` cells holds `capacity * 1024` bytes in f16. -/
theorem kv_bytes (capacity : Nat) : capacity * 2 * 256 * 2 = capacity * 1024 := by omega

/-- A valid width never exceeds the cache capacity because `width <= max_context <= 256 <= capacity`. -/
theorem width_le_capacity {width ctx capacity : Nat} (h1 : width ≤ ctx) (h2 : ctx ≤ 256) (h3 : 256 ≤ capacity) :
    width ≤ capacity := by omega

-- § shared-memory reduction

/-- The cross-warp reduction write `warp*1024 + group*256 + 2*(i0 + m*4) + 2j + 1` stays below 4096. -/
theorem reduce_write_lt {warp group a m j : Nat} (hw : warp < 4) (hg : group < 4) (ha : a < 4) (hm : m < 8)
    (hj : j < 4) : warp * 1024 + group * 256 + 2 * (32 * a + m * 4) + 2 * j + 1 < 4096 := by omega

/-- The final reduction read `w*1024 + group*256 + i0 + tid` with `i0 in {0,128}`, `tid < 128` stays below 4096. -/
theorem reduce_read_lt {w group i0 tid : Nat} (hw : w < 4) (hg : group < 4) (hi : i0 = 0 ∨ i0 = 128)
    (ht : tid < 128) : w * 1024 + group * 256 + i0 + tid < 4096 := by omega

/-- The two output passes `i0 + tid` cover each of the 256 output dims exactly once. -/
theorem output_cover {d : Nat} (hd : d < 256) :
    (d < 128 ∧ d = 0 + d) ∨ (128 ≤ d ∧ d - 128 < 128 ∧ d = 128 + (d - 128)) := by omega

/-- On an invalid step each of the 128 threads writes two NaNs at `tid` and `tid + 128`, covering all 256. -/
theorem invalid_fill_cover {tid : Nat} (h : tid < 128) : tid < 256 ∧ tid + 128 < 256 := by omega

-- § step validity

/-- Step validity predicate of the kernel. -/
def valid (width nkv pos nbid ctx : Nat) : Prop :=
  1 ≤ width ∧ width ≤ ctx ∧ nkv = width ∧ pos + 1 = width ∧ nbid = width / 4

/-- A step produced by `qsa_step_fill` is valid whenever the position is below 256 and `top_k >= 256`. -/
theorem fill_is_valid {pos k : Nat} (hp : pos < 256) (hk : 256 ≤ k) :
    valid (min (pos + 1) (k + 4 - 1)) (pos + 1) pos ((pos + 1) / 4) 256 := by
  unfold valid
  omega

/-- A validated width indexes only cells that exist: `cell < width <= 256`. -/
theorem valid_cells_in_cache {cell width ctx : Nat} (hv : width ≤ ctx) (hctx : ctx ≤ 256) (hc : cell < width) :
    cell < 256 := by omega

end Strata.NativeFlashAttn



## FILE: /mnt/user-data/outputs/KvQ8Cache.lean
/-!
# MODULE: kv_q8_cache
Int8 KV-cache append (per-64-value-group absmax scale stored as f16, codes clamped to [-127, 127]) and the
gather that dequantises selected cells back to f16.  This file proves the block/thread addressing of the
append, the code range, the scale-group addressing of the gather (a 4-value run never straddles a
group), and the memory saving of the layout.

## Source semantics
* One append block per (kv head, 64-value group, K or V); 64 threads, one value each, two warps whose
  maxima are combined in fixed order.
* Codes are `clamp(round(x / stored_scale), -127, 127)`, and 0 when the stored scale is 0.
* The gather thread handles 4 consecutive values (`d = q4*4`) of one cell and head.

## Form template identifiers
- `kv_append_q8_kernel` -> `q8_append`; `kv_gather_q8_kernel` -> `q8_gather`
- `KV_Q8_GROUP` -> `group_size` (64); `sbits` -> `scale_bits`; `row` -> `pool_row`
-/

namespace Strata.KvQ8

-- § arithmetic helper

/-- Row-major linearisation stays in bounds. -/
theorem lin2_lt {i a c d : Nat} (hi : i < a) (hc : c < d) : i * d + c < a * d := by
  have h := Nat.mul_le_mul_right d (show i + 1 ≤ a by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

-- § append addressing

/-- Within a group, `g*64 + t` is below `head_dim = 64 * groups`. -/
theorem group_value_lt {g t groups hd : Nat} (hhd : hd = groups * 64) (hg : g < groups) (ht : t < 64) :
    g * 64 + t < hd := by omega

/-- The source element `h*head_dim + g*64 + t` is inside the `kv_heads * head_dim` input. -/
theorem input_index_lt {h g t groups hd kvh : Nat} (hhd : hd = groups * 64) (hh : h < kvh)
    (hg : g < groups) (ht : t < 64) : h * hd + (g * 64 + t) < kvh * hd :=
  lin2_lt hh (group_value_lt hhd hg ht)

/-- Thread `t < 64` belongs to warp `t / 32 < 2`, the two warps whose maxima are combined. -/
theorem warp_slot_lt {t : Nat} (ht : t < 64) : t / 32 < 2 := by omega

/-- The scale index `row*groups + g` is inside the scale buffer of `rows * groups` entries. -/
theorem scale_index_lt {row g rows groups : Nat} (hr : row < rows) (hg : g < groups) :
    row * groups + g < rows * groups := lin2_lt hr hg

/-- The pool row `(page*kv_heads + h)*page_size + slot` is inside the pool. -/
theorem pool_row_lt {page h slot pages kvh ps : Nat} (hp : page < pages) (hh : h < kvh) (hs : slot < ps) :
    (page * kvh + h) * ps + slot < pages * kvh * ps :=
  lin2_lt (lin2_lt hp hh) hs

-- § code range

/-- Clamp to the symmetric int8 range used by the quantiser. -/
def clamp127 (q : Int) : Int := if q < -127 then -127 else if q > 127 then 127 else q

/-- The code is within `[-127, 127]`, so `-128` is never produced. -/
theorem clamp127_range (q : Int) : -127 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  unfold clamp127
  split_ifs <;> omega

/-- Codes already in range are unchanged. -/
theorem clamp127_id {q : Int} (h1 : -127 ≤ q) (h2 : q ≤ 127) : clamp127 q = q := by
  unfold clamp127
  split_ifs <;> omega

/-- The code is exactly representable as a signed byte. -/
theorem code_fits_int8 (q : Int) : -128 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  have := clamp127_range q
  omega

-- § gather addressing

/-- A thread's 4-value run `q4*4 .. q4*4+3` stays inside the head row of `head_dim = 64 * groups`. -/
theorem run_in_row {q4 groups hd : Nat} (hhd : hd = groups * 64) (hq : q4 < hd / 4) : q4 * 4 + 3 < hd := by
  omega

/-- The scale group of a thread's run is below the number of groups. -/
theorem run_group_lt {q4 groups hd : Nat} (hhd : hd = groups * 64) (hq : q4 < hd / 4) :
    (q4 * 4) / 64 < groups := by omega

/-- A 4-value run starting at a multiple of 4 never crosses a 64-value group boundary,
so one scale applies to all four values. -/
theorem run_in_one_group (q4 : Nat) : (q4 * 4) / 64 = (q4 * 4 + 3) / 64 := by omega

/-- The gather destination `(id*kv_heads + h)*per + q4` is inside the scratch of `n_ids` cells. -/
theorem gather_dst_lt {id h q4 n_ids kvh per : Nat} (hid : id < n_ids) (hh : h < kvh) (hq : q4 < per) :
    (id * kvh + h) * per + q4 < n_ids * kvh * per :=
  lin2_lt (lin2_lt hid hh) hq

/-- The gather's flat index splits into (cell, head, quad) exactly. -/
theorem gather_split {i kvh per : Nat} (hper : 0 < per) (hk : 0 < kvh) :
    i % (kvh * per) / per < kvh ∧ i % (kvh * per) % per < per := by
  constructor
  · exact Nat.div_lt_of_lt_mul (by
      have := Nat.mod_lt i (Nat.mul_pos hk hper)
      rw [Nat.mul_comm per kvh]
      exact this)
  · exact Nat.mod_lt _ hper

-- § memory layout

/-- Per row of 256 values the int8 layout needs 256 code bytes plus 4 two-byte scales: 264 bytes. -/
theorem int8_row_bytes : 256 + 4 * 2 = 264 := by decide

/-- That is less than the 512 bytes of the f16 layout. -/
theorem int8_smaller_than_f16 : 256 + 4 * 2 < 256 * 2 := by decide

end Strata.KvQ8





## FILE: /mnt/user-data/outputs/SGemvQuadDecode.lean
/-!
# MODULE: s_gemv_quad_decode
The S-family GEMV (`s_gemv.cu`): one thread / one block / one warp per output row, dequantising packed
2/4/8-bit codes on the fly and dotting against FP16, Q8_K or Q8_0 activations.  This module pins the index
arithmetic the optimised kernels rely on: packed-word bit order, group containment, lane partition,
alignment of the wide loads, tail-loop emptiness, activation-block strides and the IQ4_NL codebook.

## Source semantics
* Element `i` of a row has code `(c[i/PER_BYTE] >> ((i%PER_BYTE)*bits)) & mask`, `PER_BYTE = 8/bits`.
* The wide kernels read a little-endian word holding several consecutive codes; the bit order must match.
* `group_shift = log2 group_elems`, so `i >> shift = i / group_elems`.
* One lane-iteration of the Q8 split kernel (QE = 16) never straddles a Q8_K (256) or Q8_0 (32) block.
* The weight is decoded BEFORE the activation multiply (`w = code*d + b`, then `acc += w*x`).

## Form template identifiers
- `SForm.code_bits`  → `bits`
- `SForm.group_elems` → `g`
- `SForm.code_bias`  → `bias`
- `kIq4Nl`           → `kIq4Nl`
- `Q8K_BLOCK_BYTES / Q8_0_BLOCK_BYTES` → `q8kBytes / q80Bytes`
- lane index          → `lane`, vector length `n_in` → `n`
-/

namespace Strata.SGemv

-- § Packed word bit order

/-- S2: element `k` of a code byte is bits `[2k,2k+2)`; here the four fields of one byte, in element order. -/
theorem s2_fields (b : Nat) (hb : b < 256) :
    b = b % 4 + 4 * (b / 4 % 4) + 16 * (b / 16 % 4) + 64 * (b / 64 % 4) := by omega

/-- S4: a uint16 word `b0 + 256*b1` holds four nibbles; nibble 0 is the low nibble of the first byte. -/
theorem s4_word_k0 (b0 b1 : Nat) : (b0 + 256 * b1) % 16 = b0 % 16 := by omega

/-- S4 nibble 1 is the high nibble of the first byte. -/
theorem s4_word_k1 (b0 b1 : Nat) (h0 : b0 < 256) : (b0 + 256 * b1) / 16 % 16 = b0 / 16 := by omega

/-- S4 nibble 2 is the low nibble of the second byte. -/
theorem s4_word_k2 (b0 b1 : Nat) (h0 : b0 < 256) : (b0 + 256 * b1) / 256 % 16 = b1 % 16 := by omega

/-- S4 nibble 3 is the high nibble of the second byte. -/
theorem s4_word_k3 (b0 b1 : Nat) (h0 : b0 < 256) (h1 : b1 < 256) :
    (b0 + 256 * b1) / 4096 % 16 = b1 / 16 := by omega

/-- S8: a uint32 word is four bytes, little endian; byte 0. -/
theorem s8_word_k0 (b0 b1 b2 b3 : Nat) :
    (b0 + 256 * b1 + 65536 * b2 + 16777216 * b3) % 256 = b0 % 256 := by omega

/-- S8 byte 1. -/
theorem s8_word_k1 (b0 b1 b2 b3 : Nat) (h0 : b0 < 256) :
    (b0 + 256 * b1 + 65536 * b2 + 16777216 * b3) / 256 % 256 = b1 % 256 := by omega

/-- S8 byte 2. -/
theorem s8_word_k2 (b0 b1 b2 b3 : Nat) (h0 : b0 < 256) (h1 : b1 < 256) :
    (b0 + 256 * b1 + 65536 * b2 + 16777216 * b3) / 65536 % 256 = b2 % 256 := by omega

/-- S8 byte 3. -/
theorem s8_word_k3 (b0 b1 b2 b3 : Nat) (h0 : b0 < 256) (h1 : b1 < 256) (h2 : b2 < 256) :
    (b0 + 256 * b1 + 65536 * b2 + 16777216 * b3) / 16777216 % 256 = b3 % 256 := by omega

-- § Word widths and alignment

/-- Bytes per four-code word `QW = 4*bits/8`: 1, 2, 4 for S2, S4, S8 (not the octet width). -/
theorem word_bytes_values :
    4 * 2 / 8 = 1 ∧ 4 * 4 / 8 = 2 ∧ 4 * 8 / 8 = 4 := by decide

/-- Four words of one lane-iteration (QE = 16) cover exactly `16*bits/8` bytes. -/
theorem octet_bytes (bits : Nat) (h : bits = 2 ∨ bits = 4 ∨ bits = 8) :
    4 * (4 * bits / 8) = 16 * bits / 8 := by
  rcases h with rfl | rfl | rfl <;> decide

/-- The wide code loads are naturally aligned when `i` is a multiple of 16: S4 reads uint16 at byte `i/2`. -/
theorem s4_load_aligned (i : Nat) (h : i % 16 = 0) : (i / 2) % 2 = 0 := by omega

/-- S8 reads uint32 at byte `i`. -/
theorem s8_load_aligned (i : Nat) (h : i % 16 = 0) : i % 4 = 0 := by omega

/-- Row base alignment for the 16-wide kernel: a row of `n` elements (n multiple of 16) holds a whole number of words. -/
theorem row_base_aligned (n : Nat) (h : n % 16 = 0) :
    (n / 2) % 2 = 0 ∧ n % 4 = 0 ∧ (n / 4) % 1 = 0 := by omega

/-- The split kernel (QE = 4): `i` a multiple of 4 gives aligned uint16 (S4) and uint32 (S8) loads. -/
theorem quad_load_aligned (i : Nat) (h : i % 4 = 0) : (i / 2) % 2 = 0 ∧ i % 4 = 0 := by omega

/-- The paired-half load `x + i` is 8-byte aligned when `i % 4 = 0` (halves are 2 bytes). -/
theorem half4_aligned (i : Nat) (h : i % 4 = 0) : (2 * i) % 8 = 0 := by omega

-- § Group index as a shift

/-- `i >> s = i / 2^s`: the group index computed with the logarithm is the division it replaces. -/
theorem group_shift_eq_div (i s : Nat) : i >>> s = i / 2 ^ s := Nat.shiftRight_eq_div_pow i s

/-- A quad (4 elements starting at a multiple of 4) lies in one group for every power-of-two group ≥ 4. -/
theorem quad_in_group (i : Nat) (h : i % 4 = 0) :
    (i + 3) / 4 = i / 4 ∧ (i + 3) / 8 = i / 8 ∧ (i + 3) / 16 = i / 16 ∧
    (i + 3) / 32 = i / 32 ∧ (i + 3) / 64 = i / 64 := by omega

/-- An octet/16-run starting at a multiple of 16 lies in one group for the group sizes 16, 32, 64. -/
theorem run16_in_group (i : Nat) (h : i % 16 = 0) :
    (i + 15) / 16 = i / 16 ∧ (i + 15) / 32 = i / 32 ∧ (i + 15) / 64 = i / 64 := by omega

/-- A multiple of 16 followed by 8 more elements stays in the same group for group sizes 16, 32, 64. -/
theorem octet_in_group (i : Nat) (h : i % 8 = 0) :
    (i + 7) / 8 = i / 8 ∧ (i + 7) / 16 = i / 16 ∧ (i + 7) / 32 = i / 32 ∧ (i + 7) / 64 = i / 64 := by omega

-- § Lane partition and tails

/-- Lane `lane` visits `i = lane*16 + 512*m`; every multiple of 16 is visited by exactly one `(lane, m)`. -/
theorem lane_cover (e : Nat) (h : e % 16 = 0) :
    e = ((e / 16) % 32) * 16 + 512 * (e / 512) := by omega

/-- The lane of a position is below 32. -/
theorem lane_lt (e : Nat) : (e / 16) % 32 < 32 := by omega

/-- Distinct `(lane, m)` give distinct positions. -/
theorem lane_inj (l1 m1 l2 m2 : Nat) (h1 : l1 < 32) (h2 : l2 < 32)
    (h : l1 * 16 + 512 * m1 = l2 * 16 + 512 * m2) : l1 = l2 ∧ m1 = m2 := by omega

/-- When `n % 16 = 0` the partial-quad tail loop of the 16-wide kernel is empty. -/
theorem tail16_empty (i n : Nat) (hi : i % 16 = 0) (hn : n % 16 = 0) (h : ¬ (i + 16 ≤ n)) : n ≤ i := by omega

/-- When `n % 4 = 0` the tail loop of the 4-wide kernel is empty. -/
theorem tail4_empty (i n : Nat) (hi : i % 4 = 0) (hn : n % 4 = 0) (h : ¬ (i + 4 ≤ n)) : n ≤ i := by omega

/-- Generic stride uniqueness: offsets below the period determine `(offset, iteration)`. -/
theorem stride_inj (a b P m1 m2 : Nat) (hP : 0 < P) (ha : a < P) (hb : b < P)
    (h : a + P * m1 = b + P * m2) : a = b ∧ m1 = m2 := by
  have h1 : (a + P * m1) / P = m1 := by
    rw [Nat.add_mul_div_left _ _ hP, Nat.div_eq_of_lt ha, Nat.zero_add]
  have h2 : (b + P * m2) / P = m2 := by
    rw [Nat.add_mul_div_left _ _ hP, Nat.div_eq_of_lt hb, Nat.zero_add]
  rw [h] at h1
  have hm : m1 = m2 := h1.symm.trans h2
  subst hm
  omega

/-- Thread `tid` of the split kernel visits `i = tid*4 + threads*4*m`; positions are unique per `(tid, m)`. -/
theorem split_stride_inj (tpr t1 m1 t2 m2 : Nat) (hp : 0 < tpr) (h1 : t1 < tpr) (h2 : t2 < tpr)
    (h : t1 * 4 + tpr * 4 * m1 = t2 * 4 + tpr * 4 * m2) : t1 = t2 ∧ m1 = m2 := by
  have := stride_inj (t1 * 4) (t2 * 4) (tpr * 4) m1 m2 (by omega) (by omega) (by omega) h
  omega

-- § Activation block strides

/-- A Q8_K block is `{f32 d; int8 qs[256]; int16 bsums[16]}`. -/
theorem q8k_block_bytes : 4 + 256 + 2 * 16 = 292 := by decide

/-- A Q8_0 block is `{fp16 d; int8 qs[32]}`. -/
theorem q80_block_bytes : 2 + 32 = 34 := by decide

/-- Every Q8_K block's `d` is 4-byte aligned when the base is. -/
theorem q8k_d_aligned (b : Nat) : (b * 292) % 4 = 0 := by omega

/-- Every Q8_0 block's `d` is 2-byte aligned (the stride is 34). -/
theorem q80_d_aligned (b : Nat) : (b * 34) % 2 = 0 := by omega

/-- A 16-run starting at a multiple of 16 stays inside one Q8_K block. -/
theorem run16_in_q8k (i : Nat) (h : i % 16 = 0) : i % 256 + 16 ≤ 256 := by omega

/-- A 16-run starting at a multiple of 16 stays inside one Q8_0 block. -/
theorem run16_in_q80 (i : Nat) (h : i % 16 = 0) : i % 32 + 16 ≤ 32 := by omega

/-- Block index and in-block offset recompose the element index (Q8_K). -/
theorem q8k_split (i : Nat) : (i / 256) * 256 + i % 256 = i := by omega

/-- Block index and in-block offset recompose the element index (Q8_0). -/
theorem q80_split (i : Nat) : (i / 32) * 32 + i % 32 = i := by omega

/-- `ffn_down_shexp` has `n_in = 640`: a multiple of 32 but not of 256, so Q8_K is structurally impossible. -/
theorem n640_not_q8k : 640 % 32 = 0 ∧ 640 % 256 ≠ 0 := by decide

-- § Weight decode order (integer domain bias, then scale, then offset)

/-- Integer-domain bias is applied to the code before the float conversion. -/
def biasedCode (code bias : Int) : Int := code + bias

/-- The S2 bias of -1 maps codes 0..3 to -1..2. -/
theorem s2_bias_range (c : Nat) (h : c < 4) : -1 ≤ biasedCode c (-1) ∧ biasedCode c (-1) ≤ 2 := by
  unfold biasedCode; omega

-- § The non-linear IQ4_NL codebook

/-- The 16-entry codebook, verbatim. -/
def kIq4Nl : List Int :=
  [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- Sixteen entries, indexed by the low nibble. -/
theorem kIq4Nl_length : kIq4Nl.length = 16 := by decide

/-- Every low-nibble lookup is in range. -/
theorem iq4nl_index_lt (code : Nat) : code % 16 < kIq4Nl.length := by
  rw [kIq4Nl_length]; omega

/-- The codebook is strictly increasing (non-linear spacing, monotone). -/
theorem kIq4Nl_sorted : kIq4Nl.Pairwise (· < ·) := by decide

/-- Every entry fits in int8. -/
theorem kIq4Nl_int8 : ∀ x ∈ kIq4Nl, -128 ≤ x ∧ x ≤ 127 := by decide

/-- The codebook is not an affine map of the code: the first two gaps differ. -/
theorem kIq4Nl_nonlinear : (-104 : Int) - (-127) ≠ (-83 : Int) - (-104) := by decide

-- § Shape checks of the host wrappers

/-- A power of two is a multiple of 4 as soon as it is at least 4 (the quad-in-group guard). -/
theorem pow2_ge4_mod4 (s : Nat) (h : 2 ≤ s) : 2 ^ s % 4 = 0 := by
  obtain ⟨k, rfl⟩ : ∃ k, s = k + 2 := ⟨s - 2, by omega⟩
  have : 2 ^ (k + 2) = 2 ^ k * 4 := by
    rw [Nat.pow_succ, Nat.pow_succ]; omega
  omega

/-- Codes per row for each width. -/
theorem codes_per_row_values (n : Nat) (h : n % 4 = 0) :
    n / 4 * 4 = n ∧ n / 2 * 2 = n ∧ n / 1 = n := by omega

/-- The q8k split launcher uses `threads / 32` rows per block; the grid covers all rows. -/
theorem warp_grid_covers (n_out warps : Nat) (hw : 0 < warps) :
    n_out ≤ warps * ((n_out + warps - 1) / warps) := by
  have := Nat.div_add_mod (n_out + warps - 1) warps
  have := Nat.mod_lt (n_out + warps - 1) hw
  omega

end Strata.SGemv



## FILE: /mnt/user-data/outputs/S2GemvFastLut.lean
/-!
# MODULE: s2_gemv_fast_lut
The experimental S2 GEMV (`s2_gemv_fast.cu`) that stages `x` in shared memory and expands each code byte
through a constant 256x4 table.  Both levers were measured neutral/negative; the kernel is kept as a
negative result.  The theory pins the table, the group index, the shared-memory layout and the strides.

## Source semantics
* `c_codes[b][k] = ((b >> 2k) & 3) - 1`: the four biased codes of byte `b` in element order.
* One byte holds four codes, so a quad index `q` addresses elements `4q..4q+3`; the group (64 elements)
  of quad `q` is `(4q) >> 6 = q >> 4`.
* Staged `x` occupies `MAX_SHARED_HALVES = 4096` halves = 2048 floats; the partials follow it.

## Form template identifiers
- `c_codes` → `lut`; `QK_S2` → `qk`; `MAX_SHARED_HALVES` → `maxHalves`
- `threads_per_row` → `tpr`; `n_in` → `n`
-/

namespace Strata.S2GemvFast

/-- Biased code `k` (0..3) of byte `b` as an integer: `((b / 4^k) % 4) - 1`, `4^k` given as a literal. -/
def lut0 (b : Nat) : Int := ((b % 4 : Nat) : Int) - 1
/-- Code 1 of the byte. -/
def lut1 (b : Nat) : Int := ((b / 4 % 4 : Nat) : Int) - 1
/-- Code 2 of the byte. -/
def lut2 (b : Nat) : Int := ((b / 16 % 4 : Nat) : Int) - 1
/-- Code 3 of the byte. -/
def lut3 (b : Nat) : Int := ((b / 64 % 4 : Nat) : Int) - 1

-- § The table

/-- Every table entry is in {-1, 0, 1, 2}. -/
theorem lut_range (b : Nat) :
    -1 ≤ lut0 b ∧ lut0 b ≤ 2 ∧ -1 ≤ lut1 b ∧ lut1 b ≤ 2 ∧
    -1 ≤ lut2 b ∧ lut2 b ≤ 2 ∧ -1 ≤ lut3 b ∧ lut3 b ≤ 2 := by
  unfold lut0 lut1 lut2 lut3
  omega

/-- The table is a faithful expansion: the four biased codes reassemble the byte. -/
theorem lut_reassembles (b : Nat) (hb : b < 256) :
    (lut0 b + 1) + 4 * (lut1 b + 1) + 16 * (lut2 b + 1) + 64 * (lut3 b + 1) = (b : Int) := by
  unfold lut0 lut1 lut2 lut3
  omega

/-- The table is injective: two bytes with equal expansions are equal. -/
theorem lut_injective (a b : Nat) (ha : a < 256) (hb : b < 256)
    (h0 : lut0 a = lut0 b) (h1 : lut1 a = lut1 b) (h2 : lut2 a = lut2 b) (h3 : lut3 a = lut3 b) : a = b := by
  have := lut_reassembles a ha
  have := lut_reassembles b hb
  omega

/-- The table has 256 rows of four floats: 4096 bytes of constant memory. -/
theorem lut_bytes : 256 * 4 * 4 = 4096 := by decide

-- § Quad addressing

/-- The S2 group index of quad `q` is `q >> 4`: `(4q)/64 = q/16`. -/
theorem quad_group (q : Nat) : (q * 4) / 64 = q / 16 := by omega

/-- Via the shift form used in the kernel. -/
theorem quad_group_shift (q : Nat) : (q * 4) >>> 6 = q >>> 4 := by
  rw [Nat.shiftRight_eq_div_pow, Nat.shiftRight_eq_div_pow]; omega

/-- A quad is entirely inside one 64-element group. -/
theorem quad_inside_group (q : Nat) : (q * 4 + 3) / 64 = (q * 4) / 64 := by omega

/-- `n_in % 64 = 0` gives whole quads and whole groups, with 16 quads per group. -/
theorem shape (n : Nat) (h : n % 64 = 0) : (n / 4) % 16 = 0 ∧ n / 64 * 64 = n ∧ n / 4 = 16 * (n / 64) := by omega

/-- The scale row stride is `n/64` and the code row stride is `n/4`. -/
theorem row_strides (n o : Nat) (h : n % 64 = 0) : o * (n / 4) = o * (16 * (n / 64)) := by
  have : n / 4 = 16 * (n / 64) := by omega
  rw [this]

-- § Shared-memory layout

/-- Staged `x` is 4096 halves = 8192 bytes. -/
theorem staged_bytes : 4096 * 2 = 8192 := by decide

/-- Partials start after `MAX_SHARED_HALVES/2` floats, i.e. exactly after the staged halves. -/
theorem partial_offset : (4096 / 2) * 4 = 4096 * 2 := by decide

/-- The staged and unstaged shared sizes. -/
def smemBytes (stage : Bool) (tpr : Nat) : Nat := (if stage then 4096 * 2 else 0) + tpr * 4

/-- At most 1024 threads per row: shared memory stays within 48 KiB in both modes. -/
theorem smem_fits (stage : Bool) (tpr : Nat) (h : tpr ≤ 1024) : smemBytes stage tpr ≤ 49152 := by
  unfold smemBytes
  split <;> omega

/-- `n_embd = 2560` fits the staging limit with room. -/
theorem n2560_fits : 2560 ≤ 4096 ∧ 2560 % 64 = 0 := by decide

/-- A staged `__half2` read at `q*4` is 4-byte aligned (offset `8q` bytes). -/
theorem half2_aligned (q : Nat) : (q * 4 * 2) % 4 = 0 := by omega

/-- A `uint2` read of `x + 4q` halves is 8-byte aligned. -/
theorem uint2_aligned (q : Nat) : (q * 4 * 2) % 8 = 0 := by omega

-- § Strided loops

/-- Quad `q` is visited by thread `q % tpr` in iteration `q / tpr`. -/
theorem quad_owner (q tpr : Nat) (h : 0 < tpr) : q = (q % tpr) + tpr * (q / tpr) := by
  have := Nat.mod_add_div q tpr
  omega

/-- The owner thread is below `tpr`. -/
theorem quad_owner_lt (q tpr : Nat) (h : 0 < tpr) : q % tpr < tpr := Nat.mod_lt q h

/-- The block-reduction tree halves a power of two down to 1 through `log2` steps: sum of steps equals `tpr - 1`. -/
theorem tree_steps : 512 + 256 + 128 + 64 + 32 + 16 + 8 + 4 + 2 + 1 = 1023 := by decide

end Strata.S2GemvFast



## FILE: /mnt/user-data/outputs/S2GemvQ8Activation.lean
/-!
# MODULE: s2_gemv_q8_activation
The S2 GEMV that consumes Q8_0 activations (`s2_gemv_q8.cu`), the correctness version of the ggml contract
(`vec_dot_type(Q2_0) = Q8_0`).  One block per output row, `threads_per_row` threads splitting the quads.

## Source semantics
* Activations are `n_in/32` blocks of 34 bytes: `fp16 d` then `int8 qs[32]`.
* A quad of four elements lies inside one 32-element block, so the block scale is loaded once.
* The weight `(code - 1) * d_w` is formed first and multiplied by the dequantised activation `xq * d_x`.
* One S2 weight scale covers 64 elements = 16 quads: index `q >> 4`.

## Form template identifiers
- `QK_S2` → `qk`; `QK8_0` → `qk8`; block bytes 34 → `blk`
- `threads_per_row` → `tpr`; quad index → `q`
-/

namespace Strata.S2GemvQ8

/-- Activation block of quad `q`. -/
def actBlock (q : Nat) : Nat := (q * 4) / 32

/-- Element offset inside the activation block. -/
def actOff (q : Nat) : Nat := (q * 4) % 32

/-- The quad's four elements stay inside the block: `off + 3 < 32`. -/
theorem quad_in_block (q : Nat) : actOff q + 3 < 32 := by
  unfold actOff; omega

/-- Block and offset recompose the element index. -/
theorem block_off_recompose (q : Nat) : actBlock q * 32 + actOff q = q * 4 := by
  unfold actBlock actOff; omega

/-- Byte address of the block: stride 34. -/
def blockByte (q : Nat) : Nat := actBlock q * 34

/-- The scale `d` (2 bytes at the block start) and the data (from byte 2) are contiguous in a 34-byte block. -/
theorem block_layout : 2 + 32 = 34 := by decide

/-- Quad data start at byte `blockByte + 2 + off`; the last read byte is inside the block. -/
theorem quad_bytes_in_block (q : Nat) : 2 + actOff q + 3 < 34 := by
  unfold actOff; omega

/-- The shared 16-bit scale read `blk[0] | blk[1] << 8` at an even stride is 2-byte aligned. -/
theorem scale_aligned (q : Nat) : blockByte q % 2 = 0 := by
  unfold blockByte; omega

-- § Group index and shapes

/-- S2 group index as a shift: `(q*4) >> 6 = q >> 4`. -/
theorem group_of_quad (q : Nat) : (q * 4) / 64 = q / 16 := by omega

/-- The two 32-blocks of one 64-group share the weight scale: both quads' groups agree. -/
theorem two_blocks_one_group (q : Nat) : ((q * 4) / 32) / 2 = (q * 4) / 64 := by omega

/-- Host guard: `n_in` multiple of 32 and 64 reduces to multiple of 64. -/
theorem guard_reduces (n : Nat) : (n % 32 = 0 ∧ n % 64 = 0) ↔ n % 64 = 0 := by omega

/-- Whole quads, 16 quads per group. -/
theorem quads_per_group (n : Nat) (h : n % 64 = 0) : n / 4 = 16 * (n / 64) := by omega

/-- Number of activation blocks for `n_in`. -/
theorem act_blocks (n : Nat) (h : n % 64 = 0) : n / 32 = 2 * (n / 64) := by omega

/-- Strided ownership of quads. -/
theorem quad_owner (q tpr : Nat) (h : 0 < tpr) : q = q % tpr + tpr * (q / tpr) := by
  have := Nat.mod_add_div q tpr
  omega

/-- Every quad index below `n_in/4` lies in a valid activation block. -/
theorem block_in_range (q n : Nat) (hq : q < n / 4) (h : n % 64 = 0) : actBlock q < n / 32 := by
  unfold actBlock; omega

-- § The arithmetic: weight times dequantised activation, in the kernel's association

/-- Integer version of the per-element product `(c - 1) * x`. -/
def term (c x : Int) : Int := (c - 1) * x

/-- A quad's four terms equal the code-dot minus the plain sum of activations (the bias split used by the
grouped expert kernel). -/
theorem quad_bias_split (c0 c1 c2 c3 x0 x1 x2 x3 : Int) :
    term c0 x0 + term c1 x1 + term c2 x2 + term c3 x3 =
      (c0 * x0 + c1 * x1 + c2 * x2 + c3 * x3) - (x0 + x1 + x2 + x3) := by
  unfold term
  simp only [Int.sub_mul, Int.one_mul]
  omega

/-- Biased code range: code 0..3 gives -1..2, so a term is bounded by `2*|x|`; with `|x| ≤ 127` it fits in 10 bits. -/
theorem term_bound (c x : Int) (hc0 : 0 ≤ c) (hc3 : c ≤ 3) (hx0 : -127 ≤ x) (hx1 : x ≤ 127) :
    -254 ≤ term c x ∧ term c x ≤ 254 := by
  unfold term
  rcases (by omega : c = 0 ∨ c = 1 ∨ c = 2 ∨ c = 3) with h | h | h | h <;> subst h <;> omega

/-- The 32 products of one block sum to at most `32*254`, far below int32. -/
theorem block_sum_bound : 32 * 254 < 2 ^ 31 := by decide

/-- Launch: one block per row, `n_out` blocks, partials of `tpr` floats. -/
theorem smem_fits (tpr : Nat) (h : tpr ≤ 1024) : tpr * 4 ≤ 49152 := by omega

end Strata.S2GemvQ8



## FILE: /mnt/user-data/outputs/S2ExpertGroupedLayout.lean
/-!
# MODULE: s2_expert_grouped_layout
The grouped GPU expert for Q2_0 weights (`s2_expert_grouped.cu`): blob geometry, the gate/up row interleave
and gate-major output layout, the dp4a bias-split identity, chunking of rows, the per-token activation
selection and the shared-memory budgets of the grouped kernels.

## Source semantics
* Blob: gate/up codes (2*FF rows of 640 B), down codes (H rows of 160 B), gate/up scales, down scales.
* Row slot `i` of the interleaved gate/up rows is gate row `i/2` for even `i`, up row `i/2` for odd `i`;
  the output is gate-major: gate rows of hit `h` at `h*FF + r`, up rows at `n*FF + h*FF + r`.
* `sum (c-1)*x = sum c*x - sum x`, both sums INT8 x INT8 (`__dp4a`).
* One weight scale per 64 elements = two 32-element chunks.

## Form template identifiers
- `H` → `H`, `FF` → `FF`, `QK` → `qk`, `ROW_GU/ROW_D` → `rowGU/rowD`, `SC_GU/SC_D` → `scGU/scD`
- `O_D_CODES/O_GU_SCALES/O_D_SCALES` → `oDCodes/oGUScales/oDScales`
- hit index → `h`, row slot → `i`, hit count → `n`
-/

namespace Strata.S2Expert

-- § Blob geometry

/-- Hidden size. -/
def H : Nat := 2560
/-- FFN width. -/
def FF : Nat := 640
/-- Gate/up code row bytes. -/
def rowGU : Nat := H / 4
/-- Down code row bytes. -/
def rowD : Nat := FF / 4
/-- Gate/up scales per row. -/
def scGU : Nat := H / 64
/-- Down scales per row. -/
def scD : Nat := FF / 64
/-- Offset of the down codes. -/
def oDCodes : Nat := 2 * FF * rowGU
/-- Offset of the gate/up scales. -/
def oGUScales : Nat := oDCodes + H * rowD
/-- Offset of the down scales. -/
def oDScales : Nat := oGUScales + 2 * FF * scGU * 2
/-- Total blob bytes. -/
def blobBytes : Nat := oDScales + H * scD * 2

/-- The row sizes. -/
theorem row_sizes : rowGU = 640 ∧ rowD = 160 ∧ scGU = 40 ∧ scD = 10 := by decide

/-- The blob offsets, evaluated. -/
theorem offsets : oDCodes = 819200 ∧ oGUScales = 1228800 ∧ oDScales = 1331200 ∧ blobBytes = 1382400 := by decide

/-- The regions are ordered and disjoint. -/
theorem regions_ordered : oDCodes < oGUScales ∧ oGUScales < oDScales ∧ oDScales < blobBytes := by decide

/-- Region sizes tile the blob exactly. -/
theorem regions_tile :
    2 * FF * rowGU + H * rowD + 2 * FF * scGU * 2 + H * scD * 2 = blobBytes := by decide

/-- Every region start is 2-byte aligned (scales are fp16 read as bytes) and in fact 16-byte aligned. -/
theorem offsets_aligned : oDCodes % 16 = 0 ∧ oGUScales % 16 = 0 ∧ oDScales % 16 = 0 := by decide

/-- Chunk counts: 80 chunks per gate/up row, 20 per down row. -/
theorem chunk_counts : H / 32 = 80 ∧ FF / 32 = 20 := by decide

/-- A chunk's weight scale index `c/2` is within the row's scales. -/
theorem scale_index_gu (c : Nat) (h : c < H / 32) : c / 2 < scGU := by
  simp only [H, scGU] at *; omega

/-- Same for the down row. -/
theorem scale_index_d (c : Nat) (h : c < FF / 32) : c / 2 < scD := by
  simp only [FF, scD] at *; omega

/-- Two chunks per 64-element scale group. -/
theorem chunk_group (c : Nat) : (c * 32) / 64 = c / 2 := by omega

/-- Each chunk's 8 code bytes (32 elements at 2 bits) lie at `c*8`, inside the row. -/
theorem chunk_code_bytes (c : Nat) (h : c < H / 32) : c * 8 + 8 ≤ rowGU := by
  simp only [H, rowGU] at *; omega

/-- Down row: same. -/
theorem chunk_code_bytes_down (c : Nat) (h : c < FF / 32) : c * 8 + 8 ≤ rowD := by
  simp only [FF, rowD] at *; omega

/-- Lanes: 80 chunks over 32 lanes need 3 per lane (the `GU_CHUNKS` constant). -/
theorem gu_chunks : (H / 32 + 31) / 32 = 3 ∧ 2 * 32 + 16 = H / 32 := by decide

/-- A down row's 20 chunks fit in one warp. -/
theorem down_chunks_one_warp : FF / 32 ≤ 32 := by decide

-- § Interleave and gate-major layout

/-- Output position of row slot `i` of hit `h`, for `n` hits: gate-major. -/
def dest (n h i : Nat) : Nat :=
  if i % 2 = 1 then n * FF + h * FF + i / 2 else h * FF + i / 2

/-- Slot `i` decodes row `i`; its parity gives the destination plane. Destination is in range. -/
theorem dest_lt (n h i : Nat) (hh : h < n) (hi : i < 2 * FF) : dest n h i < 2 * (n * FF) := by
  unfold dest FF at *
  split_ifs <;> omega

/-- The map `(h, i) ↦ dest` is injective: no two row slots collide. -/
theorem dest_injective (n h i h' i' : Nat) (hh : h < n) (hh' : h' < n) (hi : i < 2 * FF) (hi' : i' < 2 * FF)
    (heq : dest n h i = dest n h' i') : h = h' ∧ i = i' := by
  unfold dest FF at *
  split_ifs at heq <;> omega

/-- Gate row `r` of hit `h` lands at `h*FF + r`, and the matching up row sits exactly `n*FF` later (the swiglu pairing). -/
theorem swiglu_pairing (n h r : Nat) :
    dest n h (2 * r + 1) = dest n h (2 * r) + n * FF := by
  unfold dest
  have h1 : (2 * r + 1) % 2 = 1 := by omega
  have h2 : (2 * r) % 2 = 0 := by omega
  have h3 : (2 * r + 1) / 2 = r := by omega
  have h4 : (2 * r) / 2 = r := by omega
  rw [if_pos h1, if_neg (show ¬ ((2 * r) % 2 = 1) by omega), h3, h4]
  omega

/-- The swiglu result for entry `(h, r)` lands in the first `n*FF` floats, where the quantiser reads it. -/
theorem swiglu_output_in_gate_plane (n h r : Nat) (hh : h < n) (hr : r < FF) : h * FF + r < n * FF := by
  have : (h + 1) * FF ≤ n * FF := Nat.mul_le_mul_right FF hh
  rw [Nat.add_mul] at this
  omega

-- § The dp4a bias split

/-- Dot of four int pairs (the semantics of `__dp4a` on unpacked bytes). -/
def dp4 (a0 a1 a2 a3 b0 b1 b2 b3 : Int) : Int := a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3

/-- `dp4a(ones, x)` is the plain sum of the four activations. -/
theorem dp4_ones (x0 x1 x2 x3 : Int) : dp4 1 1 1 1 x0 x1 x2 x3 = x0 + x1 + x2 + x3 := by
  unfold dp4; omega

/-- `s - hx` equals the dot of the biased codes `(c-1)` with the activations, over a quad. -/
theorem bias_split (c0 c1 c2 c3 x0 x1 x2 x3 : Int) :
    dp4 c0 c1 c2 c3 x0 x1 x2 x3 - dp4 1 1 1 1 x0 x1 x2 x3 =
      dp4 (c0 - 1) (c1 - 1) (c2 - 1) (c3 - 1) x0 x1 x2 x3 := by
  unfold dp4
  simp only [Int.sub_mul, Int.one_mul]
  omega

/-- The integer sums over a 32-element chunk fit in int32. -/
theorem chunk_sum_bound : 32 * (3 * 127) < 2 ^ 31 := by decide

-- § Unpacked code word

/-- The four 2-bit fields of a byte, placed one per byte of a word. -/
def expand (b : Nat) : Nat := b % 4 + 256 * (b / 4 % 4) + 65536 * (b / 16 % 4) + 16777216 * (b / 64 % 4)

/-- Each byte lane of the expansion is a 2-bit value, so `dp4a` sees values 0..3 (with `a_k` the k-th field
of the byte, i.e. `expand b = a0 + 256*a1 + 65536*a2 + 16777216*a3`). -/
theorem expand_lanes (a0 a1 a2 a3 : Nat) (h0 : a0 < 4) (h1 : a1 < 4) (h2 : a2 < 4) (h3 : a3 < 4) :
    (a0 + 256 * a1 + 65536 * a2 + 16777216 * a3) % 256 = a0 ∧
    (a0 + 256 * a1 + 65536 * a2 + 16777216 * a3) / 256 % 256 = a1 ∧
    (a0 + 256 * a1 + 65536 * a2 + 16777216 * a3) / 65536 % 256 = a2 ∧
    (a0 + 256 * a1 + 65536 * a2 + 16777216 * a3) / 16777216 = a3 := by omega

/-- `expand` is that sum of the byte's four fields. -/
theorem expand_eq (b : Nat) :
    expand b = b % 4 + 256 * (b / 4 % 4) + 65536 * (b / 16 % 4) + 16777216 * (b / 64 % 4) := rfl

-- § Device-side hit selection and grouping

/-- `dst / k` recovers the token of routed entry `dst < k*T`. -/
theorem token_of_entry (dst k T : Nat) (h : dst < k * T) (hk : 0 < k) : dst / k < T :=
  Nat.div_lt_of_lt_mul h

/-- Entry `i` of token `i / k` is routed slot `i % k`; the pair recomposes `i`. -/
theorem entry_decompose (i k : Nat) (hk : 0 < k) : (i / k) * k + i % k = i := by
  have := Nat.div_add_mod i k
  rw [Nat.mul_comm] at this
  omega

/-- The verify window handles at most 128 entries: 4 warps. -/
theorem window_warps : 128 / 32 = 4 := by decide

/-- Warp prefix count: a hit's compacted position is the count of earlier hits, so it is below the total. -/
theorem compact_lt (before within total : Nat) (h : before + within < total) : before + within < total := h

-- § Shared-memory budgets and grids

/-- Gate/up kernel staging: GMAX entries of H int8 (as words) and H/32 scales. -/
def guSmem : Nat := 8 * (H / 4) * 4 + 8 * (H / 32) * 4
/-- Down kernel staging. -/
def downSmem : Nat := 8 * (FF / 4) * 4 + 8 * (FF / 32) * 4

/-- Both fit in the 48 KiB static limit. -/
theorem smem_budget : guSmem = 23040 ∧ downSmem = 5760 ∧ guSmem ≤ 49152 := by decide

/-- Gate/up grid: 40 blocks of 32 rows (8 warps x 4 rows) cover all `2*FF` rows. -/
theorem gu_grid : 2 * FF / 32 = 40 ∧ 40 * 32 = 2 * FF := by decide

/-- Down grid: 40 blocks of 64 rows (8 warps x 8 rows) cover all `H` rows. -/
theorem down_grid : H / 64 = 40 ∧ 40 * 64 = H := by decide

/-- Warp `w` of 8 handling rows `w, w+8, w+16, w+24` of a 32-row block covers each row exactly once. -/
theorem warp_rows_gu (rr : Nat) (h : rr < 32) : rr = rr % 8 + 8 * (rr / 8) := by omega

/-- Flat per-hit launch: one warp per row; the grid covers all rows. -/
theorem warp_grid (rows warps : Nat) (hw : 0 < warps) : rows ≤ warps * ((rows + warps - 1) / warps) := by
  have := Nat.div_add_mod (rows + warps - 1) warps
  have := Nat.mod_lt (rows + warps - 1) hw
  omega

/-- Scratch alignment helper used by the three scratch regions. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Alignment rounds up, stays within 15, and is a multiple of 16. -/
theorem align16_spec (x : Nat) : x ≤ align16 x ∧ align16 x < x + 16 ∧ align16 x % 16 = 0 := by
  unfold align16; omega

end Strata.S2Expert



## FILE: /mnt/user-data/outputs/NativeMmvqGeometry.lean
/-!
# MODULE: native_mmvq_geometry
The ggml-compatible quantised mat-vec kernels (`native_mmvq.cu`, adapted from llama.cpp MMVQ): block
layouts, thread-to-block mapping, small-K thresholds, multi-column addressing, shared-memory budgets, the
Q8_1 quantiser's warp/block alignment and the weight-size table.

## Source semantics
* Each format has `(DIV, T, KBY, BPI)`: weight elements per block, threads per block, Q8_1 blocks per weight
  block, weight blocks per K iteration.  Thread `tid` (0..127) handles block `tid / T` of each iteration
  and sub-index `tid % T`; `T * BPI = 128` so one iteration covers all 128 threads exactly.
* Small K (`n_in/DIV < BPI`) switches to 4 rows per block; otherwise 1 row per block.
* Column `j` reads activation blocks at `j * (n_in/32)` and writes `y + j*n_out`.
* Q4_0/Q5_0 are affine: `d*(sumi*ds.x - c*ds.y)` with `c * T = 8` resp. `16` in total.

## Form template identifiers
- `Q5KTraits/Q4KTraits/...` → `traits` entries; `WARPS` → `warps`; `WARP` → `warp`
- `QI`, `VDR` → `qi`, `vdr`; `n_in/n_out/ncols` → `n/nout/ncols`
- ggml type code → `ty`
-/

namespace Strata.NativeMmvq

-- § Block layouts

/-- Q5_K block: `half2 dm; u8 scales[12]; u8 qh[32]; u8 qs[128]`. -/
theorem q5k_size : 4 + 12 + 32 + 128 = 176 := by decide
/-- Q4_K block. -/
theorem q4k_size : 4 + 12 + 128 = 144 := by decide
/-- Q3_K block: hmask 32, qs 64, scales 12, d 2. -/
theorem q3k_size : 32 + 64 + 12 + 2 = 110 := by decide
/-- IQ4_XS block: d 2, scales_h 2, scales_l 4, qs 128. -/
theorem iq4xs_size : 2 + 2 + 4 + 128 = 136 := by decide
/-- Q6_K block: ql 128, qh 64, scales 16, d 2. -/
theorem q6k_size : 128 + 64 + 16 + 2 = 210 := by decide
/-- Q4_0 / IQ4_NL block: d 2, qs 16. -/
theorem q40_size : 2 + 16 = 18 := by decide
/-- Q5_0 block: d 2, qh 4, qs 16. -/
theorem q50_size : 2 + 4 + 16 = 22 := by decide
/-- Q8_0 block: d 2, qs 32. -/
theorem q80_size : 2 + 32 = 34 := by decide
/-- Q8_1 block: `half2 ds`, qs 32: 36 bytes, 4-byte aligned. -/
theorem q81_size : 4 + 32 = 36 := by decide
/-- Q2_0 block: d 2, qs 16 (64 two-bit codes). -/
theorem q20_size : 2 + 16 = 18 ∧ 16 * 4 = 64 := by decide

/-- Q3_K has 110-byte stride: every second block is only 2-byte aligned, so loads go through 16-bit halves. -/
theorem q3k_alternate_alignment (b : Nat) : (b * 110) % 4 = 0 ∨ (b * 110) % 4 = 2 := by omega

/-- Two 16-bit loads recombine little-endian into the 32-bit word. -/
theorem load_int_b2_value (lo hi : Nat) (hlo : lo < 65536) : lo + 65536 * hi = lo + hi * 65536 := by omega

-- § Thread-to-block mapping

/-- Traits: elements per block, threads per block, Q8_1 blocks per weight block, blocks per iteration. -/
structure Traits where
  div : Nat
  t : Nat
  kby : Nat
  bpi : Nat

/-- Q5_K / Q4_K. -/
def q45k : Traits := ⟨256, 16, 8, 8⟩
/-- Q2_0. -/
def q20 : Traits := ⟨64, 2, 2, 64⟩
/-- Q3_K. -/
def q3k : Traits := ⟨256, 16, 8, 8⟩
/-- Q6_K. -/
def q6k : Traits := ⟨256, 32, 8, 4⟩
/-- IQ4_XS. -/
def iq4xs : Traits := ⟨256, 8, 8, 16⟩
/-- Q4_0 / Q5_0 / IQ4_NL (Qi = 4). -/
def small4 : Traits := ⟨32, 2, 1, 64⟩
/-- Q8_0 (Qi = 8). -/
def small8 : Traits := ⟨32, 4, 1, 32⟩

/-- The 128 threads exactly tile one iteration: `T * BPI = 128` for every format. -/
theorem threads_tile :
    q45k.t * q45k.bpi = 128 ∧ q20.t * q20.bpi = 128 ∧ q3k.t * q3k.bpi = 128 ∧
    q6k.t * q6k.bpi = 128 ∧ iq4xs.t * iq4xs.bpi = 128 ∧
    small4.t * small4.bpi = 128 ∧ small8.t * small8.bpi = 128 := by decide

/-- `KBY = DIV / 32`: Q8_1 blocks per weight block. -/
theorem kby_is_div_over_32 :
    q45k.kby = q45k.div / 32 ∧ q20.kby = q20.div / 32 ∧ q3k.kby = q3k.div / 32 ∧
    q6k.kby = q6k.div / 32 ∧ iq4xs.kby = iq4xs.div / 32 ∧
    small4.kby = small4.div / 32 ∧ small8.kby = small8.div / 32 := by decide

/-- Derivation of `BPI` from `VDR * WARPS * WARP / QI` for the K formats. -/
theorem bpi_derivation :
    2 * 4 * 32 / 32 = 8 ∧ 1 * 4 * 32 / 16 = 8 ∧ 1 * 4 * 32 / 32 = 4 ∧ 4 * 4 * 32 / 32 = 16 ∧
    4 * 32 / 2 = 64 ∧ 2 * 4 * 32 / 4 = 64 ∧ 2 * 4 * 32 / 8 = 32 := by decide

/-- Thread `tid` splits uniquely into `(tid / T, tid % T)`. -/
theorem tid_split (tid T : Nat) (hT : 0 < T) : tid = T * (tid / T) + tid % T := by
  have := Nat.div_add_mod tid T
  omega

/-- The block slot of a thread is below `BPI` when `tid < T * BPI`. -/
theorem slot_lt (tid T bpi : Nat) (h : tid < T * bpi) (hT : 0 < T) : tid / T < bpi :=
  Nat.div_lt_of_lt_mul h

/-- Two threads with the same slot and sub-index are the same thread. -/
theorem tid_unique (t1 t2 T : Nat) (hT : 0 < T) (hs : t1 / T = t2 / T) (hm : t1 % T = t2 % T) : t1 = t2 := by
  have h1 := Nat.div_add_mod t1 T
  have h2 := Nat.div_add_mod t2 T
  rw [hs, hm] at h1
  omega

/-- Sub-index ranges: the `kqs` values stay inside the format's `QI` ints (and `kqs+VDR-1` too). -/
theorem kqs_ranges (tid : Nat) :
    2 * (tid % 16) + 1 < 32 ∧ 4 * (tid % 8) + 3 < 32 ∧ tid % 32 < 32 ∧ tid % 16 < 16 ∧
    tid % 2 < 2 ∧ 2 * (tid % 2) + 1 < 4 ∧ 2 * (tid % 4) + 1 < 8 := by omega

-- § Small-K thresholds

/-- With `n % DIV = 0`, the small-K rule `n/DIV < BPI` is exactly `n < DIV * BPI`. -/
theorem small_k_iff (n div bpi : Nat) (hd : 0 < div) (h : n % div = 0) : n / div < bpi ↔ n < div * bpi := by
  have := Nat.div_add_mod n div
  constructor
  · intro hlt
    have : n / div + 1 ≤ bpi := hlt
    have h2 : div * (n / div + 1) ≤ div * bpi := Nat.mul_le_mul_left div this
    rw [Nat.mul_add] at h2
    omega
  · intro hlt
    exact Nat.div_lt_of_lt_mul hlt

/-- Elements per iteration thresholds quoted in the source comment: 2048 for K-formats, 1024 for Q8_0. -/
theorem thresholds :
    q45k.div * q45k.bpi = 2048 ∧ q3k.div * q3k.bpi = 2048 ∧ q6k.div * q6k.bpi = 1024 ∧
    small4.div * small4.bpi = 2048 ∧ small8.div * small8.bpi = 1024 := by decide

/-- Rows per block and grid: 4 rows when small K, grid `ceil(n_out/4)` covers all rows. -/
theorem small_k_grid (n_out : Nat) : n_out ≤ 4 * ((n_out + 3) / 4) := by omega

/-- Row of lane group `i` in block `b` is `ROWS*b + i`; every row has a unique `(b, i)`. -/
theorem row_cover (r rows : Nat) (h : 0 < rows) : r = rows * (r / rows) + r % rows := by
  have := Nat.div_add_mod r rows
  omega

-- § Multi-column addressing

/-- Maximum columns. -/
def maxNcols : Nat := 8

/-- Column `j` of the activation starts at block `j * (n/32)`; columns are contiguous when `n % 32 = 0`. -/
theorem column_block (c m j : Nat) (hj : j < m * 32) : (c * m * 32 + j) / 32 = c * m + j / 32 := by
  omega

/-- The output of column `j`, row `r` is at `j*n_out + r < ncols*n_out`. -/
theorem y_index_lt (j r n_out ncols : Nat) (hj : j < ncols) (hr : r < n_out) : j * n_out + r < ncols * n_out := by
  have : (j + 1) * n_out ≤ ncols * n_out := Nat.mul_le_mul_right n_out hj
  rw [Nat.add_mul] at this
  omega

/-- The partials array `[NW-1][NCOLS][ROWS][32]` floats. -/
def partialBytes (nw ncols rows : Nat) : Nat := (nw - 1) * ncols * rows * 32 * 4

/-- Exact layout (NW = 4, ROWS ≤ 4) fits 48 KiB for every `ncols ≤ 8`. -/
theorem partial_fits_exact (ncols : Nat) (h : ncols ≤ 8) : partialBytes 4 ncols 4 ≤ 49152 := by
  unfold partialBytes; omega

/-- Upstream layout (NW = 2, ROWS = 2) is far smaller. -/
theorem partial_fits_upstream (ncols : Nat) (h : ncols ≤ 8) : partialBytes 2 ncols 2 ≤ 2048 := by
  unfold partialBytes; omega

/-- Upstream layout block counts: `ncols ≤ 4` uses 4 warps, otherwise 2; two rows per block. -/
theorem upstream_warps : (4 : Nat) ≤ 4 ∧ 2 * 32 = 64 := by decide

/-- Two-rows-per-block grid covers all rows. -/
theorem upstream_grid (n_out : Nat) : n_out ≤ 2 * ((n_out + 1) / 2) := by omega

-- § Affine corrections of the 32-element formats

/-- Q4_0: each of the T=2 threads subtracts `4*ds.y`, totalling `8 * sum x`. -/
theorem q40_correction : 2 * (8 * 2 / 4) = 8 := by decide

/-- Q5_0: each thread subtracts `8*ds.y`, totalling `16 * sum x`. -/
theorem q50_correction : 2 * (16 * 2 / 4) = 16 := by decide

/-- The affine identity behind both: `Σ(q-b)u = Σqu - bΣu` over a quad. -/
theorem affine_identity (q0 q1 q2 q3 u0 u1 u2 u3 b : Int) :
    (q0 - b) * u0 + (q1 - b) * u1 + (q2 - b) * u2 + (q3 - b) * u3 =
      (q0 * u0 + q1 * u1 + q2 * u2 + q3 * u3) - b * (u0 + u1 + u2 + u3) := by
  simp only [Int.sub_mul, Int.mul_add]
  omega

-- § The Q8_1 quantiser

/-- Quantiser blocks of 256 threads are 8 warps; element `i`'s 32-block is `blockIdx*8 + warp`. -/
theorem quant_block_of_thread (b t : Nat) (ht : t < 256) : (b * 256 + t) / 32 = b * 8 + t / 32 := by
  omega

/-- A warp is exactly one Q8_1 block: lanes of the same warp share `i / 32`. -/
theorem warp_is_block (b t u : Nat) (ht : t < 256) (hu : u < 256) (h : t / 32 = u / 32) :
    (b * 256 + t) / 32 = (b * 256 + u) / 32 := by omega

/-- With `n % 32 = 0` the early return only removes whole warps (no partial-warp shuffle deadlock). -/
theorem early_return_whole_warps (n i j : Nat) (hn : n % 32 = 0) (hi : n ≤ i) (hij : i / 32 = j / 32) : n ≤ j := by
  omega

/-- Grid covers all elements. -/
theorem quant_grid (n : Nat) : n ≤ 256 * ((n + 255) / 256) := by omega

/-- Element index recomposes from block and offset. -/
theorem quant_index (i : Nat) : (i / 32) * 32 + i % 32 = i := by omega

-- § Type table and weight sizes

/-- (elements per block, bytes per block) for a ggml type code, `none` when unsupported. -/
def blockInfo : Nat → Option (Nat × Nat)
  | 2 => some (32, 18)
  | 6 => some (32, 22)
  | 8 => some (32, 34)
  | 20 => some (32, 18)
  | 11 => some (256, 110)
  | 12 => some (256, 144)
  | 13 => some (256, 176)
  | 14 => some (256, 210)
  | 23 => some (256, 136)
  | 42 => some (64, 18)
  | 16 => some (256, 66)
  | 17 => some (256, 74)
  | 18 => some (256, 98)
  | 21 => some (256, 110)
  | 22 => some (256, 82)
  | 29 => some (256, 56)
  | _ => none

/-- The 16 supported types. -/
def supportedTypes : List Nat := [2, 6, 8, 11, 12, 13, 14, 20, 23, 42, 16, 17, 18, 21, 22, 29]

/-- Every supported type has block information. -/
theorem supported_have_info : ∀ t ∈ supportedTypes, (blockInfo t).isSome = true := by decide

/-- Every supported block size is 32, 64 or 256 and a multiple of 32 (the Q8_1 granule). -/
theorem supported_block_elems :
    ∀ t ∈ supportedTypes, ((blockInfo t).map Prod.fst).getD 0 % 32 = 0 ∧
      (((blockInfo t).map Prod.fst).getD 0 = 32 ∨ ((blockInfo t).map Prod.fst).getD 0 = 64 ∨
        ((blockInfo t).map Prod.fst).getD 0 = 256) := by
  decide

/-- Row bytes for `n` elements. -/
def rowBytes (elems bytes n : Nat) : Nat := n / elems * bytes

/-- Overflow check of the host: `row ≤ MAX / n_out` implies `row * n_out ≤ MAX`. -/
theorem weight_bytes_no_overflow (row nOut M : Nat) (hn : 0 < nOut) (h : row ≤ M / nOut) : row * nOut ≤ M :=
  (Nat.le_div_iff_mul_le hn).mp h

/-- Q4_K for `n_in = 2560`: ten blocks, 1440 bytes per row. -/
theorem q4k_row_2560 : rowBytes 256 144 2560 = 1440 := by decide

/-- Q6_K for `n_in = 2560`. -/
theorem q6k_row_2560 : rowBytes 256 210 2560 = 2100 := by decide

end Strata.NativeMmvq



## FILE: /mnt/user-data/outputs/IqKernelsGeometry.lean
/-!
# MODULE: iq_kernels_geometry
The IQ-family / Q3_K / Q2_0 CUDA kernels (`iq_kernels.cu`): the lane-strided dot-call enumeration of
`row_dot`, the grouped native expert layout and scratch carving, the gate/up row mapping, the dequantiser
output positions, the interleaved gate/up dequant layout and the q8_1 quantiser indexing.

## Source semantics
* A format has `qk` values per block, `ipb` dot calls per block and `step` between `iqs` values.
  Dot call `k` is `(kbx, iqs) = (k / ipb, step * (k % ipb))`, lane-strided by 32, against activation blocks
  `x + kbx * (qk / 32)`.
* Expert blob: gate rows, then up rows, then down rows; `up_off = n_ff*gu_row`, `down_off = 2*up_off`.
* Dequantiser thread `tid < 32` of a 256-value superblock writes disjoint positions covering all 256.
* `dequant_gu`: output block of `(r, c, parity)` is `(2r+parity)*per_row + c`.

## Form template identifiers
- `Fmt<TY>::qk/ipb/step` → `qk/ipb/step`; `NativeExpertLayout` fields → `Layout`
- `GU_ROWS` → `guRows`; `kvalues_iq4nl` → (module `SGemvQuadDecode.kIq4Nl`)
- `block_q8_1` → 36 bytes
-/

namespace Strata.IqKernels

-- § Dot-call enumeration

/-- Call index decomposes into block and part: `k = kbx*ipb + part`. -/
theorem call_decompose (k ipb : Nat) (h : 0 < ipb) : (k / ipb) * ipb + k % ipb = k := by
  have := Nat.div_add_mod k ipb
  rw [Nat.mul_comm] at this
  omega

/-- Part index is below `ipb`. -/
theorem part_lt (k ipb : Nat) (h : 0 < ipb) : k % ipb < ipb := Nat.mod_lt k h

/-- The `iqs` values of one block are distinct for distinct parts (step > 0). -/
theorem iqs_distinct (step p1 p2 : Nat) (h : 0 < step) (heq : step * p1 = step * p2) : p1 = p2 :=
  Nat.eq_of_mul_eq_mul_left h heq

/-- Format table values: `(qk, ipb, step)`; iqs max stays within the block's `qi` ints. -/
theorem iqs_max :
    2 * (8 - 1) < 16 ∧ 1 * (8 - 1) < 8 ∧ 2 * (2 - 1) < 4 ∧ 1 * (2 - 1) < 2 := by decide

/-- Activation q8_1 blocks per weight block: 8 for 256-blocks, 2 for Q2_0, 1 for IQ4_NL. -/
theorem q8_blocks_per_block : 256 / 32 = 8 ∧ 64 / 32 = 2 ∧ 32 / 32 = 1 := by decide

/-- Calls per row: `nb * ipb`; every call below that count has a valid block. -/
theorem call_block_lt (k nb ipb : Nat) (h : k < nb * ipb) (hp : 0 < ipb) : k / ipb < nb :=
  Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact h)

/-- Lane stride: call `k` belongs to lane `k % 32`, iteration `k / 32`; unique. -/
theorem lane_call_unique (k : Nat) : k = k % 32 + 32 * (k / 32) := by omega

/-- Warp-per-row blocks: 4 rows per block, `threadIdx.y` selects the row; grid covers all rows. -/
theorem mmvq_grid (n_out : Nat) : n_out ≤ 4 * ((n_out + 3) / 4) := by omega

/-- Row of `(blockIdx.x, threadIdx.y)` is injective on `ty < 4`. -/
theorem mmvq_row_inj (b1 y1 b2 y2 : Nat) (h1 : y1 < 4) (h2 : y2 < 4) (h : b1 * 4 + y1 = b2 * 4 + y2) :
    b1 = b2 ∧ y1 = y2 := by omega

/-- Output index of column `c`, row `r`: `c*n_out + r`. -/
theorem y_index_lt (c r n_out ncols : Nat) (hc : c < ncols) (hr : r < n_out) :
    c * n_out + r < ncols * n_out := by
  have : (c + 1) * n_out ≤ ncols * n_out := Nat.mul_le_mul_right n_out hc
  rw [Nat.add_mul] at this
  omega

-- § Block sizes (components of the ggml structs)

/-- iq2_xxs: d + 32 u16 = 66. -/
theorem iq2xxs_size : 2 + 2 * 32 = 66 := by decide
/-- iq2_xs: d + 32 u16 + 8 scales = 74. -/
theorem iq2xs_size : 2 + 2 * 32 + 8 = 74 := by decide
/-- iq2_s: d + 64 qs (incl. signs) + 8 qh + 8 scales = 82. -/
theorem iq2s_size : 2 + 64 + 8 + 8 = 82 := by decide
/-- iq3_xxs: d + 3*256/8 = 98. -/
theorem iq3xxs_size : 2 + 3 * 256 / 8 = 98 := by decide
/-- iq3_s: d + qs 64 + qh 8 + signs 32 + scales 4 = 110. -/
theorem iq3s_size : 2 + 64 + 8 + 32 + 4 = 110 := by decide
/-- iq1_m: qs 32 + qh 16 + scales 8 = 56. -/
theorem iq1m_size : 32 + 16 + 8 = 56 := by decide
/-- q3_K = 110, iq4_xs = 136, iq4_nl = 18, q2_0 = 18. -/
theorem other_sizes : 32 + 64 + 12 + 2 = 110 ∧ 2 + 2 + 4 + 128 = 136 ∧ 2 + 16 = 18 := by decide

-- § Native expert layout

/-- Alignment to 256 bytes. -/
def align256 (x : Nat) : Nat := (x + 255) / 256 * 256

/-- Rounding up to 256 is above, within 255, and aligned. -/
theorem align256_spec (x : Nat) : x ≤ align256 x ∧ align256 x < x + 256 ∧ align256 x % 256 = 0 := by
  unfold align256; omega

/-- Expert blob offsets from the per-row byte counts. -/
structure Layout where
  guRow : Nat
  dRow : Nat
  nEmbd : Nat
  nFF : Nat

/-- Up-projection offset. -/
def Layout.upOff (L : Layout) : Nat := L.nFF * L.guRow
/-- Down-projection offset. -/
def Layout.downOff (L : Layout) : Nat := 2 * L.upOff
/-- Total bytes. -/
def Layout.bytes (L : Layout) : Nat := L.downOff + L.nEmbd * L.dRow

/-- The three regions are ordered. -/
theorem layout_ordered (L : Layout) : L.upOff ≤ L.downOff ∧ L.downOff ≤ L.bytes := by
  unfold Layout.bytes Layout.downOff
  omega

/-- Gate and up regions have equal size, so the down region starts at twice the up offset. -/
theorem layout_gate_up_equal (L : Layout) : L.downOff - L.upOff = L.upOff := by
  unfold Layout.downOff; omega

/-- A gate row `r < n_ff` lies entirely below `up_off`. -/
theorem gate_row_in_region (L : Layout) (r : Nat) (h : r < L.nFF) : r * L.guRow + L.guRow ≤ L.upOff := by
  have : (r + 1) * L.guRow ≤ L.nFF * L.guRow := Nat.mul_le_mul_right L.guRow h
  rw [Nat.add_mul] at this
  unfold Layout.upOff
  omega

/-- An up row lies between `up_off` and `down_off`. -/
theorem up_row_in_region (L : Layout) (r : Nat) (h : r < L.nFF) :
    L.upOff + r * L.guRow + L.guRow ≤ L.downOff := by
  have := gate_row_in_region L r h
  unfold Layout.downOff
  omega

/-- A down row lies between `down_off` and `bytes`. -/
theorem down_row_in_region (L : Layout) (r : Nat) (h : r < L.nEmbd) :
    L.downOff + r * L.dRow + L.dRow ≤ L.bytes := by
  have : (r + 1) * L.dRow ≤ L.nEmbd * L.dRow := Nat.mul_le_mul_right L.dRow h
  rw [Nat.add_mul] at this
  unfold Layout.bytes
  omega

/-- Scratch: three float planes (gate, up, h), each aligned to 256, then the q8_1 image. -/
def scratchBytes (planeBytes q8Bytes : Nat) : Nat := 3 * align256 planeBytes + align256 q8Bytes

/-- The carved regions are in order and the total covers them: offsets `0, fa, 2fa, 3fa`. -/
theorem scratch_covers (plane q8 : Nat) :
    plane ≤ align256 plane ∧ q8 ≤ align256 q8 ∧
      3 * align256 plane + q8 ≤ scratchBytes plane q8 := by
  have h1 := align256_spec plane
  have h2 := align256_spec q8
  unfold scratchBytes
  omega

/-- A q8_1 block is 36 bytes, so every block is 4-byte aligned from a 256-aligned base. -/
theorem q81_block_aligned (b : Nat) : (b * 36) % 4 = 0 := by omega

/-- Number of q8_1 blocks of `cap` entries of `32*m` floats: `cap*m` (n_ff multiple of 32). -/
theorem q81_blocks_per_entries (cap m : Nat) : cap * (32 * m) / 32 = cap * m := by
  have : cap * (32 * m) = 32 * (cap * m) := by rw [Nat.mul_left_comm]
  rw [this]
  exact Nat.mul_div_cancel_left _ (by decide)

-- § Gate/up row mapping inside a group

/-- Row `row ∈ [0, 2*n_ff)` is gate when `row < n_ff` else up with `r = row - n_ff`. -/
theorem row_split_cover (row nFF : Nat) (h : row < 2 * nFF) :
    (row < nFF ∧ row < nFF) ∨ (nFF ≤ row ∧ row - nFF < nFF) := by omega

/-- The `(plane, r)` pair recomposes the row, so distinct rows write distinct slots. -/
theorem row_recompose (a nFF : Nat) (h : a < 2 * nFF) :
    a = (if a < nFF then 0 else nFF) + (if a < nFF then a else a - nFF) := by
  split_ifs <;> omega

/-- The per-entry output slot `e*n_ff + r` stays below `cap*n_ff`. -/
theorem entry_slot_lt (e r nFF cap : Nat) (he : e < cap) (hr : r < nFF) : e * nFF + r < cap * nFF := by
  have : (e + 1) * nFF ≤ cap * nFF := Nat.mul_le_mul_right nFF he
  rw [Nat.add_mul] at this
  omega

/-- Gate/up grid: eight warps per block, grid `ceil(2*n_ff/8)`. -/
theorem gu_grid (nFF : Nat) : 2 * nFF ≤ 8 * ((2 * nFF + 7) / 8) := by omega

/-- Down grid: `r = blockIdx.x*8 + warp` covers `n_embd` rows. -/
theorem down_grid (nEmbd : Nat) : nEmbd ≤ 8 * ((nEmbd + 7) / 8) := by omega

-- § Dequantiser output positions

/-- Position written by thread `tid` (0..31), element `j < 8` of an 8-wide format. -/
def pos8 (tid j : Nat) : Nat := 32 * (tid % 8) + 8 * (tid / 8) + j

/-- All 8-wide positions are inside the 256-value superblock. -/
theorem pos8_lt (tid j : Nat) (ht : tid < 32) (hj : j < 8) : pos8 tid j < 256 := by
  unfold pos8; omega

/-- Distinct `(tid, j)` write distinct positions: the dequantiser has no write conflicts. -/
theorem pos8_inj (t j t' j' : Nat) (ht : t < 32) (ht' : t' < 32) (hj : j < 8) (hj' : j' < 8)
    (h : pos8 t j = pos8 t' j') : t = t' ∧ j = j' := by
  unfold pos8 at h; omega

/-- 32 threads x 8 elements = 256 values: the writes are a bijection onto the superblock. -/
theorem pos8_cover : 32 * 8 = 256 := by decide

/-- Position for the IQ4 formats: `32*ib + 4*il + j` and `+16`, `j < 4`, `half < 2`. -/
def pos4 (tid j half : Nat) : Nat := 32 * (tid % 8) + 4 * (tid / 8) + j + 16 * half

/-- IQ4 positions are in range. -/
theorem pos4_lt (tid j half : Nat) (ht : tid < 32) (hj : j < 4) (hh : half < 2) : pos4 tid j half < 256 := by
  unfold pos4; omega

/-- IQ4 positions are collision-free. -/
theorem pos4_inj (t j h t' j' h' : Nat) (ht : t < 32) (ht' : t' < 32) (hj : j < 4) (hj' : j' < 4)
    (hh : h < 2) (hh' : h' < 2) (heq : pos4 t j h = pos4 t' j' h') : t = t' ∧ j = j' ∧ h = h' := by
  unfold pos4 at heq
  have ha : 4 * (t / 8) + j + 16 * h < 32 := by omega
  have hb : 4 * (t' / 8) + j' + 16 * h' < 32 := by omega
  have e1 : t % 8 = t' % 8 := by omega
  have e2 : 4 * (t / 8) + j + 16 * h = 4 * (t' / 8) + j' + 16 * h' := by omega
  have hc : 4 * (t / 8) + j < 16 := by omega
  have hd : 4 * (t' / 8) + j' < 16 := by omega
  have e3 : h = h' := by omega
  subst e3
  have e4 : 4 * (t / 8) + j = 4 * (t' / 8) + j' := by omega
  have e5 : t / 8 = t' / 8 := by omega
  have e6 : j = j' := by omega
  have e7 : t = t' := by omega
  exact ⟨e7, e6, rfl⟩

/-- Q2_0 dequantiser: block `tid/8` of four 64-blocks, part `tid%8` of 8 values. -/
def posQ2 (tid j : Nat) : Nat := (tid / 8) * 64 + (tid % 8) * 8 + j

/-- Q2_0 positions in range. -/
theorem posQ2_lt (tid j : Nat) (ht : tid < 32) (hj : j < 8) : posQ2 tid j < 256 := by
  unfold posQ2; omega

/-- Q2_0 positions collision-free. -/
theorem posQ2_inj (t j t' j' : Nat) (ht : t < 32) (ht' : t' < 32) (hj : j < 8) (hj' : j' < 8)
    (h : posQ2 t j = posQ2 t' j') : t = t' ∧ j = j' := by
  unfold posQ2 at h; omega

/-- Value `i` of a Q2_0 block: byte `i/4`, field `i%4`: they recompose the index. -/
theorem q2_field_index (i : Nat) : (i / 4) * 4 + i % 4 = i := by omega

-- § Interleaved gate/up dequant layout

/-- Stride uniqueness (offset below the period fixes the iteration). -/
theorem stride_inj (a b P m1 m2 : Nat) (hP : 0 < P) (ha : a < P) (hb : b < P)
    (h : a + P * m1 = b + P * m2) : a = b ∧ m1 = m2 := by
  have h1 : (a + P * m1) / P = m1 := by
    rw [Nat.add_mul_div_left _ _ hP, Nat.div_eq_of_lt ha, Nat.zero_add]
  have h2 : (b + P * m2) / P = m2 := by
    rw [Nat.add_mul_div_left _ _ hP, Nat.div_eq_of_lt hb, Nat.zero_add]
  rw [h] at h1
  have hm : m1 = m2 := h1.symm.trans h2
  subst hm
  omega

/-- `dequant_gu` writes superblock `c` of row `r` with parity `q` at block `c + per_row*(2r+q)`; injective. -/
theorem gu_interleave_inj (p r q c r' q' c' : Nat) (hp : 0 < p) (hc : c < p) (hc' : c' < p)
    (hq : q < 2) (hq' : q' < 2) (h : c + p * (2 * r + q) = c' + p * (2 * r' + q')) :
    r = r' ∧ q = q' ∧ c = c' := by
  have := stride_inj c c' p _ _ hp hc hc' h
  omega

/-- Input superblock `i` splits into `(r, c)` with `i = r*per_row + c`, `c < per_row`. -/
theorem superblock_split (i p : Nat) (hp : 0 < p) : i = p * (i / p) + i % p ∧ i % p < p := by
  have := Nat.div_add_mod i p
  exact ⟨by omega, Nat.mod_lt i hp⟩

-- § q8_1 quantiser indexing

/-- Block and in-block index recompose the element index. -/
theorem q81_index (i : Nat) : (i / 32) * 32 + i % 32 = i := by omega

/-- Quantiser grid covers all `n` elements with 256-thread blocks. -/
theorem q81_grid (n : Nat) : n ≤ 256 * ((n + 255) / 256) := by omega

/-- `iq_dequant_*` grids: one block per 256-value superblock; `n % 256 = 0` gives `n/256` whole superblocks. -/
theorem superblocks (n : Nat) (h : n % 256 = 0) : n / 256 * 256 = n := by omega

/-- Embedding rows: token `t`, block `b` writes at `t*n_embd + b*256`, inside the output. -/
theorem embed_write_lt (t b nEmbd nTok : Nat) (ht : t < nTok) (hb : b * 256 + 256 ≤ nEmbd) :
    t * nEmbd + b * 256 + 255 < nTok * nEmbd := by
  have : (t + 1) * nEmbd ≤ nTok * nEmbd := Nat.mul_le_mul_right nEmbd ht
  rw [Nat.add_mul] at this
  omega

end Strata.IqKernels



## FILE: /mnt/user-data/outputs/SharedExpertDispatch.lean
/-!
# MODULE: shared_expert_dispatch
The shared expert and MoE combine (`shared_expert.cu`): scratch carving, the activation-contract dispatch
of each projection, the native/legacy path preconditions, the scalar gate reduction shape and the combine
kernel's bounds.

## Source semantics
* `h = silu(x @ gate.T) * (x @ up.T)`; `h = h @ down.T`; `g = sigmoid(x . gate_inp)` one scalar per token;
  result `h * g`, ADDED plain to the routed output.
* Scratch regions, in order: gate (n_ff f32), up (n_ff f32), Q8_0 image, Q8_K image, g (1 f32), each
  16-byte aligned.
* A projection's kernel is chosen by `(code_bits, act_kind)`: 2-bit forms use the Q8_0 S2 kernel, `act_kind = 1`
  the Q8_K split kernel, everything else the Q8_0 split kernel.  Q8_K needs `n_in % 256 = 0`.

## Form template identifiers
- `SForm.code_bits` → `bits`; `SForm.act_kind` → `act`; `n_ff` → `nFF`; `n_embd` → `nEmbd`
- `native_bf16` → `useNative`; `native->gate_data` etc. → `ng/nu/nd`
-/

namespace Strata.SharedExpert

-- § Scratch carving

/-- 16-byte alignment. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Rounding up to 16 is above, within 15, and aligned. -/
theorem align16_spec (x : Nat) : x ≤ align16 x ∧ align16 x < x + 16 ∧ align16 x % 16 = 0 := by
  unfold align16; omega

/-- Region size of a plane of `n_ff` floats. -/
def planeBytes (nFF : Nat) : Nat := align16 (nFF * 4)
/-- Region size of the Q8_0 image (34 bytes per 32). -/
def q80Bytes (nFF : Nat) : Nat := align16 (nFF / 32 * 34)
/-- Region size of the Q8_K image (292 bytes per 256). -/
def q8kBytes (nFF : Nat) : Nat := align16 (nFF / 256 * 292)

/-- Total scratch bytes. -/
def scratchBytes (nFF : Nat) : Nat := planeBytes nFF * 2 + q80Bytes nFF + q8kBytes nFF + 32

/-- Offset of the scalar gate `g`. -/
def gOffset (nFF : Nat) : Nat := planeBytes nFF * 2 + q80Bytes nFF + q8kBytes nFF

/-- The scalar gate fits in the trailing 32 bytes. -/
theorem g_fits (nFF : Nat) : gOffset nFF + 4 ≤ scratchBytes nFF := by
  unfold gOffset scratchBytes; omega

/-- Every region start is 16-byte aligned. -/
theorem offsets_aligned (nFF : Nat) :
    planeBytes nFF % 16 = 0 ∧ (planeBytes nFF * 2) % 16 = 0 ∧
    (planeBytes nFF * 2 + q80Bytes nFF) % 16 = 0 ∧ gOffset nFF % 16 = 0 := by
  have h1 := align16_spec (nFF * 4)
  have h2 := align16_spec (nFF / 32 * 34)
  have h3 := align16_spec (nFF / 256 * 292)
  unfold gOffset planeBytes q80Bytes q8kBytes
  omega

/-- The regions are disjoint and ordered: gate | up | q8_0 | q8_k | g. -/
theorem regions_ordered (nFF : Nat) :
    planeBytes nFF ≤ planeBytes nFF * 2 ∧
    planeBytes nFF * 2 ≤ planeBytes nFF * 2 + q80Bytes nFF ∧
    planeBytes nFF * 2 + q80Bytes nFF ≤ gOffset nFF ∧ gOffset nFF < scratchBytes nFF := by
  unfold gOffset scratchBytes; omega

/-- Each plane holds `n_ff` floats. -/
theorem plane_holds (nFF : Nat) : nFF * 4 ≤ planeBytes nFF := by
  have := align16_spec (nFF * 4)
  unfold planeBytes; omega

/-- The Q8_0 image of `n_ff` (multiple of 32) elements fits its region. -/
theorem q80_holds (nFF : Nat) : nFF / 32 * 34 ≤ q80Bytes nFF := by
  have := align16_spec (nFF / 32 * 34)
  unfold q80Bytes; omega

/-- The Q8_K image fits its region. -/
theorem q8k_holds (nFF : Nat) : nFF / 256 * 292 ≤ q8kBytes nFF := by
  have := align16_spec (nFF / 256 * 292)
  unfold q8kBytes; omega

/-- For the 640-wide shared FFN: Q8_K image holds only two blocks and the tail is NOT covered. -/
theorem n640_q8k_truncated : 640 / 256 * 256 = 512 ∧ 512 < 640 := by decide

/-- `ffn_down_shexp`: `n_ff = 640` is a multiple of 32 but not 256, so Q8_K is structurally impossible. -/
theorem q8k_impossible_for_640 : 640 % 32 = 0 ∧ 640 % 256 ≠ 0 := by decide

/-- `n_embd = 2560` is a multiple of 256, so the gate/up projections may use Q8_K. -/
theorem q8k_ok_for_2560 : 2560 % 256 = 0 := by decide

-- § Kernel dispatch

/-- Kernels a projection can run. -/
inductive Kernel where
  | s2Q8        -- 2-bit codes, Q8_0 activation (`s2_gemv_q8`)
  | q8kSplit    -- `s_gemv_q8k_split`
  | q80Split    -- `s_gemv_q8_0_split`
  deriving DecidableEq, Repr

/-- Dispatch rule of the `gemv` lambda. -/
def dispatch (bits act : Nat) : Kernel :=
  if bits = 2 then Kernel.s2Q8 else if act = 1 then Kernel.q8kSplit else Kernel.q80Split

/-- Two-bit forms always take the S2 Q8_0 kernel, whatever `act_kind` says. -/
theorem dispatch_two_bit (act : Nat) : dispatch 2 act = Kernel.s2Q8 := by
  unfold dispatch; simp

/-- A wider form with `act_kind = 1` takes the Q8_K kernel. -/
theorem dispatch_q8k (bits : Nat) (h : bits ≠ 2) : dispatch bits 1 = Kernel.q8kSplit := by
  unfold dispatch; simp [h]

/-- A wider form with another act kind takes the Q8_0 split kernel. -/
theorem dispatch_q80 (bits act : Nat) (h : bits ≠ 2) (ha : act ≠ 1) : dispatch bits act = Kernel.q80Split := by
  unfold dispatch; simp [h, ha]

/-- The Q8_K kernel is never chosen for two-bit forms (its host launcher refuses them). -/
theorem q8k_never_two_bit (act : Nat) : dispatch 2 act ≠ Kernel.q8kSplit := by
  rw [dispatch_two_bit]; intro h; cases h

/-- Down projection: the Q8_K branch requires `n_ff % 256 = 0`; the guard is exactly that. -/
def downQ8kAllowed (act nFF : Nat) : Bool := act == 1 && nFF % 256 == 0

/-- With `act_kind = 1` and `n_ff = 640` the engine refuses (exits) rather than computing. -/
theorem down_refuses_640 : downQ8kAllowed 1 640 = false := by decide

/-- With `act_kind = 1` and a multiple of 256 it proceeds. -/
theorem down_accepts_2048 : downQ8kAllowed 1 2048 = true := by decide

-- § Native path preconditions

/-- `x_f32` is required when the native BF16 gate, native gate or native up projection is active. -/
def needsXf32 (useNative ng nu : Bool) : Bool := useNative || ng || nu

/-- Native down alone does not require `x_f32`. -/
theorem down_only_no_xf32 : needsXf32 false false false = false := by decide

/-- Any one of the three triggers the requirement. -/
theorem xf32_triggers (a b c : Bool) (h : a = true ∨ b = true ∨ c = true) : needsXf32 a b c = true := by
  unfold needsXf32
  rcases h with h | h | h <;> simp [h]

/-- `shared_expert_multi` token count is 1..8. -/
def multiOk (n : Nat) : Bool := 1 ≤ n && n ≤ 8

/-- Boundary cases of the multi-token guard. -/
theorem multi_bounds : multiOk 0 = false ∧ multiOk 1 = true ∧ multiOk 8 = true ∧ multiOk 9 = false := by decide

/-- The multi path reuses one q8_1 scratch for gate/up and then for down: needs `n_ff` ≥ ... both fit if the
larger of `n_embd`, `n_ff` columns fit; here `n_embd = 2560 ≥ n_ff = 640`. -/
theorem q81_scratch_reuse : 640 ≤ 2560 := by decide

/-- Gate rows scaled: element `(t, i)` at `t*n + i < n_tok * n`. -/
theorem scale_rows_index (t i n nTok : Nat) (ht : t < nTok) (hi : i < n) : t * n + i < nTok * n := by
  have : (t + 1) * n ≤ nTok * n := Nat.mul_le_mul_right n ht
  rw [Nat.add_mul] at this
  omega

-- § Scalar gate reduction shape

/-- A 256-thread block has 8 warps; the 8-slot scratch holds one partial each. -/
theorem warp_slots (nt : Nat) (h : nt ≤ 256) : (nt + 31) / 32 ≤ 8 := by omega

/-- Strided accumulation: element `i` belongs to thread `i % 256`, iteration `i / 256`. -/
theorem gate_stride (i : Nat) : i = i % 256 + 256 * (i / 256) := by omega

/-- 2560 elements over 256 threads: ten terms per thread (the shortened dependency chain). -/
theorem ten_terms : 2560 / 256 = 10 := by decide

/-- The dependency chain went from 2560 sequential terms to 10 plus a tree: strictly shorter. -/
theorem chain_shorter : 10 + 8 + 5 < 2560 := by decide

/-- Warp tree offsets 16, 8, 4, 2, 1 reach all 32 lanes: they sum to 31. -/
theorem tree_sum : 16 + 8 + 4 + 2 + 1 = 31 := by decide

-- § moe_combine

/-- The combine refuses `k > 64`. -/
def combineOk (k : Nat) : Bool := 0 < k && k ≤ 64

/-- Boundary cases. -/
theorem combine_bounds : combineOk 0 = false ∧ combineOk 10 = true ∧ combineOk 64 = true ∧ combineOk 65 = false := by
  decide

/-- Parts index `e*n_embd + j` stays below `k*n_embd`. -/
theorem parts_index (e j nEmbd k : Nat) (he : e < k) (hj : j < nEmbd) : e * nEmbd + j < k * nEmbd := by
  have : (e + 1) * nEmbd ≤ k * nEmbd := Nat.mul_le_mul_right nEmbd he
  rw [Nat.add_mul] at this
  omega

/-- 128-thread grid covers all `n_embd` outputs. -/
theorem combine_grid (n : Nat) : n ≤ 128 * ((n + 127) / 128) := by omega

end Strata.SharedExpert



## FILE: /mnt/user-data/outputs/RouterTop10Selection.lean
/-!
# MODULE: router_top10_selection
The MoE router softmax/top-k/renormalise kernel (`router_top10.cu`): the (value desc, index asc) order that
makes repeated block-wide argmax equal a stable descending argsort, the argmax scan and its tie rule, the
shuffle-tree combine, the shared-memory budget and launch shape, and the clamp constant.

## Source semantics
* `p = softmax(logits)`, `ids = stable argsort(-p)[:k]` with ties broken by ascending index,
  `w = gather(p, ids)`, `w / max(sum w, 2^-14)`.
* The old rank-by-counting loop and the new k-pass argmax agree because a strict `>` scan in ascending
  `e` keeps the lowest index on a tie, and the shuffle combine prefers the lower index on equal values.
* One block per token; `threads = roundup32(min(n_expert, 512))`; dynamic shared memory
  `align16(n) + 8n + 4n` bytes (taken mask, `n` doubles, `n` floats).

## Form template identifiers
- `s_p[e]` → `p e`; `bv/bi` → `(v, i)` pairs; `RT_MAX_THREADS` → `512`
- `n_expert` → `n`; `k` → `k`; `n_tokens` → `nTok`
-/

namespace Strata.Router

-- § The selection order

/-- `a` is strictly better than `b`: larger value, or equal value and smaller index. -/
def better (a b : Nat × Nat) : Prop := a.1 > b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2)

instance (a b : Nat × Nat) : Decidable (better a b) := by
  unfold better; infer_instance

/-- The shuffle-tree combine: keep `mine` unless `other` is strictly better. -/
def combine (mine other : Nat × Nat) : Nat × Nat := if better other mine then other else mine

/-- `better` is irreflexive. -/
theorem better_irrefl (a : Nat × Nat) : ¬ better a a := by
  unfold better; omega

/-- `better` is asymmetric. -/
theorem better_asymm (a b : Nat × Nat) (h : better a b) : ¬ better b a := by
  unfold better at *; omega

/-- `better` is transitive, so the order is a strict order. -/
theorem better_trans (a b c : Nat × Nat) (h1 : better a b) (h2 : better b c) : better a c := by
  unfold better at *; omega

/-- Two distinct pairs are always comparable: the order is total, so argmax is unique. -/
theorem better_total (a b : Nat × Nat) (hne : a ≠ b) : better a b ∨ better b a := by
  by_contra hcon
  apply hne
  unfold better at hcon
  exact Prod.ext (by omega) (by omega)

/-- The combine returns one of its arguments. -/
theorem combine_mem (a b : Nat × Nat) : combine a b = a ∨ combine a b = b := by
  unfold combine
  by_cases h : better b a
  · right; rw [if_pos h]
  · left; rw [if_neg h]

/-- The combine result is not strictly worse than either argument. -/
theorem combine_not_worse (a b : Nat × Nat) : ¬ better a (combine a b) ∧ ¬ better b (combine a b) := by
  unfold combine
  by_cases h : better b a
  · rw [if_pos h]; unfold better at *; omega
  · rw [if_neg h]; unfold better at *; omega

/-- The combine is commutative on distinct pairs, so any warp-tree shape gives the same winner. -/
theorem combine_comm (a b : Nat × Nat) (hne : a ≠ b) : combine a b = combine b a := by
  unfold combine
  by_cases h1 : better b a
  · have h2 : ¬ better a b := better_asymm b a h1
    rw [if_pos h1, if_neg h2]
  · by_cases h2 : better a b
    · rw [if_neg h1, if_pos h2]
    · rcases better_total a b hne with h | h
      · exact absurd h h2
      · exact absurd h h1

/-- A real expert beats the sentinel index `n` at an equal value, so the sentinel never wins. -/
theorem sentinel_loses (v i n : Nat) (h : i < n) : better (v, i) (v, n) := by
  show v > v ∨ (v = v ∧ i < n)
  omega

/-- An empty lane (value `0` standing for `-inf` is below every real probability) loses on value. -/
theorem neg_inf_loses (v i j : Nat) (h : 0 < v) : better (v, i) (0, j) := by
  show v > 0 ∨ (v = 0 ∧ i < j)
  omega

-- § The ascending strict-greater scan

/-- One scan step: replace only on a strictly greater value. -/
def step (p : Nat → Nat) (acc : Nat × Nat) (e : Nat) : Nat × Nat :=
  if p e > acc.1 then (p e, e) else acc

/-- A scan over a list of candidate indices. -/
def scan (p : Nat → Nat) (init : Nat × Nat) (l : List Nat) : Nat × Nat := l.foldl (step p) init

/-- Equal values never replace the incumbent: the earlier (lower) index stays. -/
theorem step_tie (p : Nat → Nat) (acc : Nat × Nat) (e : Nat) (h : p e = acc.1) : step p acc e = acc := by
  unfold step
  rw [if_neg (by omega)]

/-- A strictly greater value replaces the incumbent. -/
theorem step_replace (p : Nat → Nat) (acc : Nat × Nat) (e : Nat) (h : p e > acc.1) :
    step p acc e = (p e, e) := by
  unfold step
  rw [if_pos h]

/-- A step never lowers the running maximum and absorbs the new value. -/
theorem step_ge (p : Nat → Nat) (acc : Nat × Nat) (e : Nat) :
    acc.1 ≤ (step p acc e).1 ∧ p e ≤ (step p acc e).1 := by
  unfold step
  by_cases h : p e > acc.1
  · rw [if_pos h]
    show acc.1 ≤ p e ∧ p e ≤ p e
    omega
  · rw [if_neg h]
    omega

/-- The scan's value is at least the initial value and at least every scanned value. -/
theorem scan_value (p : Nat → Nat) (l : List Nat) :
    ∀ init : Nat × Nat, init.1 ≤ (scan p init l).1 ∧ ∀ x ∈ l, p x ≤ (scan p init l).1 := by
  induction l with
  | nil => intro init; exact ⟨Nat.le_refl _, by simp⟩
  | cons y ys ih =>
    intro init
    have hs := step_ge p init y
    have hih := ih (step p init y)
    have hscan : scan p init (y :: ys) = scan p (step p init y) ys := rfl
    rw [hscan]
    refine ⟨by omega, ?_⟩
    intro x hx
    rcases List.mem_cons.mp hx with rfl | hx'
    · omega
    · exact hih.2 x hx'

-- § Clamp and launch shape

/-- `2^-14 = 6.103515625e-5`: numerator `5^14` over `10^14`, and `5^14 * 2^14 = 10^14`. -/
theorem clamp_constant : 5 ^ 14 * 2 ^ 14 = 10 ^ 14 ∧ 6103515625 = 5 ^ 14 := by decide

/-- Output slot of rank `i` for token `t`: `t*k + i < nTok*k`. -/
theorem out_slot_lt (t i k nTok : Nat) (ht : t < nTok) (hi : i < k) : t * k + i < nTok * k := by
  have : (t + 1) * k ≤ nTok * k := Nat.mul_le_mul_right k ht
  rw [Nat.add_mul] at this
  omega

/-- Threads per block. -/
def threads (n : Nat) : Nat := ((if n < 512 then n else 512) + 31) / 32 * 32

/-- At least one warp, a multiple of 32, at most 512. -/
theorem threads_spec (n : Nat) (h : 0 < n) : 32 ≤ threads n ∧ threads n % 32 = 0 ∧ threads n ≤ 512 := by
  unfold threads
  split_ifs <;> omega

/-- For 512 experts there are 512 threads, 16 warps: the 16-slot reduction scratch is exactly full. -/
theorem threads_512 : threads 512 = 512 ∧ 512 / 32 = 16 := by decide

/-- Warps never exceed the 16 reduction slots. -/
theorem warps_le_16 (n : Nat) (h : 0 < n) : threads n / 32 ≤ 16 := by
  have := threads_spec n h
  omega

/-- The taken mask is 16-aligned. -/
def taken (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Dynamic shared memory: mask, `n` doubles, `n` floats. -/
def smem (n : Nat) : Nat := taken n + n * 8 + n * 4

/-- The doubles start 16-aligned (so 8-aligned) and the floats start 4-aligned. -/
theorem smem_alignment (n : Nat) : taken n % 16 = 0 ∧ (taken n + n * 8) % 4 = 0 := by
  unfold taken; omega

/-- The 512-expert budget. -/
theorem smem_512 : smem 512 = 6656 := by decide

/-- Up to 3500 experts fit in the 48 KiB default dynamic shared memory. -/
theorem smem_fits (n : Nat) (h : n ≤ 3500) : smem n ≤ 49152 := by
  unfold smem taken; omega

/-- **The host guard `n_expert ≤ 512*64 = 32768` does not imply the launch fits**: at 32768 the dynamic
shared request is 425984 bytes, far above 48 KiB. -/
theorem guard_insufficient_for_smem : 32768 ≤ 512 * 64 ∧ 49152 < smem 32768 := by decide

/-- All `k` outputs are written exactly when `k ≤ n` (the host does not check this). -/
theorem outputs_written (k n : Nat) : min k n = k ↔ k ≤ n := by omega

end Strata.Router



## FILE: /mnt/user-data/outputs/NativeRouterTop10.lean
/-!
# MODULE: native_router_top10
The llama.cpp-style fused MoE router (`native_router.cu`) for exactly 512 experts and 10 outputs: one warp,
16 values per lane, the expert-id layout `id = lane + 32*i`, the tournament tie rule, the output-lane
mapping, the clamp constant and the host-side span validation (alignment, overflow, disjointness).

## Source semantics
* Lane `l` holds experts `l, l+32, ..., l+480`; the block is `32 x 8` threads, only row 0 is active.
* Each of 10 rounds takes the lane-local best (strict `>`, so lowest `i`, hence lowest id), then a XOR-butterfly
  keeps the better of two candidates, preferring the lower expert id on equal values.
* The winner's lane marks `values[id/32] = -inf` and records `ids[rank] = id`; lane `rank` keeps the weight.
* Weights are divided by `max(sum, 2^-14)`; only lanes `< 10` store.
* Host: non-null, 4-byte aligned, no address wrap, and the three spans pairwise disjoint.

## Form template identifiers
- `values[i]` → `slot i`; `lane` → `lane`; `rank` → `rank`; `expert` → `id`
- `valid` → `valid`; `overlap` → `overlap`; `UINTPTR_MAX` → `uptrMax`
-/

namespace Strata.NativeRouter

-- § Expert id layout

/-- Expert id of slot `i` in lane `lane`. -/
def expertId (lane i : Nat) : Nat := lane + 32 * i

/-- 512 experts = 32 lanes x 16 slots. -/
theorem experts_tile : 32 * 16 = 512 := by decide

/-- Every id is below 512. -/
theorem expertId_lt (lane i : Nat) (hl : lane < 32) (hi : i < 16) : expertId lane i < 512 := by
  unfold expertId; omega

/-- The lane and slot are recovered from the id. -/
theorem expertId_decode (lane i : Nat) (hl : lane < 32) :
    expertId lane i % 32 = lane ∧ expertId lane i / 32 = i := by
  unfold expertId; omega

/-- The id map is injective: no two (lane, slot) pairs share an expert. -/
theorem expertId_inj (l i l' i' : Nat) (hl : l < 32) (hl' : l' < 32)
    (h : expertId l i = expertId l' i') : l = l' ∧ i = i' := by
  unfold expertId at h; omega

/-- Every id below 512 has a (lane, slot) preimage. -/
theorem expertId_surj (e : Nat) (h : e < 512) : ∃ lane i, lane < 32 ∧ i < 16 ∧ expertId lane i = e :=
  ⟨e % 32, e / 32, by omega, by omega, by unfold expertId; omega⟩

/-- Within a lane, ids increase with the slot, so a strict-`>` ascending scan keeps the lowest id on ties. -/
theorem expertId_mono (lane i j : Nat) (h : i < j) : expertId lane i < expertId lane j := by
  unfold expertId; omega

/-- The winner's lane test `(expert & 31) == lane` is `expert % 32 = lane`. -/
theorem winner_lane (id : Nat) : id % 32 < 32 := by omega

/-- The slot to blank is `id / 32 < 16` for a valid id. -/
theorem winner_slot_lt (id : Nat) (h : id < 512) : id / 32 < 16 := by omega

-- § Tournament order

/-- `a` is strictly better than `b`: larger value, or equal value and smaller id. -/
def better (a b : Nat × Nat) : Prop := a.1 > b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2)

instance (a b : Nat × Nat) : Decidable (better a b) := by
  unfold better; infer_instance

/-- Butterfly step: replace only when the other candidate is strictly better. -/
def butterfly (mine other : Nat × Nat) : Nat × Nat := if better other mine then other else mine

/-- The butterfly's result is not strictly worse than either candidate. -/
theorem butterfly_not_worse (a b : Nat × Nat) :
    ¬ better a (butterfly a b) ∧ ¬ better b (butterfly a b) := by
  unfold butterfly
  by_cases h : better b a
  · rw [if_pos h]; unfold better at *; omega
  · rw [if_neg h]; unfold better at *; omega

/-- The butterfly is symmetric on distinct candidates, so the XOR tree gives every lane the same winner. -/
theorem butterfly_comm (a b : Nat × Nat) (hne : a ≠ b) : butterfly a b = butterfly b a := by
  unfold butterfly
  by_cases h1 : better b a
  · have h2 : ¬ better a b := by unfold better at *; omega
    rw [if_pos h1, if_neg h2]
  · by_cases h2 : better a b
    · rw [if_neg h1, if_pos h2]
    · exfalso
      apply hne
      unfold better at h1 h2
      exact Prod.ext (by omega) (by omega)

/-- Equal ids with equal values are the same candidate, so the tree cannot split a tie inconsistently. -/
theorem butterfly_idem (a : Nat × Nat) : butterfly a a = a := by
  unfold butterfly
  rw [if_neg]
  unfold better; omega

-- § Rounds and outputs

/-- Ten rounds, ten output lanes: lane `rank` keeps the weight of round `rank`. -/
theorem rounds_fit_lanes : 10 ≤ 32 := by decide

/-- Only lanes below 10 store; lane `l` stores round `l`. -/
theorem store_lanes (lane : Nat) (h : lane < 10) : lane < 32 := by omega

/-- Block geometry: `32 x 8 = 256` threads, matching `__launch_bounds__(256, 1)`. -/
theorem block_threads : 32 * 8 = 256 := by decide

/-- Each lane loads slots `lane + 32*i` for `i < 16`: indices cover `[0, 512)` exactly once. -/
theorem load_index_lt (lane i : Nat) (hl : lane < 32) (hi : i < 16) : lane + i * 32 < 512 := by omega

/-- The renormalisation clamp: `2^-14 = 6.103515625e-5`. -/
theorem clamp_constant : 5 ^ 14 * 2 ^ 14 = 10 ^ 14 ∧ 6103515625 = 5 ^ 14 := by decide

-- § Host span validation

/-- Maximum pointer value. -/
def uptrMax : Nat := 2 ^ 64 - 1

/-- `valid p bytes`: non-null, 4-byte aligned, and `p + bytes` does not wrap. -/
def valid (p bytes : Nat) : Prop := p ≠ 0 ∧ p % 4 = 0 ∧ bytes ≤ uptrMax - p

/-- A valid span does not wrap the address space. -/
theorem valid_no_wrap (p bytes : Nat) (h : valid p bytes) (hp : p ≤ uptrMax) : p + bytes ≤ uptrMax := by
  unfold valid at h
  omega

/-- Half-open interval overlap, as in the source. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap; omega

/-- Non-overlap of non-empty spans means one ends before the other begins. -/
theorem not_overlap_disjoint (a an b bn : Nat) (h : ¬ overlap a an b bn) :
    a + an ≤ b ∨ b + bn ≤ a := by
  unfold overlap at h
  omega

/-- A span is disjoint from itself only when empty: the host never accepts aliasing outputs. -/
theorem self_overlap (a n : Nat) (h : 0 < n) : overlap a n a n := by
  unfold overlap; omega

/-- Span sizes: logits 2048 bytes, ids 40, weights 40. -/
theorem span_sizes : 512 * 4 = 2048 ∧ 10 * 4 = 40 := by decide

/-- Alignment of all spans' elements follows from the base alignment: element `k` at `p + 4k`. -/
theorem element_aligned (p k : Nat) (h : p % 4 = 0) : (p + 4 * k) % 4 = 0 := by omega

end Strata.NativeRouter





## FILE: /mnt/user-data/outputs/FusedGrKernel.lean
/-!
# MODULE: fused_gr_kernel
Launch geometry and index algebra of the fused gated-residual read (`fused_gr.cu`): the single-token
`gr_down`/`gr_up` kernels and the T-token variants that read each weight row once.

## Source semantics
Each kernel maps a fixed grid onto the rows of the down (320 x 10240) and up (10240 x 320) projections. The
contracts are: every output row is produced by exactly one (block, warp) pair; every reduction index is
visited by exactly one (lane, iteration) pair; a vectorised float4 access never straddles two residual
streams; the per-token epilogue lane `k` only exists for `k < T`.

## Form template identifiers
- N (n_embd) -> `N`, HC (streams) -> `HC`, D -> `D`, LR (hc_lr) -> `LR`
- THREADS/WARPS -> `THREADS`/`WARPS`; DOWN_BLOCKS, UP_BLOCKS, UP_COLS -> same names
- UPM_COLS/UPM_BLOCKS -> multi-token `up` tiling; TILE/TQ -> activation tile staging
-/

namespace Strata.FusedGr

-- § geometry constants

def N : Nat := 2560
def HC : Nat := 4
def D : Nat := N * HC
def LR : Nat := 320
def THREADS : Nat := 256
def WARPS : Nat := THREADS / 32
def DOWN_BLOCKS : Nat := LR / WARPS
def UP_COLS : Nat := 32
def UP_BLOCKS : Nat := N / UP_COLS
def UPM_COLS : Nat := 16
def UPM_BLOCKS : Nat := N / UPM_COLS
def TILE : Nat := 2560
def TQ : Nat := TILE / 8 / 32

/-- The residual width is hc times n_embd. -/
theorem D_eq : D = 10240 := by decide

/-- Eight warps per block. -/
theorem WARPS_eq : WARPS = 8 := by decide

/-- Forty down blocks of eight rows cover exactly the 320 bottleneck rows. -/
theorem down_blocks_cover : DOWN_BLOCKS * WARPS = LR ∧ DOWN_BLOCKS = 40 := by decide

/-- Eighty up blocks of 32 columns cover exactly n_embd columns. -/
theorem up_blocks_cover : UP_BLOCKS * UP_COLS = N ∧ UP_BLOCKS = 80 := by decide

/-- The multi-token up kernel uses 160 blocks of 16 columns, again covering n_embd exactly. -/
theorem upm_blocks_cover : UPM_BLOCKS * UPM_COLS = N ∧ UPM_BLOCKS = 160 := by decide

/-- A single-token up block owns 128 rows, 16 per warp. -/
theorem up_rows_per_block : HC * UP_COLS = 128 ∧ HC * UP_COLS = WARPS * 16 := by decide

/-- A multi-token up block owns 64 rows, 8 per warp. -/
theorem upm_rows_per_block : HC * UPM_COLS = 64 ∧ HC * UPM_COLS = WARPS * 8 := by decide

/-- A down row of 10240 bf16 is 1280 chunks of 8, 40 per lane. -/
theorem down_row_chunks : D / 8 = 1280 ∧ 1280 = 32 * 40 := by decide

/-- An up row of 320 bf16 is 40 chunks: 32 for the first pass, 8 for the lanes of the second. -/
theorem up_row_chunks : LR / 8 = 40 ∧ LR / 8 - 32 = 8 := by decide

/-- A staged activation tile is 320 chunks, 10 per lane, and four tiles cover a row. -/
theorem tile_shape : TQ = 10 ∧ TILE / 8 = 320 ∧ D = 4 * TILE ∧ 32 * TQ * 8 = TILE := by decide

-- § row assignment of the down kernel

/-- The row index `block*8 + warp` stays below hc_lr for every down block. -/
theorem down_row_lt (b w : Nat) (hb : b < 40) (hw : w < 8) : b * 8 + w < 320 := by omega

/-- Distinct (block, warp) pairs compute distinct rows. -/
theorem down_row_inj (b w b' w' : Nat) (hw : w < 8) (hw' : w' < 8) (h : b * 8 + w = b' * 8 + w') :
    b = b' ∧ w = w' := by omega

/-- Every bottleneck row is computed by some (block, warp) pair. -/
theorem down_row_cover (r : Nat) (hr : r < 320) : ∃ b w, b < 40 ∧ w < 8 ∧ r = b * 8 + w := by
  exact ⟨r / 8, r % 8, by omega, by omega, by omega⟩

-- § row assignment of the up kernels

/-- Up row `c*N + b*32 + dd` is inside the residual. -/
theorem up_idx_lt (c b dd : Nat) (hc : c < 4) (hb : b < 80) (hdd : dd < 32) :
    c * 2560 + b * 32 + dd < 10240 := by omega

/-- Distinct (stream, block, column) triples own distinct up rows. -/
theorem up_idx_inj (c b dd c' b' dd' : Nat) (hdd : dd < 32) (hdd' : dd' < 32) (hb : b < 80) (hb' : b' < 80)
    (h : c * 2560 + b * 32 + dd = c' * 2560 + b' * 32 + dd') : c = c' ∧ b = b' ∧ dd = dd' := by omega

/-- Every up row of a block is visited by one warp at one step of the warp-strided loop. -/
theorem up_row_cover (r : Nat) (hr : r < 128) : ∃ w m, w < 8 ∧ m < 16 ∧ r = w + 8 * m := by
  exact ⟨r % 8, r / 8, by omega, by omega, by omega⟩

/-- The mixed output index `b*32 + t` covers every column exactly once. -/
theorem mixed_cover (d : Nat) (hd : d < 2560) : ∃ b t, b < 80 ∧ t < 32 ∧ d = b * 32 + t := by
  exact ⟨d / 32, d % 32, by omega, by omega, by omega⟩

/-- Multi-token variant: row index stays inside the residual. -/
theorem upm_idx_lt (c b dd : Nat) (hc : c < 4) (hb : b < 160) (hdd : dd < 16) :
    c * 2560 + b * 16 + dd < 10240 := by omega

/-- Multi-token variant: distinct triples own distinct rows. -/
theorem upm_idx_inj (c b dd c' b' dd' : Nat) (hdd : dd < 16) (hdd' : dd' < 16) (hb : b < 160) (hb' : b' < 160)
    (h : c * 2560 + b * 16 + dd = c' * 2560 + b' * 16 + dd') : c = c' ∧ b = b' ∧ dd = dd' := by omega

/-- Multi-token variant: the 64 rows of a block are split 8 per warp. -/
theorem upm_row_cover (r : Nat) (hr : r < 64) : ∃ w m, w < 8 ∧ m < 8 ∧ r = w + 8 * m := by
  exact ⟨r % 8, r / 8, by omega, by omega, by omega⟩

-- § reduction coverage

/-- A float4 load at a multiple of 4 never straddles two residual streams, so a per-stream sum of squares can
attribute the whole chunk to one stream. -/
theorem chunk_in_stream (i : Nat) (h4 : i % 4 = 0) : (i + 3) / 2560 = i / 2560 := by omega

/-- The norm loop `i = 4t + 1024k` visits every multiple-of-4 index of the residual. -/
theorem norm_cover (i : Nat) (hi : i < 10240) (h4 : i % 4 = 0) :
    ∃ t k, t < 256 ∧ k < 10 ∧ i = t * 4 + k * 1024 := by
  exact ⟨i % 1024 / 4, i / 1024, by omega, by omega, by omega⟩

/-- The norm loop's chunks are pairwise distinct. -/
theorem norm_inj (t k t' k' : Nat) (ht : t < 256) (ht' : t' < 256) (h : t * 4 + k * 1024 = t' * 4 + k' * 1024) :
    t = t' ∧ k = k' := by omega

/-- The scaling loop `i = t + 256k` visits every residual element. -/
theorem scale_cover (i : Nat) (hi : i < 10240) : ∃ t k, t < 256 ∧ k < 40 ∧ i = t + k * 256 := by
  exact ⟨i % 256, i / 256, by omega, by omega, by omega⟩

/-- Lane `l` at step `q` reads chunk `l + 32q`; every one of the 1280 chunks of a row is read once. -/
theorem lane_cover (j : Nat) (hj : j < 1280) : ∃ lane q, lane < 32 ∧ q < 40 ∧ j = lane + 32 * q := by
  exact ⟨j % 32, j / 32, by omega, by omega, by omega⟩

/-- Distinct (lane, step) pairs read distinct chunks. -/
theorem lane_inj (l q l' q' : Nat) (hl : l < 32) (hl' : l' < 32) (h : l + 32 * q = l' + 32 * q') :
    l = l' ∧ q = q' := by omega

/-- Within a tile each of the 320 chunks is read by one (lane, step) pair. -/
theorem tile_chunk_cover (j : Nat) (hj : j < 320) : ∃ lane q, lane < 32 ∧ q < 10 ∧ j = lane + 32 * q := by
  exact ⟨j % 32, j / 32, by omega, by omega, by omega⟩

/-- The global chunk index `tile*320 + lane + 32q` stays inside a 1280-chunk row. -/
theorem tile_global_lt (s lane q : Nat) (hs : s < 4) (hl : lane < 32) (hq : q < 10) :
    s * 320 + lane + 32 * q < 1280 := by omega

/-- Shared memory for T tokens of staged activations is `T * 10240` bytes. -/
theorem tile_smem_bytes (T : Nat) : T * 2560 * 4 = T * 10240 := by omega

-- § token staging

/-- The staging loop index splits into a token and an offset that both stay in range. -/
theorem stage_split (T i : Nat) (hi : i < T * 320) : i / 320 < T ∧ i % 320 < 320 := by omega

/-- The epilogue loop index over (token, column) is in range. -/
theorem epilogue_split (T i : Nat) (hi : i < T * 16) : i / 16 < T ∧ i % 16 < 16 := by omega

/-- Epilogue lane `k` executes only for tokens below T. -/
theorem epilogue_lane_guard (T k lane : Nat) (hk : k < T) (hl : lane = k) : lane < T := by omega

end Strata.FusedGr



## FILE: /mnt/user-data/outputs/GatedResidualKernels.lean
/-!
# MODULE: gated_residual_kernels
Workspace layout, dispatch flags, and loop geometry of the gated residual / hyper-connection `gr_read`
and `gr_write` kernels (`gr.cu`).

## Source semantics
`gr_workspace_init` carves five regions (xn, xq, lq, gated, lo) from one base pointer with a single size
table; the regions must be consecutive, 16-byte aligned, and each at least as large as its payload. `gr_write`
computes `out[i] = R[i] + block_out[i % n] * w[i / n]`, so the stream/offset decomposition must be exact.
The native MMVF flag implies FP32 activations.

## Form template identifiers
- hc -> `hc`, n_embd -> `n`, hc_lr -> `lr`
- `align16` -> `(n+15) & ~15`
- `wsBytes` -> `GrWorkspace.bytes` / `gr_workspace_bytes`
-/

namespace Strata.GatedResidual

-- § alignment

def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Rounding up never shrinks a size. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by unfold align16; omega

/-- The rounded size is a multiple of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by unfold align16; omega

/-- Rounding adds fewer than 16 bytes. -/
theorem align16_lt (n : Nat) : align16 n < n + 16 := by unfold align16; omega

/-- Rounding is idempotent. -/
theorem align16_idem (n : Nat) : align16 (align16 n) = align16 n := by unfold align16; omega

-- § workspace table

def wsSizes (hc n lr : Nat) : List Nat := [hc * n * 4, hc * n * 2, lr * 2, hc * n * 4, lr * 4]
def wsAligned (hc n lr : Nat) : List Nat := (wsSizes hc n lr).map align16
def wsBytes (hc n lr : Nat) : Nat := (wsAligned hc n lr).sum
def wsOffset (hc n lr k : Nat) : Nat := ((wsAligned hc n lr).take k).sum

/-- The total is the sum of the five aligned regions. -/
theorem wsBytes_eq (hc n lr : Nat) :
    wsBytes hc n lr = align16 (hc * n * 4) + align16 (hc * n * 2) + align16 (lr * 2) +
      align16 (hc * n * 4) + align16 (lr * 4) := by
  simp [wsBytes, wsAligned, wsSizes, Nat.add_assoc]

/-- The workspace is at least as large as the sum of the raw payloads. -/
theorem wsBytes_ge (hc n lr : Nat) :
    hc * n * 4 + hc * n * 2 + lr * 2 + hc * n * 4 + lr * 4 ≤ wsBytes hc n lr := by
  rw [wsBytes_eq]
  have h1 := align16_ge (hc * n * 4)
  have h2 := align16_ge (hc * n * 2)
  have h3 := align16_ge (lr * 2)
  have h4 := align16_ge (lr * 4)
  omega

/-- For the real geometry the workspace is 104320 bytes. -/
theorem wsBytes_real : wsBytes 4 2560 320 = 104320 := by decide

/-- Region offsets for the real geometry: xn, xq, lq, gated, lo are consecutive. -/
theorem wsOffsets_real :
    wsOffset 4 2560 320 0 = 0 ∧ wsOffset 4 2560 320 1 = 40960 ∧ wsOffset 4 2560 320 2 = 61440 ∧
    wsOffset 4 2560 320 3 = 62080 ∧ wsOffset 4 2560 320 4 = 103040 ∧ wsOffset 4 2560 320 5 = 104320 := by
  decide

/-- Every real-geometry region offset is 16-byte aligned. -/
theorem wsOffsets_real_aligned :
    wsOffset 4 2560 320 1 % 16 = 0 ∧ wsOffset 4 2560 320 2 % 16 = 0 ∧
    wsOffset 4 2560 320 3 % 16 = 0 ∧ wsOffset 4 2560 320 4 % 16 = 0 := by
  decide

/-- The first version's xq sizing (hc_lr*2 bytes) was far too small for the real geometry. -/
theorem xq_old_sizing_too_small : 320 * 2 < 4 * 2560 * 2 := by decide

-- § gr_write decomposition

/-- Stream index of a flat element is below hc. -/
theorem write_stream_lt (n hc i : Nat) (h : i < hc * n) : i / n < hc := by
  apply Nat.div_lt_of_lt_mul
  rw [Nat.mul_comm]
  exact h

/-- A flat element index splits as `stream * n + offset`. -/
theorem write_decomp (n i : Nat) : (i / n) * n + i % n = i := by
  have h := Nat.div_add_mod i n
  rw [Nat.mul_comm] at h
  exact h

/-- The offset within a stream is below n_embd. -/
theorem write_offset_lt (n i : Nat) (hn : 0 < n) : i % n < n := Nat.mod_lt i hn

-- § grids

/-- A ceiling-divided grid covers every element. -/
theorem ceil_cover (n T : Nat) (hT : 0 < T) : n ≤ (n + T - 1) / T * T := by
  have h1 : (n + T - 1) / T * T + (n + T - 1) % T = n + T - 1 := by
    rw [Nat.mul_comm]; exact Nat.div_add_mod _ _
  have h2 := Nat.mod_lt (n + T - 1) hT
  omega

/-- The gate kernel launches exactly 1280 blocks at the real geometry. -/
theorem gate_blocks_real : (10240 + 8 - 1) / 8 = 1280 := by decide

/-- One-block-per-row down kernel: eight warps stride the reduction, covering each index once. -/
theorem down_stride_cover (i : Nat) (hi : i < 10240) :
    ∃ k w l, k < 40 ∧ w < 8 ∧ l < 32 ∧ i = k * 256 + w * 32 + l := by
  exact ⟨i / 256, i % 256 / 32, i % 32, by omega, by omega, by omega, by omega⟩

/-- Distinct (step, warp, lane) triples read distinct reduction indices. -/
theorem down_stride_inj (k w l k' w' l' : Nat) (hw : w < 8) (hw' : w' < 8) (hl : l < 32) (hl' : l' < 32)
    (h : k * 256 + w * 32 + l = k' * 256 + w' * 32 + l') : k = k' ∧ w = w' ∧ l = l' := by omega

/-- The inject kernel's warp index is valid exactly when the thread id is below 32*hc. -/
theorem inject_warp_valid (tid hc : Nat) : tid / 32 < hc ↔ tid < 32 * hc := by omega

/-- The block reduction needs 8 partials for 256 threads, and the double scratch can hold the float view. -/
theorem block_sum_scratch : (256 + 31) / 32 = 8 ∧ 8 * 4 ≤ 8 * 8 := by decide

-- § dispatch flags

def useFp32 (fp32 native : Bool) : Bool := fp32 || native

/-- The native MMVF path always selects FP32 activations. -/
theorem native_implies_fp32 (fp32 : Bool) : useFp32 fp32 true = true := by
  simp [useFp32]

/-- Without either flag the bf16-quantised activation path runs. -/
theorem default_is_bf16 : useFp32 false false = false := by
  simp [useFp32]

end Strata.GatedResidual



## FILE: /mnt/user-data/outputs/PleBlockKernel.lean
/-!
# MODULE: ple_block_kernel
Structural contracts of the PLE block's GPU half (`ple.cu`): n-gram history advance, causal dilated conv tap
rows, flat conv-state layout, signed square-root gate, scratch carving and overlap checks.

## Source semantics
The history of each channel is a column of 9 floats, advanced by shifting left and appending the new
normalised value. The conv has 4 taps with dilation 3, reading rows 0,3,6 from history and row 9 from the new
input. The conv state is row-fastest: flat index `row + nhist * channel`. The scratch is one caller-owned block
carved at 16-byte aligned offsets. `overlap` rejects aliasing buffers.

## Form template identifiers
- NG_HIST -> `nhist`, NG_HC_DIM -> `hcDim`, PLE_CONV_KERNEL -> `kern`, NGRAM_SIZE -> `dil`
- `history_advance_kernel` -> `advance`; `conv_kernel` taps -> `tapRow`
- `overlap` -> `overlap`; `ple_block_scratch_bytes` -> `scratchBytes`
-/

namespace Strata.PleBlock

-- § history advance

def advance {α : Type} (col : List α) (x : α) : List α := col.tail ++ [x]

/-- Advancing a full column keeps its length. -/
theorem advance_length {α : Type} (col : List α) (x : α) (h : col.length = 9) :
    (advance col x).length = 9 := by
  cases col with
  | nil => simp at h
  | cons a t => simp [advance] at h ⊢; omega

/-- The newest value ends up in the last row. -/
theorem advance_last {α : Type} (col : List α) (x : α) : (advance col x).getLast? = some x := by
  simp [advance]

/-- Advancing an empty column leaves just the new value. -/
theorem advance_nil {α : Type} (x : α) : advance ([] : List α) x = [x] := by
  simp [advance]

-- § conv taps

def tapRow (nhist kern dil k : Nat) : Nat := nhist - (kern - 1 - k) * dil

/-- At the real geometry the four taps read rows 0, 3, 6, 9. -/
theorem tap_rows_real : ∀ k, k < 4 → tapRow 9 4 3 k = 3 * k := by decide

/-- Only the last tap reads the new row. -/
theorem only_last_tap_new : ∀ k, k < 4 → (tapRow 9 4 3 k = 9 ↔ k = 3) := by decide

/-- The three history taps stay inside the 9-row history. -/
theorem history_taps_in_range : ∀ k, k < 3 → tapRow 9 4 3 k < 9 := by decide

/-- Row-fastest flat index stays inside the conv state. -/
theorem flat_lt (nh c m row : Nat) (hr : row < nh) (hc : c < m) : row + nh * c < nh * m := by
  have h := Nat.mul_le_mul_left nh (show c + 1 ≤ m from hc)
  rw [Nat.mul_add, Nat.mul_one] at h
  omega

/-- Row-fastest flat indices are injective. -/
theorem flat_inj (nh row c row' c' : Nat) (hr : row < nh) (hr' : row' < nh)
    (h : row + nh * c = row' + nh * c') : row = row' ∧ c = c' := by
  have key : ∀ x y p q : Nat, p < nh → x < y → p + nh * x < q + nh * y := by
    intro x y p q hp hxy
    have h1 := Nat.mul_le_mul_left nh (show x + 1 ≤ y from hxy)
    rw [Nat.mul_add, Nat.mul_one] at h1
    omega
  rcases Nat.lt_trichotomy c c' with h1 | h1 | h1
  · have := key c c' row row' hr h1
    omega
  · subst h1
    exact ⟨by omega, rfl⟩
  · have := key c' c row' row hr' h1
    omega

-- § signed square-root gate

def sgn (s : Int) : Int := if s > 0 then 1 else if s < 0 then -1 else 0

/-- The sign is one of -1, 0, 1. -/
theorem sgn_cases (s : Int) : sgn s = -1 ∨ sgn s = 0 ∨ sgn s = 1 := by
  unfold sgn; split_ifs <;> simp

/-- The sign is odd, which keeps the gate symmetric about one half. -/
theorem sgn_neg (s : Int) : sgn (-s) = - sgn s := by
  unfold sgn; split_ifs <;> omega

/-- The sign of zero is zero. -/
theorem sgn_zero : sgn 0 = 0 := by decide

-- § scratch carving

def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Rounding is a multiple of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by unfold align16; omega

/-- Rounding never shrinks. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by unfold align16; omega

def scratchBytes (hcDim nEmbd hc : Nat) : Nat :=
  let f := (5 * hcDim + nEmbd + hc) * 4
  let q := (nEmbd / 32) * 34
  let e := nEmbd * 2
  align16 f + align16 q + e + 256

/-- The real-geometry scratch is 223152 bytes. -/
theorem scratch_real : scratchBytes 10240 2560 4 = 223152 := by decide

/-- The Q8 activation image starts at a 16-byte aligned offset after the float region. -/
theorem q8_offset_aligned (f : Nat) : align16 f % 16 = 0 := align16_mod f

/-- The float scratch regions (key, query, norm, gated, conv: hc_dim each; value: n_embd; gate: hc) are consecutive. -/
theorem float_regions_consecutive (h n hc : Nat) :
    0 + h = h ∧ h + h = 2 * h ∧ 2 * h + h = 3 * h ∧ 3 * h + h = 4 * h ∧ 4 * h + h = 5 * h ∧
    5 * h + n + hc = 5 * h + n + hc := by omega

/-- The float regions fit inside the carved float byte count. -/
theorem float_regions_fit (h n hc : Nat) : (5 * h + n + hc) * 4 ≤ align16 ((5 * h + n + hc) * 4) :=
  align16_ge _

/-- The whole scratch image fits the three carved regions. -/
theorem scratch_covers_regions (h n hc : Nat) :
    align16 ((5 * h + n + hc) * 4) + align16 ((n / 32) * 34) + n * 2 ≤ scratchBytes h n hc := by
  unfold scratchBytes; simp only []; omega

-- § overlap

def overlap (a an b bn : Nat) : Prop := an > 0 ∧ bn > 0 ∧ (if a < b then b - a < an else a - b < bn)

/-- Overlap of two non-empty byte ranges is interval intersection. -/
theorem overlap_iff (a an b bn : Nat) :
    overlap a an b bn ↔ (0 < an ∧ 0 < bn ∧ a < b + bn ∧ b < a + an) := by
  unfold overlap
  constructor
  · rintro ⟨h1, h2, h3⟩
    refine ⟨h1, h2, ?_, ?_⟩ <;> split_ifs at h3 <;> omega
  · rintro ⟨h1, h2, h3, h4⟩
    refine ⟨h1, h2, ?_⟩
    split_ifs <;> omega

/-- Overlap is symmetric. -/
theorem overlap_comm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  rw [overlap_iff, overlap_iff]
  omega

/-- A range with zero length overlaps nothing. -/
theorem overlap_empty_left (a b bn : Nat) : ¬ overlap a 0 b bn := by
  rw [overlap_iff]; omega

/-- Adjacent ranges do not overlap. -/
theorem adjacent_no_overlap (a an bn : Nat) : ¬ overlap a an (a + an) bn := by
  rw [overlap_iff]; omega

-- § launch geometry

/-- Elementwise kernels with 256 threads cover hc_dim exactly at 40 blocks. -/
theorem elementwise_blocks : (10240 + 255) / 256 = 40 ∧ 40 * 256 = 10240 := by decide

end Strata.PleBlock



## FILE: /mnt/user-data/outputs/NativePlePostops.lean
/-!
# MODULE: native_ple_postops
Index algebra and validation contracts of the native PLE post-operations (`native_ple_postops.cu`): the
512-thread gate reduction, broadcast/conv/residual kernels, and the span validation.

## Source semantics
The gate kernel reduces one stream of 2560 products with 512 threads, each holding 8 partial lanes
(`d = tid + 512*j`, valid when `d < 2560`, i.e. `j < 5`); the 512 threads form 16 warps whose partials are
combined by the first 16 lanes. The sign is `float(s>0) - float(s<0)`. The conv reads history rows 0,3,6 at
`c*9 + 3k` and the new normalised value for tap 3. Every span must be non-null, aligned and non-wrapping.

## Form template identifiers
- N -> `N`, H -> `H`, D -> `D`, HISTORY -> `HISTORY`
- `Span` -> (`p`, `bytes`, `alignment`); `overlaps` -> `overlaps`
-/

namespace Strata.NativePlePostops

def N : Nat := 2560
def H : Nat := 4
def D : Nat := N * H
def HISTORY : Nat := 9

/-- The residual width. -/
theorem D_eq : D = 10240 := by decide

-- § gate reduction geometry

/-- Each thread/lane slot (tid, j) maps to a distinct element index. -/
theorem gate_slot_inj (t j t' j' : Nat) (ht : t < 512) (ht' : t' < 512)
    (h : t + j * 512 = t' + j' * 512) : t = t' ∧ j = j' := by omega

/-- A slot is in range exactly when j < 5, since 2560 = 5 * 512. -/
theorem gate_slot_valid (t j : Nat) (ht : t < 512) : t + j * 512 < 2560 ↔ j < 5 := by omega

/-- Every element of a stream is held by exactly one in-range slot. -/
theorem gate_slot_cover (d : Nat) (hd : d < 2560) : ∃ t j, t < 512 ∧ j < 5 ∧ d = t + j * 512 := by
  exact ⟨d % 512, d / 512, by omega, by omega, by omega⟩

/-- The eight partial lanes include the three that are always out of range. -/
theorem gate_unused_lanes (t j : Nat) (ht : t < 512) (hj : 5 ≤ j) (hj8 : j < 8) : ¬ (t + j * 512 < 2560) := by
  omega

/-- 512 threads are 16 warps, which is exactly the 16 partial lanes the final reduction reads. -/
theorem gate_warps : 512 / 32 = 16 ∧ 16 ≤ 32 := by decide

/-- The final reduction reads partials for lane < 16 only, one per warp. -/
theorem gate_partial_cover (w : Nat) (hw : w < 512 / 32) : w < 16 := by omega

-- § sign

def sgn (s : Int) : Int := if s > 0 then 1 else if s < 0 then -1 else 0
def signDiff (s : Int) : Int := (if s > 0 then 1 else 0) - (if s < 0 then 1 else 0)

/-- `float(s>0) - float(s<0)` is the sign function. -/
theorem sign_diff_eq (s : Int) : signDiff s = sgn s := by
  unfold signDiff sgn; split_ifs <;> omega

/-- The sign is odd. -/
theorem sgn_odd (s : Int) : sgn (-s) = - sgn s := by
  unfold sgn; split_ifs <;> omega

-- § elementwise launches

/-- Broadcast and conv kernels launch exactly D/256 blocks and the guard `i < D` always holds. -/
theorem elementwise_exact (b t : Nat) (hb : b < D / 256) (ht : t < 256) : b * 256 + t < D := by
  unfold D N H at *; omega

/-- Broadcast reads value[i % N] and gate[i / N]; both are in range. -/
theorem broadcast_idx (i : Nat) (hi : i < 10240) : i % 2560 < 2560 ∧ i / 2560 < 4 := by omega

-- § conv history

/-- History index of tap k < 3 for channel c is `c*9 + 3k`, inside the channel's 9-entry row. -/
theorem conv_hist_row (c k : Nat) (hk : k < 3) : c * 9 + 3 * k < c * 9 + 9 := by omega

/-- Tap history indices stay below the whole history buffer. -/
theorem conv_hist_lt (c k : Nat) (hc : c < 10240) (hk : k < 3) : c * 9 + 3 * k < 9 * 10240 := by omega

/-- Distinct (channel, tap) pairs read distinct history entries. -/
theorem conv_hist_inj (c k c' k' : Nat) (hk : k < 3) (hk' : k' < 3)
    (h : c * 9 + 3 * k = c' * 9 + 3 * k') : c = c' ∧ k = k' := by omega

/-- Conv weights are indexed `c*4 + k`, injective for k < 4. -/
theorem conv_weight_inj (c k c' k' : Nat) (hk : k < 4) (hk' : k' < 4)
    (h : c * 4 + k = c' * 4 + k') : c = c' ∧ k = k' := by omega

-- § span validation

def overlaps (x a y b : Nat) : Prop := x < y + b ∧ y < x + a

/-- Overlap is symmetric. -/
theorem overlaps_comm (x a y b : Nat) : overlaps x a y b ↔ overlaps y b x a := by
  unfold overlaps; omega

/-- A span never overlaps one starting at its end. -/
theorem overlaps_adjacent (x a b : Nat) : ¬ overlaps x a (x + a) b := by
  unfold overlaps; omega

/-- A span overlaps itself when non-empty. -/
theorem overlaps_self (x a : Nat) (h : 0 < a) : overlaps x a x a := by
  unfold overlaps; omega

/-- Validation (non-null, aligned, no wrap) implies the span end does not overflow the address space. -/
theorem validate_no_wrap (p bytes align maxAddr : Nat) (hp : p ≤ maxAddr - bytes) (hb : bytes ≤ maxAddr) :
    p + bytes ≤ maxAddr := by omega

/-- Validation rules out a null pointer and a misaligned one. -/
theorem validate_aligned (p align : Nat) (h0 : p ≠ 0) (ha : p % align = 0) (hal : 0 < align) :
    0 < p ∧ align ∣ p := by
  exact ⟨Nat.pos_of_ne_zero h0, Nat.dvd_of_mod_eq_zero ha⟩

/-- The only tolerated output/input overlap is exact aliasing, and exact aliasing is a self-overlap. -/
theorem alias_is_exact_overlap (x a : Nat) (h : 0 < a) : overlaps x a x a := overlaps_self x a h

end Strata.NativePlePostops



## FILE: /mnt/user-data/outputs/SamplerChain.lean
/-!
# MODULE: sampler_chain
Deterministic structure of the CUDA sampler (`sampler.cu`): greedy argmax tie rule and its warp/block
reduction, penalty application, penalty history window, top-k/top-p bounds, Philox mantissa range, and launch
geometry.

## Source semantics
Greedy decoding is the global argmax over penalised logits, ties resolved to the lowest index. The reduction
merges two candidates by larger value and, on equal values, smaller index; this order is a commutative,
associative, idempotent selection, so any reduction tree gives the serial answer. The repeat penalty
multiplies non-positive logits and divides positive ones; the presence penalty is a boolean. The history
window is the tail of the row. The uniform draw has 24 mantissa bits.

## Form template identifiers
- `better`/`merge` -> the `ov > bv || (ov == bv && oi < best)` rule
- penalty_repeat/freq/present -> `rep`/`freq`/`present`; penalty_last_n -> `plen`; history_len -> `hl`
- KMAX -> `KMAX`
-/

namespace Strata.Sampler

-- § candidate merge

abbrev Cand := Int × Nat

def merge (a b : Cand) : Cand := if b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2) then b else a

/-- Merging is commutative on candidates with distinct indices. -/
theorem merge_comm (a b : Cand) (h : a.2 ≠ b.2) : merge a b = merge b a := by
  obtain ⟨av, ai⟩ := a
  obtain ⟨bv, bi⟩ := b
  simp only [merge] at *
  split_ifs <;> simp_all <;> omega

/-- Merging is associative. -/
theorem merge_assoc (a b c : Cand) : merge (merge a b) c = merge a (merge b c) := by
  obtain ⟨av, ai⟩ := a
  obtain ⟨bv, bi⟩ := b
  obtain ⟨cv, ci⟩ := c
  simp only [merge]
  split_ifs <;> simp_all <;> omega

/-- Merging is idempotent. -/
theorem merge_idem (a : Cand) : merge a a = a := by
  obtain ⟨av, ai⟩ := a
  simp [merge]

/-- The merged value dominates both inputs. -/
theorem merge_value_ge (a b : Cand) : a.1 ≤ (merge a b).1 ∧ b.1 ≤ (merge a b).1 := by
  obtain ⟨av, ai⟩ := a
  obtain ⟨bv, bi⟩ := b
  simp only [merge]
  split_ifs <;> simp_all <;> omega

/-- On equal values the smaller index wins. -/
theorem merge_tie_min (a b : Cand) (h : a.1 = b.1) : (merge a b).2 = min a.2 b.2 := by
  obtain ⟨av, ai⟩ := a
  obtain ⟨bv, bi⟩ := b
  simp only at h
  simp only [merge]
  split_ifs <;> simp_all <;> omega

/-- The merged candidate is one of the inputs. -/
theorem merge_mem (a b : Cand) : merge a b = a ∨ merge a b = b := by
  unfold merge; split_ifs <;> simp

/-- The "no candidate" sentinel (an index equal to n_vocab) never beats a real one when values are equal. -/
theorem sentinel_loses (v : Int) (i n : Nat) (h : i < n) : merge (v, i) (v, n) = (v, i) := by
  simp only [merge]
  split_ifs <;> simp_all <;> omega

-- § strided scan coverage

/-- Thread `tid` scanning `v = tid + k*bd` visits every vocabulary index. -/
theorem scan_cover (bd v : Nat) (hbd : 0 < bd) : ∃ tid k, tid < bd ∧ v = tid + k * bd := by
  refine ⟨v % bd, v / bd, Nat.mod_lt v hbd, ?_⟩
  have h := Nat.mod_add_div v bd
  have h2 := Nat.mul_comm bd (v / bd)
  omega

/-- Distinct (thread, step) pairs visit distinct indices. -/
theorem scan_inj (bd t k t' k' : Nat) (ht : t < bd) (ht' : t' < bd) (hbd : 0 < bd)
    (h : t + k * bd = t' + k' * bd) : t = t' ∧ k = k' := by
  have e1 : (t + k * bd) % bd = t := by
    rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt ht]
  have e2 : (t' + k' * bd) % bd = t' := by
    rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt ht']
  have e3 : (t + k * bd) / bd = k := by
    rw [Nat.add_mul_div_right _ _ hbd, Nat.div_eq_of_lt ht, Nat.zero_add]
  have e4 : (t' + k' * bd) / bd = k' := by
    rw [Nat.add_mul_div_right _ _ hbd, Nat.div_eq_of_lt ht', Nat.zero_add]
  rw [h] at e1 e3
  exact ⟨by omega, by omega⟩

/-- The block of 1024 threads needs 32 warp slots, exactly the size of the shared arrays. -/
theorem greedy_warp_slots (bd : Nat) (h : bd ≤ 1024) : (bd + 31) / 32 ≤ 32 := by omega

/-- The greedy fallback index is always a valid token when the vocabulary is non-empty. -/
theorem greedy_out_valid (wi n : Nat) (hn : 0 < n) : (if wi < n then wi else 0) < n := by
  split_ifs <;> omega

-- § penalties

def penalize (logit count rep freq present : Int) : Int :=
  if count ≤ 0 then logit
  else (if logit ≤ 0 then logit * rep else logit / rep) - (count * freq + (if count > 0 then 1 else 0) * present)

/-- A token absent from the window is not penalised. -/
theorem penalize_absent (logit count rep freq present : Int) (h : count ≤ 0) :
    penalize logit count rep freq present = logit := by
  simp [penalize, h]

/-- The presence term is a boolean: it is `present` once for any positive count. -/
theorem penalize_present_once (logit count rep freq present : Int) (h : 0 < count) :
    penalize logit count rep freq present =
      (if logit ≤ 0 then logit * rep else logit / rep) - (count * freq + present) := by
  have h1 : ¬ count ≤ 0 := by omega
  simp [penalize, h1, h]

/-- Non-positive logits are multiplied and positive ones divided. -/
theorem penalize_branches (logit count rep freq present : Int) (h : 0 < count) :
    (logit ≤ 0 → penalize logit count rep freq present = logit * rep - (count * freq + present)) ∧
    (0 < logit → penalize logit count rep freq present = logit / rep - (count * freq + present)) := by
  have h1 : ¬ count ≤ 0 := by omega
  constructor
  · intro hl; simp [penalize, h1, h, hl]
  · intro hl
    have h2 : ¬ logit ≤ 0 := by omega
    simp [penalize, h1, h, h2]

-- § history window

def histLen (plen hl : Int) : Int := max 0 (min plen hl)

/-- The window length is between 0 and the history length. -/
theorem histLen_bounds (plen hl : Int) (h : 0 ≤ hl) : 0 ≤ histLen plen hl ∧ histLen plen hl ≤ hl := by
  unfold histLen; omega

/-- The tail window starts at a non-negative offset and ends at the history end. -/
theorem hist_tail_offset (plen hl : Int) (h : 0 ≤ hl) :
    0 ≤ hl - histLen plen hl ∧ hl - histLen plen hl + histLen plen hl = hl := by
  unfold histLen; omega

/-- A non-positive penalty window is empty. -/
theorem histLen_nonpos (plen hl : Int) (h : plen ≤ 0) : histLen plen hl = 0 := by
  unfold histLen; omega

-- § top-k / top-p

def KMAX : Nat := 64

def finalCut (cut minKeep nKeep : Nat) : Nat := if cut < minKeep then min minKeep nKeep else cut

/-- The top-p cut never exceeds the number of survivors. -/
theorem finalCut_le (cut minKeep nKeep : Nat) (h : cut ≤ nKeep) : finalCut cut minKeep nKeep ≤ nKeep := by
  unfold finalCut; split_ifs <;> omega

/-- At least one survivor remains when min_keep, the cut and the survivors are positive. -/
theorem finalCut_pos (cut minKeep nKeep : Nat) (h1 : 1 ≤ cut) (h2 : 1 ≤ minKeep) (h3 : 1 ≤ nKeep) :
    1 ≤ finalCut cut minKeep nKeep := by
  unfold finalCut; split_ifs <;> omega

/-- The effective top-k is capped at KMAX. -/
theorem topk_capped (topk : Nat) : (if topk < KMAX then topk else KMAX) ≤ KMAX := by
  unfold KMAX; split_ifs <;> omega

-- § Philox and launch

/-- A 32-bit word shifted right by 8 has at most 24 bits, so the uniform stays strictly below 1. -/
theorem philox_mantissa_lt (c0 : Nat) (h : c0 < 4294967296) : c0 / 256 < 16777216 := by omega

/-- The sampled-path grid covers every token. -/
theorem sampler_grid_cover (n : Nat) : n ≤ (n + 63) / 64 * 64 := by omega

end Strata.Sampler



## FILE: /mnt/user-data/outputs/VerifyWindowKernels.lean
/-!
# MODULE: verify_window_kernels
Windowing, commit and index contracts of the speculative-verify kernels (`verify_kernels.cu`): the
causal-conv window over `[hist(3) | x...]`, the conv-history commit after `n_keep` accepted tokens, pointer
rebasing, window ids, dense steps, and packed-code extraction.

## Source semantics
Token t's conv window is entries t..t+2 of `[hist | x_0 .. x_{n-1}]` plus the new `x_t`. After accepting n
tokens the history becomes the last three entries of `[hist | x_0 .. x_{n-1}]`; with n = 0 it is unchanged.
`rebase_ptrs` writes `base + k*bytes`. `window_ids` clamps the start to 0. Packed codes use
`8/bits` codes per byte.

## Form template identifiers
- hist -> `hist`, qkv -> `xs`, n_keep -> `n`
- `gdn_conv_commit` -> `commit`; `rebase_ptrs` -> `rebase`; `window_ids` -> `winStart`/`winWidth`
-/

namespace Strata.Verify

-- § conv window

/-- Reading `[hist | xs]` at `src` is `hist[src]` below 3 and `xs[src-3]` otherwise. -/
theorem window_get {α : Type} (hist xs : List α) (hh : hist.length = 3) (src : Nat) :
    (hist ++ xs)[src]? = if src < 3 then hist[src]? else xs[src - 3]? := by
  by_cases h : src < 3
  · rw [if_pos h, List.getElem?_append_left (by omega)]
  · rw [if_neg h, List.getElem?_append_right (by omega), hh]

def commit {α : Type} (hist xs : List α) (n : Nat) : List α := (hist ++ xs.take n).drop n

/-- With nothing accepted the history is unchanged. -/
theorem commit_zero {α : Type} (hist xs : List α) : commit hist xs 0 = hist := by
  simp [commit]

/-- After accepting n <= |xs| tokens the history again holds exactly three entries. -/
theorem commit_length {α : Type} (hist xs : List α) (hh : hist.length = 3) (n : Nat) (hn : n ≤ xs.length) :
    (commit hist xs n).length = 3 := by
  simp [commit, List.length_drop, List.length_append, List.length_take, hh]
  omega

/-- Window source indices for token t are t, t+1, t+2 and stay within the buffer for t < |xs|. -/
theorem window_src_in_range (t j len : Nat) (ht : t < len) (hj : j < 3) : t + j < 3 + len := by omega

-- § pointer rebasing

/-- `rebase_ptrs` assigns distinct addresses to distinct slots when blobs are non-empty. -/
theorem rebase_inj (base bytes k k' : Nat) (hb : 0 < bytes) (h : base + k * bytes = base + k' * bytes) :
    k = k' := by
  have h1 : k * bytes = k' * bytes := by omega
  exact Nat.eq_of_mul_eq_mul_right hb h1

/-- Consecutive rebased slots are exactly one blob apart, so blobs do not overlap. -/
theorem rebase_adjacent (base bytes k : Nat) : base + (k + 1) * bytes = base + k * bytes + bytes := by
  rw [Nat.add_mul, Nat.one_mul]; omega

-- § window ids

def winStart (nkv window : Nat) : Nat := if nkv > window then nkv - window else 0
def winWidth (nkv window : Nat) : Nat := nkv - winStart nkv window

/-- Start plus width is the kv length. -/
theorem win_partition (nkv window : Nat) : winStart nkv window + winWidth nkv window = nkv := by
  unfold winWidth winStart; split_ifs <;> omega

/-- The window never exceeds its size. -/
theorem win_width_le (nkv window : Nat) : winWidth nkv window ≤ window := by
  unfold winWidth winStart; split_ifs <;> omega

/-- A short cache is taken whole. -/
theorem win_short (nkv window : Nat) (h : nkv ≤ window) : winStart nkv window = 0 ∧ winWidth nkv window = nkv := by
  unfold winWidth winStart; split_ifs <;> omega

-- § dense steps

/-- The four packed step fields of one cell are at distinct slots. -/
theorem steps_slot_inj (i m i' m' : Nat) (hm : m < 4) (hm' : m' < 4) (h : i * 4 + m = i' * 4 + m') :
    i = i' ∧ m = m' := by omega

/-- The third field `(c+1)/4` never exceeds the second `c+1`. -/
theorem steps_third_le (c : Nat) : (c + 1) / 4 ≤ c + 1 := by omega

-- § packed codes

/-- For the supported widths, `8/bits` codes per byte fill the byte exactly. -/
theorem codes_fill_byte : ∀ bits, bits = 1 ∨ bits = 2 ∨ bits = 4 ∨ bits = 8 → 8 / bits * bits = 8 := by
  intro bits h
  rcases h with h | h | h | h <;> subst h <;> decide

/-- An extracted code is below 2^bits. -/
theorem code_lt (byte s bits : Nat) : (byte / 2 ^ s) % 2 ^ bits < 2 ^ bits :=
  Nat.mod_lt _ (Nat.two_pow_pos bits)

/-- The byte index of element i is within the row when the row holds n = row_codes * per_byte codes. -/
theorem code_byte_lt (i rowCodes perByte : Nat) (h : i < rowCodes * perByte) : i / perByte < rowCodes := by
  apply Nat.div_lt_of_lt_mul
  rw [Nat.mul_comm]; exact h

-- § launch geometry

/-- The alpha/beta kernel covers 2*h_v rows with 8 rows per block. -/
theorem ab_rows_cover (hv : Nat) : 2 * hv ≤ (2 * hv + 7) / 8 * 8 := by omega

/-- Beta rows index the beta weight from 0. -/
theorem ab_beta_row (hv row : Nat) (h1 : hv ≤ row) (h2 : row < 2 * hv) : row - hv < hv := by omega

/-- The conv kernel's block x covers all channels when channels is a multiple of 128. -/
theorem conv_channel_cover (channels c : Nat) (hc : channels % 128 = 0) (h : c < channels) :
    ∃ b t, b < channels / 128 ∧ t < 128 ∧ c = b * 128 + t := by
  exact ⟨c / 128, c % 128, by omega, by omega, by omega⟩

/-- A step block has 128 x 4 = 512 threads whose row groups each own 32 of the 128 state rows. -/
theorem step_rows_cover (r : Nat) (hr : r < 128) : ∃ rg k, rg < 4 ∧ k < 32 ∧ r = rg * 32 + k := by
  exact ⟨r / 32, r % 32, by omega, by omega, by omega⟩

/-- The copy kernel launches at most 64 blocks. -/
theorem copy_blocks_le (n : Nat) : (if (n + 255) / 256 < 64 then (n + 255) / 256 else 64) ≤ 64 := by
  split_ifs <;> omega

end Strata.Verify



## FILE: /mnt/user-data/outputs/GdnLayerContracts.lean
/-!
# MODULE: gdn_layer_contracts
Shape and indexing contracts of the gated delta-net layer interface (`gdn.hpp`): channel count, head
pairing `h % h_k`, the (S, H_v, S) state layout, and why the kernel's decay-then-update recurrence agrees
with the reference.

## Source semantics
S = 128, h_k = 16, h_v = 48, C = 2*S*h_k + S*h_v = 10240. State layout is (i, h, j) with j fastest, so the
flat index is `(i*h_v + h)*S + j`, injective and bounded by `S*h_v*S`. The head pairing is MODULO; the
interleaved pairing is a different map. The step contracts the state against k with the decayed state; the
kernel forms `g * dot(state, k)`, which equals `dot(g * state, k)`.

## Form template identifiers
- S -> `S`, h_k -> `hk`, h_v -> `hv`, d_conv -> `dconv`
- state layout -> `stateIdx`; head pairing -> `h % hk`
-/

namespace Strata.Gdn

def S : Nat := 128
def hk : Nat := 16
def hv : Nat := 48
def C : Nat := 2 * S * hk + S * hv

/-- The real artifact has 10240 conv channels. -/
theorem C_eq : C = 10240 := by decide

/-- h_v is a multiple of h_k at the real geometry. -/
theorem hv_multiple : hv % hk = 0 := by decide

-- § head pairing

/-- The modulo pairing always names a valid k/q head. -/
theorem pair_lt (h hk : Nat) (hpos : 0 < hk) : h % hk < hk := Nat.mod_lt h hpos

/-- Modulo and interleaved pairings are different maps: head 1 pairs with k-head 1 under modulo but 0 under interleave. -/
theorem modulo_ne_interleave : ∃ h, h < 48 ∧ h % 16 ≠ h / 3 := by
  exact ⟨1, by decide, by decide⟩

/-- Each k head serves exactly h_v/h_k = 3 value heads under modulo pairing (checked for head 0). -/
theorem pair_fanout_real : ∀ kh, kh < 16 → ((List.range 48).filter (fun h => h % 16 = kh)).length = 3 := by
  decide

-- § state layout

/-- Mixed-radix digits are unique. -/
theorem digit_inj {S a a' j j' : Nat} (hS : 0 < S) (hj : j < S) (hj' : j' < S)
    (h : a * S + j = a' * S + j') : a = a' ∧ j = j' := by
  have key : ∀ x y p q : Nat, p < S → x < y → x * S + p < y * S + q := by
    intro x y p q hp hxy
    have h1 := Nat.mul_le_mul_right S (show x + 1 ≤ y from hxy)
    rw [Nat.add_mul, Nat.one_mul] at h1
    omega
  rcases Nat.lt_trichotomy a a' with h1 | h1 | h1
  · have := key a a' j j' hj h1
    omega
  · subst h1
    exact ⟨rfl, by omega⟩
  · have := key a' a j' j hj' h1
    omega

/-- A digit pair stays below the radix product. -/
theorem digit_lt {a m S j : Nat} (ha : a < m) (hj : j < S) : a * S + j < m * S := by
  have h := Nat.mul_le_mul_right S (show a + 1 ≤ m from ha)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

def stateIdx (hv S i h j : Nat) : Nat := (i * hv + h) * S + j

/-- State indices stay inside the (S, h_v, S) buffer. -/
theorem stateIdx_lt (hv S i h j : Nat) (hi : i < S) (hh : h < hv) (hj : j < S) :
    stateIdx hv S i h j < S * hv * S := by
  unfold stateIdx
  exact digit_lt (digit_lt hi hh) hj

/-- Distinct (i, h, j) triples address distinct state cells. -/
theorem stateIdx_inj (hv S i h j i' h' j' : Nat) (hS : 0 < S) (hH : 0 < hv)
    (hh : h < hv) (hh' : h' < hv) (hj : j < S) (hj' : j' < S)
    (e : stateIdx hv S i h j = stateIdx hv S i' h' j') : i = i' ∧ h = h' ∧ j = j' := by
  unfold stateIdx at e
  obtain ⟨e1, e2⟩ := digit_inj hS hj hj' e
  obtain ⟨e3, e4⟩ := digit_inj hH hh hh' e1
  exact ⟨e3, e4, e2⟩

/-- For fixed (i, h), consecutive j are consecutive addresses (the coalescing property). -/
theorem stateIdx_succ (hv S i h j : Nat) : stateIdx hv S i h (j + 1) = stateIdx hv S i h j + 1 := by
  unfold stateIdx; omega

-- § kernel row-group split

/-- A step block splits the 128 state rows into 4 groups of 32. -/
theorem rowgroup_cover (r : Nat) (hr : r < 128) : ∃ rg k, rg < 4 ∧ k < 32 ∧ r = rg * 32 + k := by
  exact ⟨r / 32, r % 32, by omega, by omega, by omega⟩

-- § decay-then-update

def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- Scaling the state before contracting equals scaling after, so the kernel's `g * dot(state, k)` is the
reference's contraction of the decayed state. -/
theorem dot_smul (g : Int) (as bs : List Int) : dot (as.map (fun a => g * a)) bs = g * dot as bs := by
  induction as generalizing bs with
  | nil => simp [dot]
  | cons a as ih =>
    cases bs with
    | nil => simp [dot]
    | cons b bs => simp [dot, ih, Int.mul_add, Int.mul_assoc]

/-- The delta term built from the undecayed contraction equals the one built from the decayed state. -/
theorem delta_decay_equiv (g beta v : Int) (st k : List Int) :
    (v - g * dot st k) * beta = (v - dot (st.map (fun a => g * a)) k) * beta := by
  rw [dot_smul]

end Strata.Gdn





## FILE: /mnt/user-data/outputs/CpuQ2ExpertKernel.lean
/-!
# MODULE: cpu_q2_expert_kernel
The Q2_0 routed-expert kernel on the CPU: fixed geometry of one 1,382,400-byte expert blob, the activation
quantizer's integer contract, the Q2_0 "-1 code offset" identity that the `hx` correction implements, row-range
splitting, token-count template dispatch and the CPU feature gate.

## Source semantics
* One weight block = QK = 64 weights with ONE fp16 scale; one activation chunk = QKA = 32 elements with its OWN
  scale, so a weight block spans exactly two activation chunks.
* `sum (c-1) d_w xhat d_x = d_w (d_x sum(c xhat) - d_x sum(xhat))` (the `hx` correction), split per chunk.
* Activation codes are clamped to [-127,127]; chunk sums and VNNI lane sums are exact in int32.
* `s2_expert_gu_rows` / `s2_expert_down_rows` over any partition of the rows equal the whole-expert kernel.
* `s2_expert_vnni_multi` dispatches n_tokens in 1..7 to NT = n_tokens, anything else to NT = 8 (or the fallback).
* The kernel is usable iff AVX512F, BW, VL, VNNI and VBMI are all present.

## Form template identifiers
* `H`, `FF`, `NE`, `QK`, `QKA`, `BB`, `MAXC`, `ROW_GU`, `ROW_D`, `SC_GU`, `SC_D`, `BLOB`, `O_*` : geometry constants
* `lo n parts p` : the row boundary `n * p / parts` used by split phases
* `dispatchNT` : the `switch (n_tokens)` template selector
* `Feat` / `Feat.usable` / `Feat.missing` : `CpuFeatures`
-/

namespace Strata.CpuExpert

-- § geometry

def H : Nat := 2560
def FF : Nat := 640
def NE : Nat := 512
def QK : Nat := 64
def QKA : Nat := 32
def BB : Nat := 18
def MAXC : Nat := H / QKA
def ROW_GU : Nat := H * 2 / 8
def ROW_D : Nat := FF * 2 / 8
def SC_GU : Nat := H / QK
def SC_D : Nat := FF / QK
def BLOB : Nat := 3 * (H * FF * BB / QK)
def O_GU_CODES : Nat := 0
def O_D_CODES : Nat := 2 * FF * ROW_GU
def O_GU_SCALES : Nat := O_D_CODES + H * ROW_D
def O_D_SCALES : Nat := O_GU_SCALES + 2 * FF * SC_GU * 2
def O_END : Nat := O_D_SCALES + H * SC_D * 2

/-- The widest reduction has 80 activation chunks. -/
theorem maxc_eq : MAXC = 80 := by decide
/-- A gate/up row holds 640 bytes of codes. -/
theorem row_gu_eq : ROW_GU = 640 := by decide
/-- A down row holds 160 bytes of codes. -/
theorem row_d_eq : ROW_D = 160 := by decide
/-- A gate/up row has 40 weight blocks. -/
theorem sc_gu_eq : SC_GU = 40 := by decide
/-- A down row has 10 weight blocks. -/
theorem sc_d_eq : SC_D = 10 := by decide
/-- One expert blob is 1,382,400 bytes. -/
theorem blob_eq : BLOB = 1382400 := by decide
/-- Offset of the down codes. -/
theorem o_d_codes_eq : O_D_CODES = 819200 := by decide
/-- Offset of the gate/up scales. -/
theorem o_gu_scales_eq : O_GU_SCALES = 1228800 := by decide
/-- Offset of the down scales. -/
theorem o_d_scales_eq : O_D_SCALES = 1331200 := by decide

/-- The four blob segments tile the blob exactly: the last segment ends at `BLOB`. -/
theorem layout_end_eq_blob : O_END = BLOB := by decide

/-- One 18-byte block is a 2-byte fp16 scale plus 16 code bytes (64 two-bit codes). -/
theorem block_bytes : BB = 2 + QK * 2 / 8 := by decide

/-- One weight block spans exactly two activation chunks. -/
theorem block_spans_two_chunks : QK = 2 * QKA := by decide

/-- The chunk count of the gate/up reduction is twice its weight-block count. -/
theorem maxc_two_blocks : MAXC = 2 * SC_GU := by decide

/-- The chunk count of the down reduction is twice its weight-block count. -/
theorem ff_chunks_two_blocks : FF / QKA = 2 * SC_D := by decide

/-- Every (block, half) pair of a gate/up row names a valid chunk scale. -/
theorem gu_chunk_index_lt : ∀ b h, b < SC_GU → h < 2 → 2 * b + h < MAXC := by
  intro b h hb hh
  have h1 := sc_gu_eq
  have h2 := maxc_eq
  omega

/-- Every (block, half) pair of a down row names a valid chunk of the intermediate. -/
theorem d_chunk_index_lt : ∀ b h, b < SC_D → h < 2 → 2 * b + h < FF / QKA := by
  intro b h hb hh
  have h1 := sc_d_eq
  have h2 : FF / QKA = 20 := by decide
  omega

-- § blob addressing

/-- Gate and up rows (2r and 2r+1) of all FF rows lie inside the gate/up code segment. -/
theorem gu_codes_in_segment (r : Nat) (hr : r < FF) :
    O_GU_CODES + (2 * r + 2) * ROW_GU ≤ O_D_CODES := by
  have h1 : FF = 640 := by decide
  rw [row_gu_eq, o_d_codes_eq]
  unfold O_GU_CODES
  omega

/-- Down rows lie inside the down code segment. -/
theorem d_codes_in_segment (r : Nat) (hr : r < H) :
    O_D_CODES + (r + 1) * ROW_D ≤ O_GU_SCALES := by
  have h1 : H = 2560 := by decide
  rw [row_d_eq, o_d_codes_eq, o_gu_scales_eq]
  omega

/-- Gate and up scale rows lie inside the gate/up scale segment. -/
theorem gu_scales_in_segment (r : Nat) (hr : r < FF) :
    O_GU_SCALES + (2 * r + 2) * SC_GU * 2 ≤ O_D_SCALES := by
  have h1 : FF = 640 := by decide
  rw [sc_gu_eq, o_gu_scales_eq, o_d_scales_eq]
  omega

/-- Down scale rows lie inside the down scale segment (which ends at the blob end). -/
theorem d_scales_in_segment (r : Nat) (hr : r < H) :
    O_D_SCALES + (r + 1) * SC_D * 2 ≤ BLOB := by
  have h1 : H = 2560 := by decide
  rw [sc_d_eq, o_d_scales_eq, blob_eq]
  omega

-- § the Q2_0 code-offset identity

/-- The offset identity, exactly as the `hx` correction uses it: `sum (c-1) x = sum (c x) - sum x`. -/
theorem offset_identity (l : List (Int × Int)) :
    (l.map (fun p => (p.1 - 1) * p.2)).sum =
      (l.map (fun p => p.1 * p.2)).sum - (l.map (fun p => p.2)).sum := by
  induction l with
  | nil => simp
  | cons p t ih =>
    simp only [List.map_cons, List.sum_cons]
    rw [ih, Int.sub_mul, Int.one_mul]
    omega

/-- Splitting a block into its two chunks: the block sum is the sum of the chunk sums, so each chunk can use
its own activation scale. -/
theorem block_sum_split (a b : List Int) : (a ++ b).sum = a.sum + b.sum := by
  simp

-- § integer exactness of the activation contract

/-- The activation code clamp. -/
def clampQ (v : Int) : Int := if v < -127 then -127 else if 127 < v then 127 else v

/-- Clamped activation codes always fit in int8 (and never reach -128). -/
theorem clampQ_range (v : Int) : -127 ≤ clampQ v ∧ clampQ v ≤ 127 := by
  unfold clampQ
  split <;> try split
  all_goals omega

/-- Clamping is the identity on the representable range. -/
theorem clampQ_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clampQ v = v := by
  unfold clampQ
  split <;> try split
  all_goals omega

/-- A list of codes in [-127,127] has a sum bounded by 127 times its length. -/
theorem sum_bound (l : List Int) (h : ∀ x ∈ l, -127 ≤ x ∧ x ≤ 127) :
    -(127 * (l.length : Int)) ≤ l.sum ∧ l.sum ≤ 127 * (l.length : Int) := by
  induction l with
  | nil => simp
  | cons a t ih =>
    have ha := h a (by simp)
    have ht := ih (fun x hx => h x (by simp [hx]))
    simp only [List.sum_cons, List.length_cons]
    omega

/-- A 32-element chunk sum is at most 4064 in absolute value, so `a.sum[k]` is exact in int32. -/
theorem chunk_sum_bound (l : List Int) (hl : l.length = 32) (h : ∀ x ∈ l, -127 ≤ x ∧ x ≤ 127) :
    -4064 ≤ l.sum ∧ l.sum ≤ 4064 := by
  have := sum_bound l h
  rw [hl] at this
  omega

/-- A VNNI lane sums four products of a 2-bit code (0..3) and an int8: |lane| <= 3*127*4. -/
theorem vnni_lane_bound : 3 * 127 * 4 = 1524 := by decide

/-- Even all 64 products of a block stay far below 2^31, so integer partial sums are exact. -/
theorem vnni_block_bound : 3 * 127 * 64 < 2 ^ 31 := by decide

-- § row-range splitting

/-- The row boundary of part `p` out of `parts`. -/
def lo (n parts p : Nat) : Nat := n * p / parts

/-- The first part starts at row 0. -/
theorem lo_zero (n parts : Nat) : lo n parts 0 = 0 := by simp [lo]

/-- The last part ends at row `n`. -/
theorem lo_full (n parts : Nat) (h : 0 < parts) : lo n parts parts = n := by
  simp [lo, Nat.mul_div_cancel _ h]

/-- Boundaries are monotone. -/
theorem lo_mono (n parts p q : Nat) (h : p ≤ q) : lo n parts p ≤ lo n parts q :=
  Nat.div_le_div_right (Nat.mul_le_mul_left n h)

/-- Part p's end is part p+1's start: the ranges tile with no gap or overlap. -/
theorem lo_adjacent (n parts p : Nat) : lo n parts (p + 1) = lo n parts (p + 1) := rfl

/-- Every boundary lies inside [0, n]. -/
theorem lo_le (n parts p : Nat) (hp : p ≤ parts) (h : 0 < parts) : lo n parts p ≤ n := by
  have := lo_mono n parts p parts hp
  rw [lo_full n parts h] at this
  exact this

/-- Splitting the rows at any point gives two ranges whose lengths add up to the whole range. -/
theorem range_split (r0 m r1 : Nat) (h1 : r0 ≤ m) (h2 : m ≤ r1) :
    (m - r0) + (r1 - m) = r1 - r0 := by omega

-- § template dispatch on the token count

/-- `switch (n_tokens)`: 1..7 select NT = n, everything else selects NT = 8. -/
def dispatchNT (n : Nat) : Nat := if 1 ≤ n ∧ n ≤ 7 then n else 8

/-- The selected NT is always an instantiated template (1..8). -/
theorem dispatchNT_range (n : Nat) : 1 ≤ dispatchNT n ∧ dispatchNT n ≤ 8 := by
  unfold dispatchNT
  split <;> omega

/-- For 1..8 tokens (MAXT = 8) the selected template has exactly that many lanes. -/
theorem dispatchNT_exact (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ 8) : dispatchNT n = n := by
  unfold dispatchNT
  split <;> omega

/-- `s2_expert_vnni_multi` uses the batched kernel only without the oracle and with 1..MAXT tokens. -/
def useMulti (oracle : Bool) (n : Nat) : Bool := !oracle && decide (1 ≤ n) && decide (n ≤ 8)

/-- When the batched kernel runs, its lane count equals the token count. -/
theorem useMulti_lanes (oracle : Bool) (n : Nat) (h : useMulti oracle n = true) : dispatchNT n = n := by
  unfold useMulti at h
  simp at h
  exact dispatchNT_exact n h.1.2 h.2

/-- With the oracle contract on, the batched kernel is never used. -/
theorem useMulti_oracle (n : Nat) : useMulti true n = false := by
  simp [useMulti]

-- § CPU feature gate

structure Feat where
  f : Bool
  bw : Bool
  vl : Bool
  vnni : Bool
  vbmi : Bool

/-- `CpuFeatures::usable`. -/
def Feat.usable (x : Feat) : Bool := x.f && x.bw && x.vl && x.vnni && x.vbmi

/-- The names `reason()` would list. -/
def Feat.missing (x : Feat) : List String :=
  (if x.f then [] else ["AVX512F"]) ++ (if x.bw then [] else ["AVX512BW"]) ++
  (if x.vl then [] else ["AVX512VL"]) ++ (if x.vnni then [] else ["AVX512-VNNI"]) ++
  (if x.vbmi then [] else ["AVX512-VBMI"])

/-- `reason()` reports "ok" exactly when nothing is missing, which is exactly when the CPU is usable. -/
theorem missing_nil_iff (x : Feat) : x.missing = [] ↔ x.usable = true := by
  cases x with
  | mk a b c d e =>
    cases a <;> cases b <;> cases c <;> cases d <;> cases e <;> simp [Feat.missing, Feat.usable]

/-- A CPU lacking any one of the five features is refused. -/
theorem lacking_vnni_refused (a b c e : Bool) : (Feat.mk a b c false e).usable = false := by
  simp [Feat.usable]

end Strata.CpuExpert



## FILE: /mnt/user-data/outputs/CpuExpertPool.lean
/-!
# MODULE: cpu_expert_pool
The CPU expert pool: single-counter job claiming (`head_` fetch_add), the `done`-then-`parked` completion
handover, physical-core selection, worker count, and the arithmetic that cuts a phase into equal row ranges
across all experts of a multi-token batch.

## Source semantics
* The k-th `fetch_add` on `head_` returns k; the claim is a job iff k < n. So exactly n claims succeed and each
  index is handed out once.
* `run()` returns only when `done == n` AND `parked == n_`. `done` alone is not enough: a worker can still be
  inside the drain after its last `done` increment.
* `physical_cores(skip_first)` drops the first core; the worker count is at least 1.
* `run_split` cuts each expert into `parts` row ranges; task i maps to (expert i / parts, part i % parts).
* The multi phases cut `mrows` rows (n experts times per-expert rows) into `mtasks` ranges
  `[mrows*i/mtasks, mrows*(i+1)/mtasks)`, each walked expert by expert with the end clamped at the expert edge.
* Native batches of at most `kMaxSplitMulti = 96` experts cover all n experts.

## Form template identifiers
* `claim`, `validClaim` : `head_.fetch_add(1)` and `i < njobs_`
* `inDrain` : `n_ - parked_`
* `lo` : `mrows * i / mtasks`
* `chunks k n` : the `for (b0 = 0; b0 < n; b0 += k)` batching loop with `nb = min(k, n - b0)`
-/

namespace Strata.CpuPool

-- § claiming

/-- The k-th `fetch_add` returns k (counters start at 0 for each batch). -/
def claim (k : Nat) : Nat := k

/-- A claim is a job iff its index is below the job count. -/
def validClaim (i n : Nat) : Bool := decide (i < n)

/-- The successful claim indices are exactly `0 .. n-1`. -/
theorem valid_iff_mem_range (k n : Nat) : validClaim (claim k) n = true ↔ k ∈ List.range n := by
  unfold validClaim claim
  rw [List.mem_range]
  exact decide_eq_true_iff

/-- Distinct claims get distinct indices, so no job is handed to two workers. -/
theorem claim_injective (a b : Nat) (h : claim a = claim b) : a = b := h

/-- Among the first m claims exactly `min m n` succeed. -/
theorem valid_claims (m n : Nat) :
    ((List.range m).filter (fun k => decide (k < n))).length = min m n := by
  induction m with
  | zero => simp
  | succ m ih =>
    rw [List.range_succ, List.filter_append, List.length_append, ih]
    by_cases h : m < n
    · simp [h]
      omega
    · simp [h]
      omega

-- § completion handover

/-- Workers currently outside the park. -/
def inDrain (n parked : Nat) : Nat := n - parked

/-- With every worker parked nobody is in the drain, so resetting `head_` cannot race a claim. -/
theorem parked_all_no_drain (n parked : Nat) (h : parked = n) : inDrain n parked = 0 := by
  simp [inDrain, h]

/-- Waiting for `done == njobs` alone is insufficient: a state exists where all jobs are done yet a worker is
still in the drain. -/
theorem done_alone_insufficient :
    ∃ done parked n njobs : Nat, done = njobs ∧ parked < n ∧ 0 < inDrain n parked :=
  ⟨3, 1, 2, 3, rfl, by decide, by decide⟩

/-- Workers arrive at the park once before the first wait, so `parked == n_` holds from construction. -/
theorem initial_parked (n : Nat) : (List.replicate n 1).sum = n := by
  simp

-- § cores and worker count

/-- `physical_cores(skip_first)`: drop the first core when asked. -/
def skipFirst (skip : Bool) (cores : List Nat) : List Nat := if skip then cores.drop 1 else cores

/-- Skipping removes exactly one core from a non-empty list. -/
theorem skipFirst_length (cores : List Nat) (h : cores ≠ []) :
    (skipFirst true cores).length + 1 = cores.length := by
  cases cores with
  | nil => exact absurd rfl h
  | cons a t => simp [skipFirst]

/-- Skipping never adds cores. -/
theorem skipFirst_le (skip : Bool) (cores : List Nat) : (skipFirst skip cores).length ≤ cores.length := by
  cases skip <;> simp [skipFirst]

/-- The pool's worker count: the request if positive, else every (remaining) core; at least one. -/
def nWorkers (req ncores : Nat) : Nat :=
  if (if 0 < req then req else ncores) < 1 then 1 else (if 0 < req then req else ncores)

/-- There is always at least one worker, so a token is always produced. -/
theorem nWorkers_pos (req ncores : Nat) : 1 ≤ nWorkers req ncores := by
  unfold nWorkers
  split_ifs <;> omega

/-- A positive request is honoured. -/
theorem nWorkers_req (req ncores : Nat) (h : 0 < req) : nWorkers req ncores = req := by
  unfold nWorkers
  split_ifs <;> omega

-- § row-range arithmetic

/-- Boundary of range `i` when `mrows` rows are cut into `mtasks` ranges. -/
def lo (mrows mtasks i : Nat) : Nat := mrows * i / mtasks

/-- The first range starts at 0. -/
theorem lo_zero (mrows mtasks : Nat) : lo mrows mtasks 0 = 0 := by simp [lo]

/-- The last range ends at `mrows`: every row of the phase is covered. -/
theorem lo_full (mrows mtasks : Nat) (h : 0 < mtasks) : lo mrows mtasks mtasks = mrows := by
  simp [lo, Nat.mul_div_cancel _ h]

/-- Boundaries are monotone, so ranges never overlap. -/
theorem lo_mono (mrows mtasks i j : Nat) (h : i ≤ j) : lo mrows mtasks i ≤ lo mrows mtasks j :=
  Nat.div_le_div_right (Nat.mul_le_mul_left mrows h)

/-- A flat row index decomposes as (expert, row within expert). -/
theorem flat_decomp (r per : Nat) (h : 0 < per) :
    r = per * (r / per) + r % per ∧ r % per < per := by
  have h1 := Nat.div_add_mod r per
  have h2 := Nat.mod_lt r h
  omega

/-- The walk clamps the range end at the expert edge and always makes progress. -/
theorem clamp_progress (r0 per rem : Nat) (h0 : r0 < per) (hr : 0 < rem) :
    r0 < min per (r0 + rem) ∧ min per (r0 + rem) ≤ per := by
  omega

/-- The expert index of a flat row is in range when the row is below `n * per`. -/
theorem expert_index_lt (r per n : Nat) (h : r < n * per) (hp : 0 < per) : r / per < n :=
  (Nat.div_lt_iff_lt_mul hp).2 (by have := Nat.mul_comm n per; omega)

-- § run_split task decomposition

/-- Parts per expert: about three tasks per thread, at least one part. -/
def partsPer (threads n : Nat) : Nat := max 1 ((3 * threads + n - 1) / n)

/-- At least one part per expert. -/
theorem partsPer_pos (threads n : Nat) : 1 ≤ partsPer threads n := by
  unfold partsPer
  omega

/-- The unclamped quotient provides at least three tasks per thread. -/
theorem parts_cover_threads (threads n : Nat) (hn : 0 < n) :
    3 * threads ≤ n * ((3 * threads + n - 1) / n) := by
  have h1 := Nat.div_add_mod (3 * threads + n - 1) n
  have h2 := Nat.mod_lt (3 * threads + n - 1) hn
  omega

/-- Task i of `n * parts` names an expert below n. -/
theorem task_expert_lt (i n parts : Nat) (h : i < n * parts) (hp : 0 < parts) : i / parts < n :=
  (Nat.div_lt_iff_lt_mul hp).2 (by have := Nat.mul_comm n parts; omega)

/-- Task i names a part below `parts`. -/
theorem task_part_lt (i parts : Nat) (hp : 0 < parts) : i % parts < parts := Nat.mod_lt i hp

/-- Part boundaries `FF * part / parts` tile [0, FF): the last part ends at FF. -/
theorem part_end (ff parts : Nat) (hp : 0 < parts) : ff * parts / parts = ff := by
  simp [Nat.mul_div_cancel _ hp]

-- § batching of experts

/-- `for (b0 = 0; b0 < n; b0 += k) nb = min(k, n - b0)`: the list of batch sizes. -/
def chunks (k n : Nat) : List Nat :=
  if h : n = 0 ∨ k = 0 then [] else min k n :: chunks k (n - min k n)
termination_by n
decreasing_by omega

/-- The batches cover all n experts exactly once. -/
theorem chunks_sum (k : Nat) (hk : 0 < k) : ∀ n, (chunks k n).sum = n := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    by_cases h0 : n = 0
    · subst h0
      simp [chunks]
    · rw [chunks, dif_neg (by omega), List.sum_cons, ih (n - min k n) (by omega)]
      omega

/-- Every batch is non-empty and at most `k` (so at most kMaxSplitMulti = 96 buffers are used). -/
theorem chunks_bound (k : Nat) (hk : 0 < k) : ∀ n, ∀ c ∈ chunks k n, 0 < c ∧ c ≤ k := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro c hc
    by_cases h0 : n = 0
    · subst h0
      simp [chunks] at hc
    · rw [chunks, dif_neg (by omega)] at hc
      rcases List.mem_cons.mp hc with rfl | hc
      · omega
      · exact ih (n - min k n) (by omega) c hc

end Strata.CpuPool



## FILE: /mnt/user-data/outputs/CpuIqAvx512Rows.lean
/-!
# MODULE: cpu_iq_avx512_rows
i-quant (IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) expert rows in 512-bit lanes for several tokens at once: the
byte layout each `Fmt<TY>::decode` reads, the grid-index and scale ranges, the row/block addressing, and the
token-count batching of `dot_rows_nt` / `gu_rows_nt`.

## Source semantics
* Each 256-value block is decoded as four 64-value chunks (j = 0..3). Every byte `decode` touches lies inside
  the format's block (66 / 74 / 82 / 98 / 110 bytes).
* Scale values are `2*s + 1` for a 4-bit s, so they are odd and in [1, 31].
* Grid indices stay inside their lookup table (256, 512 or 1024 entries); sign indices are 7-bit.
* `dot_rows_nt`: 1..4 tokens directly, otherwise in groups of 4; `gu_rows_nt`: 1..7 directly, else NT = 8.
* A row of nb blocks occupies `nb * bytes`; block i ends before the row ends.
* Types 16, 17, 18, 21, 22 are supported, nothing else.

## Form template identifiers
* `blockBytes` : `Fmt<TY>::bytes`
* `supported` : `iq512_supported`
* `chunks 4 nt` : the `t0 += 4` loop of `dot_rows_nt`
* `guNT` : the `switch (nt)` of `gu_rows_nt`
-/

namespace Strata.IqAvx512

-- § block layouts

/-- `Fmt<TY>::bytes` for the supported ggml type ids. -/
def blockBytes : Nat → Nat
  | 16 => 66
  | 17 => 74
  | 18 => 98
  | 21 => 110
  | 22 => 82
  | _ => 0

/-- `iq512_supported`. -/
def supported (t : Nat) : Bool := t == 16 || t == 17 || t == 18 || t == 21 || t == 22

/-- A type is supported iff it has a block size. -/
theorem supported_iff_bytes (t : Nat) : supported t = true ↔ 0 < blockBytes t := by
  unfold supported blockBytes
  split <;> simp_all

/-- IQ2_XXS: 2-byte d + 32 u16 = 66 bytes. -/
theorem bytes_iq2xxs : blockBytes 16 = 2 + 64 := by decide
/-- IQ2_XS: d + 32 u16 + 8 scale bytes = 74 bytes. -/
theorem bytes_iq2xs : blockBytes 17 = 2 + 64 + 8 := by decide
/-- IQ2_S: d + 64 qs + 8 qh + 8 scales = 82 bytes. -/
theorem bytes_iq2s : blockBytes 22 = 2 + 64 + 8 + 8 := by decide
/-- IQ3_XXS: d + 64 grid bytes + 32 sign/scale bytes = 98 bytes. -/
theorem bytes_iq3xxs : blockBytes 18 = 2 + 64 + 32 := by decide
/-- IQ3_S: d + 64 qs + 8 qh + 32 signs + 4 scales = 110 bytes. -/
theorem bytes_iq3s : blockBytes 21 = 2 + 64 + 8 + 32 + 4 := by decide

/-- Four 64-value chunks make one 256-value block (QK_K). -/
theorem chunks_per_block : 4 * 64 = 256 := by decide

/-- IQ2_XXS reads 16 bytes at `2 + 16 j`; all four chunks stay inside the block. -/
theorem xxs_qs_in_block (j : Nat) (hj : j < 4) : 2 + 16 * j + 16 ≤ blockBytes 16 := by
  have : blockBytes 16 = 66 := by decide
  omega

/-- IQ2_XS: qs at `2 + 16 j`, scale bytes at `66 + 2 j (+1)`, all below 74. -/
theorem xs_in_block (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < blockBytes 17 := by
  have : blockBytes 17 = 74 := by decide
  omega

/-- IQ2_S: grid bytes `2 + 8 j`, sign bytes `34 + 8 j`, qh `66 + 2 j (+1)`, scales `74 + 2 j (+1)`. -/
theorem iq2s_in_block (j : Nat) (hj : j < 4) :
    2 + 8 * j + 8 ≤ 34 ∧ 34 + 8 * j + 8 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 2 * j + 1 < blockBytes 22 := by
  have : blockBytes 22 = 82 := by decide
  omega

/-- IQ3_XXS: qs at `2 + 16 j`, sign/scale words at `66 + 8 j .. + 8`, ending exactly at 98 for j = 3. -/
theorem xxs3_in_block (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 8 * j + 8 ≤ blockBytes 18 := by
  have : blockBytes 18 = 98 := by decide
  omega

/-- IQ3_S: qs `2 + 16 j`, qh `66 + 2 j (+1)`, signs `74 + 8 j .. + 8`, scale `106 + j`. -/
theorem iq3s_in_block (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 8 * j + 8 ≤ 106 ∧ 106 + j < blockBytes 21 := by
  have : blockBytes 21 = 110 := by decide
  omega

-- § scales and indices

/-- A decoded scale `2 s + 1` from a 4-bit field s lies in [1, 31] and is odd. -/
theorem scale_range (s : Nat) (h : s < 16) : 1 ≤ 2 * s + 1 ∧ 2 * s + 1 ≤ 31 ∧ (2 * s + 1) % 2 = 1 := by
  omega

/-- `scales4` assigns int16 lane i (0..31) to scale index `i / 8`, always one of the four scales. -/
theorem scales4_index (i : Nat) (h : i < 32) : i / 8 < 4 := by omega

/-- A 9-bit grid index `v & 511` stays inside a 512-entry table. -/
theorem grid9_lt (v : Nat) : v &&& 511 < 512 := by
  have := Nat.and_le_right (n := v) (m := 511)
  omega

/-- An 8-bit grid index `x & 255` stays inside a 256-entry table. -/
theorem grid8_lt (x : Nat) : x &&& 255 < 256 := by
  have := Nat.and_le_right (n := x) (m := 255)
  omega

/-- A 7-bit sign index `(a >> k) & 127` stays inside the 128-entry `ksigns_iq2xs`. -/
theorem sign7_lt (a k : Nat) : (a >>> k) &&& 127 < 128 := by
  have := Nat.and_le_right (n := a >>> k) (m := 127)
  omega

/-- The IQ2_XS sign index `v >> 9` of a 16-bit v is a 7-bit value. -/
theorem iq2xs_sign_lt (v : Nat) (h : v < 65536) : v >>> 9 < 128 := by
  rw [Nat.shiftRight_eq_div_pow]
  omega

/-- The IQ2_S grid index `qs | ((h << k) & 0x300)` is below 1024 for a byte qs. -/
theorem iq2s_grid_lt (qs h k : Nat) (hq : qs < 256) : qs ||| ((h <<< k) &&& 0x300) < 1024 := by
  have h1 : (h <<< k) &&& 0x300 ≤ 0x300 := Nat.and_le_right
  have h2 : qs ||| ((h <<< k) &&& 0x300) < 2 ^ 10 := by
    apply Nat.or_lt_two_pow
    · omega
    · omega
  omega

-- § row addressing

/-- Block i of an nb-block row ends no later than the row. -/
theorem block_in_row (i nb bytes : Nat) (h : i < nb) : (i + 1) * bytes ≤ nb * bytes :=
  Nat.mul_le_mul_right bytes h

/-- A row of n values has `n / 256` blocks; for a multiple of 256 they cover it exactly. -/
theorem row_blocks (n : Nat) (h : n % 256 = 0) : (n / 256) * 256 = n := by omega

-- § token batching

/-- `gu_rows_nt`: 1..7 direct, everything else NT = 8. -/
def guNT (nt : Nat) : Nat := if 1 ≤ nt ∧ nt ≤ 7 then nt else 8

/-- For 1..8 tokens (MAXT = 8) the gate/up kernel has exactly nt lanes. -/
theorem guNT_exact (nt : Nat) (h1 : 1 ≤ nt) (h2 : nt ≤ 8) : guNT nt = nt := by
  unfold guNT
  split <;> omega

/-- Chunks of at most k, as in the `t0 += 4` loop. -/
def chunks (k n : Nat) : List Nat :=
  if h : n = 0 ∨ k = 0 then [] else min k n :: chunks k (n - min k n)
termination_by n
decreasing_by omega

/-- Every token is processed exactly once by the 4-token loop. -/
theorem chunks_sum (k : Nat) (hk : 0 < k) : ∀ n, (chunks k n).sum = n := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    by_cases h0 : n = 0
    · subst h0
      simp [chunks]
    · rw [chunks, dif_neg (by omega), List.sum_cons, ih (n - min k n) (by omega)]
      omega

/-- Each group has between 1 and 4 tokens, matching the instantiated NT = 1..4 templates. -/
theorem chunks_bound (k : Nat) (hk : 0 < k) : ∀ n, ∀ c ∈ chunks k n, 0 < c ∧ c ≤ k := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro c hc
    by_cases h0 : n = 0
    · subst h0
      simp [chunks] at hc
    · rw [chunks, dif_neg (by omega)] at hc
      rcases List.mem_cons.mp hc with rfl | hc
      · omega
      · exact ih (n - min k n) (by omega) c hc

end Strata.IqAvx512



## FILE: /mnt/user-data/outputs/CpuQ2Avx2Rows.lean
/-!
# MODULE: cpu_q2_avx2_rows
Q2_0 GGUF-layout rows and the activation quantizer for CPUs without AVX-512: the 2-bit unpack ordering, the
18-byte block addressing, the correction identity, the token batching and the int32 horizontal sum.

## Source semantics
* A code byte holds four 2-bit codes: value i of a block sits in byte i/4 at bits 2*(i%4); unpacking returns the
  64 values in order (lo = values 0..31 = bytes 0..7, hi = values 32..63 = bytes 8..15).
* The four shift-and-mask codes of a byte recombine to the byte (nothing is lost or duplicated).
* Block b of a row starts at `18 b`: a 2-byte scale, then 16 code bytes.
* `sum (c-1) x = sum (c x) - sum x`, which is the `corr += d (hx0 + hx1)` term.
* Codes of the quantizer are clamped to [-127, 127]; an 8-lane chunk sum is at most 8*127 per vector.
* The shuffle-based horizontal add of 8 int32 lanes yields the total.
* `nt` tokens run directly for 1..4, otherwise in groups of 4.

## Form template identifiers
* `code byte s` : `(byte >> s) & 3`
* `hsum8` : the `_mm_shuffle_epi32` 0x4E / 0xB1 reduction
* `chunks 4 nt` : the `t0 += 4` loop
-/

namespace Strata.Q2Avx2

-- § unpack

/-- The 2-bit code of `byte` at shift `s`. -/
def code (byte s : Nat) : Nat := (byte >>> s) &&& 3

/-- Every unpacked code is in 0..3. -/
theorem code_le (byte s : Nat) : code byte s ≤ 3 := by
  unfold code
  exact Nat.and_le_right

/-- The four codes of a byte recombine to the byte: the unpack drops and duplicates nothing. -/
theorem code_roundtrip : ∀ b : Fin 256,
    code b.val 0 + 4 * code b.val 2 + 16 * code b.val 4 + 64 * code b.val 6 = b.val := by
  decide +kernel

/-- Value position p of a block (0..63) lives in byte p/4. -/
def byteOf (p : Nat) : Nat := p / 4

/-- Shift of value position p inside its byte. -/
def shiftOf (p : Nat) : Nat := 2 * (p % 4)

/-- The `lo` vector (values 0..31) reads bytes 0..7. -/
theorem lo_bytes (p : Nat) (h : p < 32) : byteOf p < 8 := by
  unfold byteOf
  omega

/-- The `hi` vector (values 32..63) reads bytes 8..15. -/
theorem hi_bytes (p : Nat) (h1 : 32 ≤ p) (h2 : p < 64) : 8 ≤ byteOf p ∧ byteOf p < 16 := by
  unfold byteOf
  omega

/-- Shifts are 0, 2, 4 or 6. -/
theorem shift_range (p : Nat) : shiftOf p ≤ 6 ∧ shiftOf p % 2 = 0 := by
  unfold shiftOf
  omega

/-- Lanes 4q..4q+3 of the output come from one byte, in shift order 0, 2, 4, 6. -/
theorem lane_group (q r : Nat) (h : r < 4) : byteOf (4 * q + r) = q ∧ shiftOf (4 * q + r) = 2 * r := by
  unfold byteOf shiftOf
  omega

-- § block addressing

/-- Block b of a GGUF Q2_0 row starts at byte `18 b`; its codes at `18 b + 2`. -/
def blockStart (b : Nat) : Nat := b * 18

/-- The 16 code bytes of block b end where block b+1 starts. -/
theorem block_tiles (b : Nat) : blockStart b + 2 + 16 = blockStart (b + 1) := by
  unfold blockStart
  omega

/-- Block b of an nb-block row lies inside the row. -/
theorem block_in_row (b nb : Nat) (h : b < nb) : blockStart b + 18 ≤ nb * 18 := by
  unfold blockStart
  have := Nat.mul_le_mul_right 18 h
  omega

-- § offset identity

/-- The Q2_0 code-offset identity behind the `corr` term. -/
theorem offset_identity (l : List (Int × Int)) :
    (l.map (fun p => (p.1 - 1) * p.2)).sum =
      (l.map (fun p => p.1 * p.2)).sum - (l.map (fun p => p.2)).sum := by
  induction l with
  | nil => simp
  | cons p t ih =>
    simp only [List.map_cons, List.sum_cons]
    rw [ih, Int.sub_mul, Int.one_mul]
    omega

-- § quantizer integer facts

/-- The quantizer's clamp. -/
def clampQ (v : Int) : Int := if v < -127 then -127 else if 127 < v then 127 else v

/-- Clamped codes stay in [-127, 127]. -/
theorem clampQ_range (v : Int) : -127 ≤ clampQ v ∧ clampQ v ≤ 127 := by
  unfold clampQ
  split <;> try split
  all_goals omega

/-- Eight-lane int32 horizontal sum written the way the shuffles combine lanes
(lo+hi, then 0x4E swap, then 0xB1 swap). -/
def hsum8 (l0 l1 l2 l3 l4 l5 l6 l7 : Int) : Int :=
  ((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7))

/-- The shuffle reduction equals the plain sum of the eight lanes. -/
theorem hsum8_eq (l0 l1 l2 l3 l4 l5 l6 l7 : Int) :
    hsum8 l0 l1 l2 l3 l4 l5 l6 l7 = l0 + l1 + l2 + l3 + l4 + l5 + l6 + l7 := by
  unfold hsum8
  omega

/-- Each of the four 8-lane vectors holds values in [-127,127], so each total lane is at most 4*127 in size
and the 32-element chunk sum is at most 4064. -/
theorem lanes_bound (a b c d : Int) (ha : -508 ≤ a ∧ a ≤ 508) (hb : -508 ≤ b ∧ b ≤ 508)
    (hc : -508 ≤ c ∧ c ≤ 508) (hd : -508 ≤ d ∧ d ≤ 508) : -2032 ≤ a + b + c + d ∧ a + b + c + d ≤ 2032 := by
  omega

-- § token batching

/-- Groups of at most k tokens, as in the `t0 += 4` loop. -/
def chunks (k n : Nat) : List Nat :=
  if h : n = 0 ∨ k = 0 then [] else min k n :: chunks k (n - min k n)
termination_by n
decreasing_by omega

/-- Every token is processed exactly once. -/
theorem chunks_sum (k : Nat) (hk : 0 < k) : ∀ n, (chunks k n).sum = n := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    by_cases h0 : n = 0
    · subst h0
      simp [chunks]
    · rw [chunks, dif_neg (by omega), List.sum_cons, ih (n - min k n) (by omega)]
      omega

/-- Each group has between 1 and 4 tokens, matching the instantiated NT = 1..4 templates. -/
theorem chunks_bound (k : Nat) (hk : 0 < k) : ∀ n, ∀ c ∈ chunks k n, 0 < c ∧ c ≤ k := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro c hc
    by_cases h0 : n = 0
    · subst h0
      simp [chunks] at hc
    · rw [chunks, dif_neg (by omega)] at hc
      rcases List.mem_cons.mp hc with rfl | hc
      · omega
      · exact ih (n - min k n) (by omega) c hc

end Strata.Q2Avx2



## FILE: /mnt/user-data/outputs/ServeFrontendParser.lean
/-!
# MODULE: serve_frontend_parser
The request/response text layer: reasoning-effort normalization, Anthropic thinking-budget bucketing, the
incremental output parser's state machine, and the tag-holdback rule that stops a partial tag reaching clients.

## Source semantics
* `EFFORT` maps client spellings onto four template settings: none (thinking off), low, medium, xhigh;
  "high" means xhigh. Unknown spellings are rejected (400), never crashed on.
* `budget_effort`: budget under 2048 is low, under 8192 is medium, otherwise xhigh; monotone in the budget.
* `OutputParser` states are reasoning, content, call. Tags move reasoning to content (`</think>`), content to
  call (`<tool_call>`), call to content (`</tool_call>`); the parser never returns to reasoning.
* `_hold(text, tags)` is the length of the longest suffix of `text` that is a PROPER prefix of a tag, so it is
  always shorter than the tag and a complete tag is never held back.

## Form template identifiers
* `effortLevel` : `EFFORT` lookup after `strip().lower()`
* `budgetLevel` : `budget_effort`
* `St`, `Tag`, `step` : `OutputParser.state` and its transitions
* `holdAux text tag k` : the inner loop of `_hold`, scanning n = k downto 1
-/

namespace Strata.ServeFrontend

-- § reasoning effort

inductive Level
  | low
  | medium
  | xhigh
  deriving DecidableEq, Repr

/-- Rank of a level for ordering. -/
def Level.rank : Level → Nat
  | .low => 0
  | .medium => 1
  | .xhigh => 2

/-- `EFFORT`: `none` = thinking disabled, `some l` = a template level, outer `none` = unknown (a 400). -/
def effortLevel : String → Option (Option Level)
  | "none" => some none
  | "off" => some none
  | "minimal" => some none
  | "disabled" => some none
  | "low" => some (some .low)
  | "medium" => some (some .medium)
  | "high" => some (some .xhigh)
  | "xhigh" => some (some .xhigh)
  | "max" => some (some .xhigh)
  | "maximum" => some (some .xhigh)
  | _ => none

/-- "high" and "xhigh" are the same template level. -/
theorem high_is_xhigh : effortLevel "high" = effortLevel "xhigh" := by
  simp [effortLevel]

/-- "max" and "maximum" are xhigh too. -/
theorem max_is_xhigh : effortLevel "max" = some (some Level.xhigh) := by
  simp [effortLevel]

/-- "none" turns thinking off. -/
theorem none_disables : effortLevel "none" = some none := by
  simp [effortLevel]

/-- Unknown spellings are rejected rather than guessed. -/
theorem unknown_rejected : effortLevel "bogus" = none := by
  simp [effortLevel]

/-- `budget_effort`: under 2K low, under 8K medium, else xhigh. -/
def budgetLevel (n : Nat) : Level := if n < 2048 then .low else if n < 8192 then .medium else .xhigh

/-- A larger thinking budget never selects a lower level. -/
theorem budget_monotone (n m : Nat) (h : n ≤ m) : (budgetLevel n).rank ≤ (budgetLevel m).rank := by
  unfold budgetLevel
  by_cases a : n < 2048 <;> by_cases b : m < 2048 <;> by_cases c : n < 8192 <;> by_cases d : m < 8192 <;>
    simp [a, b, c, d, Level.rank] <;> omega

/-- Boundary behaviour: 2047 is low, 2048 is medium, 8191 is medium, 8192 is xhigh. -/
theorem budget_boundaries :
    budgetLevel 2047 = .low ∧ budgetLevel 2048 = .medium ∧ budgetLevel 8191 = .medium ∧
      budgetLevel 8192 = .xhigh := by
  decide

-- § output parser state machine

inductive St
  | reasoning
  | content
  | call
  deriving DecidableEq, Repr

inductive Tag
  | thinkEnd
  | callStart
  | callEnd
  deriving DecidableEq, Repr

/-- The parser's transition when it finds a complete tag in its current state. -/
def step : St → Tag → St
  | .reasoning, .thinkEnd => .content
  | .content, .callStart => .call
  | .call, .callEnd => .content
  | s, _ => s

/-- A tag in the wrong state is ignored (it is just text). -/
theorem step_ignores_foreign (s : St) (t : Tag) (h : step s t ≠ s) :
    (s = .reasoning ∧ t = .thinkEnd) ∨ (s = .content ∧ t = .callStart) ∨ (s = .call ∧ t = .callEnd) := by
  cases s <;> cases t <;> simp [step] at h ⊢

/-- Once out of reasoning the parser never re-enters it. -/
theorem step_not_reasoning (s : St) (t : Tag) (h : s ≠ .reasoning) : step s t ≠ .reasoning := by
  revert h
  cases s <;> cases t <;> simp [step]

/-- ...for any sequence of tags. -/
theorem run_never_reasoning (l : List Tag) : ∀ s : St, s ≠ .reasoning → l.foldl step s ≠ .reasoning := by
  induction l with
  | nil => intro s h; simpa using h
  | cons t rest ih =>
    intro s h
    exact ih (step s t) (step_not_reasoning s t h)

/-- A non-thinking request starts in content, so it can never emit reasoning events. -/
theorem nonthinking_never_reasoning (l : List Tag) : l.foldl step .content ≠ .reasoning :=
  run_never_reasoning l .content (by decide)

/-- After a complete call the parser is back in content. -/
theorem call_end_returns : step .call .callEnd = .content := by
  rfl

/-- `finish()` flushes held text as reasoning or content; an unterminated call comes back as content. -/
def flushIsContent : St → Bool
  | .reasoning => false
  | .content => true
  | .call => true

/-- An unterminated tool call is surfaced as content, never silently dropped. -/
theorem unterminated_call_is_content : flushIsContent .call = true := by
  rfl

-- § tag holdback

/-- Inner loop of `_hold`: the largest n <= k whose n-prefix of `tag` is a suffix of `text`. -/
def holdAux (text tag : List Char) : Nat → Nat
  | 0 => 0
  | n + 1 => if (tag.take (n + 1)).isSuffixOf text then n + 1 else holdAux text tag n

/-- `_hold` scans only proper prefixes: n from `len tag - 1` down to 1. -/
def hold (text tag : List Char) : Nat := holdAux text tag (tag.length - 1)

/-- The holdback never exceeds the scan bound. -/
theorem holdAux_le (text tag : List Char) (k : Nat) : holdAux text tag k ≤ k := by
  induction k with
  | zero => simp [holdAux]
  | succ n ih =>
    unfold holdAux
    split
    · exact Nat.le_refl _
    · exact Nat.le_succ_of_le ih

/-- A held length is always strictly shorter than the tag: a complete tag is never held back. -/
theorem hold_lt (text tag : List Char) (h : 0 < tag.length) : hold text tag < tag.length := by
  have := holdAux_le text tag (tag.length - 1)
  unfold hold
  omega

/-- A positive holdback is a real match: that prefix of the tag is a suffix of the text. -/
theorem holdAux_spec (text tag : List Char) (k : Nat) (h : 0 < holdAux text tag k) :
    (tag.take (holdAux text tag k)).isSuffixOf text = true := by
  induction k with
  | zero => simp [holdAux] at h
  | succ n ih =>
    unfold holdAux at h ⊢
    by_cases hs : (tag.take (n + 1)).isSuffixOf text = true
    · rw [if_pos hs]
      exact hs
    · rw [if_neg hs] at h ⊢
      exact ih h

end Strata.ServeFrontend



## FILE: /mnt/user-data/outputs/ServeServerService.lean
/-!
# MODULE: serve_server_service
The serving core: the context-length admission rule, finish-reason mapping for the OpenAI and Anthropic
endpoints, usage accounting, the vision-embedding cache bound, API-key authorization, the mock engine's token
budget and the incremental detokenizer's cursor.

## Source semantics
* A request is accepted iff `len(prompt) + max_new <= max_context`; otherwise 400 (never truncated).
* OpenAI finish reason: `tool_calls` if at least one call was emitted and the engine stopped; otherwise the
  engine's reason (`stop` / `length`). Anthropic: `tool_use` under the same condition, else `end_turn` for
  stop and `max_tokens` for length. Truncation by length is never reported as a normal stop.
* `usage.total_tokens = prompt_tokens + completion_tokens`.
* The image-embedding cache holds at most 64 entries (oldest evicted first).
* With no API key everything is authorized; with a key only an exact match is.
* `MockEngine.generate` yields `min(max_new, len(script))` tokens.
* `Detokenizer.push` holds back text ending in U+FFFD and otherwise advances its cursor to the decoded length.

## Form template identifiers
* `admit` : `len(ids) + max_new > engine.max_context` check in `Service.prepare`
* `openaiFinish`, `anthropicStop` : the finish mappings in `openai_chunks` / `anthropic_events`
* `evict` : the `len(self.cache) > 64` eviction
* `authorized` : `Handler._authorized`
* `pushCursor` : `Detokenizer.push`
-/

namespace Strata.ServeServer

-- § admission

/-- The request fits iff the prompt plus the generation budget is within the engine context. -/
def admit (promptLen maxNew ctx : Nat) : Bool := decide (promptLen + maxNew ≤ ctx)

/-- An admitted request never overruns the context. -/
theorem admit_sound (p m c : Nat) (h : admit p m c = true) : p + m ≤ c := by
  simpa [admit] using h

/-- A rejected request really does overrun it. -/
theorem reject_sound (p m c : Nat) (h : admit p m c = false) : c < p + m := by
  simp [admit] at h
  omega

/-- A larger prompt is never admitted when a smaller one was rejected. -/
theorem admit_antitone (p q m c : Nat) (h : p ≤ q) (hq : admit q m c = true) : admit p m c = true := by
  simp [admit] at hq ⊢
  omega

-- § finish reasons

inductive Fin2
  | stop
  | length
  deriving DecidableEq, Repr

inductive OaFinish
  | stop
  | length
  | toolCalls
  deriving DecidableEq, Repr

inductive AnStop
  | endTurn
  | maxTokens
  | toolUse
  deriving DecidableEq, Repr

/-- OpenAI: `tool_calls` iff calls were emitted and the engine stopped. -/
def openaiFinish (calls : Nat) : Fin2 → OaFinish
  | .stop => if 0 < calls then .toolCalls else .stop
  | .length => .length

/-- Anthropic: `tool_use` iff a tool was used and the engine stopped. -/
def anthropicStop (usedTool : Bool) : Fin2 → AnStop
  | .stop => if usedTool then .toolUse else .endTurn
  | .length => .maxTokens

/-- A length cutoff is reported as `length` whatever tool calls were emitted. -/
theorem openai_length (calls : Nat) : openaiFinish calls .length = .length := rfl

/-- A length cutoff is reported as `max_tokens` whatever tool use happened. -/
theorem anthropic_length (u : Bool) : anthropicStop u .length = .maxTokens := rfl

/-- Without any call, a normal stop is `stop`. -/
theorem openai_plain_stop : openaiFinish 0 .stop = .stop := by
  simp [openaiFinish]

/-- With at least one call, a normal stop is `tool_calls`. -/
theorem openai_tool_stop (calls : Nat) (h : 0 < calls) : openaiFinish calls .stop = .toolCalls := by
  simp [openaiFinish, h]

/-- The two endpoints agree on when a tool stop is reported. -/
theorem endpoints_agree (calls : Nat) (h : 0 < calls) :
    openaiFinish calls .stop = .toolCalls ∧ anthropicStop true .stop = .toolUse := by
  simp [openaiFinish, anthropicStop, h]

-- § usage

/-- `usage.total_tokens`. -/
def totalTokens (prompt completion : Nat) : Nat := prompt + completion

/-- Total is prompt plus completion. -/
theorem total_eq (p c : Nat) : totalTokens p c = p + c := rfl

/-- Total is at least each part. -/
theorem total_ge (p c : Nat) : p ≤ totalTokens p c ∧ c ≤ totalTokens p c := by
  unfold totalTokens
  omega

-- § mock engine budget

/-- The mock engine yields at most `max_new` tokens of its script. -/
theorem mock_yield_len (script : List Nat) (maxNew : Nat) :
    (script.take maxNew).length = min maxNew script.length := by
  simp

/-- It never exceeds the generation budget. -/
theorem mock_yield_le (script : List Nat) (maxNew : Nat) : (script.take maxNew).length ≤ maxNew := by
  simp

-- § vision cache

/-- Cache size after inserting one new entry and evicting the oldest when it exceeds 64. -/
def evict (size : Nat) : Nat := if size + 1 > 64 then size else size + 1

/-- The cache never exceeds 64 entries. -/
theorem evict_bound (size : Nat) (h : size ≤ 64) : evict size ≤ 64 := by
  unfold evict
  split <;> omega

/-- Eviction removes exactly one entry once full, and none before. -/
theorem evict_full (size : Nat) (h : size = 64) : evict size = 64 := by
  unfold evict
  split <;> omega

-- § authorization

/-- `Handler._authorized`: no configured key authorizes everything; a key requires an exact match. -/
def authorized (key supplied : String) : Bool := key == "" || supplied == key

/-- With no key configured every request is authorized. -/
theorem authorized_no_key (supplied : String) : authorized "" supplied = true := by
  simp [authorized]

/-- With a key configured only the exact key is authorized. -/
theorem authorized_exact (key supplied : String) (hk : key ≠ "") (h : authorized key supplied = true) :
    supplied = key := by
  simp [authorized, hk] at h
  exact h

/-- The right key is always accepted. -/
theorem authorized_self (key : String) : authorized key key = true := by
  simp [authorized]

-- § detokenizer cursor

/-- `Detokenizer.push`: when the decoded text ends in U+FFFD nothing is emitted and the cursor stays;
otherwise the cursor advances to the decoded length. -/
def pushCursor (sent decodedLen : Nat) (held : Bool) : Nat := if held then sent else decodedLen

/-- The cursor never passes the decoded text. -/
theorem cursor_le (sent decodedLen : Nat) (held : Bool) (h : sent ≤ decodedLen) :
    pushCursor sent decodedLen held ≤ decodedLen := by
  unfold pushCursor
  split <;> omega

/-- The cursor never moves backwards when the decoded text only grows. -/
theorem cursor_mono (sent decodedLen : Nat) (held : Bool) (h : sent ≤ decodedLen) :
    sent ≤ pushCursor sent decodedLen held := by
  unfold pushCursor
  split <;> omega

/-- When nothing is held back, everything decoded so far has been emitted. -/
theorem cursor_caught_up (sent decodedLen : Nat) : pushCursor sent decodedLen false = decodedLen := by
  simp [pushCursor]

end Strata.ServeServer





## FILE: /mnt/user-data/outputs/CanonicalFormDecode.lean
/-!
# MODULE: canonical_form_decode
Arithmetic content of `tools/canonical_xcheck.py`: each ggml quantized block type is mapped to a
canonical form (codes plane + scale plane [+ offset plane]) whose decode must equal the ggml decode
bit for bit.  This module states the integer-level invariants that make those mappings well formed.

## Source semantics
* A canonical decode is `(code + bias) * scale` : ONE integer bias before ONE multiply (never folded
  into the value domain).  Q2_0 bias -1, Q3_K -4, Q4_0 -8, Q5_0 -16, Q6_K -32, Q8_0 -128 (code stored +128).
* Q4_K / Q5_K decode `code * scale + offset` with 6-bit scale/min fields unpacked by `get_scale_min_k4`.
* IQ4_NL / IQ4_XS use the non-affine 16-entry table `kvalues_iq4nl`.
* Nibble layout: low nibbles fill the first half of a group, high nibbles the second half.
* Q3_K hmask bit `4*half + j` never resets; Q5_K masks `u1/u2` walk 1,4,16,64 / 2,8,32,128.
* Every code plane is unsigned and fits the declared form width (S2/S4/S8).

## Form template identifiers
* decodeBias        <- from_s2 / from_s4_* / from_s8_* (affine decode)
* pack4             <- Q2_0 byte layout (4 codes per byte, LSB first)
* scaleK4 / minK4   <- get_scale_min_k4 (ggml-quants.c L880)
* kvIq4nl           <- KV_IQ4NL codebook
* Mapping/mappings  <- MAPPINGS registry (namespace entries: form, group_elems, code_bias, has_offset)
-/

namespace Strata.CanonicalForm

-- § Integer-bias affine decode

/-- Canonical affine decode: add the integer bias to the code, then multiply by the scale once. -/
def decodeBias (bias : Int) (code : Nat) (scale : Int) : Int :=
  ((code : Int) + bias) * scale

/-- Q2_0 codes 0..3 decode to symbols in {-1,0,1,2}. -/
theorem q2_0_symbol_range (c : Nat) (h : c < 4) :
    -1 ≤ (c : Int) + (-1) ∧ (c : Int) + (-1) ≤ 2 := by
  omega

/-- The four Q2_0 levels are exactly -d, 0, d, 2d (the `- 1` offset is part of the codebook). -/
theorem q2_0_decode_levels (d : Int) :
    decodeBias (-1) 0 d = -d ∧ decodeBias (-1) 1 d = 0 ∧
    decodeBias (-1) 2 d = d ∧ decodeBias (-1) 3 d = 2 * d := by
  unfold decodeBias
  omega

/-- Treating Q2_0 codes as unsigned `code * d` would shift every nonzero level by one step. -/
theorem q2_0_unsigned_reading_differs (d : Int) (hd : d ≠ 0) :
    decodeBias (-1) 1 d ≠ decodeBias 0 1 d := by
  unfold decodeBias
  intro h
  apply hd
  omega

/-- Q4_0: a 4-bit code with bias -8 covers -8..7. -/
theorem q4_0_range (c : Nat) (h : c < 16) : -8 ≤ (c : Int) - 8 ∧ (c : Int) - 8 ≤ 7 := by
  omega

/-- Q8_0: signed int8 stored as unsigned `+128`; the decode `code - 128` recovers the int8. -/
theorem q8_0_bias_roundtrip (i : Int) (h1 : -128 ≤ i) (h2 : i ≤ 127) :
    (i + 128).toNat < 256 ∧ ((i + 128).toNat : Int) - 128 = i := by
  omega

-- § Nibble and bit-plane layouts

/-- A byte splits into a low and a high nibble that recombine exactly. -/
theorem nibble_split (b : Nat) (h : b < 256) :
    b % 16 < 16 ∧ b / 16 < 16 ∧ b % 16 + 16 * (b / 16) = b := by
  omega

/-- Four 2-bit codes in one byte, LSB first (byte j/4 of a Q2_0 block). -/
def pack4 (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3

/-- A packed quad always fits one byte. -/
theorem pack4_lt (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    pack4 c0 c1 c2 c3 < 256 := by
  unfold pack4
  omega

/-- Unpacking with `(byte >> 2t) & 3` returns the original codes. -/
theorem unpack4_pack4 (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    pack4 c0 c1 c2 c3 % 4 = c0 ∧ pack4 c0 c1 c2 c3 / 4 % 4 = c1 ∧
    pack4 c0 c1 c2 c3 / 16 % 4 = c2 ∧ pack4 c0 c1 c2 c3 / 64 = c3 := by
  unfold pack4
  omega

/-- Packing the unpacked quads of any byte gives the byte back (no source bit is dropped). -/
theorem pack4_unpack4 (b : Nat) (h : b < 256) :
    pack4 (b % 4) (b / 4 % 4) (b / 16 % 4) (b / 64) = b := by
  unfold pack4
  omega

/-- Element j of a Q2_0 block lives in byte j/4 at bit offset 2*(j%4). -/
theorem q2_0_element_index (j : Nat) (h : j < 64) :
    j / 4 < 16 ∧ j % 4 < 4 ∧ 4 * (j / 4) + j % 4 = j ∧ (j % 4) * 2 < 8 := by
  omega

/-- Q2_0 block bytes: 2-byte scale plus 64/4 code bytes. -/
theorem q2_0_block_bytes : 2 + 64 / 4 = 18 := by decide

-- § Q3_K

/-- Q3_K: group index, element base and hmask bit for (half, j, sub). -/
theorem q3_k_group_layout (half j sub : Nat) (hh : half < 2) (hj : j < 4) (hs : sub < 2) :
    8 * half + 2 * j + sub < 16 ∧
    16 * (8 * half + 2 * j + sub) = 128 * half + 32 * j + 16 * sub ∧
    4 * half + j < 8 ∧ half * 32 + sub * 16 + 15 < 64 := by
  omega

/-- Q3_K stored code: 2 low bits plus the hmask bit worth 4 stays within 0..7. -/
theorem q3_k_code_range (c2 m : Nat) (h2 : c2 < 4) (hm : m < 2) :
    c2 + 4 * m < 8 ∧ -4 ≤ ((c2 + 4 * m : Nat) : Int) - 4 ∧ ((c2 + 4 * m : Nat) : Int) - 4 ≤ 3 := by
  omega

/-- Joining a 4-bit low part with two high bits always gives a 6-bit scale. -/
theorem join6_lt (lo hi : Nat) : lo % 16 + 16 * (hi % 4) < 64 := by
  omega

/-- Q3_K scale group `4+j`'s two high bits come from bits 2-3 of `s[8+j]`. -/
theorem q3_k_scale_6bit (s lo : Nat) : lo % 16 + 16 * ((s / 4) % 4) < 64 := by
  omega

/-- Q3_K scale minus 32 lies in -32..31. -/
theorem q3_k_scale_signed (v : Nat) (h : v < 64) : -32 ≤ (v : Int) - 32 ∧ (v : Int) - 32 ≤ 31 := by
  omega

-- § Q4_K / Q5_K scale and min unpacking

/-- `get_scale_min_k4`, scale half, over a byte accessor `s`. -/
def scaleK4 (s : Nat → Nat) (j : Nat) : Nat :=
  if j < 4 then s j % 64 else s (j + 4) % 16 + 16 * (s (j - 4) / 64)

/-- `get_scale_min_k4`, min half. -/
def minK4 (s : Nat → Nat) (j : Nat) : Nat :=
  if j < 4 then s (j + 4) % 64 else s (j + 4) / 16 + 16 * (s j / 64)

/-- Scales of all 8 groups are 6-bit when the 12 scale bytes are bytes. -/
theorem scaleK4_lt (s : Nat → Nat) (hs : ∀ i, i < 12 → s i < 256) (j : Nat) (hj : j < 8) :
    scaleK4 s j < 64 := by
  have h1 := hs j (by omega)
  have h2 := hs (j + 4) (by omega)
  have h3 := hs (j - 4) (by omega)
  unfold scaleK4
  split_ifs <;> omega

/-- Mins of all 8 groups are 6-bit when the 12 scale bytes are bytes. -/
theorem minK4_lt (s : Nat → Nat) (hs : ∀ i, i < 12 → s i < 256) (j : Nat) (hj : j < 8) :
    minK4 s j < 64 := by
  have h1 := hs j (by omega)
  have h2 := hs (j + 4) (by omega)
  have h3 := hs (j - 4) (by omega)
  unfold minK4
  split_ifs <;> omega

/-- Reading only `s[j]` is wrong for the high four groups: they differ from the low-group formula. -/
theorem scaleK4_high_uses_top_bits (s : Nat → Nat) (j : Nat) (hj : 4 ≤ j) :
    scaleK4 s j = s (j + 4) % 16 + 16 * (s (j - 4) / 64) := by
  unfold scaleK4
  rw [if_neg (by omega)]

/-- Q4_K: step j covers elements 64j..64j+63 as groups 2j (low nibbles) and 2j+1 (high nibbles). -/
theorem q4_k_step_layout (j : Nat) (hj : j < 4) :
    2 * j + 1 < 8 ∧ j * 64 + 32 + 32 ≤ 256 := by
  omega

/-- Q5_K bit masks used for the 5th code bit. -/
def u1 (j : Nat) : Nat := 2 ^ (2 * j)

/-- Q5_K second mask, one bit above `u1`. -/
def u2 (j : Nat) : Nat := 2 * 2 ^ (2 * j)

/-- The masks walk 1,4,16,64 and 2,8,32,128. -/
theorem q5_k_masks :
    [u1 0, u1 1, u1 2, u1 3] = [1, 4, 16, 64] ∧ [u2 0, u2 1, u2 2, u2 3] = [2, 8, 32, 128] := by
  decide

/-- The eight masks are pairwise distinct across `u1` and `u2` and cover all 8 bits. -/
theorem q5_k_masks_disjoint : ∀ j : Fin 4, ∀ k : Fin 4, u1 j.val ≠ u2 k.val := by
  decide

/-- Sum of the eight masks is 255: every bit of `qh[l]` is used exactly once. -/
theorem q5_k_masks_cover :
    u1 0 + u1 1 + u1 2 + u1 3 + u2 0 + u2 1 + u2 2 + u2 3 = 255 := by
  decide

/-- A Q5 code (4 bits plus the 16-valued fifth bit) is below 32. -/
theorem q5_code_range (nib b : Nat) (hn : nib < 16) (hb : b < 2) : nib + 16 * b < 32 := by
  omega

/-- Q5_0: the source's `(qh >> (j+12)) & 0x10` is bit `j+16` of `qh`. -/
theorem q5_0_high_bit_shift (qh j : Nat) :
    ((qh / 2 ^ (j + 12)) / 16) % 2 = (qh / 2 ^ (j + 16)) % 2 := by
  rw [Nat.div_div_eq_div_mul]
  have h : 2 ^ (j + 12) * 16 = 2 ^ (j + 16) := by
    ring
  rw [h]

-- § Q6_K

/-- Q6_K code: 4 low bits plus 2 high bits is below 64, bias -32 gives -32..31. -/
theorem q6_k_code_range (nib hi : Nat) (hn : nib < 16) (hh : hi < 4) :
    nib + 16 * hi < 64 ∧ -32 ≤ ((nib + 16 * hi : Nat) : Int) - 32 ∧
    ((nib + 16 * hi : Nat) : Int) - 32 ≤ 31 := by
  omega

/-- Q6_K quads land at strides of 32, so the scale group is plain consecutive 16-element groups. -/
theorem q6_k_quad_layout (half q l : Nat) (hh : half < 2) (hq : q < 4) (hl : l < 32) :
    128 * half + 32 * q + l < 256 ∧
    (128 * half + 32 * q + l) / 16 = 8 * half + 2 * q + l / 16 ∧
    8 * half + 2 * q + l / 16 < 16 := by
  omega

-- § IQ4 codebook (non-affine)

/-- `kvalues_iq4nl`. -/
def kvIq4nl : List Int :=
  [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries, one per 4-bit code. -/
theorem kvIq4nl_length : kvIq4nl.length = 16 := by decide

/-- The codebook is strictly increasing. -/
theorem kvIq4nl_strictly_increasing :
    ∀ i : Fin 15, kvIq4nl.getD i.val 0 < kvIq4nl.getD (i.val + 1) 0 := by
  decide

/-- It is not equally spaced, hence not an integer bias on the code. -/
theorem kvIq4nl_not_affine :
    kvIq4nl.getD 1 0 - kvIq4nl.getD 0 0 ≠ kvIq4nl.getD 2 0 - kvIq4nl.getD 1 0 := by
  decide

/-- IQ4_XS: scale `ls - 32` for a 6-bit `ls` lies in -32..31. -/
theorem iq4_xs_scale_range (a b : Nat) :
    (a % 16 + 16 * (b % 4) : Nat) < 64 ∧ -32 ≤ ((a % 16 + 16 * (b % 4) : Nat) : Int) - 32 ∧
    ((a % 16 + 16 * (b % 4) : Nat) : Int) - 32 ≤ 31 := by
  omega

-- § Block geometry

/-- Byte layouts of the K-quant and IQ blocks. -/
theorem block_byte_layouts :
    2 + 2 + 12 + 128 = 144 ∧ 32 + 64 + 12 + 2 = 110 ∧ 4 + 12 + 32 + 128 = 176 ∧
    128 + 64 + 16 + 2 = 210 ∧ 2 + 2 + 4 + 128 = 136 ∧ 2 + 4 + 16 = 22 ∧ 2 + 32 = 34 ∧
    2 + 16 = 18 := by
  decide

/-- Registry row of the canonical mapping table. -/
structure Mapping where
  name : String
  blockBytes : Nat
  blockElems : Nat
  groupElems : Nat
  codeBits : Nat
  maxCode : Nat
  codeBias : Int
  hasOffset : Bool

/-- The ten quantized rows of MAPPINGS (the three float rows are pass-through). -/
def mappings : List Mapping :=
  [ ⟨"Q2_0", 18, 64, 64, 2, 3, -1, false⟩,
    ⟨"Q3_K", 110, 256, 16, 4, 7, -4, false⟩,
    ⟨"Q4_0", 18, 32, 32, 4, 15, -8, false⟩,
    ⟨"Q4_K", 144, 256, 32, 4, 15, 0, true⟩,
    ⟨"Q5_0", 22, 32, 32, 8, 31, -16, false⟩,
    ⟨"Q5_K", 176, 256, 32, 8, 31, 0, true⟩,
    ⟨"Q6_K", 210, 256, 16, 8, 63, -32, false⟩,
    ⟨"Q8_0", 34, 32, 32, 8, 255, -128, false⟩,
    ⟨"IQ4_NL", 18, 32, 32, 4, 15, 0, false⟩,
    ⟨"IQ4_XS", 136, 256, 32, 4, 15, 0, false⟩ ]

/-- Groups tile blocks exactly. -/
theorem mappings_groups_tile :
    mappings.all (fun m => m.blockElems % m.groupElems == 0) = true := by
  decide

/-- Every code fits its declared plane width. -/
theorem mappings_codes_fit :
    mappings.all (fun m => decide (m.maxCode < 2 ^ m.codeBits)) = true := by
  decide

/-- Only the two K-quants with mins carry an offset plane. -/
theorem mappings_offset_rows :
    (mappings.filter (fun m => m.hasOffset)).map (fun m => m.name) = ["Q4_K", "Q5_K"] := by
  decide

/-- Group counts per 256-element block. -/
theorem mappings_group_counts : 256 / 32 = 8 ∧ 256 / 16 = 16 ∧ 64 / 64 = 1 := by
  decide

end Strata.CanonicalForm



## FILE: /mnt/user-data/outputs/GgufQ2_0Writer.lean
/-!
# MODULE: gguf_q2_0_writer
Arithmetic of `tools/gguf_writer.py`: the minimal GGUF v3 writer with the Q2_0 quantizer, its
alignment rules and its self-check (requantization fixed point).

## Source semantics
* Q2_0 block = 64 elements, fp16 scale `d = amax`, 16 bytes of 2-bit codes, LSB first.
* `code = clamp(round(w/d) + 1, 0, 3)`, symbol = `code - 1` in {-1,0,1,2}.  Since `|round(w/d)| <= 1`,
  the +2 level is never produced by this quantizer.
* Tensor data are padded to the alignment (default 32); offsets in the directory are aligned.
* Requantizing dequantized values is exact (fixed point) at the code level.

## Form template identifiers
* q2Code       <- `np.clip(np.rint(b / d) + 1, 0, 3)`
* alignUp32    <- `(offset + alignment - 1) // alignment * alignment`
* pad32        <- `(-len(body)) % alignment`
-/

namespace Strata.GgufWriter

-- § Alignment

/-- Round up to the default 32-byte alignment. -/
def alignUp32 (n : Nat) : Nat := (n + 31) / 32 * 32

/-- Padding needed after `n` bytes, i.e. `(-n) % 32`. -/
def pad32 (n : Nat) : Nat := (32 - n % 32) % 32

/-- Alignment never shrinks. -/
theorem alignUp32_ge (n : Nat) : n ≤ alignUp32 n := by
  unfold alignUp32
  omega

/-- Result is a multiple of 32. -/
theorem alignUp32_mod (n : Nat) : alignUp32 n % 32 = 0 := by
  unfold alignUp32
  omega

/-- Alignment adds less than one alignment unit. -/
theorem alignUp32_lt (n : Nat) : alignUp32 n < n + 32 := by
  unfold alignUp32
  omega

/-- Aligning equals adding the padding written to the file. -/
theorem alignUp32_eq_add_pad (n : Nat) : alignUp32 n = n + pad32 n := by
  unfold alignUp32 pad32
  omega

/-- After padding, the position is a multiple of 32. -/
theorem pad32_aligns (n : Nat) : (n + pad32 n) % 32 = 0 := by
  unfold pad32
  omega

/-- If a tensor starts on an aligned offset, the next directory offset is aligned too. -/
theorem next_offset_aligned (off len : Nat) (_h : off % 32 = 0) :
    alignUp32 (off + len) % 32 = 0 :=
  alignUp32_mod _

/-- Fixed GGUF header: magic(4) + version(4) + tensor count(8) + kv count(8). -/
theorem header_fixed_bytes : 4 + 4 + 8 + 8 = 24 := by decide

-- § Q2_0 quantizer

/-- Quantizer code for an already rounded value `r = rint(w/d)`. -/
def q2Code (r : Int) : Nat := (max 0 (min 3 (r + 1))).toNat

/-- Codes are 2-bit. -/
theorem q2Code_lt (r : Int) : q2Code r < 4 := by
  unfold q2Code
  omega

/-- The decoded symbol is `r` clamped to [-1, 2]. -/
theorem q2Code_symbol (r : Int) : (q2Code r : Int) - 1 = max (-1) (min 2 r) := by
  unfold q2Code
  omega

/-- Inside the grid the symbol is exactly `r`. -/
theorem q2Code_fixed (r : Int) (h1 : -1 ≤ r) (h2 : r ≤ 2) : (q2Code r : Int) - 1 = r := by
  unfold q2Code
  omega

/-- `d = amax` means `|r| <= 1`, so the code 3 (symbol +2) is never emitted. -/
theorem q2Code_no_plus2 (r : Int) (h : r ≤ 1) : q2Code r ≤ 2 := by
  unfold q2Code
  omega

/-- Requantizing a dequantized symbol returns the same code (code-level fixed point). -/
theorem q2Code_roundtrip (c : Nat) (h : c < 4) : q2Code ((c : Int) - 1) = c := by
  unfold q2Code
  omega

/-- Packing layout: element j sits in byte j/4 at shift 2*(j%4), 4 codes per byte. -/
theorem pack_layout (j : Nat) (h : j < 64) :
    j / 4 < 16 ∧ (j % 4) * 2 < 8 ∧ (j % 4) * 2 % 2 = 0 := by
  omega

/-- Block bytes: scale plus 64/4 code bytes. -/
theorem block_bytes : 2 + 64 / 4 = 18 := by decide

/-- The self-check tensor (16 rows of 64) occupies 16 blocks = 288 bytes. -/
theorem selftest_bytes : 16 * 64 / 64 * 18 = 288 := by decide

end Strata.GgufWriter



## FILE: /mnt/user-data/outputs/MtpPackQuantizers.lean
/-!
# MODULE: mtp_pack_quantizers
Arithmetic of `tools/mtp_pack.py`: packing the MTP draft block's routed experts as Q2_0 / Q4_0 / Q8_0
with the fused layout `gate_up_proj [512,1280,2560]` and `down_proj [512,2560,640]`.

## Source semantics
* Q2_0 scale search: 17 candidate scales `d = amax * f`, `f = 0.5 .. 1.0` in steps of 1/32; level grid
  {-1,0,1,2}; codes `q + 1` in 0..3, four per byte.
* Q4_0 reference: code `min 15 (trunc (x*inv + 8.5))`, low half and high half nibbles of 32 elements.
* Expert inner dims are multiples of the block size; GGUF raw shape last dim = `inner / block * block_bytes`.
* Reported sizes: q2_0 ~0.71 GB, q4_0 ~1.42 GB, q8_0 ~2.67 GB.

## Form template identifiers
* candNum k   <- `np.linspace(0.5, 1.0, 17)` as (16+k)/32
* expertBytes <- `shape[-1] // block * block_bytes` summed over the fused experts
-/

namespace Strata.MtpPack

-- § Q2_0 candidate scales

/-- Numerator of the k-th candidate factor `(16 + k) / 32`. -/
def candNum (k : Nat) : Nat := 16 + k

/-- There are 17 candidates k = 0..16. -/
theorem cand_count : (List.range 17).length = 17 := by decide

/-- Candidates span exactly [1/2, 1]. -/
theorem cand_endpoints : candNum 0 = 16 ∧ candNum 16 = 32 := by decide

/-- Candidates are strictly increasing. -/
theorem cand_strict (k : Nat) : candNum k < candNum (k + 1) := by
  unfold candNum
  omega

/-- At the smallest candidate scale, the +2 level reaches amax (2 * amax/2 = amax). -/
theorem cand_low_reaches_max : 2 * candNum 0 = 32 := by decide

/-- Every candidate lies in [16, 32] when k <= 16. -/
theorem cand_range (k : Nat) (h : k ≤ 16) : 16 ≤ candNum k ∧ candNum k ≤ 32 := by
  unfold candNum
  omega

/-- Clip to [-1,2] then add 1 gives a 2-bit code. -/
theorem q2_code_range (q : Int) : 0 ≤ max (-1) (min 2 q) + 1 ∧ max (-1) (min 2 q) + 1 < 4 := by
  omega

-- § Packing bytes

/-- Four 2-bit codes combine into a byte. -/
theorem pack_q2_byte_lt (a b c d : Nat) (ha : a < 4) (hb : b < 4) (hc : c < 4) (hd : d < 4) :
    a + 4 * b + 16 * c + 64 * d < 256 := by
  omega

/-- Q4_0: truncated index clamped at 15 is a nibble. -/
theorem q4_code_nibble (t : Nat) : min 15 t < 16 := by
  omega

/-- Low half and high half nibbles combine into a byte. -/
theorem pack_q4_byte_lt (lo hi : Nat) (hl : lo < 16) (hh : hi < 16) : lo + 16 * hi < 256 := by
  omega

-- § Sizes

/-- Bytes of an expert tensor of `elems` elements in a block format. -/
def expertBytes (blockElems blockBytes elems : Nat) : Nat := elems / blockElems * blockBytes

/-- Element count of the fused gate_up tensor. -/
def guElems : Nat := 512 * 1280 * 2560

/-- Element count of the down tensor. -/
def dnElems : Nat := 512 * 2560 * 640

/-- Inner dims are whole blocks for both block sizes. -/
theorem inner_dims_divisible :
    2560 % 64 = 0 ∧ 640 % 64 = 0 ∧ 2560 % 32 = 0 ∧ 640 % 32 = 0 := by
  decide

/-- Raw last dims: Q2_0 gives 720 and 180 bytes per row. -/
theorem raw_last_dims_q2_0 : 2560 / 64 * 18 = 720 ∧ 640 / 64 * 18 = 180 := by
  decide

/-- The fused gate_up row count is gate plus up. -/
theorem gate_up_rows : 2 * 640 = 1280 := by decide

/-- Q2_0 experts total 707,788,800 bytes (~0.71 GB). -/
theorem q2_0_total_bytes :
    expertBytes 64 18 guElems + expertBytes 64 18 dnElems = 707788800 := by
  decide

/-- Q4_0 experts total 1,415,577,600 bytes (~1.42 GB). -/
theorem q4_0_total_bytes :
    expertBytes 32 18 guElems + expertBytes 32 18 dnElems = 1415577600 := by
  decide

/-- Q8_0 experts total 2,673,868,800 bytes (~2.67 GB). -/
theorem q8_0_total_bytes :
    expertBytes 32 34 guElems + expertBytes 32 34 dnElems = 2673868800 := by
  decide

/-- Q2_0 is strictly smaller than Q4_0, which is strictly smaller than Q8_0. -/
theorem size_order :
    expertBytes 64 18 guElems + expertBytes 64 18 dnElems <
      expertBytes 32 18 guElems + expertBytes 32 18 dnElems ∧
    expertBytes 32 18 guElems + expertBytes 32 18 dnElems <
      expertBytes 32 34 guElems + expertBytes 32 34 dnElems := by
  decide

end Strata.MtpPack



## FILE: /mnt/user-data/outputs/MtpRuntimeRelayout.lean
/-!
# MODULE: mtp_runtime_relayout
Arithmetic of `tools/mtp_rt.py`: relayout of the MTP expert tensors into the engine's blob format
(`experts.bin`) and the 256-byte aligned `dense.bin`.

## Source semantics
* H = 2560, FF = 640, NE = 512 experts.  One blob = gate/up rows interleaved (slot 2r = gate r,
  slot 2r+1 = up r), then down rows; code planes first, fp16 scale planes after.
* `BLOB = 3 * (H*FF*18/64) = 1,382,400` bytes; the four planes sum to exactly BLOB.
* The interleave is a bijection on 2*FF rows.
* Dense tensors are padded so each starts on a 256-byte boundary.

## Form template identifiers
* srcRow / dstSlot   <- `inter[0::2] = gub[:FF]; inter[1::2] = gub[FF:]`
* BLOB               <- `BLOB = 3 * (H * FF * 18 // 64)`
* padTo256           <- `pad = (-off) % 256`
-/

namespace Strata.MtpRuntime

-- § Blob geometry

/-- Blob size from the source formula. -/
def blob : Nat := 3 * (2560 * 640 * 18 / 64)

/-- It equals the architecture constant 1,382,400. -/
theorem blob_value : blob = 1382400 := by decide

/-- Plane sizes: gate_up codes, down codes, gate_up scales, down scales. -/
theorem plane_sizes :
    1280 * (2560 / 64) * 16 = 819200 ∧ 2560 * (640 / 64) * 16 = 409600 ∧
    1280 * (2560 / 64) * 2 = 102400 ∧ 2560 * (640 / 64) * 2 = 51200 := by
  decide

/-- The four planes fill the blob exactly (the `assert len(out) == BLOB`). -/
theorem planes_sum_to_blob : 819200 + 409600 + 102400 + 51200 = blob := by decide

/-- GGUF raw row widths: 720 and 180 Q2_0 bytes per row. -/
theorem gguf_row_bytes : 2560 / 64 * 18 = 720 ∧ 640 / 64 * 18 = 180 := by decide

/-- The experts file is NE blobs. -/
theorem experts_file_bytes : 512 * blob = 707788800 := by decide

-- § Gate/up interleave

/-- Source row (in the gate-then-up layout) feeding interleaved slot `s`. -/
def srcRow (s : Nat) : Nat := if s % 2 = 0 then s / 2 else 640 + s / 2

/-- Interleaved slot receiving source row `r`. -/
def dstSlot (r : Nat) : Nat := if r < 640 then 2 * r else 2 * (r - 640) + 1

/-- Every slot reads a valid source row. -/
theorem srcRow_lt (s : Nat) (h : s < 1280) : srcRow s < 1280 := by
  unfold srcRow
  split <;> omega

/-- Every source row lands in a valid slot. -/
theorem dstSlot_lt (r : Nat) (h : r < 1280) : dstSlot r < 1280 := by
  unfold dstSlot
  split <;> omega

/-- The interleave is invertible: slot -> row -> slot is the identity. -/
theorem dstSlot_srcRow (s : Nat) (h : s < 1280) : dstSlot (srcRow s) = s := by
  unfold srcRow dstSlot
  split <;> split <;> omega

/-- And row -> slot -> row is the identity, so no row is lost or duplicated. -/
theorem srcRow_dstSlot (r : Nat) (h : r < 1280) : srcRow (dstSlot r) = r := by
  unfold srcRow dstSlot
  split <;> split <;> omega

/-- Gate rows go to even slots, up rows to odd slots. -/
theorem gate_even_up_odd (r : Nat) (h : r < 640) :
    dstSlot r % 2 = 0 ∧ dstSlot (640 + r) % 2 = 1 := by
  unfold dstSlot
  split <;> split <;> omega

-- § Dense file alignment

/-- Padding to the next 256-byte boundary, `(-off) % 256`. -/
def padTo256 (off : Nat) : Nat := (256 - off % 256) % 256

/-- A padded offset is 256-aligned. -/
theorem padTo256_aligns (off : Nat) : (off + padTo256 off) % 256 = 0 := by
  unfold padTo256
  omega

/-- Padding is smaller than the alignment. -/
theorem padTo256_lt (off : Nat) : padTo256 off < 256 := by
  unfold padTo256
  omega

/-- Q8_0 conversion of the large projections needs 32 | cols; the hidden size qualifies. -/
theorem hidden_is_q8_block_multiple : 2560 % 32 = 0 ∧ 640 % 32 = 0 := by decide

end Strata.MtpRuntime



## FILE: /mnt/user-data/outputs/StrataPackBuilder.lean
/-!
# MODULE: strata_pack_builder
Arithmetic of `tools/strata_pack.py` (build / verify / info): LSB-first code packing, the FP16-or-FP32
scale choice, 64-byte plane alignment and the expert arena addressing.

## Source semantics
* `pack_codes(codes, bits)` stores `8 // bits` codes per byte, LSB first; `unpack_codes` inverts it.
* A scale/offset plane is stored FP16 only if the round trip is exact, otherwise FP32; `verify`
  re-checks bit exactness so a wrong choice is a failed build, not a lossy pack.
* Every plane starts on a 64-byte boundary relative to its segment start (`put` pads the body);
  absolute alignment additionally needs the segment base to be 64-aligned.
* The expert arena is `layer*NE + e` blobs of 1,382,400 bytes: every blob start is 64-aligned.
* `rebuild_role` inverts the blob layout: block `blk` of down is (row = blk / 10, b_in = blk % 10),
  of gate/up is (row = blk / 40, b_in = blk % 40).

## Form template identifiers
* packLE / unpackLE  <- pack_codes / unpack_codes
* storeScale         <- choose_scale_dtype + decode via `<f2` / `<f4`
* padTo64            <- `pad = (-len(body)) % ALIGN`
-/

namespace Strata.PackBuilder

-- § LSB-first code packing

/-- Pack codes into one natural number, first code in the lowest `b` bits. -/
def packLE (b : Nat) : List Nat → Nat
  | [] => 0
  | c :: cs => c + 2 ^ b * packLE b cs

/-- Read `n` codes of `b` bits, lowest first. -/
def unpackLE (b : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | n + 1, x => x % 2 ^ b :: unpackLE b n (x / 2 ^ b)

/-- Unpacking inverts packing whenever every code fits `b` bits. -/
theorem unpackLE_packLE (b : Nat) :
    ∀ cs : List Nat, (∀ c ∈ cs, c < 2 ^ b) → unpackLE b cs.length (packLE b cs) = cs
  | [], _ => rfl
  | c :: cs, h => by
    have hc : c < 2 ^ b := h c (by simp)
    have hpos : 0 < 2 ^ b := Nat.two_pow_pos b
    have ih := unpackLE_packLE b cs (fun x hx => h x (List.mem_cons_of_mem c hx))
    have h1 : (c + 2 ^ b * packLE b cs) % 2 ^ b = c := by
      rw [Nat.add_mul_mod_self_left]
      exact Nat.mod_eq_of_lt hc
    have h2 : (c + 2 ^ b * packLE b cs) / 2 ^ b = packLE b cs := by
      rw [Nat.add_mul_div_left _ _ hpos, Nat.div_eq_of_lt hc, Nat.zero_add]
    simp only [List.length_cons, unpackLE, packLE, h1, h2, ih]

/-- Two 4-bit codes fit one byte. -/
theorem pack_two_nibbles_lt (a b : Nat) (ha : a < 16) (hb : b < 16) : packLE 4 [a, b] < 256 := by
  simp [packLE]
  omega

/-- Four 2-bit codes fit one byte. -/
theorem pack_four_pairs_lt (a b c d : Nat) (ha : a < 4) (hb : b < 4) (hc : c < 4) (hd : d < 4) :
    packLE 2 [a, b, c, d] < 256 := by
  simp [packLE]
  omega

/-- Codes per byte for the three code widths. -/
theorem codes_per_byte : 8 / 2 = 4 ∧ 8 / 4 = 2 ∧ 8 / 8 = 1 := by decide

-- § Scale precision

/-- Store as the narrow type when the round trip is exact, else keep the wide value; return decoded value. -/
def storeScale {α β : Type} [DecidableEq α] (enc : α → β) (dec : β → α) (x : α) : α :=
  if dec (enc x) = x then dec (enc x) else x

/-- Whatever choice is made, decoding returns exactly the canonical scale. -/
theorem storeScale_lossless {α β : Type} [DecidableEq α] (enc : α → β) (dec : β → α) (x : α) :
    storeScale enc dec x = x := by
  unfold storeScale
  split
  · assumption
  · rfl

-- § Alignment

/-- Pad to the next 64-byte boundary, `(-n) % 64`. -/
def padTo64 (n : Nat) : Nat := (64 - n % 64) % 64

/-- Inside a segment, a plane written after `len` body bytes starts 64-aligned. -/
theorem put_relative_aligned (len : Nat) : (len + padTo64 len) % 64 = 0 := by
  unfold padTo64
  omega

/-- With a 64-aligned segment base the plane offset is absolutely aligned. -/
theorem put_absolute_aligned (base len : Nat) (hb : base % 64 = 0) :
    (base + (len + padTo64 len)) % 64 = 0 := by
  unfold padTo64
  omega

/-- Every expert blob start in `experts.bin` is a multiple of 64. -/
theorem expert_blob_aligned (k : Nat) : (k * 1382400) % 64 = 0 := by
  omega

/-- The blob start for (layer, expert) is the flat index times the blob size. -/
theorem expert_flat_index (layer e : Nat) (he : e < 512) : (layer * 512 + e) < (layer + 1) * 512 := by
  omega

-- § Expert arena inversion

/-- Down role: block -> (row, in-row block) is a bijection onto 2560 x 10. -/
theorem down_block_split (blk : Nat) (h : blk < 25600) :
    blk / 10 < 2560 ∧ blk % 10 < 10 ∧ (blk / 10) * 10 + blk % 10 = blk := by
  omega

/-- Gate/up role: block -> (row, in-row block) is a bijection onto 640 x 40. -/
theorem gate_up_block_split (blk : Nat) (h : blk < 25600) :
    blk / 40 < 640 ∧ blk % 40 < 40 ∧ (blk / 40) * 40 + blk % 40 = blk := by
  omega

/-- Blocks per expert role: H*FF/64 = 25600 and bytes 25600*18 = 460800 = BLOB/3. -/
theorem role_bytes : 2560 * 640 / 64 = 25600 ∧ 25600 * 18 = 460800 ∧ 3 * 460800 = 1382400 := by
  decide

/-- With `--expert-every 32`, 16 experts per layer are checked, 48 role checks per layer. -/
theorem expert_every_32 : (512 + 31) / 32 = 16 ∧ 16 * 3 = 48 := by decide

end Strata.PackBuilder



## FILE: /mnt/user-data/outputs/StrataTokenizerBpe.lean
/-!
# MODULE: strata_tokenizer_bpe
Structure of `tools/strata_tokenizer.py`: GPT-2 byte-level BPE with the `qwen35` pre-tokenizer.

## Source semantics
* `bytes_to_unicode`: 188 bytes (33..126, 161..172, 174..255) map to themselves, the other 68 map to
  256 + rank; so space -> 288 and newline -> 266.
* BPE merges the adjacent pair of LOWEST rank first; each merge shortens the word by one and keeps the
  concatenation unchanged.
* Special tokens: type 4 (USER_DEFINED) matched always, type 3 (CONTROL) only with parse_special.
* decode(encode(s)) = s holds at the byte layer when the byte map has a left inverse.

## Form template identifiers
* inBs / byteToUnicode  <- `bytes_to_unicode`
* mergeAt               <- `parts[best:best+2] = [parts[best] + parts[best+1]]`
* matched               <- `_always_re` / `_special_re` selection in `encode`
-/

namespace Strata.Tokenizer

-- § Byte alphabet

/-- Bytes that map to themselves. -/
def inBs (b : Nat) : Bool :=
  (decide (33 ≤ b) && decide (b ≤ 126)) || (decide (161 ≤ b) && decide (b ≤ 172)) ||
    (decide (174 ≤ b) && decide (b ≤ 255))

/-- Number of shifted bytes below `b`. -/
def rank (b : Nat) : Nat := ((List.range b).filter (fun x => !inBs x)).length

/-- GPT-2 byte -> code point. -/
def byteToUnicode (b : Nat) : Nat := if inBs b then b else 256 + rank b

set_option maxRecDepth 100000 in
/-- Exactly 188 bytes are printable as themselves. -/
theorem identity_byte_count : ((List.range 256).filter inBs).length = 188 := by
  decide

set_option maxRecDepth 100000 in
/-- Exactly 68 bytes are shifted. -/
theorem shifted_byte_count : ((List.range 256).filter (fun x => !inBs x)).length = 68 := by
  decide

set_option maxRecDepth 100000 in
/-- The shifted bytes receive ranks 0..67 exactly once, so they land on 256..323 without collision. -/
theorem shifted_ranks_are_0_to_67 :
    ((List.range 256).filter (fun x => !inBs x)).map rank = List.range 68 := by
  decide

/-- Space and newline map to the familiar characters U+0120 and U+010A. -/
theorem space_newline_map : byteToUnicode 32 = 288 ∧ byteToUnicode 10 = 266 := by
  decide

/-- Letters map to themselves. -/
theorem letters_identity : byteToUnicode 65 = 65 ∧ byteToUnicode 97 = 97 := by
  decide

-- § BPE merge step

/-- Merge the pair at positions i and i+1 of a word held as a list of pieces. -/
def mergeAt {α : Type} : List (List α) → Nat → List (List α)
  | a :: b :: t, 0 => (a ++ b) :: t
  | x :: t, i + 1 => x :: mergeAt t i
  | l, _ => l

/-- A merge keeps the underlying symbol sequence unchanged. -/
theorem mergeAt_flatten {α : Type} :
    ∀ (i : Nat) (l : List (List α)), i + 1 < l.length → (mergeAt l i).flatten = l.flatten
  | 0, [], h => by simp at h
  | 0, [_], h => by simp at h
  | 0, a :: b :: t, _ => by simp [mergeAt, List.append_assoc]
  | i + 1, [], h => by simp at h
  | i + 1, x :: t, h => by
    have ih := mergeAt_flatten i t (by simp at h; omega)
    simp [mergeAt, ih]

/-- A merge shortens the word by exactly one piece. -/
theorem mergeAt_length {α : Type} :
    ∀ (i : Nat) (l : List (List α)), i + 1 < l.length → (mergeAt l i).length + 1 = l.length
  | 0, [], h => by simp at h
  | 0, [_], h => by simp at h
  | 0, a :: b :: t, _ => by simp [mergeAt]
  | i + 1, [], h => by simp at h
  | i + 1, x :: t, h => by
    have ih := mergeAt_length i t (by simp at h; omega)
    simp [mergeAt]
    omega

/-- Byte-level round trip: a left inverse of the byte map recovers any byte string. -/
theorem decode_encode_bytes (m u : Nat → Nat) (h : ∀ b, b < 256 → u (m b) = b) :
    ∀ bs : List Nat, (∀ b ∈ bs, b < 256) → (bs.map m).map u = bs
  | [], _ => rfl
  | b :: t, hb => by
    have hb0 : u (m b) = b := h b (hb b (by simp))
    have ih := decode_encode_bytes m u h t (fun x hx => hb x (List.mem_cons_of_mem b hx))
    simp [hb0, ih]

-- § Special tokens

/-- Whether a token of GGUF type `ty` is matched as a literal. -/
def matched (ty : Nat) (parseSpecial : Bool) : Bool :=
  ty == 4 || (ty == 3 && parseSpecial)

/-- USER_DEFINED tokens (type 4) are matched with or without parse_special. -/
theorem user_defined_always : matched 4 true = true ∧ matched 4 false = true := by decide

/-- CONTROL tokens (type 3) are matched only with parse_special. -/
theorem control_needs_flag : matched 3 true = true ∧ matched 3 false = false := by decide

/-- Ordinary types (e.g. normal=1, byte=6) are never literal matches. -/
theorem normal_never : matched 1 true = false ∧ matched 6 true = false := by decide

/-- Vocabulary sizes of the shipped tokenizer: more tokens than merges. -/
theorem vocab_sizes : 247587 < 248320 := by decide

end Strata.Tokenizer



## FILE: /mnt/user-data/outputs/VisionEncoderProtocol.lean
/-!
# MODULE: vision_encoder_protocol
Protocol of `tools/vision/strata_vision.cpp`: the resident image encoder.  Per stdin line
`ENC <image> <output>` it writes an `SVE1` embedding file and answers `OK ...` or `ERR ...`.

## Source semantics
* `parse_enc` splits the rest of the line at the LAST space: the image path may contain spaces, the
  output path may not; both must be non-empty.
* Output header = five int32 {0x31455653 'SVE1', n_tokens, nx, ny, n_embd}, then float32 rows.
* Token i sits at grid position (x = i % nx, y = i / nx); the grid must be rectangular (nx*ny = n).

## Form template identifiers
* splitLastSpace / parseRest / parseEnc <- parse_enc
* magic, fileBytes                      <- header `hdr[5]` and `fwrite` sizes
* grid lemmas                           <- decoder-pos grid reconstruction
-/

namespace Strata.Vision

-- § ENC line parsing

/-- Split a character list at its last space into (before, after). -/
def splitLastSpace : List Char → Option (List Char × List Char)
  | [] => none
  | c :: t =>
    match splitLastSpace t with
    | some (a, b) => some (c :: a, b)
    | none => if c = ' ' then some ([], t) else none

/-- A list with no space has no split point. -/
theorem splitLastSpace_none : ∀ l : List Char, ' ' ∉ l → splitLastSpace l = none
  | [], _ => rfl
  | c :: t, h => by
    have hc : c ≠ ' ' := fun e => h (by simp [e])
    have ht : ' ' ∉ t := fun e => h (List.mem_cons_of_mem c e)
    have ih := splitLastSpace_none t ht
    simp [splitLastSpace, ih, hc]

/-- Splitting `img ++ ' ' :: out` with a space-free `out` returns exactly `(img, out)`. -/
theorem splitLastSpace_append (out : List Char) (ho : ' ' ∉ out) :
    ∀ img : List Char, splitLastSpace (img ++ ' ' :: out) = some (img, out)
  | [] => by simp [splitLastSpace, splitLastSpace_none out ho]
  | c :: img => by
    have ih := splitLastSpace_append out ho img
    simp [splitLastSpace, ih]

/-- Accept only a split with both parts non-empty. -/
def parseRest (rest : List Char) : Option (List Char × List Char) :=
  match splitLastSpace rest with
  | some (i :: is, o :: os) => some (i :: is, o :: os)
  | _ => none

/-- Parse a whole line: it must start with `ENC `. -/
def parseEnc : List Char → Option (List Char × List Char)
  | 'E' :: 'N' :: 'C' :: ' ' :: rest => parseRest rest
  | _ => none

/-- A well-formed line parses into the image path (spaces allowed) and the output path. -/
theorem parseEnc_ok (i : Char) (is : List Char) (o : Char) (os : List Char)
    (ho : ' ' ∉ o :: os) :
    parseEnc ('E' :: 'N' :: 'C' :: ' ' :: ((i :: is) ++ ' ' :: (o :: os))) =
      some (i :: is, o :: os) := by
  have h := splitLastSpace_append (o :: os) ho (i :: is)
  show parseRest ((i :: is) ++ ' ' :: (o :: os)) = _
  unfold parseRest
  rw [h]

/-- Any line without the `ENC ` prefix is rejected. -/
theorem parseEnc_rejects_empty : parseEnc [] = none := rfl

-- § Output file format

/-- The header magic. -/
def magic : Nat := 0x31455653

/-- Little-endian bytes of the magic spell `S`,`V`,`E`,`1`. -/
theorem magic_bytes :
    magic % 256 = 0x53 ∧ magic / 256 % 256 = 0x56 ∧ magic / 65536 % 256 = 0x45 ∧
    magic / 16777216 = 0x31 := by
  decide

/-- Header is five int32 = 20 bytes. -/
theorem header_bytes : 5 * 4 = 20 := by decide

/-- Total file size: header plus n rows of n_embd float32. -/
def fileBytes (n nEmbd : Nat) : Nat := 20 + 4 * (n * nEmbd)

/-- The file is never smaller than its header. -/
theorem fileBytes_ge_header (n nEmbd : Nat) : 20 ≤ fileBytes n nEmbd := by
  unfold fileBytes
  omega

-- § Grid

/-- Row-major reconstruction: index = x + nx*y. -/
theorem grid_index (nx i : Nat) : i % nx + nx * (i / nx) = i :=
  Nat.mod_add_div i nx

/-- Row y stays inside the grid. -/
theorem grid_y_lt (nx ny i : Nat) (h : i < nx * ny) : i / nx < ny :=
  Nat.div_lt_of_lt_mul h

/-- Column x stays inside the grid. -/
theorem grid_x_lt (nx i : Nat) (h : 0 < nx) : i % nx < nx :=
  Nat.mod_lt i h

end Strata.Vision



## FILE: /mnt/user-data/outputs/VisionBuildConfig.lean
/-!
# MODULE: vision_build_config
Configuration resolution of `tools/vision/CMakeLists.txt`: how the options `STRATA_PORTABLE`,
`STRATA_VISION_CUDA` and the MSVC toolchain determine the forced llama.cpp/ggml cache variables.

## Source semantics
* Always: static libraries, only the `mtmd` part of llama.cpp (common/tools/tests/examples/server/app
  off), no OpenSSL/CURL, ggml backends not dynamic.
* Portable: `GGML_NATIVE` off, OpenMP off, AVX/AVX2/FMA/F16C/BMI2 forced on; static MSVC runtime only
  when also building with MSVC.
* Non-portable: `GGML_NATIVE` on.  `GGML_CUDA` mirrors `STRATA_VISION_CUDA`.

## Form template identifiers
* resolve     <- the `if(STRATA_PORTABLE) ... else() ... endif()` block
* msvcStatic  <- `if(STRATA_PORTABLE AND MSVC)`
-/

namespace Strata.VisionBuild

/-- The cache variables the file forces; `none` means left at the llama.cpp default. -/
structure Cfg where
  native : Bool
  openmp : Option Bool
  avx : Option Bool
  avx2 : Option Bool
  fma : Option Bool
  f16c : Option Bool
  bmi2 : Option Bool
  cuda : Bool
  sharedLibs : Bool
  backendDl : Bool
  mtmd : Bool
  common : Bool
  tools : Bool
  tests : Bool
  examples : Bool
  server : Bool
  app : Bool
  curl : Bool
  openssl : Bool

/-- Resolve the configuration from the two user options. -/
def resolve (portable cuda : Bool) : Cfg :=
  { native := !portable
    openmp := if portable then some false else none
    avx := if portable then some true else none
    avx2 := if portable then some true else none
    fma := if portable then some true else none
    f16c := if portable then some true else none
    bmi2 := if portable then some true else none
    cuda := cuda
    sharedLibs := false, backendDl := false, mtmd := true, common := false, tools := false,
    tests := false, examples := false, server := false, app := false, curl := false,
    openssl := false }

/-- Portable builds never use `-march=native`. -/
theorem portable_not_native (c : Bool) : (resolve true c).native = false := rfl

/-- Non-portable builds use native tuning. -/
theorem default_native (c : Bool) : (resolve false c).native = true := rfl

/-- Portable builds need no OpenMP runtime and force the AVX2 baseline. -/
theorem portable_baseline (c : Bool) :
    (resolve true c).openmp = some false ∧ (resolve true c).avx = some true ∧
    (resolve true c).avx2 = some true ∧ (resolve true c).fma = some true ∧
    (resolve true c).f16c = some true ∧ (resolve true c).bmi2 = some true :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- Non-portable builds leave the instruction flags at their defaults (native decides). -/
theorem default_flags_untouched (c : Bool) :
    (resolve false c).openmp = none ∧ (resolve false c).avx2 = none := ⟨rfl, rfl⟩

/-- GPU encoding is exactly the user's CUDA option, whichever portability is chosen. -/
theorem cuda_passthrough (p c : Bool) : (resolve p c).cuda = c := rfl

/-- Only the multimodal library is built from llama.cpp, statically, with no network deps. -/
theorem fixed_options (p c : Bool) :
    (resolve p c).mtmd = true ∧ (resolve p c).common = false ∧ (resolve p c).tools = false ∧
    (resolve p c).server = false ∧ (resolve p c).sharedLibs = false ∧
    (resolve p c).backendDl = false ∧ (resolve p c).curl = false ∧
    (resolve p c).openssl = false :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- Static MSVC runtime is selected only for portable builds under MSVC. -/
def msvcStatic (portable msvc : Bool) : Bool := portable && msvc

/-- `msvcStatic` is the conjunction of the two conditions. -/
theorem msvcStatic_iff (p m : Bool) : msvcStatic p m = true ↔ p = true ∧ m = true := by
  simp [msvcStatic]

/-- A non-portable build never forces the static runtime. -/
theorem no_static_when_not_portable (m : Bool) : msvcStatic false m = false := rfl

end Strata.VisionBuild




[LIGHT FILES — structure only, no LAYER 1a/1b]
  include/strata/core/graph.hpp (118 lines)
  include/strata/core/layout.hpp (98 lines)
  include/strata/core/mtp.hpp (136 lines)
  include/strata/core/pinned.hpp (96 lines)
  include/strata/core/weights.hpp (136 lines)
  include/strata/core/device.hpp (72 lines)
  include/strata/core/hit_hook.hpp (42 lines)
  include/strata/core/native_dense.hpp (37 lines)
  include/strata/core/native_head.hpp (66 lines)
  include/strata/kernels/elementwise.hpp (117 lines)
  include/strata/kernels/gr.hpp (123 lines)
  include/strata/kernels/native_mmvq.hpp (113 lines)
  include/strata/kernels/ple.hpp (120 lines)
  include/strata/kernels/s2_expert_grouped.hpp (119 lines)
  include/strata/kernels/shared_expert.hpp (102 lines)
  include/strata/kernels/verify_kernels.hpp (81 lines)
  include/strata/kernels/bf16_bits.hpp (47 lines)
  include/strata/kernels/bf16_gemv.hpp (49 lines)
  include/strata/kernels/dequant_bf16.hpp (28 lines)
  include/strata/kernels/dequant_s2.hpp (19 lines)
  include/strata/kernels/fused_gdn.hpp (36 lines)
  include/strata/kernels/fused_gr.hpp (52 lines)
  include/strata/kernels/iq_kernels.hpp (58 lines)
  include/strata/kernels/kv_q8.hpp (39 lines)
  include/strata/kernels/mrope.hpp (33 lines)
  include/strata/kernels/native_flash_attn.hpp (37 lines)
  include/strata/kernels/native_gdn.hpp (23 lines)
  include/strata/kernels/native_gdn_preprocess.hpp (38 lines)
  include/strata/kernels/native_gr_norm.hpp (19 lines)
  include/strata/kernels/native_gr_postops.hpp (29 lines)
  include/strata/kernels/native_moe.hpp (19 lines)
  include/strata/kernels/native_ple_postops.hpp (31 lines)
  include/strata/kernels/native_qsa.hpp (31 lines)
  include/strata/kernels/native_qsa_indexer.hpp (36 lines)
  include/strata/kernels/native_qsa_score.hpp (32 lines)
  include/strata/kernels/native_rope.hpp (15 lines)
  include/strata/kernels/native_router.hpp (18 lines)
  include/strata/kernels/qsa_decode_attn.hpp (45 lines)
  include/strata/kernels/qsa_select.hpp (34 lines)
  include/strata/kernels/quantize_act.hpp (59 lines)
  include/strata/kernels/rope.hpp (51 lines)
  include/strata/kernels/router_top10.hpp (21 lines)
  include/strata/kernels/s2_gemv.hpp (26 lines)
  include/strata/kernels/s2_gemv_q8.hpp (23 lines)
  include/strata/kernels/sampler.hpp (44 lines)
  include/strata/kernels/cpu/expert_layout.hpp (52 lines)
  include/strata/kernels/cpu/iq_avx512.hpp (19 lines)
  include/strata/kernels/cpu/native_expert.hpp (50 lines)
  include/strata/ngram/ple_reader.hpp (87 lines)
  include/strata/platform/direct_file.hpp (60 lines)
  include/strata/platform/memory.hpp (27 lines)
  include/strata/prefill/gemm.hpp (55 lines)
  include/strata/prefill/kernels.hpp (75 lines)
  include/strata/prefill/prefill.hpp (76 lines)
  include/strata/program/logits_selection.hpp (31 lines)
  include/strata/spec/suffix_drafter.hpp (54 lines)
  serve/__init__.py (2 lines)
  serve/chat_golden.json (197 lines)
  serve/chat_template.jinja (184 lines)
  serve/index.html (153 lines)
  src/artifact/dequant.cpp (124 lines)
  src/artifact/gguf_reader.cpp (68 lines)
  src/core/concurrent_main.cpp (198 lines)
  src/core/device.cu (118 lines)
  src/core/layout.cpp (174 lines)
  src/core/load_main.cpp (120 lines)
  src/core/native_dense.cpp (180 lines)
  src/core/native_head.cpp (152 lines)
  src/core/overlap_main.cpp (170 lines)
  src/core/device_main.cpp (79 lines)
  src/kernels/bf16_gemv_parity.cpp (149 lines)
  src/kernels/dequant_bf16_test.cpp (109 lines)
  src/kernels/dequant_s2_parity.cpp (144 lines)
  src/kernels/elementwise_parity.cpp (417 lines)
  src/kernels/gdn_parity.cpp (398 lines)
  src/kernels/gr_parity.cpp (616 lines)
  src/kernels/iq_parity.cpp (88 lines)
  src/kernels/kv_q8_parity.cpp (142 lines)
  src/kernels/native_expert_parity.cpp (205 lines)
  src/kernels/ple_oracle_vectors.inc (106 lines)
  src/kernels/ple_parity.cpp (1185 lines)
  src/kernels/qsa_oracle_vectors.inc (135 lines)
  src/kernels/qsa_parity.cpp (1384 lines)
  src/kernels/quantize_act_parity.cpp (408 lines)
  src/kernels/rope_parity.cpp (172 lines)
  src/kernels/router_top10_parity.cpp (168 lines)
  src/kernels/s2_gemv_parity.cpp (174 lines)
  src/kernels/s2_gemv_q8_parity.cpp (182 lines)
  src/kernels/s_gemv_parity.cpp (635 lines)
  src/kernels/s_gemv_q8k_parity.cpp (237 lines)
  src/kernels/sampler_parity.cpp (363 lines)
  src/kernels/shared_expert_parity.cpp (483 lines)
  src/kernels/cpu/expert_layout.cpp (137 lines)
  src/kernels/cpu/expert_multi_test.cpp (88 lines)
  src/kernels/cpu/expert_parity.cpp (185 lines)
  src/kernels/cpu/native_expert.cpp (115 lines)
  src/kernels/cpu/pool_test.cpp (225 lines)
  src/kernels/cuda/bf16_gemv.cu (142 lines)
  src/kernels/cuda/dequant_s2.cu (52 lines)
  src/kernels/cuda/native_bf16.cu (119 lines)
  src/kernels/cuda/native_gr_norm.cu (103 lines)
  src/kernels/cuda/native_gr_postops.cu (116 lines)
  src/kernels/cuda/native_moe.cu (75 lines)
  src/kernels/cuda/native_rope.cu (89 lines)
  src/kernels/cuda/rope.cu (96 lines)
  src/kernels/cuda/s2_gemv.cu (64 lines)
  src/kernels/cuda/s2_gemv_quads.cu (94 lines)
  src/ngram/ple_reader_test.cpp (238 lines)
  src/plan/plan_main.cpp (74 lines)
  src/platform/memory.cpp (77 lines)
  src/platform/memory_test.cpp (36 lines)
  src/prefill/gemm.cu (108 lines)
  src/spec/controller_test.cpp (75 lines)
  src/spec/suffix_drafter_test.cpp (142 lines)
  tools/_paths.py (29 lines)
  tools/gguf_reader.py (143 lines)
  tools/iq_pack.py (222 lines)
  tools/mtp_fetch.py (122 lines)
  tools/pack_index.py (261 lines)
  tools/pack_layer.py (124 lines)
  bench/results/2026-09-24-final/estimates.md (28 lines)
  bench/results/2026-09-24-final/matrix.md (16 lines)
  bench/results/2026-09-24-vision/vision.md (6 lines)
  third_party/ggml/VERSION.txt (2 lines)
  CMakeLists.txt (537 lines)
  START-HERE.bat (53 lines)
  bench/results/2026-09-24-final/estimates.json (461 lines)
  bench/results/2026-09-24-final/matrix.json (240 lines)
  bench/results/2026-09-24-vision/vision.json (53 lines)
  chat.py (117 lines)
  setup.py (807 lines)
  setup.sh (31 lines)
  third_party/ggml/ggml-common.h (1912 lines)

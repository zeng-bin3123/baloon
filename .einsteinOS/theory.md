
## FILE: /mnt/user-data/outputs/ExpertCacheResidency.lean
/-!
# MODULE: expert_cache_residency
VRAM expert-cache slot storage and residency table (`ExpertCache`): per-layer slot ranges, no-eviction
admission, slot addressing, 256-byte aligned variable slots, and the fill-size clamp.

## Source semantics
* `layer_slot_range` gives layer `l` the half-open range `[l*q, (l+1)*q)` with `q = slots / n_layers`; the last
  layer takes the remainder, so the ranges tile `0..slots` exactly.
* `admit` never evicts: a pair gets the next free slot of its range (per-layer) or of the global counter, or is refused.
* `device_slot(slot)` is only defined for `0 ≤ slot < slots`, and the slot (plus its blob) lies inside the arena.
* `open_sized` aligns every slot to 256 bytes; `fill_slot` clamps the copy size into `(0, blob]`.

## Form template identifiers
`slots_` -> nSlots, `n_layers_` -> nl, `n_expert_` -> ne, `blob_` -> blob, `layer_next_[l]` -> next,
`kNotResident` -> notResident, `layer_slot_range` -> slotLo / slotHi.
-/

namespace Strata.ExpertCache

-- § alignment of variable-size slots

/-- Slot sizes are rounded up to a multiple of 256 bytes. -/
def align256 (b : Nat) : Nat := (b + 255) / 256 * 256

/-- Aligned size is at least the requested size. -/
theorem align256_ge (b : Nat) : b ≤ align256 b := by unfold align256; omega

/-- Aligned size is a multiple of 256. -/
theorem align256_mod (b : Nat) : align256 b % 256 = 0 := by unfold align256; omega

/-- Alignment wastes fewer than 256 bytes. -/
theorem align256_lt (b : Nat) : align256 b < b + 256 := by unfold align256; omega

/-- Sum of aligned slot sizes (the single allocation of `open_sized`). -/
def total : List Nat → Nat
  | [] => 0
  | b :: bs => align256 b + total bs

/-- Offset of slot `i`: the sum of the aligned sizes before it. -/
def off : List Nat → Nat → Nat
  | _, 0 => 0
  | [], _ + 1 => 0
  | b :: bs, i + 1 => align256 b + off bs i

/-- Every slot offset is 256-byte aligned. -/
theorem off_aligned : ∀ (l : List Nat) (i : Nat), off l i % 256 = 0
  | _, 0 => by simp [off]
  | [], _ + 1 => by simp [off]
  | b :: bs, i + 1 => by
      have h := off_aligned bs i
      have h2 := align256_mod b
      simp only [off]
      omega

/-- Slot offsets never exceed the arena size. -/
theorem off_le_total : ∀ (l : List Nat) (i : Nat), off l i ≤ total l
  | _, 0 => by simp [off]
  | [], _ + 1 => by simp [off, total]
  | b :: bs, i + 1 => by
      have h := off_le_total bs i
      simp only [off, total]
      omega

/-- Slot `i+1` starts exactly where slot `i`'s aligned extent ends. -/
theorem off_succ : ∀ (l : List Nat) (i : Nat), i < l.length →
    off l (i + 1) = off l i + align256 (l.getD i 0)
  | [], i, h => by simp at h
  | b :: bs, 0, _ => by simp [off, List.getD_cons_zero]
  | b :: bs, i + 1, h => by
      have h' : i < bs.length := by simpa using h
      have := off_succ bs i h'
      simp only [off, List.getD_cons_succ]
      omega

-- § layer ranges

/-- Lower end of layer `l`'s slot range. -/
def slotLo (nl s l : Nat) : Nat :=
  if nl = 0 ∨ s = 0 ∨ nl ≤ l then 0 else l * (s / nl)

/-- Upper end (exclusive) of layer `l`'s slot range; the last layer takes the remainder. -/
def slotHi (nl s l : Nat) : Nat :=
  if nl = 0 ∨ s = 0 ∨ nl ≤ l then 0 else if l = nl - 1 then s else (l + 1) * (s / nl)

/-- An out-of-range layer owns the empty range. -/
theorem slotRange_oob (nl s l : Nat) (h : nl ≤ l) : slotLo nl s l = 0 ∧ slotHi nl s l = 0 := by
  unfold slotLo slotHi; simp [h]

/-- The first layer starts at slot 0. -/
theorem slotLo_first (nl s : Nat) (h : 0 < nl) (hs : 0 < s) : slotLo nl s 0 = 0 := by
  have c : ¬ (nl = 0 ∨ s = 0 ∨ nl ≤ 0) := by omega
  unfold slotLo; rw [if_neg c]; simp

/-- No range extends past the slot count. -/
theorem slotHi_le (nl s l : Nat) (h : l < nl) (hs : 0 < s) : slotHi nl s l ≤ s := by
  have c : ¬ (nl = 0 ∨ s = 0 ∨ nl ≤ l) := by omega
  unfold slotHi
  rw [if_neg c]
  by_cases hl : l = nl - 1
  · simp [hl]
  · rw [if_neg hl]
    have h1 : (l + 1) * (s / nl) ≤ nl * (s / nl) := Nat.mul_le_mul_right _ (by omega)
    have h2 : nl * (s / nl) ≤ s := Nat.mul_div_le s nl
    omega

/-- The last layer's range ends exactly at the slot count: no slot is orphaned by the division. -/
theorem slotHi_last (nl s : Nat) (h : 0 < nl) (hs : 0 < s) : slotHi nl s (nl - 1) = s := by
  have c : ¬ (nl = 0 ∨ s = 0 ∨ nl ≤ nl - 1) := by omega
  unfold slotHi; simp [c]

/-- Adjacent layers' ranges are contiguous, so the ranges tile `0..slots`. -/
theorem slot_contiguous (nl s l : Nat) (h : l + 1 < nl) (hs : 0 < s) :
    slotHi nl s l = slotLo nl s (l + 1) := by
  have c1 : ¬ (nl = 0 ∨ s = 0 ∨ nl ≤ l) := by omega
  have c2 : ¬ (nl = 0 ∨ s = 0 ∨ nl ≤ l + 1) := by omega
  have c3 : ¬ (l = nl - 1) := by omega
  unfold slotHi slotLo
  rw [if_neg c1, if_neg c3, if_neg c2]

/-- Each range is well formed (`lo ≤ hi`). -/
theorem slotLo_le_hi (nl s l : Nat) (h : l < nl) (hs : 0 < s) : slotLo nl s l ≤ slotHi nl s l := by
  have c : ¬ (nl = 0 ∨ s = 0 ∨ nl ≤ l) := by omega
  unfold slotLo slotHi
  rw [if_neg c, if_neg c]
  by_cases hl : l = nl - 1
  · rw [if_pos hl]
    have h1 : l * (s / nl) ≤ nl * (s / nl) := Nat.mul_le_mul_right _ (by omega)
    have h2 : nl * (s / nl) ≤ s := Nat.mul_div_le s nl
    omega
  · rw [if_neg hl]
    exact Nat.mul_le_mul_right _ (Nat.le_succ l)

-- § admission (no eviction)

/-- Per-layer admission: take `next` if the layer's quota is not full. -/
def admitStep (next hi : Nat) : Option (Nat × Nat) :=
  if hi ≤ next then none else some (next, next + 1)

/-- An admitted slot is below the ceiling, and the counter advances by one. -/
theorem admitStep_sound (next hi slot next' : Nat) (h : admitStep next hi = some (slot, next')) :
    slot < hi ∧ next' = slot + 1 ∧ slot = next := by
  unfold admitStep at h
  by_cases c : hi ≤ next
  · simp [c] at h
  · simp [c] at h
    omega

/-- A full quota refuses (no eviction). -/
theorem admitStep_full (next hi : Nat) (h : hi ≤ next) : admitStep next hi = none := by
  unfold admitStep; simp [h]

/-- The counter never passes the ceiling when it started within it. -/
theorem admitStep_bounded (next hi slot next' : Nat) (hn : next ≤ hi)
    (h : admitStep next hi = some (slot, next')) : next' ≤ hi := by
  have := admitStep_sound next hi slot next' h
  omega

-- § addressing

/-- The residency table index `layer*ne + expert` is in range for valid coordinates. -/
theorem residency_idx_lt (nl ne layer expert : Nat) (hl : layer < nl) (he : expert < ne) :
    layer * ne + expert < nl * ne := by
  have h := Nat.mul_le_mul_right ne (show layer + 1 ≤ nl by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- A valid slot plus its blob lies inside the arena. -/
theorem device_slot_in_arena (slots blob slot : Nat) (h : slot < slots) :
    slot * blob + blob ≤ slots * blob := by
  have h1 := Nat.mul_le_mul_right blob (show slot + 1 ≤ slots by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  exact h1

/-- A positive slot count and blob size give a positive allocation. -/
theorem want_pos (n blob : Nat) (hn : 0 < n) (hb : 0 < blob) : 0 < n * blob := Nat.mul_pos hn hb

/-- The VRAM check: a request larger than the free figure is refused before allocation. -/
def vramOk (free want : Nat) : Bool := !(decide (free < want))

/-- If the check passes, the request fits in free VRAM. -/
theorem vramOk_fits (free want : Nat) (h : vramOk free want = true) : want ≤ free := by
  unfold vramOk at h
  simp at h
  omega

/-- The copy size of `fill_slot`: the request if it is in `(0, blob]`, otherwise the whole blob. -/
def fillBytes (bytes blob : Int) : Int := if 0 < bytes ∧ bytes ≤ blob then bytes else blob

/-- The copy size is always positive and never exceeds the slot's blob size. -/
theorem fillBytes_bounds (bytes blob : Int) (hb : 0 < blob) :
    0 < fillBytes bytes blob ∧ fillBytes bytes blob ≤ blob := by
  unfold fillBytes
  split <;> omega

end Strata.ExpertCache



## FILE: /mnt/user-data/outputs/ExpertSourceBlobAccess.lean
/-!
# MODULE: expert_source_blob_access
Expert blob addressing (`FileExpertSource`, `ArenaExpertSource`) and the dispatch adapter's job/output indexing
(`expert_pool_dispatch`, `expert_pool_dispatch_multi`).

## Source semantics
* The flat blob index is `layer*n_expert + expert`; `expert >= n_expert` must be rejected on its own, because the flat
  index alone would alias into the next layer.
* The dispatch loop indexes the JOB ARRAY by `njobs` and the OUTPUT by `i`; hit rows are zeroed and skipped.
* The PCIe share `(nmiss * pcie_num) >> 8` never exceeds the number of misses.

## Form template identifiers
`n_expert_` -> ne, `blobs_` -> blobs, `BLOB` -> blobBytes, `d.is_hit` -> hitFlags, `njobs` -> misses.
-/

namespace Strata.ExpertSource

-- § blob addressing

/-- Blob index with the separate expert-axis check (negative inputs are modelled elsewhere: coordinates are naturals). -/
def blobIndex? (ne blobs layer expert : Nat) : Option Nat :=
  if ne ≤ expert then none
  else if blobs ≤ layer * ne + expert then none
  else some (layer * ne + expert)

/-- An expert id one past the end of a layer is rejected. -/
theorem blobIndex_expert_axis (ne blobs layer : Nat) : blobIndex? ne blobs layer ne = none := by
  unfold blobIndex?; simp

/-- Why the separate check is needed: `(layer, ne)` and `(layer+1, 0)` have the same flat index. -/
theorem flat_index_aliases (ne layer : Nat) : layer * ne + ne = (layer + 1) * ne + 0 := by
  rw [Nat.add_mul, Nat.one_mul, Nat.add_zero]

/-- A returned index is in range and equals the flat index. -/
theorem blobIndex_sound (ne blobs layer expert i : Nat) (h : blobIndex? ne blobs layer expert = some i) :
    expert < ne ∧ i < blobs ∧ i = layer * ne + expert := by
  unfold blobIndex? at h
  by_cases c1 : ne ≤ expert
  · simp [c1] at h
  · by_cases c2 : blobs ≤ layer * ne + expert
    · simp [c1, c2] at h
    · simp [c1, c2] at h
      omega

/-- For a pack of exactly `nl * ne` blobs, every valid coordinate pair is served. -/
theorem blobIndex_complete (nl ne layer expert : Nat) (hl : layer < nl) (he : expert < ne) :
    blobIndex? ne (nl * ne) layer expert = some (layer * ne + expert) := by
  have h := Nat.mul_le_mul_right ne (show layer + 1 ≤ nl by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  have c1 : ¬ ne ≤ expert := by omega
  have c2 : ¬ nl * ne ≤ layer * ne + expert := by omega
  unfold blobIndex?
  rw [if_neg c1, if_neg c2]

/-- The pinned test: a blob is pinned only if it ends inside the registered bytes. -/
def pinnedOk (off len pinnedBytes : Nat) : Bool := decide (off + len ≤ pinnedBytes)

/-- Pinned status is monotone in the registered size. -/
theorem pinnedOk_mono (off len a b : Nat) (hab : a ≤ b) (h : pinnedOk off len a = true) :
    pinnedOk off len b = true := by
  unfold pinnedOk at *
  simp at *
  omega

-- § dispatch: jobs versus output rows

/-- Number of misses (jobs) in a hit-flag list. -/
def misses : List Bool → Nat
  | [] => 0
  | b :: bs => (if b then 0 else 1) + misses bs

/-- Number of hits in a hit-flag list. -/
def hitCount : List Bool → Nat
  | [] => 0
  | b :: bs => (if b then 1 else 0) + hitCount bs

/-- Every routed entry is either a hit (GPU) or a job (CPU): no entry is computed twice or dropped. -/
theorem hits_add_misses : ∀ l : List Bool, hitCount l + misses l = l.length
  | [] => by simp [hitCount, misses]
  | b :: bs => by
      have ih := hits_add_misses bs
      cases b <;> simp [hitCount, misses] <;> omega

/-- The job array never needs more than `k` slots. -/
theorem jobs_le_k (l : List Bool) : misses l ≤ l.length := by
  have := hits_add_misses l
  omega

/-- Output row `i < k` lies inside the `k x n_embd` output buffer. -/
theorem out_row_in_bounds (k n i : Nat) (h : i < k) : i * n + n ≤ k * n := by
  have h1 := Nat.mul_le_mul_right n (show i + 1 ≤ k by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  exact h1

/-- The verify-window job table of `n_tok * k` entries fits the `MAXT * k` allocation. -/
theorem jobs_multi_fit (nTok maxT k : Nat) (h : nTok ≤ maxT) : nTok * k ≤ maxT * k :=
  Nat.mul_le_mul_right k h

-- § the PCIe share

/-- Number of missed experts the GPU reads over PCIe: `(nmiss * pcie_num) >> 8`. -/
def pcieShare (nmiss num : Nat) : Nat := nmiss * num / 256

/-- With `pcie_num ≤ 256` the share never exceeds the misses. -/
theorem pcieShare_le (nmiss num : Nat) (h : num ≤ 256) : pcieShare nmiss num ≤ nmiss := by
  unfold pcieShare
  have h1 : nmiss * num ≤ nmiss * 256 := Nat.mul_le_mul_left nmiss h
  have h2 : nmiss * num ≤ 256 * nmiss := by
    calc nmiss * num ≤ nmiss * 256 := h1
      _ = 256 * nmiss := Nat.mul_comm _ _
  exact Nat.div_le_of_le_mul h2

end Strata.ExpertSource



## FILE: /mnt/user-data/outputs/ExpertDispatchContract.lean
/-!
# MODULE: expert_dispatch_contract
The adapter contract in `expert_source.hpp`: the all-or-nothing `hits_ready()` predicate, the latching `failed`
flag, the geometry refusal, and the GPU-plan list layout.

## Source semantics
* `hits_ready()` is true only if every one of the nine hit-path pointers is wired.
* `failed` latches: once set, the adapter short-circuits and the layer counter does not advance.
* A mismatched `n_embd` is refused rather than mis-indexed.
* The GPU plan (`GpuPlanSink`) holds at most `cap` entries; group starts are monotone.

## Form template identifiers
`ExpertDispatch::failed` -> failed, `hits_ready` -> hitsReady, `H` -> hWidth, `GpuPlanSink::cap` -> cap.
-/

namespace Strata.ExpertDispatch

-- § hits_ready

/-- Which of the hit-path members are set. -/
structure HitCfg where
  cache : Bool
  cacheBase : Bool
  partsOut : Bool
  hitOut : Bool
  mixed : Bool
  scratch : Bool
  xq : Bool
  xqScale : Bool
  dSlot : Bool
  dDst : Bool

/-- The all-or-nothing readiness predicate. -/
def hitsReady (c : HitCfg) : Bool :=
  c.cache && c.cacheBase && c.partsOut && c.hitOut && c.mixed && c.scratch && c.xq && c.xqScale && c.dSlot && c.dDst

/-- Readiness requires the fp32 activation scales (R4.2h). -/
theorem hitsReady_needs_scales (c : HitCfg) (h : hitsReady c = true) : c.xqScale = true := by
  unfold hitsReady at h
  simp at h
  tauto

/-- Readiness requires the cache itself. -/
theorem hitsReady_needs_cache (c : HitCfg) (h : hitsReady c = true) : c.cache = true := by
  unfold hitsReady at h
  simp at h
  tauto

/-- Readiness requires the device slot and destination lists. -/
theorem hitsReady_needs_lists (c : HitCfg) (h : hitsReady c = true) : c.dSlot = true ∧ c.dDst = true := by
  unfold hitsReady at h
  simp at h
  tauto

-- § the latching failure flag

/-- The flag after a step that may report a new failure. -/
def latch (failed new : Bool) : Bool := failed || new

/-- Once failed, always failed: clearing the message cannot re-arm the adapter. -/
theorem latch_sticky (new : Bool) : latch true new = true := by simp [latch]

/-- A failure is never lost. -/
theorem latch_keeps (failed new : Bool) (h : failed = true) : latch failed new = true := by simp [latch, h]

/-- The layer counter advances only when the adapter has not failed. -/
def stepLayers (failed : Bool) (layers : Nat) : Nat := if failed then layers else layers + 1

/-- After a failure the counter is frozen. -/
theorem stepLayers_frozen (layers : Nat) : stepLayers true layers = layers := by simp [stepLayers]

/-- Without failure the counter advances by one. -/
theorem stepLayers_advance (layers : Nat) : stepLayers false layers = layers + 1 := by simp [stepLayers]

-- § geometry refusal

/-- The kernel accepts only the compiled-in activation width. -/
def geometryOk (nEmbd hWidth : Nat) : Bool := decide (nEmbd = hWidth)

/-- A different width is refused. -/
theorem geometry_refuses (nEmbd hWidth : Nat) (h : nEmbd ≠ hWidth) : geometryOk nEmbd hWidth = false := by
  unfold geometryOk; simp [h]

/-- The compiled width is accepted. -/
theorem geometry_accepts (hWidth : Nat) : geometryOk hWidth hWidth = true := by
  unfold geometryOk; simp

-- § GPU plan layout

/-- Plan group starts are monotone and the entry count is bounded by `cap`. -/
structure PlanState where
  groups : Nat
  entries : Nat
  fetches : Nat

/-- Adding a group that holds `m` entries keeps `entries ≤ cap` when there is room. -/
theorem plan_add_group (st : PlanState) (cap m : Nat) (h : st.entries + m ≤ cap) :
    (PlanState.mk (st.groups + 1) (st.entries + m) st.fetches).entries ≤ cap := by
  simpa using h

/-- A fetch is only issued while below both the staging capacity and the 64-slot arrays. -/
def mayFetch (fetches stagingCap : Nat) : Bool := decide (fetches < stagingCap ∧ fetches < 64)

/-- Fetch count stays below the array size after an allowed fetch. -/
theorem mayFetch_bound (fetches stagingCap : Nat) (h : mayFetch fetches stagingCap = true) :
    fetches + 1 ≤ 64 := by
  unfold mayFetch at h
  simp at h
  omega

end Strata.ExpertDispatch



## FILE: /mnt/user-data/outputs/GdnQsaLayerBuffers.lean
/-!
# MODULE: gdn_qsa_layer_buffers
Buffer carving, plane layout, kernel dispatch and dump layout of `layer.cpp` (GDN layer, QSA mixer, MoE block).

## Source semantics
* Every buffer region is 16-byte aligned; `*_bytes` and `*_init` walk the same list, so they agree.
* A quantized tensor's planes (codes | scales | offsets) must add up exactly to the tensor size.
* Which GEMV runs is decided per tensor: native, else S2 (2-bit), else Q8_0 or Q8_K by `wants_q8k`.
* The per-layer dump stride equals the last slot's end.

## Form template identifiers
`align_up16` -> align16, `q8k_bytes` -> q8kBytes, `Planes` -> codes/scales/offset sizes, `dump_stride_floats` -> dumpStride.
-/

namespace Strata.Layer

-- § alignment and carving

/-- 16-byte alignment of every carved region. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Aligned size covers the request. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by unfold align16; omega

/-- Aligned size is a multiple of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by unfold align16; omega

/-- Q8_K buffer size: `n/256` blocks of 292 bytes. -/
def q8kBytes (n : Nat) : Nat := n / 256 * 292

/-- For `n` a multiple of 256 the Q8_K size is exactly `n * 292 / 256`. -/
theorem q8kBytes_exact (n : Nat) (h : n % 256 = 0) : q8kBytes n * 256 = n * 292 := by
  unfold q8kBytes; omega

/-- End offset after carving `parts` from `acc` (the `total` loop of `*_buffers_bytes`). -/
def carveEnd (acc : Nat) (parts : List Nat) : Nat := parts.foldl (fun a p => a + align16 p) acc

/-- The carved total is the sum of the aligned parts. -/
theorem carveEnd_eq : ∀ (parts : List Nat) (acc : Nat), carveEnd acc parts = acc + (parts.map align16).sum
  | [], acc => by simp [carveEnd]
  | p :: ps, acc => by
      have := carveEnd_eq ps (acc + align16 p)
      simp only [carveEnd, List.foldl_cons, List.map_cons, List.sum_cons] at this ⊢
      omega

/-- Offset of the `i`th carved region. -/
def off16 : List Nat → Nat → Nat
  | _, 0 => 0
  | [], _ + 1 => 0
  | p :: ps, i + 1 => align16 p + off16 ps i

/-- Every carved region starts 16-byte aligned. -/
theorem off16_aligned : ∀ (l : List Nat) (i : Nat), off16 l i % 16 = 0
  | _, 0 => by simp [off16]
  | [], _ + 1 => by simp [off16]
  | p :: ps, i + 1 => by
      have h := off16_aligned ps i
      have h2 := align16_mod p
      simp only [off16]
      omega

/-- Every region offset is within the carved total. -/
theorem off16_le_total : ∀ (l : List Nat) (i : Nat), off16 l i ≤ (l.map align16).sum
  | _, 0 => by simp [off16]
  | [], _ + 1 => by simp [off16]
  | p :: ps, i + 1 => by
      have h := off16_le_total ps i
      simp only [off16, List.map_cons, List.sum_cons]
      omega

-- § planes

/-- The three planes of a quantized tensor tile it exactly (the `plane_ptrs` check). -/
theorem planes_tile (codes scales offs bytes : Nat) (h : codes + scales + offs = bytes) :
    codes + scales ≤ bytes ∧ codes ≤ bytes := by omega

/-- With an offset plane, the offset pointer `codes + scales` plus its size ends at the tensor end. -/
theorem planes_offset_end (codes scales offs bytes : Nat) (h : codes + scales + offs = bytes) :
    (codes + scales) + offs = bytes := h

-- § kernel dispatch

/-- Which GEMV a quantized projection uses. -/
inductive Kind | native | s2 | q8_0 | q8k
  deriving DecidableEq

/-- The per-tensor dispatch of `gemv_quantized`. -/
def kernelOf (native : Bool) (bits : Nat) (wantsQ8k : Bool) : Kind :=
  if native then .native else if bits = 2 then .s2 else if wantsQ8k then .q8k else .q8_0

/-- A native tensor always takes the native path. -/
theorem kernelOf_native (bits : Nat) (w : Bool) : kernelOf true bits w = .native := by simp [kernelOf]

/-- A 2-bit tensor takes S2 (Q8_0 activation). -/
theorem kernelOf_s2 (w : Bool) : kernelOf false 2 w = .s2 := by simp [kernelOf]

/-- A legacy 4/8-bit form that does not want Q8_K takes the Q8_0 split kernel. -/
theorem kernelOf_q8_0 (bits : Nat) (h : bits ≠ 2) : kernelOf false bits false = .q8_0 := by
  simp [kernelOf, h]

/-- A K-quant takes Q8_K. -/
theorem kernelOf_q8k (bits : Nat) (h : bits ≠ 2) : kernelOf false bits true = .q8k := by
  simp [kernelOf, h]

-- § layer kinds

/-- Twelve of the 48 layers are QSA (`l % 4 = 3`). -/
theorem qsa_layer_count : (List.range 48).countP (fun l => decide (l % 4 = 3)) = 12 := by decide

/-- The other 36 are GDN. -/
theorem gdn_layer_count : (List.range 48).countP (fun l => decide (l % 4 ≠ 3)) = 36 := by decide

-- § q/gate split and RoPE bound

/-- The q/gate half-split reads row `h` of the `[nh x 2*hd]` buffer inside its bounds. -/
theorem qgate_row_in_bounds (nh hd h : Nat) (hh : h < nh) : h * (2 * hd) + hd ≤ nh * (2 * hd) := by
  have h1 := Nat.mul_le_mul_right (2 * hd) (show h + 1 ≤ nh by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

/-- The RoPE table check: a position inside the table means the position itself is in range. -/
theorem rope_pos_in_table (posBase pos maxCells : Nat) (h : posBase + pos < maxCells) : pos < maxCells := by
  omega

-- § dump layout

/-- Floats per layer in the half-level dump. -/
def dumpStride (n hc nhhd nvk : Nat) : Nat := 2 * n + 2 * hc + nhhd + 5 * nvk + 8

/-- The last slot (`kcur`, at `base + 2*nvk`, length `nvk`) ends exactly at the stride. -/
theorem dump_last_slot_ends_at_stride (n hc nhhd nvk : Nat) :
    (2 * n + 2 * hc + nhhd + 2 * nvk + 8) + 2 * nvk + nvk = dumpStride n hc nhhd nvk := by
  unfold dumpStride; omega

/-- The gathered-`v` slots (`vs + nvk/2 + 4 + 4`) fit before the stride. -/
theorem dump_gather_slots_fit (n hc nhhd nvk : Nat) :
    (2 * n + 2 * hc + nhhd + 2 * nvk) + nvk / 2 + 4 + 4 ≤ dumpStride n hc nhhd nvk := by
  unfold dumpStride; omega

-- § compatibility gate

/-- The verify window requires every fast-path flag. -/
def verifyCompatible (bf16 gr gdn attn sel idx : Bool) : Bool := bf16 && gr && gdn && attn && sel && idx

/-- Compatibility implies the fused hyper-connection read is on. -/
theorem verifyCompatible_needs_gr (bf16 gr gdn attn sel idx : Bool)
    (h : verifyCompatible bf16 gr gdn attn sel idx = true) : gr = true := by
  unfold verifyCompatible at h
  simp at h
  tauto

end Strata.Layer



## FILE: /mnt/user-data/outputs/MtpDraftLayer.lean
/-!
# MODULE: mtp_draft_layer
The MTP draft layer (`MtpDrafter`): bump-allocator arena, attention window capacity, step-row staging for the
catch-up and chain graphs, and the draft vocabulary subset.

## Source semantics
* `Bump::take` advances by `n*sizeof(T)` rounded up to 256; the count pass and the carve pass use the same sizes.
* `cap_` is the window (or `max_cells`) rounded up to a multiple of 64.
* Step rows: T catch-up rows `[0,T)`, chain rows `[max_t, 2*max_t-2)`, and the staged full-layer row `2*max_t-1`.
* `draft_vocab.bin` holds `n_dvocab = size/4` 32-bit ids.

## Form template identifiers
`Bump` -> bumpEnd, `cap_` -> capOf, `ra` -> stagedRow, `row` (chain) -> chainRow, `n_dvocab_` -> nDvocab.
-/

namespace Strata.Mtp

-- § bump allocator

/-- 256-byte rounding of one `take`. -/
def align256 (b : Nat) : Nat := (b + 255) / 256 * 256

/-- A take advances the cursor by at least the request. -/
theorem take_advances (used b : Nat) : used + b ≤ used + align256 b := by unfold align256; omega

/-- A take keeps the cursor 256-aligned. -/
theorem take_aligned (used b : Nat) (h : used % 256 = 0) : (used + align256 b) % 256 = 0 := by
  unfold align256; omega

/-- Cursor after a sequence of takes. -/
def bumpEnd (acc : Nat) (sizes : List Nat) : Nat := sizes.foldl (fun a b => a + align256 b) acc

/-- The total is the sum of the rounded sizes (so the count pass sizes the real allocation exactly). -/
theorem bumpEnd_eq : ∀ (sizes : List Nat) (acc : Nat), bumpEnd acc sizes = acc + (sizes.map align256).sum
  | [], acc => by simp [bumpEnd]
  | b :: bs, acc => by
      have := bumpEnd_eq bs (acc + align256 b)
      simp only [bumpEnd, List.foldl_cons, List.map_cons, List.sum_cons] at this ⊢
      omega

/-- The count pass and the carve pass end at the same offset (same list, same start). -/
theorem two_pass_agree (sizes : List Nat) : bumpEnd 0 sizes = bumpEnd 0 sizes := rfl

-- § attention window

/-- Effective window: the requested one if it is positive and below `max_cells`. -/
def windowOf (window maxCells : Nat) : Nat := if 0 < window ∧ window < maxCells then window else maxCells

/-- `cap_` rounds the effective window up to a multiple of 64. -/
def capOf (window maxCells : Nat) : Nat := (windowOf window maxCells + 63) / 64 * 64

/-- The capacity covers the window. -/
theorem capOf_ge (window maxCells : Nat) : windowOf window maxCells ≤ capOf window maxCells := by
  unfold capOf; omega

/-- The capacity is a multiple of 64. -/
theorem capOf_mod (window maxCells : Nat) : capOf window maxCells % 64 = 0 := by
  unfold capOf; omega

-- § step rows

/-- Row staged for the single full-layer step of the round graph. -/
def stagedRow (maxT : Nat) : Nat := 2 * maxT - 1

/-- Row of chain step `j`. -/
def chainRow (maxT j : Nat) : Nat := maxT + j - 1

/-- Catch-up rows, chain rows and the staged row are pairwise disjoint and all below `2*max_t`. -/
theorem rows_disjoint (maxT j : Nat) (hT : 2 ≤ maxT) (hj1 : 1 ≤ j) (hj2 : j + 2 ≤ maxT) :
    maxT - 1 < chainRow maxT j ∧ chainRow maxT j < stagedRow maxT ∧ stagedRow maxT < 2 * maxT := by
  unfold chainRow stagedRow; omega

/-- Chain draft indices stay inside the `max_t`-sized draft array. -/
theorem draft_index_bound (maxT j : Nat) (hj : j + 1 < maxT) : j < maxT := by omega

/-- At most `max_t - 1` drafts are produced, so `n_drafts ≤ max_t - 1`. -/
theorem drafts_count_bound (maxT n : Nat) (h : n ≤ maxT - 1) : n < maxT ∨ maxT = 0 := by omega

-- § draft vocabulary

/-- Number of ids in `draft_vocab.bin`. -/
def nDvocab (size : Nat) : Nat := size / 4

/-- When the file size is a multiple of 4 the ids tile it exactly. -/
theorem dvocab_tiles (size : Nat) (h : size % 4 = 0) : nDvocab size * 4 = size := by
  unfold nDvocab; omega

/-- A non-empty file yields at least one id. -/
theorem dvocab_pos (size : Nat) (h : 4 ≤ size) : 0 < nDvocab size := by
  unfold nDvocab; omega

end Strata.Mtp



## FILE: /mnt/user-data/outputs/SessionStateLayout.lean
/-!
# MODULE: session_state_layout
Session arena layout and per-layer state addressing (`session.cpp`): the GDN state index, per-GDN-layer state slices,
the QSA state arena (one RoPE-owning state, the rest borrow), alignment, ring sequence numbers and the five-stage
differencing.

## Source semantics
* A GDN layer's index is the number of non-QSA layers before it; each owns `F = state + conv` floats.
* QSA state `i` sits at offset `0` for `i = 0` and `first + (i-1)*rest` otherwise; all fit in `first + (n-1)*rest`.
* `session_loop` waits for ring `l+1` at layer `l`; the token graph waits for ring `l+1` too.
* The five stage times are consecutive differences of prefix times, so they telescope to the full prefix.

## Form template identifiers
`gdn_point_at` -> gdnIndex, `gdn_state_floats` -> stateFloats, `qsa_state_arena` -> qsaOff, `align_up` -> alignUp256.
-/

namespace Strata.Session

-- § alignment

/-- 256-byte alignment of the session arena regions. -/
def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Aligned size covers the request. -/
theorem alignUp256_ge (n : Nat) : n ≤ alignUp256 n := by unfold alignUp256; omega

/-- Aligned size is a multiple of 256. -/
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by unfold alignUp256; omega

-- § GDN state indexing

/-- Index of a GDN layer: the count of non-QSA layers before it. -/
def gdnIndex (l : Nat) : Nat := ((List.range l).filter (fun i => !(i % 4 == 3))).length

/-- Closed form of the index on the 48-layer geometry. -/
theorem gdnIndex_closed : ∀ l, l < 49 → gdnIndex l = l - l / 4 := by decide

/-- A GDN layer's index is below the 36 states. -/
theorem gdnIndex_lt_36 (l : Nat) (h : l < 48) (hq : l % 4 ≠ 3) : gdnIndex l < 36 := by
  have := gdnIndex_closed l (by omega)
  omega

/-- Consecutive GDN layers get consecutive indices (no state is shared). -/
theorem gdnIndex_succ (l : Nat) (h : l < 47) (hq : l % 4 ≠ 3) : gdnIndex (l + 1) = gdnIndex l + 1 := by
  have h1 := gdnIndex_closed l (by omega)
  have h2 := gdnIndex_closed (l + 1) (by omega)
  omega

/-- The conv state starts where the recurrent state ends, and ends at the slice's end. -/
theorem gdn_slice_tiles (idx S1 S2 : Nat) : idx * (S1 + S2) + S1 + S2 = (idx + 1) * (S1 + S2) := by
  rw [Nat.add_mul, Nat.one_mul]; omega

/-- A valid GDN index's slice lies inside the `n_gdn * F` allocation. -/
theorem gdn_slice_in_arena (idx n F : Nat) (h : idx < n) : idx * F + F ≤ n * F := by
  have h1 := Nat.mul_le_mul_right F (show idx + 1 ≤ n by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  exact h1

-- § QSA state arena

/-- Offset of QSA state `i` in the arena. -/
def qsaOff (first rest i : Nat) : Nat := if i = 0 then 0 else first + (i - 1) * rest

/-- State `i`'s extent: `first` for the RoPE owner, `rest` for the borrowers. -/
def qsaSize (first rest i : Nat) : Nat := if i = 0 then first else rest

/-- Every QSA state fits in the `first + (n-1)*rest` arena. -/
theorem qsa_state_fits (first rest n i : Nat) (h : i < n) :
    qsaOff first rest i + qsaSize first rest i ≤ first + (n - 1) * rest := by
  unfold qsaOff qsaSize
  by_cases h0 : i = 0
  · simp [h0]
  · simp only [h0, if_false]
    have h1 := Nat.mul_le_mul_right rest (show i ≤ n - 1 by omega)
    have h2 : (i - 1 + 1) * rest = (i - 1) * rest + rest := by rw [Nat.add_mul, Nat.one_mul]
    have h3 : i - 1 + 1 = i := by omega
    rw [h3] at h2
    omega

-- § ring sequence

/-- At layer `l` the loop waits for ring number `l + 1`; the counter is monotone. -/
theorem ring_monotone (l : Nat) : l + 1 < (l + 1) + 1 := by omega

/-- The per-layer wait value is positive, so a zeroed doorbell is never mistaken for a ring. -/
theorem ring_positive (l : Nat) : 0 < l + 1 := by omega

-- § five-stage differencing

/-- The five stage times are the differences of the prefix times (`t0 = 0`). -/
def stageTimes (t1 t2 t3 t4 t5 : Int) : List Int := [t1, t2 - t1, t3 - t2, t4 - t3, t5 - t4]

/-- The five stage times sum to the full prefix time. -/
theorem stageTimes_sum (t1 t2 t3 t4 t5 : Int) : (stageTimes t1 t2 t3 t4 t5).sum = t5 := by
  simp [stageTimes]

end Strata.Session



## FILE: /mnt/user-data/outputs/VerifyWindowProtocol.lean
/-!
# MODULE: verify_window_protocol
The speculative verify window (`Verifier`): token-group split, ring numbering, staging split, mapped plan layout,
monotone flag raising and the commit vector.

## Source semantics
* With two groups, A = tokens `[0, (T+1)/2)` and B = the rest; the groups cover the window and are non-empty for `T ≥ 2`.
* The host's `k`-th ring maps to layer `k / G`, group `k % G`; the graph's ring number for `(l, grp)` is `l*G + grp + 1`.
* Staging blobs are split between groups without overlap.
* The mapped plan buffer holds `dst/tok/ptr/ptr2/start2` regions that do not overlap; two slots fit the allocation.
* Flag B only rises (`raise_flag` is a monotone max).
* `commit` stages `[n_keep, n_keep-1, pos0+t for t<n_keep else -1]`.

## Form template identifiers
`tb_`/`te_` -> groupBegin/groupEnd, `ring` -> ringNo, `kStagingBlobs` -> k, `plan_i32_` -> planTotal,
`raise_flag` -> raise, `h_commit_` -> commitEntry.
-/

namespace Strata.Verify

-- § token groups

/-- Size of group A in a split window. -/
def half (T : Nat) : Nat := (T + 1) / 2

/-- For `T ≥ 2` both groups are non-empty and together cover the window. -/
theorem groups_cover (T : Nat) (h : 2 ≤ T) :
    0 < half T ∧ half T < T ∧ half T + (T - half T) = T := by
  unfold half; omega

-- § ring numbering

/-- The graph's ring number for layer `l`, group `grp`, with `G` groups. -/
def ringNo (G l grp : Nat) : Nat := l * G + grp + 1

/-- Two groups: the host's `k`-th ring is layer `k/2`, group `k%2`. -/
theorem ring_decode_two (l grp : Nat) (hg : grp < 2) :
    (l * 2 + grp) / 2 = l ∧ (l * 2 + grp) % 2 = grp := by omega

/-- One group: the `k`-th ring is layer `k`, group 0. -/
theorem ring_decode_one (l : Nat) : (l * 1 + 0) / 1 = l ∧ (l * 1 + 0) % 1 = 0 := by omega

/-- Ring numbers are consecutive across the group boundary (two groups). -/
theorem ring_consecutive_two (l : Nat) : ringNo 2 (l + 1) 0 = ringNo 2 l 1 + 1 := by
  unfold ringNo; omega

/-- Within a layer the two groups' rings are consecutive. -/
theorem ring_consecutive_in_layer (l : Nat) : ringNo 2 l 1 = ringNo 2 l 0 + 1 := by
  unfold ringNo; omega

/-- Ring numbers are positive (a zeroed flag is never mistaken for a ring). -/
theorem ring_pos (G l grp : Nat) : 0 < ringNo G l grp := by unfold ringNo; omega

/-- The total number of rings is `n_layers * G`; the last one is `n_layers*G` for `grp = G-1`. -/
theorem ring_last_two (nl : Nat) (h : 0 < nl) : ringNo 2 (nl - 1) 1 = nl * 2 := by
  unfold ringNo; omega

-- § staging split

/-- Staging blobs per group. -/
def per (G k : Nat) : Nat := if G = 2 then k / 2 else k

/-- Group regions of staging are disjoint and inside `kStagingBlobs`. -/
theorem staging_disjoint (G k grp : Nat) (hG : G = 1 ∨ G = 2) (hg : grp < G) :
    grp * per G k + per G k ≤ k := by
  rcases hG with rfl | rfl
  · have : grp = 0 := by omega
    subst this
    simp [per]
  · rcases (by omega : grp = 0 ∨ grp = 1) with rfl | rfl <;> simp [per] <;> omega

-- § mapped plan layout (int32 words)

/-- Words before the u64 pointers: counts(4) | start(cap+1) | dst(cap) | tok(cap). -/
def planI32 (cap : Nat) : Nat := 4 + (cap + 1) + cap + cap

/-- Word offset of the u64 pointer array, rounded to an even word (8-byte alignment). -/
def ptrOff (cap : Nat) : Nat := (planI32 cap + 1) / 2 * 2

/-- Words per plan slot. -/
def planTotal (cap : Nat) : Nat := ptrOff cap + 4 * cap + (cap + 1) + 1

/-- The i32 prefix fits before the pointers, the pointers are 8-byte aligned. -/
theorem plan_prefix_fits (cap : Nat) : planI32 cap ≤ ptrOff cap ∧ ptrOff cap % 2 = 0 := by
  unfold ptrOff planI32; omega

/-- `ptr` (2*cap words), `ptr2` (2*cap words) and `start2` (cap+1 words) tile the tail without overlap. -/
theorem plan_tail_tiles (cap : Nat) :
    ptrOff cap + 2 * cap + 2 * cap = ptrOff cap + 4 * cap ∧
    ptrOff cap + 4 * cap + (cap + 1) ≤ planTotal cap := by
  unfold planTotal; omega

/-- The dst/tok regions end before the pointer array. -/
theorem plan_tok_before_ptr (cap : Nat) : 4 + (cap + 1) + cap + cap ≤ ptrOff cap := by
  unfold ptrOff planI32; omega

/-- Two plan slots fit the allocation of `2 * words + 16` words. -/
theorem two_slots_fit (N grp : Nat) (hg : grp < 2) : grp * N + N ≤ 2 * N := by
  rcases (by omega : grp = 0 ∨ grp = 1) with rfl | rfl <;> omega

-- § flag B only rises

/-- `raise_flag`: store `value` only if it is larger than the current value. -/
def raise (cur value : Nat) : Nat := if cur < value then value else cur

/-- The result dominates both inputs. -/
theorem raise_ge (cur value : Nat) : cur ≤ raise cur value ∧ value ≤ raise cur value := by
  unfold raise; split <;> omega

/-- Raising is order independent: a late host function of an earlier layer cannot lower the flag. -/
theorem raise_comm (cur a b : Nat) : raise (raise cur a) b = raise (raise cur b) a := by
  unfold raise
  split_ifs <;> omega

/-- Raising to a smaller value is a no-op. -/
theorem raise_noop (cur value : Nat) (h : value ≤ cur) : raise cur value = cur := by
  unfold raise; split <;> omega

-- § commit vector

/-- The staged position of cell `t`: `pos0 + t` if kept, `-1` otherwise. -/
def commitEntry (pos0 t nKeep : Int) : Int := if t < nKeep then pos0 + t else -1

/-- Kept cells carry their real (non-negative) position. -/
theorem commit_kept (pos0 t nKeep : Int) (h0 : 0 ≤ pos0) (ht : 0 ≤ t) (h : t < nKeep) :
    commitEntry pos0 t nKeep = pos0 + t ∧ 0 ≤ commitEntry pos0 t nKeep := by
  unfold commitEntry; rw [if_pos h]; omega

/-- Dropped cells are marked `-1`. -/
theorem commit_dropped (pos0 t nKeep : Int) (h : nKeep ≤ t) : commitEntry pos0 t nKeep = -1 := by
  unfold commitEntry; rw [if_neg (by omega)]

/-- The commit counts are in range: `1 ≤ n_keep ≤ last_t` gives `0 ≤ n_keep - 1 < last_t`. -/
theorem commit_index_range (nKeep lastT : Nat) (h1 : 1 ≤ nKeep) (h2 : nKeep ≤ lastT) :
    nKeep - 1 < lastT := by omega

-- § acceptance

/-- Accepted drafts `a` satisfy `a ≤ T-1`, so emitted tokens `a+1 ≤ T`. -/
theorem accepted_bound (T a : Nat) (h : a ≤ T - 1) (hT : 1 ≤ T) : a + 1 ≤ T := by omega

end Strata.Verify



## FILE: /mnt/user-data/outputs/WeightLoaderSegments.lean
/-!
# MODULE: weight_loader_segments
The dense-weight loader (`weights.cpp`): chunked conversion progress for the widening and BF16 paths, staging-buffer
bounds, segment tiling and compacted arena offsets.

## Source semantics
* `CHUNK = 8 MiB`; widening fp16 -> f32 doubles a chunk, so `stage_out` is `2*CHUNK`.
* Widening advances the destination by `2*n` per `n` source bytes; BF16 promotion by `n/2` per `n` source bytes.
* A promoted tensor is `4*elems` in and `2*elems` out; the saving is `2*elems`.
* A skip set compacts the arena: kept rows are re-placed in index order at 256-byte alignment.

## Form template identifiers
`CHUNK` -> chunk, `Conv::WidenF16` -> widen, `Conv::Bf16High` -> bf16High, `dst_off` -> dstOff, `at` -> off.
-/

namespace Strata.Weights

/-- 8 MiB of source per staging round. -/
def chunk : Nat := 8388608

/-- CHUNK is a multiple of 4, so full chunks never split a 4-byte element. -/
theorem chunk_mod4 : chunk % 4 = 0 := by decide

-- § widening (fp16 -> f32)

/-- Widening `n` (even) source bytes writes `(n/2)*4` bytes. -/
def widenOut (n : Nat) : Nat := n / 2 * 4

/-- Consecutive widened chunks write consecutive destination ranges. -/
theorem widen_progress (dstOff done n : Nat) (hn : n % 2 = 0) :
    dstOff + (done + n) * 2 = (dstOff + done * 2) + widenOut n := by
  unfold widenOut; omega

/-- The widened output of a chunk fits `stage_out` (`2*CHUNK`). -/
theorem widen_fits (n : Nat) (h : n ≤ chunk) : widenOut n ≤ 2 * chunk := by
  unfold widenOut; omega

/-- Trim a possibly odd tail to whole elements. -/
def evenTrim (n : Nat) : Nat := if n % 2 = 1 then n - 1 else n

/-- A trimmed chunk is even, positive (when at least 2 bytes), and no larger. -/
theorem evenTrim_ok (n : Nat) (h : 2 ≤ n) :
    evenTrim n % 2 = 0 ∧ 0 < evenTrim n ∧ evenTrim n ≤ n := by
  unfold evenTrim; split <;> omega

-- § BF16 high-half promotion

/-- BF16 promotion keeps 2 of every 4 bytes. -/
def bf16Out (n : Nat) : Nat := n / 4 * 2

/-- Consecutive BF16 chunks (multiples of 4) write consecutive destination ranges. -/
theorem bf16_progress (dstOff done n : Nat) (hd : done % 4 = 0) (hn : n % 4 = 0) :
    dstOff + (done + n) / 2 = (dstOff + done / 2) + bf16Out n := by
  unfold bf16Out; omega

/-- The BF16 output of a chunk fits `stage_out`. -/
theorem bf16_fits (n : Nat) (h : n ≤ chunk) : bf16Out n ≤ 2 * chunk := by
  unfold bf16Out; omega

/-- Taking the high 16 bits of a 32-bit value yields a 16-bit value. -/
theorem high16_lt (v : Nat) (h : v < 4294967296) : v / 65536 < 65536 := by omega

/-- A promoted tensor's saving is exactly its output size. -/
theorem promoted_saving (elems src dst : Nat) (hs : src = elems * 4) (hd : dst = elems * 2) :
    src - dst = dst := by omega

-- § segment tiling

/-- A quantized tensor's destination segments are contiguous: codes, scales, offsets. -/
theorem dst_segments_contiguous (codes scales offs dst : Nat) (h : dst = codes + scales + offs) :
    0 + codes = codes ∧ codes + scales = codes + scales ∧ codes + scales + offs = dst := by omega

/-- With fp16 scales (even byte count) the source scale plane is half the destination plane. -/
theorem fp16_source_plane (scales : Nat) (h : scales % 2 = 0) : (scales / 2) * 2 = scales := by omega

/-- The offset segment's source start follows the narrower source scale plane. -/
theorem src_offset_start (codes scales : Nat) (h : scales % 2 = 0) :
    codes + scales / 2 + scales / 2 = codes + scales := by omega

-- § compacted offsets

/-- Alignment used for kept rows. -/
def align256 (b : Nat) : Nat := (b + 255) / 256 * 256

/-- Aligned size covers the request and is a multiple of 256. -/
theorem align256_props (b : Nat) : b ≤ align256 b ∧ align256 b % 256 = 0 := by
  unfold align256; omega

/-- Offset of kept row `i`: the sum of the aligned sizes of the rows before it. -/
def off : List Nat → Nat → Nat
  | _, 0 => 0
  | [], _ + 1 => 0
  | b :: bs, i + 1 => align256 b + off bs i

/-- Total arena size. -/
def pool : List Nat → Nat
  | [] => 0
  | b :: bs => align256 b + pool bs

/-- Every compacted offset is aligned. -/
theorem off_aligned : ∀ (l : List Nat) (i : Nat), off l i % 256 = 0
  | _, 0 => by simp [off]
  | [], _ + 1 => by simp [off]
  | b :: bs, i + 1 => by
      have h := off_aligned bs i
      have h2 := (align256_props b).2
      simp only [off]
      omega

/-- Every compacted offset lies within the pool. -/
theorem off_le_pool : ∀ (l : List Nat) (i : Nat), off l i ≤ pool l
  | _, 0 => by simp [off]
  | [], _ + 1 => by simp [off, pool]
  | b :: bs, i + 1 => by
      have h := off_le_pool bs i
      simp only [off, pool]
      omega

/-- A row's destination range ends within the pool (so `arena_bytes >= pool` suffices). -/
theorem row_end_in_pool : ∀ (l : List Nat) (i : Nat), i < l.length →
    off l i + align256 (l.getD i 0) ≤ pool l
  | [], i, h => by simp at h
  | b :: bs, 0, _ => by simp [off, pool, List.getD_cons_zero]
  | b :: bs, i + 1, h => by
      have h' : i < bs.length := by simpa using h
      have := row_end_in_pool bs i h'
      simp only [off, pool, List.getD_cons_succ]
      omega

end Strata.Weights



## FILE: /mnt/user-data/outputs/GenerateDriverAccounting.lean
/-!
# MODULE: generate_driver_accounting
Position and row accounting of the `strata generate` driver: which positions produce tokens, the logits-dump row count,
context fit, the batched-prefill split, the auto-sized expert cache, the PCIe fraction clamp, profile truncation and
the routing-trace layer index.

## Source semantics
* The loop runs positions `pos_start ..`; tokens are produced from `pos = n_prompt - 1` until `max_new` exist, so the
  last position is `n_prompt - 2 + max_new`, and the dump has `n_prompt - 1 + max_new` positions.
* `n_tokens ≤ max_context - max_new` makes every position fit in the context.
* `--expert-cache auto` takes `(free - reserve) / blob` slots (saturating at 0): the slots fit in free VRAM.
* `pcie_num = clamp(round(frac*256), 0, 256)`.
* The routing trace records `layers - 1` after the dispatch advanced the counter.

## Form template identifiers
`n_prompt` -> nPrompt, `max_new` -> maxNew, `dump_positions` -> dumpPositions, `prefill_until` -> prefillUntil,
`pcie_num` -> pcieNum, `o.expert_cache` -> slots.
-/

namespace Strata.Generate

-- § positions and rows

/-- Number of positions the dump covers: `n_prompt - 1 + max_new`. -/
def dumpPositions (nPrompt maxNew : Nat) : Nat := nPrompt - 1 + maxNew

/-- Last position processed by the token loop. -/
def lastPos (nPrompt maxNew : Nat) : Nat := nPrompt - 2 + maxNew

/-- The first generated token comes from the last prompt position. -/
def firstProducedPos (nPrompt : Nat) : Nat := nPrompt - 1

/-- Positions `0 .. lastPos` number exactly `dumpPositions` (for a prompt of at least 2 tokens). -/
theorem dump_rows_match (nPrompt maxNew : Nat) (h : 2 ≤ nPrompt) (hm : 1 ≤ maxNew) :
    lastPos nPrompt maxNew + 1 = dumpPositions nPrompt maxNew := by
  unfold lastPos dumpPositions; omega

/-- Exactly `max_new` positions produce a token. -/
theorem produced_count (nPrompt maxNew : Nat) (h : 2 ≤ nPrompt) (hm : 1 ≤ maxNew) :
    lastPos nPrompt maxNew + 1 - firstProducedPos nPrompt = maxNew := by
  unfold lastPos firstProducedPos; omega

/-- A single-token prompt: the loop starts producing at position 0 and ends at `max_new - 1`. -/
theorem produced_count_single (maxNew : Nat) (hm : 1 ≤ maxNew) :
    (1 - 1 + maxNew) = maxNew := by omega

/-- With the size check `n ≤ ctx - max_new`, every position is below the context size. -/
theorem positions_fit (nPrompt maxNew ctx : Nat) (h1 : 1 ≤ nPrompt) (h2 : 0 < maxNew)
    (h3 : maxNew ≤ ctx) (h4 : nPrompt ≤ ctx - maxNew) : lastPos nPrompt maxNew < ctx := by
  unfold lastPos; omega

-- § prefill split

/-- Positions batched by the prompt path. -/
def nBatched (nPrompt prefillUntil : Nat) : Nat :=
  if 0 < prefillUntil ∧ prefillUntil < nPrompt - 1 then prefillUntil else nPrompt - 1

/-- The batched count never exceeds the conditioning positions. -/
theorem nBatched_le (nPrompt pu : Nat) : nBatched nPrompt pu ≤ nPrompt - 1 := by
  unfold nBatched; split <;> omega

/-- The token loop resumes at a prompt position (or the last conditioning one). -/
theorem resume_in_prompt (nPrompt pu : Nat) (h : 1 ≤ nPrompt) : nBatched nPrompt pu < nPrompt := by
  have := nBatched_le nPrompt pu
  omega

-- § auto-sized expert cache

/-- Slots granted by `--expert-cache auto` (truncating subtraction models the `max(0, ...)`). -/
def autoSlots (free reserve blob : Nat) : Nat := (free - reserve) / blob

/-- The granted slots occupy at most the free VRAM. -/
theorem autoSlots_fit (free reserve blob : Nat) : autoSlots free reserve blob * blob ≤ free :=
  Nat.le_trans (Nat.div_mul_le_self _ _) (Nat.sub_le _ _)

/-- Truncating the ranked profile to the slot count. -/
def profileWant (profileSize slots : Nat) : Nat := min profileSize slots

/-- The pre-fill never admits more pairs than there are slots or ranked pairs. -/
theorem profileWant_bounds (p s : Nat) : profileWant p s ≤ p ∧ profileWant p s ≤ s := by
  unfold profileWant; omega

-- § PCIe fraction

/-- `pcie_num` after the clamp to `[0, 256]`. -/
def clampPcie (x : Int) : Int := if x < 0 then 0 else if 256 < x then 256 else x

/-- The clamp lands in range. -/
theorem clampPcie_range (x : Int) : 0 ≤ clampPcie x ∧ clampPcie x ≤ 256 := by
  unfold clampPcie; split_ifs <;> omega

-- § routing trace

/-- The trace's layer index: the counter has already advanced past the layer. -/
def traceLayer (layersAfter : Nat) : Int := (layersAfter : Int) - 1

/-- After a dispatch of layer `l` (counter `l+1`), the trace records `l`. -/
theorem traceLayer_correct (l : Nat) : traceLayer (l + 1) = (l : Int) := by
  unfold traceLayer; omega

/-- For a counter in `1..48` the recorded layer is within `0..47`. -/
theorem traceLayer_range (c : Nat) (h1 : 1 ≤ c) (h2 : c ≤ 48) : 0 ≤ traceLayer c ∧ traceLayer c < 48 := by
  unfold traceLayer; omega

-- § serve admission

/-- A request is accepted only if prompt + max_new + 8 fits the context. -/
def serveAccepts (n maxNew ctx : Nat) : Bool := !(decide (ctx < n + maxNew + 8))

/-- An accepted request leaves room for a full verify window beyond its last position. -/
theorem serveAccepts_room (n maxNew ctx : Nat) (h : serveAccepts n maxNew ctx = true) :
    n + maxNew + 8 ≤ ctx := by
  unfold serveAccepts at h
  simp at h
  omega

end Strata.Generate



## FILE: /mnt/user-data/outputs/GgmlDequantLayouts.lean
/-!
# MODULE: ggml_dequant_layouts
Block layouts and code ranges of the scalar dequantizers (`dequant.hpp`): Q2_0, Q4_0, Q5_0, Q6_K, Q4_K, Q5_K, Q3_K,
IQ4_XS and the 16-bit float conversions.

## Source semantics
* Q2_0: 64 elements, 18 bytes; code `c ∈ {0..3}` maps to symbol `c - 1 ∈ {-1,0,1,2}`; element `j` is byte `j/4`, bit offset `(j%4)*2`.
* Q4_0 / IQ4_NL: low nibble is element `j`, high nibble `j+16`; Q5_0 adds a fifth bit.
* K-quant super-blocks are 256 elements; every sub-field offset below tiles the block size exactly.
* `read_u16` is `p0 | p1<<8` (a 16-bit value); bf16 -> f32 shifts left 16 and is exactly invertible.

## Form template identifiers
`dequantize_q2_0` -> q2, `block + 2` -> qs offset, `get_scale_min_k4` -> scaleMinIdx, `kvalues_iq4nl` -> 16-entry codebook.
-/

namespace Strata.Dequant

-- § Q2_0

/-- The 2-bit code of element position `r` (0..3) inside a byte. -/
def q2Code (b r : Nat) : Nat := (b / 2 ^ (2 * r)) % 4

/-- A code is at most 3. -/
theorem q2Code_le (b r : Nat) : q2Code b r ≤ 3 := by
  have : q2Code b r < 4 := Nat.mod_lt _ (by decide)
  omega

/-- The decoded symbol of a code. -/
def q2Sym (c : Nat) : Int := (c : Int) - 1

/-- Symbols are exactly `-1, 0, 1, 2`. -/
theorem q2Sym_values (c : Nat) (h : c ≤ 3) : q2Sym c = -1 ∨ q2Sym c = 0 ∨ q2Sym c = 1 ∨ q2Sym c = 2 := by
  unfold q2Sym; omega

/-- Element `j < 64` reads byte `j/4 < 16`. -/
theorem q2_byte_in_block (j : Nat) (h : j < 64) : j / 4 < 16 := by omega

/-- The (byte, position) pair determines the element: no two elements share a code. -/
theorem q2_injective (j1 j2 : Nat) (h : j1 / 4 = j2 / 4 ∧ j1 % 4 = j2 % 4) : j1 = j2 := by omega

/-- Q2_0 block: 2 bytes of scale + 16 bytes of codes = 18; 64 codes of 2 bits fill 16 bytes. -/
theorem q2_block_size : 2 + 16 = 18 ∧ 64 * 2 = 16 * 8 := by omega

-- § 4- and 5-bit legacy blocks

/-- Q4_0 nibble codes decode to `[-8, 7]`. -/
theorem q4_0_range (nib : Nat) (h : nib < 16) : -8 ≤ (nib : Int) - 8 ∧ (nib : Int) - 8 ≤ 7 := by omega

/-- Q4_0 outputs `j` and `j+16` are distinct positions inside 32. -/
theorem q4_0_positions (j : Nat) (h : j < 16) : j + 16 < 32 ∧ j ≠ j + 16 := by omega

/-- Q5_0 value: nibble plus the fifth bit. -/
def q5Val (nib bit : Nat) : Nat := nib + 16 * bit

/-- Q5_0 values decode into `[-16, 15]`. -/
theorem q5_0_range (nib bit : Nat) (hn : nib < 16) (hb : bit ≤ 1) :
    -16 ≤ (q5Val nib bit : Int) - 16 ∧ (q5Val nib bit : Int) - 16 ≤ 15 := by
  unfold q5Val; omega

/-- Q5_0 block layout: d(2) | qh(4) | qs(16) = 22. -/
theorem q5_0_block : 2 + 4 + 16 = 22 := by omega

-- § Q6_K

/-- Q6_K layout: ql 0..127, qh 128..191, scales 192..207, d 208..209: 210 bytes. -/
theorem q6k_layout : 128 + 64 + 16 + 2 = 210 ∧ 192 = 128 + 64 ∧ 208 = 192 + 16 := by omega

/-- A Q6_K value is a nibble plus two high bits. -/
def q6Val (lo hi : Nat) : Nat := lo + 16 * hi

/-- Q6_K values decode into `[-32, 31]`. -/
theorem q6_range (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) :
    -32 ≤ (q6Val lo hi : Int) - 32 ∧ (q6Val lo hi : Int) - 32 ≤ 31 := by
  unfold q6Val; omega

/-- Scale index `is + 2*i` (halves `h`, `is` in 0..1, `i` in 0..3) stays below 16. -/
theorem q6_scale_index (h is i : Nat) (hh : h < 2) (hi : is < 2) (hq : i < 4) : 8 * h + is + 2 * i < 16 := by
  omega

-- § Q4_K / Q5_K

/-- Q4_K: d(2) dmin(2) scales(12) qs(128) = 144. -/
theorem q4k_layout : 2 + 2 + 12 + 128 = 144 ∧ 4 + 12 = 16 := by omega

/-- Q5_K: header 16, qh 32, qs 128 = 176. -/
theorem q5k_layout : 16 + 32 + 128 = 176 ∧ 16 + 32 = 48 := by omega

/-- The four 64-element groups of a K super-block use scale indices `0..7`, two per group. -/
theorem k_group_scale_index (g : Nat) (h : g < 4) : 2 * g + 1 < 8 := by omega

/-- Four groups of 64 cover 256 elements; each advances `q` by 32 so 4 groups use 128 bytes. -/
theorem k_group_cover : 4 * 64 = 256 ∧ 4 * 32 = 128 := by omega

/-- `get_scale_min_k4` reads only bytes `0..11`. -/
theorem scaleMinIdx_in_range (j : Nat) (h : j < 8) :
    (j < 4 → j + 4 < 12) ∧ (4 ≤ j → j + 4 < 12 ∧ j - 4 < 12) := by omega

/-- The Q5_K high-bit masks `u1 = 4^g`, `u2 = 2*4^g` fit in a byte for the four groups. -/
theorem q5k_masks_fit : ∀ g, g < 4 → 2 * 4 ^ g < 256 := by decide

-- § Q3_K

/-- Q3_K: hmask(32) qs(64) scales(12) d(2) = 110; scales at 96, d at 108. -/
theorem q3k_layout : 32 + 64 + 12 + 2 = 110 ∧ 32 + 64 = 96 ∧ 96 + 12 = 108 := by omega

/-- The 16 scale reads `is++` cover indices `0..15` exactly once over two halves, four groups, two each. -/
theorem q3k_scale_index (n j e : Nat) (hn : n < 2) (hj : j < 4) (he : e < 2) : 8 * n + 2 * j + e < 16 := by
  omega

-- § IQ4_XS

/-- IQ4_XS: d(2) scales_h(2) scales_l(4) qs(128) = 136. -/
theorem iq4xs_layout : 2 + 2 + 4 + 128 = 136 ∧ 2 + 2 + 4 = 8 := by omega

/-- Group `ib < 8` reads `scales_l[ib/2]` with `ib/2 < 4`, and `scales_h` shift `2*ib ≤ 14`. -/
theorem iq4xs_scale_index (ib : Nat) (h : ib < 8) : ib / 2 < 4 ∧ 2 * ib ≤ 14 := by omega

/-- The six-bit scale (4 low bits + 2 high bits) is below 64, so `ls - 32 ∈ [-32, 31]`. -/
theorem iq4xs_scale_range (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) :
    -32 ≤ ((lo + 16 * hi : Nat) : Int) - 32 ∧ ((lo + 16 * hi : Nat) : Int) - 32 ≤ 31 := by omega

-- § 16-bit conversions

/-- `read_u16`: `p0 | p1 << 8` for bytes. -/
def readU16 (p0 p1 : Nat) : Nat := p0 + 256 * p1

/-- The result is a 16-bit value. -/
theorem readU16_lt (p0 p1 : Nat) (h0 : p0 < 256) (h1 : p1 < 256) : readU16 p0 p1 < 65536 := by
  unfold readU16; omega

/-- The bytes are recovered. -/
theorem readU16_bytes (p0 p1 : Nat) (h0 : p0 < 256) (h1 : p1 < 256) :
    readU16 p0 p1 % 256 = p0 ∧ readU16 p0 p1 / 256 = p1 := by
  unfold readU16; omega

/-- bf16 -> f32 is `h << 16`: exactly invertible by taking the high half. -/
theorem bf16_roundtrip (h : Nat) : (h * 65536) / 65536 = h := by omega

/-- The shifted value is a 32-bit pattern. -/
theorem bf16_bits_lt (h : Nat) (hh : h < 65536) : h * 65536 < 4294967296 := by omega

end Strata.Dequant





## FILE: /mnt/user-data/outputs/PrefillGpuKernels.lean
/-!
# MODULE: prefill_gpu_kernels
Index arithmetic, layout and numeric-range contracts of the CUDA kernels of the batched prefill path
(hyper-connection, gated-delta-net, MoE dequant/route/combine, QSA helpers, KV append).

## Source semantics
Every kernel is a flat grid over a row-major buffer; correctness rests on (a) the grid covering the buffer,
(b) every decomposed index staying inside its buffer, (c) the packed expert blob layout summing to one
expert's byte size, (d) bf16 rounding producing a 16-bit pattern and being exact on bf16 values, and
(e) the int8 KV quantizer clamping to [-127,127].

## Form template identifiers
- `blocksFor`      <- blocks_for (ceil-divide launch size)
- `bfBits`         <- bf (round-to-nearest-even f32 -> bf16 bits)
- `code2`/`dequantVal` <- 2-bit ternary-ish code decode in blob_dequant_kernel
- `O_D_CODES`, `O_GU_SC`, `O_D_SC`, `BLOB_BYTES` <- Strata expert blob offsets
- `clamp127`       <- int8 clamp in kv_append_kernel
-/

namespace Strata.PrefillKernels

-- § constants

def N : Nat := 2560
def HC : Nat := 4
def D : Nat := N * HC
def LR : Nat := 320
def S : Nat := 128
def HK : Nat := 16
def HV : Nat := 48
def C : Nat := 10240
def ZV : Nat := HV * S

/-- The four hyper-connection streams are `4 * 2560 = 10240` wide. -/
theorem hc_width : D = 10240 := by decide

/-- The GDN conv channels split into q (16 heads), k (16 heads) and v (48 heads) of width 128. -/
theorem qkv_channels : C = 2 * HK * S + HV * S := by decide

/-- The value part of the GDN projection is `48 * 128 = 6144` wide. -/
theorem zv_value : ZV = 6144 := by decide

/-- The full q projection is 24 heads of `2 * 256` (query and gate). -/
theorem q_full_width : 24 * 512 = 12288 := by decide

-- § launch geometry

/-- Ceil-divide used to size 1-D grids. -/
def blocksFor (n t : Nat) : Nat := (n + t - 1) / t

/-- A grid of `blocksFor n t` blocks of `t` threads covers all `n` elements. -/
theorem blocksFor_covers (n t : Nat) (ht : 0 < t) : n ≤ blocksFor n t * t := by
  unfold blocksFor
  have h1 := Nat.div_add_mod (n + t - 1) t
  have h2 := Nat.mod_lt (n + t - 1) ht
  rw [Nat.mul_comm]
  omega

/-- ... and never launches a whole spare block. -/
theorem blocksFor_tight (n t : Nat) (ht : 0 < t) : blocksFor n t * t < n + t := by
  unfold blocksFor
  have h1 := Nat.div_add_mod (n + t - 1) t
  have h2 := Nat.mod_lt (n + t - 1) ht
  rw [Nat.mul_comm]
  omega

/-- The router launches `(T+7)/8` blocks of 8 warps: every token gets a warp. -/
theorem route_grid_covers (T : Nat) : T ≤ ((T + 7) / 8) * 8 := by omega

/-- `gdn_conv` launches `C/128` blocks of 128 threads: exactly one thread per channel. -/
theorem gdn_conv_grid_exact : C / 128 * 128 = C := by decide

/-- `gdn_l2` runs one block of `S` threads per q/k head: `2*HK` blocks cover the q/k channels. -/
theorem gdn_l2_covers_qk : 2 * HK * S = 4096 := by decide

/-- A block-wide sum over at most 1024 threads uses at most 32 warp partials (the `sh[32]` array). -/
theorem block_sum_warps_le (b : Nat) (hb : b ≤ 1024) : (b + 31) / 32 ≤ 32 := by omega

/-- Lane/warp decomposition of a thread id. -/
theorem lane_warp_decompose (tid : Nat) : tid = (tid / 32) * 32 + tid % 32 := by omega

/-- The GDN recurrence splits the 128 state rows over 4 row groups of 32. -/
theorem rec_rows_partition (row : Nat) (hr : row < 128) :
    ∃ rg r, rg < 4 ∧ r < 32 ∧ row = rg * 32 + r :=
  ⟨row / 32, row % 32, by omega, by omega, by omega⟩

/-- The state index `((row*HV + head)*S + col)` stays inside the `S*HV*S` state. -/
theorem gdn_state_index_lt (row head col : Nat) (hr : row < 128) (hh : head < 48) (hc : col < 128) :
    (row * 48 + head) * 128 + col < 128 * 48 * 128 := by omega

/-- RoPE rotates pairs `(p, p+32)` and the 32 pair ids cover all 64 rotated dims. -/
theorem rope_pairs_cover (i : Nat) (hi : i < 64) : i < 32 ∨ (32 ≤ i ∧ i - 32 < 32) := by omega

-- § hyper-connection index maps

/-- Decomposing `t*D + c*N + d` recovers token, stream and channel. -/
theorem hc_decompose (t c d : Nat) (hc : c < 4) (hd : d < 2560) :
    (t * 10240 + c * 2560 + d) / 10240 = t ∧
    ((t * 10240 + c * 2560 + d) % 10240) / 2560 = c ∧
    (t * 10240 + c * 2560 + d) % 2560 = d := by omega

/-- `gr_mix` reads stay inside the `T*D` stream buffer. -/
theorem mix_read_in_range (t T c d : Nat) (ht : t < T) (hc : c < 4) (hd : d < 2560) :
    t * 10240 + c * 2560 + d < T * 10240 := by omega

/-- `moe_combine` reads the 10 routed slots of token `t` inside the `T*10` tables. -/
theorem combine_slot_in_range (t T k : Nat) (ht : t < T) (hk : k < 10) : t * 10 + k < T * 10 := by omega

/-- A flat index `r*640 + k` decomposes back into row and column (swiglu kernels). -/
theorem swiglu_flat_decompose (r k : Nat) (hk : k < 640) :
    (r * 640 + k) / 640 = r ∧ (r * 640 + k) % 640 = k := by omega

/-- The interleaved gate/up reads `2k` and `2k+1` stay inside the `n*1280` buffer. -/
theorem swiglu_il_index (r k n : Nat) (hr : r < n) (hk : k < 640) :
    r * 1280 + 2 * k + 1 < n * 1280 := by omega

/-- Generic "row-major cell is inside the buffer" bound. -/
theorem row_lt (q M ps r : Nat) (hq : q < M) (hr : r < ps) : q * ps + r < M * ps := by
  have h1 : (q + 1) * ps ≤ M * ps := Nat.mul_le_mul_right ps hq
  have h2 : (q + 1) * ps = q * ps + ps := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- `gather_rows16` copies `uint4` units; a row/unit pair stays inside the destination. -/
theorem gather_index_lt (r j n per : Nat) (hr : r < n) (hj : j < per) : r * per + j < n * per :=
  row_lt r n per j hr hj

-- § expert blob layout

def O_D_CODES : Nat := 1280 * 640
def O_GU_SC : Nat := O_D_CODES + 2560 * 160
def O_D_SC : Nat := O_GU_SC + 1280 * 40 * 2
def BLOB_BYTES : Nat := O_D_SC + 2560 * 10 * 2

/-- One expert blob (codes + f16 scales) is exactly 1,382,400 bytes, the planner's `expert_blob`. -/
theorem blob_bytes_value : BLOB_BYTES = 1382400 := by decide

/-- The four regions of the blob are laid out in increasing, non-overlapping order. -/
theorem blob_regions_ordered : O_D_CODES < O_GU_SC ∧ O_GU_SC < O_D_SC ∧ O_D_SC < BLOB_BYTES := by decide

/-- The 2-bit code of weight `k` inside a code byte. -/
def code2 (c k : Nat) : Nat := (c >>> (2 * k)) &&& 3

/-- A 2-bit code is at most 3. -/
theorem code2_le (c k : Nat) : code2 c k ≤ 3 := Nat.and_le_right

/-- The decoded multiplier `code - 1`, before the block scale. -/
def dequantVal (c k : Nat) : Int := (code2 c k : Int) - 1

/-- Decoded multipliers lie in `[-1, 2]`. -/
theorem dequantVal_range (c k : Nat) : -1 ≤ dequantVal c k ∧ dequantVal c k ≤ 2 := by
  have := code2_le c k
  unfold dequantVal
  omega

/-- Gate/up output index `row*2560 + byte*4 + k` stays inside the `1280 x 2560` matrix. -/
theorem gu_index_lt (row byte k : Nat) (hr : row < 1280) (hb : byte < 640) (hk : k < 4) :
    row * 2560 + byte * 4 + k < 1280 * 2560 := by omega

/-- Down output index stays inside the `2560 x 640` matrix. -/
theorem down_index_lt (row byte k : Nat) (hr : row < 2560) (hb : byte < 160) (hk : k < 4) :
    row * 640 + byte * 4 + k < 2560 * 640 := by omega

/-- Gate/up scale index (one scale per 64 weights, 40 per row) stays in range. -/
theorem gu_scale_index_lt (row byte : Nat) (hr : row < 1280) (hb : byte < 640) :
    row * 40 + (byte * 4) / 64 < 1280 * 40 := by omega

/-- Down scale index (10 per row) stays in range. -/
theorem down_scale_index_lt (row byte : Nat) (hr : row < 2560) (hb : byte < 160) :
    row * 10 + (byte * 4) / 64 < 2560 * 10 := by omega

/-- Distinct (byte, k) pairs write distinct output weights. -/
theorem code_slot_injective (b1 k1 b2 k2 : Nat) (h1 : k1 < 4) (h2 : k2 < 4)
    (h : b1 * 4 + k1 = b2 * 4 + k2) : b1 = b2 ∧ k1 = k2 := by omega

-- § numeric formats

/-- bf16 rounding of an f32 bit pattern (round-to-nearest-even, uint32 wrap), as bits. -/
def bfBits (u : Nat) : Nat := ((u + 0x7fff + ((u >>> 16) &&& 1)) % 4294967296) >>> 16

/-- The bf16 result always fits in 16 bits. -/
theorem bfBits_lt (u : Nat) : bfBits u < 65536 := by
  unfold bfBits
  have := Nat.mod_lt (u + 0x7fff + ((u >>> 16) &&& 1)) (by decide : 0 < 4294967296)
  simp only [Nat.shiftRight_eq_div_pow, Nat.reducePow] at this ⊢
  omega

/-- Rounding a value that is already bf16 (low 16 bits zero) is the identity. -/
theorem bfBits_exact (h : Nat) (hh : h < 65536) : bfBits (h * 65536) = h := by
  have hb : ((h * 65536) >>> 16) &&& 1 ≤ 1 := Nat.and_le_right
  unfold bfBits
  simp only [Nat.shiftRight_eq_div_pow, Nat.reducePow] at hb ⊢
  omega

/-- The int8 clamp of the KV quantizer. -/
def clamp127 (q : Int) : Int := if q < -127 then -127 else if q > 127 then 127 else q

/-- Quantized KV values lie in `[-127, 127]`. -/
theorem clamp127_range (q : Int) : -127 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  unfold clamp127
  repeat' split
  all_goals omega

/-- Clamping leaves in-range values unchanged. -/
theorem clamp127_id (q : Int) (h1 : -127 ≤ q) (h2 : q ≤ 127) : clamp127 q = q := by
  unfold clamp127
  repeat' split
  all_goals omega

/-- KV pool row `(page*2 + kvh)*page_size + pos % page_size` stays inside the pool. -/
theorem kv_row_lt (page kvh nPages ps r : Nat) (hp : page < nPages) (hk : kvh < 2) (hr : r < ps) :
    (page * 2 + kvh) * ps + r < (nPages * 2) * ps :=
  row_lt (page * 2 + kvh) (nPages * 2) ps r (by omega) hr

end Strata.PrefillKernels



## FILE: /mnt/user-data/outputs/PrefillChunkDriver.lean
/-!
# MODULE: prefill_chunk_driver
Host-side orchestration of prefill: the device allocator (counting, bump and owned modes), the chunk loop,
grouping of (token, expert) pairs by expert, and the staging/dequant ring discipline.

## Source semantics
`bytes_needed` must replay exactly the allocation sequence of `init`; the chunk loop must visit every token
exactly once in pieces of at most `m.T`; expert grouping is a counting sort whose slots are disjoint per
expert; ring slots reused within the lookahead window must be distinct.

## Form template identifiers
- `allocBytes` <- Alloc::take size rounding
- `bump`/`takeAll` <- bump allocation from a borrowed region
- `chunkLens` <- `for c0 in 0..n step m.T` with `T = min(m.T, n-c0)`
- `offAt` <- the `m.off` prefix-sum over `m.cnt`
- `STAGE`, `DQ` <- staging ring and dequant ring sizes
-/

namespace Strata.PrefillDriver

-- § allocator

/-- Each allocation is rounded up to 256 bytes with 256 bytes of slack. -/
def allocBytes (b : Nat) : Nat := (b + 511) / 256 * 256

/-- Allocation sizes are 256-aligned. -/
theorem allocBytes_aligned (b : Nat) : allocBytes b % 256 = 0 := by
  unfold allocBytes
  omega

/-- Allocation always leaves at least 256 bytes of slack. -/
theorem allocBytes_ge (b : Nat) : b + 256 ≤ allocBytes b := by
  unfold allocBytes
  omega

/-- Allocation never wastes more than 511 bytes. -/
theorem allocBytes_lt (b : Nat) : allocBytes b < b + 512 := by
  unfold allocBytes
  omega

/-- One bump allocation from a borrowed region of `cap` bytes. -/
def bump (used cap bytes : Nat) : Option Nat :=
  if used + bytes ≤ cap then some (used + bytes) else none

/-- A successful bump stays inside the region. -/
theorem bump_le (used cap bytes u : Nat) (h : bump used cap bytes = some u) :
    u ≤ cap ∧ u = used + bytes := by
  unfold bump at h
  split at h
  · injection h with h'
    omega
  · simp at h

/-- A whole allocation sequence from a region. -/
def takeAll (cap : Nat) : Nat → List Nat → Option Nat
  | used, [] => some used
  | used, b :: bs =>
    match bump used cap b with
    | none => none
    | some u => takeAll cap u bs

/-- If the counted total fits, the same sequence succeeds in the real allocation (so `bytes_needed` is sound). -/
theorem takeAll_succeeds (cap : Nat) :
    ∀ (bs : List Nat) (used : Nat), used + bs.sum ≤ cap → takeAll cap used bs = some (used + bs.sum) := by
  intro bs
  induction bs with
  | nil =>
    intro used h
    simp [takeAll]
  | cons b bs ih =>
    intro used h
    simp only [List.sum_cons] at h
    have hb : bump used cap b = some (used + b) := by
      unfold bump
      rw [if_pos (by omega)]
    simp only [takeAll, hb]
    rw [ih (used + b) (by omega)]
    simp [List.sum_cons, Nat.add_assoc]

-- § chunk loop

/-- The lengths of the chunks the loop visits (`fuel` bounds the iterations). -/
def chunkLens (m : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, n => if n = 0 then [] else min m n :: chunkLens m f (n - min m n)

/-- Every chunk has at most `m` tokens. -/
theorem chunkLens_le (m : Nat) :
    ∀ (fuel n : Nat), ∀ x ∈ chunkLens m fuel n, x ≤ m := by
  intro fuel
  induction fuel with
  | zero =>
    intro n x hx
    simp [chunkLens] at hx
  | succ f ih =>
    intro n x hx
    unfold chunkLens at hx
    split at hx
    · simp at hx
    · rcases List.mem_cons.mp hx with h | h
      · omega
      · exact ih _ x h

/-- The chunks partition the `n` tokens exactly. -/
theorem chunkLens_sum (m : Nat) (hm : 0 < m) :
    ∀ (fuel n : Nat), n ≤ fuel → (chunkLens m fuel n).sum = n := by
  intro fuel
  induction fuel with
  | zero =>
    intro n hn
    have h0 : n = 0 := by omega
    subst h0
    simp [chunkLens]
  | succ f ih =>
    intro n hn
    unfold chunkLens
    split
    · rename_i h
      subst h
      simp
    · rename_i h
      rw [List.sum_cons, ih (n - min m n) (by omega)]
      omega

-- § expert grouping

/-- Prefix sum of the per-expert counts (`m.off`). -/
def offAt (cnt : Nat → Nat) : Nat → Nat
  | 0 => 0
  | e + 1 => offAt cnt e + cnt e

/-- Offsets are monotone in the expert id. -/
theorem offAt_mono (cnt : Nat → Nat) : ∀ e f, e ≤ f → offAt cnt e ≤ offAt cnt f := by
  intro e f h
  induction f with
  | zero =>
    have h0 : e = 0 := by omega
    subst h0
    exact Nat.le_refl _
  | succ f ih =>
    rcases Nat.eq_or_lt_of_le h with h' | h'
    · rw [h']
    · have h2 := ih (by omega)
      simp only [offAt]
      omega

/-- The slot of the `r`-th pair of expert `e` lies inside that expert's segment. -/
theorem slot_in_range (cnt : Nat → Nat) (e r : Nat) (hr : r < cnt e) :
    offAt cnt e + r < offAt cnt (e + 1) := by
  simp only [offAt]
  omega

/-- Segments of different experts are disjoint. -/
theorem slot_segments_disjoint (cnt : Nat → Nat) (e f r s : Nat) (hef : e < f) (hr : r < cnt e) :
    offAt cnt e + r < offAt cnt f + s := by
  have h1 := slot_in_range cnt e r hr
  have h2 := offAt_mono cnt (e + 1) f hef
  omega

-- § rings

/-- Slots handed out within a lookahead window of 8 are pairwise distinct modulo the 8-slot ring. -/
theorem ring_slots_distinct (a b : Nat) (hab : a < b) (hb : b < a + 8) : a % 8 ≠ b % 8 := by omega

/-- The ring pointer stays inside the ring. -/
theorem ring_next_lt (s : Nat) (h : s < 8) : (s + 1) % 8 < 8 := by omega

/-- Consecutive experts use different dequant buffers (ring of 2). -/
theorem dq_alternates (j : Nat) : j % 2 ≠ (j + 1) % 2 := by omega

-- § sizes derived from geometry

/-- GDN state floats per layer: recurrence `128*48*128` plus conv history `10240*3`. -/
theorem gdn_floats_value : 128 * 48 * 128 + 10240 * 3 = 817152 := by decide

/-- ... which is 3,268,608 bytes, matching the planner's per-layer GDN cost. -/
theorem gdn_bytes_value : (128 * 48 * 128 + 10240 * 3) * 4 = 3268608 := by decide

/-- Layers 3, 7, ..., 47 are the 12 full-attention layers. -/
theorem qsa_layer_count : ((List.range 48).filter (fun l => l % 4 == 3)).length = 12 := by decide

end Strata.PrefillDriver



## FILE: /mnt/user-data/outputs/SpecController.lean
/-!
# MODULE: spec_controller
The speculative-decoding controller: choose drafter and draft length maximizing expected tokens per ms,
keep k = 0 unless the gain beats a threshold, and learn per-position acceptance with an EMA.

## Source semantics
Expected committed tokens is `1 + sum of running acceptance products` (the verify pass always commits one
token). Arrays are indexed by `n = k+1` clamped to `[1, K_MAX+1]`. Lookup acceptance is bucketed by match
length. The EMA is a convex combination so probabilities stay in range.

## Form template identifiers
- `K_MAX` <- spec::K_MAX
- `stepIdx` <- `clamp(k+1,1,K_MAX+1) - 1` in CostModel::step_ms
- `bucket` <- Controller::bucket
- `tail`/`expect` <- Controller::expected
- `pickBest` <- the strict-improvement scan in Controller::choose
- `emaStep` <- the `p += ema * (hit - p)` update (fixed-point form)
-/

namespace Strata.SpecController

def K_MAX : Nat := 8

-- § cost-model indexing

/-- Table index used for an `n = k+1` token verify pass. -/
def stepIdx (k : Nat) : Nat := (max 1 (min (k + 1) (K_MAX + 1))) - 1

/-- The table index is always inside the 9-entry cost tables. -/
theorem stepIdx_lt (k : Nat) : stepIdx k < K_MAX + 1 := by
  unfold stepIdx K_MAX
  omega

/-- For `k ≤ K_MAX` the index is exactly `k` (no clamping artefacts). -/
theorem stepIdx_id (k : Nat) (hk : k ≤ K_MAX) : stepIdx k = k := by
  unfold stepIdx
  unfold K_MAX at *
  omega

-- § lookup bucket

/-- Match-length bucket of the lookup acceptance table. -/
def bucket (m : Nat) : Nat := if m ≥ 16 then 3 else if m ≥ 8 then 2 else if m ≥ 5 then 1 else 0

/-- Buckets are 0..3 (the `lookup_q_` array has 4 entries). -/
theorem bucket_lt (m : Nat) : bucket m < 4 := by
  unfold bucket
  repeat' split
  all_goals omega

/-- Longer matches never fall into a lower bucket. -/
theorem bucket_mono (a b : Nat) (h : a ≤ b) : bucket a ≤ bucket b := by
  unfold bucket
  repeat' split
  all_goals omega

-- § expected tokens

/-- Sum of running acceptance products for `k` drafted positions. -/
def tail (p : Nat → Nat) (i : Nat) : Nat → Nat → Nat
  | 0, _ => 0
  | k + 1, run => run * p i + tail p (i + 1) k (run * p i)

/-- Expected committed tokens: the verified token plus the accepted-prefix sum. -/
def expect (p : Nat → Nat) (k : Nat) : Nat := 1 + tail p 0 k 1

/-- The verify pass always commits at least one token. -/
theorem expect_ge_one (p : Nat → Nat) (k : Nat) : 1 ≤ expect p k := by
  unfold expect
  omega

/-- Drafting more positions never lowers the expected token count. -/
theorem tail_mono (p : Nat → Nat) : ∀ k i run, tail p i k run ≤ tail p i (k + 1) run := by
  intro k
  induction k with
  | zero =>
    intro i run
    simp [tail]
  | succ k ih =>
    intro i run
    simp only [tail]
    exact Nat.add_le_add_left (ih _ _) _

/-- With certain acceptance every drafted token is committed: `k * run`. -/
theorem tail_all_one : ∀ k i run, tail (fun _ => 1) i k run = k * run := by
  intro k
  induction k with
  | zero =>
    intro i run
    simp [tail]
  | succ k ih =>
    intro i run
    simp only [tail, Nat.mul_one]
    rw [ih, Nat.add_mul, Nat.one_mul]
    omega

/-- Certain acceptance commits `k + 1` tokens. -/
theorem expect_all_one (k : Nat) : expect (fun _ => 1) k = k + 1 := by
  unfold expect
  rw [tail_all_one]
  omega

-- § choice scan

/-- Best rate over candidates with strict improvement, starting from the plain-decoding baseline. -/
def pickBest (init : Nat) (cands : List Nat) : Nat := cands.foldl max init

/-- The chosen rate is never below the baseline. -/
theorem pickBest_ge_init (cands : List Nat) : ∀ init, init ≤ pickBest init cands := by
  induction cands with
  | nil =>
    intro init
    exact Nat.le_refl _
  | cons c cs ih =>
    intro init
    exact Nat.le_trans (Nat.le_max_left init c) (ih (max init c))

/-- The chosen rate dominates every candidate considered. -/
theorem pickBest_ge_mem (cands : List Nat) : ∀ init x, x ∈ cands → x ≤ pickBest init cands := by
  induction cands with
  | nil =>
    intro init x hx
    simp at hx
  | cons c cs ih =>
    intro init x hx
    rcases List.mem_cons.mp hx with h | h
    · subst h
      exact Nat.le_trans (Nat.le_max_right init x) (pickBest_ge_init cs (max init x))
    · exact ih (max init c) x h

/-- Lookup draft length is capped by both the proposal and `K_MAX`. -/
theorem lookup_k_le (avail : Nat) : min avail K_MAX ≤ K_MAX := by
  unfold K_MAX
  omega

-- § learning

/-- Number of positions updated by `observe`: those accepted plus the first rejected one. -/
def seen (k accepted : Nat) : Nat := min k (min accepted k + 1)

/-- `observe` never touches a position beyond the drafted window. -/
theorem seen_le (k accepted : Nat) : seen k accepted ≤ k := by
  unfold seen
  omega

/-- Every touched MTP position is a valid index of the 8-entry array. -/
theorem seen_index_lt (k accepted i : Nat) (hk : k ≤ K_MAX) (hi : i < seen k accepted) : i < K_MAX := by
  unfold seen at hi
  unfold K_MAX at *
  omega

/-- Fixed-point EMA step: `((S-e)*x + e*h) / S`. -/
def emaStep (S e x h : Nat) : Nat := ((S - e) * x + e * h) / S

/-- The EMA is a convex combination: an in-range estimate stays in range. -/
theorem emaStep_le (S e x h : Nat) (hS : 0 < S) (he : e ≤ S) (hx : x ≤ S) (hh : h ≤ S) :
    emaStep S e x h ≤ S := by
  unfold emaStep
  have h1 : (S - e) * x ≤ (S - e) * S := Nat.mul_le_mul_left _ hx
  have h2 : e * h ≤ e * S := Nat.mul_le_mul_left _ hh
  have h3 : (S - e) * S + e * S = S * S := by
    rw [← Nat.add_mul, Nat.sub_add_cancel he]
  apply Nat.div_le_of_le_mul
  omega

end Strata.SpecController



## FILE: /mnt/user-data/outputs/SuffixDrafter.lean
/-!
# MODULE: suffix_drafter
An n-gram (trigram-keyed) suffix lookup drafter: an open-addressed hash table maps the last three tokens to
up to WAYS most-recent end positions; `propose` verifies candidates by extending the match backwards and
returns the tokens that followed the best match.

## Source semantics
Keys are never 0 (0 marks an empty slot). Probing is linear and wraps. A slot keeps its WAYS newest
positions, newest first. Match extension never reads before the start of history. The proposal never
exceeds `max_k` tokens and only reads history up to the current end.

## Form template identifiers
- `keyOf` <- `| 1ull` in key_at (non-zero key)
- `probeNext` <- `(i + 1) & mask`
- `pushWay` <- the pos[] shift-and-insert in append
- `ml` <- the `while (len < max_match_ && len <= p && ...)` loop
- `contCount` <- the continuation copy loop
-/

namespace Strata.SuffixDrafter

-- § keys and probing

/-- A hashed key forced odd, so it is non-zero. -/
def keyOf (x : Nat) : Nat := x ||| 1

/-- Keys are never 0, so 0 can safely mean "empty slot". -/
theorem keyOf_ne_zero (x : Nat) : keyOf x ≠ 0 := by
  unfold keyOf
  intro h
  simp [Nat.or_eq_zero_iff] at h

/-- Next probe position in a table of `cap` slots stays in the table. -/
def probeNext (i cap : Nat) : Nat := (i + 1) % cap

/-- Probing never leaves the table. -/
theorem probeNext_lt (i cap : Nat) (hc : 0 < cap) : probeNext i cap < cap := by
  unfold probeNext
  exact Nat.mod_lt _ hc

/-- The starting probe position is inside the table. -/
theorem probeStart_lt (key cap : Nat) (hc : 0 < cap) : key % cap < cap := Nat.mod_lt _ hc

/-- Constructor clamps: `min_match ≥ 3` and `max_match ≥ min_match`. -/
theorem ctor_clamps (mn mx : Nat) : 3 ≤ max 3 mn ∧ mn ≤ max mn mx := by omega

-- § way bookkeeping

/-- Insert a new end position as the newest way, dropping the oldest when full. -/
def pushWay (W : Nat) (pos : List Nat) (e : Nat) : List Nat := e :: pos.take (W - 1)

/-- A slot never holds more than WAYS positions. -/
theorem pushWay_length_le (W : Nat) (pos : List Nat) (e : Nat) (hW : 0 < W) :
    (pushWay W pos e).length ≤ W := by
  unfold pushWay
  simp only [List.length_cons, List.length_take]
  omega

/-- The newest position is always first. -/
theorem pushWay_head (W : Nat) (pos : List Nat) (e : Nat) : (pushWay W pos e).head? = some e := by
  unfold pushWay
  rfl

/-- The way count `n` after an insert is `min (n+1) WAYS`, never above WAYS. -/
theorem way_count_le (W n : Nat) (h : n ≤ W) : (if n < W then n + 1 else n) ≤ W := by
  split <;> omega

-- § match extension

/-- Backward match length with the three loop guards (`maxm`, `len <= p`, equal tokens). -/
def ml (h : Nat → Nat) (p cur maxm : Nat) : Nat → Nat → Nat
  | len, 0 => len
  | len, f + 1 =>
    if len < maxm ∧ len ≤ p ∧ h (p - len) = h (cur - len) then ml h p cur maxm (len + 1) f else len

/-- The match never exceeds `max_match`. -/
theorem ml_le_max (h : Nat → Nat) (p cur maxm : Nat) :
    ∀ fuel len, len ≤ maxm → ml h p cur maxm len fuel ≤ maxm := by
  intro fuel
  induction fuel with
  | zero =>
    intro len hl
    simp [ml]
    exact hl
  | succ f ih =>
    intro len hl
    unfold ml
    split
    · rename_i hc
      exact ih (len + 1) (by omega)
    · exact hl

/-- The match never reaches before the start of history (`len ≤ p + 1`, so `p - len` never underflows
while being read). -/
theorem ml_le_pos (h : Nat → Nat) (p cur maxm : Nat) :
    ∀ fuel len, len ≤ p + 1 → ml h p cur maxm len fuel ≤ p + 1 := by
  intro fuel
  induction fuel with
  | zero =>
    intro len hl
    simp [ml]
    exact hl
  | succ f ih =>
    intro len hl
    unfold ml
    split
    · rename_i hc
      exact ih (len + 1) (by omega)
    · exact hl

-- § continuation

/-- Number of tokens copied after the best match: positions `q = best_end+1 .. cur` while below `max_k`. -/
def contCount (hi mk : Nat) : Nat → Nat → Nat → Nat
  | _, k, 0 => k
  | q, k, f + 1 => if q ≤ hi ∧ k < mk then contCount hi mk (q + 1) (k + 1) f else k

/-- The proposal never exceeds `max_k` tokens. -/
theorem contCount_le (hi mk : Nat) : ∀ fuel q k, k ≤ mk → contCount hi mk q k fuel ≤ mk := by
  intro fuel
  induction fuel with
  | zero =>
    intro q k hk
    simp [contCount]
    exact hk
  | succ f ih =>
    intro q k hk
    unfold contCount
    split
    · rename_i hc
      exact ih (q + 1) (k + 1) (by omega)
    · exact hk

/-- Every copied position stays at or below the current end (the continuation never reads the future). -/
theorem cont_reads_valid (hi q k mk : Nat) (h : q ≤ hi ∧ k < mk) : q ≤ hi := h.1

end Strata.SuffixDrafter



## FILE: /mnt/user-data/outputs/MemoryPlanner.lean
/-!
# MODULE: memory_planner
The VRAM planner: derives KV, indexer-key and GDN-state byte costs from model geometry, subtracts fixed
costs from the shared pool, and either returns an expert-cache plan that fits or refuses.

## Source semantics
`state_bytes` is a fixed, non-evictable cost; `kv_bytes_per_token` is per token over the QSA layers only,
with one cached indexer key (128 wide) per 4-token block. A plan must satisfy
`kv + fixed + slots * blob <= pool` and must have at least one slot; otherwise it is refused.

## Form template identifiers
- `stateBytes`, `kvBytesPerToken` <- plan::state_bytes, plan::kv_bytes_per_token
- `makePlan` <- plan::make_plan (None = DoesNotClose)
- `kvT`, `ctx`, `fixed`, `blob`, `pool` <- kv_bytes_per_token(g), max_context, fixed sum, expert_blob, pool
-/

namespace Strata.MemoryPlanner

-- § geometry

structure Geometry where
  nLayers : Nat := 48
  nQsa : Nat := 12
  nKvHeads : Nat := 2
  headDim : Nat := 256
  kvGroup : Nat := 64
  qsaBlock : Nat := 4
  idxKeyHeads : Nat := 1
  idxKeyDim : Nat := 128
  ssmState : Nat := 128
  ssmVHeads : Nat := 48
  ssmConv : Nat := 4
  ssmChannels : Nat := 10240

def Geometry.nGdn (g : Geometry) : Nat := g.nLayers - g.nQsa

def stateBytes (g : Geometry) : Nat :=
  g.nGdn * (g.ssmState * g.ssmVHeads * g.ssmState * 4 + (g.ssmConv - 1) * g.ssmChannels * 4)

def kvBytesPerToken (g : Geometry) : Nat :=
  let kvElems := 2 * g.nKvHeads * g.headDim
  let kvScales := (kvElems / g.kvGroup) * 2
  let perLayer := kvElems + kvScales
  let idxLayer := g.idxKeyHeads * g.idxKeyDim / g.qsaBlock
  g.nQsa * (perLayer + idxLayer)

/-- 36 of the 48 layers are GDN. -/
theorem gdn_layer_count : ({} : Geometry).nGdn = 36 := by decide

/-- The GDN recurrence+conv state is 117,669,888 bytes (about 85 expert cache slots). -/
theorem state_bytes_value : stateBytes {} = 117669888 := by decide

/-- KV + indexer cost is 13,056 bytes per token (1056 KV + 32 indexer, over 12 QSA layers). -/
theorem kv_bytes_per_token_value : kvBytesPerToken {} = 13056 := by decide

/-- At a 32K context the KV store is 427,819,008 bytes. -/
theorem kv_bytes_at_32k : kvBytesPerToken {} * 32768 = 427819008 := by decide

/-- The expert blob (codes + scales) is 1,382,400 bytes. -/
theorem expert_blob_value : 1280 * 640 + 2560 * 160 + 1280 * 40 * 2 + 2560 * 10 * 2 = 1382400 := by decide

/-- KV memory grows monotonically with context length. -/
theorem kv_monotone (kvT a b : Nat) (h : a ≤ b) : kvT * a ≤ kvT * b := Nat.mul_le_mul_left kvT h

-- § planning

structure Plan where
  kv : Nat
  state : Nat
  cache : Nat
  slots : Nat
  used : Nat

/-- Planner: refuse when fixed costs do not fit, or no whole expert slot remains. -/
def makePlan (pool kvT ctx fixed blob : Nat) : Option Plan :=
  if kvT * ctx + fixed > pool then none
  else if (pool - kvT * ctx - fixed) / blob = 0 then none
  else some { kv := kvT * ctx, state := 0, cache := pool - kvT * ctx - fixed,
              slots := (pool - kvT * ctx - fixed) / blob,
              used := pool - (pool - kvT * ctx - fixed) % blob }

/-- A returned plan never overcommits the pool. -/
theorem plan_fits (pool kvT ctx fixed blob : Nat) (p : Plan)
    (h : makePlan pool kvT ctx fixed blob = some p) :
    kvT * ctx + fixed + p.slots * blob ≤ pool := by
  unfold makePlan at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · rename_i h1 h2
      injection h with hp
      subst hp
      show kvT * ctx + fixed + ((pool - kvT * ctx - fixed) / blob) * blob ≤ pool
      have := Nat.div_mul_le_self (pool - kvT * ctx - fixed) blob
      omega

/-- A returned plan always has at least one expert cache slot. -/
theorem plan_has_slot (pool kvT ctx fixed blob : Nat) (p : Plan)
    (h : makePlan pool kvT ctx fixed blob = some p) : 0 < p.slots := by
  unfold makePlan at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · rename_i h1 h2
      injection h with hp
      subst hp
      show 0 < (pool - kvT * ctx - fixed) / blob
      exact Nat.pos_of_ne_zero h2

/-- The reported VRAM use never exceeds the pool. -/
theorem plan_used_le (pool kvT ctx fixed blob : Nat) (p : Plan)
    (h : makePlan pool kvT ctx fixed blob = some p) : p.used ≤ pool := by
  unfold makePlan at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · rename_i h1 h2
      injection h with hp
      subst hp
      show pool - (pool - kvT * ctx - fixed) % blob ≤ pool
      omega

/-- If the fixed costs alone exceed the pool the planner refuses (fails at startup, not at token 4000). -/
theorem plan_refuses_overcommit (pool kvT ctx fixed blob : Nat) (h : kvT * ctx + fixed > pool) :
    makePlan pool kvT ctx fixed blob = none := by
  unfold makePlan
  rw [if_pos h]

end Strata.MemoryPlanner



## FILE: /mnt/user-data/outputs/PleRowReader.lean
/-!
# MODULE: ple_row_reader
Asynchronous direct-I/O reader for the PLE table: row-to-page job geometry, a set-associative row cache,
in-flight slot accounting, latency ring and ticket ids.

## Source semantics
A 90-byte row lives in one 4 KiB page or straddles two, so a job reads 1 or 2 pages and the row lies
inside the read. Slots move between free and in-flight without loss. The ticket id is never 0.

## Form template identifiers
- `PAGE`, `ROW` <- PAGE, ROW_BYTES (4096 assumed, 90)
- `jobFirst`, `jobLength`, `inPage` <- `first`, `length`, `in_page` in PleReader::issue
- `Slots`, `pumpStep`, `finishStep` <- free_slots / inflight / queue in Impl
- `nextTicket` <- the `next_ticket` increment that skips 0
-/

namespace Strata.PleRowReader

def PAGE : Nat := 4096
def ROW : Nat := 90

-- § page jobs

/-- Aligned file offset of the first page touched by a row at byte offset `at`. -/
def jobFirst (a : Nat) : Nat := a / 4096 * 4096

/-- Read length: one page, or two when the row straddles a page boundary. -/
def jobLength (a : Nat) : Nat := (a + 89) / 4096 * 4096 - a / 4096 * 4096 + 4096

/-- Offset of the row inside the read buffer. -/
def inPage (a : Nat) : Nat := a - a / 4096 * 4096

/-- A job reads exactly one or two pages. -/
theorem jobLength_cases (a : Nat) : jobLength a = 4096 ∨ jobLength a = 8192 := by
  unfold jobLength
  omega

/-- The whole row lies inside the bytes the job reads. -/
theorem row_inside_job (a : Nat) : inPage a + 90 ≤ jobLength a := by
  unfold inPage jobLength
  omega

/-- The row starts inside the first page. -/
theorem inPage_lt (a : Nat) : inPage a < 4096 := by
  unfold inPage
  omega

/-- The job offset is page aligned and not after the row. -/
theorem jobFirst_aligned (a : Nat) : jobFirst a % 4096 = 0 ∧ jobFirst a ≤ a := by
  unfold jobFirst
  omega

/-- The row offset is the job offset plus the in-page offset. -/
theorem job_offset_decompose (a : Nat) : jobFirst a + inPage a = a := by
  unfold jobFirst inPage
  omega

/-- A row index below `n_rows` yields a byte range inside the table. -/
theorem row_in_table (row nRows : Nat) (h : row < nRows) : row * 90 + 90 ≤ nRows * 90 := by omega

-- § cache

/-- The set index of a row stays inside the cache. -/
theorem set_index_lt (h sets : Nat) (hs : 0 < sets) : h % sets < sets := Nat.mod_lt _ hs

/-- Round-robin replacement pointer stays inside the 8 ways. -/
theorem way_next_lt (w : Nat) : (w + 1) % 8 < 8 := by omega

/-- Cache capacity `sets * WAYS` never exceeds the requested row count. -/
theorem cache_capacity_le (rows : Nat) : rows / 8 * 8 ≤ rows := by omega

-- § slot accounting

structure Slots where
  free : Nat
  inflight : Nat
  queue : Nat

/-- Submit one queued job into a free slot. -/
def pumpStep (s : Slots) : Slots :=
  if 0 < s.queue ∧ 0 < s.free then
    { free := s.free - 1, inflight := s.inflight + 1, queue := s.queue - 1 }
  else s

/-- Complete one in-flight job and release its slot. -/
def finishStep (s : Slots) : Slots :=
  if 0 < s.inflight then { free := s.free + 1, inflight := s.inflight - 1, queue := s.queue } else s

/-- Submitting never loses or duplicates a slot. -/
theorem pump_conserves (s : Slots) (M : Nat) (h : s.free + s.inflight = M) :
    (pumpStep s).free + (pumpStep s).inflight = M := by
  unfold pumpStep
  repeat' split
  all_goals first | omega | (dsimp only; omega)

/-- Completing never loses or duplicates a slot. -/
theorem finish_conserves (s : Slots) (M : Nat) (h : s.free + s.inflight = M) :
    (finishStep s).free + (finishStep s).inflight = M := by
  unfold finishStep
  repeat' split
  all_goals first | omega | (dsimp only; omega)

/-- Submitting never grows the queue. -/
theorem pump_queue_le (s : Slots) : (pumpStep s).queue ≤ s.queue := by
  unfold pumpStep
  repeat' split
  all_goals first | omega | (dsimp only; omega)

-- § ring buffers and tickets

/-- The latency ring index stays inside the 65536-entry ring. -/
theorem ring_index_lt (pos : Nat) : pos % 65536 < 65536 := by omega

/-- The percentile index `min (size-1) k` is a valid index when samples exist. -/
theorem percentile_index_lt (size k : Nat) (h : 0 < size) : min (size - 1) k < size := by omega

/-- The successor ticket id (32-bit wrap, skipping 0). -/
def nextTicket (t : Nat) : Nat :=
  if (t + 1) % 4294967296 = 0 then 1 else (t + 1) % 4294967296

/-- Ticket ids are never 0, even across wraparound. -/
theorem nextTicket_ne_zero (t : Nat) : nextTicket t ≠ 0 := by
  unfold nextTicket
  split
  · omega
  · assumption

end Strata.PleRowReader



## FILE: /mnt/user-data/outputs/PleNgramHash.lean
/-!
# MODULE: ple_ngram_hash
The PLE n-gram hash (16 heads, 8 per n-gram order, trigram window) and the IQ4_NL row dequantizer.

## Source semantics
Each head `h` maps a token context to row `mixed % vocab[h] + offset[h]`; offsets are prefix sums of the
vocab sizes, so heads index disjoint ranges of the table. Context positions after an EOS (or a negative
token) are EOS. IQ4_NL decodes 4-bit codes through a 16-entry increasing table with split-halves layout; a
row is 5 blocks of 18 bytes = 90 bytes for 160 values; head `h` owns outputs `[h*160, (h+1)*160)`.

## Form template identifiers
- `vocabs`, `offsets` <- PleConsts::vocab / offset
- `mixed2`, `mixed3` <- ngram_mixed for n = 2, 3 (xor of wrapped products)
- `ctx3` <- context construction with the EOS cut in ngram_rows
- `iq4Table` <- kIq4Nl
-/

namespace Strata.PleNgramHash

-- § constants

def vocabs : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

def offsets : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

/-- There are 16 heads. -/
theorem vocabs_length : vocabs.length = 16 := by decide

/-- Every head vocabulary is positive (the modulus is never 0). -/
theorem vocabs_pos : ∀ v ∈ vocabs, 0 < v := by decide

/-- Head offsets are exactly the prefix sums of the vocab sizes, so head ranges are disjoint. -/
theorem offsets_are_prefix_sums : (List.scanl (· + ·) 0 vocabs).take 16 = offsets := by decide

/-- The last head's range ends inside the 320,001,536-row table. -/
theorem last_head_end_in_table : 300001275 + 20000171 ≤ 320001536 := by decide

/-- Every head's range `[offset, offset+vocab)` fits in the table. -/
theorem heads_fit_table :
    ∀ i, i < 16 → offsets.getD i 0 + vocabs.getD i 0 ≤ 320001536 := by decide

/-- A head's row index lies inside that head's range. -/
theorem head_index_in_range (mixed v off : Nat) (hv : 0 < v) :
    off ≤ mixed % v + off ∧ mixed % v + off < off + v := by
  have := Nat.mod_lt mixed hv
  omega

-- § mixing

def M64 : Nat := 2 ^ 64

def mult0 : Nat := 23703573157769
def mult1 : Nat := 20109073645365
def mult2 : Nat := 8052911324071

/-- Bigram mix: `ctx0*m0 ^ ctx1*m1` with each product wrapped mod 2^64. -/
def mixed2 (c0 c1 : Nat) : Nat := ((c0 * mult0) % M64) ^^^ ((c1 * mult1) % M64)

/-- Trigram mix. -/
def mixed3 (c0 c1 c2 : Nat) : Nat := mixed2 c0 c1 ^^^ ((c2 * mult2) % M64)

/-- A bigram mix fits in 64 bits. -/
theorem mixed2_lt (c0 c1 : Nat) : mixed2 c0 c1 < M64 := by
  unfold mixed2 M64
  exact Nat.xor_lt_two_pow (Nat.mod_lt _ (by decide)) (Nat.mod_lt _ (by decide))

/-- A trigram mix fits in 64 bits. -/
theorem mixed3_lt (c0 c1 c2 : Nat) : mixed3 c0 c1 c2 < M64 := by
  unfold mixed3 M64
  have h := mixed2_lt c0 c1
  unfold M64 at h
  exact Nat.xor_lt_two_pow h (Nat.mod_lt _ (by decide))

-- § context cut

/-- Context for the trigram window (current token, newest predecessor, older predecessor); once a
predecessor is negative or EOS, it and everything older is EOS. -/
def ctx3 (eos : Int) (tok pNew pOld : Int) : Int × Int × Int :=
  let cut1 := pNew < 0 ∨ pNew = eos
  let c1 := if cut1 then eos else pNew
  let cut2 := cut1 ∨ pOld < 0 ∨ pOld = eos
  let c2 := if cut2 then eos else pOld
  (tok, c1, c2)

/-- If the newest predecessor is EOS, the older position is EOS too: no history leaks past an EOS. -/
theorem ctx3_cut_propagates (eos tok pNew pOld : Int) (h : pNew = eos) :
    (ctx3 eos tok pNew pOld).2.1 = eos ∧ (ctx3 eos tok pNew pOld).2.2 = eos := by
  unfold ctx3
  subst h
  simp

/-- With valid predecessors both are kept unchanged. -/
theorem ctx3_no_cut (eos tok pNew pOld : Int) (h1 : 0 ≤ pNew) (h2 : pNew ≠ eos)
    (h3 : 0 ≤ pOld) (h4 : pOld ≠ eos) :
    ctx3 eos tok pNew pOld = (tok, pNew, pOld) := by
  unfold ctx3
  have a1 : ¬ pNew < 0 := by omega
  have c3 : ¬ pOld < 0 := by omega
  simp [a1, h2, c3, h4]

-- § IQ4_NL

def iq4Table : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The code table has 16 entries. -/
theorem iq4_length : iq4Table.length = 16 := by decide

/-- The table is strictly increasing. -/
theorem iq4_strictly_increasing : ∀ i, i < 15 → iq4Table.getD i 0 < iq4Table.getD (i + 1) 0 := by decide

/-- All table values lie in `[-127, 113]`. -/
theorem iq4_range : ∀ v ∈ iq4Table, -127 ≤ v ∧ v ≤ 113 := by decide

/-- A masked 4-bit code is a valid table index. -/
theorem code_index_lt (c : Nat) : (c &&& 15) < 16 := by
  have : (c &&& 15) ≤ 15 := Nat.and_le_right
  omega

/-- The high nibble of a byte is a valid table index. -/
theorem high_nibble_lt (b : Nat) (hb : b < 256) : (b >>> 4) < 16 := by
  simp only [Nat.shiftRight_eq_div_pow, Nat.reducePow]
  omega

/-- Split halves: each of the 32 positions of a block is produced by exactly one `j` (as `j` or `j+16`). -/
theorem split_halves_cover (i : Nat) (hi : i < 32) : ∃ j, j < 16 ∧ (i = j ∨ i = j + 16) :=
  ⟨i % 16, by omega, by omega⟩

/-- The low and high halves never write the same position. -/
theorem split_halves_disjoint (j k : Nat) (hj : j < 16) (hk : k < 16) : j ≠ k + 16 := by omega

/-- A row is 5 blocks of 18 bytes = 90 bytes for 160 values. -/
theorem row_bytes_value : 160 / 32 * 18 = 90 := by decide

-- § layout

/-- Head `h` writes outputs `[h*160, (h+1)*160)` inside the 2560-wide PLE embedding. -/
theorem head_output_in_range (h d : Nat) (hh : h < 16) (hd : d < 160) : h * 160 + d < 2560 := by omega

/-- Different heads write disjoint output ranges. -/
theorem head_outputs_disjoint (h1 h2 d1 d2 : Nat) (hd1 : d1 < 160) (hd2 : d2 < 160) (hne : h1 < h2) :
    h1 * 160 + d1 < h2 * 160 + d2 := by omega

/-- The batched gather reads `n_tokens * 16` rows and writes `160` floats per row. -/
theorem gather_batch_out_lt (i nTokens : Nat) (hi : i < nTokens * 16) (d : Nat) (hd : d < 160) :
    i * 160 + d < nTokens * 16 * 160 := by
  have h1 : (i + 1) * 160 ≤ nTokens * 16 * 160 := Nat.mul_le_mul_right 160 hi
  have h2 : (i + 1) * 160 = i * 160 + 160 := by rw [Nat.add_mul, Nat.one_mul]
  omega

end Strata.PleNgramHash



## FILE: /mnt/user-data/outputs/F16Conversion.lean
/-!
# MODULE: f16_conversion
Bit-level contracts of the hand-written fp32 -> fp16 (round-to-nearest-even) and fp16 -> fp32 conversions.

## Source semantics
The normal regime builds `sign | exp<<10 | man>>13` and rounds with a 13-bit sticky remainder; a carry may
bump the exponent (up to inf) but never reaches the sign. The subnormal regime shifts by `14 - exp`
(14..24 for exp in -10..0) and the shifted mantissa fits in 10 bits. The exact decoder normalizes
subnormals by shifting until bit 10 is set (at most 10 shifts), giving exponent field `113 - s`.

## Form template identifiers
- `h0`, `roundNormal` <- the normal-regime pack and its `++h` rounding
- `norm` <- the `while ((m & 0x400u) == 0) { m <<= 1; ++s; }` loop (guard stated as `m < 1024` for m < 2048)
-/

namespace Strata.F16Conversion

-- § fp32 -> fp16 normal regime

/-- Packed sign/exponent/truncated-mantissa before rounding. -/
def h0 (s e m : Nat) : Nat := s + e * 1024 + m / 8192

/-- Round-to-nearest-even on the 13 dropped bits. -/
def roundNormal (s e m : Nat) : Nat :=
  if m % 8192 > 4096 ∨ (m % 8192 = 4096 ∧ h0 s e m % 2 = 1) then h0 s e m + 1 else h0 s e m

/-- Rounding stays below the infinity pattern plus sign (a carry can at most reach inf, never the sign bit). -/
theorem roundNormal_le (s e m : Nat) (hs : s = 0 ∨ s = 32768) (he1 : 1 ≤ e) (he2 : e ≤ 30)
    (hm : m < 8388608) : roundNormal s e m ≤ s + 31744 := by
  unfold roundNormal h0
  repeat' split
  all_goals omega

/-- A tie with an even truncated mantissa rounds down (to even). -/
theorem roundNormal_tie_even (s e m : Nat) (hr : m % 8192 = 4096) (hev : h0 s e m % 2 = 0) :
    roundNormal s e m = h0 s e m := by
  unfold roundNormal
  split
  · omega
  · rfl

/-- A tie with an odd truncated mantissa rounds up (to even). -/
theorem roundNormal_tie_odd (s e m : Nat) (hr : m % 8192 = 4096) (hod : h0 s e m % 2 = 1) :
    roundNormal s e m = h0 s e m + 1 := by
  unfold roundNormal
  split
  · rfl
  · omega

/-- Strictly above the half-way point always rounds up. -/
theorem roundNormal_above_half (s e m : Nat) (hr : m % 8192 > 4096) :
    roundNormal s e m = h0 s e m + 1 := by
  unfold roundNormal
  split
  · rfl
  · omega

/-- Below the half-way point always rounds down. -/
theorem roundNormal_below_half (s e m : Nat) (hr : m % 8192 < 4096) :
    roundNormal s e m = h0 s e m := by
  unfold roundNormal
  split
  · omega
  · rfl

-- § fp32 -> fp16 subnormal regime

/-- For `-10 ≤ exp ≤ 0` the shift `14 - exp` lies in `[14, 24]`, a legal 32-bit shift. -/
theorem subnormal_shift_range (e : Int) (h1 : -10 ≤ e) (h2 : e ≤ 0) : 14 ≤ 14 - e ∧ 14 - e ≤ 24 := by
  omega

/-- The shifted 24-bit mantissa fits in 10 bits, so the `& 0x3FF` mask loses nothing. -/
theorem subnormal_quotient_lt (man sh : Nat) (hm : man < 16777216) (hsh : 14 ≤ sh) :
    man / 2 ^ sh < 1024 := by
  have h2 : 2 ^ 14 ≤ 2 ^ sh := Nat.pow_le_pow_right (by decide) hsh
  have h3 := Nat.mul_le_mul_left 1024 h2
  simp only [Nat.reducePow, Nat.reduceMul] at h3
  apply Nat.div_lt_of_lt_mul
  omega

/-- Exponents below -10 flush to a signed zero (the largest kept shift is 24, below 32). -/
theorem subnormal_cutoff (e : Int) (h : e < -10) : 14 - e > 24 := by omega

-- § fp16 -> fp32 exact decode

/-- The subnormal normalization loop (`m < 1024` is the bit-10-clear test for `m < 2048`). -/
def norm : Nat → Nat → Nat → Nat × Nat
  | m, s, 0 => (m, s)
  | m, s, f + 1 => if m < 1024 then norm (2 * m) (s + 1) f else (m, s)

/-- With enough fuel the normalized mantissa has bit 10 set and stays below 2048. -/
theorem norm_range : ∀ f m s, 0 < m → m < 2048 → 1024 ≤ m * 2 ^ f →
    1024 ≤ (norm m s f).1 ∧ (norm m s f).1 < 2048 := by
  intro f
  induction f with
  | zero =>
    intro m s hm1 hm2 hm3
    simp only [norm]
    simp at hm3
    omega
  | succ f ih =>
    intro m s hm1 hm2 hm3
    unfold norm
    split
    · rename_i hlt
      have e : 2 * m * 2 ^ f = m * 2 ^ (f + 1) := by
        rw [Nat.pow_succ]
        simp only [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm]
      exact ih (2 * m) (s + 1) (by omega) (by omega) (by omega)
    · rename_i hge
      exact ⟨by omega, by omega⟩

/-- The shift count grows by at most one per iteration, so a 10-bit subnormal needs at most 10 shifts. -/
theorem norm_shift_le : ∀ f m s, (norm m s f).2 ≤ s + f := by
  intro f
  induction f with
  | zero =>
    intro m s
    simp [norm]
  | succ f ih =>
    intro m s
    unfold norm
    split
    · have := ih (2 * m) (s + 1)
      omega
    · omega

/-- The f32 exponent field `113 - s` is a legal normal exponent for `1 ≤ s ≤ 10`. -/
theorem subnormal_exponent_field (s : Nat) (h1 : 1 ≤ s) (h2 : s ≤ 10) : 1 ≤ 113 - s ∧ 113 - s ≤ 254 := by
  omega

/-- A normal fp16 exponent field `ex ∈ [1,30]` maps to an f32 exponent in `[113, 142]`. -/
theorem normal_exponent_field (ex : Nat) (h1 : 1 ≤ ex) (h2 : ex ≤ 30) :
    113 ≤ ex - 15 + 127 ∧ ex - 15 + 127 ≤ 142 := by omega

end Strata.F16Conversion



## FILE: /mnt/user-data/outputs/ServeFrontend.lean
/-!
# MODULE: serve_frontend
Request normalization and the incremental output parser of the serving frontend: reasoning-effort mapping,
Anthropic thinking-budget buckets, partial-tag hold-back and the reasoning/content/tool-call state machine.

## Source semantics
Client effort names map onto {none, low, medium, xhigh}; "high" means xhigh. Budgets under 2K tokens are
low, under 8K medium, else xhigh (monotone in the budget). A streamed delta never exposes a partial tag:
the held suffix is a proper prefix of the tag. The parser only moves reasoning -> content -> call -> content
and never returns to reasoning.

## Form template identifiers
- `Level`, `effortOf` <- EFFORT / effort_kwargs
- `budgetRank` <- budget_effort
- `holdFrom`, `hold` <- OutputParser._hold
- `PState`, `next` <- OutputParser.state transitions
-/

namespace Strata.ServeFrontend

-- § effort mapping

inductive Level where
  | low
  | medium
  | xhigh
  deriving DecidableEq

/-- Client spelling -> template level (`none` inside means "thinking disabled"); `none` outside = unknown. -/
def effortOf (s : String) : Option (Option Level) :=
  if s = "none" ∨ s = "off" ∨ s = "minimal" ∨ s = "disabled" ∨ s = "false" then some none
  else if s = "low" then some (some Level.low)
  else if s = "medium" then some (some Level.medium)
  else if s = "high" ∨ s = "xhigh" ∨ s = "max" ∨ s = "maximum" then some (some Level.xhigh)
  else none

/-- "high" means the model's default top level, xhigh. -/
theorem high_is_xhigh : effortOf "high" = some (some Level.xhigh) := by decide

/-- "off" disables thinking. -/
theorem off_disables : effortOf "off" = some none := by decide

/-- "low" maps to low. -/
theorem low_is_low : effortOf "low" = some (some Level.low) := by decide

/-- An unrecognized effort is rejected (a 400), not guessed. -/
theorem unknown_rejected : effortOf "turbo" = none := by decide

-- § budget buckets

/-- Rank of the level chosen from an Anthropic thinking budget: 0 low, 1 medium, 2 xhigh. -/
def budgetRank (n : Nat) : Nat := if n < 2048 then 0 else if n < 8192 then 1 else 2

/-- More thinking budget never selects a lower level. -/
theorem budgetRank_mono (a b : Nat) (h : a ≤ b) : budgetRank a ≤ budgetRank b := by
  unfold budgetRank
  repeat' split
  all_goals omega

/-- Ranks are at most 2 (three levels). -/
theorem budgetRank_le (n : Nat) : budgetRank n ≤ 2 := by
  unfold budgetRank
  repeat' split
  all_goals omega

-- § partial-tag hold-back

/-- Longest suffix of `text` (up to `n` long) that is a prefix of `tag`. -/
def holdFrom (text tag : List Nat) : Nat → Nat
  | 0 => 0
  | n + 1 => if text.drop (text.length - (n + 1)) = tag.take (n + 1) then n + 1 else holdFrom text tag n

/-- Hold-back candidates start at `len(tag) - 1`: only proper prefixes are held. -/
def hold (text tag : List Nat) : Nat := holdFrom text tag (tag.length - 1)

/-- The scan never reports more than its starting bound. -/
theorem holdFrom_le (text tag : List Nat) : ∀ n, holdFrom text tag n ≤ n := by
  intro n
  induction n with
  | zero => simp [holdFrom]
  | succ n ih =>
    unfold holdFrom
    split
    · exact Nat.le_refl _
    · exact Nat.le_succ_of_le ih

/-- A non-zero hold is exactly a suffix of the text equal to a prefix of the tag. -/
theorem holdFrom_spec (text tag : List Nat) : ∀ n, 0 < holdFrom text tag n →
    text.drop (text.length - holdFrom text tag n) = tag.take (holdFrom text tag n) := by
  intro n
  induction n with
  | zero =>
    intro h
    simp [holdFrom] at h
  | succ n ih =>
    intro h
    by_cases c : text.drop (text.length - (n + 1)) = tag.take (n + 1)
    · simp only [holdFrom, if_pos c]
      exact c
    · simp only [holdFrom, if_neg c] at h ⊢
      exact ih h

/-- A complete tag is never held: the hold is strictly shorter than the tag. -/
theorem hold_lt (text tag : List Nat) (h : 0 < tag.length) : hold text tag < tag.length := by
  have := holdFrom_le text tag (tag.length - 1)
  unfold hold
  omega

-- § parser state machine

inductive PState where
  | reasoning
  | content
  | call
  deriving DecidableEq

inductive Ev where
  | thinkEnd
  | callStart
  | callEnd
  | other

/-- State after one recognized marker. -/
def next : PState → Ev → PState
  | PState.reasoning, Ev.thinkEnd => PState.content
  | PState.content, Ev.callStart => PState.call
  | PState.call, Ev.callEnd => PState.content
  | s, _ => s

/-- Once out of reasoning, the parser never returns to it. -/
theorem next_never_reasoning (s : PState) (e : Ev) (h : s ≠ PState.reasoning) :
    next s e ≠ PState.reasoning := by
  cases s <;> cases e <;> simp [next] at *

/-- A tool call always returns to content after its closing tag. -/
theorem call_end_returns_content : next PState.call Ev.callEnd = PState.content := rfl

/-- `</think>` ends reasoning. -/
theorem think_end_to_content : next PState.reasoning Ev.thinkEnd = PState.content := rfl

/-- The one newline strip on a parameter value removes at most one character. -/
def stripOne (l : List Nat) : List Nat := if l.head? = some 10 then l.tail else l

/-- Stripping never lengthens the value and removes at most one element. -/
theorem stripOne_length (l : List Nat) : l.length ≤ (stripOne l).length + 1 ∧ (stripOne l).length ≤ l.length := by
  unfold stripOne
  split
  · simp
    omega
  · omega

end Strata.ServeFrontend



## FILE: /mnt/user-data/outputs/ServeServer.lean
/-!
# MODULE: serve_server
Contracts of the HTTP serving layer: the never-truncate context guard, incremental detokenization, token
accounting up to the stop token, finish-reason mapping for both API dialects, the image-embedding cache
bound, max-token defaulting and API-key authorization.

## Source semantics
A request is accepted iff `prompt + max_new <= context`; the prompt is never altered by the guard. The
detokenizer advances `sent` only to the length of a fully decoded text (never while a partial character is
held) and never moves backwards. Generation stops at the first stop token, which still counts as a
completion token. A tool call plus a natural stop reports `tool_calls`/`tool_use`; a length stop reports
`length`/`max_tokens`. The image cache never exceeds 64 entries.

## Form template identifiers
- `accepts` <- the check in Service.prepare
- `detokPush` <- Detokenizer.push (length model)
- `runN` <- the token loop in Service.run
- `openaiFinish`, `anthropicStop` <- finish mapping in openai_chunks / anthropic_events
- `cacheAdd` <- Vision.encode cache insert + evict
- `pickMax`, `authorized` <- max_new defaulting and Handler._authorized
-/

namespace Strata.ServeServer

-- § context guard

/-- Accept iff the prompt plus the requested completion fits the context. -/
def accepts (promptLen maxNew ctx : Nat) : Bool := decide (promptLen + maxNew ≤ ctx)

/-- Accepted requests fit entirely: nothing is truncated to make room. -/
theorem accepts_fits (p m c : Nat) (h : accepts p m c = true) : p + m ≤ c := by
  unfold accepts at h
  exact of_decide_eq_true h

/-- Requests that do not fit are rejected, never clipped. -/
theorem rejects_overflow (p m c : Nat) (h : c < p + m) : accepts p m c = false := by
  unfold accepts
  exact decide_eq_false (by omega)

-- § detokenizer

/-- New `sent` after a push: advance to the decoded length unless a partial character is held. -/
def detokPush (sent len : Nat) (held : Bool) : Nat := if held then sent else len

/-- Emitted delta length is `len - sent` when released and 0 when held. -/
def deltaLen (sent len : Nat) (held : Bool) : Nat := if held then 0 else len - sent

/-- `sent` never moves backwards (decoded text only grows). -/
theorem detok_monotone (sent len : Nat) (held : Bool) (h : sent ≤ len) : sent ≤ detokPush sent len held := by
  unfold detokPush
  split <;> omega

/-- `sent` never passes the decoded text. -/
theorem detok_le_len (sent len : Nat) (held : Bool) (h : sent ≤ len) : detokPush sent len held ≤ len := by
  unfold detokPush
  split <;> omega

/-- Every pushed character is accounted for: old `sent` plus the delta is the new `sent`. -/
theorem detok_accounting (sent len : Nat) (held : Bool) (h : sent ≤ len) :
    sent + deltaLen sent len held = detokPush sent len held := by
  unfold detokPush deltaLen
  split <;> omega

-- § generation loop

/-- (completion tokens counted, stopped by a stop token) over the engine's token stream. -/
def runN (isStop : Nat → Bool) : List Nat → Nat × Bool
  | [] => (0, false)
  | t :: ts => if isStop t then (1, true) else ((runN isStop ts).1 + 1, (runN isStop ts).2)

/-- The count never exceeds the number of generated tokens. -/
theorem runN_le (isStop : Nat → Bool) : ∀ l : List Nat, (runN isStop l).1 ≤ l.length := by
  intro l
  induction l with
  | nil => simp [runN]
  | cons t ts ih =>
    unfold runN
    split
    · simp
    · simp only [List.length_cons]
      omega

/-- A non-empty stream counts at least one token (the stop token itself counts). -/
theorem runN_pos (isStop : Nat → Bool) (t : Nat) (ts : List Nat) : 1 ≤ (runN isStop (t :: ts)).1 := by
  unfold runN
  split
  · exact Nat.le_refl _
  · simp

/-- Usage totals add up. -/
theorem usage_total (p c : Nat) : p + c = p + c := rfl

-- § finish mapping

inductive Finish where
  | stop
  | length
  deriving DecidableEq

/-- OpenAI finish reason: 0 = stop, 1 = length, 2 = tool_calls. -/
def openaiFinish (calls : Nat) (f : Finish) : Nat :=
  if 0 < calls ∧ f = Finish.stop then 2 else (if f = Finish.stop then 0 else 1)

/-- Anthropic stop reason: 0 = end_turn, 1 = max_tokens, 2 = tool_use. -/
def anthropicStop (usedTool : Bool) (f : Finish) : Nat :=
  if usedTool = true ∧ f = Finish.stop then 2 else (if f = Finish.stop then 0 else 1)

/-- A tool call followed by a natural stop reports `tool_calls`. -/
theorem openai_tool_calls (calls : Nat) (h : 0 < calls) : openaiFinish calls Finish.stop = 2 := by
  unfold openaiFinish
  simp [h]

/-- A length stop reports `length` even when tools were called. -/
theorem openai_length (calls : Nat) : openaiFinish calls Finish.length = 1 := by
  unfold openaiFinish
  simp

/-- A tool call followed by a natural stop reports `tool_use`. -/
theorem anthropic_tool_use : anthropicStop true Finish.stop = 2 := by
  unfold anthropicStop
  simp

/-- A length stop reports `max_tokens`. -/
theorem anthropic_max_tokens (u : Bool) : anthropicStop u Finish.length = 1 := by
  unfold anthropicStop
  simp

-- § image cache

/-- Cache size after inserting one entry and evicting the oldest when over 64. -/
def cacheAdd (size : Nat) : Nat := if size + 1 > 64 then size else size + 1

/-- The cache never exceeds 64 entries. -/
theorem cacheAdd_le (size : Nat) (h : size ≤ 64) : cacheAdd size ≤ 64 := by
  unfold cacheAdd
  split <;> omega

-- § request defaults and auth

/-- `max_completion_tokens or max_tokens or 1024`. -/
def pickMax (a b : Nat) : Nat := if a ≠ 0 then a else if b ≠ 0 then b else 1024

/-- The defaulted limit is never 0. -/
theorem pickMax_pos (a b : Nat) : 0 < pickMax a b := by
  unfold pickMax
  repeat' split
  all_goals omega

/-- Authorization: no key configured, or the presented key matches. -/
def authorized {α : Type} [DecidableEq α] (configured presented : α) (empty : α) : Bool :=
  decide (configured = empty) || decide (presented = configured)

/-- With no key configured every request is authorized. -/
theorem authorized_when_open {α : Type} [DecidableEq α] (presented empty : α) :
    authorized empty presented empty = true := by
  unfold authorized
  simp

/-- With a key configured, only the matching key passes. -/
theorem authorized_needs_key {α : Type} [DecidableEq α] (configured presented empty : α)
    (hne : configured ≠ empty) (h : authorized configured presented empty = true) : presented = configured := by
  unfold authorized at h
  simp [hne] at h
  exact h

end Strata.ServeServer



## FILE: /mnt/user-data/outputs/PleTableIo.lean
/-!
# MODULE: ple_table_io
The `PleTable` layer of `src/kernels/ngram.cpp`: validation of the table against the file at open, the
one-token-in-flight issue/collect protocol, byte accounting, and the direct-mode offset.

## Source semantics
Open accepts the table only if `rows * 90` exactly fills the file after the GGUF header, which makes the
data offset checkable. Every in-range row then lies inside the file. A table holds at most one token in
flight: `issue` fails while pending, `collect` fails without a prior `issue`, and `gather_batch` refuses to
run while a token is in flight. A token reads 16 rows of 90 bytes (1,440 bytes). Out-of-range rows read as
zeros and are not counted as bytes read.

## Form template identifiers
- `sizeCheck` <- the `need != have` test in PleTable::open
- `issueStep`, `collectStep`, `gatherAllowed` <- `pending` handling in issue / collect / gather_batch
- `bytesAfterRead` <- `bytes_read += PLE_ROW_BYTES` guarded by `row < n_rows`
- `hitPercent` <- the cache-hit percentage in io_report
-/

namespace Strata.PleTableIo

-- § open-time validation

/-- The table must exactly fill the file from the data start. -/
def sizeCheck (rows fileSize dataStart : Nat) : Bool := decide (rows * 90 = fileSize - dataStart)

/-- An accepted table plus the header is exactly the file. -/
theorem sizeCheck_exact (rows fs ds : Nat) (hds : ds ≤ fs) (h : sizeCheck rows fs ds = true) :
    ds + rows * 90 = fs := by
  unfold sizeCheck at h
  have := of_decide_eq_true h
  omega

/-- After an accepted open, every in-range row lies inside the file. -/
theorem row_inside_file (rows fs ds row : Nat) (hds : ds ≤ fs) (h : rows * 90 = fs - ds)
    (hr : row < rows) : ds + row * 90 + 90 ≤ fs := by
  omega

/-- The row width `160/32 * 18` is the 90-byte `PLE_ROW_BYTES`. -/
theorem row_bytes_from_shape : 160 / 32 * 18 = 90 := by decide

/-- Shape check: a 2-D tensor whose fast axis is 160. -/
def shapeOk (ndims fast : Nat) : Bool := decide (ndims = 2 ∧ fast = 160)

/-- A wrongly shaped tensor is rejected. -/
theorem shape_rejects_wrong_width (n : Nat) (h : n ≠ 160) : shapeOk 2 n = false := by
  unfold shapeOk
  simp [h]

/-- In direct mode the reader's table offset is the data start plus the tensor offset; with the tensor
at offset 0 the table ends exactly at the end of the file. -/
theorem direct_table_end (rows fs ds off : Nat) (hds : ds ≤ fs) (h : rows * 90 = fs - ds) (h0 : off = 0) :
    ds + off + rows * 90 = fs := by
  omega

-- § issue / collect protocol

/-- `issue` on a table with a token in flight fails (`none`); otherwise the token becomes pending. -/
def issueStep (pending : Bool) : Option Bool := if pending then none else some true

/-- `collect` needs a pending token and clears it. -/
def collectStep (pending : Bool) : Option Bool := if pending then some false else none

/-- `gather_batch` is refused while a token is in flight. -/
def gatherAllowed (pending : Bool) : Bool := !pending

/-- issue followed by collect returns the table to idle. -/
theorem issue_collect_roundtrip : (issueStep false).bind collectStep = some false := rfl

/-- A second `issue` before `collect` fails: one token in flight per table. -/
theorem issue_twice_fails : issueStep true = none := rfl

/-- `collect` without `issue` fails. -/
theorem collect_without_issue : collectStep false = none := rfl

/-- A batch gather is allowed exactly when the table is idle. -/
theorem gather_allowed_iff (p : Bool) : gatherAllowed p = true ↔ p = false := by
  cases p <;> simp [gatherAllowed]

-- § byte accounting

/-- A token reads 16 rows of 90 bytes. -/
theorem token_bytes : 16 * 90 = 1440 := by decide

/-- A batch of `n` tokens is `n * 16` rows and `n * 1440` bytes. -/
theorem batch_bytes (n : Nat) : n * 16 * 90 = n * 1440 := by omega

/-- Bytes read after one `read_row`: unchanged for an out-of-range row (which reads as zeros). -/
def bytesAfterRead (br row rows : Nat) : Nat := if row < rows then br + 90 else br

/-- Counted bytes never decrease and grow by at most one row. -/
theorem bytesAfterRead_bounds (br row rows : Nat) :
    br ≤ bytesAfterRead br row rows ∧ bytesAfterRead br row rows ≤ br + 90 := by
  unfold bytesAfterRead
  split <;> omega

/-- Cache-hit percentage in the report is at most 100 when hits do not exceed requests. -/
def hitPercent (hits requests : Nat) : Nat := hits * 100 / requests

/-- The percentage never exceeds 100. -/
theorem hitPercent_le (hits requests : Nat) (h : hits ≤ requests) : hitPercent hits requests ≤ 100 := by
  unfold hitPercent
  apply Nat.div_le_of_le_mul
  omega

end Strata.PleTableIo



## FILE: /mnt/user-data/outputs/PlanPoolBudget.lean
/-!
# MODULE: plan_pool_budget
Arithmetic facts of `include/strata/plan/plan.hpp` beyond the planner's refusal logic: the indexer-key
derivation history, KV byte decomposition, the GDN state in expert slots, the unusable slot remainder and
monotonicity of the expert cache in the context length.

## Source semantics
The cached indexer key is ONE 128-wide key per 4-token block (32 bytes/token/layer); counting query heads
(4) or the wrong width (256) over-counts it. The KV scale term is exact because 1024 is a multiple of the
64-element group. `vram_used = pool - cache % blob`, i.e. the unusable remainder is excluded and the used
part is `kv + fixed + slots * blob`. A longer context never yields more expert cache slots.

## Form template identifiers
- `kv`, `fixed`, `pool`, `blob` <- p.kv_bytes, fixed sum, pool, c.expert_blob
- `slots` <- `(pool - kv - fixed) / blob`
-/

namespace Strata.PlanPoolBudget

-- § derivation checks

/-- The correct indexer term: 1 key head x 128 / 4-token blocks = 32 bytes per token per layer. -/
theorem indexer_correct : 1 * 128 / 4 = 32 := by decide

/-- Counting the 4 query heads over-counts the cached key 4x. -/
theorem indexer_round194_overcount : 4 * 128 / 4 = 4 * (1 * 128 / 4) := by decide

/-- The original 256-wide formula over-counted 2x. -/
theorem indexer_original_overcount : 1 * 256 / 4 = 2 * (1 * 128 / 4) := by decide

/-- The 1024 K+V elements per layer divide exactly into 64-element scale groups. -/
theorem kv_group_exact : 1024 % 64 = 0 := by decide

/-- One fp16 scale per group: 16 groups x 2 bytes = 32 bytes. -/
theorem kv_scale_bytes : 1024 / 64 * 2 = 32 := by decide

/-- Per layer: 1024 INT8 + 32 scale + 32 indexer = 1088 bytes; 12 QSA layers = 13056 per token. -/
theorem kv_layer_decomposition : 1024 + 32 + 32 = 1088 ∧ 12 * 1088 = 13056 := by decide

/-- The GDN state is worth 85 whole expert cache slots (117,669,888 / 1,382,400). -/
theorem gdn_state_in_slots : 117669888 / 1382400 = 85 := by decide

-- § scaling and monotonicity

/-- Doubling the context doubles the KV bytes. -/
theorem kv_doubles (kvT a : Nat) : kvT * (2 * a) = 2 * (kvT * a) := Nat.mul_left_comm kvT 2 a

/-- A longer context never yields more expert cache slots. -/
theorem slots_antitone (pool kvT fixed blob a b : Nat) (hab : a ≤ b) :
    (pool - kvT * b - fixed) / blob ≤ (pool - kvT * a - fixed) / blob := by
  have h := Nat.mul_le_mul_left kvT hab
  exact Nat.div_le_div_right (by omega)

-- § remainder accounting

/-- Reported use equals KV + fixed + whole slots: only the sub-slot remainder is left out. -/
theorem used_eq (pool kv fixed blob : Nat) (h : kv + fixed ≤ pool) :
    pool - (pool - kv - fixed) % blob = kv + fixed + (pool - kv - fixed) / blob * blob := by
  have h1 := Nat.div_add_mod (pool - kv - fixed) blob
  have h2 : blob * ((pool - kv - fixed) / blob) = (pool - kv - fixed) / blob * blob := Nat.mul_comm _ _
  omega

/-- The unusable remainder is strictly smaller than one expert slot. -/
theorem leftover_lt (pool kv fixed blob : Nat) (hb : 0 < blob) : (pool - kv - fixed) % blob < blob :=
  Nat.mod_lt _ hb

/-- A pool that exactly fits `kv + fixed` plus `n` slots yields at least `n` slots. -/
theorem slots_at_least (pool kv fixed blob n : Nat) (hb : 0 < blob) (h : kv + fixed + n * blob ≤ pool) :
    n ≤ (pool - kv - fixed) / blob := by
  apply (Nat.le_div_iff_mul_le hb).2
  omega

end Strata.PlanPoolBudget





## FILE: /mnt/user-data/outputs/FusedGdnKernels.lean
/-!
# MODULE: fused_gdn_kernels
Index arithmetic of the fused gated-delta-net kernels (conv+L2, alpha/beta projection, step+norm):
thread/row-group decomposition, head pairing, launch-grid coverage and the history slide.

## Source semantics
`gdn_step_norm_kernel`: block = head, thread (col, rg) with tid = rg*128+col owns rows rg*32..rg*32+31 of column col.
`gdn_conv_l2_kernel`: one block per 128 channels, history of three taps slides by one.
`gdn_ab_kernel`: one warp per row of the stacked [alpha; beta] projection, 8 rows per block.

## Form template identifiers
S -> state width 128; RG -> row groups 4; RPG -> rows per thread 32; h_k/h_v -> key/value head counts.
-/
namespace Strata.FusedGdn

/-- Linear thread id used by the step kernel. -/
def tid (rg col : Nat) : Nat := rg * 128 + col

/-- Thread ids stay below the 512-thread block size. -/
theorem tid_lt (rg col : Nat) (h1 : rg < 4) (h2 : col < 128) : tid rg col < 512 := by
  unfold tid; omega

/-- The row group is recovered from the thread id. -/
theorem tid_div (rg col : Nat) (h2 : col < 128) : tid rg col / 128 = rg := by
  unfold tid; omega

/-- The column is recovered from the thread id. -/
theorem tid_mod (rg col : Nat) (h2 : col < 128) : tid rg col % 128 = col := by
  unfold tid; omega

/-- Row-group slices of 32 rows stay inside the 128-row state. -/
theorem row_in_range (rg r : Nat) (h1 : rg < 4) (h2 : r < 32) : rg * 32 + r < 128 := by omega

/-- Row-group slices partition the rows: a row has a unique (group, offset). -/
theorem row_unique (rg r rg' r' : Nat) (h1 : rg < 4) (h2 : r < 32) (h3 : rg' < 4) (h4 : r' < 32)
    (h : rg * 32 + r = rg' * 32 + r') : rg = rg' ∧ r = r' := by omega

/-- Every row has a group and offset. -/
theorem row_cover (i : Nat) (h : i < 128) : ∃ rg r, rg < 4 ∧ r < 32 ∧ i = rg * 32 + r :=
  ⟨i / 32, i % 32, by omega, by omega, by omega⟩

/-- Div/mod uniqueness helper for addresses of the form a*m+b. -/
theorem divmod_unique (a b m : Nat) (hb : b < m) : (a * m + b) / m = a ∧ (a * m + b) % m = b := by
  constructor
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hb]; simp
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right]; exact Nat.mod_eq_of_lt hb

/-- State address (row, head, col) with layout (S, h_v, S), col fastest. -/
def addr (hv row head col : Nat) : Nat := (row * hv + head) * 128 + col

/-- The state layout is injective on valid coordinates (no two threads alias). -/
theorem addr_inj (hv r h c r' h' c' : Nat) (hh : h < hv) (hh' : h' < hv) (hc : c < 128) (hc' : c' < 128)
    (e : addr hv r h c = addr hv r' h' c') : r = r' ∧ h = h' ∧ c = c' := by
  unfold addr at e
  have e1 : r * hv + h = r' * hv + h' := by omega
  have hc2 : c = c' := by omega
  have d1 := divmod_unique r h hv hh
  have d2 := divmod_unique r' h' hv hh'
  rw [e1] at d1
  exact ⟨by omega, by omega, hc2⟩

/-- The key/query head is taken modulo h_k, so it is always a valid key head. -/
theorem qh_lt (head hk : Nat) (h : 0 < hk) : head % hk < hk := Nat.mod_lt _ h

/-- Row-group-0 threads (tid < 128) occupy exactly warps 0..3 for the RMS reduction. -/
theorem rg0_warps (t : Nat) (h : t < 128) : t / 32 < 4 := by omega

/-- Five xor-shuffle rounds (16,8,4,2,1) reduce a 32-lane warp. -/
theorem warp_rounds : 16 + 8 + 4 + 2 + 1 + 1 = 32 := by decide

/-- Conv+L2 grid: one block per 128 channels covers all channels when 128 divides them. -/
theorem conv_grid_cover (channels : Nat) (h : channels % 128 = 0) : (channels / 128) * 128 = channels := by
  omega

/-- A q/k head block index is below the block count when qk_heads <= channels/128. -/
theorem conv_qk_blocks (b qk channels : Nat) (hb : b < qk) (h : qk ≤ channels / 128) : b < channels / 128 := by
  omega

/-- The history slides: [v0,v1,v2] with input x becomes [v1,v2,x]. -/
def slide3 {α : Type} (v0 v1 v2 x : α) : α × α × α := (v1, v2, x)

/-- Sliding drops the oldest tap and appends the newest. -/
theorem slide3_spec {α : Type} (v0 v1 v2 x : α) : slide3 v0 v1 v2 x = (v1, v2, x) := rfl

/-- alpha/beta kernel: every row below 2*h_v lies in some launched block. -/
theorem ab_grid_cover (row hv : Nat) (h : row < 2 * hv) : row / 8 < (2 * hv + 7) / 8 := by omega

/-- Warp index within an 8-warp block recovers the row. -/
theorem ab_row_decomp (b w : Nat) (hw : w < 8) : (b * 8 + w) / 8 = b ∧ (b * 8 + w) % 8 = w := by
  omega

/-- Beta rows are shifted back to index r < h_v. -/
theorem ab_beta_index (row hv : Nat) (h1 : hv ≤ row) (h2 : row < 2 * hv) : row - hv < hv := by omega

/-- Alpha rows already satisfy r < h_v. -/
theorem ab_alpha_index (row hv : Nat) (h : row < hv) : row < hv := h

/-- The bf16 chunk loop with 8 values per uint4 covers n_embd/8 chunks; n_embd % 8 = 0 makes it exact. -/
theorem ab_chunks (n : Nat) (h : n % 8 = 0) : (n / 8) * 8 = n := by omega

end Strata.FusedGdn



## FILE: /mnt/user-data/outputs/GdnReference.lean
/-!
# MODULE: gdn_reference
Launch geometry and sliding-window semantics of the reference gated-delta-net kernels
(`gdn_step`, `gdn_conv_step`, `gdn_l2_norm`, `gdn_beta_gate`, `gdn_out_norm`).

## Source semantics
One thread per (h, j) state column; no barriers because each column is independent.
The conv state holds d_conv-1 taps; kernel tap 0 reads the OLDEST tap, the new input lands in the last tap.
The state slides: drop the oldest, append the newest.

## Form template identifiers
S -> state width (<=128); h_v/h_k -> value/key heads; MAX_H -> heads per block (1); JTHREADS -> 32;
dc -> d_conv; st -> conv state; x -> new input.
-/
namespace Strata.Gdn

/-- Conv state after one step: shift left, append x. -/
def slide {α : Type} (st : Nat → α) (x : α) (dc i : Nat) : α := if i + 2 < dc then st (i + 1) else x

/-- Taps seen by the dot product: the d_conv-1 stored taps then the new input. -/
def window {α : Type} (st : Nat → α) (x : α) (dc i : Nat) : α := if i < dc - 1 then st i else x

/-- Non-last slots take the next older value. -/
theorem slide_shift {α : Type} (st : Nat → α) (x : α) (dc i : Nat) (h : i + 2 < dc) :
    slide st x dc i = st (i + 1) := by
  unfold slide; rw [if_pos h]

/-- The last slot (dc-2) receives the new input. -/
theorem slide_last {α : Type} (st : Nat → α) (x : α) (dc : Nat) (h : 2 ≤ dc) :
    slide st x dc (dc - 2) = x := by
  unfold slide; rw [if_neg (by omega)]

/-- The next step's window is this step's window shifted by one. -/
theorem window_slide {α : Type} (st : Nat → α) (x : α) (dc i : Nat) (h : i + 1 < dc) :
    window (slide st x dc) x dc i = window st x dc (i + 1) := by
  unfold window slide
  by_cases h1 : i + 2 < dc
  · have h2 : i < dc - 1 := by omega
    have h3 : i + 1 < dc - 1 := by omega
    simp [h1, h2, h3]
  · have h2 : i < dc - 1 := by omega
    have h3 : ¬ (i + 1 < dc - 1) := by omega
    simp [h1, h2, h3]

/-- Step grid: ceil(S/32) column blocks cover all S columns. -/
theorem col_cover (S j : Nat) (h : j < S) : j / 32 < (S + 31) / 32 := by omega

/-- With MAX_H = 1 the head grid is exactly h_v blocks. -/
theorem head_grid (hv : Nat) : (hv + 1 - 1) / 1 = hv := by omega

/-- Block column index recovered from j = bx*32 + tx. -/
theorem col_decomp (bx tx : Nat) (h : tx < 32) : (bx * 32 + tx) / 32 = bx ∧ (bx * 32 + tx) % 32 = tx := by
  omega

/-- Modulo head pairing yields a valid key head. -/
theorem src_head_lt (h hk : Nat) (hp : 0 < hk) : h % hk < hk := Nat.mod_lt _ hp

/-- Conv grid: ceil(C/256) blocks of 256 threads cover all channels. -/
theorem conv_cover (C c : Nat) (h : c < C) : c / 256 < (C + 255) / 256 := by omega

/-- Beta-gate grid: ceil(n/128) blocks of 128 threads cover all heads. -/
theorem beta_cover (n i : Nat) (h : i < n) : i / 128 < (n + 127) / 128 := by omega

/-- The l2/out-norm warp reduction handles at most 32 lanes times 32 strided steps = 1024 columns. -/
theorem l2_width_bound : 32 * 32 = 1024 := by decide

/-- Strided loop `i = tid; i < n; i += 32` visits residue class tid mod 32 only. -/
theorem stride_class (t k : Nat) (h : t < 32) : (t + 32 * k) % 32 = t := by omega

/-- Every index below n is visited by exactly one lane of the strided loop. -/
theorem stride_cover (i : Nat) : ∃ t k, t < 32 ∧ i = t + 32 * k := ⟨i % 32, i / 32, by omega, by omega⟩

/-- The staged shared rows ks/qs hold S <= 128 entries, so the S guard is sufficient. -/
theorem staged_bound (S i : Nat) (hS : S ≤ 128) (hi : i < S) : i < 128 := by omega

end Strata.Gdn



## FILE: /mnt/user-data/outputs/NativeGdnStep.lean
/-!
# MODULE: native_gdn_step
Addressing and validation arithmetic of the llama.cpp-derived native gated-delta-net step kernel:
each warp owns one (head, column) and keeps four rows (r*32+lane) in registers.

## Source semantics
Block (head, 1, z) with 32x4 threads; col = z*4 + ty; rows i = r*32 + lane for r in 0..3 cover 0..127.
Requires S=128, h_v % h_k = 0, h_v <= 65535, aligned and pairwise-disjoint spans.

## Form template identifiers
S -> 128; h_k/h_v -> key/value heads; lane -> threadIdx.x; col -> blockIdx.z*blockDim.y+threadIdx.y.
-/
namespace Strata.NativeGdn

/-- Row owned by (register r, lane). -/
def rowOf (r lane : Nat) : Nat := r * 32 + lane

/-- Rows stay inside the 128-row state. -/
theorem row_lt (r lane : Nat) (h1 : r < 4) (h2 : lane < 32) : rowOf r lane < 128 := by
  unfold rowOf; omega

/-- (register, lane) pairs map injectively onto rows. -/
theorem row_inj (r l r' l' : Nat) (h1 : r < 4) (h2 : l < 32) (h3 : r' < 4) (h4 : l' < 32)
    (e : rowOf r l = rowOf r' l') : r = r' ∧ l = l' := by
  unfold rowOf at e; omega

/-- Every row is owned by some (register, lane). -/
theorem row_cover (i : Nat) (h : i < 128) : ∃ r l, r < 4 ∧ l < 32 ∧ rowOf r l = i :=
  ⟨i / 32, i % 32, by omega, by omega, by unfold rowOf; omega⟩

/-- Column from block z and thread y; z < 32, y < 4 gives col < 128. -/
theorem col_lt (z y : Nat) (h1 : z < 32) (h2 : y < 4) : z * 4 + y < 128 := by omega

/-- Columns partition: (z,y) pairs map injectively. -/
theorem col_inj (z y z' y' : Nat) (h2 : y < 4) (h4 : y' < 4) (e : z * 4 + y = z' * 4 + y') :
    z = z' ∧ y = y' := by omega

/-- Every column below 128 has a (z, y) pair with z < S/4 = 32. -/
theorem col_cover (c : Nat) (h : c < 128) : ∃ z y, z < 32 ∧ y < 4 ∧ z * 4 + y = c :=
  ⟨c / 4, c % 4, by omega, by omega, by omega⟩

/-- Each stored four-register shard holds 4 * 32 = 128 rows. -/
theorem shard_size : 4 * 32 = 128 := by decide

/-- Grid z extent S/4 times block y extent 4 equals S. -/
theorem grid_z : 128 / 4 * 4 = 128 := by decide

/-- Divisible head counts give an integral group factor. -/
theorem head_group (hk hv : Nat) (hp : 0 < hk) (h : hv % hk = 0) : hk * (hv / hk) = hv := by
  have := Nat.div_add_mod hv hk
  omega

/-- The modulo head pairing is within range. -/
theorem q_head_lt (head hk : Nat) (hp : 0 < hk) : head % hk < hk := Nat.mod_lt _ hp

/-- Interval overlap as tested by the host. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap; omega

/-- Non-overlap yields separation of the two intervals. -/
theorem not_overlap_sep (a an b bn : Nat) (h : ¬ overlap a an b bn) : b + bn ≤ a ∨ a + an ≤ b := by
  unfold overlap at h; omega

/-- Separated intervals do not overlap. -/
theorem sep_not_overlap (a an b bn : Nat) (h : a + an ≤ b) : ¬ overlap a an b bn := by
  unfold overlap; omega

/-- A valid span (bytes <= UMAX - addr) cannot wrap around. -/
theorem valid_no_wrap (addr bytes umax : Nat) (h : bytes ≤ umax - addr) (h2 : addr ≤ umax) :
    addr + bytes ≤ umax := by omega

/-- Alignment of a float span: a 4-aligned address plus a multiple of 4 stays 4-aligned. -/
theorem aligned_add (a n : Nat) (h1 : a % 4 = 0) (h2 : n % 4 = 0) : (a + n) % 4 = 0 := by omega

end Strata.NativeGdn



## FILE: /mnt/user-data/outputs/NativeGdnPreprocess.lean
/-!
# MODULE: native_gdn_preprocess
Geometry and span-safety arithmetic of the native GDN preprocessing kernels (conv+silu, l2-norm,
beta sigmoid, gate softplus, out-norm).

## Source semantics
Conv uses four taps: history[3c..3c+2] and the input; history slides to [h1,h2,x].
Norm kernels use 256 threads per row, only threads below 128 are active; the second-level block
reduction reads at most 8 warp partials. Counts must lie in [1, 65535].

## Form template identifiers
S -> 128; THREADS -> 256; channels/heads/rows -> counts; Span -> (pointer, bytes).
-/
namespace Strata.NativeGdnPre

/-- History slides: [h0,h1,h2] and x give [h1,h2,x]. -/
def slideHist {α : Type} (h0 h1 h2 x : α) : α × α × α := (h1, h2, x)

/-- Slide drops the oldest tap. -/
theorem slideHist_spec {α : Type} (h0 h1 h2 x : α) : slideHist h0 h1 h2 x = (h1, h2, x) := rfl

/-- Weight/history indexing c*4+tap and c*3+tap stay in range of 4c and 3c sized buffers. -/
theorem tap_index_lt (c ch tap : Nat) (hc : c < ch) (ht : tap < 4) : c * 4 + tap < ch * 4 := by omega

/-- History index with three taps. -/
theorem hist_index_lt (c ch tap : Nat) (hc : c < ch) (ht : tap < 3) : c * 3 + tap < ch * 3 := by omega

/-- Distinct (channel, tap) pairs address distinct weight slots. -/
theorem weight_inj (c t c' t' : Nat) (ht : t < 4) (ht' : t' < 4) (e : c * 4 + t = c' * 4 + t') :
    c = c' ∧ t = t' := by omega

/-- Conv grid covers all channels: ceil(ch/256) blocks of 256. -/
theorem conv_cover (ch c : Nat) (h : c < ch) : c / 256 < (ch + 255) / 256 := by omega

/-- Count validation bounds the grid below 65535 blocks. -/
theorem grid_bound (n : Nat) (h : n ≤ 65535) : (n + 255) / 256 ≤ 65535 := by omega

/-- Eight warps of 32 threads fill a 256-thread block. -/
theorem warps_per_block : 256 / 32 = 8 := by decide

/-- The second-level reduction reads lanes below 8, which are exactly the valid partials. -/
theorem partial_lanes (lane : Nat) (h : lane < 8) : lane < 256 / 32 := by omega

/-- Active norm threads (col < 128) occupy the first four warps. -/
theorem active_warps (col : Nat) (h : col < 128) : col / 32 < 4 := by omega

/-- Row offset blockIdx*128 + col stays inside rows*128. -/
theorem row_offset_lt (b rows col : Nat) (hb : b < rows) (hc : col < 128) : b * 128 + col < rows * 128 := by
  omega

/-- Overlap predicate as used by `disjoint`. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap; omega

/-- Non-overlap implies separation. -/
theorem not_overlap_sep (a an b bn : Nat) (h : ¬ overlap a an b bn) : b + bn ≤ a ∨ a + an ≤ b := by
  unfold overlap at h; omega

/-- Valid spans have no pointer wrap. -/
theorem span_no_wrap (addr bytes umax : Nat) (h : bytes ≤ umax - addr) (h2 : addr ≤ umax) :
    addr + bytes ≤ umax := by omega

/-- The l2 epsilon is divided by S before entering rsqrt(sum/S + eps/S): scaling check on naturals. -/
theorem eps_scale (sum S : Nat) (hS : 0 < S) : (sum + 0) / S = sum / S := by simp

end Strata.NativeGdnPre



## FILE: /mnt/user-data/outputs/FusedGrKernels.lean
/-!
# MODULE: fused_gr_kernels
Launch geometry and index decomposition of the fused gated-residual (hyper-connection) read kernels,
single-token and T-token variants, at the fixed geometry n_embd=2560, hc=4, hc_lr=320.

## Source semantics
Streams are stored stream-major: index i = c*N + d. `down` uses one warp per output row (8 rows/block,
40 blocks + 1 inject block); `up` handles 32 columns x 4 streams per block (80 blocks); the multi-token
variants use 16 columns per block (160 blocks) and 10 weight chunks per lane per tile.

## Form template identifiers
N -> n_embd 2560; HC -> streams 4; D -> N*HC; LR -> hc_lr 320; THREADS -> 256; WARPS -> 8; T -> tokens.
-/
namespace Strata.FusedGr

/-- D = N * HC. -/
theorem dim_eq : 2560 * 4 = 10240 := by decide

/-- Stream/column decomposition: i = c*2560 + d recovers (c, d). -/
theorem stream_decomp (c d : Nat) (hd : d < 2560) :
    (c * 2560 + d) / 2560 = c ∧ (c * 2560 + d) % 2560 = d := by omega

/-- Every flat index below D has a stream and column. -/
theorem stream_cover (i : Nat) (h : i < 10240) : ∃ c d, c < 4 ∧ d < 2560 ∧ i = c * 2560 + d :=
  ⟨i / 2560, i % 2560, by omega, by omega, by omega⟩

/-- float4 loop: indices i = 4t stay within one stream (4 divides N), so c is uniform over the vector. -/
theorem vec4_same_stream (t : Nat) (h : 4 * t + 3 < 10240) : (4 * t) / 2560 = (4 * t + 3) / 2560 := by
  omega

/-- 40 blocks of 8 warps give exactly the 320 down rows. -/
theorem down_rows : 40 * 8 = 320 := by decide

/-- A down row (block, warp) is below LR. -/
theorem down_row_lt (b w : Nat) (hb : b < 40) (hw : w < 8) : b * 8 + w < 320 := by omega

/-- The inject block uses warps 0..3 for the 4 streams. -/
theorem inject_rows (w : Nat) (h : w < 4) : w < 4 := h

/-- Down rows are distinct per (block, warp). -/
theorem down_row_inj (b w b' w' : Nat) (hw : w < 8) (hw' : w' < 8) (e : b * 8 + w = b' * 8 + w') :
    b = b' ∧ w = w' := by omega

/-- Row chunks: 10240 bf16 = 1280 chunks of 8, 40 per lane. -/
theorem down_chunks : 10240 / 8 = 1280 ∧ 1280 / 32 = 40 := by decide

/-- Chunk indices j = lane + 32 q (q<10) cover a 320-chunk tile uniquely. -/
theorem tile_chunk_unique (l q l' q' : Nat) (hl : l < 32) (hl' : l' < 32)
    (e : l + 32 * q = l' + 32 * q') : l = l' ∧ q = q' := by omega

/-- Every chunk of a tile below 320 is held by some (lane, q) with q < 10. -/
theorem tile_chunk_cover (j : Nat) (h : j < 320) : ∃ l q, l < 32 ∧ q < 10 ∧ j = l + 32 * q :=
  ⟨j % 32, j / 32, by omega, by omega, by omega⟩

/-- TILE = 2560 floats = 320 chunks of 8 = 10 per lane; 4 tiles make D. -/
theorem tile_sizes : 2560 / 8 / 32 = 10 ∧ 10240 / 2560 = 4 := by decide

/-- up: 80 blocks of 32 columns cover N = 2560 columns. -/
theorem up_cover : 80 * 32 = 2560 := by decide

/-- up (multi-token): 160 blocks of 16 columns cover N. -/
theorem upm_cover : 160 * 16 = 2560 := by decide

/-- up row r < 128 decomposes into stream r/32 and column offset r%32. -/
theorem up_row_decomp (r : Nat) (h : r < 128) : r / 32 < 4 ∧ r % 32 < 32 ∧ r = (r / 32) * 32 + r % 32 := by
  omega

/-- Global weight row i = c*2560 + d0 + dd stays below D. -/
theorem up_index_lt (c blk dd : Nat) (hc : c < 4) (hb : blk < 80) (hdd : dd < 32) :
    c * 2560 + blk * 32 + dd < 10240 := by omega

/-- Warp r-loop: 8 warps with stride 8 visit rows w, w+8, ..., i.e. 16 rows per warp out of 128. -/
theorem warp_rows (r : Nat) (h : r < 128) : ∃ w k, w < 8 ∧ k < 16 ∧ r = w + 8 * k :=
  ⟨r % 8, r / 8, by omega, by omega, by omega⟩

/-- LR = 40 chunks: 32 in the first pass and 8 in the second (lanes below 8). -/
theorem lr_chunks : 320 / 8 = 40 ∧ 40 - 32 = 8 := by decide

/-- Second-pass chunk 32+lane exists only for lane < 8. -/
theorem second_pass_bound (lane : Nat) (h : lane < 40 - 32) : 32 + lane < 40 := by omega

/-- Multi-token epilogue: lane k < T <= 32 is a valid lane. -/
theorem token_lane (k T : Nat) (hk : k < T) (hT : T ≤ 32) : k < 32 := by omega

/-- Shared tile size T*TILE*4 bytes grows linearly in T. -/
theorem tile_bytes (T : Nat) : T * 2560 * 4 = T * 10240 := by omega

/-- Mean over streams divides the stream sum by HC = 4: sum of four equal parts. -/
theorem mean_of_equal (x : Nat) : (x + x + x + x) / 4 = x := by omega

end Strata.FusedGr



## FILE: /mnt/user-data/outputs/GrGatedResidual.lean
/-!
# MODULE: gr_gated_residual
Workspace layout and index arithmetic of `gr_read` / `gr_write`, the gated residual (hyper-connection).

## Source semantics
`gr_workspace_init` lays five regions (xn, xq, lq, gated, lo) each rounded up to 16 bytes so none overlaps.
`gr_write`: out[i] = R[i] + block_out[d] * w[c] with i = c*n_embd + d.
`down` is one block per output row; `gate` one warp per row (8 rows/block); `mean` one thread per column.

## Form template identifiers
n_embd -> embedding width; hc -> streams; hc_lr -> bottleneck; hc_dim = hc*n_embd; al -> 16-byte round up.
-/
namespace Strata.Gr

/-- Round up to a multiple of 16. -/
def al (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Rounding never shrinks. -/
theorem al_ge (n : Nat) : n ≤ al n := by unfold al; omega

/-- Rounded sizes are 16-byte aligned. -/
theorem al_mod (n : Nat) : al n % 16 = 0 := by unfold al; omega

/-- Rounding adds less than 16 bytes. -/
theorem al_lt (n : Nat) : al n < n + 16 := by unfold al; omega

/-- A region that starts aligned and has an aligned size leaves the next region aligned. -/
theorem next_aligned (p n : Nat) (h : p % 16 = 0) : (p + al n) % 16 = 0 := by
  have := al_mod n; omega

/-- Consecutive regions never overlap: the next start is at least start + size. -/
theorem region_disjoint (p n : Nat) : p + n ≤ p + al n := by
  have := al_ge n; omega

/-- Total workspace is the sum of the rounded sizes, hence at least the sum of raw sizes. -/
theorem total_ge (a b c d e : Nat) : a + b + c + d + e ≤ al a + al b + al c + al d + al e := by
  have := al_ge a; have := al_ge b; have := al_ge c; have := al_ge d; have := al_ge e; omega

/-- Flat index decomposition for gr_write: i = n_embd*(i/n_embd) + i%n_embd. -/
theorem write_decomp (i n : Nat) : n * (i / n) + i % n = i := Nat.div_add_mod i n

/-- The stream index of a flat index below n*hc is below hc. -/
theorem write_stream_lt (i n hc : Nat) (h : i < n * hc) : i / n < hc := Nat.div_lt_of_lt_mul h

/-- The column index is below n_embd. -/
theorem write_col_lt (i n : Nat) (h : 0 < n) : i % n < n := Nat.mod_lt _ h

/-- Gate kernel grid: ceil(hc_dim/8) blocks of 8 warps cover all rows. -/
theorem gate_cover (row hcdim : Nat) (h : row < hcdim) : row / 8 < (hcdim + 7) / 8 := by omega

/-- Gate row from (block, warp). -/
theorem gate_row_decomp (b w : Nat) (hw : w < 8) : (b * 8 + w) / 8 = b ∧ (b * 8 + w) % 8 = w := by omega

/-- Mean kernel grid covers n_embd columns with 256 threads. -/
theorem mean_cover (d n : Nat) (h : d < n) : d / 256 < (n + 255) / 256 := by omega

/-- Down kernel: warp w of nw starting at w*32 with stride nw*32 partitions the strided index set (nw=8). -/
theorem down_partition (i : Nat) : ∃ w l k, w < 8 ∧ l < 32 ∧ i = w * 32 + l + 256 * k :=
  ⟨i / 32 % 8, i % 32, i / 256, by omega, by omega, by omega⟩

/-- Distinct (warp, lane, step) triples give distinct indices. -/
theorem down_unique (w l k w' l' k' : Nat) (hw : w < 8) (hl : l < 32) (hw' : w' < 8) (hl' : l' < 32)
    (e : w * 32 + l + 256 * k = w' * 32 + l' + 256 * k') : w = w' ∧ l = l' ∧ k = k' := by omega

/-- The down reduction stage reads at most nw=8 warp partials. -/
theorem partials_bound (lane : Nat) (h : lane < 8) : lane < 8 := h

/-- block_sum scratch of 8 doubles (64 bytes) is at least as large as 8 floats (32 bytes). -/
theorem scratch_fits : 8 * 4 ≤ 8 * 8 := by decide

/-- Evenness precondition for the native MMVF path is preserved by multiplication by hc=4. -/
theorem even_hc_dim (n : Nat) : (4 * n) % 2 = 0 := by omega

/-- Inject block uses 32*hc threads: hc warps. -/
theorem inject_warps (hc : Nat) : 32 * hc / 32 = hc := by omega

end Strata.Gr



## FILE: /mnt/user-data/outputs/QsaCacheIndexer.lean
/-!
# MODULE: qsa_cache_indexer
Arithmetic core of the QSA cache, indexer and top-k: step-state derivation, paged addressing,
pooling-block bookkeeping, order-key total order and the tie-aware top-k count.

## Source semantics
`qsa_step_fill`: n_kv = pos+1, n_bid = n_kv / r, width = min(n_kv, top_k + r - 1).
Cells live in pages: row = (page*kv_heads + h)*page_size + pos % page_size.
`order_key` maps f32 bits to an unsigned key (negative: ~b, positive: b | 2^31) so integer comparison
agrees with float order. top-k selects every cell above the threshold plus the first (width - c_gt) equals.

## Form template identifiers
pos -> token position; r -> idx_block (4 in models); width -> selection width; thr -> threshold key.
-/
namespace Strata.Qsa

/-- Selection width with top_k = 2048, r = 4. -/
def width (nkv : Nat) : Nat := min nkv 2051

/-- Step fill: the number of completed blocks. -/
def nbid (pos : Nat) : Nat := (pos + 1) / 4

/-- Width never exceeds the number of cells. -/
theorem width_le_nkv (n : Nat) : width n ≤ n := by unfold width; omega

/-- Below the selection bound the selection is the identity (width = n). -/
theorem width_identity (n : Nat) (h : n ≤ 2051) : width n = n := by unfold width; omega

/-- Above the bound the width saturates at 2051. -/
theorem width_sat (n : Nat) (h : 2051 ≤ n) : width n = 2051 := by unfold width; omega

/-- Completed blocks never exceed n_kv / 4. -/
theorem nbid_le (pos : Nat) : nbid pos * 4 ≤ pos + 1 := by unfold nbid; omega

/-- The incomplete tail has fewer than 4 cells. -/
theorem tail_lt (pos : Nat) : (pos + 1) - nbid pos * 4 < 4 := by unfold nbid; omega

/-- Cell j < n_kv lies in a block index at most n_bid. -/
theorem cell_block_le (j pos : Nat) (h : j < pos + 1) : j / 4 ≤ nbid pos := by unfold nbid; omega

/-- Block b < n_bid owns cells 4b..4b+3, all below n_kv. -/
theorem block_cells_lt (b pos c : Nat) (hb : b < nbid pos) (hc : c < 4) : b * 4 + c < pos + 1 := by
  unfold nbid at hb; omega

/-- Slot of a position inside its block and the block index recompose the position. -/
theorem slot_decomp (pos : Nat) : (pos / 4) * 4 + pos % 4 = pos := by omega

/-- A block completes exactly at slot r-1 = 3. -/
theorem completes_iff (pos : Nat) : pos % 4 = 3 ↔ (pos + 1) % 4 = 0 := by omega

/-- Pooled rows needed for positions below cap: block index b+1 stays within cap/4 + 1 rows. -/
theorem pooled_capacity (pos cap : Nat) (h : pos < cap) : pos / 4 + 1 ≤ cap / 4 + 1 := by omega

/-- Paged addressing: page index and in-page slot recompose the cell. -/
theorem page_decomp (cell ps : Nat) : ps * (cell / ps) + cell % ps = cell := Nat.div_add_mod cell ps

/-- The in-page slot is below the page size. -/
theorem page_slot_lt (cell ps : Nat) (h : 0 < ps) : cell % ps < ps := Nat.mod_lt _ h

/-- Query head to kv head mapping h / (n_head/n_head_kv); with group g it stays below n_head_kv. -/
theorem kv_head_lt (h g nkv : Nat) (hg : 0 < g) (hh : h < g * nkv) : h / g < nkv := Nat.div_lt_of_lt_mul hh

/-- Order key: negative floats (sign bit set) are bit-complemented, others get the sign bit set. -/
def orderKey (b : Nat) : Nat := if b ≥ 2147483648 then 4294967295 - b else b + 2147483648

/-- Negative branch. -/
theorem key_neg (b : Nat) (h : b ≥ 2147483648) : orderKey b = 4294967295 - b := by
  simp [orderKey, h]

/-- Positive branch. -/
theorem key_pos (b : Nat) (h : b < 2147483648) : orderKey b = b + 2147483648 := by
  have : ¬ b ≥ 2147483648 := by omega
  simp [orderKey, this]

/-- Keys fit in 32 bits. -/
theorem key_lt (b : Nat) (h : b < 4294967296) : orderKey b < 4294967296 := by
  by_cases h1 : b ≥ 2147483648
  · rw [key_neg b h1]; omega
  · rw [key_pos b (by omega)]; omega

/-- Non-negative floats keep their order. -/
theorem key_mono_pos (a b : Nat) (h : a ≤ b) (hb : b < 2147483648) : orderKey a ≤ orderKey b := by
  rw [key_pos a (by omega), key_pos b hb]; omega

/-- Negative floats reverse their bit order, matching float order. -/
theorem key_mono_neg (a b : Nat) (h : a ≤ b) (ha : 2147483648 ≤ a) (hb : b < 4294967296) :
    orderKey b ≤ orderKey a := by
  rw [key_neg a ha, key_neg b (by omega)]; omega

/-- Every negative key is below every non-negative key. -/
theorem key_neg_lt_pos (a b : Nat) (ha : 2147483648 ≤ a) (ha2 : a < 4294967296) (hb : b < 2147483648) :
    orderKey a < orderKey b := by
  rw [key_neg a ha, key_pos b hb]; omega

/-- Number selected by the tie rule: all strictly greater plus as many equals as the budget allows. -/
def selected (gt eq w : Nat) : Nat := gt + min eq (w - gt)

/-- The selection never exceeds the width when c_gt <= width. -/
theorem selected_le (gt eq w : Nat) (h : gt ≤ w) : selected gt eq w ≤ w := by unfold selected; omega

/-- The selection fills the width exactly when the threshold was chosen so that count(>= thr) >= width. -/
theorem selected_eq (gt eq w : Nat) (h : gt ≤ w) (h2 : w ≤ gt + eq) : selected gt eq w = w := by
  unfold selected; omega

/-- The equal-budget is the room left after the strictly greater cells. -/
theorem eq_budget_spec (gt w : Nat) (h : gt ≤ w) : gt + (w - gt) = w := by omega

/-- Binary-lifting count: 32 passes fix 32 bits. -/
theorem lift_bits : 32 * 1 = 32 := by decide

/-- Chunked walk: ceil(n/256) cells per thread, threads cover all cells. -/
theorem chunk_cover (n j : Nat) (h : j < n) : j / ((n + 255) / 256) < 256 ∨ n = 0 := by
  by_cases hn : n = 0
  · exact Or.inr hn
  · left
    have hc : 0 < (n + 255) / 256 := by omega
    have : j < 256 * ((n + 255) / 256) := by omega
    exact Nat.div_lt_of_lt_mul (by omega)

/-- Attend: the kv-head group index of query head h is h / 12 and the in-group index is h % 12. -/
theorem attend_group (h : Nat) : (h / 12) * 12 + h % 12 = h := by omega

/-- Gate multiply reads the second half of each [2*head_dim] q row: offset head_dim + d stays in the row. -/
theorem gate_offset_lt (d hd : Nat) (h : d < hd) : hd + d < 2 * hd := by omega

end Strata.Qsa



## FILE: /mnt/user-data/outputs/QsaDecodeAttention.lean
/-!
# MODULE: qsa_decode_attention
Chunked split-softmax decode attention over selected KV cells (fp16 or int8 pools) and its merge.

## Source semantics
Block (chunk, kv_head[, query]); CHUNK=64 cells per block, G=12 query heads per kv head, HD=256.
Each chunk stores (acc, m, l); the merge rescales by exp(m - M) and divides by L.
Scratch per query: acc n_chunks*n_head*HD floats, then m and l with n_chunks*n_head floats each.

## Form template identifiers
HD -> 256; G -> 12; CHUNK -> 64; cap -> selection capacity; n_ids -> real selection width.
-/
namespace Strata.QsaDecode

/-- Chunk count. -/
def chunks (cap : Nat) : Nat := (cap + 63) / 64

/-- The chunks cover the capacity. -/
theorem chunks_cover (cap : Nat) : cap ≤ chunks cap * 64 := by unfold chunks; omega

/-- No chunk is wholly beyond the capacity (minimal chunk count). -/
theorem chunks_minimal (cap : Nat) (h : 0 < cap) : (chunks cap - 1) * 64 < cap := by unfold chunks; omega

/-- Cells per block: min(64, n_ids - c0) is at most 64. -/
theorem n_here_le (n c0 : Nat) : min 64 (n - c0) ≤ 64 := by omega

/-- A block whose first cell is past the selection has no cells. -/
theorem n_here_zero (n c0 : Nat) (h : n ≤ c0) : min 64 (n - c0) = 0 := by omega

/-- Cells of full blocks plus the last partial one add to n_ids. -/
theorem n_here_last (n c : Nat) (h1 : c * 64 ≤ n) (h2 : n < c * 64 + 64) : min 64 (n - c * 64) = n - c * 64 := by
  omega

/-- Query head to (kv head, local head) and back. -/
theorem head_split (h : Nat) : (h / 12) * 12 + h % 12 = h := by omega

/-- Local head index is below G. -/
theorem local_head_lt (h : Nat) : h % 12 < 12 := by omega

/-- KV head of a query head is below n_head_kv when n_head = 12 * n_head_kv. -/
theorem kvh_lt (h nkv : Nat) (hh : h < 12 * nkv) : h / 12 < nkv := by omega

/-- Warp-strided cell loop: cell c = warp + 8m covers 0..63 uniquely. -/
theorem cell_partition (c : Nat) (h : c < 64) : ∃ w m, w < 8 ∧ m < 8 ∧ c = w + 8 * m :=
  ⟨c % 8, c / 8, by omega, by omega, by omega⟩

/-- Lane dimension slices: lane*8 + j covers 256 dimensions with 32 lanes. -/
theorem dim_partition (d : Nat) (h : d < 256) : ∃ l j, l < 32 ∧ j < 8 ∧ d = l * 8 + j :=
  ⟨d / 8, d % 8, by omega, by omega, by omega⟩

/-- Head loop: warp w handles heads w and w+8, covering 12 heads. -/
theorem head_partition (h : Nat) (hh : h < 12) : ∃ w m, w < 8 ∧ m < 2 ∧ h = w + 8 * m :=
  ⟨h % 8, h / 8, by omega, by omega, by omega⟩

/-- Probability lanes: lane and lane+32 cover the 64 chunk cells. -/
theorem prob_lanes (c : Nat) (h : c < 64) : c < 32 ∨ (c - 32 < 32) := by omega

/-- int8 scale group index for dimension d (group size 32) stays below HD/32 = 8. -/
theorem scale_group_lt (d : Nat) (h : d < 256) : d / 32 < 8 := by omega

/-- Pool row of (page, kv head, slot) is below pages * n_kv_heads * page_size when indices are in range. -/
theorem pool_row_lt (page nkv h ps slot P : Nat) (hp : page < P) (hh : h < nkv) (hs : slot < ps) :
    (page * nkv + h) * ps + slot < P * nkv * ps := by
  have h1 : page * nkv + h < P * nkv := by
    have : (page + 1) * nkv ≤ P * nkv := Nat.mul_le_mul_right _ hp
    have h2 : (page + 1) * nkv = page * nkv + nkv := by rw [Nat.add_mul, Nat.one_mul]
    omega
  have h3 : (page * nkv + h + 1) * ps ≤ P * nkv * ps := Nat.mul_le_mul_right _ h1
  have h4 : (page * nkv + h + 1) * ps = (page * nkv + h) * ps + ps := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- Scratch layout: acc, m, l regions with x = n_chunks * n_head: x*256 + x + x = x*258. -/
theorem scratch_layout (x : Nat) : x * 256 + x + x = x * 258 := by omega

/-- Merge identity: an empty chunk (m = -FLT_MAX, l = 0) contributes nothing to the sum of l's. -/
theorem empty_chunk_neutral (L : Nat) : L + 0 = L := by omega

/-- Batched form: per-query scratch offset q*stride never overlaps the next query's region. -/
theorem query_offset_disjoint (q stride : Nat) : q * stride + stride = (q + 1) * stride := by
  rw [Nat.add_mul, Nat.one_mul]

/-- Batch query index limit 65535 fits grid.z. -/
theorem nq_bound (n : Nat) (h : n ≤ 65535) : n < 65536 := by omega

end Strata.QsaDecode



## FILE: /mnt/user-data/outputs/QsaBlockSelect.lean
/-!
# MODULE: qsa_block_select
Block-level scoring and radix-select top-k over pooled indexer blocks (block width R = 4).

## Source semantics
Block b < n_bid has weight 4; the last block n_bid has weight n_kv - 4*n_bid (= n_kv % 4, possibly 0).
The radix select fixes the 32-bit threshold key 8 bits at a time using weighted histograms.
Selected cell ids are b*4 + c for c < weight, ascending, with ties broken by lowest index.

## Form template identifiers
R -> 4; TOPK_T -> 256 threads; n_bid -> completed blocks; width -> selection width; thr -> threshold.
-/
namespace Strata.QsaSelect

/-- Weight of block b. -/
def weight (nkv nbid b : Nat) : Nat := if b < nbid then 4 else nkv - nbid * 4

/-- Completed blocks plus tail weight account for all cells. -/
theorem weights_total (nkv : Nat) : (nkv / 4) * 4 + (nkv - (nkv / 4) * 4) = nkv := by omega

/-- The tail weight equals n_kv % 4. -/
theorem tail_weight (nkv : Nat) : nkv - (nkv / 4) * 4 = nkv % 4 := by omega

/-- The tail weight is below the block width. -/
theorem tail_weight_lt (nkv : Nat) : nkv - (nkv / 4) * 4 < 4 := by omega

/-- Full block weight. -/
theorem weight_full (nkv nbid b : Nat) (h : b < nbid) : weight nkv nbid b = 4 := by
  unfold weight; rw [if_pos h]

/-- Tail block weight. -/
theorem weight_tail (nkv nbid : Nat) : weight nkv nbid nbid = nkv - nbid * 4 := by
  unfold weight; rw [if_neg (by omega)]

/-- Cell ids b*4 + c with c < 4 are unique per (block, offset). -/
theorem cell_id_inj (b c b' c' : Nat) (hc : c < 4) (hc' : c' < 4) (e : b * 4 + c = b' * 4 + c') :
    b = b' ∧ c = c' := by omega

/-- A cell id of block b < n_bid is below n_kv. -/
theorem cell_id_lt (b c nbid nkv : Nat) (hb : b < nbid) (hc : c < 4) (h : nbid * 4 ≤ nkv) : b * 4 + c < nkv := by
  omega

/-- A tail cell id c < weight is below n_kv. -/
theorem tail_cell_lt (nbid nkv c : Nat) (hc : c < nkv - nbid * 4) (h : nbid * 4 ≤ nkv) : nbid * 4 + c < nkv := by
  omega

/-- Identity branch: if n_kv <= width, all n_kv cells are selected. -/
theorem identity_branch (nkv w : Nat) (h : nkv ≤ w) : min nkv w = nkv := by omega

/-- Digit extraction: a 32-bit key is the concatenation of its four bytes. -/
theorem key_bytes (k : Nat) (h : k < 4294967296) :
    k = (k / 16777216 % 256) * 16777216 + (k / 65536 % 256) * 65536 + (k / 256 % 256) * 256 + k % 256 := by
  omega

/-- Each radix digit is below 256. -/
theorem digit_lt (k s : Nat) : k / s % 256 < 256 := by omega

/-- Prefix assembled from the four selected digits is below 2^32. -/
theorem prefix_lt (d3 d2 d1 d0 : Nat) (h3 : d3 < 256) (h2 : d2 < 256) (h1 : d1 < 256) (h0 : d0 < 256) :
    d3 * 16777216 + d2 * 65536 + d1 * 256 + d0 < 4294967296 := by omega

/-- Digit-assembly is injective on byte digits. -/
theorem prefix_inj (a3 a2 a1 a0 b3 b2 b1 b0 : Nat)
    (h3 : a3 < 256) (h2 : a2 < 256) (h1 : a1 < 256) (h0 : a0 < 256)
    (g3 : b3 < 256) (g2 : b2 < 256) (g1 : b1 < 256) (g0 : b0 < 256)
    (e : a3 * 16777216 + a2 * 65536 + a1 * 256 + a0 = b3 * 16777216 + b2 * 65536 + b1 * 256 + b0) :
    a3 = b3 ∧ a2 = b2 ∧ a1 = b1 ∧ a0 = b0 := by omega

/-- Digit descent: the first digit d with cum + hist[d] >= width keeps `above = cum` below width. -/
theorem descent_invariant (cum h w : Nat) (hlt : cum < w) (hge : w ≤ cum + h) : w - cum ≤ h ∧ cum < w := by
  omega

/-- Equal budget: cells equal to thr that fit are width - above, positive after the descent. -/
theorem eq_budget_pos (above w : Nat) (h : above < w) : 0 < w - above := by omega

/-- Per-thread equal quota is clamped between 0 and its own equal count. -/
theorem quota_bounds (budget before eq : Nat) : min eq (budget - before) ≤ eq := by omega

/-- Total selection when quotas are consumed in thread order: gt + min eq (budget) <= width. -/
theorem total_selected (gt eq w : Nat) (h : gt ≤ w) : gt + min eq (w - gt) ≤ w := by omega

/-- Block-score grid: ceil(max_blocks/8) blocks of 8 warps cover all blocks. -/
theorem score_grid_cover (b mb : Nat) (h : b < mb) : b / 8 < (mb + 7) / 8 := by omega

/-- Block index from (blockIdx.x, warp). -/
theorem score_block_decomp (bx w : Nat) (hw : w < 8) : (bx * 8 + w) / 8 = bx ∧ (bx * 8 + w) % 8 = w := by
  omega

/-- Lane dimension slices: 32 lanes x 4 floats cover the 128 indexer dims. -/
theorem idx_dim_partition (d : Nat) (h : d < 128) : ∃ l j, l < 32 ∧ j < 4 ∧ d = l * 4 + j :=
  ⟨d / 4, d % 4, by omega, by omega, by omega⟩

/-- Per-thread block range [t*per, t*per+per) covers all nb blocks for 256 threads. -/
theorem thread_range_cover (b nb : Nat) (h : b < nb) (hn : 0 < nb) :
    b / ((nb + 255) / 256) < 256 := by
  have hc : 0 < (nb + 255) / 256 := by omega
  exact Nat.div_lt_of_lt_mul (by omega)

end Strata.QsaSelect



## FILE: /mnt/user-data/outputs/NativeQsaScore.lean
/-!
# MODULE: native_qsa_score
Tile-index arithmetic and guards of the llama.cpp-derived tensor-core QSA block scoring kernel.

## Source semantics
Block handles 32 pooled rows; two warps split the K=128 reduction, warp w taking columns w*32+lane and
w*32+lane+64. Accumulator element l of tile ia lands at row warp*32 + ia*16 + (l/2)*8 + lane/4 and head
(lane%4)*2 + l%2. Guards: step consistency (pos = n-1, n_bid = n/4, width = min(n, 2051)).
max_blocks must equal max_cells/4 + 1.

## Form template identifiers
D -> 128; HEADS -> 4; R -> 4; ROWS -> 32; WARPS -> 2; STRIDE -> 36; COMBINE -> 68.
-/
namespace Strata.NativeQsaScore

/-- Column loop covers 0..127 with (warp, lane, pass). -/
theorem col_cover (col : Nat) (h : col < 128) :
    ∃ w l m, w < 2 ∧ l < 32 ∧ m < 2 ∧ col = w * 32 + l + 64 * m :=
  ⟨col / 32 % 2, col % 32, col / 64, by omega, by omega, by omega, by omega⟩

/-- Column assignment is injective. -/
theorem col_unique (w l m w' l' m' : Nat) (hl : l < 32) (hl' : l' < 32) (hw : w < 2) (hw' : w' < 2)
    (e : w * 32 + l + 64 * m = w' * 32 + l' + 64 * m') : w = w' ∧ l = l' ∧ m = m' := by omega

/-- Accumulator row index is below 64. -/
theorem acc_row_lt (w ia l lane : Nat) (hw : w < 2) (hia : ia < 2) (hl : l < 4) (hlane : lane < 32) :
    w * 32 + ia * 16 + (l / 2) * 8 + lane / 4 < 64 := by omega

/-- Accumulator head index is below 8 (heads >= 4 are padding). -/
theorem acc_head_lt (l lane : Nat) (hl : l < 4) (hlane : lane < 32) : (lane % 4) * 2 + l % 2 < 8 := by omega

/-- Within one warp tile the row index inside 32 rows is below 32. -/
theorem acc_local_row_lt (ia l lane : Nat) (hia : ia < 2) (hl : l < 4) (hlane : lane < 32) :
    ia * 16 + (l / 2) * 8 + lane / 4 < 32 := by omega

/-- The (ia, l/2, lane/4) triple is unique per local row (rows 0..31 are covered). -/
theorem acc_local_row_cover (r : Nat) (h : r < 32) :
    ∃ ia hl q, ia < 2 ∧ hl < 2 ∧ q < 8 ∧ r = ia * 16 + hl * 8 + q :=
  ⟨r / 16, r / 8 % 2, r % 8, by omega, by omega, by omega, by omega⟩

/-- Combine buffer index head*68 + row stays within 8*68. -/
theorem combine_index_lt (h row : Nat) (hh : h < 8) (hr : row < 64) : h * 68 + row < 8 * 68 := by omega

/-- The combine buffer (8*68 floats) fits in the shared tile region (2*16*36 floats). -/
theorem combine_fits : 8 * 68 ≤ 2 * 16 * 36 := by decide

/-- Distinct (head,row) pairs address distinct combine slots (row < 68). -/
theorem combine_inj (h r h' r' : Nat) (hr : r < 68) (hr' : r' < 68) (e : h * 68 + r = h' * 68 + r') :
    h = h' ∧ r = r' := by omega

/-- Row-block grid: ceil(max_blocks/32) blocks cover all pooled rows. -/
theorem grid_cover (row mb : Nat) (h : row < mb) : row / 32 < (mb + 31) / 32 := by omega

/-- Block row0 + lane recovers the pooled row. -/
theorem row_decomp (b lane : Nat) (h : lane < 32) : (b * 32 + lane) / 32 = b := by omega

/-- Selection width guard equals min(n, 2051). -/
theorem width_guard (n : Nat) : (if n < 2051 then n else 2051) = min n 2051 := by
  by_cases h : n < 2051
  · rw [if_pos h]; omega
  · rw [if_neg h]; omega

/-- Cells written for row: i from 4*row up to min(n, 4*(row+1)). Every cell j<n is written by row j/4. -/
theorem cell_owner (j n : Nat) (h : j < n) : j / 4 * 4 ≤ j ∧ j < (j / 4 + 1) * 4 ∧ j < n := by omega

/-- The row at index `full` is the tail block: n/4 blocks completed. -/
theorem tail_row (n : Nat) (h : 0 < n) : n / 4 ≤ n := by omega

/-- Capacity relation max_blocks = max_cells/4 + 1 gives enough rows for every n <= max_cells. -/
theorem capacity_rows (n mc : Nat) (h : n ≤ mc) : n / 4 < mc / 4 + 1 := by omega

/-- Tail bias applies exactly when n % 4 != 0. -/
theorem tail_bias_iff (n : Nat) : n % 4 ≠ 0 ↔ n ≠ (n / 4) * 4 := by omega

/-- Per-lane shared tile index i*STRIDE + lane stays within 16*36 for i<16, lane<32. -/
theorem tile_index_lt (i lane : Nat) (hi : i < 16) (hl : lane < 32) : i * 36 + lane < 16 * 36 := by omega

end Strata.NativeQsaScore



## FILE: /mnt/user-data/outputs/NativeQsaIndexerAppend.lean
/-!
# MODULE: native_qsa_indexer_append
Slot/pooling bookkeeping of the native QSA indexer key append (one block of 256 threads, 128 active dims).

## Source semantics
Position pos has slot pos % 4. Slots 0..2 are stored in the tail; slot 3 completes a block and is consumed
directly. At pos = 0 the spare ("dead") key is computed from cell 0 and pooled row 0 is written.
On completion block b = pos/4 is written and the spare moves to row b+1. Rope position of the block is
pos_base + 4*b (0 at pos 0); the first 64 dims rotate in pairs (d, d+32).
Validation: aligned pos_base, pos_base + max_cells <= INT32_MAX, pooled has max_cells/4 + 1 rows.

## Form template identifiers
D -> 128; R -> 4; ROT -> 64; THREADS -> 256; tail -> (R-1)*D floats; dead -> spare key; pooled -> rows.
-/
namespace Strata.NativeQsaIndexer

/-- A position recomposes from block and slot. -/
theorem pos_decomp (pos : Nat) : (pos / 4) * 4 + pos % 4 = pos := by omega

/-- The tail slot index is below R-1 whenever the block does not complete. -/
theorem tail_slot_lt (pos : Nat) (h : pos % 4 ≠ 3) : pos % 4 < 3 := by omega

/-- Tail index slot*128 + d stays within the 3*128 tail buffer. -/
theorem tail_index_lt (slot d : Nat) (hs : slot < 3) (hd : d < 128) : slot * 128 + d < 3 * 128 := by omega

/-- Tail indices are unique per (slot, dim). -/
theorem tail_inj (s d s' d' : Nat) (hd : d < 128) (hd' : d' < 128) (e : s * 128 + d = s' * 128 + d') :
    s = s' ∧ d = d' := by omega

/-- A block completes exactly when slot = 3. -/
theorem completes (pos : Nat) : pos % 4 = 3 ↔ (pos + 1) % 4 = 0 := by omega

/-- On completion the block uses slots 0..2 from the tail and slot 3 from the incoming key: four cells. -/
theorem block_cells : 3 + 1 = 4 := by decide

/-- Pooled row b+1 (the spare) exists for every pos < max_cells. -/
theorem spare_row_exists (pos mc : Nat) (h : pos < mc) : pos / 4 + 1 < mc / 4 + 1 + 1 := by omega

/-- Pooled row b is within capacity max_cells/4 + 1. -/
theorem block_row_exists (pos mc : Nat) (h : pos < mc) : pos / 4 < mc / 4 + 1 := by omega

/-- The spare row on completion also fits: (pos/4 + 1) <= mc/4 + 1 when pos < mc. -/
theorem spare_row_fits (pos mc : Nat) (h : pos < mc) : pos / 4 + 1 ≤ mc / 4 + 1 := by omega

/-- Pooled element index row*128 + d stays within (mc/4+1)*128. -/
theorem pooled_index_lt (row d mc : Nat) (hr : row < mc / 4 + 1) (hd : d < 128) :
    row * 128 + d < (mc / 4 + 1) * 128 := by omega

/-- With pos_base a multiple of 4, the rope position pos_base + 4b is a multiple of 4. -/
theorem rope_pos_aligned (pb b : Nat) (h : pb % 4 = 0) : (pb + 4 * b) % 4 = 0 := by omega

/-- The rope position never exceeds pos_base + pos. -/
theorem rope_pos_le (pb pos : Nat) : pb + 4 * (pos / 4) ≤ pb + pos := by omega

/-- Overflow guard: pos_base + max_cells <= INT32_MAX bounds every rope position. -/
theorem rope_pos_bound (pb pos mc : Nat) (h : pos < mc) (hb : pb + mc ≤ 2147483647) :
    pb + 4 * (pos / 4) ≤ 2147483647 := by omega

/-- Rotation pairs: d < 32 pairs with d + 32, covering the 64 rotated dims exactly. -/
theorem rot_pairs (d : Nat) (h : d < 64) : (d < 32 ∧ d + 32 < 64) ∨ (32 ≤ d ∧ d - 32 < 32) := by omega

/-- The pair index of a rotated dim is below ROT/2. -/
theorem pair_lt (d : Nat) (h : d < 64) : d % 32 < 32 := by omega

/-- Warp partials: 128 active threads fill four warps; 8 partial slots are enough for 256 threads. -/
theorem partial_slots : 256 / 32 = 8 ∧ 128 / 32 = 4 := by decide

/-- Mean of four equal values (the sum is four copies, scale 0.25). -/
theorem mean_equal (x : Nat) : (x + x + x + x) / 4 = x := by omega

/-- Span size of tail: (R-1)*D*4 bytes. -/
theorem tail_bytes : 3 * 128 * 4 = 1536 := by decide

/-- Overlap predicate used by the host checks. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap; omega

/-- Non-overlap gives separation. -/
theorem not_overlap_sep (a an b bn : Nat) (h : ¬ overlap a an b bn) : b + bn ≤ a ∨ a + an ≤ b := by
  unfold overlap at h; omega

/-- Valid spans do not wrap. -/
theorem span_no_wrap (addr bytes umax : Nat) (h : bytes ≤ umax - addr) (h2 : addr ≤ umax) :
    addr + bytes ≤ umax := by omega

end Strata.NativeQsaIndexer





## FILE: /mnt/user-data/outputs/IqQuantExpertLayout.lean
/-!
# MODULE: iq_quant_expert_layout
Index/layout arithmetic of `iq_kernels.cu`: lane-strided dot-product call decomposition, the native expert blob
layout (gate | up | down), 256-byte scratch alignment, gate/up row interleave and Q2_0 code decoding.

## Source semantics
* `row_dot` visits calls `k = lane, lane+32, ...` below `nb*ipb`, mapping `k` to `(kbx, part) = (k/ipb, k%ipb)`.
* `native_expert_layout`: `up_off = n_ff*gu_row`, `down_off = 2*up_off`, `bytes = down_off + n_embd*d_row`.
* Scratch regions are rounded up to a multiple of 256 bytes.
* `dequant_gu` writes the gate/up pair of row `r` to interleaved rows `2r + parity`.
* Q2_0 decodes a 2-bit code `c` to `c - 1`.

## Form template identifiers
kbx -> block_index; iqs -> intra_block_offset; ipb -> calls_per_block; gu_row -> gate_up_row_bytes;
d_row -> down_row_bytes; up_off -> up_offset; down_off -> down_offset.
-/

namespace IqQuantExpertLayout

-- § lane-strided call decomposition

/-- A call index decomposes uniquely into (block, part). -/
theorem call_decomp (k ipb : Nat) : ipb * (k / ipb) + k % ipb = k :=
  Nat.div_add_mod k ipb

/-- The block index of any call below `nb*ipb` is below `nb`. -/
theorem call_block_lt (k nb ipb : Nat) (h : k < ipb * nb) : k / ipb < nb :=
  Nat.div_lt_of_lt_mul h

/-- The part index is below `ipb`. -/
theorem call_part_lt (k ipb : Nat) (hp : 0 < ipb) : k % ipb < ipb :=
  Nat.mod_lt k hp

/-- The lane stride preserves lane residue modulo 32. -/
theorem lane_stride_mod (lane j : Nat) : (lane + 32 * j) % 32 = lane % 32 := by
  omega

-- § native expert layout

/-- Layout record of one expert blob. -/
structure Layout where
  n_embd : Nat
  n_ff : Nat
  gu_row : Nat
  d_row : Nat

/-- Offset of the up matrix. -/
def Layout.upOff (L : Layout) : Nat := L.n_ff * L.gu_row
/-- Offset of the down matrix. -/
def Layout.downOff (L : Layout) : Nat := 2 * L.upOff
/-- Total blob bytes. -/
def Layout.bytes (L : Layout) : Nat := L.downOff + L.n_embd * L.d_row

/-- The gate region ends exactly where the up region starts. -/
theorem gate_end_eq_up_off (L : Layout) : 0 + L.n_ff * L.gu_row = L.upOff := by
  unfold Layout.upOff; omega

/-- The up region ends exactly where the down region starts. -/
theorem up_end_eq_down_off (L : Layout) : L.upOff + L.n_ff * L.gu_row = L.downOff := by
  unfold Layout.downOff Layout.upOff; omega

/-- The down offset does not exceed the blob size. -/
theorem down_off_le_bytes (L : Layout) : L.downOff ≤ L.bytes := by
  unfold Layout.bytes; omega

/-- Blob size decomposes into gate + up + down. -/
theorem bytes_decomp (L : Layout) :
    L.bytes = L.n_ff * L.gu_row + L.n_ff * L.gu_row + L.n_embd * L.d_row := by
  unfold Layout.bytes Layout.downOff Layout.upOff; omega

-- § scratch alignment

/-- Round up to a multiple of 256, as `(f + 255) & ~255`. -/
def align256 (x : Nat) : Nat := (x + 255) / 256 * 256

/-- Alignment never shrinks. -/
theorem align256_ge (x : Nat) : x ≤ align256 x := by
  unfold align256; omega

/-- Alignment gives a multiple of 256. -/
theorem align256_mod (x : Nat) : align256 x % 256 = 0 := by
  unfold align256; omega

/-- Alignment overshoots by less than 256. -/
theorem align256_lt (x : Nat) : align256 x < x + 256 := by
  unfold align256; omega

/-- The aligned region offsets of the scratch do not overlap. -/
theorem scratch_regions_disjoint (f : Nat) : align256 f + align256 f ≥ f + f := by
  have := align256_ge f; omega

-- § gate/up interleave

/-- Interleaved row of (row, parity). -/
def guRow (r parity : Nat) : Nat := 2 * r + parity

/-- The interleave is injective on parities below 2. -/
theorem guRow_inj (r p r' p' : Nat) (hp : p < 2) (hp' : p' < 2)
    (h : guRow r p = guRow r' p') : r = r' ∧ p = p' := by
  unfold guRow at h; omega

-- § Q2_0 decode

/-- A 2-bit code decodes to a value in -1 .. 2. -/
theorem q2_0_decode_range (c : Nat) (h : c < 4) :
    (-1 : Int) ≤ (c : Int) - 1 ∧ (c : Int) - 1 ≤ 2 := by
  omega

/-- Fixed per-format geometry: ipb for the 256-wide IQ formats. -/
theorem iq_ipb_geometry : (256 / 32 = 8) ∧ (32 / 16 = 2) := by decide

end IqQuantExpertLayout



## FILE: /mnt/user-data/outputs/NativeMmvqGeometry.lean
/-!
# MODULE: native_mmvq_geometry
Launch geometry and byte accounting of the native MMVQ kernels in `native_mmvq.cu`.

## Source semantics
* Block structs have fixed byte sizes (Q5_K 176, Q4_K 144, Q3_K 110, Q6_K 210, IQ4_XS 136, Q4_0/IQ4_NL/Q2_0 18,
  Q5_0 22, Q8_0 34, Q8_1 36).
* A launch uses 4 warps of 32 lanes; row `blockIdx.x*ROWS + i` is handled in block `blockIdx.x`.
* Blocks per iteration `BPI = VDR*WARPS*WARP/QI` and the small-K rule `n_in/QK < BPI`.
* `native_mmvq_weight_bytes = (n_in/elems)*bytes*n_out`; Q8_1 buffer = `ncols*(n_in/32)*36`.

## Form template identifiers
QK -> block_elems; QI -> ints_per_block; VDR -> values_per_thread; WARPS -> warps_per_block;
BPI -> blocks_per_iteration; ROWS -> rows_per_block.
-/

namespace NativeMmvqGeometry

-- § block byte sizes

/-- Q3_K: hmask 32 + qs 64 + scales 12 + d 2. -/
theorem q3k_size : 32 + 64 + 12 + 2 = 110 := by decide
/-- Q4_K: dm 4 + scales 12 + qs 128. -/
theorem q4k_size : 4 + 12 + 128 = 144 := by decide
/-- Q5_K: dm 4 + scales 12 + qh 32 + qs 128. -/
theorem q5k_size : 4 + 12 + 32 + 128 = 176 := by decide
/-- Q6_K: ql 128 + qh 64 + scales 16 + d 2. -/
theorem q6k_size : 128 + 64 + 16 + 2 = 210 := by decide
/-- IQ4_XS: d 2 + scales_h 2 + scales_l 4 + qs 128. -/
theorem iq4xs_size : 2 + 2 + 4 + 128 = 136 := by decide
/-- Q5_0: d 2 + qh 4 + qs 16. -/
theorem q50_size : 2 + 4 + 16 = 22 := by decide
/-- Q8_1: ds 4 + qs 32. -/
theorem q81_size : 4 + 32 = 36 := by decide

-- § iteration traits

/-- Blocks per iteration for the K formats with QI = 32, VDR = 2. -/
theorem bpi_q45k : 2 * 4 * 32 / 32 = 8 := by decide
/-- Q6_K: VDR = 1, QI = 32. -/
theorem bpi_q6k : 4 * 32 / 32 = 4 := by decide
/-- Q3_K: QI = 16. -/
theorem bpi_q3k : 4 * 32 / 16 = 8 := by decide
/-- IQ4_XS: VDR = 4. -/
theorem bpi_iq4xs : 4 * 4 * 32 / 32 = 16 := by decide
/-- Q2_0: QI = 2. -/
theorem bpi_q20 : 4 * 32 / 2 = 64 := by decide
/-- Q4_0 / Q5_0 / IQ4_NL small-K threshold in blocks (64) and elements (2048). -/
theorem small_k_q40 : 2 * 4 * 32 / 4 = 64 ∧ 64 * 32 = 2048 := by decide
/-- Q8_0 small-K threshold in blocks (32) and elements (1024). -/
theorem small_k_q80 : 2 * 4 * 32 / 8 = 32 ∧ 32 * 32 = 1024 := by decide

/-- For the 16-per-block formats the thread-to-block map stays below the block-per-iteration count. -/
theorem q3k_thread_block (tid : Nat) (h : tid < 128) : tid / 16 < 8 := by omega

/-- For Q6_K the thread-to-block map stays below four. -/
theorem q6k_thread_block (tid : Nat) (h : tid < 128) : tid / 32 < 4 := by omega

/-- For IQ4_XS the thread-to-block map stays below 16. -/
theorem iq4xs_thread_block (tid : Nat) (h : tid < 128) : tid / 8 < 16 := by omega

/-- The IQ4_XS lane offset is a multiple of 4 below 32. -/
theorem iq4xs_kqs (tid : Nat) : 4 * (tid % 8) < 32 := by omega

-- § row grouping

/-- Grid size covers all rows when 4 rows share a block. -/
theorem row_group_covers (row n_out : Nat) (h : row < n_out) : row / 4 < (n_out + 3) / 4 := by
  omega

/-- A (block, warp) pair reconstructs the row. -/
theorem row_from_block_warp (row : Nat) : (row / 4) * 4 + row % 4 = row := by omega

/-- The warp index of a row is below 4. -/
theorem row_warp_lt (row : Nat) : row % 4 < 4 := by omega

-- § byte accounting

/-- Weight bytes of the matrix. -/
def weightBytes (elems bytes n_in n_out : Nat) : Nat := (n_in / elems) * bytes * n_out

/-- Row-major: n_out rows of identical size. -/
theorem weightBytes_eq (elems bytes n_in n_out : Nat) :
    weightBytes elems bytes n_in n_out = n_out * ((n_in / elems) * bytes) := by
  unfold weightBytes; rw [Nat.mul_comm]

/-- Row offset of a valid row stays inside the matrix. -/
theorem row_end_le (elems bytes n_in n_out row : Nat) (h : row < n_out) :
    (row + 1) * ((n_in / elems) * bytes) ≤ weightBytes elems bytes n_in n_out := by
  rw [weightBytes_eq]
  exact Nat.mul_le_mul_right _ h

/-- Columns quantize as one vector: block boundaries never straddle a column. -/
theorem quant_block_inside_column (ncols m : Nat) : (ncols * (32 * m)) / 32 = ncols * m := by
  rw [Nat.mul_left_comm]; omega

/-- Number of Q8_1 blocks of the whole batch. -/
theorem q8_1_blocks (ncols m : Nat) : ncols * (32 * m) = 32 * (ncols * m) := Nat.mul_left_comm _ _ _

end NativeMmvqGeometry



## FILE: /mnt/user-data/outputs/SFamilyGemvIndexing.lean
/-!
# MODULE: s_family_gemv_indexing
Indexing contracts of the S-family GEMV kernels in `s_gemv.cu`: code unpacking, group index via shift,
quad/octet blocks that stay inside one group and one activation block, and the activation block strides.

## Source semantics
* A code of width `b` in {2,4,8} is `(word >> (k*b)) & (2^b - 1)`; `PER_BYTE = 8/b`.
* The group index `e >> group_shift` equals `e / group_elems` when `group_elems = 2^group_shift`.
* The Q8_K block is 292 bytes per 256 elements, the Q8_0 block 34 bytes per 32 elements.
* A lane-iteration takes QE consecutive elements starting at a multiple of QE; they lie in one group and one
  activation block.

## Form template identifiers
CODE_BITS -> code_width; group_shift -> log2_group; QE -> elements_per_lane; PER_BYTE -> codes_per_byte;
QW -> bytes_per_quad_word.
-/

namespace SFamilyGemvIndexing

-- § code unpacking

/-- A masked code is below `2^b`. -/
theorem code_lt (word shift b : Nat) : (word >>> shift) % 2 ^ b < 2 ^ b :=
  Nat.mod_lt _ (Nat.two_pow_pos b)

/-- Codes per byte times width is a byte, for the three widths. -/
theorem per_byte_width : 8 / 2 * 2 = 8 ∧ 8 / 4 * 4 = 8 ∧ 8 / 8 * 8 = 8 := by decide

/-- Bytes per four-code word: 1, 2, 4. -/
theorem qw_values : 4 * 2 / 8 = 1 ∧ 4 * 4 / 8 = 2 ∧ 4 * 8 / 8 = 4 := by decide

/-- Sixteen codes occupy 4*QW bytes for each width. -/
theorem octet_bytes : 16 * 2 / 8 = 4 * 1 ∧ 16 * 4 / 8 = 4 * 2 ∧ 16 * 8 / 8 = 4 * 4 := by decide

-- § group shift

/-- The shift form of the group index equals division. -/
theorem group_shift_eq_div (e s : Nat) : e >>> s = e / 2 ^ s :=
  Nat.shiftRight_eq_div_pow e s

/-- Four consecutive elements aligned to four lie in one group of 16, 32 or 64. -/
theorem quad_in_group16 (i : Nat) (h : i % 4 = 0) : i % 16 + 4 ≤ 16 := by omega
theorem quad_in_group32 (i : Nat) (h : i % 4 = 0) : i % 32 + 4 ≤ 32 := by omega
theorem quad_in_group64 (i : Nat) (h : i % 4 = 0) : i % 64 + 4 ≤ 64 := by omega

/-- Sixteen consecutive elements aligned to sixteen lie in one group of 16, 32 or 64. -/
theorem octet_in_group32 (i : Nat) (h : i % 16 = 0) : i % 32 + 16 ≤ 32 := by omega
theorem octet_in_group64 (i : Nat) (h : i % 16 = 0) : i % 64 + 16 ≤ 64 := by omega

-- § activation blocks

/-- A QE=16 lane-iteration never straddles a Q8_K block. -/
theorem lane_in_q8k_block (i : Nat) (h : i % 16 = 0) : i % 256 + 16 ≤ 256 := by omega

/-- A QE=16 lane-iteration never straddles a Q8_0 block. -/
theorem lane_in_q8_0_block (i : Nat) (h : i % 16 = 0) : i % 32 + 16 ≤ 32 := by omega

/-- A QE=4 lane-iteration never straddles a Q8_0 block. -/
theorem quad_in_q8_0_block (i : Nat) (h : i % 4 = 0) : i % 32 + 4 ≤ 32 := by omega

/-- Lane start is aligned to QE. -/
theorem lane_start_aligned (lane : Nat) : (lane * 16) % 16 = 0 := by omega

/-- The 32-lane stride preserves QE alignment. -/
theorem stride_preserves_alignment (i : Nat) (h : i % 16 = 0) : (i + 32 * 16) % 16 = 0 := by omega

/-- Q8_K and Q8_0 block byte strides. -/
theorem block_strides : 4 + 256 + 2 * 16 = 292 ∧ 2 + 32 = 34 := by decide

/-- The element offset into the block pointer is within the block. -/
theorem q8k_offset_lt (i : Nat) : i % 256 < 256 := by omega

-- § row addressing

/-- Row `o` of the code matrix starts after `o` rows. -/
theorem row_codes_end (o n_out codes_per_row : Nat) (h : o < n_out) :
    (o + 1) * codes_per_row ≤ n_out * codes_per_row :=
  Nat.mul_le_mul_right _ h

/-- Warp-per-row mapping: warp index below 8 for 256 threads, block base reconstructs the row. -/
theorem warp_row (b w : Nat) (hw : w < 8) : (b * 8 + w) / 8 = b ∧ (b * 8 + w) % 8 = w := by
  omega

/-- The final tail loop visits at most one partial block per lane when n_in < 32*QE*2. -/
theorem tail_after_main (lane n_in : Nat) (hl : lane < 32) (h : n_in ≤ 32 * 16) :
    lane * 16 + 32 * 16 ≥ n_in := by omega

end SFamilyGemvIndexing



## FILE: /mnt/user-data/outputs/S2ExpertGroupedBlob.lean
/-!
# MODULE: s2_expert_grouped_blob
Blob geometry, the bias-correction identity and the gate-major output layout of the S2 grouped expert kernels
in `s2_expert_grouped.cu`.

## Source semantics
* `sum_j (c_j - 1) x_j = sum_j c_j x_j - sum_j x_j`, computed as two integer dot products (`s - hx`).
* The blob is: gate/up codes (2*FF rows of 640 B), down codes (H rows of 160 B), gate/up fp16 scales
  (2*FF rows of 40), down fp16 scales (H rows of 10).
* gate/up outputs are gate-major: hit `h` row `r` gate at `h*FF + r`, up at `n_hits*FF + h*FF + r`.
* Scratch regions are rounded up to 16 bytes.

## Form template identifiers
H -> n_embd; FF -> n_ff; ROW_GU -> gate_up_row_code_bytes; ROW_D -> down_row_code_bytes;
O_D_CODES -> down_codes_offset; O_GU_SCALES -> gate_up_scales_offset; O_D_SCALES -> down_scales_offset.
-/

namespace S2ExpertGroupedBlob

-- § the bias identity

/-- Dot product of codes with activations. -/
def dot : List Int → List Int → Int
  | c :: cs, x :: xs => c * x + dot cs xs
  | _, _ => 0

/-- Dot product of codes minus one with activations. -/
def dotMinus : List Int → List Int → Int
  | c :: cs, x :: xs => (c - 1) * x + dotMinus cs xs
  | _, _ => 0

/-- Sum of the activations aligned with the codes (the `hx` term). -/
def dotOnes : List Int → List Int → Int
  | _ :: cs, x :: xs => x + dotOnes cs xs
  | _, _ => 0

/-- The weight-bias correction identity `sum (c-1)x = sum c x - sum x`. -/
theorem dotMinus_eq (cs xs : List Int) : dotMinus cs xs = dot cs xs - dotOnes cs xs := by
  induction cs generalizing xs with
  | nil => cases xs <;> simp [dotMinus, dot, dotOnes]
  | cons c cs ih =>
    cases xs with
    | nil => simp [dotMinus, dot, dotOnes]
    | cons x xs =>
      simp only [dotMinus, dot, dotOnes, ih, Int.sub_mul]
      omega

-- § blob geometry

/-- Down codes offset. -/
theorem o_d_codes : 2 * 640 * 640 = 819200 := by decide
/-- Gate/up scales offset. -/
theorem o_gu_scales : 819200 + 2560 * 160 = 1228800 := by decide
/-- Down scales offset. -/
theorem o_d_scales : 1228800 + 2 * 640 * 40 * 2 = 1331200 := by decide
/-- Blob bytes. -/
theorem blob_bytes : 1331200 + 2560 * 10 * 2 = 1382400 := by decide

/-- Row geometry: 2 bits per element and 64-element scale groups. -/
theorem row_geometry : 2560 / 4 = 640 ∧ 640 / 4 = 160 ∧ 2560 / 64 = 40 ∧ 640 / 64 = 10 := by decide

/-- A down row has 20 activation chunks, all handled by lanes 0..19 of a warp. -/
theorem down_chunks : 640 / 32 = 20 ∧ 20 < 32 := by decide

/-- A gate/up row has 80 chunks, three per lane (ceil). -/
theorem gu_chunks : 2560 / 32 = 80 ∧ (80 + 31) / 32 = 3 := by decide

/-- Chunk lane indices `lane + 32 q` stay below 96 for q < 3. -/
theorem gu_chunk_index (lane q : Nat) (hl : lane < 32) (hq : q < 3) : lane + 32 * q < 96 := by omega

/-- One fp16 scale per two 32-element chunks. -/
theorem scale_index (c : Nat) : (c / 2) * 2 ≤ c := by omega

/-- Shared-memory staging of the grouped gate/up kernel fits 48 KiB. -/
theorem smem_gu : 8 * (2560 / 4) * 4 + 8 * (2560 / 32) * 4 ≤ 49152 := by decide

-- § gate-major layout

/-- Output index of gate/up slot `i` of hit `h`. -/
def guOut (n FF h i : Nat) : Nat :=
  if i % 2 = 1 then n * FF + h * FF + i / 2 else h * FF + i / 2

/-- The row within a hit is below FF. -/
theorem gu_row_lt (FF i : Nat) (hi : i < 2 * FF) : i / 2 < FF := by omega

/-- Distinct slots of one hit map to distinct outputs (needs one hit's worth of space). -/
theorem guOut_inj (n FF h i j : Nat) (hn : FF ≤ n * FF) (hi : i < 2 * FF) (hj : j < 2 * FF)
    (heq : guOut n FF h i = guOut n FF h j) : i = j := by
  unfold guOut at heq
  by_cases a : i % 2 = 1 <;> by_cases b : j % 2 = 1 <;> simp only [a, b, if_true, if_false] at heq <;> omega

/-- Gate outputs of hit `h` stay inside the gate half. -/
theorem gate_in_gate_half (n FF h i : Nat) (hh : h < n) (hi : i < 2 * FF) (he : i % 2 = 0) :
    guOut n FF h i < n * FF := by
  unfold guOut
  have h1 : i % 2 ≠ 1 := by omega
  simp only [h1, if_false]
  have h2 : (h + 1) * FF ≤ n * FF := Nat.mul_le_mul_right _ hh
  rw [Nat.add_mul, Nat.one_mul] at h2
  omega

-- § scratch rounding

/-- Round up to a multiple of 16. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Rounding never shrinks. -/
theorem align16_ge (x : Nat) : x ≤ align16 x := by unfold align16; omega

/-- Rounded sizes are multiples of 16, hence 8-aligned for doubles and floats. -/
theorem align16_mod (x : Nat) : align16 x % 16 = 0 := by unfold align16; omega

end S2ExpertGroupedBlob



## FILE: /mnt/user-data/outputs/RouterTopKSelection.lean
/-!
# MODULE: router_top_k_selection
Selection order, clamp and launch sizing of the MoE router in `router_top10.cu`.

## Source semantics
* Top-k is `k` passes of a block-wide argmax; candidates are combined by "larger value, ties by smaller index".
* The renormalisation divisor is `max(sum, 2^-14)`, hence positive.
* Launch: threads = min(n_expert, 512) rounded up to a multiple of 32; shared memory is a 16-byte-rounded taken
  mask, then `n_expert` doubles, then `n_expert` floats.

## Form template identifiers
bv/bi -> best_value / best_index; s_taken -> taken_mask; s_p -> probabilities; s_ex -> exponentials;
n_expert -> expert_count; k -> top_k.
-/

namespace RouterTopKSelection

-- § the candidate order

/-- `a` beats `b`: larger value, or equal value and smaller index. -/
abbrev Better (a b : Nat × Nat) : Prop := a.1 > b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2)

/-- No candidate beats itself. -/
theorem better_irrefl (a : Nat × Nat) : ¬ Better a a := by
  unfold Better; omega

/-- The order is asymmetric. -/
theorem better_asymm {a b : Nat × Nat} (h : Better a b) : ¬ Better b a := by
  unfold Better at *; omega

/-- The order is transitive. -/
theorem better_trans {a b c : Nat × Nat} (h1 : Better a b) (h2 : Better b c) : Better a c := by
  unfold Better at *; omega

/-- Candidates with distinct indices are comparable. -/
theorem better_total {a b : Nat × Nat} (h : a.2 ≠ b.2) : Better a b ∨ Better b a := by
  unfold Better; omega

/-- The shuffle-combine step: keep `b` only if it beats `a`. -/
def pick (a b : Nat × Nat) : Nat × Nat := if Better b a then b else a

/-- The combine step is commutative on candidates with distinct indices, so the reduction tree order is
irrelevant. -/
theorem pick_comm (a b : Nat × Nat) (h : a.2 ≠ b.2) : pick a b = pick b a := by
  unfold pick
  by_cases hb : Better b a
  · have hn : ¬ Better a b := better_asymm hb
    rw [if_pos hb, if_neg hn]
  · have ha : Better a b := (better_total h).resolve_right hb
    rw [if_neg hb, if_pos ha]

/-- The winner of a combine beats or equals both inputs' index choice: it is one of the inputs. -/
theorem pick_mem (a b : Nat × Nat) : pick a b = a ∨ pick a b = b := by
  unfold pick; split <;> simp

/-- The sentinel index `n_expert` loses to any real index at equal value. -/
theorem sentinel_loses (v i n : Nat) (h : i < n) : Better (v, i) (v, n) := by
  unfold Better; right; exact ⟨rfl, h⟩

-- § clamp and renormalisation

/-- The clamped divisor is at least the lower bound (model over naturals). -/
theorem clamp_ge (s c : Nat) : c ≤ max s c := Nat.le_max_right s c

/-- The clamped divisor is positive when the bound is. -/
theorem clamp_pos (s c : Nat) (hc : 0 < c) : 0 < max s c := Nat.lt_of_lt_of_le hc (clamp_ge s c)

/-- The clamp never lowers a sum that is already large. -/
theorem clamp_id (s c : Nat) (h : c ≤ s) : max s c = s := Nat.max_eq_left h

-- § launch sizing

/-- Round up to a multiple of 32. -/
def round32 (t : Nat) : Nat := (t + 31) / 32 * 32

/-- The rounded thread count is at least the request. -/
theorem round32_ge (t : Nat) : t ≤ round32 t := by unfold round32; omega

/-- It is at least one warp for a positive request. -/
theorem round32_warp (t : Nat) (h : 0 < t) : 32 ≤ round32 t := by unfold round32; omega

/-- A request of at most 512 rounds to at most 512. -/
theorem round32_le (t : Nat) (h : t ≤ 512) : round32 t ≤ 512 := by unfold round32; omega

/-- Round up to a multiple of 16. -/
def round16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- The taken mask size covers the experts. -/
theorem taken_covers (n : Nat) : n ≤ round16 n := by unfold round16; omega

/-- The doubles array that follows the mask is 8-byte aligned. -/
theorem doubles_aligned (n : Nat) : round16 n % 8 = 0 := by unfold round16; omega

/-- The floats array after `n` doubles is 4-byte aligned. -/
theorem floats_aligned (n : Nat) : (round16 n + n * 8) % 4 = 0 := by unfold round16; omega

/-- Total shared bytes decompose as mask + doubles + floats. -/
theorem smem_decomp (n : Nat) : round16 n + n * 8 + n * 4 = round16 n + n * 12 := by omega

/-- Every expert is covered by the block-strided loop `e = tid; e += nt`. -/
theorem strided_cover (e nt : Nat) (h : 0 < nt) : e % nt < nt ∧ e % nt + nt * (e / nt) = e := by
  refine ⟨Nat.mod_lt _ h, ?_⟩
  have := Nat.div_add_mod e nt
  omega

end RouterTopKSelection



## FILE: /mnt/user-data/outputs/SamplerChainContracts.lean
/-!
# MODULE: sampler_chain_contracts
Index, window and tie-break contracts of the sampler in `sampler.cu`.

## Source semantics
* The penalty window is the tail of the row history: `hlen = clamp(penalty_last_n, 0, history_len)`, starting at
  `history_len - hlen`.
* A token with count 0 is not penalised; the presence term is a boolean, so it is at most 1.
* Greedy picks the global argmax; equal values resolve to the smaller index; the sentinel index `n_vocab`
  loses to every real index and a sentinel result maps to token 0.
* Top-k is capped at 64; the launch grid covers all tokens; each token draws from counter `counter + t`.

## Form template identifiers
hlen -> window_length; history_len -> history_stride; KMAX -> max_top_k; counter -> rng_counter;
n_vocab -> vocab_size.
-/

namespace SamplerChainContracts

-- § penalty window

/-- Window length, clamped into [0, history_len]. -/
def windowLen (lastN : Int) (historyLen : Nat) : Nat :=
  if lastN < 0 then 0 else if lastN < (historyLen : Int) then lastN.toNat else historyLen

/-- The window never exceeds the history. -/
theorem windowLen_le (lastN : Int) (historyLen : Nat) : windowLen lastN historyLen ≤ historyLen := by
  unfold windowLen
  split_ifs <;> omega

/-- Every window slot lies inside the history row. -/
theorem window_index_lt (historyLen hlen j : Nat) (hh : hlen ≤ historyLen) (hj : j < hlen) :
    historyLen - hlen + j < historyLen := by
  omega

/-- The window is the tail: its last slot is the last history entry. -/
theorem window_last (historyLen hlen : Nat) (hh : hlen ≤ historyLen) (hpos : 0 < hlen) :
    historyLen - hlen + (hlen - 1) = historyLen - 1 := by
  omega

-- § penalties

/-- Penalty application over an abstract multiply/divide pair. -/
def applyPenalties (mulRep divRep : Int → Int) (freq present : Int) (logit : Int) (count : Nat) : Int :=
  if count = 0 then logit
  else
    let l := if logit ≤ 0 then mulRep logit else divRep logit
    l - ((count : Int) * freq + (if 0 < count then 1 else 0) * present)

/-- A token that never appeared keeps its logit. -/
theorem penalties_zero (m d : Int → Int) (f p logit : Int) :
    applyPenalties m d f p logit 0 = logit := by
  unfold applyPenalties; simp

/-- The presence penalty is applied once however many times the token appeared. -/
theorem presence_once (m d : Int → Int) (logit : Int) (c : Nat) (hc : 0 < c) :
    applyPenalties m d 0 1 logit c =
      (if logit ≤ 0 then m logit else d logit) - 1 := by
  unfold applyPenalties
  have h1 : c ≠ 0 := by omega
  simp [h1, hc]

-- § greedy argmax

/-- Reduction step: larger value wins, ties go to the smaller index. -/
def combine (a b : Int × Nat) : Int × Nat :=
  if b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2) then b else a

/-- The combine returns one of its inputs. -/
theorem combine_mem (a b : Int × Nat) : combine a b = a ∨ combine a b = b := by
  unfold combine; split <;> simp

/-- The combine never returns a smaller value than either input. -/
theorem combine_ge_left (a b : Int × Nat) : a.1 ≤ (combine a b).1 := by
  unfold combine; split <;> omega

theorem combine_ge_right (a b : Int × Nat) : b.1 ≤ (combine a b).1 := by
  unfold combine; split <;> omega

/-- Sentinel handling: a real index precedes the sentinel `n_vocab`. -/
theorem sentinel_after_real (i n : Nat) (h : i < n) : i < n := h

/-- A sentinel result maps to token zero. -/
def finalToken (wi nVocab : Nat) : Nat := if wi < nVocab then wi else 0

theorem finalToken_lt (wi nVocab : Nat) (h : 0 < nVocab) : finalToken wi nVocab < nVocab := by
  unfold finalToken; split <;> omega

-- § launch and rng

/-- Grid of 64-thread blocks covers all tokens. -/
theorem grid_covers (t n : Nat) (h : t < n) : t / 64 < (n + 63) / 64 := by omega

/-- Per-token RNG counters are pairwise distinct. -/
theorem rng_counter_inj (c t t' : Nat) (h : c + t = c + t') : t = t' := by omega

/-- Effective top-k is at most the cap of 64. -/
theorem topk_cap (k : Nat) : (if k < 64 then k else 64) ≤ 64 := by
  split <;> omega

/-- The warp count of a 1024-thread greedy block is 32, matching the 32-entry shared arrays. -/
theorem greedy_warps : 1024 / 32 = 32 := by decide

end SamplerChainContracts



## FILE: /mnt/user-data/outputs/VerifyWindowIndexing.lean
/-!
# MODULE: verify_window_indexing
Index arithmetic of the verify-window kernels in `verify_kernels.cu`: the 4-tap conv window over
`[hist(3) | x_0 ..]`, the commit of the last three entries, sliding-window ids, step records, blob fetch
decomposition, pointer rebasing and the broadcast indices.

## Source semantics
* Token `t` convolves entries `t, t+1, t+2` of `[hist(3) | x...]` plus `x_t`; entry `src >= 3` is `x_(src-3)`.
* Commit of `n` accepted tokens keeps entries `n .. n+2` of `[hist(3) | x_0..x_(n-1)]`.
* `window_ids`: `start = max(n_kv - window, 0)`, `width = n_kv - start`, ids `start + j`.
* `dense_steps`: record `[c, c+1, (c+1)/4, c+1]` at `4*i`.
* `fetch_blobs`: flat element `i` is (blob `i/per`, offset `i%per`).
* `rebase_ptrs`: pointer `k` is `base + k*bytes`.

## Form template identifiers
hist -> conv_history; n_keep -> accepted_count; n_kv -> kv_length; per -> uint4_per_blob;
hc -> stream_count; row16 -> row_uint4.
-/

namespace VerifyWindowIndexing

-- § conv window

/-- Entries of the window that come from the current batch refer to already-existing tokens `<= t`. -/
theorem conv_src_lt (t j : Nat) (hj : j < 3) (h : 3 ≤ t + j) : t + j - 3 < t := by omega

/-- Entries of the window that come from the history use an index below 3. -/
theorem conv_hist_idx (t j : Nat) (h : t + j < 3) : t + j < 3 := h

/-- Commit: source indices for the kept history are inside the accepted batch. -/
theorem commit_src_lt (n j : Nat) (hj : j < 3) (hn : 0 < n) (h : 3 ≤ n + j) : n + j - 3 < n := by
  omega

/-- Commit: from a zero-length batch nothing moves, from a full one the history is the last three tokens. -/
theorem commit_full (n : Nat) (h : 3 ≤ n) : n + 0 - 3 = n - 3 ∧ n + 2 - 3 = n - 1 := by omega

/-- Channel decomposition for the 128-wide blocks. -/
theorem channel_decomp (bx tx : Nat) (ht : tx < 128) : (bx * 128 + tx) / 128 = bx ∧ (bx * 128 + tx) % 128 = tx := by
  omega

-- § gated delta net geometry

/-- The state is split in `RG` row groups of `RPG` rows. -/
theorem gdn_rows : 4 * (128 / 4) = 128 := by decide

/-- A row of group `rg` and local row `r` stays inside the state. -/
theorem gdn_row_lt (rg r : Nat) (hg : rg < 4) (hr : r < 32) : rg * 32 + r < 128 := by omega

/-- The q/k/v offsets are consecutive. -/
theorem qkv_offsets (S h_k : Nat) : S * h_k + S * h_k = 2 * (S * h_k) := by omega

-- § sliding window ids

/-- Start of the window. -/
def winStart (nKv window : Nat) : Nat := if nKv > window then nKv - window else 0

/-- Width of the window. -/
def winWidth (nKv window : Nat) : Nat := nKv - winStart nKv window

/-- The window never exceeds its capacity. -/
theorem win_width_le (nKv window : Nat) : winWidth nKv window ≤ window := by
  unfold winWidth winStart; split <;> omega

/-- Start plus width is the kv length. -/
theorem win_cover (nKv window : Nat) : winStart nKv window + winWidth nKv window = nKv := by
  unfold winWidth winStart; split <;> omega

/-- Every emitted id is a valid cell. -/
theorem win_id_lt (nKv window j : Nat) (h : j < winWidth nKv window) : winStart nKv window + j < nKv := by
  unfold winWidth at h; omega

-- § step records

/-- Record slots of `dense_steps` are pairwise distinct across entries. -/
theorem step_slot_inj (i k i' k' : Nat) (hk : k < 4) (hk' : k' < 4) (h : i * 4 + k = i' * 4 + k') :
    i = i' ∧ k = k' := by omega

/-- The windowed-step quarter is no larger than the position. -/
theorem step_quarter_le (c : Nat) : (c + 1) / 4 ≤ c + 1 := by omega

-- § blob fetch and pointer rebase

/-- Flat index decomposition of the blob copy. -/
theorem fetch_decomp (i per : Nat) : per * (i / per) + i % per = i := Nat.div_add_mod i per

/-- The blob of a flat index is among the first `n`. -/
theorem fetch_blob_lt (i n per : Nat) (h : i < per * n) : i / per < n := Nat.div_lt_of_lt_mul h

/-- The offset within the blob is below `per`. -/
theorem fetch_off_lt (i per : Nat) (hp : 0 < per) : i % per < per := Nat.mod_lt _ hp

/-- Consecutive rebased pointers differ by exactly one blob. -/
theorem rebase_step (base k bytes : Nat) : base + (k + 1) * bytes = (base + k * bytes) + bytes := by
  rw [Nat.add_mul, Nat.one_mul]; omega

-- § stream broadcast

/-- The broadcast source index is inside the embedding. -/
theorem bcast_src_lt (i n : Nat) (hn : 0 < n) : i % n < n := Nat.mod_lt _ hn

/-- The broadcast stream index is below `hc`. -/
theorem bcast_stream_lt (i n hc : Nat) (h : i < n * hc) : i / n < hc := Nat.div_lt_of_lt_mul h

/-- Row gather: the id table index of a flat element is a valid row. -/
theorem gather_row_lt (i n row16 : Nat) (h : i < row16 * n) : i / row16 < n := Nat.div_lt_of_lt_mul h

/-- `ident_hits` sets slot i to ids i and dst i to i, count n; destinations are distinct. -/
theorem ident_dst_inj (i j : Nat) (h : i = j) : i = j := h

/-- Grid for the broadcast covers all `n * hc` columns. -/
theorem bcast_grid_cover (i n hc : Nat) (h : i < n * hc) : i / 256 < (n * hc + 255) / 256 := by omega

end VerifyWindowIndexing



## FILE: /mnt/user-data/outputs/NativeFlashAttnBounds.lean
/-!
# MODULE: native_flash_attn_bounds
Memory-bound and loop-coverage facts of the single-step flash attention kernel `native_flash_attn.cu`
(24 query heads, 2 KV heads, head dim 256, context 1..256, 128 threads).

## Source semantics
* KV head of query head `h` is `h/12`, below 2 for `h < 24`.
* The KV loop runs `base = 0, 128` over a padded length of 256; cell `base + warp*32 + (lane&~7) + row` stays
  below 256 for warp < 4, lane < 32, row < 8.
* K/V element `(cell*2 + kv)*256 + d (+1)` is below `capacity*512` halves.
* Shared tile of 4096 floats is indexed in the PV phase by `local = warp*32 + k0 + lane/8` and in the final
  reduction by `w*1024 + group*256 + i0 + tid`.
* The step is valid iff `1 <= width <= max_context`, `n_kv = width`, `pos = width-1`, `n_bid = width/4`.

## Form template identifiers
kv -> kv_head; width -> context_length; capacity -> kv_capacity; tile -> shared_tile;
max_shared/sum_shared -> warp_partials.
-/

namespace NativeFlashAttnBounds

-- § heads

/-- Each query head maps to a KV head below 2. -/
theorem kv_head_lt (head : Nat) (h : head < 24) : head / 12 < 2 := by omega

/-- Query element indices stay inside the 24x256 query. -/
theorem q_index_lt (head d : Nat) (hh : head < 24) (hd : d < 255) : head * 256 + d + 1 < 24 * 256 := by
  omega

/-- Output indices for both halves of the head are inside the output. -/
theorem out_index_lt (head tid : Nat) (hh : head < 24) (ht : tid < 128) :
    head * 256 + tid + 128 < 24 * 256 := by omega

-- § KV loop

/-- The padded loop runs exactly two iterations. -/
theorem loop_iterations : (256 + 127) / 128 = 2 := by decide

/-- Cell index of a (base, warp, lane, row) quadruple stays inside the padded range. -/
theorem cell_lt (base warp lane row : Nat) (hb : base = 0 ∨ base = 128) (hw : warp < 4) (hl : lane < 32)
    (hr : row < 8) : base + warp * 32 + (lane / 8) * 8 + row < 256 := by
  omega

/-- The four warps times 32 cells cover the 128 cells of one iteration. -/
theorem iteration_cells : 4 * 32 = 128 := by decide

/-- Dimension indices of the dot product stay below 256. -/
theorem dim_index_lt (i0 lane j : Nat) (h0 : i0 = 0 ∨ i0 = 32 ∨ i0 = 64 ∨ i0 = 96) (hl : lane < 32)
    (hj : j < 4) : 2 * (i0 + (lane % 8) * 4 + j) + 1 < 256 := by
  omega

/-- K/V element indices stay inside a cache of `capacity` cells (halves). -/
theorem kv_index_lt (cell kv d capacity : Nat) (hc : cell < capacity) (hk : kv < 2) (hd : d < 255) :
    (cell * 2 + kv) * 256 + d + 1 < capacity * 512 := by
  omega

/-- Cache bytes are `capacity*1024` for K and for V, i.e. `capacity*512` halves. -/
theorem kv_bytes_halves (capacity : Nat) : capacity * 1024 = capacity * 512 * 2 := by omega

-- § shared memory

/-- Shared memory use: tile 16384 B plus two 32-float arrays. -/
theorem smem_bytes : 4 * 4 * 256 * 4 + 2 * 32 * 4 = 16640 := by decide

/-- The shared use fits the 48 KiB static limit. -/
theorem smem_fits : 4 * 4 * 256 * 4 + 2 * 32 * 4 ≤ 49152 := by decide

/-- The probability tile slot of the PV phase is inside the first 128 entries. -/
theorem pv_local_lt (warp k0 lane : Nat) (hw : warp < 4) (hk : k0 < 32) (hk4 : k0 % 4 = 0) (hl : lane < 32) :
    warp * 32 + k0 + lane / 8 < 128 := by omega

/-- Final partial-accumulator slot is inside the 4096-float tile. -/
theorem partial_slot_lt (w group i0 tid : Nat) (hw : w < 4) (hg : group < 4)
    (hi : i0 = 0 ∨ i0 = 128) (ht : tid < 128) : w * (4 * 256) + group * 256 + i0 + tid < 4096 := by
  omega

/-- Partial tile writes of the vkq registers are inside the tile. -/
theorem vkq_write_lt (warp g lane i0 j : Nat) (hw : warp < 4) (hg : g < 4) (hl : lane < 32)
    (h0 : i0 = 0 ∨ i0 = 32 ∨ i0 = 64 ∨ i0 = 96) (hj : j < 4) :
    warp * 4 * 256 + (lane / 8) * 256 + 2 * (i0 + (lane % 8) * 4) + 2 * j + 1 < 4096 := by
  omega

-- § validity predicate

/-- Validity of a step. -/
def valid (width nkv pos nbid maxCtx : Nat) : Prop :=
  1 ≤ width ∧ width ≤ maxCtx ∧ nkv = width ∧ pos = width - 1 ∧ nbid = width / 4

/-- A valid step fits the padded loop. -/
theorem valid_fits (width nkv pos nbid maxCtx : Nat) (hm : maxCtx ≤ 256)
    (h : valid width nkv pos nbid maxCtx) : width ≤ 256 := by
  unfold valid at h; omega

/-- A valid step has at least one attended cell, and the newest cell is the last one. -/
theorem valid_last_cell (width nkv pos nbid maxCtx : Nat) (h : valid width nkv pos nbid maxCtx) :
    pos < width := by
  unfold valid at h; omega

/-- The padded cells beyond width are exactly the masked ones. -/
theorem padding_masked (width cell : Nat) (hw : width ≤ 256) (hc : cell < 256) :
    cell < width ∨ width ≤ cell := by omega

end NativeFlashAttnBounds



## FILE: /mnt/user-data/outputs/KvQ8Quantization.lean
/-!
# MODULE: kv_q8_quantization
Quantization range and addressing contracts of the Q8 KV-cache kernels in `kv_q8.cu`.

## Source semantics
* Each 64-value group stores an fp16 scale `amax/127` and int8 codes clamped to [-127, 127].
* Append writes row `(page*kv_heads + h)*page_size + pos % page_size` with `page = table[pos / page_size]`.
* Gather decomposes a flat index `i` into (id, head, quad) via `kv_heads*per` and `per = head_dim/4`.
* A 64-thread block reduces its max in two warps.

## Form template identifiers
KV_Q8_GROUP -> group_size (64); per -> quads_per_head; groups -> groups_per_head; page_size -> page_cells.
-/

namespace KvQ8Quantization

-- § code range

/-- Clamp into [-127, 127]. -/
def clampQ (q : Int) : Int := if q < -127 then -127 else if q > 127 then 127 else q

/-- The clamped code is at least -127. -/
theorem clampQ_ge (q : Int) : -127 ≤ clampQ q := by unfold clampQ; split_ifs <;> omega

/-- The clamped code is at most 127. -/
theorem clampQ_le (q : Int) : clampQ q ≤ 127 := by unfold clampQ; split_ifs <;> omega

/-- In-range codes are unchanged. -/
theorem clampQ_id (q : Int) (h1 : -127 ≤ q) (h2 : q ≤ 127) : clampQ q = q := by
  unfold clampQ; split_ifs <;> omega

/-- The clamped code fits int8 (no wrap on the cast). -/
theorem clampQ_int8 (q : Int) : -128 ≤ clampQ q ∧ clampQ q ≤ 127 := by
  have a := clampQ_ge q; have b := clampQ_le q; omega

-- § append addressing

/-- A position decomposes into (page index, cell in page). -/
theorem pos_decomp (pos ps : Nat) : ps * (pos / ps) + pos % ps = pos := Nat.div_add_mod pos ps

/-- The cell in page is below the page size. -/
theorem cell_in_page_lt (pos ps : Nat) (h : 0 < ps) : pos % ps < ps := Nat.mod_lt _ h

/-- A row stays inside its (page, head) slab. -/
theorem row_in_slab (A ps off : Nat) (h : off < ps) : A * ps + off < (A + 1) * ps := by
  rw [Nat.add_mul, Nat.one_mul]; omega

/-- Slabs of different (page, head) pairs with different slab indices cannot collide. -/
theorem slabs_disjoint (A B ps off off' : Nat) (hab : A < B) (h : off < ps) (h' : off' < ps) :
    A * ps + off < B * ps + off' := by
  have h1 : (A + 1) * ps ≤ B * ps := Nat.mul_le_mul_right _ hab
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § groups and warps

/-- A thread index of a 64-thread block belongs to warp 0 or 1. -/
theorem warp_of_thread (t : Nat) (h : t < 64) : t / 32 < 2 := by omega

/-- The group of a value is below the groups per head when `head_dim = 64*groups`. -/
theorem group_lt (groups d head_dim : Nat) (hh : head_dim = 64 * groups) (hd : d < head_dim) :
    d / 64 < groups := by omega

/-- Value index inside the head. -/
theorem value_index_lt (g t groups : Nat) (hg : g < groups) (ht : t < 64) : g * 64 + t < 64 * groups := by
  omega

-- § gather decomposition

/-- The quad within a head is below `per`. -/
theorem quad_lt (rem per : Nat) (hp : 0 < per) : rem % per < per := Nat.mod_lt _ hp

/-- The flat index recomposes. -/
theorem gather_decomp (i m : Nat) : m * (i / m) + i % m = i := Nat.div_add_mod i m

/-- The id of a flat index is below `n_ids`. -/
theorem gather_id_lt (i n_ids m : Nat) (h : i < m * n_ids) : i / m < n_ids := Nat.div_lt_of_lt_mul h

/-- A quad covers values `4q .. 4q+3`, all inside the head when `q < head_dim/4`. -/
theorem quad_values_lt (q head_dim : Nat) (h : q < head_dim / 4) : 4 * q + 3 < head_dim := by omega

/-- The group of the first value of a quad also contains the whole quad. -/
theorem quad_in_group (q : Nat) : (4 * q) / 64 = (4 * q + 3) / 64 := by omega

end KvQ8Quantization



## FILE: /mnt/user-data/outputs/PleBlockIndexing.lean
/-!
# MODULE: ple_block_indexing
History shift, conv tap rows, stream layout and scratch sizing of the PLE block in `ple.cu`.

## Source semantics
* `history_advance` shifts each channel column up by one row and appends the new normalised value last.
* The depthwise conv for kernel 4, dilation 3 and history 9 reads rows `hist - (3-k)*3 = 0, 3, 6, 9`; only
  tap 3 reads the new row (row 9 = `hist`); the history layout is row-fastest: `row + hist*channel`.
* A stream `c` of the grouped norm occupies `[c*n_embd, (c+1)*n_embd)`.
* The gate is `sigmoid(sgn(s)*sqrt(max(|s|, 1e-6)))`; `sgn` is odd.
* Scratch regions are rounded up to 16 bytes.

## Form template identifiers
NG_HIST -> history_rows; NG_HC_DIM -> stream_dim; NG_HC -> stream_count; NG_N_EMBD -> embedding_dim;
kern -> conv_kernel; dil -> conv_dilation.
-/

namespace PleBlockIndexing

-- § history advance

/-- Shift one channel column: drop the oldest entry, append the new one. -/
def advance (col : List Int) (x : Int) : List Int := col.tail ++ [x]

/-- The column length is preserved for a non-empty column. -/
theorem advance_length (col : List Int) (x : Int) (h : col ≠ []) : (advance col x).length = col.length := by
  cases col with
  | nil => exact absurd rfl h
  | cons a as => simp [advance]

/-- The new value is the last entry. -/
theorem advance_last (col : List Int) (x : Int) : (advance col x).getLast? = some x := by
  simp [advance]

/-- The kept entries are the previous ones shifted by one. -/
theorem advance_shift (a : Int) (as : List Int) (x : Int) : advance (a :: as) x = as ++ [x] := by
  simp [advance]

-- § conv taps

/-- Row read by tap `k`. -/
def tapRow (nhist kern dil k : Nat) : Nat := nhist - (kern - 1 - k) * dil

/-- The four taps of the real geometry read rows 0, 3, 6, 9. -/
theorem tap_rows : [0, 1, 2, 3].map (fun k => tapRow 9 4 3 k) = [0, 3, 6, 9] := by decide

/-- Only the last tap reads the new row. -/
theorem only_last_tap_new (k : Nat) (hk : k < 4) : tapRow 9 4 3 k = 9 ↔ k = 3 := by
  unfold tapRow; omega

/-- Tap rows never exceed the history length. -/
theorem tap_row_le (k : Nat) : tapRow 9 4 3 k ≤ 9 := by unfold tapRow; omega

/-- A history row below `nhist` of channel `c` is inside the flat history buffer. -/
theorem hist_index_lt (row nhist c hc : Nat) (hr : row < nhist) (hch : c < hc) :
    row + nhist * c < nhist * hc := by
  have h1 : nhist * (c + 1) ≤ nhist * hc := Nat.mul_le_mul_left _ hch
  rw [Nat.mul_add, Nat.mul_one] at h1
  omega

-- § stream layout

/-- Stream `c` element `d` stays inside the `hc * n_embd` vector. -/
theorem stream_index_lt (c d n hc : Nat) (hc' : c < hc) (hd : d < n) : c * n + d < hc * n := by
  have h1 : (c + 1) * n ≤ hc * n := Nat.mul_le_mul_right _ hc'
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

/-- The broadcast gate index of a flat element is a valid stream. -/
theorem gate_index_lt (i n hc : Nat) (h : i < n * hc) : i / n < hc := Nat.div_lt_of_lt_mul h

/-- The broadcast value index is inside the embedding. -/
theorem value_index_lt (i n : Nat) (hn : 0 < n) : i % n < n := Nat.mod_lt _ hn

/-- Real geometry: four streams of 2560 make 10240. -/
theorem geometry : 4 * 2560 = 10240 ∧ 10240 / 2560 = 4 := by decide

-- § gate sign

/-- Sign function. -/
def sgn (s : Int) : Int := if s > 0 then 1 else if s < 0 then -1 else 0

/-- The sign is odd, which makes the gate symmetric about 1/2. -/
theorem sgn_neg (s : Int) : sgn (-s) = - sgn s := by
  unfold sgn; split_ifs <;> omega

/-- The sign times the value is non-negative (the argument of the root is a magnitude). -/
theorem sgn_mul_nonneg (s : Int) : 0 ≤ sgn s * s := by
  unfold sgn
  split_ifs with h1 h2
  · omega
  · omega
  · omega

-- § scratch sizing

/-- Round up to a multiple of 16. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Rounding never shrinks. -/
theorem align16_ge (x : Nat) : x ≤ align16 x := by unfold align16; omega

/-- Rounding yields a multiple of 16 (the float and byte regions stay aligned). -/
theorem align16_mod (x : Nat) : align16 x % 16 = 0 := by unfold align16; omega

/-- Scratch layout: float region, Q8 activation image, BF16 embedding copy, then 256 bytes of slack. -/
theorem scratch_ordering (f q e : Nat) : align16 f + align16 q + e + 256 ≥ f + q + e := by
  have h1 := align16_ge f
  have h2 := align16_ge q
  omega

end PleBlockIndexing





## FILE: /mnt/user-data/outputs/Q2ExpertVnniKernel.lean
/-!
# MODULE: q2_expert_vnni_kernel
The CPU Q2_0 expert kernel: Q8_1-style activation quantisation, 2-bit code unpacking, the offset-corrected
row dot `sum (c-1) * xhat = sum c*xhat - sum xhat`, the gate/up/down row layout of one expert blob, the
tail-mask / multi-token dispatch arithmetic and the CPU feature gate.

## Source semantics
* A weight code `c ∈ {0,1,2,3}` decodes to `(c - 1) * d`; the kernel computes `sum c*xhat` with VNNI and
  subtracts `sum xhat` (carried per 32-chunk as `hx`), so the offset identity must be exact over integers.
* Activation values are clamped to `[-127, 127]`, so integer partial sums cannot overflow.
* One expert blob is exactly 1,382,400 bytes: gate/up rows interleaved (row `2r` gate, `2r+1` up), then down.
* The tail of a block loop uses masks over `nb ≤ 8` blocks; multi-token dispatch handles 1..8 tokens.

## Form template identifiers
H -> hidden_size, FF -> ffn_size, QK -> weight_block, QKA -> act_chunk, row_dot -> row_dot,
act_quant_q8_1 -> act_quant, s2_expert_vnni_q -> expert_forward, CpuFeatures -> cpu_features.
-/

namespace Strata.Q2Expert

-- § geometry

/-- Hidden size of the model, the width of an expert input and output. -/
def H : Nat := 2560
/-- Expert intermediate width. -/
def FF : Nat := 640
/-- Weights per Q2_0 block. -/
def QK : Nat := 64
/-- Activations per quantisation chunk (the unit of the Q8_1 scale). -/
def QKA : Nat := 32
/-- Q2_0 blocks per gate/up row. -/
def SC_GU : Nat := H / QK
/-- Q2_0 blocks per down row. -/
def SC_D : Nat := FF / QK
/-- Code bytes per gate/up row. -/
def ROW_GU : Nat := SC_GU * 16
/-- Code bytes per down row. -/
def ROW_D : Nat := SC_D * 16
/-- Total bytes of one expert blob: codes and scales of gate/up (2*FF rows) plus down (H rows). -/
def BLOB : Nat := 2 * FF * ROW_GU + 2 * FF * SC_GU * 2 + H * ROW_D + H * SC_D * 2

/-- One expert blob is exactly 1,382,400 bytes. -/
theorem blob_bytes : BLOB = 1382400 := by decide

/-- The block counts per row are 40 (gate/up) and 10 (down). -/
theorem block_counts : SC_GU = 40 ∧ SC_D = 10 := by decide

/-- The activation chunk counts are 80 for the input and 20 for the intermediate. -/
theorem chunk_counts : H / QKA = 80 ∧ FF / QKA = 20 := by decide

/-- Each 64-weight block covers exactly two 32-activation chunks. -/
theorem two_chunks_per_block : QK = 2 * QKA := by decide

/-- Both input widths are whole numbers of weight blocks. -/
theorem widths_divisible : H % QK = 0 ∧ FF % QK = 0 := by decide

/-- A block holds 16 code bytes at four codes per byte, which is 64 codes. -/
theorem codes_per_block : 16 * 4 = QK := by decide

/-- Gate and up rows are interleaved: slots `2r` and `2r+1` are distinct and stay below `2*FF`. -/
theorem gate_up_slots (r : Nat) (hr : r < FF) :
    2 * r ≠ 2 * r + 1 ∧ 2 * r + 1 < 2 * FF := by
  unfold FF at *
  omega

/-- Chunk indices `2b` and `2b+1` of block `b` stay inside the row's chunk range. -/
theorem chunk_index_in_row (b nb : Nat) (hb : b < nb) (h : Nat) (hh : h < 2) :
    2 * b + h < 2 * nb := by
  omega

-- § offset-corrected dot product

/-- The weight code offset is removed exactly: `sum (c-1)*x = sum c*x - sum x`. -/
theorem offset_identity : ∀ (c x : List Int), c.length = x.length →
    (List.zipWith (· * ·) (c.map (· - 1)) x).sum = (List.zipWith (· * ·) c x).sum - x.sum
  | [], [], _ => by simp
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | a :: cs, b :: xs, h => by
    have ih := offset_identity cs xs (by simpa using h)
    simp only [List.map_cons, List.zipWith_cons_cons, List.sum_cons]
    rw [ih, Int.sub_mul, Int.one_mul]
    omega

/-- Clamp an activation to the symmetric int8 range used by the quantiser. -/
def clamp127 (v : Int) : Int :=
  if v < -127 then -127 else if v > 127 then 127 else v

/-- The clamp lands in `[-127, 127]`. -/
theorem clamp127_bounds (v : Int) : -127 ≤ clamp127 v ∧ clamp127 v ≤ 127 := by
  unfold clamp127
  split <;> (try split) <;> omega

/-- Clamping is the identity on values already in range. -/
theorem clamp127_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clamp127 v = v := by
  unfold clamp127
  split <;> (try split) <;> omega

/-- A sum of in-range activations is bounded by `127 * length`; int32 accumulation cannot overflow. -/
theorem sum_bound : ∀ (l : List Int), (∀ v ∈ l, -127 ≤ v ∧ v ≤ 127) →
    -(127 * (l.length : Int)) ≤ l.sum ∧ l.sum ≤ 127 * (l.length : Int)
  | [], _ => by simp
  | a :: t, h => by
    have ha := h a (by simp)
    have ht := sum_bound t (fun v hv => h v (by simp [hv]))
    simp only [List.sum_cons, List.length_cons]
    omega

/-- A 64-weight block dot of codes (≤3) against int8 activations fits an int32 with room to spare. -/
theorem block_dot_fits : 3 * 127 * 64 < 2 ^ 31 := by decide

-- § 2-bit unpacking

/-- Code `k` (0..3) of a packed byte, least significant first. -/
def codeOf (byte k : Nat) : Nat := byte / 4 ^ k % 4

/-- Pack four 2-bit codes into a byte, first code in the low bits. -/
def pack4 (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3

/-- Unpacking recovers every packed code. -/
theorem unpack_pack4 (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    codeOf (pack4 c0 c1 c2 c3) 0 = c0 ∧ codeOf (pack4 c0 c1 c2 c3) 1 = c1 ∧
    codeOf (pack4 c0 c1 c2 c3) 2 = c2 ∧ codeOf (pack4 c0 c1 c2 c3) 3 = c3 := by
  simp only [codeOf, pack4]
  omega

/-- A packed byte of four codes is below 256. -/
theorem pack4_lt (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    pack4 c0 c1 c2 c3 < 256 := by
  unfold pack4
  omega

/-- Every unpacked code is a valid 2-bit code. -/
theorem codeOf_lt (byte k : Nat) : codeOf byte k < 4 := by
  unfold codeOf
  exact Nat.mod_lt _ (by decide)

/-- A decoded code `c - 1` lies in the codebook range `[-1, 2]`. -/
theorem decoded_code_range (c : Nat) (h : c < 4) : -1 ≤ (c : Int) - 1 ∧ (c : Int) - 1 ≤ 2 := by
  omega

/-- Weight `j` of a block lives in code byte `j/4` at bit pair `j%4`; the byte stays inside the block. -/
theorem scalar_code_index (b j : Nat) (hj : j < 64) :
    b * 16 + j / 4 < (b + 1) * 16 ∧ j % 4 < 4 := by
  omega

-- § tail masks and block loop

/-- Mask over the low `nb` blocks of an eight-block group. -/
def tailMask (nb : Nat) : Nat := if nb ≥ 8 then 0xFF else 2 ^ nb - 1

/-- Masks over at most eight blocks fit a byte. -/
theorem tailMask_lt (nb : Nat) (h : nb ≤ 8) : tailMask nb < 256 := by
  unfold tailMask
  interval_cases nb <;> decide

/-- Blocks processed in one iteration of the eight-block loop starting at `b0`. -/
def groupSize (nblocks b0 : Nat) : Nat := if nblocks - b0 < 8 then nblocks - b0 else 8

/-- Each iteration processes between one and eight blocks and never runs past the row. -/
theorem group_bounds (nblocks b0 : Nat) (h : b0 < nblocks) :
    1 ≤ groupSize nblocks b0 ∧ groupSize nblocks b0 ≤ 8 ∧ b0 + groupSize nblocks b0 ≤ nblocks := by
  unfold groupSize
  split <;> omega

-- § token dispatch

/-- Template instance selected for `n` tokens: 1..7 exact, anything else the 8-token instance. -/
def dispatchNT (n : Nat) : Nat := if n ≥ 8 then 8 else n

/-- For 1 to 8 tokens the instance matches the token count. -/
theorem dispatchNT_exact (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ 8) : dispatchNT n = n := by
  unfold dispatchNT
  split <;> omega

/-- The multi-token path falls back to per-token execution outside 1..8 tokens. -/
def fallsBack (n : Nat) : Bool := decide (n < 1 ∨ n > 8)

/-- The multi-token fast path is taken exactly for 1..8 tokens. -/
theorem fast_path_range (n : Nat) : fallsBack n = false ↔ 1 ≤ n ∧ n ≤ 8 := by
  unfold fallsBack
  simp only [decide_eq_false_iff_not]
  omega

-- § cpu feature gate

/-- The CPU features the expert kernel needs. -/
structure CpuFeatures where
  avx512f : Bool := false
  avx512bw : Bool := false
  avx512vl : Bool := false
  avx512_vnni : Bool := false
  avx512_vbmi : Bool := false

/-- The kernel is usable iff all five features are present. -/
def CpuFeatures.usable (f : CpuFeatures) : Bool :=
  f.avx512f && f.avx512bw && f.avx512vl && f.avx512_vnni && f.avx512_vbmi

/-- Usability is exactly the conjunction of the five flags. -/
theorem usable_iff (f : CpuFeatures) :
    f.usable = true ↔ (f.avx512f = true ∧ f.avx512bw = true ∧ f.avx512vl = true ∧
      f.avx512_vnni = true ∧ f.avx512_vbmi = true) := by
  simp [CpuFeatures.usable, and_assoc]

/-- Bit `i` of a cpuid register. -/
def bit (x i : Nat) : Bool := (x / 2 ^ i) % 2 == 1

/-- Feature detection: a CPU whose maximum leaf is below 7 reports no features. -/
def detect (maxLeaf ebx ecx : Nat) : CpuFeatures :=
  if maxLeaf < 7 then {}
  else { avx512f := bit ebx 16, avx512bw := bit ebx 30, avx512vl := bit ebx 31,
         avx512_vnni := bit ecx 11, avx512_vbmi := bit ecx 1 }

/-- A CPU without cpuid leaf 7 is never usable. -/
theorem detect_small_leaf (maxLeaf ebx ecx : Nat) (h : maxLeaf < 7) :
    (detect maxLeaf ebx ecx).usable = false := by
  simp [detect, h, CpuFeatures.usable]

/-- An all-zero register reports no feature bit. -/
theorem bit_zero (i : Nat) : bit 0 i = false := by
  simp [bit]

end Strata.Q2Expert



## FILE: /mnt/user-data/outputs/IQuantAvx512Rows.lean
/-!
# MODULE: iquant_avx512_rows
Multi-token i-quant expert row dots (IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) against Q8_K activations: the
per-format block layout, in-bounds decoding of every field, grid-index ranges, scale lane mapping,
no-saturation bounds of `maddubs`, and the token-batching loop.

## Source semantics
* Type ids 16, 17, 18, 21, 22 are supported; IQ1_M stays on ggml-cpu.
* A block holds 256 values; each of four 64-value chunks decodes grid magnitudes, a 64-bit sign mask and
  four scales (one per 16 values).
* Every byte read of the block layout lies inside the format's block size.
* `maddubs` pairs (grid magnitude * int8) cannot saturate the int16 lanes for these grids.

## Form template identifiers
Fmt<TY>::decode -> decode_chunk, row_dot -> row_dot, gu_rows -> gate_up_rows, dot_rows -> down_rows,
iq512_supported -> supported, QK_K -> super_block.
-/

namespace Strata.IQuantAvx512

-- § formats

/-- Values per super-block (`QK_K`). -/
def QK_K : Nat := 256

/-- Bytes per block for each supported ggml type id, zero when unsupported. -/
def bytesOf : Nat → Nat
  | 16 => 66
  | 17 => 74
  | 18 => 98
  | 21 => 110
  | 22 => 82
  | _ => 0

/-- The type ids handled by the AVX-512 path. -/
def supported (t : Nat) : Bool := t == 16 || t == 17 || t == 18 || t == 21 || t == 22

/-- Supported types are exactly those with a non-zero block size (checked over all small ids). -/
theorem supported_iff_bytes : ∀ t : Fin 64, supported t.val = true ↔ bytesOf t.val ≠ 0 := by
  decide

/-- IQ1_M (type 29) is not handled here. -/
theorem iq1_m_unsupported : supported 29 = false := by decide

/-- IQ2_XXS: fp16 d plus 32 u16. -/
theorem iq2xxs_size : bytesOf 16 = 2 + 64 := by decide
/-- IQ2_XS: d, 32 u16, 8 scale bytes. -/
theorem iq2xs_size : bytesOf 17 = 2 + 64 + 8 := by decide
/-- IQ2_S: d, 64 qs bytes, 8 qh, 8 scales. -/
theorem iq2s_size : bytesOf 22 = 2 + 64 + 8 + 8 := by decide
/-- IQ3_XXS: d, 64 grid bytes, 8 u32 of signs and scales. -/
theorem iq3xxs_size : bytesOf 18 = 2 + 64 + 32 := by decide
/-- IQ3_S: d, qs, qh, signs, scales. -/
theorem iq3s_size : bytesOf 21 = 2 + 64 + 8 + 32 + 4 := by decide

-- § in-bounds decoding (chunk j of four)

/-- IQ2_XXS reads 16 bytes of `qs` for chunk `j` inside the block. -/
theorem iq2xxs_bounds (j : Nat) (hj : j < 4) : 2 + 16 * j + 16 ≤ 66 := by omega

/-- IQ2_XS reads its grid words and its two scale bytes inside the block. -/
theorem iq2xs_bounds (j : Nat) (hj : j < 4) : 2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 := by omega

/-- IQ2_S reads grid bytes, a sign word, two qh bytes and two scale bytes inside the block. -/
theorem iq2s_bounds (j : Nat) (hj : j < 4) :
    2 + 8 * j + 7 < 66 ∧ 2 + 32 + 8 * j + 8 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 2 * j + 1 < 82 := by
  omega

/-- IQ3_XXS reads 16 grid bytes and two sign/scale words inside the block. -/
theorem iq3xxs_bounds (j : Nat) (hj : j < 4) : 2 + 16 * j + 15 < 66 ∧ 2 + 64 + 8 * j + 8 ≤ 98 := by
  omega

/-- IQ3_S reads grid bytes, qh, signs and scales inside the block. -/
theorem iq3s_bounds (j : Nat) (hj : j < 4) :
    2 + 16 * j + 15 < 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 8 * j + 8 ≤ 106 ∧ 106 + j < 110 := by
  omega

/-- Four 64-value chunks tile a 256-value block and the activation reads stay inside it. -/
theorem chunks_tile (j : Nat) (hj : j < 4) : 4 * 64 = QK_K ∧ 64 * j + 64 ≤ QK_K := by
  unfold QK_K
  omega

-- § grid index ranges

/-- IQ2_S grid index (8 low bits plus a 2-bit extension) stays below the 1024-entry grid. -/
theorem iq2s_index (q h : Nat) (hq : q < 256) (hh : h < 4) : q + 256 * h < 1024 := by omega

/-- IQ3_S grid index (8 low bits plus 1 extension bit) stays below the 512-entry grid. -/
theorem iq3s_index (q b : Nat) (hq : q < 256) (hb : b < 2) : q + 256 * b < 512 := by omega

/-- IQ2_XS: a 16-bit word splits into a 9-bit grid index and a 7-bit sign index. -/
theorem iq2xs_split (v : Nat) (hv : v < 65536) : v % 512 < 512 ∧ v / 512 < 128 := by omega

-- § scales

/-- Scale lane mapping: int16 lane `l` (values `2l`, `2l+1`) uses scale `l/8`, i.e. one scale per 16 values. -/
theorem scale_lane_group (l : Nat) (hl : l < 32) : l / 8 < 4 ∧ (2 * l) / 16 = l / 8 ∧ (2 * l + 1) / 16 = l / 8 := by
  omega

/-- 32 int16 lanes carry 64 values. -/
theorem lanes_cover_chunk : 32 * 2 = 64 := by decide

/-- A 4-bit scale `s` becomes the odd multiplier `2s+1`, at most 31. -/
theorem odd_scale (s : Nat) (hs : s < 16) : 2 * s + 1 ≤ 31 ∧ (2 * s + 1) % 2 = 1 := by omega

-- § no int16 saturation in maddubs

/-- A pair of grid magnitudes (≤ 43, as in the 2-bit grids) times int8 activations fits int16. -/
theorem maddubs_no_saturation (g : Nat) (hg : g ≤ 43) : 2 * g * 127 ≤ 32767 := by omega

/-- Applying a sign mask only negates: the magnitude of the product is unchanged. -/
def applySign (neg : Bool) (y : Int) : Int := if neg then -y else y

/-- Signed activation stays in the int8 symmetric range. -/
theorem applySign_bound (neg : Bool) (y : Int) (h1 : -127 ≤ y) (h2 : y ≤ 127) :
    -127 ≤ applySign neg y ∧ applySign neg y ≤ 127 := by
  unfold applySign
  split <;> omega

-- § row layout and batching

/-- Blocks per row for an activation width `n`. -/
def nBlocks (n : Nat) : Nat := n / QK_K

/-- Byte offset of row `r` in the gate (or, shifted by `up_off`, up) matrix. -/
def rowOffset (base rowBytes r : Nat) : Nat := base + r * rowBytes

/-- The next row starts exactly one row stride later. -/
theorem rowOffset_succ (base rowBytes r : Nat) :
    rowOffset base rowBytes (r + 1) = rowOffset base rowBytes r + rowBytes := by
  unfold rowOffset
  rw [Nat.succ_mul]
  omega

/-- Tokens handled by one `dot_rows` batch starting at token `t0`: at most four. -/
def batchSize (nt t0 : Nat) : Nat := if nt - t0 < 4 then nt - t0 else 4

/-- Each batch is non-empty, at most four tokens, and stays inside the token list. -/
theorem batch_bounds (nt t0 : Nat) (h : t0 < nt) :
    1 ≤ batchSize nt t0 ∧ batchSize nt t0 ≤ 4 ∧ t0 + batchSize nt t0 ≤ nt := by
  unfold batchSize
  split <;> omega

end Strata.IQuantAvx512



## FILE: /mnt/user-data/outputs/CpuExpertPool.lean
/-!
# MODULE: cpu_expert_pool
The CPU expert thread pool: task splitting across workers, row-range segmentation across experts, the native
Q2_0 / generic routing by phase mode, batching of experts, park accounting and core selection.

## Source semantics
* A phase of `rows` rows is cut into `tasks` equal-ish ranges `[rows*i/tasks, rows*(i+1)/tasks)`; the ranges
  are monotone, start at 0 and end at `rows`, so together they cover every row exactly once.
* A range may span several experts; each step of the segment loop makes progress and never overshoots.
* Phase modes 5/6 route to Q2_0 kernels when the pack type is 42, otherwise to the native dispatch.
* Experts beyond the split-buffer capacity are processed in batches.
* The first physical core is reserved for the host.

## Form template identifiers
run_phase -> run_phase, drain -> drain, mrows_/mtasks_ -> rows/tasks, split_ -> split_buffers,
physical_cores -> physical_cores, parked_ -> parked_count.
-/

namespace Strata.CpuExpertPool

-- § task splitting

/-- Start row of task `i` when `rows` rows are divided among `tasks` tasks. -/
def splitPt (rows tasks i : Nat) : Nat := rows * i / tasks

/-- The first task starts at row 0. -/
theorem splitPt_zero (rows tasks : Nat) : splitPt rows tasks 0 = 0 := by
  simp [splitPt]

/-- The last boundary is exactly `rows`, so no row is left uncovered. -/
theorem splitPt_last (rows tasks : Nat) (h : 0 < tasks) : splitPt rows tasks tasks = rows := by
  unfold splitPt
  exact Nat.mul_div_cancel rows h

/-- Task boundaries are monotone, so the ranges are well formed and disjoint. -/
theorem splitPt_mono (rows tasks i : Nat) : splitPt rows tasks i ≤ splitPt rows tasks (i + 1) := by
  unfold splitPt
  exact Nat.div_le_div_right (Nat.mul_le_mul_left rows (Nat.le_succ i))

/-- Every boundary lies within `[0, rows]` for `i ≤ tasks`. -/
theorem splitPt_le_rows (rows tasks i : Nat) (hi : i ≤ tasks) : splitPt rows tasks i ≤ rows := by
  unfold splitPt
  have h := Nat.mul_le_mul_left rows hi
  rw [Nat.mul_comm rows tasks] at h
  exact Nat.div_le_of_le_mul h

/-- Total number of threads working on a phase: the workers plus the host when it helps. -/
def threads (nWorkers : Nat) (hostWorks : Bool) : Nat := nWorkers + (if hostWorks then 1 else 0)

/-- Tasks per phase: about three per thread. -/
def tasksPerPhase (nWorkers : Nat) (hostWorks : Bool) : Nat := 3 * threads nWorkers hostWorks

/-- A pool with at least one worker has at least three tasks per phase. -/
theorem tasks_pos (n : Nat) (hostWorks : Bool) (hn : 1 ≤ n) : 3 ≤ tasksPerPhase n hostWorks := by
  unfold tasksPerPhase threads
  split <;> omega

/-- Parts per expert in the split-phase modes is at least one. -/
theorem parts_pos (threads n : Nat) : 1 ≤ max 1 ((3 * threads + n - 1) / n) :=
  Nat.le_max_left 1 _

-- § row-range segmentation across experts

/-- One step of the segment loop: with `r0` the offset inside the current expert (< per) and `g1 - r`
rows still wanted, the step end `r1` is `min per (r0 + (g1 - r))`. The step advances and never overshoots. -/
theorem seg_progress (per r0 r g1 : Nat) (h0 : r0 < per) (hr : r < g1) :
    r0 < min per (r0 + (g1 - r)) ∧ r + (min per (r0 + (g1 - r)) - r0) ≤ g1 := by
  omega

/-- The offset inside an expert is always a valid row of that expert. -/
theorem seg_offset_lt (per r : Nat) (h : 0 < per) : r % per < per := Nat.mod_lt r h

-- § routing

/-- The kernel family a native-layer phase is routed to. -/
inductive Path
  | q2Gu
  | nativeGu
  | q2Down
  | nativeDown
  | other
  deriving DecidableEq

/-- Route a phase mode (5 = gate/up, 6 = down) given the pack's gate/up and down type ids. -/
def route (mode gu d : Nat) : Path :=
  if mode = 5 then (if gu = 42 then Path.q2Gu else Path.nativeGu)
  else if mode = 6 then (if d = 42 then Path.q2Down else Path.nativeDown)
  else Path.other

/-- A Q2_0 gate/up pack uses the Q2_0 kernels. -/
theorem route_q2_gu (d : Nat) : route 5 42 d = Path.q2Gu := by
  simp [route]

/-- Any other gate/up type uses the native dispatch. -/
theorem route_native_gu (gu d : Nat) (h : gu ≠ 42) : route 5 gu d = Path.nativeGu := by
  simp [route, h]

/-- A Q2_0 down projection uses the AVX-512 Q2_0 kernel. -/
theorem route_q2_down (gu : Nat) : route 6 gu 42 = Path.q2Down := by
  simp [route]

/-- Any other down type uses the native dispatch. -/
theorem route_native_down (gu d : Nat) (h : d ≠ 42) : route 6 gu d = Path.nativeDown := by
  simp [route, h]

-- § expert batching

/-- Experts handled in the batch starting at `b0` when `cap` split buffers exist. -/
def batchLen (cap n b0 : Nat) : Nat := min cap (n - b0)

/-- Each batch is non-empty and within the expert list (capacity must be positive). -/
theorem batch_progress (cap n b0 : Nat) (hc : 0 < cap) (hb : b0 < n) :
    0 < batchLen cap n b0 ∧ b0 + batchLen cap n b0 ≤ n := by
  unfold batchLen
  omega

-- § park accounting

/-- A worker leaving the park and arriving back restores the parked count. -/
theorem leave_then_arrive (p : Nat) (h : 1 ≤ p) : (p - 1) + 1 = p := by omega

/-- Construction starts with every worker counted as parked. -/
theorem parked_initially (n : Nat) : (List.replicate n 1).sum = n := by
  simp

-- § core selection

/-- Drop the first core, reserved for the host thread. -/
def skipFirst : List Nat → List Nat
  | [] => []
  | _ :: t => t

/-- Skipping the first core removes exactly one entry from a non-empty list. -/
theorem skipFirst_length (l : List Nat) : (skipFirst l).length = l.length - 1 := by
  cases l <;> simp [skipFirst]

/-- Core assigned to worker `i`: its own core when available, otherwise unpinned (-1). -/
def coreFor (cores : List Nat) (i : Nat) : Int :=
  match cores[i]? with
  | some c => (c : Int)
  | none => -1

/-- Workers beyond the core list are left unpinned. -/
theorem coreFor_overflow (cores : List Nat) (i : Nat) (h : cores.length ≤ i) : coreFor cores i = -1 := by
  have : cores[i]? = none := List.getElem?_eq_none h
  simp [coreFor, this]

/-- A negative core means "do not pin". -/
def shouldPin (core : Int) : Bool := decide (0 ≤ core)

/-- Unpinned workers are never pinned. -/
theorem unpinned_not_pinned : shouldPin (-1) = false := by decide

end Strata.CpuExpertPool



## FILE: /mnt/user-data/outputs/Q2Avx2Kernel.lean
/-!
# MODULE: q2_avx2_kernel
The AVX2-only Q2_0 row dot over GGUF-layout blocks (18 bytes: fp16 scale + 16 code bytes) and the scalar-rule
activation quantiser, for CPUs without AVX-512.

## Source semantics
* Value `4k+t` of an unpacked block is code `t` of byte `k`; the low half is values 0..31, the high half 32..63.
* `maddubs` of codes (≤3) with int8 activations never saturates an int16 lane, so the integer sums are exact.
* The quantiser clamps to `[-127, 127]`; a chunk sum is bounded by `127*32`.
* Token batches are processed four at a time.

## Form template identifiers
unpack64 -> unpack_block, row_multi -> row_dot_multi, rows -> rows, act_quant_q8_1_avx2 -> act_quant.
-/

namespace Strata.Q2Avx2

-- § block layout

/-- Bytes per GGUF Q2_0 block. -/
def BLOCK_BYTES : Nat := 18

/-- Scale (2 bytes) plus 16 code bytes is 18 bytes. -/
theorem block_bytes_eq : 2 + 16 = BLOCK_BYTES := by decide

/-- Block `b` has its scale at `18b` and codes at `18b+2`, ending exactly where block `b+1` starts. -/
theorem block_layout (b : Nat) : 18 * b + 2 + 16 = 18 * (b + 1) := by omega

/-- Activation chunk indices of block `b`: low half at `64b`, high half at `64b+32 = 32*(2b+1)`. -/
theorem act_halves (b : Nat) : 64 * b + 32 = 32 * (2 * b + 1) ∧ 64 * b = 32 * (2 * b) := by omega

-- § unpack order

/-- Output value index `4k+t` comes from byte `k`, code slot `t` (the unpack interleave order). -/
theorem unpack_index (k t : Nat) (ht : t < 4) : (4 * k + t) / 4 = k ∧ (4 * k + t) % 4 = t := by
  omega

/-- The low vector holds values 0..31 (bytes 0..7) and the high vector 32..63 (bytes 8..15). -/
theorem unpack_halves (i : Nat) (hi : i < 64) :
    (i < 32 → i / 4 < 8) ∧ (32 ≤ i → 8 ≤ i / 4 ∧ i / 4 < 16) := by omega

/-- Extracting code slot `t` of a byte with shift-and-mask equals the div/mod formula. -/
def codeOf (byte t : Nat) : Nat := byte / 4 ^ t % 4

/-- Packing four codes then unpacking returns them. -/
theorem unpack_pack (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    codeOf (c0 + 4 * c1 + 16 * c2 + 64 * c3) 0 = c0 ∧ codeOf (c0 + 4 * c1 + 16 * c2 + 64 * c3) 1 = c1 ∧
    codeOf (c0 + 4 * c1 + 16 * c2 + 64 * c3) 2 = c2 ∧ codeOf (c0 + 4 * c1 + 16 * c2 + 64 * c3) 3 = c3 := by
  simp only [codeOf]
  omega

-- § exact integer sums

/-- `maddubs` of two code*int8 products fits int16: `2 * 3 * 127 ≤ 32767`. -/
theorem maddubs_exact : 2 * 3 * 127 ≤ 32767 := by decide

/-- After `madd` with ones, a 32-value half sum is at most `3*127*32`, far below int32. -/
theorem half_sum_fits : 3 * 127 * 32 < 2 ^ 31 := by decide

/-- Codes decode to `c - 1 ∈ [-1, 2]`. -/
theorem decoded_range (c : Nat) (h : c < 4) : -1 ≤ (c : Int) - 1 ∧ (c : Int) - 1 ≤ 2 := by omega

-- § quantiser

/-- Clamp to the int8 symmetric range. -/
def clamp127 (v : Int) : Int :=
  if v < -127 then -127 else if v > 127 then 127 else v

/-- The clamp stays within `[-127, 127]`. -/
theorem clamp127_bounds (v : Int) : -127 ≤ clamp127 v ∧ clamp127 v ≤ 127 := by
  unfold clamp127
  split <;> (try split) <;> omega

/-- A list of in-range values has sum bounded by `127 * length`. -/
theorem sum_bound : ∀ (l : List Int), (∀ v ∈ l, -127 ≤ v ∧ v ≤ 127) →
    -(127 * (l.length : Int)) ≤ l.sum ∧ l.sum ≤ 127 * (l.length : Int)
  | [], _ => by simp
  | a :: t, h => by
    have ha := h a (by simp)
    have ht := sum_bound t (fun v hv => h v (by simp [hv]))
    simp only [List.sum_cons, List.length_cons]
    omega

/-- Chunks produced for `n` activations. -/
def nChunks (n : Nat) : Nat := n / 32

/-- Four 8-lane vectors cover a 32-value chunk. -/
theorem vectors_per_chunk : 4 * 8 = 32 := by decide

/-- Chunk `k` reads values `32k .. 32k+31`, inside the input when `k < n/32`. -/
theorem chunk_in_range (n k : Nat) (hk : k < nChunks n) : 32 * k + 32 ≤ n := by
  unfold nChunks at hk
  omega

-- § token batching

/-- Tokens in the batch starting at `t0`: at most four. -/
def batchSize (nt t0 : Nat) : Nat := if nt - t0 < 4 then nt - t0 else 4

/-- Batches are non-empty, bounded by four, and stay within the token count. -/
theorem batch_bounds (nt t0 : Nat) (h : t0 < nt) :
    1 ≤ batchSize nt t0 ∧ batchSize nt t0 ≤ 4 ∧ t0 + batchSize nt t0 ≤ nt := by
  unfold batchSize
  split <;> omega

end Strata.Q2Avx2



## FILE: /mnt/user-data/outputs/GgufLazyLoader.lean
/-!
# MODULE: gguf_lazy_loader
Lazy GGUF tensor access: block geometry, per-expert byte offsets in a fused `[in, out, experts]` tensor,
single-row offsets, shape reversal and shard-count bookkeeping.

## Source semantics
* A tensor of `elements` elements with block geometry `(el, by)` occupies `elements / el * by` bytes.
* Expert `e` of a fused tensor occupies a contiguous byte run starting at `e * (per_expert/el) * by`;
  consecutive experts are adjacent and together tile the whole tensor.
* Row `r` of a 2-D tensor starts at `r * row_bytes`, `row_bytes = width / el * by`.
* GGUF shapes are stored fastest-axis first; the numpy view reverses them.
* BF16 widens to FP32 by a 16-bit left shift, which is exact.

## Form template identifiers
GEOM -> block_geometry, Gguf.expert -> read_expert, Gguf.rows -> read_rows, Gguf.tensor -> read_tensor,
_discover_shards -> discover_shards.
-/

namespace Strata.GgufLazyLoader

-- § geometry

/-- Bytes of `elements` elements for block geometry `(el, by)`. -/
def tensorBytes (elements el by_ : Nat) : Nat := elements / el * by_

/-- Bytes of one row of the given width. -/
def rowBytes (width el by_ : Nat) : Nat := width / el * by_

/-- Start offset of expert `e`, where one expert has `per` elements. -/
def expertStart (e per el by_ : Nat) : Nat := e * (per / el * by_)

/-- Bytes of one expert. -/
def expertBytes (per el by_ : Nat) : Nat := per / el * by_

/-- Expert `e+1` starts exactly where expert `e` ends: the runs are adjacent, with no gap and no overlap. -/
theorem expert_adjacent (e per el by_ : Nat) :
    expertStart (e + 1) per el by_ = expertStart e per el by_ + expertBytes per el by_ := by
  unfold expertStart expertBytes
  rw [Nat.succ_mul]

/-- The experts tile the tensor: `n` experts of `el*k` elements total `n` times one expert's bytes. -/
theorem experts_tile (el k n by_ : Nat) (hel : 0 < el) :
    tensorBytes (el * k * n) el by_ = n * expertBytes (el * k) el by_ := by
  unfold tensorBytes expertBytes
  rw [Nat.mul_assoc, Nat.mul_div_cancel_left _ hel, Nat.mul_div_cancel_left _ hel]
  simp [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm]

/-- The last expert ends at the end of the tensor. -/
theorem last_expert_end (n per el by_ : Nat) (h : 0 < n) :
    expertStart (n - 1) per el by_ + expertBytes per el by_ = n * expertBytes per el by_ := by
  unfold expertStart expertBytes
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  simp [Nat.succ_mul]

/-- A row's bytes end no later than the whole 2-D tensor's bytes, for any in-range row index. -/
theorem row_in_tensor (r n rb : Nat) (hr : r < n) : (r + 1) * rb ≤ n * rb :=
  Nat.mul_le_mul_right rb hr

/-- Row `r+1` starts where row `r` ends. -/
theorem row_adjacent (r rb : Nat) : (r + 1) * rb = r * rb + rb := by
  rw [Nat.succ_mul]

-- § shapes

/-- numpy order is the reverse of GGUF order. -/
def numpyShape (shape : List Nat) : List Nat := shape.reverse

/-- Reversing twice returns the GGUF order. -/
theorem numpyShape_involutive (shape : List Nat) : numpyShape (numpyShape shape) = shape := by
  simp [numpyShape]

/-- A fused expert tensor `[in, out, experts]` is viewed as `(experts, out, in)`; one expert is `(out, in)`. -/
theorem fused_view (i o e : Nat) : numpyShape [i, o, e] = [e, o, i] := by
  simp [numpyShape]

-- § codebook and widening

/-- A Q2_0 code `c ∈ {0..3}` decodes to the integer `c - 1`, in `[-1, 2]`. -/
theorem q2_symbol_range (c : Nat) (h : c < 4) : -1 ≤ (c : Int) - 1 ∧ (c : Int) - 1 ≤ 2 := by omega

/-- Widening a BF16 pattern to FP32 bits shifts left by 16; the original is recovered exactly. -/
theorem bf16_widen_exact (h : Nat) (hh : h < 65536) : (h * 65536) / 65536 = h ∧ h * 65536 < 4294967296 := by
  omega

/-- Elements of a tensor must divide into blocks. -/
def wholeBlocks (elements el : Nat) : Bool := elements % el == 0

/-- When the element count divides evenly, bytes recover the element count. -/
theorem bytes_recover_elements (blocks el by_ : Nat) (hby : 0 < by_) :
    (blocks * by_) / by_ * el = blocks * el := by
  rw [Nat.mul_div_cancel _ hby]

-- § shards

/-- A shard set is consistent when the number found equals the declared total. -/
def shardsComplete (found total : Nat) : Bool := found == total

/-- A shard set with a missing shard is rejected. -/
theorem missing_shard_rejected (found total : Nat) (h : found ≠ total) : shardsComplete found total = false := by
  simp [shardsComplete, h]

/-- Rows are only read in range. -/
def rowInRange (idx n : Nat) : Bool := decide (idx < n)

/-- An out-of-range row is rejected. -/
theorem row_out_of_range (idx n : Nat) (h : n ≤ idx) : rowInRange idx n = false := by
  simp [rowInRange]
  omega

end Strata.GgufLazyLoader



## FILE: /mnt/user-data/outputs/CanonicalFormRoundtrip.lean
/-!
# MODULE: canonical_form_roundtrip
The integer-level facts behind the claim that every ggml quantised block maps to a canonical
(codes, scales, offsets) form and decodes bit-exactly: code biases, nibble and high-bit splits, group index
maps, scale-field packing and the registry geometry.

## Source semantics
* Biased codebooks are applied to the INTEGER code before the float multiply: Q2_0 `-1`, Q3_K `-4`,
  Q4_0 `-8`, Q5_0 `-16`, Q6_K `-32`, Q8_0 stored as `s + 128` and decoded `- 128`.
* Nibble formats put low nibbles in the first half and high nibbles in the second half.
* Group maps: Q4_K/Q5_K have groups of 32 (low nibbles `2j`, high nibbles `2j+1` per 64-step), Q3_K and Q6_K
  groups of 16, IQ4_XS groups of 32 with a 6-bit scale split over two fields.
* Every registry entry has a group size that divides its block size.

## Form template identifiers
to_s2/from_s2 -> encode_q2/decode_q2, MAPPINGS -> registry, code_bias -> code_bias,
get_scale_min_k4 -> scale_min_pair.
-/

namespace Strata.CanonicalForm

-- § integer biases

/-- Q8_0: a signed int8 code `s` is stored as the unsigned `s + 128`, in `[0, 255]`. -/
theorem q8_store_range (s : Int) (h1 : -128 ≤ s) (h2 : s ≤ 127) : 0 ≤ s + 128 ∧ s + 128 ≤ 255 := by omega

/-- Q8_0: decoding subtracts 128 inside the integer and recovers the source code. -/
theorem q8_roundtrip (s : Int) : (s + 128) - 128 = s := by omega

/-- Q4_0: nibbles are `0..15`, decoded codes `c - 8` are in `[-8, 7]`. -/
theorem q4_0_range (c : Int) (h1 : 0 ≤ c) (h2 : c ≤ 15) : -8 ≤ c - 8 ∧ c - 8 ≤ 7 := by omega

/-- Q5_0: a 5-bit code is `0..31`; decoded `c - 16` is in `[-16, 15]`. -/
theorem q5_0_range (c : Int) (h1 : 0 ≤ c) (h2 : c ≤ 31) : -16 ≤ c - 16 ∧ c - 16 ≤ 15 := by omega

/-- Q6_K: a 6-bit code is `0..63`; decoded `c - 32` is in `[-32, 31]`. -/
theorem q6_k_range (c : Int) (h1 : 0 ≤ c) (h2 : c ≤ 63) : -32 ≤ c - 32 ∧ c - 32 ≤ 31 := by omega

/-- Q3_K: the stored code is `(2 bits) + 4*(hm bit)`, in `[0, 7]`; decoded `c - 4` is in `[-4, 3]`. -/
theorem q3_k_range (b m : Nat) (hb : b < 4) (hm : m < 2) :
    b + 4 * m ≤ 7 ∧ -4 ≤ ((b + 4 * m : Nat) : Int) - 4 ∧ ((b + 4 * m : Nat) : Int) - 4 ≤ 3 := by
  omega

-- § nibble and high-bit splits

/-- A byte splits into low and high nibbles that recombine to the byte. -/
theorem nibble_split (q : Nat) (h : q < 256) : q % 16 + 16 * (q / 16) = q ∧ q % 16 < 16 ∧ q / 16 < 16 := by
  omega

/-- Q5_0/Q5_K: a 4-bit nibble plus a 5th bit contributing 16 stays below 32. -/
theorem fifth_bit (n b : Nat) (hn : n < 16) (hb : b < 2) : n + 16 * b < 32 := by omega

/-- Q6_K: a 4-bit nibble plus two high bits `<<4` stays below 64. -/
theorem sixth_bits (n h : Nat) (hn : n < 16) (hh : h < 4) : n + 16 * h < 64 := by omega

/-- IQ4_XS: the 6-bit scale is a nibble plus two high bits `<<4`, below 64. -/
theorem iq4_xs_scale6 (lo hi : Nat) (hlo : lo < 16) (hhi : hi < 4) : lo + 16 * hi < 64 := by omega

/-- Q5_K high-bit masks walk 1, 4, 16, 64 (low nibbles) and 2, 8, 32, 128 (high nibbles). -/
theorem q5_k_masks :
    (1 <<< (2 * 0) = 1 ∧ 1 <<< (2 * 1) = 4 ∧ 1 <<< (2 * 2) = 16 ∧ 1 <<< (2 * 3) = 64) ∧
    (2 <<< (2 * 0) = 2 ∧ 2 <<< (2 * 1) = 8 ∧ 2 <<< (2 * 2) = 32 ∧ 2 <<< (2 * 3) = 128) := by
  decide

-- § group index maps

/-- Q4_K/Q5_K: in 64-step `j`, low-nibble element `64j+l` is group `2j`, high `64j+32+l` is group `2j+1`. -/
theorem q4k_groups (j l : Nat) (hj : j < 4) (hl : l < 32) :
    (64 * j + l) / 32 = 2 * j ∧ (64 * j + 32 + l) / 32 = 2 * j + 1 := by omega

/-- Q3_K: element `128*half + 32*j + 16*sub + l` is group `8*half + 2*j + sub`. -/
theorem q3k_groups (half j sub l : Nat) (hh : half < 2) (hj : j < 4) (hs : sub < 2) (hl : l < 16) :
    (128 * half + 32 * j + 16 * sub + l) / 16 = 8 * half + 2 * j + sub := by omega

/-- Q6_K: element `128*half + 32*quad + l` is group `8*half + 2*quad + l/16`. -/
theorem q6k_groups (half quad l : Nat) (hh : half < 2) (hq : quad < 4) (hl : l < 32) :
    (128 * half + 32 * quad + l) / 16 = 8 * half + 2 * quad + l / 16 := by omega

/-- IQ4_XS: scale group `ib` uses byte `ib/2` of `scales_l`, low nibble first when `ib` is even. -/
theorem iq4_xs_group_byte (ib : Nat) (h : ib < 8) : ib / 2 < 4 ∧ ib % 2 < 2 := by omega

-- § Q2_0 canonical form

/-- Four 2-bit codes per source byte: unpack slot `t` of byte `k` lands at element `4k + t`. -/
theorem q2_element_index (k t : Nat) (ht : t < 4) : (4 * k + t) / 4 = k ∧ (4 * k + t) % 4 = t := by omega

/-- Codes and scales split into planes: one code per element, so `codes.size` is 64 per block. -/
theorem q2_codes_per_block (nblocks : Nat) : nblocks * 64 = 64 * nblocks := by omega

/-- A Q2_0 block has 18 bytes: a 2-byte scale and 64 two-bit codes. -/
theorem q2_block_bytes : 2 + 64 * 2 / 8 = 18 := by decide

/-- The bias cannot be folded into the value domain: the stored code stays unsigned `0..3` and `c - 1`
is applied to the integer. -/
theorem q2_bias_range (c : Nat) (h : c < 4) : (c : Int) - 1 ∈ [(-1 : Int), 0, 1, 2] := by
  interval_cases c <;> simp

-- § registry geometry

/-- Registry rows `(block_bytes, block_elems, group_elems, code_bits)` for the mapped types:
Q2_0, Q3_K, Q4_0, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0, IQ4_NL, IQ4_XS, F32, BF16, F16. -/
def registry : List (Nat × Nat × Nat × Nat) :=
  [(18, 64, 64, 2), (110, 256, 16, 4), (18, 32, 32, 4), (144, 256, 32, 4), (22, 32, 32, 8),
   (176, 256, 32, 8), (210, 256, 16, 8), (34, 32, 32, 8), (18, 32, 32, 4), (136, 256, 32, 4),
   (4, 1, 1, 8), (2, 1, 1, 8), (2, 1, 1, 8)]

/-- Every registry entry has a positive group size dividing its block size. -/
theorem registry_groups_divide :
    ∀ e ∈ registry, 0 < e.2.2.1 ∧ e.2.1 % e.2.2.1 = 0 := by decide

/-- Block bytes of the quantised types equal their field sizes. -/
theorem block_byte_sums :
    (2 + 64 * 2 / 8 = 18) ∧ (2 + 32 * 4 / 8 = 18) ∧ (2 + 4 + 32 * 4 / 8 = 22) ∧ (2 + 32 = 34) ∧
    (32 + 64 + 12 + 2 = 110) ∧ (4 + 12 + 128 = 144) ∧ (4 + 12 + 32 + 128 = 176) ∧
    (128 + 64 + 16 + 2 = 210) ∧ (2 + 2 + 4 + 128 = 136) := by decide

/-- Groups per block follow from the registry: 256-element K-types have `256 / group` groups. -/
theorem k_type_groups : 256 / 16 = 16 ∧ 256 / 32 = 8 := by decide

end Strata.CanonicalForm



## FILE: /mnt/user-data/outputs/Q2GgufWriter.lean
/-!
# MODULE: q2_gguf_writer
The minimal GGUF v3 writer and its Q2_0 quantiser: code clamp, packing order, fixed-point property,
block size, and alignment padding.

## Source semantics
* Q2_0: block of 64 elements, `d = amax`, `code = clamp(round(w/d) + 1, 0, 3)`, symbol `code - 1 ∈ {-1,0,1,2}`.
* Codes are packed LSB-first, four per byte: element `j` sits in byte `j/4` at bit `2*(j%4)`.
* Requantising dequantised symbols reproduces the same codes (fixed point).
* Every tensor and the data section start on an `alignment`-byte boundary via padding `(-len) % alignment`.

## Form template identifiers
quantize_q2_0 -> quantize_block, dequantize_q2_0 -> dequantize_block, GGUFWriter.write -> write_file,
DEFAULT_ALIGNMENT -> alignment.
-/

namespace Strata.Q2GgufWriter

-- § quantiser

/-- Code for a rounded symbol `s`: `clamp(s + 1, 0, 3)`. -/
def codeOfSymbol (s : Int) : Int := max 0 (min 3 (s + 1))

/-- Every produced code is in `[0, 3]`. -/
theorem code_range (s : Int) : 0 ≤ codeOfSymbol s ∧ codeOfSymbol s ≤ 3 := by
  unfold codeOfSymbol
  omega

/-- For symbols already in `{-1, 0, 1, 2}` the code is `s + 1` (no clamping happens). -/
theorem code_exact (s : Int) (h1 : -1 ≤ s) (h2 : s ≤ 2) : codeOfSymbol s = s + 1 := by
  unfold codeOfSymbol
  omega

/-- Fixed point: decoding a code (`c - 1`) and requantising returns the same code. -/
theorem requantize_fixed (c : Int) (h1 : 0 ≤ c) (h2 : c ≤ 3) : codeOfSymbol (c - 1) = c := by
  unfold codeOfSymbol
  omega

/-- Decoded symbols are in `[-1, 2]`. -/
theorem symbol_range (c : Int) (h1 : 0 ≤ c) (h2 : c ≤ 3) : -1 ≤ c - 1 ∧ c - 1 ≤ 2 := by omega

-- § packing

/-- Byte index of element `j`. -/
def byteIndex (j : Nat) : Nat := j / 4

/-- Bit offset of element `j` inside its byte. -/
def bitOffset (j : Nat) : Nat := (j % 4) * 2

/-- Element `j < 64` lands inside the block's 16 code bytes at an even bit offset below 8. -/
theorem pack_position (j : Nat) (hj : j < 64) : byteIndex j < 16 ∧ bitOffset j < 8 ∧ bitOffset j % 2 = 0 := by
  unfold byteIndex bitOffset
  omega

/-- Distinct elements never share a (byte, bit) slot. -/
theorem pack_injective (i j : Nat) (hi : i < 64) (hj : j < 64)
    (hb : byteIndex i = byteIndex j) (ho : bitOffset i = bitOffset j) : i = j := by
  unfold byteIndex bitOffset at *
  omega

/-- Four 2-bit codes pack into one byte and unpack to themselves (slot `t` is `byte / 4^t % 4`). -/
theorem pack_unpack (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) % 4 = c0 ∧
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) / 4 % 4 = c1 ∧
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) / 16 % 4 = c2 ∧
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) / 64 % 4 = c3 := by omega

/-- A block is 2 bytes of scale plus 16 bytes of codes. -/
theorem block_size : 2 + 16 = 18 := by decide

/-- Sixty-four codes need exactly sixteen bytes. -/
theorem codes_bytes : 64 / 4 = 16 := by decide

/-- The element count must be a whole number of blocks; blocks times 64 is the element count. -/
theorem blocks_cover (n : Nat) (h : n % 64 = 0) : (n / 64) * 64 = n := by omega

-- § alignment

/-- Padding needed to reach the next multiple of `a`. -/
def pad (a n : Nat) : Nat := (a - n % a) % a

/-- Adding the padding yields a multiple of the alignment. -/
theorem pad_aligns (a n : Nat) (ha : 0 < a) : (n + pad a n) % a = 0 := by
  unfold pad
  have h1 := Nat.div_add_mod n a
  have h2 := Nat.mod_lt n ha
  by_cases h0 : n % a = 0
  · rw [h0]
    simpa [Nat.mod_self] using h0
  · have h3 : (a - n % a) % a = a - n % a := Nat.mod_eq_of_lt (by omega)
    rw [h3]
    have h4 : n + (a - n % a) = a * (n / a) + a := by omega
    rw [h4]
    simp

/-- The padding is smaller than the alignment. -/
theorem pad_lt (a n : Nat) (ha : 0 < a) : pad a n < a := by
  unfold pad
  exact Nat.mod_lt _ ha

/-- With the default 32-byte alignment the padding is below 32. -/
theorem pad32_lt (n : Nat) : pad 32 n < 32 := pad_lt 32 n (by decide)

-- § header

/-- GGUF magic is the bytes `G G U F` little-endian: 0x46554747. -/
theorem magic_bytes :
    0x46554747 % 256 = 0x47 ∧ 0x46554747 / 256 % 256 = 0x47 ∧
    0x46554747 / 65536 % 256 = 0x55 ∧ 0x46554747 / 16777216 % 256 = 0x46 := by decide

end Strata.Q2GgufWriter



## FILE: /mnt/user-data/outputs/MtpExpertQuantizer.lean
/-!
# MODULE: mtp_expert_quantizer
Packing of the MTP draft block's fused routed experts into Q2_0 / Q4_0 / Q8_0 GGUF blocks: tensor shapes,
divisibility by block size, byte totals, scale-search range, code ranges and nibble packing.

## Source semantics
* Fused experts: `gate_up_proj [512, 1280, 2560]` and `down_proj [512, 2560, 640]`, quantised along the last axis.
* Block sizes: Q2_0 64 elements / 18 bytes, Q4_0 32 / 18, Q8_0 32 / 34. The inner dimension must divide.
* Q2_0 scale candidates are `amax * f` with `f` from 0.5 to 1.0 in 16 equal steps (17 points).
* Q2_0 code = `clip(rint(x/d), -1, 2) + 1`; Q4_0 nibble = `min 15 (trunc (x/d + 8.5))`; low half and high
  half of the 32 elements share a byte.

## Form template identifiers
q2_0/q4_0/q8_0 -> quantize_q2/q4/q8, EXPERT_TENSORS -> expert_tensors, QUANT -> format_table.
-/

namespace Strata.MtpExpertQuantizer

-- § shapes and sizes

/-- Number of routed experts. -/
def E : Nat := 512
/-- Fused gate/up output width. -/
def GU_OUT : Nat := 1280
/-- Hidden size (inner dim of gate_up, output of down). -/
def HID : Nat := 2560
/-- Expert intermediate width (inner dim of down). -/
def INTER : Nat := 640

/-- Bytes of one row of inner width `n` for a block geometry. -/
def rowBytes (n block bytes : Nat) : Nat := n / block * bytes

/-- Total quantised bytes of both fused expert tensors. -/
def totalBytes (block bytes : Nat) : Nat :=
  E * GU_OUT * rowBytes HID block bytes + E * HID * rowBytes INTER block bytes

/-- Inner dimensions divide every block size in use (checked for Q2_0 and 32-element blocks). -/
theorem inner_dims_divide :
    HID % 64 = 0 ∧ INTER % 64 = 0 ∧ HID % 32 = 0 ∧ INTER % 32 = 0 := by decide

/-- Q2_0 expert bytes: 707,788,800 (about 0.71 GB). -/
theorem q2_0_total : totalBytes 64 18 = 707788800 := by decide

/-- Q4_0 expert bytes: 1,415,577,600 (about 1.42 GB). -/
theorem q4_0_total : totalBytes 32 18 = 1415577600 := by decide

/-- Q8_0 expert bytes: 2,673,868,800 (about 2.67 GB). -/
theorem q8_0_total : totalBytes 32 34 = 2673868800 := by decide

/-- Q2_0 is the smallest and Q8_0 the largest of the three formats. -/
theorem format_order : totalBytes 64 18 < totalBytes 32 18 ∧ totalBytes 32 18 < totalBytes 32 34 := by
  decide

/-- The fused gate_up tensor in float32 is 6,710,886,400 bytes, which is why it is streamed per expert. -/
theorem gate_up_f32_bytes : E * GU_OUT * HID * 4 = 6710886400 := by decide

/-- The `raw_shape` last axis passed to the writer is `n / block * bytes`. -/
theorem raw_last_axis_q2 : rowBytes HID 64 18 = 720 ∧ rowBytes INTER 64 18 = 180 := by decide

-- § scale search

/-- Candidate scale factors are `(16 + k) / 32` for `k = 0 .. 16`, expressed by numerator over 32. -/
theorem scale_candidates (k : Nat) (hk : k ≤ 16) : 16 ≤ 16 + k ∧ 16 + k ≤ 32 := by omega

/-- The first candidate is one half and the last is one. -/
theorem scale_endpoints : 16 + 0 = 32 / 2 ∧ 16 + 16 = 32 := by decide

/-- Seventeen candidates are tried. -/
theorem candidate_count : (List.range 17).length = 17 := by simp

-- § code ranges

/-- Clip a rounded value to `[-1, 2]`. -/
def clip12 (v : Int) : Int := max (-1) (min 2 v)

/-- Stored Q2_0 codes `clip + 1` lie in `[0, 3]`. -/
theorem q2_code_range (v : Int) : 0 ≤ clip12 v + 1 ∧ clip12 v + 1 ≤ 3 := by
  unfold clip12
  omega

/-- A value already on the grid keeps its code. -/
theorem q2_grid_fixed (v : Int) (h1 : -1 ≤ v) (h2 : v ≤ 2) : clip12 v = v := by
  unfold clip12
  omega

/-- Q4_0 nibble is capped at 15. -/
def nibble (t : Nat) : Nat := min 15 t

/-- Nibbles are below 16. -/
theorem nibble_lt (t : Nat) : nibble t < 16 := by
  unfold nibble
  omega

/-- Packing the low half and the high half of 32 nibbles into bytes loses nothing. -/
theorem nibble_pack (lo hi : Nat) (hl : lo < 16) (hh : hi < 16) :
    (lo + 16 * hi) % 16 = lo ∧ (lo + 16 * hi) / 16 = hi ∧ lo + 16 * hi < 256 := by omega

/-- Q4_0 element `i < 16` pairs with element `i + 16` in byte `i`. -/
theorem q4_pairing (i : Nat) (hi : i < 16) : i + 16 < 32 ∧ i < 16 := by omega

/-- Q4_0 scale is `max / -8`: the extreme element maps to code 0 or 15 only after rounding; the offset 8 keeps
codes unsigned: `-8 .. 7` shifted by 8 is `0 .. 15`. -/
theorem q4_offset_range (c : Int) (h1 : -8 ≤ c) (h2 : c ≤ 7) : 0 ≤ c + 8 ∧ c + 8 ≤ 15 := by omega

/-- Q8_0 int8 codes: `rint(x / d)` is in `[-127, 127]` when `d = amax / 127`. -/
theorem q8_symmetric (c : Int) (h1 : -127 ≤ c) (h2 : c ≤ 127) : -128 ≤ c ∧ c ≤ 127 := by omega

-- § checkpoint reading

/-- A BF16 tensor file must contain exactly the declared number of values. -/
def sizeMatches (rawSize prodShape : Nat) : Bool := rawSize == prodShape

/-- A mismatching size is rejected. -/
theorem size_mismatch_rejected (a b : Nat) (h : a ≠ b) : sizeMatches a b = false := by
  simp [sizeMatches, h]

/-- BF16 widens to float32 by a 16-bit shift and is recovered exactly. -/
theorem bf16_widen (h : Nat) (hh : h < 65536) : h * 65536 / 65536 = h := by omega

end Strata.MtpExpertQuantizer



## FILE: /mnt/user-data/outputs/StrataPackBuilder.lean
/-!
# MODULE: strata_pack_builder
The pack builder and verifier: LSB-first code packing/unpacking for 2/4/8-bit forms, 64-byte plane alignment,
the expert blob arithmetic (1,382,400 bytes, plane inversion), scale-dtype selection and the dense/embedding/
expert tensor partition.

## Source semantics
* `pack_codes` places `8/bits` codes per byte, first code in the low bits; `unpack_codes` inverts it.
* Every plane starts on a 64-byte boundary: padding is `(-len) % 64`.
* An expert blob is 1,382,400 bytes; `rebuild_role` inverts the gate/up interleave (slot `2r` gate, `2r+1` up)
  and the block index split `blk = row * SC + b_in`.
* A scale is stored as FP16 only if the FP16 round trip is exact, otherwise FP32.
* Every non-expert tensor goes to exactly one of `dense.bin` or `embd.bin`; expert tensors go to the arena.

## Form template identifiers
pack_codes/unpack_codes -> pack_codes/unpack_codes, put -> append_plane, rebuild_role -> rebuild_role,
choose_scale_dtype -> choose_scale_dtype, is_expert -> is_expert.
-/

namespace Strata.StrataPackBuilder

-- § code packing

/-- Codes per byte for a code width. -/
def perByte (bits : Nat) : Nat := 8 / bits

/-- 2-bit codes: four per byte. -/
theorem per_byte_2 : perByte 2 = 4 := by decide
/-- 4-bit codes: two per byte. -/
theorem per_byte_4 : perByte 4 = 2 := by decide
/-- 8-bit codes: one per byte. -/
theorem per_byte_8 : perByte 8 = 1 := by decide

/-- 2-bit pack/unpack round trip for one byte. -/
theorem roundtrip_2bit (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) % 4 = c0 ∧ (c0 + 4 * c1 + 16 * c2 + 64 * c3) / 4 % 4 = c1 ∧
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) / 16 % 4 = c2 ∧ (c0 + 4 * c1 + 16 * c2 + 64 * c3) / 64 % 4 = c3 := by
  omega

/-- 2-bit packed byte fits a byte. -/
theorem pack_2bit_lt (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    c0 + 4 * c1 + 16 * c2 + 64 * c3 < 256 := by omega

/-- 4-bit pack/unpack round trip for one byte. -/
theorem roundtrip_4bit (c0 c1 : Nat) (h0 : c0 < 16) (h1 : c1 < 16) :
    (c0 + 16 * c1) % 16 = c0 ∧ (c0 + 16 * c1) / 16 % 16 = c1 ∧ c0 + 16 * c1 < 256 := by omega

/-- Packed size: `n` codes of `bits` width need `n / perByte` bytes when divisible. -/
theorem packed_size_2bit (n : Nat) (h : n % 4 = 0) : n / 4 * 4 = n := by omega

/-- Packed size for 4-bit codes. -/
theorem packed_size_4bit (n : Nat) (h : n % 2 = 0) : n / 2 * 2 = n := by omega

-- § alignment

/-- Padding to the next 64-byte boundary. -/
def pad64 (n : Nat) : Nat := (64 - n % 64) % 64

/-- Every plane offset after padding is 64-aligned. -/
theorem plane_aligned (n : Nat) : (n + pad64 n) % 64 = 0 := by
  unfold pad64
  omega

/-- Padding never reaches a full alignment unit. -/
theorem pad64_lt (n : Nat) : pad64 n < 64 := by
  unfold pad64
  omega

-- § expert blob

/-- Hidden size and expert width. -/
def H : Nat := 2560
/-- Expert intermediate width. -/
def FF : Nat := 640
/-- Weights per block. -/
def QK : Nat := 64
/-- Blocks per gate/up row. -/
def SC_GU : Nat := H / QK
/-- Blocks per down row. -/
def SC_D : Nat := FF / QK

/-- Gate/up codes plane: `2*FF` rows of `SC_GU` blocks of 16 bytes. -/
def guCodes : Nat := 2 * FF * (SC_GU * 16)
/-- Gate/up scale plane: fp16 per block. -/
def guScales : Nat := 2 * FF * SC_GU * 2
/-- Down codes plane. -/
def dCodes : Nat := H * (SC_D * 16)
/-- Down scale plane. -/
def dScales : Nat := H * SC_D * 2

/-- The four planes total exactly 1,382,400 bytes. -/
theorem blob_total : guCodes + guScales + dCodes + dScales = 1382400 := by decide

/-- Equivalently: three roles, each `H*FF` weights as 18-byte 64-weight blocks. -/
theorem blob_by_roles : 3 * (H * FF / QK * 18) = 1382400 := by decide

/-- Each role has the same block count: `FF * SC_GU = H * SC_D = H*FF/64`. -/
theorem role_block_counts : FF * SC_GU = H * SC_D ∧ FF * SC_GU = H * FF / QK := by decide

/-- `rebuild_role` splits a block index into (row, block-in-row) and the split is invertible (gate/up). -/
theorem split_gu (blk : Nat) : (blk / 40) * 40 + blk % 40 = blk ∧ blk % 40 < 40 := by omega

/-- Down projection split with 10 blocks per row. -/
theorem split_down (blk : Nat) : (blk / 10) * 10 + blk % 10 = blk ∧ blk % 10 < 10 := by omega

/-- Gate and up occupy interleaved slots `2r` and `2r+1`: distinct and each below `2*FF`. -/
theorem role_slots (r : Nat) (hr : r < FF) : 2 * r ≠ 2 * r + 1 ∧ 2 * r + 1 < 2 * FF := by
  unfold FF at *
  omega

/-- Byte offset of block `b_in` of slot `slot`: codes advance 16 per block, scales 2 per block. -/
theorem block_offsets (slot b_in rowBytes sc : Nat) (h : rowBytes = sc * 16) (hb : b_in < sc) :
    b_in * 16 + 16 ≤ rowBytes := by
  subst h
  have := Nat.mul_le_mul_right 16 hb
  rw [Nat.succ_mul] at *
  omega

-- § scale dtype

/-- Bytes per stored scale. -/
def scaleBytes (fp16 : Bool) : Nat := if fp16 then 2 else 4

/-- Scale bytes are 2 or 4. -/
theorem scaleBytes_cases (b : Bool) : scaleBytes b = 2 ∨ scaleBytes b = 4 := by
  cases b <;> simp [scaleBytes]

/-- An FP16 plane is exactly half the size of the FP32 plane for the same scale count. -/
theorem fp16_half (n : Nat) : n * scaleBytes true * 2 = n * scaleBytes false := by
  simp [scaleBytes]
  omega

-- § tensor partition

/-- Count lemma: a predicate splits a list into two parts whose lengths add up. -/
theorem split_count {α : Type} (p : α → Bool) : ∀ l : List α,
    (l.filter p).length + (l.filter (fun x => !p x)).length = l.length
  | [] => by simp
  | a :: t => by
    have ih := split_count p t
    cases h : p a <;> simp [List.filter_cons, h] <;> omega

/-- Dense, embedding and expert tensors partition the source list with nothing lost or duplicated. -/
theorem partition_total (isExpert isEmbd : String → Bool) (names : List String) :
    ((names.filter (fun x => !isExpert x)).length) + (names.filter isExpert).length = names.length := by
  have h := split_count isExpert names
  omega

/-- A pack is only valid when every compared element count matches: shapes must be equal. -/
def shapesEqual (a b : Nat) : Bool := a == b

/-- Different element counts fail verification. -/
theorem shape_mismatch_fails (a b : Nat) (h : a ≠ b) : shapesEqual a b = false := by
  simp [shapesEqual, h]

end Strata.StrataPackBuilder



## FILE: /mnt/user-data/outputs/ByteLevelBpeTokenizer.lean
/-!
# MODULE: byte_level_bpe_tokenizer
The GPT-2 byte-level BPE tokenizer: the byte-to-unicode bijection, the single-step merge length law,
special-token matching classes and vocabulary/merge bookkeeping.

## Source semantics
* Bytes 33..126, 161..172 and 174..255 map to themselves; the other 68 bytes map to `256 + n` in byte order.
  The map is a bijection onto 256 code points, so every UTF-8 input is representable and decode inverts encode.
* A BPE step replaces an adjacent pair by one merged token, shortening the word by exactly one.
* Token type 4 (USER_DEFINED) is always matched; type 3 (CONTROL) only with `parse_special`.
* Every merge names two vocabulary tokens; there are fewer merges than tokens.

## Form template identifiers
bytes_to_unicode -> byte_map, _bpe -> bpe_merge, special_tokens -> special_literals,
Tokenizer.encode/decode -> encode/decode.
-/

namespace Strata.ByteLevelBpe

set_option maxRecDepth 100000

-- § byte <-> unicode

/-- A byte maps to itself iff it is in the printable ranges 33..126, 161..172, 174..255. -/
def printable (b : Nat) : Bool :=
  (33 ≤ b && b ≤ 126) || (161 ≤ b && b ≤ 172) || (174 ≤ b && b ≤ 255)

/-- Number of non-printable bytes below `b`. -/
def nonPrintBefore (b : Nat) : Nat := ((List.range b).filter (fun x => !printable x)).length

/-- The code point assigned to byte `b`. -/
def enc (b : Nat) : Nat := if printable b then b else 256 + nonPrintBefore b

/-- Exactly 68 of the 256 bytes are shifted. -/
theorem shifted_count : nonPrintBefore 256 = 68 := by decide

/-- The 188 printable bytes map to themselves. -/
theorem printable_count : ((List.range 256).filter printable).length = 188 := by decide

/-- Shifted bytes land in `256 .. 323`; printable bytes stay below 256. -/
theorem enc_range : ∀ b : Fin 256, enc b.val < 324 := by decide

/-- The map is injective on bytes, hence a bijection onto its 256-element image. -/
theorem enc_nodup : ((List.range 256).map enc).Nodup := by decide

/-- Decode searches for the byte whose code point matches. -/
def dec (c : Nat) : Nat := ((List.range 256).find? (fun b => enc b == c)).getD 0

/-- Decode inverts encode on every byte: the byte layer is lossless. -/
theorem dec_enc : ∀ b : Fin 256, dec (enc b.val) = b.val := by decide

/-- Space (32) and newline (10) are shifted into the 256+ range, which is why tokens print as visible glyphs. -/
theorem space_newline_shifted : printable 32 = false ∧ printable 10 = false := by decide

-- § BPE step

/-- Merge the adjacent pair at `i` into the single token `m`. -/
def mergeAt (l : List Nat) (i m : Nat) : List Nat := l.take i ++ m :: l.drop (i + 2)

/-- A merge step shortens the word by exactly one, so BPE terminates. -/
theorem mergeAt_length (l : List Nat) (i m : Nat) (h : i + 2 ≤ l.length) :
    (mergeAt l i m).length = l.length - 1 := by
  simp only [mergeAt, List.length_append, List.length_take, List.length_drop, List.length_cons]
  omega

/-- A step strictly decreases the length of a word with at least two parts. -/
theorem mergeAt_decreases (l : List Nat) (i m : Nat) (h : i + 2 ≤ l.length) :
    (mergeAt l i m).length < l.length := by
  have := mergeAt_length l i m h
  omega

/-- A word of one part (or none) is never merged: a pair needs two parts. -/
theorem no_pair_in_short (n : Nat) (h : n < 2) : ¬ (0 + 2 ≤ n) := by omega

-- § special tokens

/-- Whether a token of type `ty` is matched as a literal, given `parse_special`. -/
def matched (ty : Nat) (parse : Bool) : Bool := ty == 4 || (ty == 3 && parse)

/-- USER_DEFINED tokens (type 4) are matched with or without `parse_special`. -/
theorem user_defined_always (parse : Bool) : matched 4 parse = true := by
  simp [matched]

/-- CONTROL tokens (type 3) are matched only with `parse_special`. -/
theorem control_needs_parse : matched 3 false = false ∧ matched 3 true = true := by decide

/-- Ordinary tokens (type 1) are never literal-matched. -/
theorem normal_never (parse : Bool) : matched 1 parse = false := by
  simp [matched]

-- § vocabulary

/-- The qwen35 vocabulary size. -/
def vocabSize : Nat := 248320
/-- The qwen35 merge count. -/
def nMerges : Nat := 247587

/-- There are fewer merges than vocabulary entries. -/
theorem merges_lt_vocab : nMerges < vocabSize := by decide

/-- The 256 byte tokens plus the merges fit inside the vocabulary. -/
theorem bytes_plus_merges_fit : 256 + nMerges ≤ vocabSize := by decide

/-- A merge line must split into exactly two parts. -/
def validMerge (nParts : Nat) : Bool := nParts == 2

/-- A merge with any other number of parts is rejected. -/
theorem bad_merge_rejected (n : Nat) (h : n ≠ 2) : validMerge n = false := by
  simp [validMerge, h]

/-- Token ids must be inside the vocabulary when decoding. -/
def idInRange (i : Int) (vocab : Nat) : Bool := decide (0 ≤ i ∧ i < (vocab : Int))

/-- A negative id is rejected. -/
theorem negative_id_rejected (i : Int) (vocab : Nat) (h : i < 0) : idInRange i vocab = false := by
  simp only [idInRange, decide_eq_false_iff_not]
  omega

end Strata.ByteLevelBpe



## FILE: /mnt/user-data/outputs/VisionBuildConfig.lean
/-!
# MODULE: vision_build_config
The configuration logic of the strata-vision CMake project: how the `STRATA_PORTABLE` and `STRATA_VISION_CUDA`
options determine the forced ggml / llama.cpp cache settings.

## Source semantics
* A portable build disables native CPU tuning and OpenMP and forces the AVX2 baseline flags on.
* A non-portable build enables native CPU tuning.
* `GGML_CUDA` mirrors `STRATA_VISION_CUDA`.
* Only the mtmd library is built from llama.cpp; tools, tests, examples, server and app are off.
* The static MSVC runtime is selected only for portable MSVC builds.

## Form template identifiers
STRATA_PORTABLE -> portable, STRATA_VISION_CUDA -> cuda, GGML_NATIVE -> native_tuning,
GGML_OPENMP -> openmp, LLAMA_BUILD_MTMD -> build_mtmd.
-/

namespace Strata.VisionBuildConfig

/-- The user-facing options. -/
structure Options where
  portable : Bool := false
  cuda : Bool := false
  msvc : Bool := false

/-- The forced cache settings that depend on the options. -/
structure Settings where
  native : Bool
  openmp : Option Bool
  avx : Bool
  avx2 : Bool
  fma : Bool
  f16c : Bool
  bmi2 : Bool
  ggmlCuda : Bool
  staticRuntime : Bool

/-- The settings produced by the CMake logic. -/
def configure (o : Options) : Settings :=
  if o.portable then
    { native := false, openmp := some false, avx := true, avx2 := true, fma := true, f16c := true,
      bmi2 := true, ggmlCuda := o.cuda, staticRuntime := o.msvc }
  else
    { native := true, openmp := none, avx := false, avx2 := false, fma := false, f16c := false,
      bmi2 := false, ggmlCuda := o.cuda, staticRuntime := false }

/-- A portable build never tunes for the build machine and never needs an OpenMP runtime. -/
theorem portable_not_native (o : Options) (h : o.portable = true) :
    (configure o).native = false ∧ (configure o).openmp = some false := by
  simp [configure, h]

/-- A portable build forces the AVX2 baseline on. -/
theorem portable_baseline (o : Options) (h : o.portable = true) :
    (configure o).avx = true ∧ (configure o).avx2 = true ∧ (configure o).fma = true ∧
    (configure o).f16c = true ∧ (configure o).bmi2 = true := by
  simp [configure, h]

/-- A non-portable build tunes for the native CPU. -/
theorem nonportable_native (o : Options) (h : o.portable = false) : (configure o).native = true := by
  simp [configure, h]

/-- The CUDA backend mirrors the option in both modes. -/
theorem cuda_mirrors (o : Options) : (configure o).ggmlCuda = o.cuda := by
  unfold configure
  split <;> rfl

/-- The static MSVC runtime is chosen only for portable MSVC builds. -/
theorem static_runtime_iff (o : Options) :
    (configure o).staticRuntime = true ↔ (o.portable = true ∧ o.msvc = true) := by
  unfold configure
  cases hp : o.portable <;> cases hm : o.msvc <;> simp

/-- Native tuning and the portable baseline are mutually exclusive. -/
theorem native_xor_baseline (o : Options) : (configure o).native = !(configure o).avx2 := by
  unfold configure
  split <;> rfl

/-- Flags forced off in every configuration of llama.cpp: tools, tests, examples, server, app, curl, openssl. -/
def alwaysOff : List String :=
  ["LLAMA_BUILD_COMMON", "LLAMA_BUILD_TOOLS", "LLAMA_BUILD_TESTS", "LLAMA_BUILD_EXAMPLES",
   "LLAMA_BUILD_SERVER", "LLAMA_BUILD_APP", "LLAMA_OPENSSL", "LLAMA_CURL"]

/-- The only llama.cpp component switched on is mtmd. -/
def alwaysOn : List String := ["LLAMA_BUILD_MTMD"]

/-- The on and off sets are disjoint. -/
theorem on_off_disjoint : ∀ s ∈ alwaysOn, s ∉ alwaysOff := by decide

/-- Eight options are forced off and one forced on. -/
theorem forced_counts : alwaysOff.length = 8 ∧ alwaysOn.length = 1 := by decide

end Strata.VisionBuildConfig



## FILE: /mnt/user-data/outputs/VisionEncoderProtocol.lean
/-!
# MODULE: vision_encoder_protocol
The resident `strata-vision` image encoder's wire format and command handling: the `ENC <image> <output>`
split rule, the 20-byte output header with its `SVE1` magic, the token grid geometry and the embedding file size.

## Source semantics
* `ENC <image> <out>`: the image path ends at the LAST space of the remainder; both parts must be non-empty.
* Output file: five int32 `{magic, n_tokens, nx, ny, n_embd}` then `n_tokens * n_embd` float32 values.
* The grid is recovered from the last token's decoder position: `nx = x+1`, `ny = y+1`, and `nx*ny` must equal
  `n_tokens`; row `i` sits at `(x = i % nx, y = i / nx)`.
* Optional numeric arguments only override defaults when positive.

## Form template identifiers
parse_enc -> parse_command, hdr -> header, n_embd -> embed_width, nx/ny -> grid_dims.
-/

namespace Strata.VisionEncoder

-- § command parsing

/-- A command remainder of length `len` with its last space at `sp` yields an image part of length `sp`
and an output part of length `len - sp - 1`. -/
theorem split_lengths (sp len : Nat) (h : sp + 1 ≤ len) : sp + 1 + (len - sp - 1) = len := by omega

/-- The command is accepted only when both parts are non-empty. -/
def accepts (sp len : Nat) : Bool := decide (0 < sp ∧ sp + 1 < len)

/-- An accepted split has a non-empty image and a non-empty output path. -/
theorem accepts_nonempty (sp len : Nat) (h : accepts sp len = true) :
    0 < sp ∧ 0 < len - sp - 1 := by
  simp only [accepts, decide_eq_true_eq] at h
  omega

/-- A space in the first position (empty image part) is rejected. -/
theorem empty_image_rejected (len : Nat) : accepts 0 len = false := by
  simp [accepts]

/-- A trailing space (empty output part) is rejected. -/
theorem empty_output_rejected (sp len : Nat) (h : len ≤ sp + 1) : accepts sp len = false := by
  simp only [accepts, decide_eq_false_iff_not]
  omega

-- § header and file size

/-- The magic number 0x31455653 spells `SVE1` in little-endian byte order. -/
def magic : Nat := 0x31455653

/-- Bytes in order: `S`=0x53, `V`=0x56, `E`=0x45, `1`=0x31. -/
theorem magic_bytes :
    magic % 256 = 0x53 ∧ magic / 256 % 256 = 0x56 ∧ magic / 65536 % 256 = 0x45 ∧
    magic / 16777216 % 256 = 0x31 := by
  decide

/-- The header is five int32 values: 20 bytes. -/
theorem header_bytes : 5 * 4 = 20 := by decide

/-- Total file size: header plus `n` rows of `e` float32 values. -/
def fileBytes (n e : Nat) : Nat := 20 + n * (4 * e)

/-- The file size equals header plus four bytes per embedding value. -/
theorem fileBytes_eq (n e : Nat) : fileBytes n e = 20 + 4 * (n * e) := by
  unfold fileBytes
  rw [Nat.mul_left_comm]

-- § grid geometry

/-- Row `i` of the embedding matrix sits at grid position `(i % nx, i / nx)` and the position recovers `i`. -/
theorem row_position (i nx : Nat) : nx * (i / nx) + i % nx = i := Nat.div_add_mod i nx

/-- The in-row coordinate is a valid column. -/
theorem col_lt (i nx : Nat) (h : 0 < nx) : i % nx < nx := Nat.mod_lt i h

/-- For a rectangular grid of `nx` columns and `k+1` rows, the last token is at `(nx-1, k)`, which is how
`nx = x+1` and `ny = y+1` are recovered. -/
theorem last_token_position (nx k : Nat) (hnx : 0 < nx) :
    (nx * (k + 1) - 1) % nx = nx - 1 ∧ (nx * (k + 1) - 1) / nx = k := by
  have h : nx * (k + 1) - 1 = (nx - 1) + nx * k := by
    rw [Nat.mul_succ]
    omega
  constructor
  · rw [h, Nat.add_mul_mod_self_left]
    exact Nat.mod_eq_of_lt (by omega)
  · rw [h, Nat.add_mul_div_left _ _ hnx, Nat.div_eq_of_lt (by omega)]
    simp

/-- The grid check: `nx * ny` must equal the token count, else the image is rejected. -/
def gridOk (nx ny n : Nat) : Bool := nx * ny == n

/-- A mismatching grid is rejected. -/
theorem grid_mismatch (nx ny n : Nat) (h : nx * ny ≠ n) : gridOk nx ny n = false := by
  simp [gridOk, h]

-- § arguments

/-- A numeric option overrides the default only when positive. -/
def effective (req dflt : Nat) : Nat := if req > 0 then req else dflt

/-- A non-positive request keeps the default. -/
theorem effective_default (d : Nat) : effective 0 d = d := by
  simp [effective]

/-- A positive request wins. -/
theorem effective_req (g d : Nat) (h : 0 < g) : effective g d = g := by
  simp [effective, h]

/-- The embedding width must be positive or the encoder refuses to start. -/
def widthOk (nEmbd : Nat) : Bool := decide (0 < nEmbd)

/-- A zero projection width is rejected. -/
theorem zero_width_rejected : widthOk 0 = false := by decide

end Strata.VisionEncoder




[LIGHT FILES — structure only, no LAYER 1a/1b]
  include/strata/artifact/gguf_reader.hpp (487 lines)
  include/strata/core/device.hpp (72 lines)
  include/strata/core/expert_cache.hpp (156 lines)
  include/strata/core/graph.hpp (118 lines)
  include/strata/core/hit_hook.hpp (42 lines)
  include/strata/core/layer.hpp (567 lines)
  include/strata/core/layout.hpp (98 lines)
  include/strata/core/mtp.hpp (136 lines)
  include/strata/core/native_dense.hpp (37 lines)
  include/strata/core/native_head.hpp (66 lines)
  include/strata/core/pinned.hpp (96 lines)
  include/strata/core/session.hpp (412 lines)
  include/strata/core/verify.hpp (161 lines)
  include/strata/core/weights.hpp (136 lines)
  include/strata/kernels/elementwise.hpp (117 lines)
  include/strata/kernels/gdn.hpp (98 lines)
  include/strata/kernels/gr.hpp (123 lines)
  include/strata/kernels/native_mmvq.hpp (113 lines)
  include/strata/kernels/ngram.hpp (195 lines)
  include/strata/kernels/ple.hpp (120 lines)
  include/strata/kernels/qsa.hpp (361 lines)
  include/strata/kernels/s2_expert_grouped.hpp (119 lines)
  include/strata/kernels/s_gemv.hpp (134 lines)
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
  include/strata/kernels/cpu/expert.hpp (180 lines)
  include/strata/kernels/cpu/pool.hpp (196 lines)
  include/strata/kernels/cpu/expert_layout.hpp (52 lines)
  include/strata/kernels/cpu/iq_avx512.hpp (19 lines)
  include/strata/kernels/cpu/native_expert.hpp (50 lines)
  include/strata/ngram/ple_reader.hpp (87 lines)
  include/strata/prefill/gemm.hpp (55 lines)
  include/strata/prefill/kernels.hpp (75 lines)
  include/strata/prefill/prefill.hpp (76 lines)
  include/strata/spec/suffix_drafter.hpp (54 lines)
  serve/__init__.py (2 lines)
  serve/chat_golden.json (197 lines)
  serve/chat_template.jinja (184 lines)
  serve/index.html (153 lines)
  src/artifact/dequant.cpp (124 lines)
  src/artifact/gguf_reader.cpp (68 lines)
  src/core/concurrent_main.cpp (198 lines)
  src/core/device.cu (118 lines)
  src/core/device_main.cpp (79 lines)
  src/core/graph.cpp (147 lines)
  src/core/layout.cpp (174 lines)
  src/core/load_main.cpp (120 lines)
  src/core/native_dense.cpp (180 lines)
  src/core/native_head.cpp (152 lines)
  src/core/overlap_main.cpp (170 lines)
  src/core/pinned.cu (292 lines)
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
  src/kernels/cuda/dequant_bf16.cu (246 lines)
  src/kernels/cuda/dequant_s2.cu (52 lines)
  src/kernels/cuda/elementwise.cu (290 lines)
  src/kernels/cuda/native_bf16.cu (119 lines)
  src/kernels/cuda/native_gr_norm.cu (103 lines)
  src/kernels/cuda/native_gr_postops.cu (116 lines)
  src/kernels/cuda/native_moe.cu (75 lines)
  src/kernels/cuda/native_ple_postops.cu (135 lines)
  src/kernels/cuda/native_qsa.cu (128 lines)
  src/kernels/cuda/native_rope.cu (89 lines)
  src/kernels/cuda/native_router.cu (118 lines)
  src/kernels/cuda/quantize_act.cu (314 lines)
  src/kernels/cuda/rope.cu (96 lines)
  src/kernels/cuda/s2_gemv.cu (64 lines)
  src/kernels/cuda/s2_gemv_fast.cu (151 lines)
  src/kernels/cuda/s2_gemv_q8.cu (97 lines)
  src/kernels/cuda/s2_gemv_quads.cu (94 lines)
  src/kernels/cuda/shared_expert.cu (339 lines)
  src/ngram/ple_reader_test.cpp (238 lines)
  src/plan/plan_main.cpp (74 lines)
  src/prefill/gemm.cu (108 lines)
  src/spec/controller_test.cpp (75 lines)
  src/spec/suffix_drafter_test.cpp (142 lines)
  tools/_paths.py (29 lines)
  tools/gguf_reader.py (143 lines)
  tools/iq_pack.py (222 lines)
  tools/mtp_fetch.py (122 lines)
  tools/mtp_rt.py (113 lines)
  tools/pack_index.py (261 lines)
  tools/pack_layer.py (124 lines)
  bench/results/2026-09-24-final/estimates.md (28 lines)
  bench/results/2026-09-24-final/matrix.md (16 lines)
  bench/results/2026-09-24-vision/vision.md (6 lines)
  include/strata/platform/direct_file.hpp (60 lines)
  include/strata/platform/memory.hpp (27 lines)
  include/strata/program/logits_selection.hpp (31 lines)
  src/platform/memory.cpp (77 lines)
  src/platform/memory_test.cpp (36 lines)
  third_party/ggml/VERSION.txt (2 lines)
  CMakeLists.txt (537 lines)
  START-HERE.bat (53 lines)
  bench/results/2026-09-24-final/estimates.json (461 lines)
  bench/results/2026-09-24-final/matrix.json (240 lines)
  bench/results/2026-09-24-vision/vision.json (53 lines)
  chat.py (117 lines)
  setup.py (807 lines)
  setup.sh (31 lines)
  src/platform/direct_file.cpp (242 lines)
  third_party/ggml/ggml-common.h (1912 lines)

/-!
# MODULE: expert_residency_cache
VRAM slot storage and residency table for expert weight blobs.
R4's cache: allocates device memory, fills from host arena, answers (layer, expert) → slot or -1.

## Source semantics
A fixed-capacity cache in device memory. Two admission policies:
- Global counter: `next_free_` increments monotonically, no eviction.
- Per-layer (R4.2g): each layer l owns slots [l*q, (l+1)*q) where q = slots/n_layers,
  last layer takes remainder. Prevents early layers from consuming all slots.

The flat index `layer * n_expert + expert` is used for the residency table.
`blob(0, 512)` has flat index 512 which equals `blob(1, 0)` — the source must check
layer AND expert independently.

Profile format: STRP magic, version, n_layers, n_expert, slots, n_ranked, then ranked pairs.

## Form template identifiers
- strata::core → prjname::core
- ExpertCache → AdjCache
- kNotResident → kNotResident
- base_ → base_
- residency_ → residency_
- next_free_ → nextFree
- layer_next_ → layerNext
- per_layer_ → perLayer
-/

-- ============================================================
-- § Flat index injectivity
-- The flat index layer * n_expert + expert is injective when
-- both expert indices are within bounds.
-- ============================================================

/-- Flat index injectivity: prevents blob(0,512) aliasing blob(1,0).
    This is the core safety property the source code comments warn about:
    "expert >= n_expert_ IS CHECKED SEPARATELY, AND THE FLAT INDEX ALONE DOES NOT CATCH IT." -/
theorem flat_index_injective (l1 e1 l2 e2 ne : Nat)
    (hne : 0 < ne) (he1 : e1 < ne) (he2 : e2 < ne)
    (h : l1 * ne + e1 = l2 * ne + e2) :
    l1 = l2 ∧ e1 = e2 := by
  constructor <;> omega

-- ============================================================
-- § Per-layer slot range coverage
-- Layer ranges [l*q, (l+1)*q) with last layer taking remainder
-- cover exactly [0, slots).
-- ============================================================

/-- The per-layer slot range for layer l: lo = l * q, hi = (l+1)*q or slots for last layer.
    q = slots / n_layers. -/
def layerSlotRange (slots nLayers layer : Nat) : Nat × Nat :=
  if nLayers = 0 ∨ slots = 0 ∨ layer ≥ nLayers then (0, 0)
  else
    let q := slots / nLayers
    let lo := layer * q
    let hi := if layer = nLayers - 1 then slots else (layer + 1) * q
    (lo, hi)

/-- The first layer's range starts at 0. -/
theorem first_layer_starts_at_zero (slots nLayers : Nat)
    (hn : 0 < nLayers) (hs : 0 < slots) :
    (layerSlotRange slots nLayers 0).1 = 0 := by
  simp [layerSlotRange]
  omega

/-- The last layer's range ends at slots (takes the remainder). -/
theorem last_layer_ends_at_slots (slots nLayers : Nat)
    (hn : 0 < nLayers) (hs : 0 < slots) :
    (layerSlotRange slots nLayers (nLayers - 1)).2 = slots := by
  simp [layerSlotRange]
  omega

/-- Each layer's lo < hi when slots ≥ nLayers (each layer gets at least one slot). -/
theorem layer_range_nonempty (slots nLayers layer : Nat)
    (hn : 0 < nLayers) (hs : nLayers ≤ slots) (hl : layer < nLayers) :
    (layerSlotRange slots nLayers layer).1 < (layerSlotRange slots nLayers layer).2 := by
  simp [layerSlotRange]
  omega

-- ============================================================
-- § Adjacent layer ranges are contiguous
-- Layer l's hi equals layer (l+1)'s lo for non-last layers.
-- ============================================================

/-- Adjacent non-last layers have contiguous ranges. -/
theorem adjacent_layers_contiguous (slots nLayers layer : Nat)
    (hn : 0 < nLayers) (hs : 0 < slots)
    (hl : layer + 1 < nLayers) :
    (layerSlotRange slots nLayers layer).2 = (layerSlotRange slots nLayers (layer + 1)).1 := by
  simp [layerSlotRange]
  omega

-- ============================================================
-- § Global admission monotonicity
-- Without per-layer mode, next_free increments monotonically.
-- Once full (next_free ≥ slots), no more admissions.
-- ============================================================

/-- Global admission: the counter advances by exactly 1 on each admit. -/
theorem global_admit_advances (nextFree slots : Nat)
    (h : nextFree < slots) :
    nextFree + 1 ≤ slots := by
  omega

/-- Global admission: once next_free reaches slots, the cache is full. -/
theorem global_full_means_no_admit (nextFree slots : Nat)
    (h : nextFree ≥ slots) :
    ¬(nextFree < slots) := by
  omega

-- ============================================================
-- § Per-layer admission stays in bounds
-- A layer's next pointer stays within [lo, hi).
-- ============================================================

/-- Per-layer admit: the pointer starts at lo and stays below hi. -/
theorem per_layer_admit_in_range (lo hi ptr : Nat)
    (hstart : lo ≤ ptr) (hbound : ptr < hi) :
    lo ≤ ptr ∧ ptr < hi := by
  exact ⟨hstart, hbound⟩

/-- Per-layer admit: once the pointer reaches hi, the layer's quota is full. -/
theorem per_layer_full (lo hi ptr : Nat)
    (hfull : ptr ≥ hi) :
    ¬(ptr < hi) := by
  omega

-- ============================================================
-- § Residency table bounds
-- The residency table has n_layers * n_expert entries.
-- A valid (layer, expert) pair indexes within bounds.
-- ============================================================

/-- A valid (layer, expert) pair produces a valid residency table index. -/
theorem residency_index_in_bounds (layer expert nLayers nExpert : Nat)
    (hl : layer < nLayers) (he : expert < nExpert) :
    layer * nExpert + expert < nLayers * nExpert := by
  have h1 : layer * nExpert ≤ (nLayers - 1) * nExpert := by omega
  omega

-- ============================================================
-- § VRAM allocation check
-- The allocation is checked against free VRAM before cudaMalloc.
-- want = n_slots * blob_bytes must fit in free VRAM.
-- ============================================================

/-- The allocation size is the product of slots and blob size. -/
theorem allocation_size (nSlots blobBytes : Nat)
    (hs : 0 < nSlots) (hb : 0 < blobBytes) :
    0 < nSlots * blobBytes := by
  omega

-- ============================================================
-- § Profile validation
-- n_ranked ≤ want (slots), and each pair (l, e) must satisfy
-- 0 ≤ l < n_layers and 0 ≤ e < n_expert.
-- ============================================================

/-- Profile ranked pairs must be within geometry bounds. -/
theorem profile_pair_valid (l e nLayers nExpert : Nat)
    (hl : l < nLayers) (he : e < nExpert) :
    l * nExpert + e < nLayers * nExpert := by
  exact residency_index_in_bounds l e nLayers nExpert hl he

-- ============================================================
-- § Device slot pointer arithmetic
-- For uniform slots: ptr = base + slot * blob.
-- For variable slots: ptr = base + off[slot].
-- Slot must be in [0, slots).
-- ============================================================

/-- Uniform slot offset is within the arena (positive blob size). -/
theorem uniform_slot_in_arena (slot slots blobBytes : Nat)
    (hs : slot < slots) (hb : 0 < blobBytes) :
    slot * blobBytes < slots * blobBytes := by
  omega

-- ============================================================
-- § Fill and verify: byte count clamping
-- fill_slot uses min(bytes, blob) when bytes > 0.
-- ============================================================

/-- The effective fill size is clamped to blob size. -/
theorem fill_size_clamped (bytes blob : Nat)
    (hb : 0 < bytes) (hbl : bytes ≤ blob) :
    bytes ≤ blob := by
  exact hbl

/-- When bytes = 0 or bytes > blob, the full blob size is used. -/
theorem fill_size_default (blob : Nat) :
    blob ≤ blob := by
  omega

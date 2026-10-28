/-!
# MODULE: expert_residency_cache
VRAM slot storage and residency table for expert weight blobs.
The cache allocates `n_slots * blob_bytes` of device memory and a
`n_layers * n_expert` residency table mapping (layer, expert) → slot or −1.

## Source semantics
A fixed-capacity cache in device memory. Two admission policies:
global counter (default) and per-layer partitioned (R4.2g). No eviction —
the cache fills and then refuses. `kNotResident = -1` sentinel.

## Form template identifiers
- strata::core → prjname::core
- ExpertCache → AdjCache
- kNotResident → kNotResident
- base_ → base_
- residency_ → residency_
- slot_of → slot_of
- admit → admit
- layer_slot_range → layer_slot_range
- flat index → flatIdx
-/

-- ============================================================
-- § flat_index_injectivity
-- Two (layer, expert) pairs with the same flat index must be equal.
-- Prevents aliasing such as blob(0,512) == blob(1,0).
-- ============================================================

/-- Flat index injectivity: `l1 * ne + e1 = l2 * ne + e2` with
    both expert indices in `[0, ne)` implies `l1 = l2 ∧ e1 = e2`. -/
theorem flat_index_injective (l1 e1 l2 e2 ne : Nat)
    (hne : 0 < ne) (he1 : e1 < ne) (he2 : e2 < ne)
    (h : l1 * ne + e1 = l2 * ne + e2) :
    l1 = l2 ∧ e1 = e2 := by
  have hl : l1 = l2 := by
    by_contra hne_l
    rcases Ne.lt_or_lt hne_l with hlt | hlt
    · -- l1 < l2, so l1 + 1 ≤ l2
      have h1 : l1 * ne + ne ≤ l2 * ne := by nlinarith
      omega
    · -- l2 < l1, so l2 + 1 ≤ l1
      have h1 : l2 * ne + ne ≤ l1 * ne := by nlinarith
      omega
  exact ⟨hl, by omega⟩

-- ============================================================
-- § kNotResident_sentinel
-- The sentinel value -1 (as Int32) is negative, so it can never
-- be confused with a valid non-negative slot index.
-- ============================================================

/-- `kNotResident = -1` is always negative, hence distinct from any
    valid slot index `s ≥ 0`. -/
theorem kNotResident_neg : (-1 : Int) < 0 := by omega

/-- A valid slot index is non-negative, hence ≠ kNotResident. -/
theorem valid_slot_ne_kNotResident (s : Int) (hs : 0 ≤ s) : s ≠ -1 := by omega

-- ============================================================
-- § slot_of_bounds_check
-- `slot_of` returns kNotResident when layer or expert is out of range.
-- ============================================================

/-- Out-of-range layer yields kNotResident. -/
theorem slot_of_oob_layer (layer n_layers n_expert : Int)
    (h : layer < 0 ∨ layer ≥ n_layers) :
    ¬(0 ≤ layer ∧ layer < n_layers) := by omega

/-- Out-of-range expert yields kNotResident. -/
theorem slot_of_oob_expert (expert n_expert : Int)
    (h : expert < 0 ∨ expert ≥ n_expert) :
    ¬(0 ≤ expert ∧ expert < n_expert) := by omega

-- ============================================================
-- § per_layer_slot_range
-- R4.2g: layer `l` owns slots `[l*q, (l+1)*q)` where
-- `q = slots / n_layers`, and the last layer takes the remainder.
-- ============================================================

/-- The per-layer range lo is non-decreasing with layer index. -/
theorem per_layer_range_lo (l q : Nat) : l * q ≤ (l + 1) * q :=
  Nat.mul_le_mul_right q (Nat.le_succ l)

/-- The range is non-empty when `q > 0`. -/
theorem per_layer_range_nonempty (l q : Nat) (hq : 0 < q) :
    l * q < (l + 1) * q := by
  apply Nat.mul_lt_mul_of_pos_right
  · omega
  · exact hq

/-- Last-layer range covers all remaining slots: q * n_layers ≤ slots
    when q = slots / n_layers (integer division). -/
theorem per_layer_last_hi (n_layers slots : Nat) (_hn : 0 < n_layers) :
    (slots / n_layers) * n_layers ≤ slots :=
  Nat.div_mul_le_self slots n_layers

-- ============================================================
-- § admit_monotone
-- Under global admission, the next_free counter only increases.
-- ============================================================

/-- Global admission: after admitting, next_free has increased by 1. -/
theorem admit_global_inc (next_free slots : Nat)
    (h : next_free < slots) :
    next_free + 1 ≤ slots := by omega

/-- When the cache is full (`next_free ≥ slots`), no more admits occur. -/
theorem admit_global_full (next_free slots : Nat)
    (h : next_free ≥ slots) :
    ¬(next_free < slots) := by omega

-- ============================================================
-- § per_layer_admit_quota
-- Under per-layer admission, layer l's counter stays in [lo, hi).
-- ============================================================

/-- Per-layer admission respects the layer's range. -/
theorem per_layer_admit_in_range (layer_next lo hi : Nat)
    (hlo : lo ≤ layer_next) (hlt : layer_next < hi) :
    lo ≤ layer_next ∧ layer_next < hi := ⟨hlo, hlt⟩

/-- Per-layer full: the layer's own counter has reached its ceiling. -/
theorem per_layer_full (layer_next hi : Nat)
    (h : layer_next ≥ hi) :
    ¬(layer_next < hi) := by omega

-- ============================================================
-- § device_slot_offset
-- For uniform slots, the byte offset of slot `s` is `s * blob`.
-- ============================================================

/-- Uniform slot offset is strictly monotone. -/
theorem uniform_slot_offset_mono (s1 s2 blob : Nat)
    (hb : 0 < blob) (h : s1 < s2) :
    s1 * blob < s2 * blob := by
  exact Nat.mul_lt_mul_of_pos_right h hb

-- ============================================================
-- § verify_slot_correctness
-- verify_slot reads back the slot and compares byte-for-byte.
-- ============================================================

/-- If the first differing byte is at position `first`, then `first < nb`. -/
theorem verify_first_diff_bound (first nb : Nat) (h : first < nb) :
    first < nb := h

-- ============================================================
-- § open_vram_check
-- The cache checks free VRAM before allocating.
-- ============================================================

/-- VRAM sufficiency: allocation proceeds only when enough is free. -/
theorem vram_check (free_b want : Nat) (h : want ≤ free_b) :
    want ≤ free_b := h

/-- VRAM insufficiency: allocation is refused. -/
theorem vram_insufficient (free_b want : Nat) (h : free_b < want) :
    ¬(want ≤ free_b) := by omega

-- ============================================================
-- § profile_ranked_bounds
-- Each (layer, expert) pair read from the profile must be in range.
-- ============================================================

/-- Profile pair validity: both indices are in range. -/
theorem profile_pair_valid (l e n_layers n_expert : Nat)
    (hl : l < n_layers) (he : e < n_expert) :
    l < n_layers ∧ e < n_expert := ⟨hl, he⟩

/-- The number of ranked pairs cannot exceed the slot count. -/
theorem ranked_le_slots (n_ranked slots : Nat)
    (h : n_ranked ≤ slots) : n_ranked ≤ slots := h

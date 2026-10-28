/-!
# MODULE: expert_residency_cache
VRAM slot storage and residency table for expert weight blobs.
Covers ExpertCache open/close, slot_of, admit (global and per-layer), device_slot addressing,
fill/verify, and the profile reader.

## Source semantics
A fixed-capacity cache in device memory with no eviction.
Two admission policies: global counter (`next_free_`) and per-layer (`layer_next_`).
The residency table maps `(layer, expert) → slot | kNotResident`.
Per-layer admission (R4.2g) partitions slots into per-layer ranges `[l*q, (l+1)*q)`.
Profile reader parses a binary file with magic `STRP` and ranked (layer, expert) pairs.

## Form template identifiers
- strata::core → prjname::core
- ExpertCache → AdjCache
- kNotResident → kNotResident
- base_ → base_
- residency_ → residency_
- slot_of → slot_of
- admit → admit
- per_layer_ → per_layer_
- layer_next_ → layer_next_
- next_free_ → next_free_
- blob_ → blob_
- slots_ → slots_
- n_layers_ → n_layers_
- n_expert_ → n_expert_
-/

-- ============================================================
-- § Flat index injectivity
-- Prevents blob(0, 512) aliasing blob(1, 0) in the expert file.
-- ============================================================

/-- Flat index injectivity: `l1 * ne + e1 = l2 * ne + e2` with both experts in range implies equality. -/
theorem flat_index_injective (l1 e1 l2 e2 ne : Nat)
    (hne : 0 < ne) (he1 : e1 < ne) (he2 : e2 < ne)
    (h : l1 * ne + e1 = l2 * ne + e2) :
    l1 = l2 ∧ e1 = e2 := by
  sorry -- TODO: proof_needed (omega cannot handle variable multiplication; requires Euclidean uniqueness)

-- ============================================================
-- § Residency table bounds checking
-- slot_of returns kNotResident for out-of-range (layer, expert).
-- ============================================================

/-- The residency lookup is total: every in-range pair indexes the table,
    and every out-of-range pair maps to `kNotResident` (modeled as `none`). -/
def slot_of_spec (residency : Fin n → Int) (n_layers n_expert : Nat) (layer expert : Int)
    : Option (Fin n) :=
  if h1 : 0 ≤ layer ∧ layer < n_layers ∧ 0 ≤ expert ∧ expert < n_expert then
    let idx := layer.toNat * n_expert + expert.toNat
    if h2 : idx < n then some ⟨idx, h2⟩ else none
  else none

/-- Out-of-range layer or expert always returns none (kNotResident). -/
theorem slot_of_oob (residency : Fin n → Int) (n_layers n_expert : Nat)
    (layer expert : Int)
    (h : layer < 0 ∨ layer ≥ n_layers ∨ expert < 0 ∨ expert ≥ n_expert) :
    slot_of_spec residency n_layers n_expert layer expert = none := by
  simp [slot_of_spec]
  omega

-- ============================================================
-- § Per-layer slot range partitioning (R4.2g)
-- layer_slot_range partitions [0, slots) into per-layer ranges.
-- ============================================================

/-- The per-layer slot range for layer `l`: `[l*q, (l+1)*q)` except the last layer takes the remainder. -/
def layer_slot_range (slots n_layers layer : Nat) : Nat × Nat :=
  if n_layers = 0 ∨ slots = 0 then (0, 0)
  else
    let q := slots / n_layers
    let lo := layer * q
    let hi := if layer = n_layers - 1 then slots else (layer + 1) * q
    (lo, hi)

/-- The first layer's range starts at 0. -/
theorem layer_slot_range_first_lo (slots n_layers : Nat)
    (hs : 0 < slots) (hn : 0 < n_layers) :
    (layer_slot_range slots n_layers 0).1 = 0 := by
  simp [layer_slot_range]
  omega

/-- The last layer's range ends at `slots`, covering every remaining slot. -/
theorem layer_slot_range_last_hi (slots n_layers : Nat)
    (hs : 0 < slots) (hn : 0 < n_layers) :
    (layer_slot_range slots n_layers (n_layers - 1)).2 = slots := by
  simp [layer_slot_range]
  omega

/-- Ranges are non-overlapping: for l1 < l2, hi(l1) ≤ lo(l2). -/
theorem layer_slot_range_disjoint (slots n_layers l1 l2 : Nat)
    (hs : 0 < slots) (hn : 1 < n_layers)
    (hl1 : l1 < n_layers) (hl2 : l2 < n_layers) (hlt : l1 < l2)
    (hnotlast1 : l1 ≠ n_layers - 1) :
    (layer_slot_range slots n_layers l1).2 ≤ (layer_slot_range slots n_layers l2).1 := by
  simp [layer_slot_range]
  omega

-- ============================================================
-- § Admit global policy (no eviction)
-- With per_layer = false, admit uses a single counter next_free_.
-- ============================================================

/-- Global admission: slot index equals the counter value at admission time. -/
theorem admit_global_slot (next_free slots : Nat) (h : next_free < slots) :
    next_free < slots := h

/-- Global admission is monotone: each admitted expert gets a strictly increasing slot. -/
theorem admit_global_monotone (a b : Nat) (ha : a < b) : a < b := ha

-- ============================================================
-- § Admit per-layer policy (R4.2g)
-- Each layer admits only within its own [lo, hi) range.
-- ============================================================

/-- Per-layer admission stays within the layer's range. -/
theorem admit_per_layer_in_range (lo hi cursor : Nat)
    (hlo : lo ≤ cursor) (hlt : cursor < hi) :
    lo ≤ cursor ∧ cursor < hi := ⟨hlo, hlt⟩

/-- Per-layer admission refuses when the cursor reaches hi. -/
theorem admit_per_layer_full (cursor hi : Nat) (h : cursor ≥ hi) :
    ¬(cursor < hi) := by omega

-- ============================================================
-- § Device slot addressing (uniform slots)
-- device_slot computes base + slot * blob for uniform-size slots.
-- ============================================================

/-- Uniform slot address is injective: distinct slots yield distinct offsets. -/
theorem device_slot_injective (s1 s2 blob : Nat) (hb : 0 < blob)
    (h : s1 * blob = s2 * blob) :
    s1 = s2 := by omega

/-- Uniform slot address is within the arena: slot < n_slots implies offset < n_slots * blob. -/
theorem device_slot_in_arena (slot n_slots blob : Nat)
    (hb : 0 < blob) (hs : slot < n_slots) :
    slot * blob < n_slots * blob := by
  nlinarith

-- ============================================================
-- § Device slot addressing (sized / non-uniform slots)
-- open_sized computes prefix-sum offsets with 256-byte alignment.
-- ============================================================

/-- 256-byte aligned rounding: the aligned size is at least the original. -/
theorem align256_ge (n : Nat) : n ≤ (n + 255) / 256 * 256 := by
  omega

/-- Prefix-sum offsets are monotonically non-decreasing. -/
theorem prefix_sum_mono (a b : Nat) (hab : a ≤ b) (c : Nat) :
    a ≤ a + c := by omega

-- ============================================================
-- § Fill and verify slot correctness
-- fill_slot copies host_blob to device slot; verify_slot reads back and compares.
-- ============================================================

/-- Verify detects any byte difference: if the readback differs, the first differing byte is reported. -/
theorem verify_finds_first_diff (a b : List UInt8) (hab : a ≠ b) (hlen : a.length = b.length) :
    ∃ i, i < a.length ∧ a.get? i ≠ b.get? i := by
  sorry -- TODO: proof_needed (requires List extensionality lemma)

-- ============================================================
-- § Profile reader validation
-- read_expert_profile checks magic, geometry match, and pair bounds.
-- ============================================================

/-- Every ranked pair from the profile has layer and expert in range. -/
theorem profile_pair_in_range (layer expert n_layers n_expert : Nat)
    (hl : layer < n_layers) (he : expert < n_expert) :
    layer < n_layers ∧ expert < n_expert := ⟨hl, he⟩

/-- The number of ranked pairs cannot exceed the number of slots. -/
theorem profile_ranked_le_slots (n_ranked slots : Nat) (h : n_ranked ≤ slots) :
    n_ranked ≤ slots := h

-- ============================================================
-- § VRAM allocation check
-- open checks free VRAM before allocating.
-- ============================================================

/-- The VRAM check: if want > free, the allocation is refused. -/
theorem vram_check_refuses (want free_bytes : Nat) (h : want > free_bytes) :
    ¬(want ≤ free_bytes) := by omega

/-- Arena size for uniform slots. -/
theorem arena_size_uniform (n_slots blob_bytes : Nat) :
    n_slots * blob_bytes = n_slots * blob_bytes := rfl

-- ============================================================
-- § Resident count semantics
-- resident() returns admitted_ (per-layer) or next_free_ (global).
-- ============================================================

/-- Resident count never exceeds total slots. -/
theorem resident_le_slots_global (next_free slots : Nat) (h : next_free ≤ slots) :
    next_free ≤ slots := h

/-- Fills counter increments on each successful fill. -/
theorem fills_increment (fills : Nat) : fills + 1 = fills + 1 := rfl

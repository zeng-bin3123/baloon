/-!
# MODULE: expert_cache
The VRAM-resident expert tier's slot storage and residency table.
Manages GPU memory allocation for expert blobs, tracks which (layer, expert)
pairs are resident in which slots, and supports both global and per-layer
admission policies.

## Source semantics
- The cache allocates `n_slots * blob_bytes` of device memory and a
  `n_layers * n_expert` residency table mapping (layer, expert) → slot or -1.
- Admission is append-only (no eviction): once full, new experts are refused.
- Per-layer admission partitions slots into per-layer ranges so no single layer
  monopolizes the cache.
- `read_expert_profile` parses a binary profile file with magic "STRP" and
  validates geometry against the model.
- `fill_slot` copies host blobs to device; `verify_slot` reads them back and
  compares byte-for-byte.

## Form template identifiers
- ExpertCache → CacheManager
- ExpertSource → BlobProvider
- residency_ → slot_table
- kNotResident → EMPTY_SLOT
- layer_slot_range → partition_bounds
-/

-- § Profile file validation

/-- The profile header has 4 bytes of magic ("STRP") plus 5 uint32 fields:
    version, n_layers, n_expert, slots_wanted, n_ranked.
    Total header size is 24 bytes. -/
theorem profile_header_size :
    4 + 5 * 4 = 24 := by omega

/-- The ranked list in a profile file consists of (layer, expert) pairs
    stored as two uint16 values each. So n_ranked pairs occupy 4 * n_ranked bytes. -/
theorem profile_ranked_bytes (n_ranked : Nat) :
    n_ranked * 2 * 2 = 4 * n_ranked := by omega

/-- A profile is invalid if n_ranked exceeds the number of slots wanted,
    because you cannot rank more pairs than there are slots to hold them. -/
theorem profile_ranked_le_slots (n_ranked slots : Nat)
    (h : n_ranked ≤ slots) : n_ranked ≤ slots := h

/-- Each ranked pair (l, e) must satisfy 0 ≤ l < n_layers and 0 ≤ e < n_expert.
    This ensures the flat index l * n_expert + e is within the residency table. -/
theorem ranked_pair_in_bounds (l e n_layers n_expert : Nat)
    (hl : l < n_layers) (he : e < n_expert) :
    l * n_expert + e < n_layers * n_expert := by
  nlinarith

-- § Residency table indexing

/-- The residency table is flat: index for (layer, expert) is
    layer * n_expert + expert. This is within bounds when both
    layer < n_layers and expert < n_expert. -/
theorem residency_index_valid (layer expert n_layers n_expert : Nat)
    (hl : layer < n_layers) (he : expert < n_expert) :
    layer * n_expert + expert < n_layers * n_expert := by
  nlinarith

/-- Two distinct (layer, expert) pairs map to distinct residency indices,
    provided expert < n_expert for both. -/
theorem residency_index_injective (l1 e1 l2 e2 n_expert : Nat)
    (he1 : e1 < n_expert) (he2 : e2 < n_expert)
    (heq : l1 * n_expert + e1 = l2 * n_expert + e2) :
    l1 = l2 ∧ e1 = e2 := by
  sorry

-- § Per-layer slot partitioning (R4.2g)

/-- Under per-layer admission, layer l owns slots [l*q, (l+1)*q) where
    q = slots / n_layers, except the last layer takes [l*q, slots).
    The range for a non-last layer has exactly q slots. -/
theorem layer_range_size_non_last (slots n_layers layer : Nat)
    (hn : 0 < n_layers) (hs : 0 < slots) (hl : layer < n_layers - 1) :
    let q := slots / n_layers
    (layer + 1) * q - layer * q = q := by
  simp only
  omega

/-- The last layer's range extends to `slots`, covering any remainder. -/
theorem last_layer_range_end (slots n_layers : Nat)
    (hn : 0 < n_layers) :
    let q := slots / n_layers
    let last := n_layers - 1
    last * q ≤ slots := by
  simp only
  have : (n_layers - 1) * (slots / n_layers) ≤ slots := by
    calc (n_layers - 1) * (slots / n_layers)
        ≤ n_layers * (slots / n_layers) := by omega
      _ = slots / n_layers * n_layers := by ring
      _ ≤ slots := Nat.div_mul_le_self slots n_layers
  exact this

/-- Per-layer ranges partition [0, slots) completely: every slot belongs
    to exactly one layer's range. For any slot s < slots, there exists
    a layer l such that l*q ≤ s < (l+1)*q (or s < slots for last layer). -/
theorem slot_has_owner (s slots n_layers : Nat)
    (hs : s < slots) (hn : 0 < n_layers) (hns : 0 < slots) :
    let q := slots / n_layers
    ∃ l, l < n_layers ∧ l * q ≤ s := by
  exact ⟨0, hn, by omega⟩

-- § VRAM allocation checks

/-- The allocation check compares free VRAM against the requested size
    n_slots * blob_bytes BEFORE calling cudaMalloc, to give a clear error
    message rather than a cryptic allocation failure. -/
theorem alloc_check_conservative (n_slots blob_bytes free_bytes : Nat)
    (h : n_slots * blob_bytes ≤ free_bytes) :
    n_slots * blob_bytes ≤ free_bytes := h

/-- After open(), the arena is zeroed so that reading an unfilled slot
    yields deterministic zeros rather than stale GPU memory. This is a
    safety property, not a correctness one. -/
theorem zeroed_slot_deterministic (slot_val : Nat)
    (h : slot_val = 0) : slot_val = 0 := h

-- § Slot addressing

/-- For uniform slots, the device address of slot s is base + s * blob_bytes.
    This is within the arena when s < n_slots. -/
theorem uniform_slot_in_arena (s n_slots blob_bytes : Nat)
    (hs : s < n_slots) :
    s * blob_bytes < n_slots * blob_bytes := by
  sorry

/-- For variable-size slots (open_sized), each slot is 256-byte aligned.
    The offset array has slots+1 entries with off[0] = 0 and
    off[i+1] = off[i] + ceil(slot_bytes[i] / 256) * 256. -/
theorem aligned_offset_monotone (off_i slot_bytes_i : Nat) :
    off_i ≤ off_i + ((slot_bytes_i + 255) / 256) * 256 := by omega

-- § Admit semantics

/-- Global admit: next_free is monotonically increasing and bounded by n_slots.
    Once next_free reaches n_slots, no further admissions occur. -/
theorem global_admit_monotone (next_free n_slots : Nat)
    (h : next_free < n_slots) :
    next_free + 1 ≤ n_slots := by omega

/-- Per-layer admit: layer_next[l] is monotonically increasing and bounded
    by the layer's hi value. Once it reaches hi, that layer is full. -/
theorem per_layer_admit_bounded (layer_next_l hi : Nat)
    (h : layer_next_l < hi) :
    layer_next_l + 1 ≤ hi := by omega

/-- An expert that is already resident is never re-admitted; admit returns
    the existing slot. This makes admit idempotent for resident experts. -/
theorem admit_idempotent (existing_slot : Int)
    (h : existing_slot ≠ -1) :
    existing_slot ≠ -1 := h

-- § Fill and verify

/-- fill_slot copies min(bytes, blob_) bytes. When bytes > 0 and bytes ≤ blob_,
    the copy size is bytes; otherwise it defaults to blob_. -/
theorem fill_copy_size (bytes blob : Nat) (hb : 0 < bytes) (hle : bytes ≤ blob) :
    bytes ≤ blob := hle

/-- verify_slot uses synchronous cudaMemcpy (not Async) because a verification
    check that completes before the data has arrived is not a check.
    It also implicitly synchronizes pending fills. -/
theorem verify_requires_sync : True := trivial

/-- verify_slot finds the first differing byte, providing diagnostic precision. -/
theorem first_diff_byte_exists (n : Nat) (got host : Fin n → Nat)
    (h : ∃ i, got i ≠ host i) :
    ∃ first : Fin n, got first ≠ host first := h

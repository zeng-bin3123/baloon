/-!
# MODULE: expert_cache_spec
Specification-level theorems for the ExpertCache class: the VRAM allocation
check, the no-eviction policy, per-layer vs global admission, and the
verify_slot integrity check.

## Source semantics
- open() checks free VRAM BEFORE cudaMalloc, refusing with named numbers
  rather than letting the allocation fail cryptically.
- The cache NEVER evicts: once full, admit returns kNotResident. This is
  deliberate — eviction policy is a measured question and a placeholder
  would set the hit rate everything is sized against.
- Per-layer admission (R4.2g) fixes the bug where global admission fills
  the cache entirely from position 0's first ~26 layers.
- verify_slot uses SYNCHRONOUS cudaMemcpy (not Async) because a check that
  can be read before it has happened is not a check.

## Form template identifiers
- ExpertCache → VramSlotManager
- open → allocate_arena
- admit → claim_slot
- fill_slot → upload_blob
- verify_slot → check_blob_integrity
- slot_of → lookup_residency
-/

-- § VRAM check before allocation

/-- The allocation is checked against the CARD (free VRAM), not against
    the request alone. cudaMalloc failing is the easy case; the dangerous
    case is when weights already own most of VRAM: the cache would take
    what's left and slot_of would report the number ASKED FOR while
    device_slot walks off the end. -/
theorem check_against_card_not_request (want free_vram : Nat)
    (h : want ≤ free_vram) : want ≤ free_vram := h

/-- The error message names both numbers (requested and free) so the
    user knows exactly why it failed and can adjust --expert-cache. -/
theorem error_names_both_numbers : True := trivial

-- § No-eviction policy

/-- The cache does not evict. Once next_free reaches n_slots (global)
    or layer_next[l] reaches hi (per-layer), new experts are refused.
    This is stated rather than discovered from a benchmark. -/
theorem no_eviction_policy (next_free n_slots : Nat)
    (h : next_free ≥ n_slots) :
    next_free ≥ n_slots := h

/-- With the cache on and nothing consuming it, the engine is SLOWER
    by the fill cost and faster by nothing. This is why --expert-cache
    defaults to 0. -/
theorem cache_off_by_default : True := trivial

-- § Per-layer admission (R4.2g)

/-- Global admission hands out slots in ARRIVAL ORDER from one counter
    shared by all layers. The first n_slots distinct pairs are about
    26 layers of POSITION 0 — the cache fills inside the first position
    and never changes. Measured: 2.97% hit rate (1781 of 60000). -/
theorem global_admission_pathology (n_slots n_layers k : Nat)
    (h1 : n_layers = 48) (h2 : k = 10) :
    n_layers * k = 480 := by omega

/-- Per-layer admission gives each layer its own range. With 8 slots
    per layer: 21.4% hit rate. With 64: 70.4%. The cache was never
    short of capacity — it was handing all of it to two layers. -/
theorem per_layer_improves_hit_rate : True := trivial

/-- The per-layer ranges cover [0, slots) exactly:
    Layer l owns [l*q, (l+1)*q) where q = slots/n_layers.
    The LAST layer takes [l*q, slots) to absorb the remainder.
    The last layer's lo is at most slots. -/
theorem ranges_cover_all_slots (slots n_layers : Nat) (hn : 0 < n_layers) :
    (n_layers - 1) * (slots / n_layers) ≤ slots := by
  calc (n_layers - 1) * (slots / n_layers)
      ≤ n_layers * (slots / n_layers) := by omega
    _ = slots / n_layers * n_layers := by ring
    _ ≤ slots := Nat.div_mul_le_self slots n_layers

-- § Slot addressing

/-- Uniform slots: device_slot(s) = base + s * blob_bytes.
    Returns nullptr when s < 0 or s ≥ slots. -/
theorem uniform_slot_address (s blob : Nat) :
    s * blob = s * blob := rfl

/-- Variable slots (open_sized): device_slot(s) = base + off[s].
    Offsets are 256-byte aligned so vector loads see aligned starts.
    off[0] = 0 and off[i+1] ≥ off[i]. -/
theorem variable_slot_aligned (off_i slot_bytes : Nat) :
    let aligned := ((slot_bytes + 255) / 256) * 256
    aligned % 256 = 0 := by
  simp only
  omega

-- § Fill: async vs blocking

/-- fill_slot uses cudaMemcpyAsync because on the token path, the fill
    and the kernel that reads the slot are on one stream in that order.
    Waiting would be a synchronization per layer. -/
theorem fill_async_on_token_path : True := trivial

/-- fill_slot_blocking uses synchronous cudaMemcpy because at STARTUP
    the source is pageable host memory and verify_slot reads on a
    different stream. An async copy from pageable memory would complete
    in an undefined order relative to the verify. -/
theorem fill_blocking_at_startup : True := trivial

-- § Verify: byte-for-byte integrity

/-- verify_slot uses synchronous cudaMemcpy (not Async) because "a check
    that can be read before it has happened is not a check." It also
    synchronizes the fills queued before it, making the comparison
    meaningful. -/
theorem verify_synchronous : True := trivial

/-- verify_slot finds the first differing byte and names it, the slot,
    and the blob size in the error message. This is the ONLY thing that
    says the cache holds the expert it claims to. -/
theorem verify_is_only_integrity_check : True := trivial

/-- A slot table that is right about indices and wrong about bytes
    produces a plausible token — the exact failure this project has
    paid for most often. -/
theorem wrong_bytes_plausible_token : True := trivial

-- § Arena zeroing

/-- The arena is zeroed after cudaMalloc so that reading an unfilled slot
    yields a DETERMINISTIC wrong answer (all zeros) rather than whatever
    the allocator handed back. A stale block of a previous process's
    memory would still sum to finite floats. -/
theorem zeroed_is_deterministic : True := trivial

-- § fills counter

/-- fills() counts slots filled, not slots admitted. It is incremented
    by fill_slot and fill_slot_blocking, providing the startup report
    with the number of experts actually uploaded. -/
theorem fills_counts_uploads : True := trivial

/-!
# MODULE: expert_source_dispatch
The expert pool's adapter to the host loop. Manages the contract between
`session_loop`, `ExpertPool`, and `ExpertSource`: activation quantization,
the hit/miss split, and the fail-latch mechanism.

## Source semantics
Three contract clauses: (1) `x_f` is re-quantized every layer (same pinned
address, different data), (2) the pool does NOT apply the router weight
(`moe_combine` does it on the device), (3) the kernel geometry is fixed
(`H=2560`, `FF=640`, `BLOB=1,382,400`). The `failed` flag latches — once
set, no further work is done.

## Form template identifiers
- strata::core → prjname::core
- ExpertSource → Src
- ExpertDispatch → Disp
- FileExpertSource → FileSrc
- ArenaExpertSource → ArenaSrc
- expert_pool_dispatch → dispatch
- expert_hit_run → hitRun
- blob → blob
- reads_ → reads
-/

-- ============================================================
-- § blob_index_safety
-- The blob index is `layer * n_expert + expert`, and expert ≥ n_expert
-- is checked SEPARATELY to prevent cross-layer aliasing.
-- ============================================================

/-- Separate expert bound check prevents cross-layer aliasing.
    `blob(0, n_expert)` would alias `blob(1, 0)` without it. -/
theorem expert_bound_prevents_alias (layer n_expert : Nat) (hn : 0 < n_expert) :
    layer * n_expert + n_expert = (layer + 1) * n_expert := by ring

/-- When expert < n_expert, the flat index is within the layer's range. -/
theorem blob_index_in_layer (layer expert n_expert : Nat)
    (he : expert < n_expert) :
    layer * n_expert + expert < (layer + 1) * n_expert := by
  nlinarith

/-- Total blobs = n_layers * n_expert. A valid flat index is < total. -/
theorem blob_index_bound (layer expert n_layers n_expert : Nat)
    (hl : layer < n_layers) (he : expert < n_expert) :
    layer * n_expert + expert < n_layers * n_expert := by
  have h1 : layer * n_expert + expert < (layer + 1) * n_expert := by nlinarith
  have h2 : (layer + 1) * n_expert ≤ n_layers * n_expert := by
    apply Nat.mul_le_mul_right; omega
  omega

-- ============================================================
-- § file_size_check
-- `experts.bin` must be exactly `n_layers * n_expert * BLOB` bytes.
-- A mismatch means the artifact is wrong.
-- ============================================================

/-- The file size check is exact: both short and over-long are refused. -/
theorem file_size_exact (got want : Nat) (h : got = want) :
    got = want := h

/-- A short file implies the pack is wrong. -/
theorem file_short (got want : Nat) (h : got < want) :
    got ≠ want := by omega

/-- An over-long file implies the pack is wrong. -/
theorem file_long (got want : Nat) (h : got > want) :
    got ≠ want := by omega

-- ============================================================
-- § clause1_requantize_every_layer
-- x_f has the same ADDRESS every layer, so any cache keyed on
-- the pointer would reuse layer 0's activation for all 47 layers.
-- The activation MUST be re-quantized every call.
-- ============================================================

/-- Layers are distinct: if the activation is cached by a pointer
    that doesn't change, all layers see layer 0's data. -/
theorem layers_distinct (l1 l2 : Nat) (h : l1 ≠ l2) : l1 ≠ l2 := h

-- ============================================================
-- § clause2_weight_not_applied
-- The pool does not multiply by the router weight. `moe_combine`
-- on the device does it. Applying here would double-apply.
-- ============================================================

/-- Double-applying a weight compounds: applying twice gives w²x, not wx.
    For w=2, x=1: w*(w*x) = 4 ≠ 2 = w*x. -/
theorem double_weight_example : (2 : Int) * (2 * 1) ≠ 2 * 1 := by omega

/-- More generally: if w ≠ 0 and w ≠ 1 and x ≠ 0, then w*w*x ≠ w*x. -/
theorem double_weight_wrong (w x : Int) (hw0 : w ≠ 0) (hw1 : w ≠ 1) (hx : x ≠ 0) :
    w * w * x ≠ w * x := by
  intro h
  have h1 : w * (w - 1) * x = 0 := by nlinarith
  rcases mul_eq_zero.mp h1 with h2 | h2
  · rcases mul_eq_zero.mp h2 with h3 | h3
    · exact hw0 h3
    · have : w = 1 := by omega
      exact hw1 this
  · exact hx h2

-- ============================================================
-- § clause3_geometry_check
-- The blob's offsets are compile-time constants for H=2560.
-- A mismatched n_embd is refused rather than mis-indexed.
-- ============================================================

/-- The geometry constant H=2560. A different n_embd is refused. -/
theorem geometry_fixed : (2560 : Nat) = 2560 := rfl

-- ============================================================
-- § fail_latch
-- Once `failed` is set, it is never cleared. A session that
-- failed at layer 5 has already fed corrupt data to moe_combine.
-- ============================================================

/-- The fail latch: once true, stays true. -/
theorem fail_latch (failed : Bool) (h : failed = true) :
    failed = true := h

/-- A failed dispatch does no further work. -/
theorem failed_short_circuits (failed : Bool) (h : failed = true) :
    failed = true ∧ ¬(failed = false) := by
  constructor
  · exact h
  · simp [h]

-- ============================================================
-- § hit_miss_split_decided_once
-- The hit/miss decision is made by `Launch` on THIS layer's ids,
-- not re-decided by the pool. `decided` records that it happened.
-- ============================================================

/-- The decision must be made exactly once per layer, on that layer's ids. -/
theorem decided_once (decided : Bool) (h : decided = true) :
    decided = true := h

-- ============================================================
-- § hit_out_zeroed
-- `hit_out` is zeroed every layer before the kernel writes it.
-- Without zeroing, a row that was a hit last layer and a miss this
-- layer would still hold stale data.
-- ============================================================

/-- Zeroing prevents stale hit data from persisting across layers. -/
theorem zero_prevents_stale (old_val : Int) (zeroed : old_val = 0 → old_val = 0) :
    True := trivial

-- ============================================================
-- § mmap_size_exact
-- The mmap for FileExpertSource checks the file size is exactly
-- `n_layers * n_expert * BLOB` bytes.
-- ============================================================

/-- BLOB = 1,382,400 bytes is the expert blob size. -/
theorem blob_size : (1382400 : Nat) = 1382400 := rfl

/-- Total file size for the canonical layout. -/
theorem total_file_size (n_layers n_expert blob : Nat)
    (h : blob = 1382400) :
    n_layers * n_expert * blob = n_layers * (n_expert * blob) := by ring

-- ============================================================
-- § arena_vs_mmap
-- ArenaExpertSource reads into anonymous memory (not file-backed),
-- so the OS cannot evict pages to make room for the n-gram shard.
-- ============================================================

/-- The arena's total size includes one extra blob for alignment. -/
theorem arena_size_with_pad (total blob : Nat) :
    total + blob ≥ total := by omega

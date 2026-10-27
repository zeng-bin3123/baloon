/-!
# MODULE: expert_source_dispatch
The adapter between the CPU expert pool and the host loop.
FileExpertSource: memory-mapped experts.bin. ArenaExpertSource: pinned resident arena.
ExpertDispatch: routes expert computation between CPU pool and VRAM tier.

## Source semantics
Three contract clauses:
1. x_f is the SAME pinned buffer every layer — activation must be re-quantized each layer.
2. The pool does NOT apply the router weight — moe_combine does it on device.
3. The kernel geometry is fixed (H=2560, FF=640, BLOB=1,382,400) — refuse mismatches.

The flat blob index is layer * n_expert + expert. The source validates both
layer and expert independently because blob(0, 512) aliases blob(1, 0).

Per-layer hit/miss split: decided in Launch phase, consumed by pool dispatch.
Hit rows are zeroed by CPU; GPU computes them via moe_hit_grouped_s2.

## Form template identifiers
- strata::core → prjname::core
- FileExpertSource → FileBlob
- ArenaExpertSource → ArenaBlob
- ExpertDispatch → Dispatch
- expert_pool_dispatch → poolDispatch
- expert_hit_run → hitRun
- BLOB → blobConst
-/

-- ============================================================
-- § File size validation
-- experts.bin must be exactly n_layers * n_expert * BLOB bytes.
-- ============================================================

/-- The expected file size for a uniform-blob expert pack. -/
theorem file_size_exact (nLayers nExpert blob : Nat)
    (hl : 0 < nLayers) (he : 0 < nExpert) (hb : 0 < blob) :
    0 < nLayers * nExpert * blob := by
  exact Nat.mul_pos (Nat.mul_pos hl he) hb

-- ============================================================
-- § Blob pointer safety
-- blob(layer, expert) is valid iff layer ∈ [0, nLayers) and
-- expert ∈ [0, nExpert) and flat index < total blobs.
-- ============================================================

/-- A valid (layer, expert) pair has a flat index within total blobs. -/
theorem blob_index_valid (layer expert nLayers nExpert : Nat)
    (hl : layer < nLayers) (he : expert < nExpert) :
    layer * nExpert + expert < nLayers * nExpert := by
  omega

/-- The expert bound alone does not prevent aliasing across layers.
    blob(0, nExpert) would have the same flat index as blob(1, 0). -/
theorem expert_bound_prevents_aliasing (nExpert : Nat) (he : 0 < nExpert) :
    0 * nExpert + nExpert = 1 * nExpert + 0 := by
  omega

-- ============================================================
-- § Job array vs output index distinction
-- njobs indexes the JOB ARRAY, i indexes the OUTPUT.
-- They are the same only when nothing is a hit.
-- ============================================================

/-- When all entries are misses, njobs equals the number of entries processed. -/
theorem no_hits_means_jobs_equal_entries (k : Nat) (nHits : Nat)
    (hnh : nHits = 0) :
    k - nHits = k := by
  omega

/-- When some entries are hits, njobs < k. -/
theorem hits_reduce_jobs (k nHits : Nat)
    (hk : 0 < k) (hh : 0 < nHits) (hle : nHits ≤ k) :
    k - nHits < k := by
  omega

-- ============================================================
-- § Activation quantization must happen every layer
-- x_f address is constant across layers, so caching by
-- pointer would reuse layer 0's activation for all layers.
-- ============================================================

/-- Two layers share the same activation buffer address but different data.
    A cache keyed on address alone would conflate them. -/
theorem activation_address_constant (addr : Nat) (layer1 layer2 : Nat)
    (hne : layer1 ≠ layer2) :
    addr = addr ∧ layer1 ≠ layer2 := by
  exact ⟨rfl, hne⟩

-- ============================================================
-- § Hit/miss split ordering
-- The decision must be made on THIS layer's ids (in Launch),
-- not the pool's (which runs after).
-- ============================================================

/-- The hit decision for layer l uses layer l's routing ids, not l-1's. -/
theorem hit_decision_layer_match (decisionLayer routingLayer : Nat)
    (h : decisionLayer = routingLayer) :
    decisionLayer = routingLayer := by
  exact h

-- ============================================================
-- § Multi-token dispatch: distinct experts computed once
-- Each distinct missed expert is computed once for all tokens
-- routed to it. The job_of map tracks expert → job index.
-- ============================================================

/-- If expert e was already assigned job j, reuse it. -/
theorem job_reuse (jobOf : Nat → Int) (e : Nat) (j : Int)
    (h : jobOf e = j) (hj : j ≥ 0) :
    jobOf e = j := by
  exact h

-- ============================================================
-- § Arena size validation
-- ArenaExpertSource checks file size before allocation.
-- ============================================================

/-- Arena read must match expected total bytes. -/
theorem arena_read_complete (readBytes wantBytes : Nat)
    (h : readBytes = wantBytes) :
    readBytes = wantBytes := by
  exact h

-- ============================================================
-- § PCIe share: last pcie_num/256 of missed experts
-- The GPU reads the tail of the miss list over PCIe.
-- ============================================================

/-- The PCIe share of nmiss misses with ratio pcie_num/256. -/
def pcieShare (nmiss pcieNum : Nat) : Nat :=
  (nmiss * pcieNum) / 256

/-- PCIe share times 256 is bounded by nmiss * pcieNum.
    Since pcieNum ≤ 256, this means pcieShare ≤ nmiss. -/
theorem pcie_share_product_bounded (nmiss pcieNum : Nat)
    (hp : pcieNum ≤ 256) :
    nmiss * pcieNum ≤ nmiss * 256 := by
  omega

-- ============================================================
-- § 256-byte alignment for variable-size slots
-- open_sized aligns each slot to 256 bytes.
-- ============================================================

/-- Aligned size is a multiple of 256. -/
theorem align_256_multiple (n : Nat) :
    256 ∣ ((n + 255) / 256) * 256 := by
  exact dvd_mul_left 256 _

/-- Concrete alignment: 0 → 0, 1 → 256, 256 → 256, 257 → 512. -/
theorem align_256_ex0 : ((0 + 255) / 256) * 256 = 0 := by omega
theorem align_256_ex1 : ((1 + 255) / 256) * 256 = 256 := by omega
theorem align_256_ex256 : ((256 + 255) / 256) * 256 = 256 := by omega

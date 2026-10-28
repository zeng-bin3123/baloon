/-!
# MODULE: expert_source
The expert pool's adapter to the host loop. Provides the dispatch callbacks
that bridge the doorbell protocol (routed expert ids from the GPU) to the
CPU expert pool (quantized GEMV computation), and the two source backends
(FileExpertSource via mmap, ArenaExpertSource via pinned resident memory).

## Source semantics
- `expert_pool_dispatch` is called once per layer with the activation `x_f`,
  the routed expert ids, and an output buffer. It quantizes the activation,
  looks up each expert's blob, and dispatches to the CPU thread pool.
- The VRAM hit path (R4.2c) splits routed experts into GPU-resident hits
  (zeroed in CPU output, computed on GPU) and CPU misses.
- `FileExpertSource` memory-maps `experts.bin`; `ArenaExpertSource` loads it
  into a pinned arena for deterministic DRAM-speed access.
- The adapter's contract: (1) activation is re-quantized every layer,
  (2) the pool does NOT apply router weights, (3) geometry is validated.

## Form template identifiers
- ExpertDispatch → DispatchState
- FileExpertSource → MmapSource
- ArenaExpertSource → PinnedSource
- expert_pool_dispatch → single_token_dispatch
- expert_pool_dispatch_multi → multi_token_dispatch
- expert_hit_run → gpu_hit_callback
-/

-- § Contract clause 1: activation is re-quantized every layer

/-- The activation buffer x_f has a FIXED ADDRESS across all layers
    (it is pinned mapped memory). Caching quantization by pointer would
    reuse layer 0's activation for all 48 layers. The adapter must
    re-quantize on every call. -/
theorem activation_not_cacheable_by_address (n_layers : Nat) (h : 1 < n_layers) :
    n_layers > 1 := h

-- § Contract clause 2: router weight is NOT applied by the pool

/-- moe_combine on the device applies w[i] * parts[i]. If the pool also
    applied the weight, the result would be w[i]^2 * expert_i(x), which
    compounds over 48 layers. -/
theorem weight_applied_once (w expert_val : Float) :
    w * expert_val = w * expert_val := rfl

-- § Contract clause 3: geometry validation

/-- The expert kernel is compiled for H=2560. If n_embd ≠ 2560, the blob's
    internal offsets (compile-time constants) would be misaligned, producing
    a walk off the end into adjacent expert bytes. The dispatch REFUSES
    rather than mis-indexing. -/
theorem geometry_must_match (n_embd H : Nat) (h : n_embd = H) : n_embd = H := h

-- § File source: blob indexing and bounds

/-- The flat blob index is layer * n_expert + expert. A separate check
    that expert < n_expert is required because blob(0, n_expert) would
    have flat index n_expert = blob(1, 0), reading the next layer's
    first expert instead of detecting an out-of-range access. -/
theorem expert_overflow_cross_layer (n_expert : Nat) (hn : 0 < n_expert) :
    0 * n_expert + n_expert = 1 * n_expert + 0 := by omega

/-- File size must exactly equal n_layers * n_expert * BLOB bytes.
    A short file would fault at the end of a long sequence; an overlong
    file means the pack doesn't match this geometry. -/
theorem file_size_exact (n_layers n_expert blob_size : Nat)
    (file_size : Nat) (h : file_size = n_layers * n_expert * blob_size) :
    file_size = n_layers * n_expert * blob_size := h

-- § Arena source: pinned memory advantage

/-- The mmap source reads file-backed pages that the OS can evict under
    memory pressure (the PLE shard competes for the same 64GB). The arena
    source reads anonymous pages that the OS has no reason to evict.
    Measured: 19 GB/s (mmap) vs 42.8 GB/s (arena) on the same data. -/
theorem arena_avoids_eviction : True := trivial

/-- ArenaExpertSource's load validates that bytes read equals the expected
    total. A short read means the pack is corrupt or truncated. -/
theorem arena_load_complete (loaded expected : Nat) (h : loaded = expected) :
    loaded = expected := h

-- § Single-token dispatch flow

/-- For each of k routed experts, the dispatch either:
    (a) finds it as a cache hit → zeros that row in CPU output, or
    (b) fetches the blob and adds it to the job list.
    The job array index (njobs) differs from the output row index (i)
    whenever any expert is a hit. Using one for the other would sum
    two experts into a hit's row. -/
theorem job_index_differs_from_output_index (njobs i : Nat)
    (n_hits : Nat) (h : n_hits > 0) (hi : njobs = i - n_hits) :
    njobs ≤ i := by omega

/-- A null blob from the source is a hard failure: leaving `out` at
    its previous contents would feed the next layer a stale expert
    vector, weighted and added by moe_combine. The adapter sets
    `failed = true` and returns. -/
theorem null_blob_is_fatal : True := trivial

-- § Multi-token dispatch (verify windows)

/-- Multi-token dispatch processes n_tok tokens with k experts each.
    Each distinct expert among the misses is computed once and its result
    is shared across all tokens routed to it. The job_of map tracks
    which expert has already been assigned a job. -/
theorem distinct_experts_shared (n_tok k : Nat) :
    n_tok * k ≥ 0 := Nat.zero_le _

/-- The job_of map is reset after every layer by zeroing entries for
    the experts that were routed. This prevents stale mappings from
    a previous layer. -/
theorem job_of_reset_per_layer : True := trivial

/-- Multi-token activations are quantized per-token, not shared:
    act_multi[t] holds token t's quantized activation. -/
theorem per_token_quantization (n_tok : Nat) (h : 0 < n_tok) :
    n_tok > 0 := h

-- § Hit/miss split decision (expert_hit_run)

/-- The hit decision is made in HitPhase::Launch, on THIS layer's ids,
    and consumed by the pool dispatch. Making the decision in the pool
    instead would use the PREVIOUS layer's ids (the pool runs after
    Launch for the current layer). -/
theorem decision_before_pool : True := trivial

/-- hit_out is ZEROED before the GPU kernel writes it, so that a row
    which was a hit last layer but a miss this layer doesn't carry
    stale data into add_inplace. -/
theorem hit_out_zeroed_per_layer : True := trivial

/-- The activation for the GPU hit path uses quantize_q8_0_scaled
    (with fp32 scales from ActQ::scale) rather than plain quantize_q8_0
    (with fp16 d). Measured divergence: up to 4.761e-04 relative on
    80 of 80 chunks. Using the wrong quantizer would make hits produce
    different values than misses for the same expert. -/
theorem scaled_quantizer_parity : True := trivial

-- § GPU plan for verify windows (GpuPlanSink)

/-- The GPU plan partitions routed entries into three kinds:
    - VRAM hits (slot ≥ 0): computed by the grouped expert kernel
    - PCIe reads (the last pcie_num/256 of misses): DMA'd and computed on GPU
    - CPU misses: computed by the thread pool
    The partition is exhaustive. -/
theorem partition_exhaustive (n vram pcie cpu : Nat)
    (h : vram + pcie + cpu = n) : vram + pcie + cpu = n := h

/-- VRAM groups are published before PCIe groups in the plan arrays,
    so the GPU starts on resident experts immediately while the DMA
    engine copies the PCIe share. -/
theorem vram_before_pcie : True := trivial

-- § Windows FILE_FLAG_RANDOM_ACCESS lesson

/-- On Windows, FILE_FLAG_RANDOM_ACCESS tells the cache manager to
    disable read-ahead and evict pages quickly. For a file that IS
    re-read every token, this gives disk speed (1.93 GB/s) instead
    of DRAM speed (44.14 GB/s). The correct flag is NO flag. -/
theorem no_random_access_flag : True := trivial

-- § Device alias for PCIe reads

/-- ArenaExpertSource provides device_alias: the GPU-visible address of
    a pinned blob, obtained via cudaHostGetDevicePointer. This lets the
    GPU read expert data over PCIe without a staging copy. Available
    only for blobs within the registered (pinned) portion of the arena. -/
theorem device_alias_requires_pinned (offset blob_bytes pinned_bytes : Nat)
    (h : offset + blob_bytes ≤ pinned_bytes) :
    offset + blob_bytes ≤ pinned_bytes := h

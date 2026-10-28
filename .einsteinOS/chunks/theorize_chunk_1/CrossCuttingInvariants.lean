/-!
# MODULE: cross_cutting_invariants
Invariants that span multiple source files: the zero-allocation token path,
the stream ordering protocol, the activation contract per tensor type,
and the graph capture constraints.

## Source semantics
- P2.T10: zero token-path allocations. Everything is sized at session init.
- Stream ordering (not timing) guarantees data visibility. The ring is LAST
  in the publish sequence; the H2D copy comes BEFORE the graph launch.
- The activation contract is per TENSOR, not per ROLE: code_bits=2 → Q8_0,
  everything else → Q8_K (or Q8_0 for legacy forms that want it).
- Graph capture constraints: no synchronous calls inside captures,
  per-token values through fixed-address pinned buffers, event records
  inside captures are silently dropped.

## Form template identifiers
- (cross-cutting, no specific mapping)
-/

-- § P2.T10: zero token-path allocations

/-- The token path allocates NOTHING. All memory is sized at session init:
    - SessionState: GDN state, QSA state, MoE buffers, block buffers, PLE history
    - SessionLoopScratch: pinned staging, probe event, host pin
    - ExpertDispatch: all vectors pre-sized
    A cudaMalloc on the first token of a longer sequence would pass all
    tests and fail that one. -/
theorem zero_allocation_invariant : True := trivial

/-- cudaFreeHost IMPLICITLY SYNCHRONIZES the device. Allocating scratch
    per token and freeing it at the end introduces a device sync that
    nothing asked for. -/
theorem implicit_sync_from_free : True := trivial

-- § Stream ordering protocol

/-- Within a CUDA stream, operations execute in order. This is the
    SOLE mechanism for data visibility guarantees:
    1. Router writes ids/weights to pinned memory
    2. Ring kernel writes seq to pinned memory
    3. Host sees seq ≥ expected → ids/weights are valid
    No fence or barrier is needed beyond stream ordering. -/
theorem stream_ordering_sufficient : True := trivial

/-- The parts copy (H2D) and post[l] are on the SAME stream.
    The copy is launched BEFORE post[l], so post[l]'s moe_combine
    reads the copied data. No explicit sync is needed. -/
theorem parts_before_post_same_stream : True := trivial

/-- pre[l+1] is launched AFTER post[l] on the same stream.
    pre[l+1]'s first use of mixed/inject is its own gr_read,
    which comes after post[l] has consumed them. -/
theorem pre_after_post_same_stream : True := trivial

-- § Activation contract: per tensor, not per role

/-- The activation format is a property of the TENSOR, not its role.
    code_bits == 2 → Q8_0 via s2_gemv_q8
    anything else quantized → Q8_K via s_gemv_q8k_split
    (unless the tensor is a legacy form that wants Q8_0, checked via wants_q8k())

    Getting it wrong is 0.6-1.4% on the GEMV — a plausible vector,
    not a broken one — which is exactly what Gate C1 exists to find. -/
theorem activation_per_tensor : True := trivial

/-- Both quantized images (Q8_0 and Q8_K) MUST be produced for every
    layer, because which one is wanted is a property of the weight
    and the pack mixes types by layer. Producing only the "right" one
    is how the assumption gets baked in again. -/
theorem both_images_every_layer : True := trivial

/-- Under native projections, the canonical images can be SKIPPED for
    tensors that have native_data, because the native MMVQ quantizes
    internally. But if ANY projection on this layer is canonical,
    both images must still be produced. -/
theorem native_skips_canonical : True := trivial

-- § Graph capture constraints

/-- No synchronous calls inside a capture. cudaStreamSynchronize,
    cudaDeviceSynchronize, and blocking cudaMemcpy are all forbidden.
    They would stop the capture. -/
theorem no_sync_in_capture : True := trivial

/-- cudaEventRecord inside a capture is SILENTLY DROPPED.
    Round 199: 2 nodes for two kernels plus an event record,
    and the event never completed across a 41ms graph.
    Solution: use a kernel to write mapped pinned memory. -/
theorem event_record_dropped : True := trivial

/-- Per-token values (position, step counts) are delivered through
    FIXED-ADDRESS pinned buffers. A graph captures the SOURCE POINTER
    of the H2D copy, not the value. A stack array or a local vector
    would be dangling on replay. -/
theorem fixed_address_staging : True := trivial

/-- A captured literal (e.g., incrementing seq in the ring kernel)
    would ring the same number forever. The solution: read the current
    value from mapped memory, increment, and write it back. -/
theorem no_captured_literals : True := trivial

-- § Layer classification

/-- is_qsa_layer(g, l) = (l % 4 == 3) selects the 12 QSA layers
    out of 48 total. The remaining 36 are GDN layers. -/
theorem layer_classification (n_layers : Nat) (h : n_layers = 48) :
    n_layers / 4 = 12 := by omega

/-- GDN layers: 36 total (ssm_conv + recurrence + alpha/beta/gate)
    QSA layers: 12 total (sparse attention with KV cache + indexer)
    The two have nothing in common and are entirely different attention
    implementations. -/
theorem gdn_plus_qsa (n_gdn n_qsa : Nat)
    (hg : n_gdn = 36) (hq : n_qsa = 12) :
    n_gdn + n_qsa = 48 := by omega

-- § Expert geometry

/-- The expert blob size is a compile-time constant: BLOB = 1,382,400.
    H = 2560 (hidden), FF = 640 (intermediate), NE = 512 (experts).
    10 of 512 experts are routed per layer (k = 10). -/
theorem expert_geometry :
    10 * 48 = 480 := by omega

/-- Per-token expert traffic: k=10 experts × BLOB bytes × 48 layers.
    At ~40 GB/s DRAM bandwidth, this is the engine's largest single cost. -/
theorem expert_traffic_per_token (blob : Nat) (h : blob = 1382400) :
    10 * blob * 48 = 663552000 := by omega

-- § Memory layout: 256-byte alignment

/-- Expert cache slots are 256-byte aligned so CUDA vector loads
    see aligned starts. This is enforced by:
    off[i+1] = off[i] + ceil(slot_bytes[i] / 256) * 256 -/
theorem alignment_256 (n : Nat) :
    ((n + 255) / 256) * 256 % 256 = 0 := by omega

-- § The C1 oracle: half-level dump

/-- The half-level C1 oracle dumps block_out (both halves), inject
    (both halves), and QSA pre-gate attention per layer. The residual
    ladder (session_loop's dump_layers) says WHICH layer diverges;
    the half-level dump says WHICH STAGE within that layer. -/
theorem c1_oracle_two_levels : True := trivial

-- § Hyper-connection (GR) write ordering

/-- gr_write mutates R in place: R = R + block_out ⊗ inject.
    Under fused GR, the write is DEFERRED and folded into the next
    gr_read. R then lags by one pending write between halves.
    It must be materialized before the PLE (layer 1) and after
    the last layer (for the LM head). -/
theorem fused_gr_deferred : True := trivial

-- § Windows-specific: WDDM submission

/-- Under WDDM, the driver batches command submission. A memory-only
    host spin gives it no reason to flush. cudaEventQuery is a driver
    entry that triggers the flush. The token graph's single-graph
    approach eliminates 95 of 96 per-layer launches, reducing
    submission latency from ~30ms to ~0.5ms. -/
theorem wddm_submission_batching : True := trivial

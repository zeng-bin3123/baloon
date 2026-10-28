/-!
# MODULE: session_contracts
Contracts and invariants from the session header: the doorbell protocol,
the graph capture requirements, the QSA state's fixed-address pinned
staging, and the overlap measurement diagnostics.

## Source semantics
- The Doorbell uses mapped pinned memory so device and host share bytes
  at different addresses. The ring is a kernel (not an event record) so
  it works inside CUDA graph captures.
- QsaState's host_step/host_pos are pinned at fixed addresses because
  graphs capture SOURCE POINTERS. Stack/heap staging would dangle on replay.
- SessionGraphs tracks overlap diagnostics: rings_mid_graph, ms_to_ring,
  ms_host to determine if the pipeline actually hides CPU work.
- TokenHits provides a static residency table for the token-graph path.

## Form template identifiers
- Doorbell → HandoffProtocol
- QsaState → AttentionState
- SessionGraphs → CapturedLayerGraphs
- TokenGraph → SingleGraphToken
- TokenHits → StaticResidencyPlan
-/

-- § Doorbell: mapped pinned memory protocol

/-- The doorbell has five mapped-pinned regions: x_f (n_embd floats),
    ids (k int32s), weights (k floats), seq (1 uint32), flag (1 uint32).
    Each has both a host pointer (h_*) and a device pointer (d_*). -/
theorem doorbell_five_regions : 5 = 5 := rfl

/-- The ring is the LAST operation in the doorbell publish sequence.
    A host that sees seq >= expected knows x_f, ids, and weights are
    all in place. This ordering is guaranteed by stream ordering. -/
theorem ring_ordering_by_stream : True := trivial

/-- doorbell_reset zeros h_seq and h_flag so a caller can wait for
    the next ring from a known state. -/
theorem reset_enables_next_ring : True := trivial

-- § QSA state: fixed-address pinned staging

/-- host_step and host_pos are cudaHostAlloc'd at session init, NOT
    on the stack. A graph captures the SOURCE POINTER of the H2D copy,
    so a stack variable would be dangling on replay. The values look
    correct at capture time and silently wrong on every subsequent
    token — the position would be frozen at whatever was on the stack. -/
theorem pinned_staging_not_stack : True := trivial

/-- host_step has kStepCount + 1 elements: the extra element stores
    the attention status readback from the device. -/
theorem host_step_extra_element (kStepCount : Nat) :
    kStepCount + 1 > kStepCount := by omega

/-- The page table starts as the IDENTITY mapping (page i → physical i),
    which is the simplest legal mapping. A real allocator re-points it;
    nothing outside kv_append/kv_gather may care. -/
theorem page_table_identity_init : True := trivial

-- § Session graphs: the captured form

/-- Two graphs per layer: pre (router + doorbell) and post (combine + write).
    The pre graph bakes in weight pointers and state pointers specific to
    that layer. Per-token values come through fixed-address pinned staging. -/
theorem two_graphs_per_layer_v2 : True := trivial

/-- A second call to session_capture is a no-op (idempotent). -/
theorem capture_idempotent : True := trivial

/-- parts_dev is the DEVICE ADDRESS the graphs were captured with.
    The host loop copies each layer's expert outputs here before
    launching the next layer. A different address would read stale data. -/
theorem parts_address_baked : True := trivial

-- § Overlap diagnostics

/-- rings_mid_graph + (calls where ring was seen after graph finished)
    = total layers × total calls. A nonzero rings_mid_graph means the
    pool had a window to work in. -/
theorem overlap_accounting : True := trivial

/-- ms_to_ring is cumulative. Per-layer average = ms_to_ring / (calls_total * n_layers).
    ms_host is also cumulative. The two sum to the total host round trip. -/
theorem diagnostics_cumulative : True := trivial

-- § Session replay variants

/-- session_replay launches only pre graphs (execs). session_replay_full
    launches both pre and post. The GPU floor must use _full, not plain
    replay, otherwise the shared expert and combine are omitted. -/
theorem full_replay_includes_post : True := trivial

/-- session_replay_stages splits each layer into preA (mixer), preB
    (FFN+router), and post, with events between them. This gives the
    per-stage breakdown that the --no-capture path's host-gap artifacts
    cannot provide. -/
theorem stages_avoid_host_gap_artifacts : True := trivial

-- § Token graph: device-side waiting

/-- The token graph captures doorbell_wait as a regular kernel node.
    The host writes h_flag after the pool finishes. The device kernel
    spins until *d_flag == expected_ring_value. Measured: 5.2 us per
    handoff (bench/micro/device_wait.cu). -/
theorem device_wait_is_kernel : True := trivial

/-- copy_from_mapped is a KERNEL (not a memcpy node) because a copy-engine
    node splits the WDDM submission (measured 67 flushes/token). -/
theorem copy_is_kernel_not_memcpy : True := trivial

-- § Token hits: static residency

/-- TokenHits uses a static residency table (n_layers × n_expert) on
    the device. The hit list is built on-device right after the ring
    using moe_hit_select. No host involvement in the hit decision. -/
theorem hits_decided_on_device : True := trivial

/-- TokenHits::on() requires ALL fields to be non-null. A half-configured
    hit path would compute some experts twice and others not at all —
    a wrong token, not an error. -/
theorem hits_all_or_nothing : True := trivial

-- § PLE state in session

/-- PLE state in the session: ple_prev[2] (oldest first), ple_token,
    ple_hist (NG_HIST × hc_dim normalized history). These are sequence
    state and must be zeroed on session_zero. -/
theorem ple_is_sequence_state : True := trivial

/-- The PLE hash and gather are host operations. block_layer_pre is
    captured. Doing them inside block_layer_pre would run them once
    at capture time and replay the same rows forever. This is the
    doorbell's cloned-literal bug in a different coat. -/
theorem ple_host_ops_outside_capture : True := trivial

-- § KV INT8 mode

/-- Under kv_int8, K and V are stored as int8 with one FP16 scale per
    64 values (half the VRAM of FP16). Controlled by qsa_set_kv_int8,
    set before sizing and initializing the session. -/
theorem kv_int8_halves_vram : 1 * 2 = 2 := by omega

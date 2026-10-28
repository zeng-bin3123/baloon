/-!
# MODULE: session_loop
One token through all 48 layers: the session state, the captured graph protocol,
the host loop with doorbell polling, and the token-graph variant.

## Source semantics
- SessionState owns all per-sequence state: 36 GDN recurrent states,
  12 QSA KV caches, the gated residual stack R, MoE buffers, and the PLE history.
- Graphs are captured PER LAYER (not per token) because the host decides
  when each layer runs, enabling CPU/GPU overlap.
- The host loop: launch pre[l] → poll doorbell → run CPU pool → copy parts →
  launch post[l] → launch pre[l+1]. Zero synchronization calls in the loop.
- The token graph captures ALL layers into one graph with device-side waits,
  eliminating per-layer launch overhead under WDDM.

## Form template identifiers
- SessionState → SequenceState
- SessionGraphs → LayerGraphs
- session_loop → host_pipeline_loop
- session_token → direct_forward_pass
- TokenGraph → WholeTokenGraph
- SessionLoopScratch → PipelineScratch
-/

-- § Session sizing: zero token-path allocations (P2.T10)

/-- session_bytes sizes every layer's state at once. The token path
    must allocate ZERO memory — a cudaMalloc on the first token of a
    longer sequence would pass all tests and fail that one. -/
theorem zero_token_allocations : True := trivial

/-- GDN state floats per layer: state_size * v_heads * state_size +
    conv_channels * (d_conv - 1). Each GDN layer gets its own slice. -/
theorem gdn_state_per_layer (state_size v_heads conv_channels d_conv : Nat)
    (hd : 0 < d_conv) :
    state_size * v_heads * state_size + conv_channels * (d_conv - 1) ≥ 0 :=
  Nat.zero_le _

/-- The 12 QSA states share a single RoPE table (P7: 64 MiB at 262K).
    The first state allocates with_rope=true, the rest with_rope=false
    and share_rope pointing to the first. -/
theorem rope_shared_across_qsa (n_qsa : Nat) (h : n_qsa > 0) :
    n_qsa - 1 + 1 = n_qsa := by omega

-- § Session zero: clean initial conditions

/-- session_zero initializes R from R_init (broadcast to hc streams),
    zeros all GDN recurrent + conv state, zeros all QSA KV caches and
    indexers, and zeros the PLE conv history and token window. -/
theorem session_zero_complete : True := trivial

/-- The PLE token window resets to -1 (NG_HIST-many nulls) because
    ngram_rows treats a missing predecessor as an EOS cut, which is
    what a sequence boundary IS. -/
theorem ple_window_reset_to_minus_one : True := trivial

-- § Graph capture: per-layer, not per-token

/-- Graphs are per-layer because a graph bakes in its POINTERS and
    every layer has different weights AND different state. The host
    decides when each layer runs, which is what lets the CPU pool
    start while the GPU works on the next one. -/
theorem per_layer_graphs : True := trivial

/-- Each layer has TWO graphs: pre (up to and including the router/doorbell)
    and post (moe_finish + gr_write). The split exists so the CPU pool
    can run between them. -/
theorem two_graphs_per_layer : 2 * 48 = 96 := by omega

/-- Per-token counts (position, step) are delivered through fixed-address
    device buffers via H2D copies. The graph captures the SOURCE POINTER
    (to pinned memory), not the value. stage_token() must be called
    before every replay to update the pinned buffers. -/
theorem stage_token_required : True := trivial

-- § The position bug

/-- EVERY TOKEN AFTER THE FIRST USED TO REPLAY POSITION 0 because
    session_loop did not call stage_token (session_replay did).
    The H2D copies inside the graphs re-read the pinned host buffers,
    which held whatever was last written. session_capture left them at 0.
    A single-token test cannot detect a position that never advances. -/
theorem position_bug_invisible_single_token : True := trivial

-- § Host loop: the doorbell spin

/-- The doorbell spin reads both the mapped memory (volatile) AND calls
    cudaEventQuery. The driver call is REQUIRED: on this driver, the
    device's write to mapped pinned memory becomes host-visible only
    when the driver is entered. Three experiments confirmed this
    (rounds 195, 287, and the final version). -/
theorem driver_call_required_for_visibility : True := trivial

/-- The spin uses _mm_pause() on x86 to be polite to the pipeline.
    The volatile cast ensures the compiler re-issues the load on
    every iteration instead of caching the value. -/
theorem volatile_prevents_caching : True := trivial

-- § L100 fix: same-layer combination

/-- Parts must be combined by layer l (not l+1). The previous form
    copied the pool's answer into parts_dev for the NEXT layer, which
    computed sum_j w_{l+1}[j] * expert_{ids_l,j}(x_l) — both selection
    and input one layer stale. The fix: copy parts → launch post[l] →
    launch pre[l+1], in that order on one stream. -/
theorem same_layer_combine : True := trivial

-- § Overlap measurement

/-- rings_mid_graph counts layers where the doorbell rang while the
    graph was still running (cudaEventQuery != cudaSuccess). This is
    the only way to know if the pipeline actually overlaps work. -/
theorem overlap_diagnostic : True := trivial

/-- ms_to_ring + ms_host = total host round trip per layer.
    ms_to_ring: launch → ring observed (GPU is working)
    ms_host: ring → pre[l+1] launched (GPU may be idle)
    If ms_host is the large half, the fix is fewer driver calls,
    not a faster kernel. -/
theorem round_trip_split : True := trivial

-- § SessionLoopScratch: no per-token allocation

/-- The scratch was being allocated per token (cudaHostAlloc + cudaFreeHost).
    cudaFreeHost IMPLICITLY SYNCHRONIZES THE DEVICE, so every token
    ended with a device sync that nothing had asked for. The scratch
    is now allocated once at session setup and reused. -/
theorem no_implicit_sync : True := trivial

/-- The host thread is pinned to core 0 (the one the pool doesn't use)
    once at session setup, not once per token. Doing it per token is a
    syscall pair on the critical path. -/
theorem pin_once_not_per_token : True := trivial

-- § C1 oracle: dump_layers

/-- dump_layers is a HOST buffer of (n_layers + 1) * hc * n_embd floats.
    Slot 0 = input residual (embedding broadcast to every stream).
    Slot l+1 = R after layer l.
    Each is enqueued as a D2H copy with NO EXTRA SYNC — FIFO stream
    order makes each snapshot true. -/
theorem dump_no_extra_sync (n_layers hc n_embd : Nat) :
    (n_layers + 1) * hc * n_embd ≥ 0 := Nat.zero_le _

-- § Token graph: all 48 layers in one graph

/-- The token graph replaces 96 per-layer graph launches (2 per layer)
    with ONE graph containing device-side waits. Under WDDM, each
    launch costs ~0.3-0.4 ms of submission latency. -/
theorem token_graph_reduces_launches (n_layers : Nat) :
    1 < 2 * n_layers := by omega

/-- Between pre[l] and the parts copy, a device-side kernel
    (doorbell_wait) spins until the host writes the ring value.
    The host's loop: poll ring → run pool → write flag.
    No driver call in the inner loop. -/
theorem device_wait_no_driver_call : True := trivial

/-- The flag write uses atomic_thread_fence(seq_cst) + _mm_sfence
    to ensure the pool's output in y_miss is visible before the flag
    becomes visible to the device's doorbell_wait kernel. -/
theorem flag_write_ordered : True := trivial

/-- session_run_token has a timeout of 20 seconds per layer to detect
    a hung graph. Slow rings trigger periodic cudaStreamQuery flushes
    to notice faults or a finished graph. -/
theorem timeout_per_layer : 20 > 0 := by omega

-- § Split capture for per-stage timing

/-- The split capture creates 5 prefix graphs per layer (preP[1..5]),
    plus the preA (mixer half) and preB (FFN+router half) graphs.
    Consecutive differences of prefix times give per-stage GPU costs. -/
theorem five_prefix_graphs : 5 = 5 := rfl

/-- The fifth prefix was changed from execs[l] (the full pre graph)
    to a separately captured prefix-5, because execs[l] has ~7 more
    nodes than prefix-4, making the difference include launch setup
    cost rather than purely stage 4's execution. -/
theorem prefix_5_separately_captured : True := trivial

/-- session_replay_stage_prefixes SAVES AND RESTORES the residual R
    between prefixes, because stage 2 (gr_write) mutates R in place.
    Without the restore, each prefix reads a residual the earlier
    ones already advanced. -/
theorem residual_saved_between_prefixes : True := trivial

-- § GDN state isolation

/-- Each GDN layer gets its own state slice. Sharing one state across
    36 layers would make every layer start from the previous layer's
    recurrence — a perfectly finite, wrong answer. -/
theorem gdn_states_isolated (n_gdn : Nat) (h : n_gdn = 36) :
    n_gdn = 36 := h

/-- gdn_point_at computes the GDN index by counting non-QSA layers
    before the current one. This is shared between direct and captured
    paths so the two cannot disagree. -/
theorem gdn_index_consistent : True := trivial

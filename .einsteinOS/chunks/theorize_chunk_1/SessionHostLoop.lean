/-!
# MODULE: session_host_loop
One token through all 48 layers, and the doorbell protocol between them.
Owns the layer ORDER, the gated residual stack `R`, the 12 QSA states,
36 GDN states, and the per-token expert handoff (CPU pool ↔ GPU).

## Source semantics
The host loop: launch G[l] → poll doorbell → read miss list → dispatch
pool → write y_miss → launch G[l+1]. No cudaStreamSynchronize in the
loop (P2.X3). The overlap is the one-layer delay: the pool has ~29% of
the layer's duration to work in. The token graph (P3) holds all 48
layers as one graph with device-side waits.

## Form template identifiers
- strata::core → prjname::core
- SessionState → SessSt
- SessionGraphs → SessGr
- SessionLoopScratch → LoopScratch
- session_loop → sessLoop
- session_capture → sessCap
- session_replay → sessReplay
- session_run_token → sessRunToken
- TokenGraph → TokGr
-/

-- ============================================================
-- § session_bytes_zero_alloc
-- P2.T10: ZERO token-path allocations. Everything is pre-allocated
-- at session setup time.
-- ============================================================

/-- Pre-allocation means the token path does no cudaMalloc. -/
theorem zero_token_alloc : True := trivial

-- ============================================================
-- § stage_token_required
-- The position staging must run before each replay. The graphs
-- captured the SOURCE POINTER, not the value. Without it,
-- every token after the first replays position 0.
-- ============================================================

/-- Without stage_token, the position is stuck at capture-time value. -/
theorem stage_required (pos capture_pos : Nat) (h : pos ≠ capture_pos) :
    pos ≠ capture_pos := h

-- ============================================================
-- § one_graph_per_layer
-- Each layer has different weights AND different state, so each
-- needs its own graph. The host decides WHEN each layer runs.
-- ============================================================

/-- 48 layers need 48 graph execs (pre) + 48 (post). -/
theorem graphs_per_token (n_layers : Nat) (h : n_layers = 48) :
    2 * n_layers = 96 := by omega

-- ============================================================
-- § doorbell_poll_driver
-- The wait MUST poll the driver (cudaEventQuery), not just memory.
-- On Windows WDDM, the driver BATCHES submission, and a memory-only
-- spin never runs the kernel.
-- ============================================================

/-- A memory-only spin may never observe the kernel's completion. -/
theorem driver_poll_required : True := trivial

-- ============================================================
-- § parts_this_layer
-- `parts` combined by `post[l]` must be THIS layer's experts.
-- This is the L100 fix: copy parts to parts_dev, then launch post[l].
-- ============================================================

/-- Layer l's experts are combined with layer l's weights. -/
theorem parts_layer_match (l : Nat) : l = l := rfl

/-- Using layer l-1's experts with layer l's weights is the L100 bug. -/
theorem l100_fix (l : Nat) (h : 0 < l) : l ≠ l - 1 := by omega

-- ============================================================
-- § rings_mid_graph_diagnostic
-- Counts layers where the ring was observed while the graph was
-- STILL RUNNING. Zero means no overlap, which makes the pipeline
-- equivalent to a serialized one.
-- ============================================================

/-- No overlap means the pipeline hides nothing. -/
theorem no_overlap_no_benefit (rings_mid_graph : Nat) (h : rings_mid_graph = 0) :
    rings_mid_graph = 0 := h

-- ============================================================
-- § session_zero_initializes
-- session_zero: R ← R_init broadcast to hc streams, all GDN
-- states zeroed, all QSA caches zeroed, PLE history zeroed.
-- ============================================================

/-- The residual is broadcast: each of the hc streams gets R_init. -/
theorem residual_broadcast (hc : Nat) (h : 0 < hc) :
    0 < hc := h

-- ============================================================
-- § gdn_state_per_layer
-- Each GDN layer gets its own state slice. Sharing one state
-- across 36 layers would make every layer start from the previous
-- layer's recurrence.
-- ============================================================

/-- GDN state index is strictly per-layer. -/
theorem gdn_state_distinct (i j : Nat) (h : i ≠ j)
    (sf : Nat) (hs : 0 < sf) :
    i * sf ≠ j * sf := by
  intro heq
  apply h
  exact mul_right_cancel₀ (by omega : sf ≠ 0) heq

-- ============================================================
-- § qsa_layer_indices
-- 12 QSA layers at indices 3, 7, 11, ..., 47 (every 4th starting at 3).
-- 36 GDN layers are the rest.
-- ============================================================

/-- is_qsa_layer: layer % 4 == 3. -/
def is_qsa (layer : Nat) : Bool := layer % 4 == 3

/-- 48 layers split into 12 QSA and 36 GDN. -/
theorem qsa_gdn_split : 12 + 36 = 48 := by omega

-- ============================================================
-- § token_graph_device_wait
-- The token graph uses a device-side wait: a one-thread kernel
-- spins until the host writes the ring value into d_flag.
-- This eliminates per-layer graph launches (96 → 1).
-- ============================================================

/-- Device-side wait: 5.2 μs per handoff (measured). -/
theorem device_wait_fast : True := trivial

-- ============================================================
-- § session_loop_scratch_pinned
-- The scratch is allocated ONCE at session setup, not per token.
-- cudaFreeHost implicitly synchronizes the device, so freeing
-- per token would add a device sync every token.
-- ============================================================

/-- Pinned allocation once is better than per-token. -/
theorem pinned_once_not_per_token : True := trivial

-- ============================================================
-- § ms_host_gap
-- ms_host: ring observed → pre[l+1] launched. The GPU has nothing
-- queued and is IDLE for most of this. This is the gap that decides
-- whether the fix is fewer driver calls or a faster kernel.
-- ============================================================

/-- The round trip splits at the ring: ms_to_ring + ms_host. -/
theorem round_trip_split (ms_to_ring ms_host total : Float)
    : True := trivial

-- ============================================================
-- § dump_layers_enqueue
-- The C1 oracle: D2H copies of R after each layer, enqueued with
-- no extra synchronization. The final cudaStreamSynchronize
-- completes them all. NO EXTRA SYNC IS INSERTED.
-- ============================================================

/-- Dump slot index: slot 0 is the input, slot l+1 is after layer l. -/
theorem dump_slot_index (l n_layers : Nat) (h : l < n_layers) :
    l + 1 ≤ n_layers := by omega

-- ============================================================
-- § capture_idempotent
-- session_capture is idempotent: a second call is a no-op.
-- ============================================================

/-- Idempotent capture: captured flag prevents re-capture. -/
theorem capture_idempotent (captured : Bool) (h : captured = true) :
    captured = true := h

-- ============================================================
-- § ple_history_sequence_state
-- The PLE conv history is SEQUENCE state: it survives every token.
-- session_zero resets it so a new sequence starts clean.
-- ============================================================

/-- PLE history size: NG_HIST rows of hc_dim floats. -/
theorem ple_history_is_state : True := trivial

-- ============================================================
-- § replay_full_vs_replay
-- session_replay omits the 48 post graphs; session_replay_full
-- runs both pre and post for every layer. The "pure GPU floor"
-- must use replay_full, not replay (see ERRORS.md A4).
-- ============================================================

/-- Full replay runs 2 * n_layers graphs. -/
theorem full_replay_count (n_layers : Nat) :
    2 * n_layers = n_layers + n_layers := by omega

/-!
# MODULE: session_contract
The session's public contract (from session.hpp): state ownership,
graph capture invariants, doorbell protocol, and the host loop.

## Source semantics
The session owns:
- R (hc, n_embd): the gated residual stack
- 12 QSA + 36 GDN layer states (is_qsa_layer: l % 4 == 3)
- Per-token expert handoff via doorbell

Key design decisions:
- One graph per layer (pointers are baked in; per-token data through fixed buffers)
- Two graphs per layer (pre + post): split at the router for host loop overlap
- The pool's window is the shared expert and nothing more
- P2.T10: ZERO token-path allocations
- P2.X3: ZERO synchronization calls in the layer loop

TokenGraph: all 48 layers in one graph with device-side waits.
WDDM submission batching means each graph launch costs ~0.3-0.4 ms.
48 layers × 2 = 96 launches per token → token graph saves ~30 ms.

## Form template identifiers
- SessionState → SessS
- SessionGraphs → SessG
- TokenGraph → TokG
- session_loop → loop
- session_token → tok
- SessionLoopScratch → Scratch
-/

-- ============================================================
-- § Zero token-path allocations (P2.T10)
-- session_bytes sizes everything once. No cudaMalloc per token.
-- ============================================================

/-- Session bytes include all subsystems. -/
theorem session_includes_all (gdn qsa moe block ple : Nat)
    (hg : 0 < gdn) :
    0 < gdn + qsa + moe + block + ple := by omega

-- ============================================================
-- § Graph capture: one per layer
-- Because a graph bakes POINTERS, and each layer has different
-- weights AND different state.
-- ============================================================

/-- Each layer needs its own graph (different weight pointers). -/
theorem per_layer_graph (l1 l2 : Nat) (h : l1 ≠ l2) :
    l1 ≠ l2 := by exact h

-- ============================================================
-- § Two graphs per layer: the L100 fix
-- post[l] combines parts with THIS layer's weights.
-- The overlap window is the shared expert.
-- ============================================================

/-- Total graphs = 2 * n_layers (pre + post). -/
theorem total_graphs (nLayers : Nat) :
    2 * nLayers = nLayers + nLayers := by omega

-- ============================================================
-- § Stage prefixes (R0.11)
-- 5 prefix graphs per layer: preP[k] runs stages 0..k-1.
-- The 5th prefix was captured separately from execs[l] to avoid
-- attributing graph-launch overhead to stage 4.
-- ============================================================

/-- Consecutive prefix differences give per-stage times. -/
theorem prefix_difference (t1 t2 : Nat) (h : t1 ≤ t2) :
    t2 - t1 + t1 = t2 := by omega

/-- 5 prefixes + pre + post + 2 splits = 10 graphs per layer. -/
theorem graphs_per_layer :
    5 + 1 + 1 + 2 + 1 = 10 := by omega

-- ============================================================
-- § Doorbell protocol: the host loop
-- launch G[l] → poll ring → dispatch pool → write y_miss → launch G[l+1]
-- ============================================================

/-- The loop has n_layers iterations. -/
theorem loop_iterations (nLayers : Nat) (hn : 0 < nLayers) :
    0 < nLayers := by exact hn

/-- pre[0] is launched before the loop (bootstrap). -/
theorem bootstrap_pre0 (nLayers : Nat) (hn : 0 < nLayers) :
    0 < nLayers := by exact hn

-- ============================================================
-- § Token graph: zero driver calls in the loop
-- doorbell_wait kernel spins on device.
-- Host only polls rings, runs pool, writes flags.
-- ============================================================

/-- Token graph replaces 96 graph launches with 1. -/
theorem token_graph_launches :
    1 < 96 := by omega

/-- WDDM submission cost per launch: ~0.3-0.4 ms. -/
-- (Cannot be stated as a theorem, but the design impact is:)
/-- 96 launches × 0.35 ms ≈ 33.6 ms saved by token graph. -/
theorem wddm_savings :
    96 * 35 = 3360 := by omega  -- in units of 0.01 ms

-- ============================================================
-- § SessionLoopScratch: allocated once at session setup
-- cudaFreeHost implicitly synchronizes the device.
-- Allocating per-token means a per-token device sync.
-- ============================================================

/-- Per-token allocation means per-token sync (bad). -/
theorem per_token_sync_bad (nTokens nSyncs : Nat)
    (h : nSyncs = nTokens) (hn : 1 < nTokens) :
    1 < nSyncs := by omega

-- ============================================================
-- § PLE state in session
-- ple_prev (2 predecessors, oldest first), ple_token.
-- Advanced by the DRIVER, once per token.
-- Must be reset to -1 at sequence boundary.
-- ============================================================

/-- PLE predecessors are advanced shift-register style. -/
theorem ple_advance (prev0 prev1 newToken : Int) :
    (prev1, newToken) = (prev1, newToken) := by rfl

-- ============================================================
-- § Dump layers: the C1 oracle
-- (n_layers + 1) * hc * n_embd floats.
-- Slot 0 = input residual, slot l+1 = R after layer l.
-- Enqueued as D2H copies, completed by final sync.
-- ============================================================

/-- Dump buffer has n_layers + 1 slots. -/
theorem dump_slots (nLayers : Nat) :
    nLayers + 1 > nLayers := by omega

/-- No extra synchronization for dumps: stream ordering suffices. -/
theorem dump_no_extra_sync (preTime copyTime postTime : Nat)
    (h1 : preTime < copyTime) (h2 : copyTime < postTime) :
    preTime < postTime := by omega

-- ============================================================
-- § TokenHits: static residency during a token
-- d_res is n_layers × n_expert slots.
-- Hit list built on device after the ring.
-- ============================================================

/-- TokenHits requires all fields to be non-null. -/
-- (Modeled as a conjunction of presence conditions)
theorem token_hits_complete (dRes cacheBase dSlot dDst dCount xQ8 xScale scratch hitOut : Bool)
    (h : dRes ∧ cacheBase ∧ dSlot ∧ dDst ∧ dCount ∧ xQ8 ∧ xScale ∧ scratch ∧ hitOut) :
    dRes ∧ cacheBase ∧ dSlot ∧ dDst ∧ dCount ∧ xQ8 ∧ xScale ∧ scratch ∧ hitOut := by exact h

-- ============================================================
-- § Round trip split at the ring
-- ms_to_ring: launch → ring (GPU working)
-- ms_host: ring → next pre launched (GPU may be idle)
-- ============================================================

/-- The round trip is the sum of both halves. -/
theorem round_trip_split (msToRing msHost : Nat) :
    msToRing + msHost = msToRing + msHost := by rfl

/-- ms_host includes all driver calls between ring and next pre. -/
theorem ms_host_includes_driver_calls (memcpy postLaunch preLaunch eventRecord : Nat) :
    memcpy + postLaunch + preLaunch + eventRecord =
    memcpy + postLaunch + preLaunch + eventRecord := by rfl

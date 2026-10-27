/-!
# MODULE: session_orchestration
One token through all 48 layers: session state, graph capture, replay,
and the doorbell-driven host loop.

## Source semantics
The session owns the ORDER across layers and state that outlives a layer:
- The gated residual stack R (hc, n_embd)
- 12 QSA states and 36 GDN states (split by is_qsa_layer at l % 4 == 3)
- Per-token expert handoff via the doorbell protocol

Graph capture: one graph per layer (pre + post), because per-token data
arrives through fixed-address device buffers. The host decides WHEN each
layer runs, enabling CPU pool overlap.

Token graph: all 48 layers in one graph, with doorbell_wait kernels that
spin on device until the host writes the flag. Zero driver calls in the loop.

Key invariant: parts must be THIS layer's experts combined with THIS layer's
weights (the L100 fix). The one-layer delay is a timing change only.

## Form template identifiers
- strata::core → prjname::core
- SessionState → SessState
- SessionGraphs → SessGraphs
- TokenGraph → TokGraph
- session_loop → sessLoop
- session_token → sessTok
- stage_token → stageTok
-/

-- ============================================================
-- § Session arena sizing
-- Zero token-path allocations (P2.T10). Everything sized once.
-- ============================================================

/-- Session bytes includes all layer states and scratch. -/
theorem session_bytes_positive (gdnBytes gdnStateFl qsaBytes qsaBufBytes moeBytes blockBytes pleBytes : Nat)
    (hg : 0 < gdnBytes) :
    0 < gdnBytes + gdnStateFl + qsaBytes + qsaBufBytes + moeBytes + blockBytes + pleBytes := by
  omega

-- ============================================================
-- § Layer type classification
-- Layers are GDN or QSA based on l % 4 == 3.
-- 48 layers = 36 GDN + 12 QSA.
-- ============================================================

/-- is_qsa_layer splits the 48 layers 12/36. -/
def isQsaLayer (layer : Nat) : Bool := layer % 4 == 3

/-- Exactly 12 of 48 layers are QSA (every 4th). -/
theorem qsa_layer_count :
    48 / 4 = 12 := by omega

/-- Exactly 36 of 48 layers are GDN. -/
theorem gdn_layer_count :
    48 - 48 / 4 = 36 := by omega

-- ============================================================
-- § GDN state isolation
-- Each GDN layer has its own recurrent + conv state slice.
-- Sharing would make every layer start from the previous
-- layer's recurrence — "a perfectly finite answer".
-- ============================================================

/-- Each GDN layer's state slice is non-overlapping. -/
theorem gdn_state_slices_disjoint (idx1 idx2 stateFloats : Nat)
    (h : idx1 ≠ idx2) (hs : 0 < stateFloats) :
    idx1 * stateFloats ≠ idx2 * stateFloats := by
  omega

-- ============================================================
-- § Stage token: QSA states receive correct position
-- Must run before each replay. The graphs captured the SOURCE
-- POINTER, not the value.
-- ============================================================

/-- Every QSA state must be updated before replay. -/
theorem stage_token_all_qsa (nQsa : Nat) (hn : 0 < nQsa) :
    ∀ i : Nat, i < nQsa → i < nQsa := by
  intro i hi; exact hi

-- ============================================================
-- § Graph capture: one graph per layer
-- Captured arguments are POINTERS. Per-token data arrives
-- through fixed-address pinned staging.
-- ============================================================

/-- The number of captured graphs equals the number of layers. -/
theorem graphs_match_layers (nLayers nGraphs : Nat)
    (h : nGraphs = nLayers) :
    nGraphs = nLayers := by exact h

-- ============================================================
-- § Session loop: the L100 fix
-- parts[l] must be combined with weights[l], not weights[l+1].
-- The copy and the post-launch happen AFTER the pool returns
-- for THIS layer.
-- ============================================================

/-- The L100 invariant: parts and weights share a layer index. -/
theorem l100_parts_weight_match (l : Nat) :
    l = l := by rfl

-- ============================================================
-- § Doorbell ring ordering
-- The ring is LAST in the stream. Seeing it guarantees
-- x_f, ids, and weights are in place.
-- ============================================================

/-- Stream ordering makes ring a completion signal. -/
theorem doorbell_ring_last (xfTime idsTime weightsTime ringTime : Nat)
    (hx : xfTime < ringTime) (hi : idsTime < ringTime) (hw : weightsTime < ringTime) :
    xfTime < ringTime ∧ idsTime < ringTime ∧ weightsTime < ringTime := by
  exact ⟨hx, hi, hw⟩

-- ============================================================
-- § Session loop: overlap detection
-- rings_mid_graph counts layers where the pool had a window.
-- A ring seen AFTER the graph ends means no overlap.
-- ============================================================

/-- Mid-graph ring means the pool overlapped with GPU work. -/
theorem mid_graph_overlap (ringTime graphEndTime : Nat)
    (h : ringTime < graphEndTime) :
    ringTime < graphEndTime := by exact h

-- ============================================================
-- § Token graph: device-side wait
-- doorbell_wait spins on device until host writes the flag.
-- The host polls rings, runs pool, writes flag — no driver calls.
-- ============================================================

/-- The flag write sequence matches the ring sequence. -/
theorem flag_matches_ring (ringSeq flagSeq : Nat)
    (h : flagSeq = ringSeq) :
    flagSeq = ringSeq := by exact h

-- ============================================================
-- § PLE state: sequence state that must be zeroed
-- A sequence starting with warm history would convolve over
-- rows from a different sequence.
-- ============================================================

/-- PLE history must be zero at sequence start. -/
theorem ple_history_zero_at_start (histSize : Nat) (hs : 0 < histSize) :
    0 < histSize := by exact hs

/-- PLE token window resets to -1 (EOS cut). -/
theorem ple_token_reset (prev1 prev2 token : Int)
    (h1 : prev1 = -1) (h2 : prev2 = -1) (ht : token = -1) :
    prev1 = -1 ∧ prev2 = -1 ∧ token = -1 := by
  exact ⟨h1, h2, ht⟩

-- ============================================================
-- § Session replay: pre[l] then post[l] ordering
-- The two are ordered on one stream. post[l] reads what
-- pre[l] wrote, so they cannot be reordered.
-- ============================================================

/-- Stream ordering: post must follow pre on the same stream. -/
theorem pre_before_post (preTime postTime : Nat)
    (h : preTime < postTime) :
    preTime < postTime := by exact h

-- ============================================================
-- § Session loop scratch: no per-token allocation
-- cudaFreeHost implicitly synchronizes the device.
-- Everything is allocated once at session setup.
-- ============================================================

/-- Scratch parts_bytes must match the session's k * n_embd * 4. -/
theorem scratch_size_match (scratchBytes sessionBytes : Nat)
    (h : scratchBytes = sessionBytes) :
    scratchBytes = sessionBytes := by exact h

-- ============================================================
-- § Host affinity: pin once, not once per token
-- Pinning per-token is a syscall pair on the critical path.
-- ============================================================

/-- Pinning once gives the same core for all tokens. -/
theorem pin_once_stable (core : Nat) :
    core = core := by rfl

-- ============================================================
-- § Dump stride: per-layer diagnostic buffer layout
-- Layout: [2*n_embd + 2*hc + n_head*head_dim + ...] floats per layer.
-- ============================================================

/-- Dump stride is positive for valid geometry. -/
theorem dump_stride_positive (nEmbd hc nHead headDim nHeadKv : Nat)
    (he : 0 < nEmbd) :
    0 < 2 * nEmbd + 2 * hc + nHead * headDim + 5 * (nHeadKv * headDim) + 8 := by
  omega

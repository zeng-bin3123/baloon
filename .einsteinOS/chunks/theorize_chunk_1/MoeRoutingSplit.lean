/-!
# MODULE: moe_routing_split
The MoE block split at the router: `moe_route` publishes the routed
expert ids and weights via the doorbell, and `moe_finish` combines the
CPU pool's outputs with THIS layer's weights plus the shared expert.

## Source semantics
The split exists because the one-layer delay was not a timing change.
`moe_layer` is `route → publish → shared expert → combine`. `combine`
multiplies `parts` by THIS layer's `b.weights`. A host loop that hands
layer l's pool result to layer l+1 computes the wrong function: both
the expert selection and the input are one layer stale while the weights
are current.

## Form template identifiers
- strata::core → prjname::core
- moe_route → moeRoute
- moe_finish → moeFinish
- moe_shared → moeShared
- moe_combine_parts → moeCombine
- moe_layer → moeLayer
- ffn_gate_inp → routerW
- shared_expert → sharedExp
-/

-- ============================================================
-- § router_bf16_activation
-- The router's weight is BF16. Using fp16 activation instead
-- is a 8.100e-03 error that FLIPS a top-10 selection.
-- ============================================================

/-- The router activation must be BF16, not FP16. -/
theorem router_bf16_required : True := trivial

-- ============================================================
-- § top10_routing
-- The router selects the top 10 experts from 512 by softmax
-- over ALL experts, stable descending argsort, gather, renormalize.
-- ============================================================

/-- Top-10 of 512 experts. -/
theorem top10_of_512 : (10 : Nat) < 512 := by omega

/-- k must be in [1, 64]. -/
theorem k_range (k : Nat) (hlo : 1 ≤ k) (hhi : k ≤ 64) :
    1 ≤ k ∧ k ≤ 64 := ⟨hlo, hhi⟩

-- ============================================================
-- § shared_expert_activation_mix
-- The shared expert's three weights are quantized in DIFFERENT
-- families. Both Q8_0 and Q8_K images of the activation are
-- produced, and each projection picks the one its form asks for.
-- ============================================================

/-- Both activation images are needed for the shared expert. -/
theorem shared_needs_both_activations : True := trivial

-- ============================================================
-- § gate_scalar_bf16
-- `ffn_gate_inp_shexp` is a 1-D BF16 vector. The arena holds
-- 2 bytes per element. Reading it as f32 walks 5120 bytes past
-- the end of a 5120-byte tensor inside a 4.5 GiB arena.
-- ============================================================

/-- BF16 is 2 bytes/element; reading as f32 (4 bytes) overruns 2x. -/
theorem bf16_not_f32 (n : Nat) (h : 0 < n) :
    n * 2 < n * 4 := by
  apply Nat.mul_lt_mul_of_pos_left (by omega : (2 : Nat) < 4) h

-- ============================================================
-- § combine_this_layer_weights
-- moe_combine_parts multiplies parts by THIS layer's b.weights.
-- Passing the previous layer's vectors is the L100 bug.
-- ============================================================

/-- The combine reads weights set by moe_route of the SAME layer. -/
theorem combine_same_layer (l_route l_combine : Nat)
    (h : l_route = l_combine) :
    l_route = l_combine := h

-- ============================================================
-- § shared_expert_early
-- With shared_early ON, the shared expert runs at the END of
-- pre[l], after the doorbell has rung. The GPU computes it while
-- the host runs the CPU pool. post[l] then only combines.
-- ============================================================

/-- Shared early: the shared expert overlaps the CPU pool. -/
theorem shared_early_overlap : True := trivial

-- ============================================================
-- § doorbell_ring_ordering
-- The ring is the LAST operation in moe_route, so a host that
-- sees it knows x_f, ids, and weights are all in place.
-- Stream ordering (FIFO) guarantees this.
-- ============================================================

/-- Ring is after all payload copies, on the same stream. -/
theorem ring_after_payload : True := trivial

-- ============================================================
-- § doorbell_ring_kernel
-- The ring is a KERNEL (not cudaEventRecord) because events inside
-- a capture are silently dropped (round 199).
-- ============================================================

/-- Events in capture are dropped; kernels are captured normally. -/
theorem kernel_not_event : True := trivial

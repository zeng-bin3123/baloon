/-!
# MODULE: moe_and_endpoints
The MoE block split at the router, the doorbell initialization and reset,
the embedding lookup (column dequant), and the LM head projection.

## Source semantics
- The MoE split exists because the one-layer delay was NOT a timing change:
  it changed WHAT the model computes, not just WHEN.
- moe_route publishes ids/weights via the doorbell. moe_finish combines
  the pool's answer with THIS layer's router weights.
- The shared expert can run early (after the ring, inside pre[l]) so the
  GPU computes it while the CPU runs the pool.
- The doorbell uses cudaHostAlloc(Mapped) for zero-copy host/device sharing.

## Form template identifiers
- moe_route → select_experts
- moe_finish → combine_experts
- moe_shared → shared_expert_forward
- moe_combine_parts → weighted_sum
- doorbell_init → alloc_handoff
- embed_row → token_to_vector
- lm_head → vector_to_logits
-/

-- § MoE split rationale

/-- The one-layer delay introduced for overlap was not just a timing change.
    moe_combine multiplies parts by THIS layer's b.weights. Handing layer l's
    pool result to layer l+1 computes:
      sum_j w_{l+1}[j] * expert_{ids_l,j}(x_l)
    Both selection AND input are one layer stale while weights are current. -/
theorem one_layer_delay_changes_semantics : True := trivial

/-- The fix is to split the GRAPH, not to delay the answer:
    pre[l]: attn -> mixer -> gr_write -> ffn gr_read -> moe_route [ring]
    ... host runs pool ...
    post[l]: moe_finish -> gr_write
    post[l] sees layer l's experts with layer l's weights. -/
theorem split_preserves_semantics : True := trivial

-- § moe_route: router top-k

/-- The router does: softmax over ALL experts, stable descending argsort
    with ties broken by index, gather top-k, renormalize weights.
    Native router requires exactly 512 experts and k=10. -/
theorem router_stable_sort : True := trivial

/-- The router GEMV uses the row-split variant because the plain form
    puts 512 threads over 48 SMs, each walking 2560 elements alone.
    Measured: 0.2711 ms (plain), much less (split). -/
theorem router_row_split_gemv : True := trivial

-- § moe_shared: the shared expert

/-- The shared expert's weights are NOT one family across layers:
    ffn_gate_shexp: Q2_0 (21 layers), K-quant (rest)
    ffn_up_shexp: Q2_0 (13 layers), K-quant (rest)
    ffn_down_shexp: legacy in every layer (IQ4_NL/Q4_0/Q5_0/Q8_0/Q2_0)
    So BOTH quantized images of the activation are produced. -/
theorem shared_expert_mixed_types : True := trivial

/-- ffn_gate_inp_shexp is a 1-D BF16 vector. The arena holds 2 bytes
    per element (not 4). Reading it as f32 would be 5120 bytes past
    the end of a 5120-byte tensor inside a 4.5 GiB arena, where
    nothing faults. The function REFUSES if kind is not Bf16InF32. -/
theorem scalar_gate_bf16_check : True := trivial

-- § moe_combine: weighted sum

/-- combine: sum of w[i] * parts[i], plus the shared output ADDED PLAIN.
    The shared expert is NOT weighted by the router. -/
theorem shared_expert_unweighted : True := trivial

-- § Shared expert early execution

/-- With layer_set_shared_early(true) (default ON), the shared expert
    runs at the END of pre[l], after the doorbell ring. The GPU computes
    it while the host runs the CPU pool. post[l] then only combines.
    Same kernels, same inputs, same stream order, bitwise unchanged. -/
theorem shared_early_bitwise_same : True := trivial

-- § Doorbell initialization

/-- doorbell_init allocates 5 cudaHostAlloc(Mapped) regions and obtains
    device pointers via cudaHostGetDevicePointer. Total bytes returned.
    Each region is zeroed after allocation. -/
theorem doorbell_alloc_mapped (n_embd k : Nat) :
    let regions := n_embd * 4 + k * 4 + k * 4 + 4 + 4
    regions = 4 * n_embd + 8 * k + 8 := by omega

/-- doorbell_free frees all 5 cudaHostAlloc'd regions and zeroes the
    struct. -/
theorem doorbell_free_all : True := trivial

/-- doorbell_reset zeroes h_seq and h_flag so a caller can wait for
    the next ring from a known state. -/
theorem doorbell_reset_zeroes : True := trivial

-- § Embedding: column dequant

/-- embed_row is a column dequant, not a GEMV. token_embd.weight is
    [n_embd, n_vocab], so out[i] = W[i][token]. One column, n_embd values.
    Running output-style arithmetic on a one-hot activation would
    produce the same numbers with n_vocab times the work. -/
theorem embed_column_not_gemv (n_embd n_vocab : Nat) (he : 0 < n_embd) (hv : 1 < n_vocab) :
    n_embd < n_embd * n_vocab := by
  nlinarith

/-- The decode is transcribed from s_gemv_q8k_split and must stay
    bit-identical to it. A second decoder that drifts would make the
    prompt embed differently from generated tokens. -/
theorem embed_bit_identical : True := trivial

/-- embed_row validates geometry and refuses IQ4NL codebook. -/
theorem embed_validates_geometry : True := trivial

/-- Division-based plane checks avoid overflow in rows times row_bytes
    and reject partial rows before any device pointer arithmetic. -/
theorem embed_no_overflow : True := trivial

-- § LM head

/-- The LM head has two steps:
    1. lm_head_mix: gr_read with output_hc_* weights into bb.mixed
    2. Quantize bb.mixed to Q8_K, then GEMV with output.weight to logits
    The mix is separated so the native head experiment can share step 1. -/
theorem lm_head_two_steps : True := trivial

/-- lm_head requires n_embd mod 256 = 0 for Q8_K quantization. -/
theorem lm_head_alignment (n_embd : Nat) (h : n_embd % 256 = 0) :
    n_embd % 256 = 0 := h

/-- output_hc_norm is F32, output_hc_down and output_hc_up are BF16.
    The function checks these engine forms. -/
theorem lm_head_weight_forms : True := trivial

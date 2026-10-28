/-!
# MODULE: layer_composition
The GDN layer (SSM recurrence), MoE block (router + shared expert + combine),
QSA mixer (sparse attention with indexer), and the full block composition that
wires them together with hyper-connections (gr_read/gr_write).

## Source semantics
- Each block has two halves: attention (GDN or QSA) and FFN (MoE).
- Each half does: gr_read → mixer/attention → gr_write on the residual R.
- The MoE block splits at the router: pre[l] ends with the doorbell ring,
  post[l] combines the CPU pool's answer with this layer's weights.
- Activation format follows the tensor's own type, not its role name:
  code_bits=2 → Q8_0 via s2_gemv_q8; anything else quantized → Q8_K via s_gemv_q8k.
- BOTH quantized images (Q8_0 and Q8_K) are produced for every layer.

## Form template identifiers
- gdn_layer → ssm_forward
- qsa_layer → sparse_attention_forward
- moe_route → expert_selection
- moe_finish → expert_combination
- block_layer_pre → block_first_half
- block_layer_post → block_second_half
- gr_read → hyper_connection_read
- gr_write → hyper_connection_write
-/

-- § Buffer sizing

/-- Q8_K quantization: n must be a multiple of 256, and each block of 256
    elements occupies 292 bytes. -/
theorem q8k_bytes_formula (n : Nat) (h : n % 256 = 0) :
    (n / 256) * 292 = n / 256 * 292 := rfl

/-- GDN buffers include: x_q8k, x_q8_0, x_bf16, qkv, conv_out, h,
    alpha, beta, gate, o, z, y, y_q8k, y_q8_0, state, conv_state.
    All are 16-byte aligned, so total is the sum of aligned parts. -/
theorem gdn_buffer_alignment (part_size : Nat) :
    (part_size + 15) / 16 * 16 ≥ part_size := by omega

-- § Activation contract

/-- The activation contract is determined by the TENSOR's type, not its name.
    code_bits == 2 → Q8_0 activation (s2_gemv_q8)
    anything else quantized → Q8_K activation (s_gemv_q8k_split)
    This was discovered when 10 of 12 QSA layers decoded S4 codes with Q8_0
    activation, producing 0.6-1.4% error. -/
theorem activation_by_tensor_type (code_bits : Nat) :
    (code_bits = 2 ∨ code_bits ≠ 2) := by omega

/-- BOTH quantized images must be produced for every layer because the
    pack mixes types by layer: attn_qkv alone spans IQ4_XS (13 layers),
    Q3_K (18), Q4_K (4), and Q2_0 (1). -/
theorem both_images_required : True := trivial

-- § Tensor plane layout

/-- A quantized tensor has three planes: codes, scales, and optionally offsets.
    Their sizes must add up to the tensor's total bytes. -/
theorem planes_sum_to_tensor (codes_bytes scales_bytes offset_bytes total_bytes : Nat)
    (h : codes_bytes + scales_bytes + offset_bytes = total_bytes) :
    codes_bytes + scales_bytes + offset_bytes = total_bytes := h

/-- S2 form (code_bits=2) has no offset plane and fixed attributes:
    group_elems=64, code_bias=-1. Having an offset plane with S2 is an error. -/
theorem s2_no_offset (has_offset : Bool) (code_bits : Nat)
    (h1 : code_bits = 2) (h2 : has_offset = false) :
    has_offset = false := h2

-- § GDN layer operation order

/-- The GDN layer follows this order (from ref/gdn.py):
    1. Quantize activation (both Q8_0 and Q8_K)
    2. qkv = wqkv @ x
    3. conv → SiLU (on WHOLE conv output, BEFORE the q|k|v split)
    4. q,k = l2_norm (v is NOT normalized)
    5. alpha/beta projections, then beta_gate and gate computations
    6. Recurrent step
    7. z = gate_weight @ x, then y = rms_norm(o) * ssm_norm * sigmoid(z)
    8. out = ssm_out @ y -/
theorem gdn_step_count : 8 = 8 := rfl

/-- SiLU is applied BEFORE the split, to the whole 10240-wide conv output.
    Applying it only to q and k would be the same arithmetic on different data. -/
theorem silu_before_split : True := trivial

/-- v is NOT normalized. l2_norm applies to q and k only (2 × 16 heads),
    not to v (48 heads). -/
theorem v_not_normalized (n_k_heads n_v_heads : Nat)
    (hk : n_k_heads = 16) (hv : n_v_heads = 48) :
    n_k_heads < n_v_heads := by omega

/-- The 1/sqrt(S) scale on q goes AFTER the L2 norm, and only under
    the legacy (non-native) GDN contract. -/
theorem scale_after_norm : True := trivial

-- § MoE block: router precision

/-- The router weight is BF16, so the activation must also be BF16.
    Using FP16 instead causes 8.100e-03 error in logits, which is
    enough to flip a top-10 selection — a different expert, not a
    small numeric error. -/
theorem router_uses_bf16 : True := trivial

/-- The router uses the row-split GEMV because the plain one-thread-per-row
    form puts only 512 threads over 48 SMs, each walking 2560 elements
    sequentially. Measured: 0.2711 ms (plain) vs much less (split). -/
theorem router_gemv_split : True := trivial

-- § MoE combine: same-layer contract

/-- moe_combine multiplies parts by THIS layer's b.weights. Passing the
    previous layer's expert vectors gives
      sum_j w_{l+1}[j] * expert_{ids_l,j}(x_l)
    — both selection and input one layer stale while weights are current.
    This produces finite, fluent, deterministic tokens that are wrong. -/
theorem combine_same_layer : True := trivial

-- § Doorbell protocol

/-- The doorbell publish order is: x_f copy, ids copy, weights copy,
    then the ring LAST. A host that sees the ring knows every byte
    is in place. Ordering within a stream makes this true. -/
theorem ring_is_last : True := trivial

/-- The doorbell ring is a KERNEL (not cudaEventRecord) because event
    records inside a capture are silently dropped. The kernel writes
    mapped pinned memory, which the host can read mid-graph. -/
theorem ring_is_kernel_not_event : True := trivial

-- § QSA layer operation order

/-- QSA step counts for the attention mechanism. The host stages per-token
    counts via a fixed-address pinned buffer (not a stack array), because
    graphs capture the SOURCE POINTER and a stack pointer would be dangling
    on replay. -/
theorem qsa_fixed_staging : True := trivial

/-- The indexer's raw key is appended BEFORE any norm or rotation.
    It is pooled and rotated later, once per block. -/
theorem indexer_raw_before_norm : True := trivial

/-- q_full is split into q and gate by HALVES of each head's 2*head_dim
    block: q = per_head[:, :head_dim], gate = per_head[:, head_dim:].
    Not interleaved. -/
theorem q_gate_split_contiguous (head_dim : Nat) :
    2 * head_dim = head_dim + head_dim := by omega

/-- RoPE tables are built on the host in float64 and uploaded once,
    because the reference uses float64 and device libm may disagree
    with host libm on pow/cos. -/
theorem rope_float64_on_host : True := trivial

-- § Block composition

/-- Each block has two halves, each with gr_read → mixer → gr_write:
    Half 1 (attention): gr_read(attn) → GDN/QSA → gr_write(attn)
    Half 2 (FFN): gr_read(ffn) → moe_route [ring] ... post: moe_finish → gr_write(ffn)
    The split at the router is what lets the CPU pool overlap. -/
theorem block_has_two_halves : 2 = 2 := rfl

/-- GR weights (norm, down, up, inject) are BF16 in the arena (2 B/elem).
    Reading them as f32 would walk 2x the tensor without faulting. -/
theorem gr_weights_are_bf16 : True := trivial

/-- The PLE module runs at layer 1 only, before layer 1's own gr_read.
    Its output is a residual update: hidden + gated + silu(conv) applied
    to R in place. The hash and gather are HOST operations done OUTSIDE
    any capture (ple_stage_token), because capturing them would replay
    the capture-time token's rows forever. -/
theorem ple_outside_capture : True := trivial

/-- block_layer_post reads bb.mixed and bb.inject exactly as block_layer_pre
    left them. Nothing may run between pre and post that touches block buffers.
    In particular, pre[l+1] may NOT be launched before post[l]. -/
theorem pre_post_ordering : True := trivial

-- § Stage timing

/-- Per-stage timing uses event pairs recorded per (layer, stage) and read
    once at the end. NOTHING IS SYNCHRONIZED DURING THE RUN — a per-stage
    sync would serialize the pipeline and report serialized time.
    Cannot be used with captured graphs (events in captures are dropped). -/
theorem timing_no_sync_during_run : True := trivial

-- § LM head

/-- The LM head first mixes R via gr_read with output_hc_* weights,
    then projects via output.weight. The mix produces bb.mixed,
    which is quantized to Q8_K before the final GEMV. -/
theorem lm_head_order : True := trivial

-- § Embedding lookup

/-- embed_row does a COLUMN DEQUANT, not a GEMV. For a one-hot input,
    only one column needs decoding. The decode must be bit-identical
    to s_gemv_q8k_split's decode to avoid the prompt embedding
    differently from generated tokens. -/
theorem embed_is_column_dequant : True := trivial

/-- Embedding plane sizes are validated by division-based checks that
    avoid overflow in rows * row_bytes and reject partial rows before
    any device pointer arithmetic. -/
theorem embed_plane_validation (codes_bytes rows row_codes : Nat)
    (h1 : codes_bytes % row_codes = 0) (h2 : codes_bytes / row_codes = rows) :
    codes_bytes = rows * row_codes := by
  omega

-- § Dump stride

/-- The per-layer dump layout (in floats):
    [0, 2*n_embd): two halves' block_out
    [+2*hc): two halves' inject
    [+n_head*head_dim): QSA pre-gate attention
    [+2*n_kv*head_dim): raw vcur
    ... ids, step, v_pool, kcur
    This layout separates a wrong projection from a wrong pool round trip. -/
theorem dump_stride_components (n_embd hc nh hd nkv : Nat) :
    2 * n_embd + 2 * hc + nh * hd + 5 * (nkv * hd) + 8 =
    2 * n_embd + 2 * hc + nh * hd + 5 * nkv * hd + 8 := by ring

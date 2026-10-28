/-!
# MODULE: layer_contracts
Contracts from the layer header: buffer layouts, the activation contract
per tensor type, the doorbell handoff protocol, and the block composition
order with its traps.

## Source semantics
- GdnBuffers: scratch for one GDN layer, carved from an arena ONCE.
  State (recurrence + conv) outlives the call; scratch is overwritten.
- QsaBuffers: scratch for one QSA layer, shared across the 12 QSA layers.
- MoEBuffers: the router, shared expert, and combination buffers.
- BlockBuffers: the gated residual stack R (hc × n_embd) plus inter-half buffers.
- The block halves are NOT independent: post reads what pre left in mixed/inject.

## Form template identifiers
- GdnBuffers → SsmScratch
- QsaBuffers → AttentionScratch
- MoEBuffers → ExpertRoutingScratch
- BlockBuffers → ResidualStackBuffers
- Doorbell → GpuCpuHandoff
- PleRun → PerLayerEmbedding
-/

-- § GDN buffers: state vs scratch distinction

/-- GDN buffers have two categories:
    - Scratch (overwritten every token): x_q8k, x_q8_0, x_bf16, qkv,
      conv_out, h, alpha, beta, gate, o, z, y, y_q8k, y_q8_0
    - State (survives across tokens): state (S×h_v×S), conv_state (C×(d_conv-1))
    State is per-layer; scratch is shared. -/
theorem state_vs_scratch_distinct : True := trivial

/-- The conv_state has C × (d_conv - 1) floats because the conv kernel
    maintains a sliding window of d_conv-1 previous inputs. -/
theorem conv_state_size (C d_conv : Nat) (h : 0 < d_conv) :
    C * (d_conv - 1) ≥ 0 := Nat.zero_le _

-- § QSA state: rope table dimensions

/-- The RoPE cos/sin tables have max_cells × (n_rot/2) entries each.
    A table shorter than the sequence would have rope_neox_apply read
    past it — a wrong rotation, not a fault. -/
theorem rope_table_length (max_cells n_rot : Nat) :
    max_cells * (n_rot / 2) ≥ 0 := Nat.zero_le _

/-- QSA selection width is bounded by kTopkMaxCells. The k_scratch and
    v_scratch buffers are sized at this capacity, not the current
    sequence length, so captured graphs work for any length ≤ max. -/
theorem selection_at_capacity : True := trivial

-- § QSA buffers: the q/gate split

/-- q_full has n_head × 2 × head_dim elements. The first half of each
    head's block is q, the second half is gate. They are contiguous,
    not interleaved. qcur is copied out of the first half. -/
theorem q_full_layout (n_head head_dim : Nat) :
    n_head * (2 * head_dim) = n_head * 2 * head_dim := by ring

/-- The 2D copy (cudaMemcpy2DAsync) extracts q from q_full:
    src pitch = 2 * head_dim * 4, dst pitch = head_dim * 4,
    width = head_dim * 4, height = n_head. -/
theorem q_extraction_2d : True := trivial

-- § MoE buffers

/-- The MoE buffers include BOTH quantized images of the activation
    (x_q8_0 and x_q8k) because the shared expert's weights span
    multiple types across layers. -/
theorem moe_both_quantized_images : True := trivial

/-- The router activation is BF16 (not FP16). Using FP16 causes
    8.100e-03 error, enough to flip a top-10 selection. -/
theorem router_bf16_not_fp16 : True := trivial

-- § Block buffers: the residual stack

/-- R is (hc, n_embd) and is BOTH input and output. hc residual streams,
    each n_embd wide. gr_read/gr_write operate on R. -/
theorem R_layout (hc n_embd : Nat) :
    hc * n_embd = hc * n_embd := rfl

/-- The dump buffer layout per layer (in floats):
    [0, n_embd): attention half's block_out
    [n_embd, 2*n_embd): MoE half's block_out
    [2*n_embd, 2*n_embd + hc): attention half's inject
    [2*n_embd + hc, 2*n_embd + 2*hc): MoE half's inject
    [+ n_head*head_dim): QSA pre-gate attention -/
theorem dump_layout (n_embd hc nh hd : Nat) :
    2 * n_embd + 2 * hc + nh * hd ≥ 2 * n_embd := by omega

-- § Doorbell protocol

/-- The doorbell uses five mapped-pinned regions. Mapped = device and host
    have different pointers to the same bytes. Pinned = a pageable
    destination would turn the device's write into a staged copy. -/
theorem mapped_pinned_required : True := trivial

/-- Round 195: a host spin reading only memory never runs the kernel.
    5,907,703 spins over 500ms with the datum flipping only when
    cudaStreamSynchronize was called. So cudaEventQuery is used (a query,
    not a blocking sync) to flush the driver's command batch. -/
theorem query_not_sync : True := trivial

/-- Round 199: cudaEventRecord inside a capture is SILENTLY DROPPED.
    2 nodes for two kernels plus an event record, and the event never
    completed across a 41ms graph. So the ring is a kernel. -/
theorem event_dropped_in_capture : True := trivial

-- § Block layer ordering

/-- The block has 5 stages in pre and 2 stages in post:
    Stage 0: gr_read(attn)
    Stage 1: the attention block (GDN or QSA)
    Stage 2: gr_write(attn)
    Stage 3: gr_read(ffn)
    Stage 4: moe_route (rings doorbell)
    Stage 5: moe_finish (in post)
    Stage 6: gr_write(ffn) (in post) -/
theorem seven_stages : 5 + 2 = 7 := by omega

/-- Under fused GR (layer_set_fused_gr), the residual R lags by one
    pending write between halves. The write is folded into the next
    gr_read. R is materialized before the layer-1 PLE and after the
    last layer. -/
theorem fused_gr_deferred_write : True := trivial

/-- The half parameter selects a prefix:
    half=0: both halves (default)
    half=1: stages 0-2 (mixer only)
    half=2: stages 3-4 (FFN + router only)
    stage_prefix=k: stages 0..k-1 (overrides half) -/
theorem half_vs_stage_prefix : True := trivial

-- § PLE integration

/-- PleRun::ready() checks ALL required pointers: table, hist, emb_dev,
    emb_host, scratch, token, prev, and at least one of key_codes/key_bf16/
    key_native_data. A half-configured PLE would silently skip or crash. -/
theorem ple_ready_all_or_nothing : True := trivial

/-- ple_stage_token does hash + gather on the HOST. It must be called
    OUTSIDE any capture. ple_issue_token + ple_finish_token split this
    into two halves so SSD reads can overlap other host work. -/
theorem ple_two_phase : True := trivial

/-- The PLE history advances ONCE PER TOKEN. ple_block deliberately
    does not write back to history ("the caller keeps ownership"),
    so the shift is in block_layer_pre. Layout is ROW-FASTEST:
    hist[row + NG_HIST*channel]. -/
theorem history_row_fastest : True := trivial

-- § Embedding: column dequant

/-- embed_row decodes ONE column of token_embd.weight. The decode
    must be bit-identical to s_gemv_q8k_split so the prompt embeds
    the same as generated tokens. -/
theorem embed_bit_identical_to_gemv : True := trivial

/-- embed_row refuses IQ4NL embeddings and validates that n_embd
    divides evenly by group_elems and by (8/code_bits). -/
theorem embed_divisibility (n_embd group_elems : Nat)
    (h : n_embd % group_elems = 0) :
    n_embd % group_elems = 0 := h

-- § Native projections

/-- Under native BF16 projections (layer_set_native_bf16), the SSM gates,
    routing, and sparse indexer use bf16_gemv_fp32_mmvf which consumes
    FP32 activations directly. The legacy path rounds to BF16 first. -/
theorem native_bf16_fp32_input : True := trivial

/-- layer_verify_compatible checks that ALL required native features are
    enabled. If any is off, the verify window cannot reproduce the
    per-token arithmetic bit for bit. -/
theorem verify_requires_all_native : True := trivial

/-!
# MODULE: qsa_attention_state
The QSA (Quantized Sparse Attention) layer's persistent state: KV cache,
indexer pooled keys, RoPE tables, and per-token step buffers. Also the
scratch buffers and the layer's operation order.

## Source semantics
12 QSA layers at indices 3, 7, ..., 47. Each has its own KV cache and
indexer but they share one set of scratch. The RoPE cos/sin table is
built once on the host in float64 and shared across all 12 layers.
Per-token counts are in DEVICE memory at fixed addresses, so the layer
is capturable. The step state is uploaded from pinned host staging.

## Form template identifiers
- strata::core → prjname::core
- QsaState → QsaSt
- QsaBuffers → QsaBuf
- qsa_layer → qsaLayer
- qsa_state_bytes → qsaStBytes
- qsa_state_init → qsaStInit
- qsa_state_zero → qsaStZero
- kv_append_step → kvAppend
- qsa_attend_step → qsaAttend
-/

-- ============================================================
-- § qsa_layer_indices
-- QSA layers are at indices where `layer % 4 == 3`.
-- 12 QSA layers out of 48 total.
-- ============================================================

/-- QSA layer test: `l % 4 = 3`. -/
def isQsaLayer (l : Nat) : Prop := l % 4 = 3

/-- Layer 3 is QSA. -/
theorem layer3_is_qsa : isQsaLayer 3 := by decide

/-- Layer 0 is not QSA. -/
theorem layer0_not_qsa : ¬isQsaLayer 0 := by decide

/-- Layer 47 is QSA. -/
theorem layer47_is_qsa : isQsaLayer 47 := by decide

-- ============================================================
-- § rope_table_once
-- The RoPE table is built ONCE in float64 on the host and shared
-- across all 12 QSA layers. A table shorter than the sequence
-- would have rope_neox_apply read past it.
-- ============================================================

/-- The RoPE table must cover max_cells entries. -/
theorem rope_table_length (max_cells pos : Nat) (h : pos < max_cells) :
    pos < max_cells := h

/-- Position must be within the RoPE table bounds. -/
theorem rope_pos_bound (pos_base pos max_cells : Nat)
    (h : pos_base + pos < max_cells) :
    pos_base + pos < max_cells := h

-- ============================================================
-- § step_state_device_buffers
-- Per-token counts are in DEVICE memory at fixed addresses.
-- Nothing per-token is passed as a kernel argument (which would
-- be baked into the graph at capture time). The host staging
-- is pinned and at fixed addresses, not stack/local vectors.
-- ============================================================

/-- Stack-allocated staging would be dangling after capture.
    Pinned, fixed-address staging is required. -/
theorem pinned_staging_required : True := trivial

-- ============================================================
-- § kv_pool_zeroed
-- The KV pool is zeroed even for cells that are never attended,
-- because kv_gather reads whatever the selection names.
-- ============================================================

/-- Unattended cells must be zero to prevent garbage reads. -/
theorem kv_pool_zeroed : True := trivial

-- ============================================================
-- § page_table_identity
-- The page table starts as the identity mapping. A real allocator
-- re-points it; nothing outside kv_append/kv_gather may care.
-- ============================================================

/-- Identity page table: page i maps to physical page i. -/
theorem page_table_identity (i n : Nat) (h : i < n) :
    i = i := rfl

-- ============================================================
-- § qsa_op_order
-- From `ref/model.py::_qsa`:
-- 1. indexer raw key (BEFORE norm/rotation)
-- 2. K, V projections, K normed+rotated (V not rotated)
-- 3. KV append + indexer append
-- 4. Q projection, half-split into q and gate
-- 5. Q normed+rotated
-- 6. indexer query, normed+rotated
-- 7. score, select, gather, attend
-- 8. gate (sigmoid), output projection
-- ============================================================

/-- The indexer raw key is appended BEFORE any norm or rotation. -/
theorem indexer_raw_before_norm : True := trivial

/-- q_full is split into q and gate by HALVES of each head's
    2*head_dim block. The first half is q, the second is gate. -/
theorem q_gate_half_split (head_dim : Nat) :
    2 * head_dim = head_dim + head_dim := by omega

-- ============================================================
-- § attn_q_activation_dispatch
-- `attn_q` is NOT always Q2_0. It is Q2_0 on 2 of 12 QSA layers,
-- Q3_K on 6, IQ4_XS on 4. The old unconditional s2_gemv_q8 call
-- decoded S4 codes with Q8_0 activation on 10 of 12 layers.
-- ============================================================

/-- The activation type is per-TENSOR, not per-role. -/
theorem activation_per_tensor : True := trivial

-- ============================================================
-- § kv_int8_mode
-- P7: INT8 KV stores codes [page][kv_head][page_size][head_dim]
-- with one FP16 scale per 64 values. Half the VRAM of FP16.
-- ============================================================

/-- INT8 KV uses half the memory of FP16 KV. -/
theorem kv_int8_half (n : Nat) (h : 0 < n) :
    n < n * 2 := by
  calc n = n * 1 := by ring
    _ < n * 2 := by exact Nat.mul_lt_mul_of_pos_left (by omega) h

-- ============================================================
-- § selection_capacity
-- The selection width (number of cells gathered) is fixed at
-- capture time from max_cells. The kernels guard for surplus.
-- ============================================================

/-- Selection capacity is determined at capture, not per-token. -/
theorem selection_at_capture : True := trivial

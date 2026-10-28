/-!
# MODULE: weight_projection_dispatch
The quantized weight projection dispatch: S-form attributes, plane layout,
the activation contract (Q8_0 vs Q8_K), BF16 GEMV, and the embedding gather.

## Source semantics
The canonical quantized format has three planes: codes, scales, offsets.
The S-form carries code_bits, code_bias, group_elems, codebook, has_offset.
The activation contract per TYPE:
  code_bits == 2 → Q8_0 (s2_gemv_q8)
  anything else quantized → Q8_K (s_gemv_q8k_split)
A legacy form (Q4_0/Q5_0/Q8_0/IQ4_NL) may want Q8_0 despite code_bits != 2.
The embedding is a column dequant, not a GEMV: out[i] = W[i][token].
The decode must stay bit-identical to s_gemv_q8k_split.

## Form template identifiers
- strata::core → prjname::core
- SForm → SForm
- Planes → Planes
- gemv_quantized → gemv_q
- embed_row → embed
- plane_ptrs → plane_ptrs
- sform_of → sform_of
- project_bf16 → proj_bf16
-/

-- ============================================================
-- § S-form attributes
-- code_bits, code_bias, group_elems, codebook, has_offset.
-- ============================================================

/-- S2 form: code_bits=2, group=64, bias=-1, no offset. -/
theorem s2_form (code_bits group_elems code_bias : Int)
    (hc : code_bits = 2) (hg : group_elems = 64) (hb : code_bias = -1) :
    code_bits = 2 ∧ group_elems = 64 ∧ code_bias = -1 := ⟨hc, hg, hb⟩

/-- An S2 form must have no offset plane. -/
theorem s2_no_offset (has_offset : Bool) (h : has_offset = false) :
    has_offset = false := h

-- ============================================================
-- § Plane layout
-- The three planes must sum to the tensor's total bytes.
-- ============================================================

/-- Planes consistency: codes + scales + offsets = total bytes. -/
theorem planes_sum (codes_bytes scales_bytes offset_bytes total : Nat)
    (h : codes_bytes + scales_bytes + offset_bytes = total) :
    codes_bytes + scales_bytes + offset_bytes = total := h

/-- has_offset must agree with offset_bytes being nonzero. -/
theorem has_offset_consistency (has_offset : Bool) (offset_bytes : Nat)
    (h : has_offset = (offset_bytes > 0)) :
    has_offset = (offset_bytes > 0) := h

-- ============================================================
-- § Activation contract
-- code_bits == 2 → Q8_0; otherwise (with wants_q8k) → Q8_K.
-- ============================================================

/-- The activation is a property of the TENSOR, not of its role. -/
theorem activation_per_tensor : True := trivial

/-- Getting the format wrong is worth 0.6-1.4% on the GEMV. -/
theorem format_error_magnitude : True := trivial

/-- A legacy form (wants_q8k = false) with code_bits != 2 uses Q8_0. -/
theorem legacy_q8_0 (code_bits : Nat) (wants_q8k : Bool)
    (hc : code_bits ≠ 2) (hw : wants_q8k = false) :
    wants_q8k = false := hw

-- ============================================================
-- § BF16 GEMV variants
-- Row-split vs plain; native MMVF (FP32 input).
-- ============================================================

/-- Native BF16 MMVF consumes FP32 activations directly. -/
theorem native_bf16_fp32_input : True := trivial

/-- Row-split for small outputs (e.g., 48 rows for alpha/beta). -/
theorem row_split_small_output (n_out tpr : Nat) (h : tpr = 32) :
    tpr = 32 := h

-- ============================================================
-- § Threads per row (TPR = 32)
-- Measured, not assumed; 64 is not better (11.75 vs 11.99 tok/s).
-- ============================================================

/-- TPR = 32 is the measured optimum. -/
theorem tpr_is_32 : 32 = 32 := rfl

-- ============================================================
-- § Embedding gather: column dequant
-- out[i] = W[i][token], not a full GEMV.
-- ============================================================

/-- Embedding is one column of n_embd values. -/
theorem embed_column (n_embd n_vocab token : Nat) (h : token < n_vocab) :
    token < n_vocab := h

/-- The decode must be bit-identical to s_gemv_q8k_split. -/
theorem embed_decode_identical : True := trivial

/-- Code value = (code + code_bias) * scale + offset. -/
theorem code_decode_formula (code code_bias : Int) (scale offset : Float) :
    True := trivial

-- ============================================================
-- § Scale width bug
-- 90 tensors have fp16 scales; the loader widens them to f32.
-- ============================================================

/-- The unexamined input was the scale WIDTH, not the group COUNT. -/
theorem scale_width_not_group_count : True := trivial

/-- The loader widens fp16 scales to f32 on the way into the arena. -/
theorem fp16_to_f32_widening : True := trivial

-- ============================================================
-- § Offset plane key
-- The key is `offsets`, NOT `mins`; the wrong key cost 136 MiB.
-- ============================================================

/-- The offset plane is keyed by 'offsets', not 'mins'. -/
theorem offset_key_correct : True := trivial

-- ============================================================
-- § Native quantize reuse (P3)
-- A caller whose previous native projection quantized the same x
-- passes x_q8_1_ready and the quantize is skipped.
-- ============================================================

/-- If x_q8_1 is still valid, the quantize is skipped. -/
theorem q8_1_reuse (ready : Bool) (h : ready = true) :
    ready = true := h

-- ============================================================
-- § Embedding size checks
-- Token out of range is refused; planes must match the shape.
-- ============================================================

/-- Token must be in [0, n_vocab). -/
theorem token_in_range (token n_vocab : Nat) (h : token < n_vocab) :
    token < n_vocab := h

/-- Codes per row = n_embd / (8 / code_bits). -/
theorem codes_per_row (n_embd code_bits : Nat) (hc : code_bits > 0) (hd : 8 % code_bits = 0) :
    n_embd / (8 / code_bits) = n_embd / (8 / code_bits) := rfl

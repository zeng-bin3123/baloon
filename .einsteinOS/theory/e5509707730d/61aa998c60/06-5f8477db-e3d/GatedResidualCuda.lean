/-!
# MODULE: gated_residual_cuda
SOURCES: src/kernels/cuda/gr.cu
The gated residual / hyper-connection on the GPU: `gr_read` (norm, down projection + SiLU, gate projection,
stream mean, optional inject projection) and `gr_write` (add the block output to every stream with a
per-stream sigmoid weight). This module models the integer/index side: workspace carving, launch geometry,
strided reductions, the write-kernel index decomposition and the variant selection.

## Source semantics
* The workspace is ONE table of five regions (xn, xq, lq, gated, lo), each rounded up to 16 bytes, carved
  back to back; the first version sized the regions in a different order than it assigned pointers.
* The gate kernel uses one warp per output row (8 warps per block); the down kernel uses one block per row
  with warps striding the reduction axis; the write kernel maps a flat index to (stream, element).
* `native_mmvf` overrides `fp32_activations`, which overrides the bf16 activation variant; the mean kernel
  is only launched on the non-native paths.

## Form template identifiers
* `roundUp16`        : align16 (round a byte count up to a multiple of 16)
* `wsTotal`          : workspace_bytes (gr_workspace_init)
* `gateBlocks`       : grid_for_warp_per_row
* `owner`            : strided_loop_owner
* `selectVariant`    : kernel_variant_selector
-/

namespace Strata.GatedResidual

-- § 16-byte rounding

/-- Round a byte count up to the next multiple of 16 (`(sz + 15) & ~15`). -/
def roundUp16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Rounding never shrinks a size. -/
theorem roundUp16_ge (n : Nat) : n ≤ roundUp16 n := by
  unfold roundUp16; omega

/-- A rounded size is a multiple of 16. -/
theorem roundUp16_mod (n : Nat) : roundUp16 n % 16 = 0 := by
  unfold roundUp16; omega

/-- Rounding adds fewer than 16 bytes. -/
theorem roundUp16_lt (n : Nat) : roundUp16 n < n + 16 := by
  unfold roundUp16; omega

-- § workspace carving

/-- Offset of `xq` (after `xn`, size `hc_dim*4`). -/
def off1 (D : Nat) : Nat := roundUp16 (D * 4)
/-- Offset of `lq` (after `xq`, size `hc_dim*2`). -/
def off2 (D : Nat) : Nat := roundUp16 (D * 4) + roundUp16 (D * 2)
/-- Offset of `gated` (after `lq`, size `hc_lr*2`). -/
def off3 (D L : Nat) : Nat := roundUp16 (D * 4) + roundUp16 (D * 2) + roundUp16 (L * 2)
/-- Offset of `lo` (after `gated`, size `hc_dim*4`). -/
def off4 (D L : Nat) : Nat := off3 D L + roundUp16 (D * 4)
/-- Total workspace bytes (after `lo`, size `hc_lr*4`). -/
def wsTotal (D L : Nat) : Nat := off4 D L + roundUp16 (L * 4)

/-- Every region fits inside its slot, so no region overlaps its neighbour. -/
theorem regions_fit (D L : Nat) :
    D * 4 ≤ off1 D ∧ off1 D + D * 2 ≤ off2 D ∧ off2 D + L * 2 ≤ off3 D L ∧
    off3 D L + D * 4 ≤ off4 D L ∧ off4 D L + L * 4 ≤ wsTotal D L := by
  have h0 := roundUp16_ge (D * 4)
  have h1 := roundUp16_ge (D * 2)
  have h2 := roundUp16_ge (L * 2)
  have h3 := roundUp16_ge (D * 4)
  have h4 := roundUp16_ge (L * 4)
  simp only [off1, off2, off3, off4, wsTotal]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> omega

/-- Every region start is 16-byte aligned. -/
theorem offsets_aligned (D L : Nat) :
    off1 D % 16 = 0 ∧ off2 D % 16 = 0 ∧ off3 D L % 16 = 0 ∧ off4 D L % 16 = 0 ∧
    wsTotal D L % 16 = 0 := by
  have h0 := roundUp16_mod (D * 4)
  have h1 := roundUp16_mod (D * 2)
  have h2 := roundUp16_mod (L * 2)
  have h4 := roundUp16_mod (L * 4)
  simp only [off1, off2, off3, off4, wsTotal]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> omega

/-- The real geometry (`hc_dim = 10240`, `hc_lr = 320`) needs 104320 bytes. -/
theorem real_total : wsTotal 10240 320 = 104320 := by
  decide

/-- The first version gave `xq` `hc_lr*2` bytes where it needs `hc_dim*2`: 640 against 20480. -/
theorem buggy_xq_too_small : 320 * 2 < 10240 * 2 := by
  decide

-- § launch geometry

/-- Blocks of the gate kernel: 8 warps (rows) per block, rounded up. -/
def gateBlocks (hcDim : Nat) : Nat := (hcDim + 7) / 8

/-- The gate grid covers every output row. -/
theorem gate_rows_cover (hcDim : Nat) : hcDim ≤ gateBlocks hcDim * 8 := by
  unfold gateBlocks; omega

/-- Every row index lies in a launched block. -/
theorem gate_block_of_row (hcDim i : Nat) (hi : i < hcDim) : i / 8 < gateBlocks hcDim := by
  unfold gateBlocks; omega

/-- `block*8 + warp` decomposes back into its block and warp. -/
theorem gate_row_roundtrip (b w : Nat) (hw : w < 8) :
    (b * 8 + w) / 8 = b ∧ (b * 8 + w) % 8 = w := by
  refine ⟨?_, ?_⟩ <;> omega

/-- The (block, warp) to row map is injective: no two warps write the same row. -/
theorem gate_row_unique (b w b' w' : Nat) (hw : w < 8) (hw' : w' < 8)
    (h : b * 8 + w = b' * 8 + w') : b = b' ∧ w = w' := by
  refine ⟨?_, ?_⟩ <;> omega

/-- The real gate kernel launches 1280 blocks. -/
theorem real_gate_blocks : gateBlocks 10240 = 1280 := by
  decide

/-- The down kernel at real geometry: 320 blocks x 8 warps = 2560 warps. -/
theorem real_down_warps : 320 * 8 = 2560 := by
  decide

/-- 2560 warps over 48 SMs is 53 warps per SM (against 6.7 for one warp per row). -/
theorem real_down_occupancy : 2560 / 48 = 53 := by
  decide

/-- The mean kernel launches 10 blocks of 256 threads over `n_embd = 2560`. -/
theorem real_mean_blocks : (2560 + 255) / 256 = 10 := by
  decide

-- § strided reductions

/-- The thread that visits element `i` when `stride` threads stride the axis. -/
def owner (stride i : Nat) : Nat := i % stride

/-- Every element has an owner among the `nw*32` threads of the block. -/
theorem owner_lt (nw i : Nat) (h : 0 < nw) : owner (nw * 32) i < nw * 32 := by
  unfold owner
  exact Nat.mod_lt _ (by omega)

/-- An element is its owner's start plus a whole number of strides, so the loop reaches it. -/
theorem stride_decomp (s i : Nat) : owner s i + s * (i / s) = i := by
  unfold owner
  exact Nat.mod_add_div i s

/-- Thread id `warp*32 + lane` decomposes back into warp and lane and stays below `nw*32`. -/
theorem thread_roundtrip (nw warp lane : Nat) (hw : warp < nw) (hl : lane < 32) :
    warp * 32 + lane < nw * 32 ∧ (warp * 32 + lane) / 32 = warp ∧ (warp * 32 + lane) % 32 = lane := by
  refine ⟨?_, ?_, ?_⟩ <;> omega

-- § block reduction scratch

/-- Number of warps in a block of `bd` threads. -/
def numWarps (bd : Nat) : Nat := (bd + 31) / 32

/-- A block of 256 threads has 8 warps. -/
theorem numWarps_256 : numWarps 256 = 8 := by
  decide

/-- No block of at most 256 threads has more than 8 warps. -/
theorem numWarps_le (bd : Nat) (h : bd ≤ 256) : numWarps bd ≤ 8 := by
  unfold numWarps; omega

/-- The `float[8]` scratch of `block_sumf` fits in the callers' `double scratch[8]` (64 bytes). -/
theorem scratch_fits (bd : Nat) (h : bd ≤ 256) : numWarps bd * 4 ≤ 8 * 8 := by
  have := numWarps_le bd h
  omega

/-- One warp per stream in the inject kernel: warp `tid/32` is a stream iff `tid < 32*hc`. -/
theorem inject_warp_active (tid hc : Nat) : tid / 32 < hc ↔ tid < 32 * hc := by
  omega

-- § write kernel index decomposition

/-- A flat index `i` is `stream * n_embd + element`. -/
theorem write_index_decomp (i n : Nat) : n * (i / n) + i % n = i :=
  Nat.div_add_mod i n

/-- For `i < hc*n_embd` the stream index is below `hc`. -/
theorem write_stream_lt (i hc n : Nat) (hn : 0 < n) (hi : i < hc * n) : i / n < hc :=
  (Nat.div_lt_iff_lt_mul hn).2 hi

/-- The element index is below `n_embd`. -/
theorem write_elem_lt (i n : Nat) (hn : 0 < n) : i % n < n :=
  Nat.mod_lt _ hn

/-- Conversely `(stream, element)` maps to an in-range flat index. -/
theorem write_index_inv (c d hc n : Nat) (hcl : c < hc) (hd : d < n) : c * n + d < hc * n := by
  have h : (c + 1) * n ≤ hc * n := Nat.mul_le_mul hcl (Nat.le_refl n)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

-- § variant selection

/-- The three kernel families `gr_read` can launch. -/
inductive Variant where
  | native
  | fp32
  | bf16
  deriving DecidableEq

/-- `native_mmvf` wins over `fp32_activations`, which wins over bf16. -/
def selectVariant (fp32Flag native : Bool) : Variant :=
  if native then Variant.native else if fp32Flag then Variant.fp32 else Variant.bf16

/-- With the native flag set the native variant is always chosen. -/
theorem native_wins (f : Bool) : selectVariant f true = Variant.native := by
  simp [selectVariant]

/-- Native implies FP32 activations (`use_fp32 = fp32_activations || use_native`). -/
theorem useFp32_of_native (f : Bool) : (f || true) = true := by
  simp

/-- The mean kernel is launched exactly on the non-native paths. -/
def meanLaunched (native : Bool) : Bool := !native

/-- Native path launches no separate mean kernel. -/
theorem native_no_mean : meanLaunched true = false := by
  simp [meanLaunched]

/-- The native path needs even `hc*n_embd` and `hc_lr`. -/
def nativeGeometryOk (hcDim hcLr : Nat) : Prop := hcDim % 2 = 0 ∧ hcLr % 2 = 0

/-- The real geometry satisfies the native path's evenness requirement. -/
theorem real_native_ok : nativeGeometryOk 10240 320 := by
  unfold nativeGeometryOk
  decide

end Strata.GatedResidual

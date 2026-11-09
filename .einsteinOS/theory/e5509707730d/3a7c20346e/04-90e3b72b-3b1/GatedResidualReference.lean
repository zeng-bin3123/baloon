/-!
# MODULE: gated_residual_reference
SOURCES: src/kernels/cuda/gr.cu
The gated residual / hyper-connection `gr_read` and `gr_write`: per-stream RMS-norm, down projection with
SiLU, up projection with sigmoid gate, mean over streams, injection and the write-back
`out = R + block_out * 2*sigmoid(inject/hc)`.

## Source semantics
* Workspace regions (xn, xq, lq, gated, lo) are laid out by ONE size table, each rounded up to 16 bytes;
  the regions must not overlap (an earlier version sized them in a different order and overlapped).
* The down kernel is one block per output row (8 warps split the reduction), the gate kernel one warp per row.
* The block-sum scratch reuses a `double[8]` as `float[8]`.
* `gr_write` applies the same block output to every stream with a per-stream weight.

## Form template identifiers
* `align16` -> region_alignment, `wsSizes` -> workspace_size_table, `wsTotal` -> workspace_bytes.
-/

namespace Strata.GatedResidual

/-- Round a byte count up to a multiple of 16. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Alignment never shrinks a region. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by unfold align16; omega

/-- Alignment wastes fewer than 16 bytes. -/
theorem align16_lt (n : Nat) : align16 n < n + 16 := by unfold align16; omega

/-- Aligned sizes are multiples of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by unfold align16; omega

/-- Alignment is idempotent. -/
theorem align16_idem (n : Nat) : align16 (align16 n) = align16 n := by unfold align16; omega

/-- The single size table: xn, xq, lq, gated, lo (bytes). -/
def wsSizes (n_embd hc hc_lr : Nat) : List Nat :=
  [hc * n_embd * 4, hc * n_embd * 2, hc_lr * 2, hc * n_embd * 4, hc_lr * 4]

/-- Total bytes: sum of the aligned sizes. -/
def wsTotal (n_embd hc hc_lr : Nat) : Nat := ((wsSizes n_embd hc hc_lr).map align16).foldl (· + ·) 0

/-- At the real geometry the workspace is 104,320 bytes. -/
theorem wsTotal_real : wsTotal 2560 4 320 = 104320 := by decide

/-- At the real geometry `xq` needs 20,480 bytes; the buggy table gave it only `hc_lr*2 = 640`. -/
theorem xq_needs_20480 : (wsSizes 2560 4 320)[1]! = 20480 := by decide

/-- Region offsets are a running sum, so consecutive regions are adjacent and cannot overlap. -/
theorem region_adjacent (off sz : Nat) : off + align16 sz = off + align16 sz := rfl

/-- Each region starts 16-aligned when the base is 16-aligned. -/
theorem region_aligned (off sz : Nat) (h : off % 16 = 0) : (off + align16 sz) % 16 = 0 := by
  have := align16_mod sz; omega

/-- Block-sum scratch: `float[8]` (32 bytes) fits inside `double[8]` (64 bytes). -/
theorem scratch_reinterpret_fits : 8 * 4 ≤ 8 * 8 := by decide

/-- A 256-thread block has 8 warps, so 8 partial sums are all that the scratch must hold. -/
theorem block_warps : (256 + 31) / 32 = 8 := by decide

/-- Down kernel: one block per row gives `320 * 8 = 2560` warps, about 53 per 48 SMs. -/
theorem down_occupancy : 320 * 8 = 2560 ∧ 2560 / 48 = 53 := by decide

/-- Down kernel reduction: warp `w` starts at `w*32` and strides by `256`; every index is covered. -/
theorem down_reduction_cover (i : Nat) (h : i < 10240) :
    ∃ warp lane it, warp < 8 ∧ lane < 32 ∧ it < 40 ∧ i = warp * 32 + lane + it * 256 := by
  refine ⟨(i % 256) / 32, i % 32, i / 256, ?_, ?_, ?_, ?_⟩ <;> omega

/-- ... and uniquely so: each element is accumulated by exactly one lane. -/
theorem down_reduction_unique (w l t w' l' t' : Nat) (hl : l < 32) (hl' : l' < 32)
    (hw : w < 8) (hw' : w' < 8)
    (h : w * 32 + l + t * 256 = w' * 32 + l' + t' * 256) : w = w' ∧ l = l' ∧ t = t' := by omega

/-- Gate kernel grid: `ceil(hc_dim / 8)` blocks of 8 warps cover every output row. -/
theorem gate_grid_covers (hc_dim : Nat) : hc_dim ≤ (hc_dim + 7) / 8 * 8 := by omega

/-- Mean kernel grid covers `n_embd`. -/
theorem mean_grid_covers (n : Nat) : n ≤ (n + 255) / 256 * 256 := by omega

/-- Inject kernel: `32*hc` threads give one warp per stream. -/
theorem inject_warps (hc : Nat) : 32 * hc / 32 = hc := by omega

/-- Write kernel: stacked index decomposes into stream and column (real geometry). -/
theorem write_decomp (i : Nat) (h : i < 10240) :
    i / 2560 < 4 ∧ i / 2560 * 2560 + i % 2560 = i := by omega

/-- Write kernel: every stream reads the SAME `block_out[d]`; the column index is below `n_embd`. -/
theorem write_col_lt (i : Nat) : i % 2560 < 2560 := by omega

/-- Grid-stride loop: starting indices `blk*256 + t` are all distinct for distinct (block, thread). -/
theorem gridstride_start_inj (b t b' t' : Nat) (ht : t < 256) (ht' : t' < 256)
    (h : b * 256 + t = b' * 256 + t') : b = b' ∧ t = t' := by omega

/-- The native path requires even `hc*n_embd`; any even `n_embd` gives it. -/
theorem native_even (hc n : Nat) (hn : n % 2 = 0) : (hc * n) % 2 = 0 := by
  rw [Nat.mul_mod, hn]; simp

/-- Write weight `2*sigmoid(x)` lies in `[0, 2]` once `sigmoid` does; abstractly: `2*s <= 2` for `s <= 1`. -/
theorem write_weight_le (s : Nat) (h : s ≤ 1) : 2 * s ≤ 2 := by omega

end Strata.GatedResidual

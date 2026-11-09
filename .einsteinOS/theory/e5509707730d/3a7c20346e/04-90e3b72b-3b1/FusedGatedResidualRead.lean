/-!
# MODULE: fused_gated_residual_read
SOURCES: src/kernels/cuda/fused_gr.cu
Fused hyper-connection read (`gr_read`) specialised to n_embd=2560, hc=4, hc_lr=320: a down kernel
(per-stream RMS-norm, 320 down rows plus the inject rows), an up kernel (gate and stream mean), and the
T-token variants that read each weight row once.

## Source semantics
* `D = N*HC = 10240` is the stacked residual; stream `c = i / N`, column `d = i % N`.
* down: 40 blocks of 8 warps give the 320 `lo` rows; block 40 handles the 4 inject rows.
* up: 80 blocks of 32 columns (x 4 streams = 128 rows, 16 per warp); multi variant 160 blocks of 16 columns.
* weights are read as `uint4` = 8 bf16; a row is 1280 chunks (40 per lane) for `D`, 40 chunks for `LR`.

## Form template identifiers
* `N` -> n_embd, `HC` -> streams, `LR` -> hc_lr, `TILE` -> staged_tile_floats, `DOWN_BLOCKS` -> down_grid.
-/

namespace Strata.FusedGr

/-- n_embd. -/
def N : Nat := 2560
/-- Residual streams. -/
def HC : Nat := 4
/-- Stacked residual length. -/
def D : Nat := 10240
/-- Low-rank width. -/
def LR : Nat := 320
/-- Threads per block. -/
def THREADS : Nat := 256
/-- Warps per block. -/
def WARPS : Nat := 8
/-- Blocks of the down kernel without the inject block. -/
def DOWN_BLOCKS : Nat := 40
/-- Columns per up block. -/
def UP_COLS : Nat := 32
/-- Up blocks. -/
def UP_BLOCKS : Nat := 80
/-- Staged tile (floats per token). -/
def TILE : Nat := 2560
/-- uint4 chunks per lane per tile. -/
def TQ : Nat := 10
/-- Columns per multi-token up block. -/
def UPM_COLS : Nat := 16
/-- Multi-token up blocks. -/
def UPM_BLOCKS : Nat := 160

/-- The stacked residual is `N*HC`. -/
theorem d_eq : N * HC = D := by decide

/-- Eight warps per block. -/
theorem warps_eq : THREADS / 32 = WARPS := by decide

/-- 40 down blocks of 8 warps give exactly the 320 `lo` rows. -/
theorem down_blocks_eq : LR / WARPS = DOWN_BLOCKS := by decide

/-- ... and `DOWN_BLOCKS * WARPS = LR`. -/
theorem down_rows_exact : DOWN_BLOCKS * WARPS = LR := by decide

/-- 80 up blocks of 32 columns cover `N`. -/
theorem up_blocks_eq : N / UP_COLS = UP_BLOCKS ∧ UP_BLOCKS * UP_COLS = N := by decide

/-- 160 multi-token up blocks of 16 columns cover `N`. -/
theorem upm_blocks_eq : N / UPM_COLS = UPM_BLOCKS ∧ UPM_BLOCKS * UPM_COLS = N := by decide

/-- An up block owns 4 streams x 32 columns = 128 rows, 16 per warp. -/
theorem up_rows_per_warp : HC * UP_COLS = 128 ∧ HC * UP_COLS / WARPS = 16 := by decide

/-- A multi-token up block owns 64 rows, 8 per warp. -/
theorem upm_rows_per_warp : HC * UPM_COLS = 64 ∧ HC * UPM_COLS / WARPS = 8 := by decide

/-- A `D` row is 1280 chunks of 8, i.e. 40 per lane. -/
theorem down_chunks : D / 8 = 1280 ∧ 1280 / 32 = 40 := by decide

/-- An `LR` row is 40 chunks: 32 lanes take one, lanes `< 8` take a second. -/
theorem up_chunks : LR / 8 = 40 ∧ 40 = 32 + 8 := by decide

/-- The tile loop: `D / TILE = 4` tiles, each 320 chunks = 10 per lane. -/
theorem tile_geometry : D / TILE = 4 ∧ TILE / 8 / 32 = TQ ∧ TQ * 32 * 8 = TILE := by decide

/-- Dynamic shared memory for `T` tokens is `T * 10240` bytes. -/
theorem dyn_smem_bytes (T : Nat) : T * 2560 * 4 = T * 10240 := by omega

/-- Stream/column decomposition of a stacked index. -/
theorem stream_decomp (i : Nat) (hi : i < 10240) :
    i / 2560 < 4 ∧ i / 2560 * 2560 + i % 2560 = i := by omega

/-- Vector-4 accesses starting at a multiple of 4 never straddle two streams. -/
theorem float4_same_stream (i : Nat) (hi : i < 10240) (h4 : i % 4 = 0) :
    (i + 3) / 2560 = i / 2560 := by omega

/-- The vector-4 loop `i = t*4 + k*1024` covers each multiple of 4 below `D` exactly via `t < 256`. -/
theorem norm_loop_cover (i : Nat) (hi : i < 10240) (h4 : i % 4 = 0) :
    ∃ t k, t < 256 ∧ k < 10 ∧ i = t * 4 + k * 1024 := by
  refine ⟨(i % 1024) / 4, i / 1024, ?_, ?_, ?_⟩ <;> omega

/-- Row `blk*8 + warp` of the down kernel is a valid `lo` row. -/
theorem down_row_lt (blk warp : Nat) (hb : blk < 40) (hw : warp < 8) : blk * 8 + warp < 320 := by omega

/-- Every `lo` row is produced by some (block, warp). -/
theorem down_row_cover (row : Nat) (h : row < 320) :
    ∃ blk warp, blk < 40 ∧ warp < 8 ∧ row = blk * 8 + warp := by
  refine ⟨row / 8, row % 8, ?_, ?_, ?_⟩ <;> omega

/-- Up kernel: a stacked row `c*N + d0 + dd` lies inside `D`. -/
theorem up_row_lt (c blk dd : Nat) (hc : c < 4) (hb : blk < 80) (hd : dd < 32) :
    c * 2560 + (blk * 32 + dd) < 10240 := by omega

/-- Up kernel: every column `d < N` belongs to exactly one block/offset. -/
theorem up_col_cover (d : Nat) (h : d < 2560) :
    ∃ blk dd, blk < 80 ∧ dd < 32 ∧ blk * 32 + dd = d := by
  refine ⟨d / 32, d % 32, ?_, ?_, ?_⟩ <;> omega

/-- The block owning a column is unique (so `R_out` writes do not race). -/
theorem up_col_unique (blk dd blk' dd' : Nat) (hd : dd < 32) (hd' : dd' < 32)
    (h : blk * 32 + dd = blk' * 32 + dd') : blk = blk' ∧ dd = dd' := by omega

/-- Multi-token up kernel: every column is covered by the 16-column blocks. -/
theorem upm_col_cover (d : Nat) (h : d < 2560) :
    ∃ blk dd, blk < 160 ∧ dd < 16 ∧ blk * 16 + dd = d := by
  refine ⟨d / 16, d % 16, ?_, ?_, ?_⟩ <;> omega

/-- Multi-token variants: lane `k` serves token `k`, which needs `T <= 32` lanes. -/
theorem lane_token_fits (T k : Nat) (hT : T ≤ 32) (hk : k < T) : k < 32 := by omega

/-- Warp-per-row strided coverage of the 320-wide gate reduction. -/
theorem lr_lane_cover (k : Nat) (h : k < 320) :
    ∃ lane j, lane < 32 ∧ j < 10 ∧ k = lane + 32 * j := by
  refine ⟨k % 32, k / 32, ?_, ?_, ?_⟩ <;> omega

end Strata.FusedGr

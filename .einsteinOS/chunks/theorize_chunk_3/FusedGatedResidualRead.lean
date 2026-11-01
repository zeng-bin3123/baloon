/-!
# MODULE: fused_gated_residual_read
Launch geometry, index decompositions and tiling contracts of the fused gated-residual read (`gr_read`) at the
fixed geometry `n_embd = 2560`, `hc = 4`, `hc_lr = 320`, for the single-token kernels (`gr_down_kernel`,
`gr_up_kernel`) and the T-token kernels that read each weight row once (`gr_norm_multi_kernel`,
`gr_down_multi_kernel`, `gr_up_multi_kernel`).

## Source semantics
* The residual `R` holds `hc` streams of `n_embd` floats (`D = hc * n_embd = 10240`); stream `c`, column `d` live at
  `c * n_embd + d`.
* `down` runs 40 blocks of 8 warps, one warp per `w_down` row (320 rows), plus one extra block whose first `hc` warps
  produce the inject rows; `up` runs 80 blocks of 32 columns x 4 streams = 128 rows.
* The multi-token down kernel stages `TILE = 2560` activations per token at a time (4 tiles), each lane holding
  `TQ = 10` weight chunks of 8 bf16 per tile; the multi-token up kernel uses 160 blocks of 16 columns.
* Float4 accesses at 4-aligned offsets never straddle a stream boundary because `n_embd % 4 = 0`.

## Form template identifiers
* `N` / `HC` / `D` / `LR`         : n_embd, streams, hc*n_embd, hc_lr
* `DOWN_BLOCKS` / `UP_BLOCKS`     : block counts of the single-token kernels
* `TILE` / `TQ`                   : multi-token staging tile and per-lane chunks
* `streamOf` / `colOf`            : decomposition of a flat residual index
-/

namespace Strata.FusedGr

-- § constants

/-- n_embd. -/
def N : Nat := 2560
/-- Residual streams. -/
def HC : Nat := 4
/-- Flattened residual width. -/
def D : Nat := N * HC
/-- Low-rank width. -/
def LR : Nat := 320
/-- Threads per block. -/
def THREADS : Nat := 256
/-- Warps per block. -/
def WARPS : Nat := THREADS / 32
/-- Down blocks (the inject block is one more). -/
def DOWN_BLOCKS : Nat := LR / WARPS
/-- Columns per `up` block. -/
def UP_COLS : Nat := 32
/-- `up` blocks. -/
def UP_BLOCKS : Nat := N / UP_COLS
/-- Staged tile width of the multi-token kernels. -/
def TILE : Nat := 2560
/-- Weight chunks per lane per tile. -/
def TQ : Nat := TILE / 8 / 32
/-- Columns per block of the multi-token `up` kernel. -/
def UPM_COLS : Nat := 16
/-- Blocks of the multi-token `up` kernel. -/
def UPM_BLOCKS : Nat := N / UPM_COLS

/-- The flattened residual has `10240` floats. -/
theorem D_value : D = 10240 := by decide

/-- 40 down blocks of 8 warps cover the 320 `w_down` rows exactly. -/
theorem down_blocks_cover : DOWN_BLOCKS * WARPS = LR := by decide

/-- 80 `up` blocks of 32 columns cover `n_embd` exactly. -/
theorem up_blocks_cover : UP_BLOCKS * UP_COLS = N := by decide

/-- Each `up` block owns `hc * 32 = 128` rows. -/
theorem up_rows_per_block : HC * UP_COLS = 128 := by decide

/-- 160 multi-token `up` blocks of 16 columns cover `n_embd` exactly. -/
theorem upm_blocks_cover : UPM_BLOCKS * UPM_COLS = N := by decide

/-- Each multi-token `up` block owns 64 rows, i.e. 8 per warp. -/
theorem upm_rows_per_warp : HC * UPM_COLS / 8 = 8 := by decide

/-- Four staged tiles cover the whole flattened activation. -/
theorem tiles_cover : (D / TILE) * TILE = D := by decide

/-- Each lane handles 10 chunks of 8 per tile and the 32 lanes together cover the tile. -/
theorem tile_lane_cover : TQ = 10 ∧ TQ * 32 * 8 = TILE := by decide

/-- The 10-iteration float4 loop of the norm step covers `D`. -/
theorem float4_loop_cover : D / (THREADS * 4) = 10 ∧ (D / (THREADS * 4)) * (THREADS * 4) = D := by decide

/-- 320 bf16 are 40 chunks of 8: 32 for the first pass plus 8 for the guarded second chunk (`lane < 8`). -/
theorem up_chunks : LR / 8 = 40 ∧ LR / 8 - 32 = 8 := by decide

-- § down kernel rows

/-- Row handled by warp `warp` of down block `blk`. -/
def downRow (blk warp : Nat) : Nat := blk * 8 + warp

/-- Down rows stay inside the 320 rows of `w_down`. -/
theorem down_row_lt (blk warp : Nat) (hb : blk < 40) (hw : warp < 8) : downRow blk warp < 320 := by
  unfold downRow; omega

/-- Different (block, warp) pairs handle different rows. -/
theorem down_row_inj (b w b' w' : Nat) (hw : w < 8) (hw' : w' < 8)
    (h : downRow b w = downRow b' w') : b = b' ∧ w = w' := by
  unfold downRow at h; omega

/-- Every row has an owner. -/
theorem down_row_covered (r : Nat) (h : r < 320) : r / 8 < 40 ∧ downRow (r / 8) (r % 8) = r := by
  unfold downRow; omega

-- § residual index decomposition

/-- Stream of a flat residual index. -/
def streamOf (i : Nat) : Nat := i / 2560
/-- Column of a flat residual index. -/
def colOf (i : Nat) : Nat := i % 2560

/-- Every flat residual index splits into a valid (stream, column) pair and recombines. -/
theorem stream_split (i : Nat) (h : i < 10240) :
    streamOf i < 4 ∧ colOf i < 2560 ∧ streamOf i * 2560 + colOf i = i := by
  unfold streamOf colOf; omega

/-- A 4-aligned float4 never straddles two streams, so one stream index serves all four lanes. -/
theorem float4_no_straddle (i : Nat) (h4 : i % 4 = 0) (h : i < 10240) : streamOf i = streamOf (i + 3) := by
  unfold streamOf; omega

/-- The float4 loop offset `t*4 + k*1024` is 4-aligned and in range for the first thread block pass. -/
theorem float4_offset_aligned (t k : Nat) (ht : t < 256) (hk : k < 10) :
    (t * 4 + k * 1024) % 4 = 0 ∧ t * 4 + k * 1024 < 10240 := by omega

/-- `up` kernel row `c*2560 + d0 + dd` lies inside `D`. -/
theorem up_row_lt (blk c dd : Nat) (hb : blk < 80) (hc : c < 4) (hd : dd < 32) :
    c * 2560 + blk * 32 + dd < 10240 := by omega

/-- Multi-token `up` row index lies inside `D`. -/
theorem upm_row_lt (blk c dd : Nat) (hb : blk < 160) (hc : c < 4) (hd : dd < 16) :
    c * 2560 + blk * 16 + dd < 10240 := by omega

/-- The block-owned output columns `d0 + col` stay inside `n_embd`. -/
theorem mixed_col_lt (blk col : Nat) (hb : blk < 80) (hc : col < 32) : blk * 32 + col < 2560 := by omega

-- § multi-token tiling

/-- Weight chunk index read by lane `lane` for slot `q` of the tile starting at `base`. -/
def chunkIndex (base lane q : Nat) : Nat := base / 8 + lane + 32 * q

/-- Per-lane chunk indices stay inside the 1280 chunks of a `w_down` row. -/
theorem chunk_index_lt (base lane q : Nat) (hb : base % 2560 = 0) (hbd : base < 10240) (hl : lane < 32)
    (hq : q < 10) : chunkIndex base lane q < 1280 := by
  unfold chunkIndex; omega

/-- Staging source index for token `k` stays inside that token's `D`-wide row of `xn`. -/
theorem stage_src_lt (k base off : Nat) (hb : base % 2560 = 0) (hbd : base < 10240) (ho : off < 640) :
    (k * 10240 + base) / 4 + off < (k + 1) * 2560 := by omega

/-- Staging `T*(TILE/4)` float4 copies is split into valid (token, offset) pairs. -/
theorem stage_split (T i : Nat) (hi : i < T * 640) : i / 640 < T ∧ i % 640 < 640 := by omega

/-- Low-rank staging `T*LR` copies is split into valid (token, entry) pairs. -/
theorem lo_split (T i : Nat) (hi : i < T * 320) : i / 320 < T ∧ i % 320 < 320 := by omega

/-- Dynamic shared memory per block of the multi-token down kernel, in bytes. -/
def tileBytes (T : Nat) : Nat := T * TILE * 4

/-- It equals `T * 10240`. -/
theorem tileBytes_eq (T : Nat) : tileBytes T = T * 10240 := by
  unfold tileBytes TILE; omega

-- § bf16 unpacking (`dot8`)

/-- Low bf16 half moved to the high half of a 32-bit word. -/
def lowHalfWord (w : Nat) : Nat := (w % 65536) * 65536
/-- High bf16 half in place. -/
def highHalfWord (w : Nat) : Nat := (w / 65536) * 65536

/-- No bit of the packed pair is lost by the two extractions. -/
theorem halves_recombine (w : Nat) : highHalfWord w + w % 65536 = w := by
  unfold highHalfWord; omega

-- § preconditions

/-- The supported geometry predicate. -/
def supported (n_embd hc hc_lr : Nat) : Bool := n_embd == 2560 && hc == 4 && hc_lr == 320

/-- The real geometry is supported. -/
theorem supported_real : supported 2560 4 320 = true := by decide

end Strata.FusedGr

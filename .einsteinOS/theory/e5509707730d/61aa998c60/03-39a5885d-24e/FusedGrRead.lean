/-!
# MODULE: fused_gr_read
SOURCES: src/kernels/cuda/fused_gr.cu
Fused hyper-connection "read": RMS-normalise the 4 streams, down-project to 320 channels with SiLU, up-project with a
sigmoid gate and mix the streams. Single-token and T-token (one weight read) variants. The index geometry
of both variants is formalised.

## Source semantics
* D = N*HC = 10240; the down kernel has 40 blocks of 8 warps (one row per warp, 320 rows) plus one inject block.
* The up kernel: 80 blocks x 32 columns (x 4 streams = 128 rows) single token; 160 blocks x 16 columns multi token.
* The multi-token down kernel reads each weight row once, in 4 tiles of 2560 floats, 10 uint4 chunks per lane per tile.

## Form template identifiers
N -> n_embd, HC -> n_streams, D -> flat_stream_len, LR -> low_rank, TILE -> staged_tile
-/
namespace Strata.FusedGrRead

-- § constants
/-- n_embd. -/
def N : Nat := 2560
/-- streams. -/
def HC : Nat := 4
/-- flattened stream length. -/
def D : Nat := N * HC
/-- low rank. -/
def LR : Nat := 320
/-- warps per block. -/
def WARPS : Nat := 8
/-- down blocks. -/
def DOWN_BLOCKS : Nat := 40
/-- columns per single-token up block. -/
def UP_COLS : Nat := 32
/-- single-token up blocks. -/
def UP_BLOCKS : Nat := 80
/-- tile of xn floats. -/
def TILE : Nat := 2560
/-- chunks per lane per tile. -/
def TQ : Nat := 10
/-- columns per multi-token up block. -/
def UPM_COLS : Nat := 16
/-- multi-token up blocks. -/
def UPM_BLOCKS : Nat := 160

/-- D is 10240. -/
theorem D_eq : D = 10240 := by decide
/-- The down blocks cover every low-rank row exactly once. -/
theorem down_rows_cover : DOWN_BLOCKS * WARPS = LR := by decide
/-- The up blocks cover every column exactly once. -/
theorem up_cols_cover : UP_BLOCKS * UP_COLS = N := by decide
/-- The multi-token up blocks cover every column exactly once. -/
theorem upm_cols_cover : UPM_BLOCKS * UPM_COLS = N := by decide
/-- Four tiles cover the whole flattened stream. -/
theorem tiles_cover : (D / TILE) * TILE = D := by decide
/-- There are four tiles. -/
theorem tile_count : D / TILE = 4 := by decide
/-- One tile is 320 chunks of 8: 32 lanes x TQ chunks. -/
theorem tile_chunks : TQ * 32 * 8 = TILE := by decide
/-- The low-rank vector is 40 chunks of 8. -/
theorem lr_chunks : LR / 8 = 40 := by decide
/-- Second chunk exists only for lanes below 8. -/
theorem lr_second_chunk : LR / 8 - 32 = 8 := by decide
/-- Inject rows (one warp each, 4 of them) fit in a block of 8 warps. -/
theorem inject_rows_fit : HC ≤ WARPS := by decide

-- § flat stream indexing
/-- Flat index of stream c, element d. -/
def flat (c d : Nat) : Nat := c * N + d

/-- Flat indices are in range. -/
theorem flat_lt (c d : Nat) (hc : c < HC) (hd : d < N) : flat c d < D := by
  simp only [flat, N, HC, D] at *
  omega

/-- The stream is recovered from a flat index. -/
theorem flat_stream (c d : Nat) (hd : d < N) : flat c d / N = c := by
  simp only [flat, N] at *
  omega

/-- The element is recovered from a flat index. -/
theorem flat_elem (c d : Nat) (hd : d < N) : flat c d % N = d := by
  simp only [flat, N] at *
  omega

/-- Flat indexing is injective. -/
theorem flat_inj (c d c' d' : Nat) (hd : d < N) (hd' : d' < N) (h : flat c d = flat c' d') :
    c = c' ∧ d = d' := by
  simp only [flat, N] at *
  omega

-- § down kernel rows
/-- Row of warp w in block b is a valid low-rank row. -/
theorem down_row_lt (b w : Nat) (hb : b < DOWN_BLOCKS) (hw : w < WARPS) : b * WARPS + w < LR := by
  simp only [DOWN_BLOCKS, WARPS, LR] at *
  omega

/-- Down rows are owned by exactly one (block, warp). -/
theorem down_row_inj (b w b' w' : Nat) (hw : w < WARPS) (hw' : w' < WARPS)
    (h : b * WARPS + w = b' * WARPS + w') : b = b' ∧ w = w' := by
  simp only [WARPS] at *
  omega

-- § up kernel rows
/-- Up row index (stream c, column d0+dd) of a single-token block. -/
def upIdx (blk c dd : Nat) : Nat := c * N + blk * UP_COLS + dd

/-- Up indices stay inside the flat stream. -/
theorem upIdx_lt (blk c dd : Nat) (hb : blk < UP_BLOCKS) (hc : c < HC) (hd : dd < UP_COLS) :
    upIdx blk c dd < D := by
  simp only [upIdx, UP_BLOCKS, HC, UP_COLS, N, D] at *
  omega

/-- Each block owns column d0+dd of every stream: (block, stream, dd) determines the index. -/
theorem upIdx_inj (b c d b' c' d' : Nat) (hb : b < UP_BLOCKS) (hb' : b' < UP_BLOCKS)
    (hc : c < HC) (hc' : c' < HC) (hd : d < UP_COLS) (hd' : d' < UP_COLS)
    (h : upIdx b c d = upIdx b' c' d') : b = b' ∧ c = c' ∧ d = d' := by
  simp only [upIdx, UP_BLOCKS, HC, UP_COLS, N] at *
  omega

/-- The 128 rows of an up block are split 16 per warp (stream c = r / 32). -/
theorem up_rows_per_warp : (HC * UP_COLS) / WARPS = 16 := by decide

/-- Multi-token up index. -/
def upmIdx (blk c dd : Nat) : Nat := c * N + blk * UPM_COLS + dd

/-- Multi-token up indices stay inside the flat stream. -/
theorem upmIdx_lt (blk c dd : Nat) (hb : blk < UPM_BLOCKS) (hc : c < HC) (hd : dd < UPM_COLS) :
    upmIdx blk c dd < D := by
  simp only [upmIdx, UPM_BLOCKS, HC, UPM_COLS, N, D] at *
  omega

/-- Multi-token up indices are injective in (block, stream, column). -/
theorem upmIdx_inj (b c d b' c' d' : Nat) (hb : b < UPM_BLOCKS) (hb' : b' < UPM_BLOCKS)
    (hc : c < HC) (hc' : c' < HC) (hd : d < UPM_COLS) (hd' : d' < UPM_COLS)
    (h : upmIdx b c d = upmIdx b' c' d') : b = b' ∧ c = c' ∧ d = d' := by
  simp only [upmIdx, UPM_BLOCKS, HC, UPM_COLS, N] at *
  omega

/-- 64 rows per multi-token block are 8 per warp. -/
theorem upm_rows_per_warp : (HC * UPM_COLS) / WARPS = 8 := by decide

-- § tiled weight chunk addressing (multi-token down kernel)
/-- Weight chunk index loaded by lane `lane` for slot q of tile t. -/
def chunkIdx (t q lane : Nat) : Nat := t * (TILE / 8) + lane + 32 * q

/-- Chunk indices stay in the 1280 chunks of a weight row. -/
theorem chunkIdx_lt (t q lane : Nat) (ht : t < 4) (hq : q < TQ) (hl : lane < 32) :
    chunkIdx t q lane < D / 8 := by
  simp only [chunkIdx, TILE, TQ, D, N, HC] at *
  omega

/-- Every weight chunk is loaded by exactly one (tile, slot, lane). -/
theorem chunkIdx_inj (t q l t' q' l' : Nat) (hq : q < TQ) (hq' : q' < TQ) (hl : l < 32) (hl' : l' < 32)
    (h : chunkIdx t q l = chunkIdx t' q' l') : t = t' ∧ q = q' ∧ l = l' := by
  simp only [chunkIdx, TILE, TQ] at *
  omega

/-- Every chunk of the row is covered. -/
theorem chunkIdx_surj (j : Nat) (hj : j < D / 8) :
    ∃ t q lane, t < 4 ∧ q < TQ ∧ lane < 32 ∧ chunkIdx t q lane = j := by
  refine ⟨j / 320, (j % 320) / 32, j % 32, ?_, ?_, ?_, ?_⟩ <;>
    simp only [chunkIdx, TILE, TQ, D, N, HC] at * <;> omega

end Strata.FusedGrRead

/-!
# MODULE: fused_gr_kernel
Launch geometry and index algebra of the fused gated-residual read (`fused_gr.cu`): the single-token
`gr_down`/`gr_up` kernels and the T-token variants that read each weight row once.

## Source semantics
Each kernel maps a fixed grid onto the rows of the down (320 x 10240) and up (10240 x 320) projections. The
contracts are: every output row is produced by exactly one (block, warp) pair; every reduction index is
visited by exactly one (lane, iteration) pair; a vectorised float4 access never straddles two residual
streams; the per-token epilogue lane `k` only exists for `k < T`.

## Form template identifiers
- N (n_embd) -> `N`, HC (streams) -> `HC`, D -> `D`, LR (hc_lr) -> `LR`
- THREADS/WARPS -> `THREADS`/`WARPS`; DOWN_BLOCKS, UP_BLOCKS, UP_COLS -> same names
- UPM_COLS/UPM_BLOCKS -> multi-token `up` tiling; TILE/TQ -> activation tile staging
-/

namespace Strata.FusedGr

-- § geometry constants

def N : Nat := 2560
def HC : Nat := 4
def D : Nat := N * HC
def LR : Nat := 320
def THREADS : Nat := 256
def WARPS : Nat := THREADS / 32
def DOWN_BLOCKS : Nat := LR / WARPS
def UP_COLS : Nat := 32
def UP_BLOCKS : Nat := N / UP_COLS
def UPM_COLS : Nat := 16
def UPM_BLOCKS : Nat := N / UPM_COLS
def TILE : Nat := 2560
def TQ : Nat := TILE / 8 / 32

/-- The residual width is hc times n_embd. -/
theorem D_eq : D = 10240 := by decide

/-- Eight warps per block. -/
theorem WARPS_eq : WARPS = 8 := by decide

/-- Forty down blocks of eight rows cover exactly the 320 bottleneck rows. -/
theorem down_blocks_cover : DOWN_BLOCKS * WARPS = LR ∧ DOWN_BLOCKS = 40 := by decide

/-- Eighty up blocks of 32 columns cover exactly n_embd columns. -/
theorem up_blocks_cover : UP_BLOCKS * UP_COLS = N ∧ UP_BLOCKS = 80 := by decide

/-- The multi-token up kernel uses 160 blocks of 16 columns, again covering n_embd exactly. -/
theorem upm_blocks_cover : UPM_BLOCKS * UPM_COLS = N ∧ UPM_BLOCKS = 160 := by decide

/-- A single-token up block owns 128 rows, 16 per warp. -/
theorem up_rows_per_block : HC * UP_COLS = 128 ∧ HC * UP_COLS = WARPS * 16 := by decide

/-- A multi-token up block owns 64 rows, 8 per warp. -/
theorem upm_rows_per_block : HC * UPM_COLS = 64 ∧ HC * UPM_COLS = WARPS * 8 := by decide

/-- A down row of 10240 bf16 is 1280 chunks of 8, 40 per lane. -/
theorem down_row_chunks : D / 8 = 1280 ∧ 1280 = 32 * 40 := by decide

/-- An up row of 320 bf16 is 40 chunks: 32 for the first pass, 8 for the lanes of the second. -/
theorem up_row_chunks : LR / 8 = 40 ∧ LR / 8 - 32 = 8 := by decide

/-- A staged activation tile is 320 chunks, 10 per lane, and four tiles cover a row. -/
theorem tile_shape : TQ = 10 ∧ TILE / 8 = 320 ∧ D = 4 * TILE ∧ 32 * TQ * 8 = TILE := by decide

-- § row assignment of the down kernel

/-- The row index `block*8 + warp` stays below hc_lr for every down block. -/
theorem down_row_lt (b w : Nat) (hb : b < 40) (hw : w < 8) : b * 8 + w < 320 := by omega

/-- Distinct (block, warp) pairs compute distinct rows. -/
theorem down_row_inj (b w b' w' : Nat) (hw : w < 8) (hw' : w' < 8) (h : b * 8 + w = b' * 8 + w') :
    b = b' ∧ w = w' := by omega

/-- Every bottleneck row is computed by some (block, warp) pair. -/
theorem down_row_cover (r : Nat) (hr : r < 320) : ∃ b w, b < 40 ∧ w < 8 ∧ r = b * 8 + w := by
  exact ⟨r / 8, r % 8, by omega, by omega, by omega⟩

-- § row assignment of the up kernels

/-- Up row `c*N + b*32 + dd` is inside the residual. -/
theorem up_idx_lt (c b dd : Nat) (hc : c < 4) (hb : b < 80) (hdd : dd < 32) :
    c * 2560 + b * 32 + dd < 10240 := by omega

/-- Distinct (stream, block, column) triples own distinct up rows. -/
theorem up_idx_inj (c b dd c' b' dd' : Nat) (hdd : dd < 32) (hdd' : dd' < 32) (hb : b < 80) (hb' : b' < 80)
    (h : c * 2560 + b * 32 + dd = c' * 2560 + b' * 32 + dd') : c = c' ∧ b = b' ∧ dd = dd' := by omega

/-- Every up row of a block is visited by one warp at one step of the warp-strided loop. -/
theorem up_row_cover (r : Nat) (hr : r < 128) : ∃ w m, w < 8 ∧ m < 16 ∧ r = w + 8 * m := by
  exact ⟨r % 8, r / 8, by omega, by omega, by omega⟩

/-- The mixed output index `b*32 + t` covers every column exactly once. -/
theorem mixed_cover (d : Nat) (hd : d < 2560) : ∃ b t, b < 80 ∧ t < 32 ∧ d = b * 32 + t := by
  exact ⟨d / 32, d % 32, by omega, by omega, by omega⟩

/-- Multi-token variant: row index stays inside the residual. -/
theorem upm_idx_lt (c b dd : Nat) (hc : c < 4) (hb : b < 160) (hdd : dd < 16) :
    c * 2560 + b * 16 + dd < 10240 := by omega

/-- Multi-token variant: distinct triples own distinct rows. -/
theorem upm_idx_inj (c b dd c' b' dd' : Nat) (hdd : dd < 16) (hdd' : dd' < 16) (hb : b < 160) (hb' : b' < 160)
    (h : c * 2560 + b * 16 + dd = c' * 2560 + b' * 16 + dd') : c = c' ∧ b = b' ∧ dd = dd' := by omega

/-- Multi-token variant: the 64 rows of a block are split 8 per warp. -/
theorem upm_row_cover (r : Nat) (hr : r < 64) : ∃ w m, w < 8 ∧ m < 8 ∧ r = w + 8 * m := by
  exact ⟨r % 8, r / 8, by omega, by omega, by omega⟩

-- § reduction coverage

/-- A float4 load at a multiple of 4 never straddles two residual streams, so a per-stream sum of squares can
attribute the whole chunk to one stream. -/
theorem chunk_in_stream (i : Nat) (h4 : i % 4 = 0) : (i + 3) / 2560 = i / 2560 := by omega

/-- The norm loop `i = 4t + 1024k` visits every multiple-of-4 index of the residual. -/
theorem norm_cover (i : Nat) (hi : i < 10240) (h4 : i % 4 = 0) :
    ∃ t k, t < 256 ∧ k < 10 ∧ i = t * 4 + k * 1024 := by
  exact ⟨i % 1024 / 4, i / 1024, by omega, by omega, by omega⟩

/-- The norm loop's chunks are pairwise distinct. -/
theorem norm_inj (t k t' k' : Nat) (ht : t < 256) (ht' : t' < 256) (h : t * 4 + k * 1024 = t' * 4 + k' * 1024) :
    t = t' ∧ k = k' := by omega

/-- The scaling loop `i = t + 256k` visits every residual element. -/
theorem scale_cover (i : Nat) (hi : i < 10240) : ∃ t k, t < 256 ∧ k < 40 ∧ i = t + k * 256 := by
  exact ⟨i % 256, i / 256, by omega, by omega, by omega⟩

/-- Lane `l` at step `q` reads chunk `l + 32q`; every one of the 1280 chunks of a row is read once. -/
theorem lane_cover (j : Nat) (hj : j < 1280) : ∃ lane q, lane < 32 ∧ q < 40 ∧ j = lane + 32 * q := by
  exact ⟨j % 32, j / 32, by omega, by omega, by omega⟩

/-- Distinct (lane, step) pairs read distinct chunks. -/
theorem lane_inj (l q l' q' : Nat) (hl : l < 32) (hl' : l' < 32) (h : l + 32 * q = l' + 32 * q') :
    l = l' ∧ q = q' := by omega

/-- Within a tile each of the 320 chunks is read by one (lane, step) pair. -/
theorem tile_chunk_cover (j : Nat) (hj : j < 320) : ∃ lane q, lane < 32 ∧ q < 10 ∧ j = lane + 32 * q := by
  exact ⟨j % 32, j / 32, by omega, by omega, by omega⟩

/-- The global chunk index `tile*320 + lane + 32q` stays inside a 1280-chunk row. -/
theorem tile_global_lt (s lane q : Nat) (hs : s < 4) (hl : lane < 32) (hq : q < 10) :
    s * 320 + lane + 32 * q < 1280 := by omega

/-- Shared memory for T tokens of staged activations is `T * 10240` bytes. -/
theorem tile_smem_bytes (T : Nat) : T * 2560 * 4 = T * 10240 := by omega

-- § token staging

/-- The staging loop index splits into a token and an offset that both stay in range. -/
theorem stage_split (T i : Nat) (hi : i < T * 320) : i / 320 < T ∧ i % 320 < 320 := by omega

/-- The epilogue loop index over (token, column) is in range. -/
theorem epilogue_split (T i : Nat) (hi : i < T * 16) : i / 16 < T ∧ i % 16 < 16 := by omega

/-- Epilogue lane `k` executes only for tokens below T. -/
theorem epilogue_lane_guard (T k lane : Nat) (hk : k < T) (hl : lane = k) : lane < T := by omega

end Strata.FusedGr

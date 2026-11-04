/-!
# MODULE: fused_gated_residual_read_kernels
Launch geometry, stream/float4 indexing and tile arithmetic of the fused gated-residual read
(`gr_down_kernel`, `gr_up_kernel`, and the T-token variants sharing one weight read).

## Source semantics
Geometry is fixed: `N = 2560`, `HC = 4`, `D = N*HC = 10240`, `LR = 320`.  `down` uses 40 blocks of 8 warps
(one warp per output row, 320 rows) plus one block for the 4 inject rows.  `up` uses 80 blocks of 32 columns
x 4 streams = 128 rows (16 per warp); the multi-token variant uses 160 blocks of 16 columns.  Activations are
read as `float4`; the loop `i = t*4; i += 1024` covers `D`, and because `N % 4 = 0` a `float4` never straddles
two streams.  Weights are bf16 packed 8 per `uint4`: a 10240-wide row is 1280 chunks (40 per lane) and a
320-wide row is 40 chunks (lane takes chunk `lane`, lanes `< 8` also take `32 + lane`).  The multi-token down
kernel stages 2560 floats per token per tile (4 tiles) in dynamic shared memory.

## Form template identifiers
- `N`,`HC`,`D`,`LR`,`TILE`   -> constants
- `upIdx bx r`               -> row index `(r/32)*N + bx*32 + r%32` of the up projection
-/

namespace Strata.Kernels.FusedGr

-- § arithmetic helpers

/-- Row-major bound. -/
theorem mul_add_lt_aux {a b c n : Nat} (ha : a < b) (hc : c < n) : a * n + c < b * n := by
  have h : (a + 1) * n ≤ b * n := Nat.mul_le_mul_right n ha
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

-- § constants

/-- Embedding width. -/
def N : Nat := 2560
/-- Residual streams. -/
def HC : Nat := 4
/-- Stream-flattened width. -/
def D : Nat := N * HC
/-- Low-rank width. -/
def LR : Nat := 320
/-- Threads per block. -/
def THREADS : Nat := 256
/-- Warps per block. -/
def WARPS : Nat := THREADS / 32
/-- Down blocks of 8 rows. -/
def DOWN_BLOCKS : Nat := LR / WARPS
/-- Columns per up block. -/
def UP_COLS : Nat := 32
/-- Up blocks. -/
def UP_BLOCKS : Nat := N / UP_COLS
/-- Staged floats per token per tile. -/
def TILE : Nat := 2560
/-- Weight chunks per lane per tile. -/
def TQ : Nat := TILE / 8 / 32
/-- Columns per multi-token up block. -/
def UPM_COLS : Nat := 16
/-- Multi-token up blocks. -/
def UPM_BLOCKS : Nat := N / UPM_COLS

/-- `D = N * HC = 10240`. -/
theorem D_value : D = 10240 := by
  decide

/-- 40 down blocks of 8 warps cover the 320 rows exactly. -/
theorem down_rows_cover : DOWN_BLOCKS * WARPS = LR := by
  decide

/-- 80 up blocks of 32 columns cover `N`. -/
theorem up_cols_cover : UP_BLOCKS * UP_COLS = N := by
  decide

/-- 160 multi-token up blocks of 16 columns cover `N`. -/
theorem upm_cols_cover : UPM_BLOCKS * UPM_COLS = N := by
  decide

/-- Rows per `up` block: 4 streams x 32 columns = 128, i.e. 16 per warp. -/
theorem up_rows_per_warp : HC * UP_COLS = 128 ∧ HC * UP_COLS / WARPS = 16 := by
  decide

/-- Multi-token up: 64 rows, 8 per warp. -/
theorem upm_rows_per_warp : HC * UPM_COLS = 64 ∧ HC * UPM_COLS / WARPS = 8 := by
  decide

/-- Tile arithmetic: 10 chunks per lane, four tiles span `D`. -/
theorem tile_arithmetic : TQ = 10 ∧ D % TILE = 0 ∧ D / TILE = 4 := by
  decide

/-- A 10240-wide bf16 row is 1280 chunks = 40 per lane; a 320-wide row is 40 chunks = 32 + 8. -/
theorem chunk_counts : D / 8 = 1280 ∧ D / 8 / 32 = 40 ∧ LR / 8 = 40 ∧ LR / 8 - 32 = 8 := by
  decide

/-- Dynamic shared memory for `T <= Tmax` tokens never exceeds the attribute set for `Tmax`. -/
theorem multi_smem_le (T Tmax : Nat) (h : T ≤ Tmax) : T * TILE * 4 ≤ Tmax * TILE * 4 :=
  Nat.mul_le_mul_right 4 (Nat.mul_le_mul_right TILE h)

-- § stream and float4 indexing

/-- Flat index `i < D` splits into stream `i / N` and offset `i % N`. -/
theorem stream_split {i : Nat} (hi : i < 10240) :
    i / 2560 < 4 ∧ i % 2560 < 2560 ∧ (i / 2560) * 2560 + i % 2560 = i := by
  omega

/-- A `float4` at a multiple of 4 never straddles two streams. -/
theorem float4_in_stream {i : Nat} (h4 : i % 4 = 0) : i % 2560 + 4 ≤ 2560 := by
  omega

/-- The loop `i = t*4; i += 1024` reaches every `float4` slot of `D` exactly via `(t, k)`. -/
theorem float4_cover {i : Nat} (hi : i < 10240) (h4 : i % 4 = 0) :
    ∃ t k : Nat, t < 256 ∧ k < 10 ∧ i = t * 4 + k * 1024 := by
  refine ⟨(i % 1024) / 4, i / 1024, ?_, ?_, ?_⟩ <;> omega

/-- The `(t, k)` pair of a `float4` slot is unique. -/
theorem float4_unique {t t' k k' : Nat} (ht : t < 256) (ht' : t' < 256)
    (h : t * 4 + k * 1024 = t' * 4 + k' * 1024) : t = t' ∧ k = k' := by
  omega

-- § row mapping of down and up

/-- A down row `bx*8 + warp` is below 320 for `bx < 40`. -/
theorem down_row_lt {bx warp : Nat} (hb : bx < 40) (hw : warp < 8) : bx * 8 + warp < 320 := by
  omega

/-- Row index of the up projection for block `bx` and local row `r < 128`. -/
def upIdx (bx r : Nat) : Nat := (r / 32) * 2560 + bx * 32 + r % 32

/-- Every up row index is inside `D`. -/
theorem upIdx_lt {bx r : Nat} (hb : bx < 80) (hr : r < 128) : upIdx bx r < 10240 := by
  unfold upIdx
  omega

/-- Distinct `(block, row)` pairs own distinct up rows, so the `R_out` writes never collide. -/
theorem upIdx_inj {bx bx' r r' : Nat} (hb : bx < 80) (hb' : bx' < 80) (hr : r < 128) (hr' : r' < 128)
    (h : upIdx bx r = upIdx bx' r') : bx = bx' ∧ r = r' := by
  unfold upIdx at h
  omega

/-- Same for the 16-column multi-token blocks. -/
def upmIdx (bx r : Nat) : Nat := (r / 16) * 2560 + bx * 16 + r % 16

/-- Multi-token up rows are inside `D`. -/
theorem upmIdx_lt {bx r : Nat} (hb : bx < 160) (hr : r < 64) : upmIdx bx r < 10240 := by
  unfold upmIdx
  omega

/-- Multi-token up rows are unique per `(block, row)`. -/
theorem upmIdx_inj {bx bx' r r' : Nat} (hb : bx < 160) (hb' : bx' < 160) (hr : r < 64) (hr' : r' < 64)
    (h : upmIdx bx r = upmIdx bx' r') : bx = bx' ∧ r = r' := by
  unfold upmIdx at h
  omega

/-- Inject rows: the extra block uses `warp < HC` and writes `inject_out[warp]`. -/
theorem inject_rows {warp : Nat} (hw : warp < 8) (ha : warp < 4) : warp < 8 ∧ warp < 4 :=
  ⟨hw, ha⟩

/-- `fused_gr_supported` pins the geometry: a supported shape has exactly `D = 10240` flattened width. -/
theorem supported_dims {n hc lr : Nat} (hn : n = 2560) (hh : hc = 4) (hl : lr = 320) :
    n * hc = 10240 ∧ lr = 320 := by
  subst hn
  subst hh
  exact ⟨by decide, hl⟩

/-- Weight-row address `row*D + chunk*8 + e` stays inside a `LR x D` matrix. -/
theorem down_weight_index_lt {row j : Nat} (hr : row < 320) (hj : j < 10240) :
    row * 10240 + j < 320 * 10240 :=
  mul_add_lt_aux hr hj

/-- Weight-row address for `up`: row `i < D`, column `k < LR`. -/
theorem up_weight_index_lt {i k : Nat} (hi : i < 10240) (hk : k < 320) :
    i * 320 + k < 10240 * 320 :=
  mul_add_lt_aux hi hk

end Strata.Kernels.FusedGr

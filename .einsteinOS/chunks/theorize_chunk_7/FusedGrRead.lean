/-!
# MODULE: fused_gr_read
Launch geometry of the fused hyper-connection read (`gr_down`, `gr_up`, and the multi-token variants).

## Source semantics
n_embd 2560, hc 4 streams, D = n_embd*hc = 10240, hc_lr 320. Down: 40 blocks x 8 warps = 320 output rows plus one inject block. Up: each block owns UP_COLS columns of every stream. Stream of flat index i is i / N.

## Form template identifiers
N -> n_embd, HC -> streams, LR -> low_rank, DOWN_BLOCKS -> down_grid, UP_BLOCKS -> up_grid
-/
namespace Strata.FusedGr

def N : Nat := 2560
def HC : Nat := 4
def D : Nat := 10240
def LR : Nat := 320
def WARPS : Nat := 8
def DOWN_BLOCKS : Nat := 40
def UP_COLS : Nat := 32
def UP_BLOCKS : Nat := 80
def UPM_COLS : Nat := 16
def UPM_BLOCKS : Nat := 160
def TILE : Nat := 2560
def TQ : Nat := 10

-- § constants

/-- D is the stream-concatenated width. -/
theorem d_eq : D = N * HC := by decide

/-- Down blocks of 8 warps cover the 320 low-rank rows. -/
theorem down_rows : DOWN_BLOCKS * WARPS = LR := by decide

/-- Up blocks cover all columns (single token). -/
theorem up_cols : UP_BLOCKS * UP_COLS = N := by decide

/-- Up blocks cover all columns (multi token). -/
theorem upm_cols : UPM_BLOCKS * UPM_COLS = N := by decide

/-- D splits into whole tiles. -/
theorem tiles_exact : D / TILE * TILE = D := by decide

/-- A tile is TQ chunks of 8 per lane across 32 lanes. -/
theorem tile_chunks : TQ * 32 * 8 = TILE := by decide

/-- 40 chunks per lane over a 10240-wide row, as the comment states. -/
theorem down_chunks : D / 8 = 32 * 40 := by decide

/-- A 320-wide up row is 40 chunks: 32 in the first pass and 8 in the second. -/
theorem up_chunks : LR / 8 = 32 + 8 := by decide

-- § stream indexing

/-- A flat index lies in a valid stream. -/
theorem stream_lt (i : Nat) (h : i < D) : i / N < HC := by (try simp only [D, N, HC] at *) <;> omega

/-- Stream and column recompose the flat index. -/
theorem stream_decomp (i : Nat) : (i / N) * N + i % N = i := by (try simp only [N] at *) <;> omega

/-- The up row of (stream c, column d0+dd) stays inside D. -/
theorem up_row_lt (b c dd : Nat) (hb : b < UP_BLOCKS) (hc : c < HC) (hd : dd < UP_COLS) : c * N + (b * UP_COLS + dd) < D := by (try simp only [UP_BLOCKS, UP_COLS, HC, N, D] at *) <;> omega

/-- The multi-token up row stays inside D. -/
theorem upm_row_lt (b c dd : Nat) (hb : b < UPM_BLOCKS) (hc : c < HC) (hd : dd < UPM_COLS) : c * N + (b * UPM_COLS + dd) < D := by (try simp only [UPM_BLOCKS, UPM_COLS, HC, N, D] at *) <;> omega

/-- Row reduction: the mean over streams divides the sum by HC. -/
def meanOver (s : Int) : Int := s / 4

/-- The first lanes' second pass is only lanes below 8. -/
theorem second_pass_lanes (lane : Nat) (h : lane < LR / 8 - 32) : lane < 8 := by (try simp only [LR] at *) <;> omega

end Strata.FusedGr

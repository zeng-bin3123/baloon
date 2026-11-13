/-!
# MODULE: fused_gated_residual
SOURCES: src/kernels/cuda/fused_gr.cu
Two-kernel fused `gr_read` at the fixed geometry n_embd = 2560, hc = 4, hc_lr = 320, which also
folds in the PREVIOUS layer's `gr_write` (`apply`), and its T-token variant that reads each
weight row once for all tokens.
-/

namespace Ref.FusedGr

-- § geometry

/-- n_embd. -/
def N : Nat := 2560
/-- Residual streams. -/
def HC : Nat := 4
/-- `hc * n_embd`. -/
def D : Nat := N * HC
/-- Low-rank width. -/
def LR : Nat := 320
/-- Warps per 256-thread block. -/
def WARPS : Nat := 8
/-- Columns per up block. -/
def UP_COLS : Nat := 32

/-- The launch constants are consistent with the geometry. -/
theorem geometry : D = 10240 ∧ LR / WARPS = 40 ∧ N / UP_COLS = 80 ∧ D / 8 / 32 = 40 ∧
    LR / 8 = 40 ∧ HC * UP_COLS = 128 := by decide

/-- `fused_gr_supported`. -/
def supported (n hc lr : Nat) : Bool := n == N && hc == HC && lr == LR

/-- Support holds exactly at the fixed geometry. -/
theorem supported_iff (n hc lr : Nat) : supported n hc lr = true ↔ n = 2560 ∧ hc = 4 ∧ lr = 320 := by
  simp [supported, N, HC, LR, and_assoc]

-- § down / inject blocks

/-- Down row handled by warp `w` of block `b` (`b < 40`). -/
def downRow (b w : Nat) : Nat := b * WARPS + w

/-- The 40 down blocks cover all 320 low-rank rows exactly once. -/
theorem downRow_cover {k : Nat} (hk : k < LR) : ∃ b w, b < 40 ∧ w < WARPS ∧ downRow b w = k := by
  refine ⟨k / 8, k % 8, ?_, ?_, ?_⟩ <;> simp [downRow, WARPS, LR] at * <;> omega

/-- The inject block (index 40) uses only warps `< HC`: four injection logits. -/
theorem inject_rows : (List.range WARPS).filter (· < HC) = [0, 1, 2, 3] := by decide

-- § up blocks

/-- Flat row `i = c*N + d0 + dd` handled by block `b` (with `d0 = b*32`) for stream `c`. -/
def upRow (b c dd : Nat) : Nat := c * N + b * UP_COLS + dd

/-- Every residual element `i < D` is owned by exactly one up block / stream / column. -/
theorem upRow_cover {i : Nat} (hi : i < D) :
    ∃ b c dd, b < 80 ∧ c < HC ∧ dd < UP_COLS ∧ upRow b c dd = i := by
  refine ⟨i % 2560 / 32, i / 2560, i % 2560 % 32, ?_, ?_, ?_, ?_⟩ <;>
    simp [upRow, D, N, HC, UP_COLS] at * <;> omega

/-- Ownership is unique, so `R_out` has a single writer per element. -/
theorem upRow_unique {b c dd b' c' dd' : Nat} (hb : b < 80) (hb' : b' < 80) (hd : dd < UP_COLS)
    (hd' : dd' < UP_COLS) (h : upRow b c dd = upRow b' c' dd') : b = b' ∧ c = c' ∧ dd = dd' := by
  simp [upRow, N, UP_COLS] at *
  omega

/-- Chunk `j` (of 40) of a `w_up` row is loaded by lane `j` if `j < 32`, else by lane `j-32`. -/
def chunkOwner (j : Nat) : Nat := if j < 32 then j else j - 32

/-- Each of the 40 chunks has an owning lane below 32, and second-pass owners are lanes `< 8`. -/
theorem chunkOwner_ok : ∀ j < 40, chunkOwner j < 32 ∧ (32 ≤ j → chunkOwner j < 8) := by decide

-- § fused previous write

/-- `R + bo * 2*sigmoid(inj/HC)` with the sigmoid abstracted as `sg`. -/
def applyPrev (sg : Int → Int) (apply : Bool) (R bo inj : Int) : Int :=
  if apply then R + bo * (2 * sg inj) else R

/-- Without `apply` the residual is read as is. -/
theorem applyPrev_off (sg : Int → Int) (R bo inj : Int) : applyPrev sg false R bo inj = R := rfl

/-- With `apply` the fused read sees exactly the previous `gr_write` result. -/
theorem applyPrev_on (sg : Int → Int) (R bo inj : Int) :
    applyPrev sg true R bo inj = R + bo * (2 * sg inj) := rfl

/-- `apply` reads `inj_prev` while the inject block writes `inject_out`; the host rejects
    aliasing them, so the read is not raced. -/
def argsOk (apply : Bool) (injPrev injectOut : Nat) : Bool := !(apply && injPrev == injectOut)

/-- With `apply`, a passing call has distinct injection buffers. -/
theorem argsOk_distinct {p o : Nat} (h : argsOk true p o = true) : p ≠ o := by
  simp [argsOk] at h; exact h

-- § mixing

/-- Numerator of `mixed[d] = (g0 + g1 + g2 + g3) / HC`. -/
def mixNum (g0 g1 g2 g3 : Int) : Int := g0 + g1 + g2 + g3

/-- Equal streams give numerator `HC * x`, so dividing by `HC` returns `x`. -/
theorem mix_const (x : Int) : mixNum x x x x = 4 * x := by
  unfold mixNum; omega

/-- Mixing is symmetric in the streams. -/
theorem mix_symm (a b c d : Int) : mixNum a b c d = mixNum d c b a := by
  unfold mixNum; omega

-- § multi-token variant

/-- Token count accepted by `fused_gr_read_multi`. -/
def nTokOk (maxT n : Nat) : Bool := 1 ≤ n && n ≤ maxT

/-- Lane `k` writes token `k`: needs `n_tok ≤ 32` lanes, which holds whenever `maxT ≤ 32`. -/
theorem lane_per_token {maxT n k : Nat} (hm : maxT ≤ 32) (h : nTokOk maxT n = true) (hk : k < n) :
    k < 32 := by
  simp [nTokOk] at h; omega

/-- Down-multi tiles: `D / TILE = 4` tiles of 2560 floats, 10 uint4 chunks per lane per tile. -/
theorem tiles : D / 2560 = 4 ∧ 2560 / 8 / 32 = 10 ∧ N / 16 = 160 := by decide

end Ref.FusedGr


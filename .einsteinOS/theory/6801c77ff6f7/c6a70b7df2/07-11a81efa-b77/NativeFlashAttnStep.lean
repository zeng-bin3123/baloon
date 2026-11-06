/-!
# MODULE: native_flash_attn_step
SOURCES: src/kernels/cuda/native_flash_attn.cu
Short-context (<= 256 cells) vector flash attention specialised to Q24x256 / KV2x256. The kernel
validates the device step and refuses (NaN output + status) on an inconsistent step.

## Source semantics
* valid = width >= 1, width <= max_context, nkv == width, pos == width - 1, n_bid == width / 4.
* head h reads kv head h / 12; cell tiles of 128 over padded length 256.
* The K/V row of cell c, kv head kv is (c * 2 + kv) * 256.

## Form template identifiers
StepV <- the four step entries, stepValid <- `valid` expression, mkStep <- qsa_step_fill (top_k + 3 bound).
-/

namespace Strata.NativeFlash

-- § step validity

/-- The four step entries read by the kernel. -/
structure StepV where
  pos : Nat
  nKv : Nat
  nBid : Nat
  width : Nat

/-- The kernel's `valid` predicate. -/
def stepValid (maxCtx : Nat) (s : StepV) : Prop :=
  1 ≤ s.width ∧ s.width ≤ maxCtx ∧ s.nKv = s.width ∧ s.pos = s.width - 1 ∧ s.nBid = s.width / 4

/-- A step as `qsa_step_fill` builds it (r = 4). -/
def mkStep (pos topK : Nat) : StepV :=
  { pos := pos, nKv := pos + 1, nBid := (pos + 1) / 4, width := min (pos + 1) (topK + 3) }

/-- Filled steps within the short-context window are accepted. -/
theorem mkStep_valid {maxCtx pos topK : Nat} (hmax : maxCtx ≤ 256) (hctx : pos + 1 ≤ maxCtx)
    (htop : 256 ≤ topK) : stepValid maxCtx (mkStep pos topK) := by
  simp only [stepValid, mkStep]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> omega

/-- A step whose width exceeds the supported context is rejected. -/
theorem step_rejected_too_long {maxCtx : Nat} {s : StepV} (h : maxCtx < s.width) :
    ¬ stepValid maxCtx s := fun hv => by
  have := hv.2.1
  omega

/-- A stale step (pos not width - 1) is rejected: the frozen-argument failure mode. -/
theorem step_rejected_stale {maxCtx : Nat} {s : StepV} (h : s.pos ≠ s.width - 1) :
    ¬ stepValid maxCtx s := fun hv => h hv.2.2.2.1

-- § tiling and addressing

/-- Query head to kv head. -/
theorem head_kv_lt : ∀ h : Nat, h < 24 → h / 12 < 2 := by
  intro h hh; omega

/-- Every cell `c < 256` is visited by some (warp, lane, row) of some 128-wide tile. -/
theorem cell_covered {c : Nat} (hc : c < 256) :
    ∃ warp lane row : Nat, warp < 4 ∧ lane < 32 ∧ row < 8 ∧
      c = (c / 128) * 128 + warp * 32 + (lane / 8) * 8 + row :=
  ⟨(c % 128) / 32, ((c % 32) / 8) * 8, c % 8, by omega, by omega, by omega, by omega⟩

/-- Two tiles of 128 cover the padded length 256. -/
theorem tiles_cover : (256 + 127) / 128 = 2 := by decide

/-- K/V row index of (cell, kv head) with two kv heads. -/
def kvRow (cell kv : Nat) : Nat := cell * 2 + kv

/-- Rows are distinct per (cell, kv). -/
theorem kvRow_inj {c1 k1 c2 k2 : Nat} (h1 : k1 < 2) (h2 : k2 < 2)
    (h : kvRow c1 k1 = kvRow c2 k2) : c1 = c2 ∧ k1 = k2 := by
  unfold kvRow at h; omega

/-- Per-cell byte size is 2 heads * 256 dims * 2 bytes. -/
theorem kv_cell_bytes : 2 * 256 * 2 = 1024 := by decide

/-- The shared tile is 4 * 4 * 256 floats = 16 KiB. -/
theorem tile_bytes : 4 * 4 * 256 * 4 = 16384 := by decide

end Strata.NativeFlash

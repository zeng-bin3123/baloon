-- Baloon.WeightLoader: weight-loader segment lists, fp16/plain scale conversions,
-- widen/narrow tiling and chunk-loop progress invariants.
-- Independent implementation proving the same mathematical facts as Ref.WeightLoader.

namespace Baloon.WeightLoader

def CHUNK : Nat := 8388608

def srcScaleBytes (scalesFp16 : Bool) (scales : Nat) : Nat :=
  if scalesFp16 then scales / 2 else scales

def sumSrc (codes scales offset : Nat) (fp16 : Bool) : Nat :=
  codes + srcScaleBytes fp16 scales + offset

def sumDst (codes scales offset : Nat) : Nat := codes + scales + offset

theorem sumDst_eq (c s o : Nat) : sumDst c s o = c + s + o := rfl

theorem sumSrc_plain (c s o : Nat) : sumSrc c s o false = sumDst c s o := by
  simp [sumSrc, sumDst, srcScaleBytes]

theorem sumSrc_fp16 (c s o : Nat) (he : s % 2 = 0) : sumSrc c s o true + s / 2 = sumDst c s o := by
  simp only [sumSrc, sumDst, srcScaleBytes, if_true]; omega

theorem fp16_half_double (s : Nat) (he : s % 2 = 0) : s / 2 * 2 = s := by omega

theorem widen_out_bytes (n : Nat) (he : n % 2 = 0) : 4 * (n / 2) = 2 * n := by omega
theorem widen_tiles (done n : Nat) (he : n % 2 = 0) : done * 2 + 4 * (n / 2) = (done + n) * 2 := by omega
theorem widen_fits_stage (n : Nat) (h : n ≤ CHUNK) : 4 * (n / 2) ≤ 2 * CHUNK := by
  unfold CHUNK at *; omega
theorem narrow_tiles (done n : Nat) (hd : done % 4 = 0) (hn : n % 4 = 0) :
    done / 2 + 2 * (n / 4) = (done + n) / 2 := by omega
theorem promoted_saves (elems src dst : Nat) (h1 : src = elems * 4) (h2 : dst = elems * 2) :
    src - dst = elems * 2 := by omega

def chunkLen (rem : Nat) : Nat := min CHUNK rem

theorem chunk_progress (rem : Nat) (h : 0 < rem) : 0 < chunkLen rem ∧ rem - chunkLen rem < rem := by
  unfold chunkLen CHUNK; omega

theorem chunk_le (rem : Nat) : chunkLen rem ≤ CHUNK := by unfold chunkLen; omega

-- ── conversion rules and arena sufficiency ───────────────────────────────────

/-- Conversion applied to an unquantized row. -/
inductive Conv
  | copy | bf16High | toF16

/-- Selection rule: raw 16-bit rows are copied, promoted rows convert, others copy. -/
def convFor (raw16 : Bool) (kind : Nat) : Conv :=
  if raw16 then Conv.copy else if kind = 1 then Conv.bf16High else if kind = 3 then Conv.toF16 else Conv.copy

/-- Raw 16-bit rows are never converted. -/
theorem convFor_raw (k : Nat) : convFor true k = Conv.copy := by simp [convFor]

/-- A promoted BF16 row takes the high half. -/
theorem convFor_bf16 : convFor false 1 = Conv.bf16High := by simp [convFor]

/-- A promoted F16 row converts with round-to-nearest-even. -/
theorem convFor_f16 : convFor false 3 = Conv.toF16 := by simp [convFor]

/-- The arena check rejects exactly when the arena is smaller than the pool. -/
def arenaOk (arena pool : Nat) : Bool := decide (pool ≤ arena)

theorem arenaOk_sufficient (arena pool : Nat) (h : arenaOk arena pool = true) : pool ≤ arena := by
  unfold arenaOk at h; exact of_decide_eq_true h

theorem arenaOk_reject (arena pool : Nat) (h : arena < pool) : arenaOk arena pool = false := by
  unfold arenaOk; exact decide_eq_false (by omega)

end Baloon.WeightLoader

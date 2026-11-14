/-!
# NativePlePostops — native PLE post-projection geometry
Formalises the 512-thread gate reduction lanes, the row-fastest history
address used by the 4-tap conv, the initialise-then-add tap accumulation,
the residual ordering, and the two aliasing exceptions of native_ple_postops.

Sources: src/kernels/cuda/native_ple_postops.cu,
         src/kernels/gpu/ple_postops_native.cu
-/

namespace Baloon.PlePostops

-- ── geometry constants ────────────────────────────────────────────────────────

/-- Embedding width `N = 2560`. -/
def kN : Nat := 2560
/-- Number of streams `H = 4`. -/
def kH : Nat := 4
/-- `D = N × H = 10240`. -/
def kD : Nat := kN * kH
/-- History depth per channel. -/
def kHist : Nat := 9

-- ── gate reduction lanes ──────────────────────────────────────────────────────

/-- Column that thread `tid` processes in partial slot `j`. -/
def gateCol (tid j : Nat) : Nat := tid + j * 512

/-- Every column `d < N` is covered by thread `d % 512` in slot `d / 512`. -/
theorem gateCol_cover {d : Nat} (hd : d < kN) :
    ∃ tid j, tid < 512 ∧ j < 8 ∧ gateCol tid j = d := by
  refine ⟨d % 512, d / 512, ?_, ?_, ?_⟩
  all_goals (simp only [gateCol]; unfold kN at hd; omega)

/-- A column is covered by exactly one (thread, slot) pair. -/
theorem gateCol_unique {t j t' j' : Nat} (ht : t < 512) (ht' : t' < 512)
    (h : gateCol t j = gateCol t' j') : t = t' ∧ j = j' := by
  unfold gateCol at h; omega

/-- `N / 512 = 5`: only the first 5 slots carry live data. -/
theorem gateActive_slots : kN / 512 = 5 ∧ kN % 512 = 0 := by decide

/-- `512 / 32 = 16` warp-partials fit in the per-block shared array. -/
theorem gateWarp_count : 512 / 32 = 16 := by decide

-- ── conv history addressing ───────────────────────────────────────────────────

/-- Linear address of history row `r` of channel `c` (row-fastest layout). -/
def convHistAddr (c r : Nat) : Nat := c * kHist + r

/-- The address for tap `k < 3` is `c * kHist + 3 * k`. -/
def convTapAddr (c k : Nat) : Nat := c * kHist + 3 * k

/-- `convTapAddr` agrees with the legacy formula `(nhist - (3-k)*3) + nhist*c`. -/
theorem convTapAddr_legacy : ∀ c k, k < 3 →
    convTapAddr c k = (9 - (4 - 1 - k) * 3) + 9 * c := by
  intro c k hk; unfold convTapAddr kHist
  have : 9 - (4 - 1 - k) * 3 = 3 * k := by
    rcases (by omega : k = 0 ∨ k = 1 ∨ k = 2) with h | h | h <;> subst h <;> rfl
  rw [this]; omega

/-- All history addresses stay inside the `kHist × kD` buffer. -/
theorem convTapAddr_inBuf {c k : Nat} (hc : c < kD) (hk : k < 3) :
    convTapAddr c k < kHist * kD := by
  unfold convTapAddr kHist kD kN kH at *; omega

-- ── tap accumulation ─────────────────────────────────────────────────────────

/-- Accumulate 4 taps as written in the kernel: initialise with tap 0, add taps 1–3. -/
def tapAccum (t0 t1 t2 t3 : Int) : Int := ((t0 + t1) + t2) + t3

/-- `tapAccum` equals the plain sum. -/
theorem tapAccum_sum (t0 t1 t2 t3 : Int) : tapAccum t0 t1 t2 t3 = t0 + t1 + t2 + t3 := rfl

-- ── residual ordering ────────────────────────────────────────────────────────

/-- Residual: `hidden + (gated + conv)` as written in `conv_residual_kernel`. -/
def residualAdd (hidden gated conv : Int) : Int := hidden + (gated + conv)

/-- Equivalent to the flat sum `hidden + gated + conv`. -/
theorem residualAdd_flat (h g c : Int) : residualAdd h g c = h + g + c := by
  unfold residualAdd; omega

/-- Conv grid covers `kD` exactly: `kD / 256 * 256 = kD` and `kD / 256 = 40`. -/
theorem convGrid_exact : kD / 256 * 256 = kD ∧ kD / 256 = 40 := by decide

-- ── aliasing policy ───────────────────────────────────────────────────────────

/-- Output/input alias allowed: only `result` (index 6) may alias `hidden` (index 1). -/
def ioAliasOk (outIdx inpIdx : Nat) : Bool := outIdx == 6 && inpIdx == 1

/-- Output/output alias allowed: only `query` (1) may alias `normalized` (4). -/
def ooAliasOk (i j : Nat) : Bool := i == 1 && j == 4

/-- Exactly one output/input and one output/output pair are exempt. -/
theorem alias_exemptCount :
    ((List.range 7).flatMap (fun i => (List.range 8).filter (ioAliasOk i))).length = 1 ∧
    ((List.range 7).flatMap
      (fun i => (List.range 7).filter (fun j => i < j && ooAliasOk i j))).length = 1 := by
  decide

/-- There are 21 unordered output pairs checked by the validator. -/
theorem outPair_count :
    ((List.range 7).flatMap (fun i => (List.range 7).filter (i < ·))).length = 21 := by
  decide

end Baloon.PlePostops

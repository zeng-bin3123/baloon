/-!
# MODULE: native_ple_postops
SOURCES: src/kernels/cuda/native_ple_postops.cu
The llama.cpp-pinned PLE post-projection operations: weighted RMS norms of key and hidden,
the per-stream gate from a 512-thread SUM_ROWS reduction with eight partial slots, the value
broadcast, the norm of the gated value, and a fused 4-tap conv + SiLU + residual add, together
with the span validation (inputs/outputs disjoint except two documented aliases).

## Source semantics
* Geometry `N = 2560, H = 4, D = 10240, HISTORY = 9`.
* Gate: thread `tid < 512` accumulates `d = tid + j*512`, `j < 8`, for `d < N` (products
  rounded before adding), partials meet over 16 warps; `s = scale*sum + 0`,
  `gate = sigmoid(sgn(s) * sqrt(max(|s|, 1e-6)))`.
* Conv: tap `k < 3` reads `history[c*9 + 3k]`, tap 3 reads `normalized[c]`; the sum is formed
  `k == 0 ? term : sum + term`; `conv = silu(sum)`, `result = hidden + (gated + conv)`.
* Allowed aliases: `result == hidden` (each thread owns its element) and
  `query == normalized` (query is dead once the gate has consumed it). All other writable
  spans are pairwise disjoint and disjoint from all inputs/weights.

## Form template identifiers
gate lane slots ↦ `gateSlot`, conv history address ↦ `convAddr`, tap accumulation ↦ `tapSum`,
alias exceptions ↦ `aliasAllowed`.
-/

namespace Strata.NativePlePostops

/-- n_embd. -/
def N : Nat := 2560
/-- Streams. -/
def H : Nat := 4
/-- `N * H`. -/
def D : Nat := N * H
/-- History rows. -/
def HISTORY : Nat := 9

-- § gate reduction

/-- Column handled by thread `tid` in slot `j`. -/
def gateSlot (tid j : Nat) : Nat := tid + j * 512

/-- Every column `d < N` is accumulated by thread `d % 512` in slot `d / 512 < 8`. -/
theorem gateSlot_cover {d : Nat} (hd : d < N) :
    ∃ tid j, tid < 512 ∧ j < 8 ∧ gateSlot tid j = d := by
  refine ⟨d % 512, d / 512, ?_, ?_, ?_⟩ <;> (try simp only [gateSlot]) <;> (try unfold N at hd) <;> omega

/-- No column is accumulated twice. -/
theorem gateSlot_unique {t j t' j' : Nat} (ht : t < 512) (ht' : t' < 512)
    (h : gateSlot t j = gateSlot t' j') : t = t' ∧ j = j' := by
  unfold gateSlot at h
  omega

/-- Only the first five slots ever see data (`2560 = 5 * 512`); the rest add zeros. -/
theorem active_slots : N / 512 = 5 ∧ N % 512 = 0 := by decide

/-- 512 threads are 16 warps: the second reduction stage reads exactly lanes `< 16`. -/
theorem gate_warps : 512 / 32 = 16 := by decide

-- § conv + residual

/-- Address of tap `k < 3` of channel `c` in the row-fastest 9-row history. -/
def convAddr (c k : Nat) : Nat := c * HISTORY + 3 * k

/-- These are exactly the rows ple.cu's `conv_kernel` reads (`nhist - (3-k)*3` for k < 3) at
    the same row-fastest address `row + nhist*c`: the native and legacy paths agree. -/
theorem convAddr_matches_legacy : ∀ c k, k < 3 → convAddr c k = (9 - (4 - 1 - k) * 3) + 9 * c := by
  intro c k hk
  unfold convAddr HISTORY
  have : 9 - (4 - 1 - k) * 3 = 3 * k := by
    rcases (by omega : k = 0 ∨ k = 1 ∨ k = 2) with h | h | h <;> subst h <;> rfl
  rw [this]; omega

/-- History taps stay inside the `HISTORY * D` buffer. -/
theorem convAddr_lt {c k : Nat} (hc : c < D) (hk : k < 3) : convAddr c k < HISTORY * D := by
  unfold convAddr HISTORY D N H at *
  omega

/-- Tap accumulation as written: the first term initialises, later terms are added. -/
def tapSum (t0 t1 t2 t3 : Int) : Int := ((t0 + t1) + t2) + t3

/-- In exact arithmetic it is the plain sum of the four terms. -/
theorem tapSum_eq (t0 t1 t2 t3 : Int) : tapSum t0 t1 t2 t3 = t0 + t1 + t2 + t3 := rfl

/-- Residual as written: `hidden + (gated + conv)`. -/
def residual (h g c : Int) : Int := h + (g + c)

/-- It equals the legacy `hidden + gated + conv`. -/
theorem residual_eq_legacy (h g c : Int) : residual h g c = h + g + c := by
  unfold residual; omega

/-- The conv grid `D/256` covers all channels exactly (no tail block). -/
theorem conv_grid : D / 256 * 256 = D ∧ D / 256 = 40 := by decide

-- § aliasing policy

/-- Output index → name: 0 key, 1 query, 2 gate, 3 gated, 4 normalized, 5 conv, 6 result;
    input index 1 is `hidden`. Allowed output/input alias: result/hidden. -/
def ioAliasAllowed (out inp : Nat) : Bool := out == 6 && inp == 1

/-- Allowed output/output alias: query/normalized. -/
def ooAliasAllowed (i j : Nat) : Bool := i == 1 && j == 4

/-- Exactly one output/input pair and one output/output pair are exempt from the check. -/
theorem alias_exemptions :
    ((List.range 7).flatMap (fun i => (List.range 8).filter (ioAliasAllowed i))).length = 1 ∧
    ((List.range 7).flatMap (fun i => ((List.range 7).filter (fun j => i < j && ooAliasAllowed i j)))).length = 1 := by
  decide

/-- Every unordered pair of the 7 outputs is checked once (`j > i`): 21 pairs. -/
theorem output_pairs : ((List.range 7).flatMap (fun i => (List.range 7).filter (i < ·))).length = 21 := by
  decide

end Strata.NativePlePostops

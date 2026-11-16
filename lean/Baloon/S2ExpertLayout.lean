-- Baloon.S2ExpertLayout: S2 MoE expert blob geometry for the Q2_0 format:
-- region offsets, row bounds, lane/chunk coverage.
-- Independent implementation proving the same mathematical facts as Ref.S2ExpertLayout.

namespace Baloon.S2ExpertLayout

def H : Nat := 2560
def FF : Nat := 640
def QK : Nat := 64
def rowGU : Nat := H / 4
def rowD : Nat := FF / 4
def scGU : Nat := H / QK
def scD : Nat := FF / QK
def oDCodes : Nat := 2 * FF * rowGU
def oGUScales : Nat := oDCodes + H * rowD
def oDScales : Nat := oGUScales + 2 * FF * scGU * 2
def blobBytes : Nat := oDScales + H * scD * 2

theorem blob_constants :
    rowGU = 640 ∧ rowD = 160 ∧ scGU = 40 ∧ scD = 10 ∧ oDCodes = 819200 ∧ oGUScales = 1228800 ∧
      oDScales = 1331200 ∧ blobBytes = 1382400 := by decide

theorem gu_code_row_in_region (i : Nat) (h : i < 1280) : i * 640 + 640 ≤ 819200 := by omega
theorem gu_scale_row_in_region (i : Nat) (h : i < 1280) :
    1228800 ≤ 1228800 + i * 80 ∧ 1228800 + i * 80 + 80 ≤ 1331200 := by omega
theorem down_code_row_in_region (r : Nat) (h : r < 2560) :
    819200 + r * 160 + 160 ≤ 1228800 := by omega
theorem down_scale_row_in_region (r : Nat) (h : r < 2560) :
    1331200 + r * 20 + 20 ≤ 1382400 := by omega
theorem regions_ordered : oDCodes ≤ oGUScales ∧ oGUScales ≤ oDScales ∧ oDScales ≤ blobBytes := by decide

theorem gu_chunks_per_lane : (80 + 31) / 32 = 3 := by decide
theorem gu_chunk_decomp (c : Nat) (h : c < 80) :
    c = c % 32 + 32 * (c / 32) ∧ c % 32 < 32 ∧ c / 32 < 3 := by omega
theorem down_chunks_fit : 640 / 32 = 20 ∧ 20 ≤ 32 := by decide
theorem down_scale_index (c : Nat) (h : c < 20) : c / 2 < 10 := by omega
theorem slot_decomp (slot : Nat) : slot = (slot / 1280) * 1280 + slot % 1280 ∧ slot % 1280 < 1280 := by
  omega

end Baloon.S2ExpertLayout

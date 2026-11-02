/-!
# MODULE: s2_expert_grouped_gemv
Grouped GPU routed-expert FFN over Q2_0 blobs: gate/up rows (interleaved in the blob, written gate-major),
SwiGLU, requantise to Q8_0, down rows; dp4a with bias term `sum c*x - sum x`.

## Source semantics
(c-1)*x summed equals sum c*x minus sum x. 2-bit codes are LSB-first: element 4j+k is bits [2k,2k+2) of byte j.
Blob layout: [gate/up codes | down codes | gate/up scales | down scales]. Row slot i: even = gate row i/2, odd = up row.

## Form template identifiers
H=2560 -> n_embd; FF=640 -> n_ff_exp; QK=64 -> weight_group
-/
namespace Strata.S2Expert

/-- Integer sum of a list. -/
def isum : List Int → Int
  | [] => 0
  | x :: xs => x + isum xs

/-- Dot of code with activation, with the bias removed per element. -/
def dotBias : List (Int × Int) → Int
  | [] => 0
  | (c, x) :: r => (c - 1) * x + dotBias r

/-- Plain code-activation dot. -/
def dotRaw : List (Int × Int) → Int
  | [] => 0
  | (c, x) :: r => c * x + dotRaw r

/-- Sum of activations. -/
def sumX : List (Int × Int) → Int
  | [] => 0
  | (_, x) :: r => x + sumX r

/-- The dp4a decomposition: sum (c-1)x = sum c x - sum x. -/
theorem bias_decomposition (l : List (Int × Int)) : dotBias l = dotRaw l - sumX l := by
  induction l with
  | nil => simp [dotBias, dotRaw, sumX]
  | cons p r ih =>
    obtain ⟨c, x⟩ := p
    simp only [dotBias, dotRaw, sumX, ih, Int.sub_mul, Int.one_mul]
    omega

/-- Byte layout of the blob (bytes). -/
def oDCodes : Nat := 2 * 640 * 640
/-- Offset of gate/up scales. -/
def oGuScales : Nat := oDCodes + 2560 * 160
/-- Offset of down scales. -/
def oDScales : Nat := oGuScales + 2 * 640 * 40 * 2
/-- Total blob size. -/
def blobBytes : Nat := oDScales + 2560 * 10 * 2

/-- Blob total is 1382400 bytes. -/
theorem blob_total : blobBytes = 1382400 := by decide

/-- Regions are ordered and non-overlapping. -/
theorem blob_ordered : oDCodes < oGuScales ∧ oGuScales < oDScales ∧ oDScales < blobBytes := by decide

/-- Code byte fields: four 2-bit fields reconstruct the byte (LSB-first). -/
theorem code_fields : ∀ c : Fin 256,
    (c.val &&& 3) + 4 * ((c.val >>> 2) &&& 3) + 16 * ((c.val >>> 4) &&& 3) + 64 * ((c.val >>> 6) &&& 3) = c.val := by
  decide +kernel

/-- Interleaved row slot to (row, is-up). -/
theorem interleave (i : Nat) : (i / 2) * 2 + i % 2 = i := by omega

/-- Gate-major layout: hit h row r (gate) is below every up index. -/
theorem gate_below_up (n h r : Nat) (hh : h < n) (hr : r < 640) :
    h * 640 + r < n * 640 + h * 640 + r := by omega

/-- Gate indices stay inside the first n*FF floats. -/
theorem gate_in_first_half (n h r : Nat) (hh : h < n) (hr : r < 640) : h * 640 + r < n * 640 := by omega

/-- Up indices stay inside the second half. -/
theorem up_in_second_half (n h r : Nat) (hh : h < n) (hr : r < 640) :
    n * 640 + h * 640 + r < 2 * (n * 640) := by omega

/-- Gate/up row loop of 80 chunks with 32 lanes needs 3 chunks per lane. -/
theorem gu_chunks : (2560 / 32 + 31) / 32 = 3 := by decide

/-- Down row has 20 chunks, fewer than a warp. -/
theorem down_chunks : 640 / 32 = 20 ∧ 20 < 32 := by decide

/-- Gate/up grid exactly covers 1280 rows with 32-row blocks. -/
theorem gu_grid : 2 * 640 / 32 * 32 = 2 * 640 := by decide

/-- Down grid exactly covers 2560 rows with 64-row blocks. -/
theorem down_grid : 2560 / 64 * 64 = 2560 := by decide

/-- One weight scale per 64 elements, i.e. per two 32-element chunks. -/
theorem scale_pairing (c : Nat) : (c / 2) * 2 ≤ c ∧ c < (c / 2) * 2 + 2 := by omega

end Strata.S2Expert

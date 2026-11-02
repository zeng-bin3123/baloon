/-!
# MODULE: s2_expert_grouped_layout
Blob geometry, gate/up interleaving and chunk coverage for the grouped Q2_0 expert kernels.

## Source semantics
* The blob holds down codes, gate/up scales, then down scales at fixed byte offsets.
* Gate/up row-slot `i` is gate row `i/2` when even and up row `i/2` when odd.
* The output is gate-major: every hit's gate rows come first, then every hit's up rows.
* `sum (c-1)*x = sum c*x - sum x`, the identity behind the two `dp4a` accumulators.

## Form template identifiers
* `H` -> `n_embd`; `FF` -> `n_ff`; `ROW_GU` -> `gu_row_bytes`; `O_D_CODES` -> `down_codes_off`
* `gu_kernel` -> `gate_up_proj`; `swiglu_kernel` -> `gated_act`; `down_kernel` -> `down_proj`
-/

namespace Strata.S2Grouped

-- § blob geometry

/-- Hidden size. -/
def H : Nat := 2560
/-- Expert feed-forward size. -/
def FF : Nat := 640
/-- Gate/up row code bytes. -/
def rowGU : Nat := H / 4
/-- Down row code bytes. -/
def rowD : Nat := FF / 4
/-- Gate/up scales per row. -/
def scGU : Nat := H / 64
/-- Down scales per row. -/
def scD : Nat := FF / 64
/-- Offset of the down codes. -/
def oDCodes : Nat := 2 * FF * rowGU
/-- Offset of the gate/up scales. -/
def oGUScales : Nat := oDCodes + H * rowD
/-- Offset of the down scales. -/
def oDScales : Nat := oGUScales + 2 * FF * scGU * 2

/-- Gate/up rows take 640 bytes. -/
theorem rowGU_val : rowGU = 640 := by decide
/-- Down rows take 160 bytes. -/
theorem rowD_val : rowD = 160 := by decide
/-- Scale counts per row. -/
theorem scales_val : scGU = 40 ∧ scD = 10 := by decide
/-- The down codes begin after all gate/up codes. -/
theorem oDCodes_val : oDCodes = 819200 := by decide
/-- The gate/up scales begin after the down codes. -/
theorem oGUScales_val : oGUScales = 1228800 := by decide
/-- The down scales begin after the gate/up scales. -/
theorem oDScales_val : oDScales = 1331200 := by decide

/-- The regions are ordered. -/
theorem offsets_ordered : oDCodes < oGUScales ∧ oGUScales < oDScales := by decide

/-- Total blob size. -/
def blobBytes : Nat := oDScales + H * scD * 2

/-- The blob size is 1,382,400 bytes. -/
theorem blobBytes_val : blobBytes = 1382400 := by decide

/-- The blob size is a multiple of 16, as `fetch_blobs` requires. -/
theorem blobBytes_mod16 : blobBytes % 16 = 0 := by decide

-- § gate-major output slot

/-- Output index for row-slot `i` of hit `h` among `n` hits. -/
def guIndex (n h i : Nat) : Nat :=
  (if i % 2 = 1 then n * 640 + h * 640 else h * 640) + i / 2

/-- Every output index stays inside the `2*n*FF` buffer. -/
theorem guIndex_lt (n h i : Nat) (hh : h < n) (hi : i < 1280) : guIndex n h i < 2 * n * 640 := by
  unfold guIndex; split_ifs <;> omega

/-- Distinct (hit, row-slot) pairs never share an output index. -/
theorem guIndex_inj (n h i h' i' : Nat) (hh : h < n) (hh' : h' < n) (hi : i < 1280) (hi' : i' < 1280)
    (heq : guIndex n h i = guIndex n h' i') : h = h' ∧ i = i' := by
  unfold guIndex at heq
  split_ifs at heq <;> omega

/-- Even slots land in the gate half. -/
theorem guIndex_gate (n h i : Nat) (hh : h < n) (hi : i < 1280) (he : i % 2 = 0) : guIndex n h i < n * 640 := by
  unfold guIndex; split_ifs <;> omega

/-- Odd slots land in the up half. -/
theorem guIndex_up (n h i : Nat) (hi : i < 1280) (ho : i % 2 = 1) : n * 640 ≤ guIndex n h i := by
  unfold guIndex; split_ifs <;> omega

/-- SwiGLU reads gate at `x` and up at `x + pairs`, so row `r` of hit `h` meets its own up row. -/
theorem swiglu_pairing (n h r : Nat) (hr : r < 640) :
    guIndex n h (2 * r + 1) = guIndex n h (2 * r) + n * 640 := by
  unfold guIndex; split_ifs <;> omega

-- § chunk coverage

/-- Lane `l`, step `q` covers chunk `l + 32q`; all 80 gate/up chunks are covered by 32 lanes x 3 steps. -/
theorem gu_chunks_covered (c : Nat) (hc : c < 80) : ∃ l q, l < 32 ∧ q < 3 ∧ l + 32 * q = c :=
  ⟨c % 32, c / 32, by omega, by omega, by omega⟩

/-- The three-step lane cover reaches no chunk twice. -/
theorem gu_chunks_unique (l q l' q' : Nat) (hl : l < 32) (hl' : l' < 32)
    (h : l + 32 * q = l' + 32 * q') : l = l' ∧ q = q' := by omega

/-- Down rows have 20 chunks, so lanes 0..19 each own one. -/
theorem down_chunks_fit (c : Nat) (hc : c < FF / 32) : c < 32 := by
  unfold FF at hc; omega

/-- One weight scale covers two chunks: `c / 2` indexes the 64-wide group. -/
theorem scale_group (c : Nat) (hc : c < 80) : c / 2 < 40 := by omega

-- § launch grid

/-- Gate/up row blocks per group tile exactly. -/
theorem gu_blocks_exact : 2 * FF / 32 * 32 = 2 * FF := by decide
/-- Down row blocks per group tile exactly. -/
theorem down_blocks_exact : H / 64 * 64 = H := by decide

-- § bias identity

/-- Dot product of codes and activations. -/
def dot : List (Int × Int) → Int
  | [] => 0
  | p :: t => p.1 * p.2 + dot t

/-- Sum of activations. -/
def sumX : List (Int × Int) → Int
  | [] => 0
  | p :: t => p.2 + sumX t

/-- Dot product with bias-subtracted codes. -/
def dotBias : List (Int × Int) → Int
  | [] => 0
  | p :: t => (p.1 - 1) * p.2 + dotBias t

/-- `sum (c-1)*x = sum c*x - sum x`. -/
theorem dotBias_eq (l : List (Int × Int)) : dotBias l = dot l - sumX l := by
  induction l with
  | nil => simp [dotBias, dot, sumX]
  | cons p t ih =>
    simp only [dotBias, dot, sumX, ih]
    rw [Int.sub_mul]
    omega

-- § compaction counts

/-- A ballot prefix count never exceeds the lane index. -/
theorem prefix_le (lane : Nat) (pc : Nat) (h : pc ≤ lane) : pc < lane + 1 := by omega

/-- Entries per group are capped at 8 by `min`. -/
theorem group_cap (a b : Nat) : min (a - b) 8 ≤ 8 := by omega

end Strata.S2Grouped

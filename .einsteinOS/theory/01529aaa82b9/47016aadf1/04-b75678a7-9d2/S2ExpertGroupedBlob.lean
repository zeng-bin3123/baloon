/-!
# MODULE: s2_expert_grouped_blob
Blob geometry, the bias-correction identity and the gate-major output layout of the S2 grouped expert kernels
in `s2_expert_grouped.cu`.

## Source semantics
* `sum_j (c_j - 1) x_j = sum_j c_j x_j - sum_j x_j`, computed as two integer dot products (`s - hx`).
* The blob is: gate/up codes (2*FF rows of 640 B), down codes (H rows of 160 B), gate/up fp16 scales
  (2*FF rows of 40), down fp16 scales (H rows of 10).
* gate/up outputs are gate-major: hit `h` row `r` gate at `h*FF + r`, up at `n_hits*FF + h*FF + r`.
* Scratch regions are rounded up to 16 bytes.

## Form template identifiers
H -> n_embd; FF -> n_ff; ROW_GU -> gate_up_row_code_bytes; ROW_D -> down_row_code_bytes;
O_D_CODES -> down_codes_offset; O_GU_SCALES -> gate_up_scales_offset; O_D_SCALES -> down_scales_offset.
-/

namespace S2ExpertGroupedBlob

-- § the bias identity

/-- Dot product of codes with activations. -/
def dot : List Int → List Int → Int
  | c :: cs, x :: xs => c * x + dot cs xs
  | _, _ => 0

/-- Dot product of codes minus one with activations. -/
def dotMinus : List Int → List Int → Int
  | c :: cs, x :: xs => (c - 1) * x + dotMinus cs xs
  | _, _ => 0

/-- Sum of the activations aligned with the codes (the `hx` term). -/
def dotOnes : List Int → List Int → Int
  | _ :: cs, x :: xs => x + dotOnes cs xs
  | _, _ => 0

/-- The weight-bias correction identity `sum (c-1)x = sum c x - sum x`. -/
theorem dotMinus_eq (cs xs : List Int) : dotMinus cs xs = dot cs xs - dotOnes cs xs := by
  induction cs generalizing xs with
  | nil => cases xs <;> simp [dotMinus, dot, dotOnes]
  | cons c cs ih =>
    cases xs with
    | nil => simp [dotMinus, dot, dotOnes]
    | cons x xs =>
      simp only [dotMinus, dot, dotOnes, ih, Int.sub_mul]
      omega

-- § blob geometry

/-- Down codes offset. -/
theorem o_d_codes : 2 * 640 * 640 = 819200 := by decide
/-- Gate/up scales offset. -/
theorem o_gu_scales : 819200 + 2560 * 160 = 1228800 := by decide
/-- Down scales offset. -/
theorem o_d_scales : 1228800 + 2 * 640 * 40 * 2 = 1331200 := by decide
/-- Blob bytes. -/
theorem blob_bytes : 1331200 + 2560 * 10 * 2 = 1382400 := by decide

/-- Row geometry: 2 bits per element and 64-element scale groups. -/
theorem row_geometry : 2560 / 4 = 640 ∧ 640 / 4 = 160 ∧ 2560 / 64 = 40 ∧ 640 / 64 = 10 := by decide

/-- A down row has 20 activation chunks, all handled by lanes 0..19 of a warp. -/
theorem down_chunks : 640 / 32 = 20 ∧ 20 < 32 := by decide

/-- A gate/up row has 80 chunks, three per lane (ceil). -/
theorem gu_chunks : 2560 / 32 = 80 ∧ (80 + 31) / 32 = 3 := by decide

/-- Chunk lane indices `lane + 32 q` stay below 96 for q < 3. -/
theorem gu_chunk_index (lane q : Nat) (hl : lane < 32) (hq : q < 3) : lane + 32 * q < 96 := by omega

/-- One fp16 scale per two 32-element chunks. -/
theorem scale_index (c : Nat) : (c / 2) * 2 ≤ c := by omega

/-- Shared-memory staging of the grouped gate/up kernel fits 48 KiB. -/
theorem smem_gu : 8 * (2560 / 4) * 4 + 8 * (2560 / 32) * 4 ≤ 49152 := by decide

-- § gate-major layout

/-- Output index of gate/up slot `i` of hit `h`. -/
def guOut (n FF h i : Nat) : Nat :=
  if i % 2 = 1 then n * FF + h * FF + i / 2 else h * FF + i / 2

/-- The row within a hit is below FF. -/
theorem gu_row_lt (FF i : Nat) (hi : i < 2 * FF) : i / 2 < FF := by omega

/-- Distinct slots of one hit map to distinct outputs (needs one hit's worth of space). -/
theorem guOut_inj (n FF h i j : Nat) (hn : FF ≤ n * FF) (hi : i < 2 * FF) (hj : j < 2 * FF)
    (heq : guOut n FF h i = guOut n FF h j) : i = j := by
  unfold guOut at heq
  by_cases a : i % 2 = 1 <;> by_cases b : j % 2 = 1 <;> simp only [a, b, if_true, if_false] at heq <;> omega

/-- Gate outputs of hit `h` stay inside the gate half. -/
theorem gate_in_gate_half (n FF h i : Nat) (hh : h < n) (hi : i < 2 * FF) (he : i % 2 = 0) :
    guOut n FF h i < n * FF := by
  unfold guOut
  have h1 : i % 2 ≠ 1 := by omega
  simp only [h1, if_false]
  have h2 : (h + 1) * FF ≤ n * FF := Nat.mul_le_mul_right _ hh
  rw [Nat.add_mul, Nat.one_mul] at h2
  omega

-- § scratch rounding

/-- Round up to a multiple of 16. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Rounding never shrinks. -/
theorem align16_ge (x : Nat) : x ≤ align16 x := by unfold align16; omega

/-- Rounded sizes are multiples of 16, hence 8-aligned for doubles and floats. -/
theorem align16_mod (x : Nat) : align16 x % 16 = 0 := by unfold align16; omega

end S2ExpertGroupedBlob

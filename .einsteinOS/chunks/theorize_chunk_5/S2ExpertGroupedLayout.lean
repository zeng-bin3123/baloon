/-!
# MODULE: s2_expert_grouped_layout
Blob geometry, gate/up interleave, gate-major output layout, lane/chunk coverage and the bias-split
identity behind the grouped Q2_0 x Q8_0 expert kernels (`s2_expert_grouped.cu`).

## Source semantics
A routed expert blob holds `2*FF` gate/up rows (interleaved: even slot = gate, odd slot = up), then `H` down
rows, then fp16 scales. Gate/up results are written gate-major so `swiglu` pairs gate `h*FF+r` with up
`n_hits*FF + h*FF + r`. Each weight is `code-1`, evaluated as `sum c*x - sum x`.

## Form template identifiers
H -> n_embd; FF -> n_ff; rowGU/rowD -> code bytes per row; outIndex -> gate_up scratch slot.
-/

namespace Strata.S2ExpertGrouped

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

-- § geometry

/-- Row sizes match the literals the CUDA file restates. -/
theorem row_sizes : rowGU = 640 ∧ rowD = 160 ∧ scGU = 40 ∧ scD = 10 := by
  decide

/-- The blob regions are laid out in increasing order. -/
theorem offsets_monotone : oDCodes ≤ oGUScales ∧ oGUScales ≤ oDScales := by
  unfold oDScales oGUScales oDCodes rowGU rowD scGU H FF QK
  decide

-- § interleave and gate-major layout

/-- Output slot of gate/up row-slot `i` of hit `h` in the gate-major scratch. -/
def outIndex (nHits h i : Nat) : Nat :=
  if i % 2 = 1 then nHits * FF + h * FF + i / 2 else h * FF + i / 2

/-- Every written slot is inside the `2 * nHits * FF` buffer. -/
theorem outIndex_lt (nHits h i : Nat) (hh : h < nHits) (hi : i < 2 * FF) :
    outIndex nHits h i < 2 * (nHits * FF) := by
  unfold outIndex FF at *
  split <;> omega

/-- The up row sits exactly `nHits*FF` after its gate partner (what `swiglu_kernel` assumes). -/
theorem up_is_gate_plus_pairs (nHits h i : Nat) (hodd : i % 2 = 1) :
    outIndex nHits h i = outIndex nHits h (i - 1) + nHits * FF := by
  unfold outIndex
  have h1 : (i - 1) % 2 = 0 := by omega
  have h2 : (i - 1) / 2 = i / 2 := by omega
  rw [if_pos hodd, if_neg (by omega), h2]
  omega

/-- Gate slots of a hit are contiguous from `h*FF`. -/
theorem gate_contiguous (nHits h r : Nat) :
    outIndex nHits h (2 * r) = h * FF + r := by
  unfold outIndex
  rw [if_neg (by omega)]
  omega

-- § lane coverage

/-- Chunks per lane in the gate/up row loop. -/
theorem gu_chunks_per_lane : (H / 32 + 31) / 32 = 3 := by
  decide

/-- Lane `c % 32` at stride step `c / 32` visits every chunk of a gate/up row. -/
theorem every_chunk_visited (c : Nat) (hc : c < H / 32) :
    ∃ lane q, lane < 32 ∧ q < 3 ∧ c = lane + 32 * q := by
  refine ⟨c % 32, c / 32, ?_, ?_, ?_⟩ <;> unfold H at hc <;> omega

-- § 2-bit unpack and bias split

/-- The four 2-bit fields of a code byte reconstruct the byte. -/
theorem fields_reconstruct (c : Nat) (hc : c < 256) :
    c % 4 + 4 * ((c / 4) % 4) + 16 * ((c / 16) % 4) + 64 * ((c / 64) % 4) = c := by
  omega

/-- Per-element bias split: `(c-1)*x = c*x - x`. -/
theorem bias_split (c x : Int) : (c - 1) * x = c * x - x := by
  rw [Int.sub_mul, Int.one_mul]

/-- Four-lane (`dp4a`) version of the bias split. -/
theorem bias_split4 (c0 c1 c2 c3 x0 x1 x2 x3 : Int) :
    (c0 - 1) * x0 + (c1 - 1) * x1 + (c2 - 1) * x2 + (c3 - 1) * x3 =
      (c0 * x0 + c1 * x1 + c2 * x2 + c3 * x3) - (x0 + x1 + x2 + x3) := by
  rw [bias_split, bias_split, bias_split, bias_split]
  omega

/-- One fp16 weight scale per two 32-element chunks. -/
theorem scale_index (c : Nat) : c / 2 = (c * 32) / 64 := by
  omega

end Strata.S2ExpertGrouped

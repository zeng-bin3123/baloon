/-!
# MODULE: s2_grouped_experts
SOURCES: src/kernels/cuda/s2_expert_grouped.cu
Grouped GPU MoE experts over Q2_0 blobs (gate/up/down), warp per output row with dp4a against Q8_0 activations;
device-side hit selection and compaction; grouped variants; CPU-order bit-exact variant.

## Source semantics
Blob geometry (H = 2560, FF = 640): gate/up codes 2*FF rows of 640 B, down codes H rows of 160 B, then fp16
scales. A weight code c in {0..3} decodes to c - 1, so sum (c-1)x = sum cx - sum x (two dp4a). Gate/up row slots
are interleaved (even = gate, odd = up) and written to a gate-major output: gate at e*FF + r, up at
cap*FF + e*FF + r with r = slot/2. Hit selection compacts resident experts in lane order by popcount of lower
lanes.

## Form template identifiers
O_* -> "blob section offsets"; gmIdx -> "gate-major output index"; rank -> "ballot prefix rank".
-/
namespace Strata.S2Grouped

-- § blob geometry
/-- Hidden size. -/
def H : Nat := 2560
/-- Expert FFN width. -/
def FF : Nat := 640
/-- Bytes of codes per gate/up row. -/
def rowGu : Nat := H / 4
/-- Bytes of codes per down row. -/
def rowD : Nat := FF / 4
/-- fp16 scales per gate/up row. -/
def scGu : Nat := H / 64
/-- fp16 scales per down row. -/
def scD : Nat := FF / 64
/-- Offset of the down codes. -/
def oDCodes : Nat := 2 * FF * rowGu
/-- Offset of the gate/up scales. -/
def oGuScales : Nat := oDCodes + H * rowD
/-- Offset of the down scales. -/
def oDScales : Nat := oGuScales + 2 * FF * scGu * 2
/-- Total blob bytes. -/
def blobBytes : Nat := oDScales + H * scD * 2

/-- The literal row sizes of the source. -/
theorem row_sizes : rowGu = 640 ∧ rowD = 160 ∧ scGu = 40 ∧ scD = 10 := by
  decide

/-- The literal section offsets. -/
theorem section_offsets : oDCodes = 819200 ∧ oGuScales = 1228800 ∧ oDScales = 1331200 := by
  decide

/-- Total blob size. -/
theorem blob_bytes_value : blobBytes = 1382400 := by
  decide

/-- Sections are laid out in increasing order. -/
theorem sections_ordered : oDCodes < oGuScales ∧ oGuScales < oDScales ∧ oDScales < blobBytes := by
  decide

/-- 80 chunks of 32 split into 3 lane passes (GU_CHUNKS), and a down row has 20 chunks, under one warp. -/
theorem chunk_counts : (H / 32 + 31) / 32 = 3 ∧ FF / 32 = 20 ∧ FF / 32 < 32 := by
  decide

-- § bias identity
/-- `__dp4a` without accumulator. -/
def dp4 (a b : Fin 4 → Int) : Int := a 0 * b 0 + a 1 * b 1 + a 2 * b 2 + a 3 * b 3

/-- sum (c - 1) x = sum c x - sum x, the identity behind the (s - hx) form. -/
theorem bias_identity (c x : Fin 4 → Int) :
    dp4 (fun i => c i - 1) x = dp4 c x - (x 0 + x 1 + x 2 + x 3) := by
  unfold dp4
  simp only [Int.sub_mul, Int.mul_add, Int.mul_one, Int.one_mul]
  omega

-- § gate-major output layout
/-- Output index of gate/up slot `i` of entry `h` among `n` entries. -/
def gmIdx (n h i : Nat) : Nat :=
  if i % 2 = 0 then h * 640 + i / 2 else n * 640 + h * 640 + i / 2

/-- Every output index is inside the 2*n*FF buffer. -/
theorem gmIdx_lt (n h i : Nat) (hh : h < n) (hi : i < 1280) : gmIdx n h i < 2 * n * 640 := by
  unfold gmIdx
  split <;> omega

/-- The interleaved-to-gate-major map is injective: no two (entry, slot) pairs collide. -/
theorem gmIdx_injective (n h i h' i' : Nat) (hh : h < n) (hh' : h' < n) (hi : i < 1280) (hi' : i' < 1280)
    (e : gmIdx n h i = gmIdx n h' i') : h = h' ∧ i = i' := by
  unfold gmIdx at e
  split at e <;> split at e <;> omega

/-- Gate values of consecutive entries are contiguous (what `swiglu` and the quantizer read). -/
theorem gmIdx_gate_contiguous (n h r : Nat) : gmIdx n h (2 * r) = h * 640 + r := by
  unfold gmIdx
  split <;> omega

-- § device compaction by ballot prefix
/-- Number of hits strictly before position i (popcount of lower lanes). -/
def rank (p : Nat → Bool) : Nat → Nat
  | 0 => 0
  | i + 1 => rank p i + (if p i = true then 1 else 0)

/-- Rank never decreases by one step. -/
theorem rank_le_succ (p : Nat → Bool) (i : Nat) : rank p i ≤ rank p (i + 1) := by
  show rank p i ≤ rank p i + (if p i = true then 1 else 0)
  split <;> omega

/-- Rank is monotone. -/
theorem rank_mono_add (p : Nat → Bool) (i d : Nat) : rank p i ≤ rank p (i + d) := by
  induction d with
  | zero => exact Nat.le_refl _
  | succ d ih => exact Nat.le_trans ih (rank_le_succ p (i + d))

/-- A hit's compacted slot is strictly below the slot of every later position: compaction preserves order
and gives distinct slots to distinct hits. -/
theorem rank_strict (p : Nat → Bool) (i j : Nat) (hp : p i = true) (hij : i < j) : rank p i < rank p j := by
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_lt hij
  have h1 : rank p (i + 1) = rank p i + 1 := by
    show rank p i + (if p i = true then 1 else 0) = _
    rw [if_pos hp]
  have h2 := rank_mono_add p (i + 1) d
  have e : i + d + 1 = i + 1 + d := by omega
  rw [e]
  omega

end Strata.S2Grouped

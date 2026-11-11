/-!
# MODULE: mtp_runtime_blob_layout
SOURCES: tools/mtp_rt.py
Relays the packed MTP GGUF into the engine's runtime files: experts.bin in the CPU expert blob layout
(gate/up rows interleaved, then down rows; a code plane then a scale plane), and dense.bin with large
projections in Q8_0, hyper-connection/router weights BF16, and RMSNorm weights as F32 `1 + w`.

## Source semantics
* `BLOB = 3·(H·FF·18/64) = 1,382,400` with H = 2560, FF = 640 — the same per-expert size the memory planner
  uses (`Costs::expert_blob`).
* Interleave: row `2r` = gate row `r`, row `2r+1` = up row `r`, r < 640.
* Blob = [gu codes (1280×40×16)] [down codes (2560×10×16)] [gu scales (1280×40×2)] [down scales (2560×10×2)].
* dense.bin: each tensor is preceded by `(-off) % 256` zero bytes, so every offset is 256-aligned.
* Q8_0 here rounds `sign(v)·floor(|v| + 0.5)` (ggml's half-away-from-zero), whereas `mtp_pack.py`'s Q8_0
  uses `np.rint` (half-to-even). The two agree except on exact ties (proved below).
Rounding is modelled on half-units: an integer `x` stands for the value `x/2`.

## Form template identifiers
BLOB → blobBytes, H/FF/NE → hidden/ffn/nExpert, blob_of interleave → interleave / rowSource,
plane offsets → planeOffsets, dense pad → pad256, q8_0 rounding → roundAway, np.rint → rintEven
-/

namespace Strata.MtpRt

-- § geometry

/-- Hidden width H. -/
def hidden : Nat := 2560
/-- Expert FFN width FF. -/
def ffn : Nat := 640
/-- Routed experts NE. -/
def nExpert : Nat := 512

/-- `BLOB = 3 · (H·FF·18 / 64)`. -/
def blobBytes : Nat := 3 * (hidden * ffn * 18 / 64)

/-- The runtime blob is 1,382,400 B — equal to `Costs::expert_blob` in the memory planner, so the planner's
slot size and the file's blob size agree. -/
theorem blob_matches_planner : blobBytes = 1382400 := by decide

/-- The four planes, in file order, sum to exactly one blob (the `assert len(out) == BLOB`). -/
theorem planes_sum_to_blob :
    2 * ffn * (hidden / 64) * 16 + hidden * (ffn / 64) * 16 +
      2 * ffn * (hidden / 64) * 2 + hidden * (ffn / 64) * 2 = blobBytes := by decide

/-- Byte offsets of the four planes inside a blob. -/
def planeOffsets : List Nat := [0, 819200, 1228800, 1331200]

/-- The plane offsets are the running sums of the plane sizes. -/
theorem planeOffsets_correct :
    planeOffsets = [0, 2 * ffn * (hidden / 64) * 16,
      2 * ffn * (hidden / 64) * 16 + hidden * (ffn / 64) * 16,
      2 * ffn * (hidden / 64) * 16 + hidden * (ffn / 64) * 16 + 2 * ffn * (hidden / 64) * 2] := by decide

/-- The source tensors' shapes: 720 = 2560/64·18 bytes per gate/up row, 180 = 640/64·18 per down row. -/
theorem source_row_bytes : hidden / 64 * 18 = 720 ∧ ffn / 64 * 18 = 180 := by decide

/-- experts.bin is 512 blobs. -/
theorem experts_bin_size : nExpert * blobBytes = 707788800 := by decide

-- § gate/up interleave

/-- `inter[0::2] = gate; inter[1::2] = up`. -/
def interleave {α : Type} : List α → List α → List α
  | a :: as, b :: bs => a :: b :: interleave as bs
  | _, _ => []

/-- Equal-length gate and up give `2·FF` interleaved rows. -/
theorem interleave_length {α : Type} : ∀ (g u : List α), g.length = u.length →
    (interleave g u).length = 2 * g.length
  | [], [], _ => rfl
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | a :: as, b :: bs, h => by
    simp only [List.length_cons, Nat.add_right_cancel_iff] at h
    simp only [interleave, List.length_cons, interleave_length as bs h]
    omega

/-- Row `2r` is gate row `r`. -/
theorem interleave_even {α : Type} : ∀ (g u : List α) (r : Nat), g.length = u.length →
    (interleave g u)[2 * r]? = g[r]?
  | [], [], _, _ => rfl
  | [], _ :: _, _, h => by simp at h
  | _ :: _, [], _, h => by simp at h
  | a :: as, b :: bs, 0, _ => rfl
  | a :: as, b :: bs, r + 1, h => by
    simp only [List.length_cons, Nat.add_right_cancel_iff] at h
    have : 2 * (r + 1) = 2 * r + 2 := by omega
    simp only [interleave, this, List.getElem?_cons_succ]
    exact interleave_even as bs r h

/-- Row `2r + 1` is up row `r`. -/
theorem interleave_odd {α : Type} : ∀ (g u : List α) (r : Nat), g.length = u.length →
    (interleave g u)[2 * r + 1]? = u[r]?
  | [], [], _, _ => rfl
  | [], _ :: _, _, h => by simp at h
  | _ :: _, [], _, h => by simp at h
  | a :: as, b :: bs, 0, _ => rfl
  | a :: as, b :: bs, r + 1, h => by
    simp only [List.length_cons, Nat.add_right_cancel_iff] at h
    have : 2 * (r + 1) + 1 = 2 * r + 1 + 2 := by omega
    simp only [interleave, this, List.getElem?_cons_succ]
    exact interleave_odd as bs r h

/-- Index form: interleaved row `i` comes from gate (`i` even) or up (`i` odd), row `i / 2`. -/
def rowSource (i : Nat) : Bool × Nat := (i % 2 == 1, i / 2)

/-- The index map is a bijection between `[0, 2·FF)` and `{gate, up} × [0, FF)`. -/
theorem rowSource_bijective (i : Nat) (hi : i < 2 * ffn) :
    (rowSource i).2 < ffn ∧ 2 * (rowSource i).2 + (if (rowSource i).1 then 1 else 0) = i := by
  unfold rowSource ffn at *
  constructor
  · simp only; omega
  · by_cases h : i % 2 = 1
    · simp only [h, beq_self_eq_true, if_true]; omega
    · have h0 : (i % 2 == 1) = false := by simp only [beq_eq_false_iff_ne, ne_eq]; exact h
      simp only [h0, Bool.false_eq_true, if_false]; omega

-- § dense.bin alignment

/-- `(-off) % 256`. -/
def pad256 (off : Nat) : Nat := (256 - off % 256) % 256

/-- Every dense tensor starts 256-aligned. -/
theorem pad256_aligns (off : Nat) : (off + pad256 off) % 256 = 0 ∧ pad256 off < 256 := by
  unfold pad256; omega

-- § Q8_0 rounding: ggml (mtp_rt) vs np.rint (mtp_pack)

/-- `sign(v)·floor(|v| + 0.5)` for `v = x/2`. -/
def roundAway (x : Int) : Int := if 0 ≤ x then (x + 1) / 2 else -((-x + 1) / 2)

/-- `np.rint(v)` for `v = x/2`: exact halves go to the even neighbour. -/
def rintEven (x : Int) : Int :=
  if x % 2 = 0 then x / 2 else
    let lo := (x - 1) / 2
    if lo % 2 = 0 then lo else lo + 1

/-- ggml's rounding is odd-symmetric: `round(−v) = −round(v)`. -/
theorem roundAway_neg (x : Int) : roundAway (-x) = -roundAway x := by
  unfold roundAway
  by_cases h : x = 0
  · subst h; decide
  · by_cases hp : 0 ≤ x
    · have : ¬ (0 ≤ -x) := by omega
      simp only [this, hp, if_true, if_false, Int.neg_neg]
    · have : 0 ≤ -x := by omega
      simp only [this, hp, if_true, if_false, Int.neg_neg]

/-- Off ties (whole values `v`) the two roundings agree. -/
theorem rounding_agree_whole (x : Int) (h : x % 2 = 0) : roundAway x = rintEven x := by
  unfold roundAway rintEven
  simp only [h, if_true]
  split <;> omega

/-- On ties they differ: 0.5 → 1 vs 0, 2.5 → 3 vs 2, −0.5 → −1 vs 0. -/
theorem rounding_differ_on_ties :
    roundAway 1 = 1 ∧ rintEven 1 = 0 ∧ roundAway 5 = 3 ∧ rintEven 5 = 2 ∧
    roundAway (-1) = -1 ∧ rintEven (-1) = 0 := by decide

/-- With `d = amax/127`, |v| ≤ 127, so the rounded code fits int8. -/
theorem roundAway_fits_int8 (x : Int) (h1 : -254 ≤ x) (h2 : x ≤ 254) :
    -127 ≤ roundAway x ∧ roundAway x ≤ 127 := by
  unfold roundAway
  split <;> omega

/-- The rounding error of ggml's rule is at most one half unit. -/
theorem roundAway_error (x : Int) : -1 ≤ 2 * roundAway x - x ∧ 2 * roundAway x - x ≤ 1 := by
  unfold roundAway
  split <;> omega

end Strata.MtpRt

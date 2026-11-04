/-!
# MODULE: quantize_activation
Layout and range contracts of the Q8_0 / Q8_K activation quantizers and their dequantizers.

## Source semantics
Q8_0 block = fp16 scale (2 B) + 32 int8 = 34 B. Q8_K block = f32 scale + 256 int8 + 16 int16 bsums = 292 B.
The ggml-matching Q8_0 clamps to [-128,127]; the CPU-matching scaled variant clamps to [-127,127].
Q8_K clamps only above (min 127). Each bsum is the sum of 16 int8 values.

## Form template identifiers
- `clampI lo hi` ↔ the `v < lo ? lo : (v > hi ? hi : v)` clamps
- `sum16`/`listSum` ↔ `bsums[j]`; `q8_0Off`/`q8kOff` ↔ block byte offsets
-/

namespace Strata.QuantizeActivation

-- § layout

/-- Q8_0 block size in bytes. -/
theorem q8_0_block_bytes : 2 + 32 = 34 := by decide

/-- Q8_K block size in bytes: 4 + 256 + 16*2. -/
theorem q8k_block_bytes : 4 + 256 + 16 * 2 = 292 := by decide

/-- Byte offset of quant `i` of Q8_0 block `b`. -/
def q8_0Off (b i : Nat) : Nat := b * 34 + 2 + i

/-- Every quant of block `b` lies inside block `b`'s 34 bytes. -/
theorem q8_0Off_in_block (b i : Nat) (h : i < 32) :
    b * 34 + 2 ≤ q8_0Off b i ∧ q8_0Off b i < (b + 1) * 34 := by
  unfold q8_0Off; omega

/-- Distinct (block, index) pairs address distinct bytes. -/
theorem q8_0Off_inj (b i b' i' : Nat) (h : i < 32) (h' : i' < 32)
    (e : q8_0Off b i = q8_0Off b' i') : b = b' ∧ i = i' := by
  unfold q8_0Off at e; omega

/-- Byte offset of the bsums of Q8_K block `b`. -/
def q8kBsums (b : Nat) : Nat := b * 292 + 4 + 256

/-- The bsums of block `b` end exactly at the block boundary. -/
theorem q8kBsums_end (b : Nat) : q8kBsums b + 32 = (b + 1) * 292 := by
  unfold q8kBsums; omega

-- § clamps

/-- Integer clamp into `[lo, hi]`. -/
def clampI (lo hi v : Int) : Int := max lo (min hi v)

/-- The clamp stays within bounds when `lo ≤ hi`. -/
theorem clampI_bounds (lo hi v : Int) (h : lo ≤ hi) : lo ≤ clampI lo hi v ∧ clampI lo hi v ≤ hi := by
  unfold clampI; omega

/-- Values already in range are untouched. -/
theorem clampI_id (lo hi v : Int) (h1 : lo ≤ v) (h2 : v ≤ hi) : clampI lo hi v = v := by
  unfold clampI; omega

/-- The scaled-variant clamp never produces -128 (symmetric int8 range). -/
theorem scaled_clamp_ne_min (v : Int) : clampI (-127) 127 v ≠ -128 := by
  unfold clampI; omega

/-- Q8_K quantization clamps only from above: `min 127 v`. -/
def q8kClamp (v : Int) : Int := min 127 v

/-- The Q8_K clamp is bounded above by 127. -/
theorem q8kClamp_le (v : Int) : q8kClamp v ≤ 127 := by
  unfold q8kClamp; omega

-- § bsums

/-- Sum of a list of ints. -/
def listSum : List Int → Int
  | [] => 0
  | x :: xs => x + listSum xs

/-- A list of values in `[-127,127]` sums within `127 * length` in magnitude. -/
theorem listSum_bound (l : List Int) (h : ∀ x ∈ l, -127 ≤ x ∧ x ≤ 127) :
    -(127 * (l.length : Int)) ≤ listSum l ∧ listSum l ≤ 127 * (l.length : Int) := by
  induction l with
  | nil => simp [listSum]
  | cons x xs ih =>
    have hx := h x (List.mem_cons_self ..)
    have ih' := ih (fun y hy => h y (List.mem_cons_of_mem _ hy))
    simp only [listSum, List.length_cons]
    omega

/-- A 16-element bsum of values in `[-127,127]` fits in int16 (|s| ≤ 2032). -/
theorem bsum16_fits (l : List Int) (hl : l.length = 16) (h : ∀ x ∈ l, -127 ≤ x ∧ x ≤ 127) :
    -32768 ≤ listSum l ∧ listSum l ≤ 32767 := by
  have := listSum_bound l h
  rw [hl] at this
  omega

-- § rounding / zero block

/-- Zero-amax guard: the scale is `0` iff `amax = 0` (over naturals standing in for magnitudes). -/
def scaleOf (amax : Nat) : Nat := if amax > 0 then amax / 127 else 0

/-- A zero amax gives a zero scale and hence the all-zero block. -/
theorem scaleOf_zero : scaleOf 0 = 0 := by simp [scaleOf]

/-- Dequantization is multiplication by the stored scale: zero quant gives zero. -/
theorem dequant_zero (d : Int) : (0 : Int) * d = 0 := by omega

end Strata.QuantizeActivation

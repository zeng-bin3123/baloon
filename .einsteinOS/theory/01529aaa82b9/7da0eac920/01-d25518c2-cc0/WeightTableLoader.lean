/-!
# MODULE: weight_table_loader
Dense-weight loader: pool-size accounting with and without a skip set, the arena-fit check, exact
bf16 narrowing and fp16 scale widening, and the activation-kind flag.

## Source semantics
* `pool_bytes` with a skip set is never larger than without it; skipped tensors contribute nothing.
* `load` refuses an arena smaller than `pool_bytes` rather than writing past the end.
* A bf16 value stored promoted to f32 is recovered exactly by taking its high 16 bits.
* Widening fp16 scales to f32 doubles their byte count, changing no value.
* `wants_q8k` is true exactly for `act_kind == 1`.

## Form template identifiers
* `WeightTable::pool_bytes` -> `poolBytes`
* `WeightTable::load` arena check -> `loadOk`
* `WeightKind::Bf16InF32` -> `bf16Narrow`
* `WeightRef::wants_q8k` -> `wantsQ8k`
-/

namespace Strata.WeightTableLoader

-- § pool accounting

/-- Bytes a pool needs: tensors flagged as skipped contribute nothing. -/
def poolBytes : List (Nat × Bool) → Nat
  | [] => 0
  | (b, s) :: t => (if s then 0 else b) + poolBytes t

/-- Bytes with nothing skipped. -/
def totalBytes : List (Nat × Bool) → Nat
  | [] => 0
  | (b, _) :: t => b + totalBytes t

/-- Skipping tensors never grows the pool. -/
theorem poolBytes_le_total (l : List (Nat × Bool)) : poolBytes l ≤ totalBytes l := by
  induction l with
  | nil => simp [poolBytes, totalBytes]
  | cons x t ih =>
    obtain ⟨b, s⟩ := x
    cases s <;> simp [poolBytes, totalBytes] <;> omega

/-- A pool of all-skipped tensors needs zero bytes. -/
theorem poolBytes_all_skipped (l : List Nat) : poolBytes (l.map (fun b => (b, true))) = 0 := by
  induction l with
  | nil => simp [poolBytes]
  | cons b t ih => simp [poolBytes, ih]

/-- A pool of nothing-skipped tensors needs exactly the total. -/
theorem poolBytes_none_skipped (l : List Nat) :
    poolBytes (l.map (fun b => (b, false))) = l.sum := by
  induction l with
  | nil => simp [poolBytes]
  | cons b t ih => simp [poolBytes, ih]

/-- The arena is accepted only if it is at least `pool_bytes`. -/
def loadOk (arena pool : Nat) : Bool := decide (pool ≤ arena)

/-- An arena smaller than the pool is refused instead of overrun. -/
theorem load_refuses_small (arena pool : Nat) (h : arena < pool) : loadOk arena pool = false := by
  unfold loadOk
  exact decide_eq_false (by omega)

/-- An arena at least as large as the pool is accepted. -/
theorem load_accepts_fit (arena pool : Nat) (h : pool ≤ arena) : loadOk arena pool = true := by
  unfold loadOk
  exact decide_eq_true h

-- § exact conversions

/-- The f32 bit pattern of a bf16 value promoted into f32. -/
def bf16InF32 (h : Nat) : Nat := h * 65536

/-- Narrowing keeps the high 16 bits. -/
def bf16Narrow (bits : Nat) : Nat := bits / 65536

/-- Narrowing a promoted bf16 recovers it exactly, so re-rounding to 16 bits loses nothing. -/
theorem bf16_narrow_exact (h : Nat) : bf16Narrow (bf16InF32 h) = h := by
  unfold bf16Narrow bf16InF32
  omega

/-- The promoted pattern has its low 16 bits clear, so it is a bf16 value, not a general f32. -/
theorem bf16_promoted_low_clear (h : Nat) : bf16InF32 h % 65536 = 0 := by
  unfold bf16InF32
  omega

/-- Each re-rounded tensor of `n` elements saves `2n` bytes against the 4-byte container. -/
theorem rerounded_saving (n : Nat) : 4 * n - 2 * n = 2 * n := by
  omega

/-- Widening fp16 scales to f32 adds exactly `2 * count` bytes, and the arena width is `4 * count`. -/
theorem widen_scales (count : Nat) : 4 * count = 2 * count + 2 * count := by
  omega

/-- The widened-scales total of 11,407,360 B is an even byte count and about 10.88 MiB. -/
theorem widened_scales_total :
    11407360 % 2 = 0 ∧ 10 * 1048576 < 11407360 ∧ 11407360 < 11 * 1048576 := by
  decide

-- § activation kind

/-- Whether a weight wants a Q8_K activation (`act_kind == 1`) rather than Q8_0 (`act_kind == 0`). -/
def wantsQ8k (actKind : Nat) : Bool := decide (actKind = 1)

/-- Q8_0 weights do not want Q8_K, and the Q8_K kind does. -/
theorem act_kind_cases : wantsQ8k 0 = false ∧ wantsQ8k 1 = true := by
  decide

/-- A tensor is quantized exactly when its code width is nonzero. -/
def quantized (codeBits : Nat) : Bool := decide (codeBits ≠ 0)

/-- Q2_0 (2-bit codes) is quantized; F32 (no code planes) is not. -/
theorem quantized_cases : quantized 2 = true ∧ quantized 0 = false := by
  decide

/-- Plane sizes: codes, scales and offsets sum to the tensor's arena bytes. -/
def planeTotal (codes scales offset : Nat) : Nat := codes + scales + offset

/-- A form with no offset plane has total equal to codes plus scales. -/
theorem plane_total_no_offset (codes scales : Nat) : planeTotal codes scales 0 = codes + scales := by
  unfold planeTotal
  omega

end Strata.WeightTableLoader

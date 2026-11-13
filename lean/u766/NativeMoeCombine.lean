/-!
# MODULE: native_moe_combine
SOURCES: src/kernels/cuda/native_moe.cu
Model of the pinned ggml `moe-weighted-reduction` combine: argument validation (stream,
width, 1..15 experts), pointer-span validity without address overflow, output/input
disjointness, and the float accumulation order `p₀·w₀ + p₁·w₁ + … (+ shared)`.

## Source semantics
* Accepts `1 ≤ k ≤ 15`, `0 < n_embd ≤ INT_MAX`, a non-null stream; spans are non-null,
  4-byte aligned and `bytes ≤ UINTPTR_MAX − address`.
* The output row must not overlap parts, weights or shared (half-open interval overlap).
* Column `col` computes `sum = parts[col]*w[0]`, then `sum += parts[e*n+col]*w[e]` for
  `e = 1..k-1`, then `sum += shared[col]` — no `0 +` seed and `part * weight` operand order,
  which differs from the legacy double-precision `moe_combine`.
* The grid is `⌈n_embd/256⌉` blocks of 256 threads, one column per thread.

## Form template identifiers
`valid_span` → `validSpan`; `overlap` → `overlaps`; argument guard → `argsOk`; `combine`
kernel → `nativeSum`; legacy kernel → `legacySum`; byte sizes → `partBytes`/`rowBytes`.
-/

namespace NativeMoeCombine

-- § argument validation

/-- `INT_MAX`. -/
def INT_MAX : Nat := 2^31 - 1

/-- The scalar guard of `native_moe_combine`. -/
def argsOk (hasStream : Bool) (nEmbd k : Nat) : Bool :=
  hasStream && 0 < nEmbd && nEmbd ≤ INT_MAX && 1 ≤ k && k ≤ 15

/-- The guard is exactly the documented range. -/
theorem argsOk_iff (s : Bool) (n k : Nat) :
    argsOk s n k = true ↔ s = true ∧ 0 < n ∧ n ≤ INT_MAX ∧ 1 ≤ k ∧ k ≤ 15 := by
  simp [argsOk, Bool.and_eq_true, decide_eq_true_eq, and_assoc]

/-- Row bytes. -/
def rowBytes (n : Nat) : Nat := n * 4
/-- All routed parts. -/
def partBytes (n k : Nat) : Nat := rowBytes n * k

/-- Within the accepted range the part span is at most 15 rows. -/
theorem partBytes_le (n k : Nat) (hk : k ≤ 15) : partBytes n k ≤ 15 * rowBytes n := by
  unfold partBytes; rw [Nat.mul_comm]; exact Nat.mul_le_mul_right _ hk

-- § spans

/-- `UINTPTR_MAX` on a 64-bit host. -/
def UPTR_MAX : Nat := 2^64 - 1

/-- `valid_span`: non-null, float-aligned, and the span does not wrap the address space. -/
def validSpan (addr bytes : Nat) : Prop := addr ≠ 0 ∧ addr % 4 = 0 ∧ bytes ≤ UPTR_MAX - addr

/-- The subtraction form `bytes ≤ MAX − addr` is the overflow-free spelling of
    `addr + bytes ≤ MAX` (for in-range addresses). -/
theorem validSpan_no_wrap (addr bytes : Nat) (ha : addr ≤ UPTR_MAX) :
    bytes ≤ UPTR_MAX - addr ↔ addr + bytes ≤ UPTR_MAX := by omega

/-- `overlap(a, an, b, bn)`: half-open intervals intersect. -/
def overlaps (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlaps_symm (a an b bn : Nat) : overlaps a an b bn ↔ overlaps b bn a an := by
  unfold overlaps; omega

/-- Non-overlap means one span ends before the other begins. -/
theorem not_overlaps_iff (a an b bn : Nat) :
    ¬ overlaps a an b bn ↔ (b + bn ≤ a ∨ a + an ≤ b) := by
  unfold overlaps; omega

/-- Two non-empty overlapping spans share an address. -/
theorem overlaps_witness (a an b bn : Nat) (h : overlaps a an b bn) (ha : 0 < an) (hb : 0 < bn) :
    ∃ x, a ≤ x ∧ x < a + an ∧ b ≤ x ∧ x < b + bn := by
  unfold overlaps at h
  exact ⟨max a b, by omega, by omega, by omega, by omega⟩

/-- A disjoint output means no output column aliases an input byte the kernel reads. -/
theorem disjoint_no_alias (o n p pn x : Nat) (h : ¬ overlaps o n p pn)
    (hx : o ≤ x ∧ x < o + n) : ¬ (p ≤ x ∧ x < p + pn) := by
  unfold overlaps at h; omega

-- § grid

/-- One thread per column: `⌈n/256⌉` blocks cover every column exactly once. -/
theorem grid_covers (n c : Nat) (h : c < n) :
    c / 256 < (n + 255) / 256 ∧ c / 256 * 256 + c % 256 = c := by omega

-- § accumulation order

/-- Free syntax of float expressions (no algebraic laws assumed). -/
inductive FExpr where
  | part (e : Nat)
  | weight (e : Nat)
  | shared
  | zero
  | add (a b : FExpr)
  | mul (a b : FExpr)
  deriving DecidableEq

/-- The native kernel's expression for `k` experts. -/
def nativeSum (k : Nat) (hasShared : Bool) : FExpr :=
  let routed := (List.range (k - 1)).foldl
    (fun acc i => FExpr.add acc (FExpr.mul (FExpr.part (i + 1)) (FExpr.weight (i + 1))))
    (FExpr.mul (FExpr.part 0) (FExpr.weight 0))
  if hasShared then FExpr.add routed FExpr.shared else routed

/-- The legacy `moe_combine` expression: seeded with `0`, `weight * part` order. -/
def legacySum (k : Nat) (hasShared : Bool) : FExpr :=
  let routed := (List.range k).foldl
    (fun acc i => FExpr.add acc (FExpr.mul (FExpr.weight i) (FExpr.part i))) FExpr.zero
  if hasShared then FExpr.add routed FExpr.shared else routed

/-- Two experts with a shared output: `((p₀w₀ + p₁w₁) + s)`. -/
theorem nativeSum_two :
    nativeSum 2 true =
      FExpr.add (FExpr.add (FExpr.mul (.part 0) (.weight 0)) (FExpr.mul (.part 1) (.weight 1)))
        FExpr.shared := rfl

/-- The pinned native expression differs syntactically from the legacy one for every accepted
    `k`, so the two are not interchangeable bit-for-bit; selecting the path is a real switch. -/
theorem native_ne_legacy : ∀ k, k < 16 → 1 ≤ k → ∀ s : Bool, nativeSum k s ≠ legacySum k s := by
  decide

/-- Integer evaluation (exact arithmetic) of an expression. -/
def eval (p w : Nat → Int) (sh : Int) : FExpr → Int
  | .part e => p e
  | .weight e => w e
  | .shared => sh
  | .zero => 0
  | .add a b => eval p w sh a + eval p w sh b
  | .mul a b => eval p w sh a * eval p w sh b

/-- In exact arithmetic both orders compute the same weighted sum (checked here for two
    experts with a shared output); only rounding can tell them apart. -/
theorem native_legacy_exact (p w : Nat → Int) (sh : Int) :
    eval p w sh (nativeSum 2 true) = eval p w sh (legacySum 2 true) := by
  simp only [nativeSum, legacySum, List.range, List.range.loop, List.foldl, eval, ite_true, Nat.zero_add]
  rw [Int.mul_comm (p 0), Int.mul_comm (p 1)]
  omega

end NativeMoeCombine

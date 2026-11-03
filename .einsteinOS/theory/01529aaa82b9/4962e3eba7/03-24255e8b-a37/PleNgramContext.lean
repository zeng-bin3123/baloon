/-!
# MODULE: ple_ngram_context
Context construction and incremental mixing of the PLE n-gram hash (`ngram_rows` / `ngram_mixed`).

## Source semantics
* `ctx[0]` is the current token; `ctx[s]` is the predecessor `s` positions back.
* Once a predecessor is negative or equals EOS the context is cut: that position and every later one is EOS.
  Before the cut, positions hold the real predecessor.
* `mixed(n) = ctx[0]*m[0] ^ ctx[1]*m[1] ^ ... ^ ctx[n-1]*m[n-1]` over 64-bit wrapping arithmetic, so the order-3
  hash extends the order-2 hash by one XOR term, and an order-`n` hash depends only on `ctx[0..n-1]`.

## Form template identifiers
cut flag -> `cutAt`; stored context value -> `ctxVal`; ngram_mixed -> `mixed`.
-/

namespace Strata.Kernels.NgramContext

-- § context cut

/-- Cut flag after processing `s` predecessors; `prev s` is the predecessor at step `s + 1`. -/
def cutAt (prev : Nat → Int) (eos : Int) : Nat → Bool
  | 0 => false
  | s + 1 => cutAt prev eos s || decide (prev s < 0) || decide (prev s = eos)

/-- Once cut, always cut. -/
theorem cut_mono (prev : Nat → Int) (eos : Int) (s : Nat) (h : cutAt prev eos s = true) :
    cutAt prev eos (s + 1) = true := by
  simp [cutAt, h]

/-- Cut persists over any number of further steps. -/
theorem cut_persists (prev : Nat → Int) (eos : Int) (s d : Nat) (h : cutAt prev eos s = true) :
    cutAt prev eos (s + d) = true := by
  induction d with
  | zero => simpa using h
  | succ d ih => exact cut_mono prev eos (s + d) ih

/-- Stored context value at step `s + 1`: EOS once cut, otherwise the real predecessor. -/
def ctxVal (prev : Nat → Int) (eos : Int) (s : Nat) : Int :=
  if cutAt prev eos (s + 1) = true then eos else prev s

/-- After a cut every stored value is EOS. -/
theorem ctx_eos_after_cut (prev : Nat → Int) (eos : Int) (s d : Nat) (h : cutAt prev eos s = true) :
    ctxVal prev eos (s + d) = eos := by
  have h1 := cut_persists prev eos s (d + 1) h
  unfold ctxVal
  have e : s + d + 1 = s + (d + 1) := by omega
  rw [e, if_pos h1]

/-- Before any cut, the stored value is the real predecessor. -/
theorem ctx_real_before_cut (prev : Nat → Int) (eos : Int) (s : Nat) (h : cutAt prev eos (s + 1) = false) :
    ctxVal prev eos s = prev s := by
  unfold ctxVal
  rw [if_neg (by simp [h])]

/-- A negative or EOS predecessor cuts at its own step, so the position holding it is EOS. -/
theorem eos_predecessor_is_eos (prev : Nat → Int) (eos : Int) (s : Nat) (h : prev s = eos) :
    ctxVal prev eos s = eos := by
  unfold ctxVal
  have hc : cutAt prev eos (s + 1) = true := by simp [cutAt, h]
  rw [if_pos hc]

-- § incremental mixing

/-- Wrapping 64-bit mixing: first term assigned, the rest XORed in. -/
def mixed (ctx mult : Nat → UInt64) : Nat → UInt64
  | 0 => ctx 0 * mult 0
  | j + 1 => mixed ctx mult j ^^^ (ctx (j + 1) * mult (j + 1))

/-- Order `j + 2` extends order `j + 1` by exactly one XOR term. -/
theorem mixed_succ (ctx mult : Nat → UInt64) (j : Nat) :
    mixed ctx mult (j + 1) = mixed ctx mult j ^^^ (ctx (j + 1) * mult (j + 1)) := rfl

/-- The order-`j` hash depends only on `ctx[0..j]`. -/
theorem mixed_congr (ctx ctx' mult : Nat → UInt64) :
    ∀ j, (∀ i, i ≤ j → ctx i = ctx' i) → mixed ctx mult j = mixed ctx' mult j := by
  intro j
  induction j with
  | zero =>
    intro h
    simp only [mixed]
    rw [h 0 (Nat.le_refl 0)]
  | succ j ih =>
    intro h
    simp only [mixed]
    rw [ih (fun i hi => h i (by omega)), h (j + 1) (Nat.le_refl _)]

/-- Head row index stays inside the head's vocabulary window. -/
theorem row_window (m v off : Nat) (hv : 0 < v) : off ≤ m % v + off ∧ m % v + off < off + v := by
  have := Nat.mod_lt m hv
  omega

end Strata.Kernels.NgramContext

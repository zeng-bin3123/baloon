/-!
# MODULE: layer_shape_check
SOURCES: src/core/layout.cpp
Load-time shape and engine-form checks for every layer's tensors, plus the whole-model check that
the layer-type predicate splits the 48 layers into the 12 QSA / 36 GDN the geometry promises.

## Source semantics
* A 2-D requirement `Want2` is skipped on the wrong layer family (`qsa_only` on a GDN layer,
  `gdn_only` on a QSA layer); otherwise `ne0` and `ne1` must match exactly.
* A 1-D requirement `Want1` also fixes the ENGINE FORM: F32 holds 4 B/elem, Bf16InF32 holds 2.  A
  consumer that reads a 2 B/elem tensor as f32 walks twice its length (the `ffn_gate_inp_shexp` bug).
* `check_all` counts the QSA layers via `is_qsa_layer` and compares with the geometry; with a full
  attention interval of 4 the QSA layers are exactly those with `l % 4 = 3`.

## Form template identifiers
isQsa ↦ is_qsa_layer; applies ↦ the `qsa_only`/`gdn_only` skip in check_one; countQsa ↦ n_qsa in
check_all; countGdn ↦ n_gdn; Kind ↦ WeightKind; formBytes ↦ want_bytes; want2Flags ↦ want2[];
want1Flags ↦ want1[].
-/

namespace Strata.Layout

-- § the layer-type predicate

/-- A layer is QSA when it is the last of each group of four (`full_attention_interval = 4`). -/
def isQsa (l : Nat) : Bool := decide (l % 4 = 3)

/-- QSA layers counted over `0..n-1` (the `n_qsa` of `check_all`). -/
def countQsa : Nat → Nat
  | 0 => 0
  | n + 1 => countQsa n + if isQsa n then 1 else 0

/-- GDN layers counted over `0..n-1` (the `n_gdn` of `check_all`). -/
def countGdn : Nat → Nat
  | 0 => 0
  | n + 1 => countGdn n + if isQsa n then 0 else 1

/-- The two counters always account for every layer. -/
theorem count_split (n : Nat) : countQsa n + countGdn n = n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [countQsa, countGdn]
    cases isQsa n <;> simp <;> omega

/-- Layer `4m + 3` is QSA. -/
theorem isQsa_last (m : Nat) : isQsa (4 * m + 3) = true := decide_eq_true (by omega)

/-- Layers `4m`, `4m+1`, `4m+2` are GDN. -/
theorem isQsa_other (m r : Nat) (hr : r < 3) : isQsa (4 * m + r) = false :=
  decide_eq_false (by omega)

/-- Each group of four layers contributes exactly one QSA layer. -/
theorem countQsa_step (m : Nat) : countQsa (4 * (m + 1)) = countQsa (4 * m) + 1 := by
  have e : 4 * (m + 1) = 4 * m + 3 + 1 := by omega
  have h0 := isQsa_other m 0 (by omega)
  have h1 := isQsa_other m 1 (by omega)
  have h2 := isQsa_other m 2 (by omega)
  have h3 := isQsa_last m
  rw [e]
  simp only [countQsa]
  simp only [Nat.add_zero] at h0
  rw [show 4 * m + 3 = 4 * m + 2 + 1 by omega, show 4 * m + 2 = 4 * m + 1 + 1 by omega] at *
  simp [countQsa, h0, h1, h2, h3]

/-- Over `4m` layers there are exactly `m` QSA layers. -/
theorem countQsa_mul (m : Nat) : countQsa (4 * m) = m := by
  induction m with
  | zero => rfl
  | succ m ih => rw [countQsa_step, ih]

/-- Over `4m` layers there are exactly `3m` GDN layers. -/
theorem countGdn_mul (m : Nat) : countGdn (4 * m) = 3 * m := by
  have h := count_split (4 * m)
  rw [countQsa_mul] at h
  omega

/-- **The artifact's split**: 48 layers give 12 QSA and 36 GDN, which `check_all` demands. -/
theorem split_48 : countQsa 48 = 12 ∧ countGdn 48 = 36 :=
  ⟨countQsa_mul 12, countGdn_mul 12⟩

-- § the per-family filter

/-- Whether a requirement with these flags is checked on a layer of this family. -/
def applies (qsaOnly gdnOnly qsa : Bool) : Bool := !(qsaOnly && !qsa) && !(gdnOnly && qsa)

/-- A GDN-only tensor is checked on exactly the GDN layers. -/
theorem gdnOnly_applies (qsa : Bool) : applies false true qsa = !qsa := by
  cases qsa <;> rfl

/-- A QSA-only tensor is checked on exactly the QSA layers. -/
theorem qsaOnly_applies (qsa : Bool) : applies true false qsa = qsa := by
  cases qsa <;> rfl

/-- `(qsa_only, gdn_only)` of the 22 two-dimensional requirements, in source order. -/
def want2Flags : List (Bool × Bool) :=
  [(false, false), (false, false), (false, false), (false, false), (false, false), (false, false),
   (false, false), (false, false), (false, false), (false, false),
   (false, true), (false, true), (false, true), (false, true), (false, true), (false, true),
   (true, false), (true, false), (true, false), (true, false), (true, false), (true, false)]

/-- 2-D tensors checked on one layer of the given family. -/
def checked2 (qsa : Bool) : Nat := (want2Flags.filter (fun f => applies f.1 f.2 qsa)).length

/-- A GDN layer checks 16 matrices (6 GR + 4 MoE + 6 GDN), a QSA layer also 16 (6 + 4 + 6). -/
theorem checked2_counts : checked2 false = 16 ∧ checked2 true = 16 := by decide

/-- No requirement is both QSA-only and GDN-only, so none is dead on every layer. -/
theorem want2_no_dead : ∀ f ∈ want2Flags, ¬(f.1 = true ∧ f.2 = true) := by decide

-- § engine forms of 1-D tensors

/-- The engine form a loader gives a 1-D tensor. -/
inductive Kind where
  | f32
  | bf16InF32
  deriving DecidableEq, Repr

/-- Bytes held in the arena for `e` elements of a given form. -/
def formBytes : Kind → Nat → Nat
  | .f32, e => 4 * e
  | .bf16InF32, e => 2 * e

/-- `(form, qsa_only, gdn_only)` of the ten 1-D requirements, in source order. -/
def want1Flags : List (Kind × Bool × Bool) :=
  [(.f32, false, false), (.f32, false, false), (.bf16InF32, false, false),
   (.f32, false, true), (.f32, false, true), (.f32, false, true),
   (.f32, true, false), (.f32, true, false), (.f32, true, false), (.f32, true, false)]

/-- Exactly one 1-D requirement is BF16: the shared expert's scalar gate. -/
theorem want1_single_bf16 : (want1Flags.filter (fun f => f.1 = .bf16InF32)).length = 1 := by decide

/-- **The misread.**  Reading a BF16 tensor of `e > 0` elements as f32 walks `2e` bytes past its end. -/
theorem bf16_as_f32_overrun (e : Nat) (he : 0 < e) :
    formBytes .f32 e = formBytes .bf16InF32 e + 2 * e ∧ formBytes .bf16InF32 e < formBytes .f32 e := by
  simp only [formBytes]
  omega

/-- The byte count distinguishes the two forms, which is what makes the form check real. -/
theorem formBytes_injective (k1 k2 : Kind) (e : Nat) (he : 0 < e)
    (h : formBytes k1 e = formBytes k2 e) : k1 = k2 := by
  cases k1 <;> cases k2 <;> simp [formBytes] at h ⊢ <;> omega

end Strata.Layout

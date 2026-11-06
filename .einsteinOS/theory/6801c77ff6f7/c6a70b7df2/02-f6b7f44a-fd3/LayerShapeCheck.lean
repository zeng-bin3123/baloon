/-!
# MODULE: layer_shape_check
SOURCES: src/core/layout.cpp
The pre-flight shape check of the weight table: for each layer, every required 2-D tensor has the exact `(ne0, ne1)` the
kernels read, every 1-D tensor has the exact element count AND engine form (f32 vs bf16-in-f32), and the QSA/GDN layer
split agrees with the geometry.

## Source semantics
* Per layer the required tensor set depends on the layer family: 16 two-D tensors for either family, and 6 (GDN) or 7
  (QSA) one-D tensors.
* The engine form fixes the byte count: 4 B/elem for f32, 2 B/elem for bf16. Reading a bf16 tensor as f32 overreads by
  2x its length, which is why the form is checked and not just the element count.
* `check_all` additionally counts layer kinds: QSA + GDN equals the number of layers, and the counts must equal the
  geometry's (12 QSA, 36 GDN).
* A layer index outside `0..n_layers-1` is refused.

## Form template identifiers
Want2 -> want2Flags ; Want1 -> want1Flags ; is_qsa_layer -> isQsa ; n_qsa_layers -> geometry QSA count ;
WeightKind::Bf16InF32 -> bf16 form (2 B) ; hc_dim -> hc * n_embd
-/

namespace Strata.LayerShape

-- § which tensors apply to which family

/-- A requirement applies unless it is QSA-only on a GDN layer or GDN-only on a QSA layer. -/
def applies (qsaOnly gdnOnly : Bool) (isQsaLayer : Bool) : Bool :=
  !(qsaOnly && !isQsaLayer) && !(gdnOnly && isQsaLayer)

/-- The 22 two-D requirements as `(qsaOnly, gdnOnly)` flags: 10 common, 6 GDN, 6 QSA. -/
def want2Flags : List (Bool × Bool) :=
  List.replicate 10 (false, false) ++ List.replicate 6 (false, true) ++ List.replicate 6 (true, false)

/-- The 10 one-D requirements: 3 common, 3 GDN, 4 QSA. -/
def want1Flags : List (Bool × Bool) :=
  List.replicate 3 (false, false) ++ List.replicate 3 (false, true) ++ List.replicate 4 (true, false)

/-- Number of requirements checked for one layer. -/
def countApplicable (flags : List (Bool × Bool)) (isQsaLayer : Bool) : Nat :=
  (flags.filter (fun f => applies f.1 f.2 isQsaLayer)).length

/-- A QSA layer is checked against 16 two-D tensors. -/
theorem want2_qsa : countApplicable want2Flags true = 16 := by decide
/-- A GDN layer is checked against 16 two-D tensors. -/
theorem want2_gdn : countApplicable want2Flags false = 16 := by decide
/-- A QSA layer is checked against 7 one-D tensors. -/
theorem want1_qsa : countApplicable want1Flags true = 7 := by decide
/-- A GDN layer is checked against 6 one-D tensors. -/
theorem want1_gdn : countApplicable want1Flags false = 6 := by decide

-- § engine forms and byte counts

/-- Engine form of a 1-D tensor. -/
inductive Form where
  | f32
  | bf16

/-- Bytes per element of an engine form. -/
def Form.bytesPer : Form → Nat
  | .f32 => 4
  | .bf16 => 2

/-- Bytes a tensor of `elements` occupies in a form. -/
def formBytes (f : Form) (elements : Nat) : Nat := elements * f.bytesPer

/-- THE SHARED-EXPERT GATE BUG: 2560 bf16 elements are 5120 bytes... -/
theorem shexp_gate_bytes : formBytes .bf16 2560 = 5120 := by decide

/-- ...and reading them as f32 would read 10240 bytes, past the end of the tensor. -/
theorem shexp_gate_f32_overreads : formBytes .bf16 2560 < formBytes .f32 2560 := by decide

/-- Reading a bf16 tensor as f32 always overreads when it is non-empty. -/
theorem bf16_as_f32_overreads (n : Nat) (h : 0 < n) : formBytes .bf16 n < formBytes .f32 n := by
  simp only [formBytes, Form.bytesPer]; omega

/-- The form check passes only when the recorded form is the wanted form. -/
def formOk (have_ want : Form) : Bool :=
  match have_, want with
  | .f32, .f32 => true
  | .bf16, .bf16 => true
  | _, _ => false

/-- A matching form gives a matching byte count. -/
theorem formOk_bytes (h w : Form) (elements : Nat) (hk : formOk h w = true) :
    formBytes h elements = formBytes w elements := by
  cases h <;> cases w <;> simp [formOk] at hk <;> rfl

/-- The hyper-connection width: 4 streams of 2560. -/
theorem hc_dim_value : 4 * 2560 = 10240 := by decide

-- § layer split

/-- Number of layers below `n` satisfying `p`. -/
def countP' (p : Nat → Bool) : Nat → Nat
  | 0 => 0
  | n + 1 => countP' p n + (if p n then 1 else 0)

/-- QSA layers plus GDN layers is every layer: the predicate partitions the layer range. -/
theorem split_partitions (isQsa : Nat → Bool) (n : Nat) :
    countP' isQsa n + countP' (fun l => !isQsa l) n = n := by
  induction n with
  | zero => simp [countP']
  | succ n ih =>
    simp only [countP']
    rcases Bool.eq_false_or_eq_true (isQsa n) with h | h <;> simp [h] <;> omega

/-- The geometry's split: 12 QSA plus 36 GDN is all 48 layers. -/
theorem geometry_split : 12 + 36 = 48 := by decide

/-- The count check of `check_all`. -/
def splitOk (nQsa nGdn gQsa gGdn : Nat) : Bool := decide (nQsa = gQsa) && decide (nGdn = gGdn)

/-- If the counted split matches the geometry and the geometry covers all layers, the counted kinds cover all layers. -/
theorem splitOk_covers (nQsa nGdn gQsa gGdn n : Nat) (h : splitOk nQsa nGdn gQsa gGdn = true)
    (hg : gQsa + gGdn = n) : nQsa + nGdn = n := by
  simp only [splitOk, Bool.and_eq_true, decide_eq_true_eq] at h
  omega

-- § layer index bound

/-- `check_layer` refuses a layer outside the range. -/
def layerInRange (layer nLayers : Nat) : Bool := decide (layer < nLayers)

/-- The last valid layer is `n - 1`; `n` itself is refused. -/
theorem layer_n_refused (n : Nat) : layerInRange n n = false := by
  simp [layerInRange]

end Strata.LayerShape

/-!
# MODULE: session_layer_state
Per-layer state indexing, session arena alignment, doorbell sequence numbers and stage-timing arithmetic
of the 48-layer token session.

## Source semantics
* Layers with `l % 4 = 3` are QSA; the others are GDN. A GDN layer's state slice index is the number of
  GDN layers before it, so two layers never share a recurrence.
* The session arena is aligned to 256 bytes.
* Layer `l` of a token waits for ring number `l + 1`; the sequence strictly increases.
* Stage-prefix timings are cumulative; per-stage times are differences and telescope back to the full prefix.
* Event slots `l*6 + k` (k < 6) are pairwise distinct across layers.

## Form template identifiers
* `is_qsa_layer`, `gdn_point_at` → `countQsa`, `countGdn`
* `align_up(n, 256)` → `alignUp256`
* `want = ++expected` → `ringWant`
* `t[i] = t_{i+1} - t_i` → `stageDiff`
-/

namespace Strata.Session

-- § layer indexing

/-- Number of QSA layers strictly before layer `l`. -/
def countQsa : Nat → Nat
  | 0 => 0
  | l + 1 => countQsa l + (if l % 4 = 3 then 1 else 0)

/-- Number of GDN layers strictly before layer `l`. -/
def countGdn : Nat → Nat
  | 0 => 0
  | l + 1 => countGdn l + (if l % 4 = 3 then 0 else 1)

/-- Closed form for the QSA count. -/
theorem countQsa_eq (l : Nat) : countQsa l = l / 4 := by
  induction l with
  | zero => rfl
  | succ l ih =>
    simp only [countQsa]
    split <;> omega

/-- Every earlier layer is either GDN or QSA. -/
theorem countGdn_add_countQsa (l : Nat) : countGdn l + countQsa l = l := by
  induction l with
  | zero => rfl
  | succ l ih =>
    simp only [countGdn, countQsa]
    split <;> omega

/-- The real model: 12 QSA layers among 48. -/
theorem countQsa_48 : countQsa 48 = 12 := by decide

/-- The real model: 36 GDN layers among 48. -/
theorem countGdn_48 : countGdn 48 = 36 := by decide

/-- A GDN layer's slot index is strictly increasing: successive GDN layers get distinct slices. -/
theorem countGdn_strict (l : Nat) (h : l % 4 ≠ 3) : countGdn l < countGdn (l + 1) := by
  simp only [countGdn]
  split <;> omega

-- § arena alignment

/-- Round up to a multiple of 256. -/
def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Session arena sizes are multiples of 256. -/
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by
  unfold alignUp256; omega

/-- Alignment never shrinks the arena. -/
theorem le_alignUp256 (n : Nat) : n ≤ alignUp256 n := by
  unfold alignUp256; omega

/-- Alignment is idempotent. -/
theorem alignUp256_idem (n : Nat) : alignUp256 (alignUp256 n) = alignUp256 n := by
  unfold alignUp256; omega

-- § doorbell ring numbers

/-- Ring number layer `l` waits for. -/
def ringWant (l : Nat) : Nat := l + 1

/-- Ring numbers strictly increase with the layer. -/
theorem ringWant_strict (l : Nat) : ringWant l < ringWant (l + 1) := by
  unfold ringWant; omega

/-- A token of `n` layers needs rings `1..n`, so the last wanted ring equals the layer count. -/
theorem ringWant_last (n : Nat) (h : 0 < n) : ringWant (n - 1) = n := by
  unfold ringWant; omega

-- § stage timing

/-- Per-stage time from consecutive cumulative prefix times. -/
def stageDiff (hi lo : Int) : Int := hi - lo

/-- The five per-stage times add up to the full-prefix time. -/
theorem stage_telescope (t1 t2 t3 t4 t5 : Int) :
    t1 + stageDiff t2 t1 + stageDiff t3 t2 + stageDiff t4 t3 + stageDiff t5 t4 = t5 := by
  unfold stageDiff; omega

/-- The mixer (stages 0..2) is the third prefix time. -/
theorem mixer_is_prefix3 (t1 t2 t3 : Int) : t1 + stageDiff t2 t1 + stageDiff t3 t2 = t3 := by
  unfold stageDiff; omega

-- § event slots

/-- Event slot of layer `l`, mark `k`. -/
def evSlot (l k : Nat) : Nat := l * 6 + k

/-- Slots stay inside the `n * 6` event vector. -/
theorem evSlot_lt (n l k : Nat) (hl : l < n) (hk : k < 6) : evSlot l k < n * 6 := by
  unfold evSlot; omega

/-- Distinct (layer, mark) pairs get distinct slots. -/
theorem evSlot_inj (l k l' k' : Nat) (hk : k < 6) (hk' : k' < 6) (h : evSlot l k = evSlot l' k') :
    l = l' ∧ k = k' := by
  unfold evSlot at h; omega

end Strata.Session

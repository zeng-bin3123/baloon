/-!
# MODULE: session_graph_loop
Per-layer state indexing and timing arithmetic of the 48-layer session: how many GDN/QSA layers precede a layer,
arena alignment, ring sequence numbers, and the stage-prefix telescoping used by the profiler.

## Source semantics
A layer `l` is QSA iff `l % 4 = 3`; the others are GDN. Each GDN layer owns the slice `gdn_index` of the GDN state
arena where `gdn_index` is the number of GDN layers before it. The doorbell of layer `l` must reach sequence `l+1`.
Stage-prefix timings are differences of cumulative prefix times, so they telescope back to the full-prefix time.

## Form template identifiers
- isQsa ← `is_qsa_layer`, gdnBefore ← `gdn_index` computation in `gdn_point_at`,
- alignUp ← `align_up`, prefixStage ← `t[k] = t(k+1) - t(k)` in `session_replay_stage_prefixes`.
-/

namespace Strata.Session

-- § layer classification

/-- QSA layers are those with `l % 4 = 3`. -/
def isQsa (l : Nat) : Bool := l % 4 == 3

/-- Number of GDN layers strictly before layer `l`. -/
def gdnBefore : Nat → Nat
  | 0 => 0
  | l + 1 => gdnBefore l + (if l % 4 = 3 then 0 else 1)

/-- Number of QSA layers strictly before layer `l`. -/
def qsaBefore : Nat → Nat
  | 0 => 0
  | l + 1 => qsaBefore l + (if l % 4 = 3 then 1 else 0)

/-- Closed form: `l - l/4` GDN layers precede layer `l`. -/
theorem gdnBefore_eq (l : Nat) : gdnBefore l = l - l / 4 := by
  induction l with
  | zero => simp [gdnBefore]
  | succ n ih =>
    simp only [gdnBefore]
    split <;> omega

/-- Closed form: `l / 4` QSA layers precede layer `l`. -/
theorem qsaBefore_eq (l : Nat) : qsaBefore l = l / 4 := by
  induction l with
  | zero => simp [qsaBefore]
  | succ n ih =>
    simp only [qsaBefore]
    split <;> omega

/-- Every earlier layer is GDN or QSA, exactly once. -/
theorem layers_partition (l : Nat) : gdnBefore l + qsaBefore l = l := by
  rw [gdnBefore_eq, qsaBefore_eq]; omega

/-- A 48-layer model has 36 GDN and 12 QSA layers. -/
theorem counts_48 : gdnBefore 48 = 36 ∧ qsaBefore 48 = 12 := by
  rw [gdnBefore_eq, qsaBefore_eq]; omega

/-- Consecutive GDN layers get consecutive state slices: a GDN layer's index is one more than the previous
GDN layer's, so no two layers share a slice. -/
theorem gdn_slices_distinct (l : Nat) (h : l % 4 ≠ 3) : gdnBefore (l + 1) = gdnBefore l + 1 := by
  simp only [gdnBefore]
  rw [if_neg h]

/-- A GDN layer's slice index is below the total GDN count for a model of `n` layers. -/
theorem gdn_index_lt (n l : Nat) (hl : l < n) (h : l % 4 ≠ 3) : gdnBefore l < n - n / 4 := by
  rw [gdnBefore_eq]; omega

-- § alignment

/-- `align_up(n, 256)`. -/
def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Aligned sizes hold the request. -/
theorem alignUp256_ge (n : Nat) : n ≤ alignUp256 n := by
  unfold alignUp256; omega

/-- Aligned sizes are multiples of 256. -/
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by
  unfold alignUp256; omega

/-- Aligned sizes waste fewer than 256 bytes. -/
theorem alignUp256_lt (n : Nat) : alignUp256 n < n + 256 := by
  unfold alignUp256; omega

-- § doorbell sequencing

/-- The sequence value the host waits for at layer `l` (`want = l + 1`). -/
def ringWant (l : Nat) : Nat := l + 1

/-- Ring numbers strictly increase across layers. -/
theorem ringWant_strict {l m : Nat} (h : l < m) : ringWant l < ringWant m := by
  unfold ringWant; omega

/-- The last layer rings exactly `n_layers`. -/
theorem ringWant_last (n : Nat) (h : 1 ≤ n) : ringWant (n - 1) = n := by
  unfold ringWant; omega

-- § profiler arithmetic

/-- Event slot of layer `l`, prefix `k` (0..5) in the 6-per-layer event table. -/
def eventSlot (l k : Nat) : Nat := l * 6 + k

/-- Event slots stay inside the `6n` table. -/
theorem eventSlot_lt (n l k : Nat) (hl : l < n) (hk : k < 6) : eventSlot l k < n * 6 := by
  unfold eventSlot; omega

/-- Event slots of distinct (layer, prefix) pairs differ. -/
theorem eventSlot_inj {l k l' k' : Nat} (hk : k < 6) (hk' : k' < 6)
    (h : eventSlot l k = eventSlot l' k') : l = l' ∧ k = k' := by
  unfold eventSlot at h; omega

/-- The five per-stage times telescope: their sum is the cumulative prefix-5 time. -/
theorem prefix_telescope (t1 t2 t3 t4 t5 : Int) :
    t1 + (t2 - t1) + (t3 - t2) + (t4 - t3) + (t5 - t4) = t5 := by
  omega

/-- Four events per layer index without overlap. -/
theorem stage_event_lt (n l j : Nat) (hl : l < n) (hj : j < 4) : l * 4 + j < n * 4 := by
  omega

end Strata.Session

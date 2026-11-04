/-!
# MODULE: expert_cache_slot_arena
The VRAM slot arena and residency table for routed experts: per-layer slot ranges, a no-eviction admit rule,
256-byte aligned variable-size slots, and the profile-file header check.

## Source semantics
* Layer `l` owns `[l*q, (l+1)*q)` with `q = slots / n_layers`; the last layer takes the remainder, so every
  range lies inside `0..slots`.
* `admit` never evicts: it hands out the next counter value while it is below the ceiling, else refuses.
* Slot sizes are rounded up to 256 bytes and offsets are prefix sums of the rounded sizes.
* A profile is accepted only when its geometry matches and it claims no more ranked pairs than slots.

## Form template identifiers
* `ExpertCache::layer_slot_range` -> `layerLo` / `layerHi`
* `ExpertCache::admit` (per-layer path) -> `admitPL`; (global path) -> `admitGlobal`
* `ExpertCache::open_sized` offsets -> `offAt`, `align256`
* `read_expert_profile` header guard -> `profileOk`
-/

namespace Strata.ExpertCache

-- § layer ranges

/-- Per-layer quota `q = slots / n_layers`. -/
def quota (slots nLayers : Nat) : Nat := slots / nLayers

/-- Lower bound of layer `l`'s slot range. -/
def layerLo (slots nLayers l : Nat) : Nat := l * quota slots nLayers

/-- Upper bound of layer `l`'s slot range; the last layer absorbs the remainder. -/
def layerHi (slots nLayers l : Nat) : Nat :=
  if l = nLayers - 1 then slots else (l + 1) * quota slots nLayers

/-- `n * q` never exceeds the slot count. -/
theorem nq_le_slots (slots n : Nat) : n * quota slots n ≤ slots := by
  unfold quota
  rw [Nat.mul_comm]
  exact Nat.div_mul_le_self slots n

/-- Every layer's range is well formed: `lo ≤ hi`. -/
theorem lo_le_hi (slots n l : Nat) (hl : l < n) :
    layerLo slots n l ≤ layerHi slots n l := by
  have h1 : l * quota slots n ≤ n * quota slots n := Nat.mul_le_mul_right _ (Nat.le_of_lt hl)
  have h2 := nq_le_slots slots n
  unfold layerLo layerHi
  by_cases h : l = n - 1
  · simp [h]
    rw [h] at h1
    omega
  · simp [h]
    exact Nat.mul_le_mul_right _ (Nat.le_succ l)

/-- Ranges never leave the arena: `hi ≤ slots`. -/
theorem hi_le_slots (slots n l : Nat) (hl : l < n) : layerHi slots n l ≤ slots := by
  have h2 := nq_le_slots slots n
  unfold layerHi
  by_cases h : l = n - 1
  · simp [h]
  · simp [h]
    have h1 : (l + 1) * quota slots n ≤ n * quota slots n := Nat.mul_le_mul_right _ hl
    omega

/-- Adjacent ranges abut exactly: no slot is orphaned between layers. -/
theorem hi_eq_next_lo (slots n l : Nat) (hl : l + 1 < n) :
    layerHi slots n l = layerLo slots n (l + 1) := by
  have h : l ≠ n - 1 := by omega
  simp [layerHi, layerLo, h]

/-- The first layer starts at slot 0. -/
theorem lo_zero (slots n : Nat) : layerLo slots n 0 = 0 := by
  simp [layerLo]

/-- The last layer ends exactly at `slots`. -/
theorem hi_last (slots n : Nat) (hn : 0 < n) : layerHi slots n (n - 1) = slots := by
  simp [layerHi]

/-- Every layer owns at least `q` slots. -/
theorem width_ge_quota (slots n l : Nat) (hl : l < n) :
    quota slots n ≤ layerHi slots n l - layerLo slots n l := by
  have h2 := nq_le_slots slots n
  have h1 : (l + 1) * quota slots n ≤ n * quota slots n := Nat.mul_le_mul_right _ hl
  have h3 : (l + 1) * quota slots n = l * quota slots n + quota slots n := Nat.succ_mul l _
  by_cases h : l = n - 1
  · have hn : n = l + 1 := by omega
    subst hn
    unfold layerLo layerHi
    simp
    omega
  · unfold layerLo layerHi
    simp [h]
    omega

-- § admit

/-- Per-layer admit counters. -/
structure PerLayer where
  slots : Nat
  nLayers : Nat
  next : Nat → Nat

/-- Per-layer admit: hand out the next counter value while below the layer's ceiling. -/
def admitPL (c : PerLayer) (l : Nat) : Option Nat :=
  if layerHi c.slots c.nLayers l ≤ c.next l then none else some (c.next l)

/-- An admitted slot is below the layer's ceiling. -/
theorem admitPL_lt_hi (c : PerLayer) (l r : Nat) (h : admitPL c l = some r) :
    r < layerHi c.slots c.nLayers l := by
  unfold admitPL at h
  by_cases hc : layerHi c.slots c.nLayers l ≤ c.next l
  · simp [hc] at h
  · simp [hc] at h
    omega

/-- An admitted slot is a valid arena index, so `device_slot` cannot walk off the end. -/
theorem admitPL_lt_slots (c : PerLayer) (l r : Nat) (hl : l < c.nLayers)
    (h : admitPL c l = some r) : r < c.slots := by
  have h1 := admitPL_lt_hi c l r h
  have h2 := hi_le_slots c.slots c.nLayers l hl
  omega

/-- A layer whose counter starts inside its range stays inside it. -/
theorem admitPL_ge_lo (c : PerLayer) (l r : Nat)
    (hlo : layerLo c.slots c.nLayers l ≤ c.next l) (h : admitPL c l = some r) :
    layerLo c.slots c.nLayers l ≤ r := by
  unfold admitPL at h
  by_cases hc : layerHi c.slots c.nLayers l ≤ c.next l
  · simp [hc] at h
  · simp [hc] at h
    omega

/-- Global (single-counter) admit. -/
def admitGlobal (slots nextFree : Nat) : Option Nat :=
  if slots ≤ nextFree then none else some nextFree

/-- The global admit never exceeds the arena. -/
theorem admitGlobal_lt (slots nf r : Nat) (h : admitGlobal slots nf = some r) : r < slots := by
  unfold admitGlobal at h
  by_cases hc : slots ≤ nf
  · simp [hc] at h
  · simp [hc] at h
    omega

/-- A full global arena refuses (no eviction). -/
theorem admitGlobal_full (slots nf : Nat) (h : slots ≤ nf) : admitGlobal slots nf = none := by
  simp [admitGlobal, h]

-- § aligned slots

/-- Round up to a multiple of 256. -/
def align256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Aligned sizes are multiples of 256. -/
theorem align256_mod (n : Nat) : align256 n % 256 = 0 := by
  unfold align256; omega

/-- Alignment never shrinks. -/
theorem le_align256 (n : Nat) : n ≤ align256 n := by
  unfold align256; omega

/-- Alignment adds fewer than 256 bytes. -/
theorem align256_lt (n : Nat) : align256 n < n + 256 := by
  unfold align256; omega

/-- Offset of slot `i` = sum of the aligned sizes of slots before it. -/
def offAt : List Nat → Nat → Nat
  | [], _ => 0
  | _ :: _, 0 => 0
  | b :: bs, i + 1 => align256 b + offAt bs i

/-- Offset of slot 0 is 0. -/
theorem offAt_zero (bs : List Nat) : offAt bs 0 = 0 := by
  cases bs <;> simp [offAt]

/-- Every slot offset is 256-aligned, so vector loads start where the kernels expect. -/
theorem offAt_mod (bs : List Nat) (i : Nat) : offAt bs i % 256 = 0 := by
  induction bs generalizing i with
  | nil => simp [offAt]
  | cons b bs ih =>
    cases i with
    | zero => simp [offAt]
    | succ i =>
      have h1 := align256_mod b
      have h2 := ih i
      simp only [offAt]
      omega

/-- Offsets are monotone. -/
theorem offAt_mono (bs : List Nat) (i : Nat) : offAt bs i ≤ offAt bs (i + 1) := by
  induction bs generalizing i with
  | nil => simp [offAt]
  | cons b bs ih =>
    cases i with
    | zero => simp [offAt]
    | succ i =>
      have h := ih i
      simp only [offAt]
      omega

/-- Slot `i` fits before slot `i+1`: its blob never overlaps the next one. -/
theorem offAt_step (bs : List Nat) (i s : Nat) (h : bs[i]? = some s) :
    offAt bs i + s ≤ offAt bs (i + 1) := by
  induction bs generalizing i with
  | nil => simp at h
  | cons b bs ih =>
    cases i with
    | zero =>
      simp at h
      have h1 := le_align256 b
      have h2 := offAt_zero bs
      simp only [offAt]
      omega
    | succ i =>
      simp at h
      have h3 := ih i h
      simp only [offAt]
      omega

-- § profile header

/-- The header guard of `read_expert_profile`: same geometry, not more ranked pairs than slots. -/
def profileOk (nl ne want nRanked nLayers nExpert : Nat) : Bool :=
  nl == nLayers && ne == nExpert && nRanked ≤ want

/-- A profile for a different artifact is refused. -/
theorem profile_geometry (nl ne want nr L E : Nat) (h : profileOk nl ne want nr L E = true) :
    nl = L ∧ ne = E := by
  unfold profileOk at h
  simp at h
  exact ⟨h.1.1, h.1.2⟩

/-- An accepted profile never claims more pairs than slots. -/
theorem profile_ranked_le (nl ne want nr L E : Nat) (h : profileOk nl ne want nr L E = true) :
    nr ≤ want := by
  unfold profileOk at h
  simp at h
  exact h.2

end Strata.ExpertCache

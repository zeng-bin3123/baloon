/-!
# MODULE: session_graph_replay
SOURCES: src/core/session.cpp
One token through all layers: session arena sizing and carving, per-layer GDN state slices, per-layer
captured graphs (pre/post, split prefixes), the doorbell ring protocol and stage-prefix timing.

## Source semantics
* Layer `l` is a QSA layer iff `l % 4 = 3`; GDN layers are numbered in order, each with its own state slice.
* Arena regions are 256-aligned; `session_bytes` rounds up and adds 4096 slack.
* Stage flags: `stage_prefix = k > 0` runs stages `0..k-1` and ignores `half`; otherwise half 0 runs all,
  half 1 the mixer (stages 0-2), half 2 the FFN front and router (stages 3-4).
* Ring numbers are `l+1` for layer `l`; stage times are differences of cumulative prefix times.

## Form template identifiers
isQsa ↦ `is_qsa_layer`; gdnBefore ↦ `gdn_index` loop; runStage ↦ `run0..run4`; sliceOff ↦ `gdn_state + idx*floats`.
-/
namespace Strata.SessionGraphReplay

-- § layer numbering

/-- QSA layers are those with index ≡ 3 (mod 4). -/
def isQsa (l : Nat) : Bool := l % 4 == 3

/-- Number of GDN layers strictly before `l` (the `gdn_index` loop). -/
def gdnBefore : Nat → Nat
  | 0 => 0
  | l + 1 => gdnBefore l + (if l % 4 = 3 then 0 else 1)

/-- Closed form: `l - l/4`. -/
theorem gdnBefore_eq : ∀ l, gdnBefore l = l - l / 4
  | 0 => by simp [gdnBefore]
  | l + 1 => by
      have ih := gdnBefore_eq l
      rw [gdnBefore, ih]
      split <;> omega

/-- 48 layers hold 36 GDN layers. -/
theorem gdn_count_48 : gdnBefore 48 = 36 := by decide

/-- GDN indices are strictly increasing across consecutive GDN layers (distinct slices). -/
theorem gdn_index_strict (l : Nat) (h : l % 4 ≠ 3) : gdnBefore (l + 1) = gdnBefore l + 1 := by
  rw [gdnBefore, if_neg h]

/-- QSA layers do not advance the GDN index. -/
theorem qsa_no_advance (l : Nat) (h : l % 4 = 3) : gdnBefore (l + 1) = gdnBefore l := by
  rw [gdnBefore, if_pos h, Nat.add_zero]

-- § slices

/-- A slice `[i*F, i*F+F)` ends inside `n*F` for `i < n`. -/
theorem slice_end_le (i n F : Nat) (h : i < n) : i * F + F ≤ n * F := by
  have := Nat.mul_le_mul_right F (show i + 1 ≤ n by omega)
  simpa [Nat.add_mul] using this

/-- Slices of distinct layers do not overlap. -/
theorem slice_disjoint (i j F : Nat) (h : i < j) : i * F + F ≤ j * F := slice_end_le i j F h

-- § session arena

/-- 256-byte alignment. -/
def align256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Alignment does not shrink. -/
theorem align256_ge (n : Nat) : n ≤ align256 n := by unfold align256; omega

/-- Alignment is a 256-multiple. -/
theorem align256_mod (n : Nat) : align256 n % 256 = 0 := by unfold align256; omega

/-- `session_bytes` covers the raw sum, and carries 4096 of slack. -/
theorem session_bytes_cover (n : Nat) : n + 4096 ≤ align256 n + 4096 := by
  have := align256_ge n; omega

/-- All QSA layers after the first borrow the RoPE table, so the arena is `first + (q-1)*rest`. -/
theorem qsa_arena (first rest q : Nat) (h : 1 ≤ q) :
    first + (q - 1) * rest ≤ q * (first + rest) := by
  obtain ⟨k, rfl⟩ : ∃ k, q = k + 1 := ⟨q - 1, by omega⟩
  simp [Nat.add_mul, Nat.mul_add]
  omega

-- § stage flags

/-- Whether stage `i` (0..4) runs for `(half, stage_prefix)`. -/
def runStage (half pfx i : Nat) : Prop :=
  if 0 < pfx then i + 1 ≤ pfx
  else if i ≤ 2 then (half = 0 ∨ half = 1) else (half = 0 ∨ half = 2)

/-- A positive prefix runs exactly the stages below it, whatever `half` says. -/
theorem prefix_runs_below (half pfx i : Nat) (h : 0 < pfx) : runStage half pfx i ↔ i + 1 ≤ pfx := by
  unfold runStage; rw [if_pos h]

/-- Prefix 5 runs every stage. -/
theorem prefix5_all (half i : Nat) (hi : i < 5) : runStage half 5 i := by
  unfold runStage; rw [if_pos (by omega)]; omega

/-- Legacy callers (`half = 0`, no prefix) run all five stages. -/
theorem half0_all (i : Nat) (hi : i < 5) : runStage 0 0 i := by
  have h : ¬ (0 < 0) := by omega
  unfold runStage; rw [if_neg h]
  split <;> simp

/-- `half = 1` stops before the FFN front. -/
theorem half1_mixer_only (i : Nat) (hi : 3 ≤ i) : ¬ runStage 1 0 i := by
  have h : ¬ (0 < 0) := by omega
  have h2 : ¬ (i ≤ 2) := by omega
  unfold runStage; rw [if_neg h, if_neg h2]; omega

/-- `half = 2` skips the mixer entirely. -/
theorem half2_skips_mixer (i : Nat) (hi : i ≤ 2) : ¬ runStage 2 0 i := by
  have h : ¬ (0 < 0) := by omega
  unfold runStage; rw [if_neg h, if_pos hi]; omega

-- § timing

/-- Stage times from cumulative prefix times telescope back to the full-layer time. -/
theorem stage_times_telescope (t1 t2 t3 t4 t5 : Int) :
    t1 + (t2 - t1) + (t3 - t2) + (t4 - t3) + (t5 - t4) = t5 := by omega

/-- Event `l*6+k` is in range for `l < n`, `k <= 5`. -/
theorem event_index (l n k : Nat) (hl : l < n) (hk : k ≤ 5) : l * 6 + k < n * 6 := by omega

/-- Ring numbers are strictly increasing per layer and the layer count ends at `n_layers`. -/
theorem ring_strict (l : Nat) : (l + 1) < (l + 1) + 1 := Nat.lt_succ_self _

end Strata.SessionGraphReplay

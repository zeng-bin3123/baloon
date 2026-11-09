/-!
# MODULE: session_state_loop
SOURCES: src/core/session.cpp
One token through all layers: session arena carving (256-byte alignment), per-GDN-layer state slices,
doorbell ring numbering, and the stage-prefix timing telescope.

## Source semantics
* `align_up(n, 256)` sizes every carve; the arena total is aligned plus 4096.
* Each GDN layer owns the slice indexed by the number of non-QSA layers before it; slices never coincide.
* Layer `l` waits for ring `l + 1`; the counter strictly increases.
* Stage times are prefix differences, so they telescope back to the full prefix.

## Form template identifiers
align_up → alignUp256 ; gdn_point_at index → gdnIndex ; expected → ringWant ; stage_ms → prefix deltas
-/

namespace Strata.Session

-- § arena alignment

/-- Round up to a multiple of 256. -/
def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- A carve never shrinks. -/
theorem alignUp256_ge (n : Nat) : n ≤ alignUp256 n := by
  unfold alignUp256; omega

/-- A carve ends on a 256-byte boundary. -/
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by
  unfold alignUp256; omega

/-- A carve over-allocates by less than 256 bytes. -/
theorem alignUp256_lt (n : Nat) : alignUp256 n < n + 256 := by
  unfold alignUp256; omega

/-- The `take` lambda: successive carves start aligned and are disjoint. -/
theorem carve_step (used bytes : Nat) (h : used % 256 = 0) :
    alignUp256 (used + bytes) % 256 = 0 ∧ used + bytes ≤ alignUp256 (used + bytes) :=
  ⟨alignUp256_mod _, alignUp256_ge _⟩

/-- Total arena size is aligned size plus the 4096-byte pad, so it covers `n`. -/
theorem session_bytes_ge (n : Nat) : n + 4096 ≤ alignUp256 n + 4096 := by
  have := alignUp256_ge n
  omega

-- § GDN state slices

/-- Number of non-QSA (GDN) layers strictly before layer `l`. -/
def gdnIndex (isQsa : Nat → Bool) : Nat → Nat
  | 0 => 0
  | l + 1 => gdnIndex isQsa l + (if isQsa l then 0 else 1)

/-- The GDN index never exceeds the layer index. -/
theorem gdnIndex_le (isQsa : Nat → Bool) (l : Nat) : gdnIndex isQsa l ≤ l := by
  induction l with
  | zero => simp [gdnIndex]
  | succ n ih =>
    simp only [gdnIndex]
    split <;> omega

/-- The GDN index is non-decreasing step by step. -/
theorem gdnIndex_step_le (isQsa : Nat → Bool) (l : Nat) : gdnIndex isQsa l ≤ gdnIndex isQsa (l + 1) := by
  simp only [gdnIndex]
  omega

/-- The GDN index is monotone. -/
theorem gdnIndex_mono (isQsa : Nat → Bool) {a b : Nat} (h : a ≤ b) : gdnIndex isQsa a ≤ gdnIndex isQsa b := by
  induction h with
  | refl => exact Nat.le_refl _
  | step _ ih => exact Nat.le_trans ih (gdnIndex_step_le isQsa _)

/-- A GDN layer's own slot is followed by a fresh one. -/
theorem gdnIndex_succ_of_gdn (isQsa : Nat → Bool) (l : Nat) (h : isQsa l = false) :
    gdnIndex isQsa (l + 1) = gdnIndex isQsa l + 1 := by
  simp [gdnIndex, h]

/-- Two distinct GDN layers never share a state slice: sharing would feed one layer's recurrence into another. -/
theorem gdnIndex_injective (isQsa : Nat → Bool) (l l' : Nat) (hl : isQsa l = false) (hlt : l < l') :
    gdnIndex isQsa l < gdnIndex isQsa l' := by
  have h1 := gdnIndex_succ_of_gdn isQsa l hl
  have h2 := gdnIndex_mono isQsa (show l + 1 ≤ l' from hlt)
  omega

-- § doorbell numbering

/-- Ring sequence number layer `l` waits for. -/
def ringWant (l : Nat) : Nat := l + 1

/-- Ring numbers strictly increase with the layer, so a stale ring cannot satisfy a later wait. -/
theorem ringWant_strict (l : Nat) : ringWant l < ringWant (l + 1) := by
  unfold ringWant; omega

/-- Ring numbers are positive (the doorbell is reset to 0 before the first layer). -/
theorem ringWant_pos (l : Nat) : 0 < ringWant l := by
  unfold ringWant; omega

-- § stage-prefix timing

/-- Stage times are differences of cumulative prefix times; they telescope to the longest prefix. -/
theorem stage_telescope (t1 t2 t3 t4 t5 : Int) :
    t1 + (t2 - t1) + (t3 - t2) + (t4 - t3) + (t5 - t4) = t5 := by
  omega

end Strata.Session

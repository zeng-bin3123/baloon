/-!
# MODULE: session_loop_indexing
Per-layer state indexing, arena sizing and doorbell sequence numbers of the session loop.

## Source semantics
Layer `l` is a QSA layer iff `l % 4 == 3`. GDN and QSA layers are numbered separately: `gdn_point_at` counts the
non-QSA layers before `l`. `session_bytes` rounds up to 256. Layer `l` waits for ring number `l+1`. The staged-prefix
stage times are differences of cumulative prefix times and telescope.

## Form template identifiers
- gdn_point_at / session_token index counters -> gdnIdx / qsaIdx
- align_up                                    -> alignUp
- session_loop `want = ++expected`            -> want
- session_replay_stage_prefixes               -> telescope
-/
namespace Strata.SessionLoopIndexing

def gdnIdx : Nat → Nat
  | 0 => 0
  | l + 1 => gdnIdx l + (if l % 4 == 3 then 0 else 1)
def qsaIdx : Nat → Nat
  | 0 => 0
  | l + 1 => qsaIdx l + (if l % 4 == 3 then 1 else 0)

/-- Every layer before `l` is exactly one of GDN or QSA. -/
theorem gdn_add_qsa : ∀ l, gdnIdx l + qsaIdx l = l
  | 0 => rfl
  | l + 1 => by
    have := gdn_add_qsa l
    simp only [gdnIdx, qsaIdx]
    split <;> omega

/-- The QSA index of layer `l` is `l / 4`. -/
theorem qsaIdx_eq : ∀ l, qsaIdx l = l / 4
  | 0 => rfl
  | l + 1 => by
    have := qsaIdx_eq l
    simp only [qsaIdx]
    split <;> simp_all <;> omega

/-- The GDN index of layer `l` is `l - l/4`. -/
theorem gdnIdx_eq (l : Nat) : gdnIdx l = l - l / 4 := by
  have h1 := gdn_add_qsa l
  have h2 := qsaIdx_eq l
  omega

/-- In a 48-layer model every GDN layer's state slot is among the 36 allocated. -/
theorem gdn_slot_in_range {l : Nat} (hl : l < 48) (hg : l % 4 ≠ 3) : gdnIdx l < 36 := by
  rw [gdnIdx_eq]; omega

/-- Every QSA layer's state slot is among the 12 allocated. -/
theorem qsa_slot_in_range {l : Nat} (hl : l < 48) : qsaIdx l < 12 := by
  rw [qsaIdx_eq]; omega

/-- Two distinct GDN layers never share a state slot. -/
theorem gdn_slots_distinct {l l' : Nat} (h : l < l') (hl : l % 4 ≠ 3) (hl' : l' % 4 ≠ 3) :
    gdnIdx l ≠ gdnIdx l' := by
  rw [gdnIdx_eq, gdnIdx_eq]; omega

def alignUp (n a : Nat) : Nat := (n + a - 1) / a * a

/-- Session arena rounding never shrinks. -/
theorem alignUp_ge (n a : Nat) (ha : 0 < a) : n ≤ alignUp n a := by
  unfold alignUp
  have h := Nat.div_add_mod (n + a - 1) a
  have h2 := Nat.mod_lt (n + a - 1) ha
  have h3 := Nat.mul_comm ((n + a - 1) / a) a
  omega

/-- Session arena rounding is a multiple of `a`. -/
theorem alignUp_mod (n a : Nat) : alignUp n a % a = 0 := by unfold alignUp; simp

def want (expected : Nat) : Nat := expected + 1

/-- Layer `l` (0-based) waits for ring `l+1`, matching the doorbell's monotone counter. -/
theorem want_layer (l : Nat) : want l = l + 1 := rfl

/-- Per-stage times derived from cumulative prefix times sum back to the full prefix. -/
theorem telescope (t1 t2 t3 t4 t5 : Int) : t1 + (t2 - t1) + (t3 - t2) + (t4 - t3) + (t5 - t4) = t5 := by omega

end Strata.SessionLoopIndexing

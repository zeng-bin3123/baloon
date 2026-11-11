/-!
# MODULE: session_token_pipeline
SOURCES: src/core/session.cpp
One token through all 48 layers: the session arena sizing, the per-layer GDN state slices
(`gdn_point_at`), the QSA/GDN layer interleave, the doorbell sequence the host loop waits on
(`session_loop`, `session_run_token`), and the prefix-differencing stage table
(`session_replay_stage_prefixes`).

## Source semantics
* Layer `l` is QSA iff `l % 4 = 3`; the remaining layers are GDN.  Each GDN layer owns its own
  recurrent + conv state slice at `gdn_index * gdn_state_floats`; slices of distinct GDN layers are
  disjoint, which is what keeps one layer's recurrence from leaking into the next.
* `gdn_index(l)` counts the GDN layers strictly before `l`; with `qsa_index` it partitions `l`.
* The arena is `align_up(n, 256) + 4096` and therefore covers the raw sum `n`.
* The host waits for `seq ≥ l + 1` at layer `l`; the ring is monotone, so the l-th wait is satisfied by
  the l-th ring and the token completes after exactly `n_layers` rings.
* The five-stage table is built from cumulative prefix times t1..t5; the differences telescope back to
  t5, which is the check the source prints ("the five must account for it").

## Form template identifiers
isQsa ↔ is_qsa_layer · gdnIndex/qsaIndex ↔ gdn_point_at / qsa_index · sliceStart ↔ s.gdn_state + i*F ·
alignUp ↔ align_up · sessionArena ↔ session_bytes · ringSatisfied ↔ `*seq >= want` · stageDiffs ↔ t[0..4]
-/

namespace SessionTokenPipeline

-- § layer kinds

/-- `full_attention_interval = 4`: every fourth layer, index 3 mod 4, is a QSA layer. -/
def isQsa (l : Nat) : Bool := l % 4 == 3

/-- The number of GDN layers strictly before `l` (the slice index `gdn_point_at` computes). -/
def gdnIndex : Nat → Nat
  | 0 => 0
  | l + 1 => gdnIndex l + (if isQsa l then 0 else 1)

/-- The number of QSA layers strictly before `l` (the `qsa_index` counter of the capture loop). -/
def qsaIndex : Nat → Nat
  | 0 => 0
  | l + 1 => qsaIndex l + (if isQsa l then 1 else 0)

/-- Every layer before `l` is counted exactly once, as GDN or as QSA. -/
theorem index_partition (l : Nat) : gdnIndex l + qsaIndex l = l := by
  induction l with
  | zero => rfl
  | succ n ih =>
    simp only [gdnIndex, qsaIndex]
    by_cases h : isQsa n = true <;> simp [h] <;> omega

/-- The artifact's 48 layers split into 36 GDN and 12 QSA layers. -/
theorem artifact_split : gdnIndex 48 = 36 ∧ qsaIndex 48 = 12 := by
  decide

/-- The GDN index is monotone: a later layer never gets an earlier slice. -/
theorem gdnIndex_mono (l : Nat) : gdnIndex l ≤ gdnIndex (l + 1) := by
  simp only [gdnIndex]
  omega

/-- Two distinct GDN layers get distinct slice indices: the index strictly grows past a GDN layer. -/
theorem gdnIndex_step (l : Nat) (h : isQsa l = false) : gdnIndex (l + 1) = gdnIndex l + 1 := by
  simp [gdnIndex, h]

/-- The GDN index never exceeds the layer index. -/
theorem gdnIndex_le (l : Nat) : gdnIndex l ≤ l := by
  have := index_partition l
  omega

-- § per-layer state slices

/-- Start of GDN slice `i` in floats. -/
def sliceStart (F i : Nat) : Nat := i * F

/-- Slices of distinct GDN indices are disjoint: slice `i` ends before slice `j` starts. -/
theorem slice_disjoint (F i j : Nat) (h : i < j) : sliceStart F i + F ≤ sliceStart F j := by
  unfold sliceStart
  have h1 : (i + 1) * F ≤ j * F := Nat.mul_le_mul_right F h
  rw [Nat.add_mul, Nat.one_mul] at h1
  exact h1

/-- The last GDN slice ends inside the `n_gdn * F` floats `session_init` carves. -/
theorem slice_in_carve (F i n : Nat) (h : i < n) : sliceStart F i + F ≤ n * F := by
  have := slice_disjoint F i n h
  unfold sliceStart at this
  exact this

/-- The conv state sits after the recurrent state inside the same slice. -/
theorem conv_after_state (F S C i : Nat) (hF : F = S + C) :
    sliceStart F i + S + C = sliceStart F (i + 1) := by
  unfold sliceStart
  rw [Nat.add_mul, Nat.one_mul, hF]
  omega

-- § arena sizing

/-- Round up to a multiple of 256 (`align_up(n, 256)`). -/
def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- The session arena request (`session_bytes`). -/
def sessionArena (n : Nat) : Nat := alignUp256 n + 4096

/-- The arena covers every region the init carves, with 4 KiB of slack. -/
theorem sessionArena_covers (n : Nat) : n + 4096 ≤ sessionArena n := by
  unfold sessionArena alignUp256; omega

/-- The aligned size is a multiple of 256. -/
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by
  unfold alignUp256; omega

-- § the doorbell sequence

/-- The host's wait at layer `l` is satisfied once the ring counter reaches `l + 1`. -/
def ringSatisfied (seq l : Nat) : Prop := l + 1 ≤ seq

/-- A monotone ring: if layer `l`'s wait was satisfied, so is every earlier layer's. -/
theorem ring_monotone (seq l l' : Nat) (hl : l' ≤ l) (h : ringSatisfied seq l) :
    ringSatisfied seq l' := by
  unfold ringSatisfied at *; omega

/-- After the last layer has rung, `seq = n_layers` satisfies every layer's wait. -/
theorem all_rang (n l : Nat) (hl : l < n) : ringSatisfied n l := by
  unfold ringSatisfied; omega

/-- A ring counter still below `l + 1` means layer `l` has not published: the host must not read it. -/
theorem not_rung (seq l : Nat) (h : seq < l + 1) : ¬ ringSatisfied seq l := by
  unfold ringSatisfied; omega

-- § the five-stage table

/-- Per-stage times from cumulative prefix times (`t[0] = t1`, `t[k] = t_{k+1} - t_k`). -/
def stageDiffs (t1 t2 t3 t4 t5 : Int) : List Int := [t1, t2 - t1, t3 - t2, t4 - t3, t5 - t4]

/-- Sum of a list of integers. -/
def isum : List Int → Int
  | [] => 0
  | x :: xs => x + isum xs

/-- The five differences telescope to the full-prefix time: the table accounts for the whole graph. -/
theorem stages_telescope (t1 t2 t3 t4 t5 : Int) : isum (stageDiffs t1 t2 t3 t4 t5) = t5 := by
  simp only [stageDiffs, isum]
  omega

/-- The mixer is stages 0..2, i.e. exactly prefix 3, which is why `mixer_per_layer` reads `c`. -/
theorem mixer_is_prefix3 (t1 t2 t3 : Int) : t1 + (t2 - t1) + (t3 - t2) = t3 := by
  omega

end SessionTokenPipeline

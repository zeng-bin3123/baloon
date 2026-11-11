/-!
# MODULE: session_token_loop
SOURCES: src/core/session.cpp
One token through all layers: the session arena (256-byte aligned regions, one GDN state slice per GDN
layer, one QSA state per QSA layer with a shared RoPE table), the per-layer graph capture, and the
doorbell protocol between the GPU's `pre[l]` and the host pool before `post[l]`.

## Source semantics
* `align_up(n, 256)` rounds every region; `session_bytes` adds a final 4096-byte slack.
* `gdn_point_at` gives GDN layer `l` the slice number `#{l' < l : l' is GDN}`; distinct GDN layers get
  disjoint slices of `gdn_state_floats` floats, all inside `n_gdn_layers * gdn_state_floats`.
* QSA state `i` lives at `0` for `i = 0` (it carries the RoPE table) and at `first + (i-1)*rest`
  otherwise; every state fits inside `first + (n_qsa - 1) * rest`.
* The doorbell: layer `l` rings sequence `l + 1`, the host acknowledges with flag `l + 1` after the pool;
  after `n` layers both counters read `n`.  The pool's answer for layer `l` is combined by layer `l`.
* `session_replay_stage_prefixes` reports stage times as differences of prefix times, which telescope.

## Form template identifiers
align256 ↦ align_up(·, SESSION_STATE_ALIGN); gdnIndex ↦ gdn_point_at's counter / session_token's
gdn_index; qsaOffset ↦ the `qp + ...` carve in session_init; Bell ↦ Doorbell (h_seq, h_flag);
runLayers ↦ session_run_token's host loop; dumpOffset ↦ dump_layers slot of session_loop.
-/

namespace Strata.Session

-- § 256-byte alignment

/-- `align_up(n, SESSION_STATE_ALIGN)` with the alignment fixed at 256. -/
def align256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Rounding never shrinks. -/
theorem align256_ge (n : Nat) : n ≤ align256 n := by unfold align256; omega

/-- The result is a multiple of 256. -/
theorem align256_mod (n : Nat) : align256 n % 256 = 0 := by unfold align256; omega

/-- `session_bytes` covers everything it summed. -/
theorem session_bytes_cover (n : Nat) : n + 4096 ≤ align256 n + 4096 := by
  have := align256_ge n; omega

-- § layer types and the GDN slice index

/-- QSA layers are the last of each group of four. -/
def isQsa (l : Nat) : Bool := decide (l % 4 = 3)

/-- `gdn_point_at`'s slice index: the number of GDN layers strictly before `l`. -/
def gdnIndex : Nat → Nat
  | 0 => 0
  | l + 1 => gdnIndex l + if isQsa l then 0 else 1

/-- The index never decreases with the layer. -/
theorem gdnIndex_mono {a b : Nat} (h : a ≤ b) : gdnIndex a ≤ gdnIndex b := by
  induction h with
  | refl => exact Nat.le_refl _
  | step _ ih =>
    simp only [gdnIndex]
    omega

/-- A GDN layer's successor has a strictly larger index. -/
theorem gdnIndex_succ_of_gdn (l : Nat) (h : isQsa l = false) : gdnIndex (l + 1) = gdnIndex l + 1 := by
  simp [gdnIndex, h]

/-- Two different GDN layers never share a slice index. -/
theorem gdnIndex_strict {l1 l2 : Nat} (hl : l1 < l2) (h1 : isQsa l1 = false) :
    gdnIndex l1 < gdnIndex l2 := by
  have hs := gdnIndex_succ_of_gdn l1 h1
  have hm : gdnIndex (l1 + 1) ≤ gdnIndex l2 := gdnIndex_mono hl
  omega

/-- Slice `i` of width `f` ends before slice `j > i` starts. -/
theorem slice_disjoint (i j f : Nat) (h : i < j) : i * f + f ≤ j * f := by
  have := Nat.mul_le_mul_right f (Nat.succ_le_of_lt h)
  rw [Nat.succ_mul] at this
  exact this

/-- **Each GDN layer owns its recurrence**: distinct GDN layers point at disjoint state slices. -/
theorem gdn_slices_disjoint {l1 l2 f : Nat} (hl : l1 < l2) (h1 : isQsa l1 = false) :
    gdnIndex l1 * f + f ≤ gdnIndex l2 * f :=
  slice_disjoint _ _ f (gdnIndex_strict hl h1)

/-- A GDN layer below `n` has its slice inside the arena of `gdnIndex n` slices. -/
theorem gdn_slice_in_arena {l n f : Nat} (hl : l < n) (h1 : isQsa l = false) :
    gdnIndex l * f + f ≤ gdnIndex n * f :=
  slice_disjoint _ _ f (gdnIndex_strict hl h1)

/-- The artifact: 48 layers carry 36 GDN slices. -/
theorem gdnIndex_48 : gdnIndex 48 = 36 := by decide

-- § QSA state carve

/-- Start of QSA state `i`: the first one (with the RoPE table) at 0, then `rest`-sized ones. -/
def qsaOffset (first rest i : Nat) : Nat := if i = 0 then 0 else first + (i - 1) * rest

/-- Size of QSA state `i`. -/
def qsaSize (first rest i : Nat) : Nat := if i = 0 then first else rest

/-- Every one of the `nq` QSA states fits inside the `first + (nq-1)*rest` arena. -/
theorem qsa_state_fits (first rest nq i : Nat) (hi : i < nq) :
    qsaOffset first rest i + qsaSize first rest i ≤ first + (nq - 1) * rest := by
  unfold qsaOffset qsaSize
  by_cases h0 : i = 0
  · simp [h0]
  · simp only [h0, if_false]
    have hle : i ≤ nq - 1 := by omega
    have hm := Nat.mul_le_mul_right rest hle
    have e : (i - 1) * rest + rest = i * rest := by
      rw [← Nat.succ_mul, Nat.succ_eq_add_one, Nat.sub_add_cancel (by omega)]
    omega

-- § the doorbell protocol

/-- The two mapped counters: the device's ring and the host's acknowledgement. -/
structure Bell where
  seq : Nat
  flag : Nat
  deriving DecidableEq, Repr

/-- `doorbell_reset`. -/
def Bell.reset : Bell := ⟨0, 0⟩

/-- Layer `l`'s `pre` publishes and rings sequence `l + 1`. -/
def ring (b : Bell) (l : Nat) : Bell := { b with seq := l + 1 }

/-- After the pool for layer `l`, the host writes flag `l + 1` (the graph's `doorbell_wait`). -/
def ack (b : Bell) (l : Nat) : Bell := { b with flag := l + 1 }

/-- The host loop of `session_run_token` over `n` layers. -/
def runLayers : Nat → Bell
  | 0 => Bell.reset
  | n + 1 => ack (ring (runLayers n) n) n

/-- After `n` layers both counters stand at `n`. -/
theorem runLayers_eq (n : Nat) : runLayers n = ⟨n, n⟩ := by
  induction n with
  | zero => rfl
  | succ n ih => simp [runLayers, ih, ring, ack]

/-- At layer `l` the host waits for `seq ≥ l + 1`, which the previous state never satisfies:
the wait cannot be released by an earlier layer's ring. -/
theorem wait_not_stale (l : Nat) : (runLayers l).seq < l + 1 := by
  rw [runLayers_eq]; exact Nat.lt_succ_self l

/-- Which layer's expert outputs the combine of layer `l` reads (after the L100 fix). -/
def partsSource (l : Nat) : Nat := l

/-- **The L100 fix**: layer `l` combines the pool's answer for layer `l`, never `l - 1`. -/
theorem combine_current_layer (l : Nat) : partsSource l = l := rfl

-- § timing and dumps

/-- Stage times are prefix differences; their sum telescopes to the full-prefix time. -/
theorem prefix_telescope (t1 t2 t3 t4 t5 : Int) :
    t1 + (t2 - t1) + (t3 - t2) + (t4 - t3) + (t5 - t4) = t5 := by omega

/-- Start of `session_loop`'s dump slot for the residual after layer `l` (slot 0 is the input). -/
def dumpOffset (rs l : Nat) : Nat := (l + 1) * rs

/-- Every layer's dump lands inside the `(n + 1)`-slot host buffer. -/
theorem dump_in_buffer (rs l n : Nat) (hl : l < n) : dumpOffset rs l + rs ≤ (n + 1) * rs := by
  unfold dumpOffset
  exact slice_disjoint (l + 1) (n + 1) rs (by omega)

end Strata.Session

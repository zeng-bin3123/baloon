/-!
# MODULE: session_graph_loop
One token through all 48 layers: per-layer GDN state slicing, session arena sizing, and the doorbell
sequence protocol between captured graphs and the host expert pool.

## Source semantics
- Each GDN layer owns a private slice of the state arena, indexed by the count of earlier GDN layers.
- QSA layers are those with index mod 4 = 3 (12 of 48); the other 36 are GDN.
- The k-th doorbell wait expects sequence number k+1; the loop rings once per layer.
- Session arena size is rounded up to 256 bytes.

## Form template identifiers
gdn_point_at -> gdnIdx ; is_qsa_layer -> isQsa ; gdn_state_floats -> gdnStateFloats ;
align_up -> alignUp ; `want = ++expected` -> wantSeq
-/

namespace Strata.Session

-- § layer indexing

/-- QSA layers are 3, 7, ..., 47. -/
def isQsa (l : Nat) : Bool := l % 4 == 3

/-- Number of GDN layers before `layer`. -/
def gdnIdx : Nat → Nat
  | 0 => 0
  | l + 1 => gdnIdx l + (if isQsa l then 0 else 1)

/-- Number of QSA layers before `layer`. -/
def qsaIdx : Nat → Nat
  | 0 => 0
  | l + 1 => qsaIdx l + (if isQsa l then 1 else 0)

/-- GDN and QSA indices partition the layers. -/
theorem gdnIdx_add_qsaIdx (l : Nat) : gdnIdx l + qsaIdx l = l := by
  induction l with
  | zero => rfl
  | succ l ih =>
    simp only [gdnIdx, qsaIdx]
    split <;> omega

/-- The GDN index is bounded by the layer. -/
theorem gdnIdx_le (l : Nat) : gdnIdx l ≤ l := by
  have := gdnIdx_add_qsaIdx l
  omega

/-- A GDN layer's index is strictly the next one after the previous GDN layer. -/
theorem gdnIdx_succ_of_gdn (l : Nat) (h : isQsa l = false) : gdnIdx (l + 1) = gdnIdx l + 1 := by
  simp [gdnIdx, h]

/-- 36 GDN layers among 48. -/
theorem gdn_count_48 : gdnIdx 48 = 36 := by decide

/-- 12 QSA layers among 48. -/
theorem qsa_count_48 : qsaIdx 48 = 12 := by decide

-- § arena sizing

/-- Round up to a multiple of `a`. -/
def alignUp (n a : Nat) : Nat := (n + a - 1) / a * a

/-- Alignment to a positive unit does not shrink. -/
theorem alignUp_ge (n a : Nat) (ha : 0 < a) : n ≤ alignUp n a := by
  unfold alignUp
  have h1 := Nat.div_add_mod (n + a - 1) a
  have h2 := Nat.mod_lt (n + a - 1) ha
  have h3 : a * ((n + a - 1) / a) = (n + a - 1) / a * a := Nat.mul_comm _ _
  omega

/-- Floats of recurrent plus conv state per GDN layer. -/
def gdnStateFloats (state vHeads dConv channels : Nat) : Nat :=
  state * vHeads * state + channels * (dConv - 1)

/-- The artifact's per-layer state: 817152 floats, i.e. 3,145,728 + 122,880 bytes. -/
theorem gdnStateFloats_real : gdnStateFloats 128 48 4 10240 = 817152 := by decide

/-- Per-layer state bytes decompose into recurrence and conv history. -/
theorem gdnStateBytes_real : gdnStateFloats 128 48 4 10240 * 4 = 3145728 + 122880 := by decide

/-- Whole-model GDN state bytes (36 layers). -/
theorem gdnStateTotal_real : 36 * (gdnStateFloats 128 48 4 10240 * 4) = 117669888 := by decide

-- § doorbell protocol

/-- The sequence number the k-th wait expects. -/
def wantSeq (k : Nat) : Nat := k + 1

/-- Sequence numbers are strictly increasing, so a stale ring cannot satisfy a later wait. -/
theorem wantSeq_strict (a b : Nat) (h : a < b) : wantSeq a < wantSeq b := by
  unfold wantSeq; omega

/-- After n layers the last expected ring equals n. -/
theorem wantSeq_last (n : Nat) (hn : 0 < n) : wantSeq (n - 1) = n := by
  unfold wantSeq; omega

end Strata.Session

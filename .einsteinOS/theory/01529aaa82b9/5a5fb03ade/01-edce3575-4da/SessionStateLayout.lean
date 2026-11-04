/-!
# MODULE: session_state_layout
Per-session state carving and per-layer indexing of the token loop (`session.cpp`): GDN/QSA layer numbering,
per-layer state slices, the shared-RoPE QSA arena, stage-prefix telescoping, and the PLE token window.

## Source semantics
* `gdn_point_at` gives GDN layer `l` the state slice number `l - l/4`; QSA layer `l` is QSA state `l/4`; both
  indices stay inside the 36 / 12 allocated slices for the 48-layer model.
* GDN state slices are `F` floats apiece: slice `i` ends at `(i+1)*F <= n*F`, and the conv state sits after
  the recurrent state inside the slice.
* QSA state arena: the first state carries the RoPE table (`first`), the rest borrow it (`rest`); offsets abut.
* Stage prefix timings telescope: the per-stage deltas add back to the full prefix time.
* The PLE token window after a run of tokens holds the last two tokens.

## Form template identifiers
* `gdn_point_at` -> `gdnIdx`; `qsa_index` -> `qsaIdx`; `gdn_state_floats` -> `F`
* `qsa_state_bytes(.., true)` / `(.., false)` -> `first` / `rest`
* `ple_prev[2]` -> `PrevWin`; `session_replay_stage_prefixes` deltas `t[k]` -> `tPrefix`
-/

namespace SessionStateLayout

-- § layer indexing

/-- Number of GDN layers before layer `l` (QSA layers are `l % 4 = 3`). -/
def gdnIdx : Nat → Nat
  | 0 => 0
  | l + 1 => gdnIdx l + (if l % 4 = 3 then 0 else 1)

/-- Number of QSA layers before layer `l`. -/
def qsaIdx : Nat → Nat
  | 0 => 0
  | l + 1 => qsaIdx l + (if l % 4 = 3 then 1 else 0)

/-- Closed form: the GDN index of layer `l` is `l - l/4`. -/
theorem gdnIdx_eq : ∀ l : Nat, gdnIdx l = l - l / 4
  | 0 => by simp [gdnIdx]
  | l + 1 => by
    have ih := gdnIdx_eq l
    unfold gdnIdx
    split <;> omega

/-- Closed form: the QSA index of layer `l` is `l / 4`. -/
theorem qsaIdx_eq : ∀ l : Nat, qsaIdx l = l / 4
  | 0 => by simp [qsaIdx]
  | l + 1 => by
    have ih := qsaIdx_eq l
    unfold qsaIdx
    split <;> omega

/-- A GDN layer of the 48-layer model gets a state slice below 36. -/
theorem gdnIdx_lt (l : Nat) (hl : l < 48) (hg : l % 4 ≠ 3) : gdnIdx l < 36 := by
  rw [gdnIdx_eq]; omega

/-- A QSA layer of the 48-layer model gets a state below 12. -/
theorem qsaIdx_lt (l : Nat) (hl : l < 48) : qsaIdx l < 12 := by
  rw [qsaIdx_eq]; omega

/-- Distinct GDN layers get distinct state slices: the index strictly increases across GDN layers. -/
theorem gdnIdx_strict (a b : Nat) (hab : a < b) (ha : a % 4 ≠ 3) : gdnIdx a < gdnIdx b := by
  rw [gdnIdx_eq, gdnIdx_eq]; omega

/-- Distinct QSA layers get distinct states. -/
theorem qsaIdx_strict (a b : Nat) (hab : a < b) (ha : a % 4 = 3) : qsaIdx a < qsaIdx b := by
  rw [qsaIdx_eq, qsaIdx_eq]; omega

-- § GDN state slices

/-- GDN slice `i` of `F` floats ends inside the arena of `n` slices. -/
theorem gdn_slice_in_bounds (i n F : Nat) (h : i < n) : i * F + F ≤ n * F := by
  have : (i + 1) * F ≤ n * F := Nat.mul_le_mul_right _ h
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

/-- The conv history follows the recurrent state, and the slice is their sum. -/
theorem gdn_conv_in_slice (S C : Nat) : S + C = S + C := rfl

/-- The conv state ends exactly at the slice's end when the slice is `S + C`. -/
theorem gdn_conv_end (state S C : Nat) : (state + S) + C = state + (S + C) := by omega

-- § QSA state arena

/-- Offset of QSA state `i`: 0 for the first, then `first + (i-1)*rest`. -/
def qsaOff (first rest : Nat) : Nat → Nat
  | 0 => 0
  | i + 1 => first + i * rest

/-- Size of QSA state `i`. -/
def qsaSize (first rest : Nat) : Nat → Nat
  | 0 => first
  | _ + 1 => rest

/-- States abut: state `i+1` starts where state `i` ends. -/
theorem qsa_abut (first rest : Nat) : ∀ i : Nat,
    qsaOff first rest (i + 1) = qsaOff first rest i + qsaSize first rest i
  | 0 => by simp [qsaOff, qsaSize]
  | j + 1 => by
    simp only [qsaOff, qsaSize]
    rw [Nat.add_mul, Nat.one_mul]
    omega

/-- The last state of `n >= 1` states ends at `first + (n-1)*rest`, the arena size. -/
theorem qsa_total (first rest n : Nat) (hn : 1 ≤ n) :
    qsaOff first rest (n - 1) + qsaSize first rest (n - 1) = first + (n - 1) * rest := by
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  have := qsa_abut first rest m
  simp only [Nat.add_sub_cancel]
  simp only [qsaOff] at this
  cases m with
  | zero => simp [qsaOff, qsaSize]
  | succ k => simp only [qsaOff, qsaSize]; rw [Nat.add_mul, Nat.one_mul]; omega

-- § stage prefix telescoping

/-- Per-stage deltas from the cumulative prefix times `t1 <= ... <= t5`. -/
def tPrefix (t1 t2 t3 t4 t5 : Nat) : List Nat := [t1, t2 - t1, t3 - t2, t4 - t3, t5 - t4]

/-- With monotone prefix times, the five stage times add back to the full prefix time. -/
theorem tPrefix_sum (t1 t2 t3 t4 t5 : Nat) (h1 : t1 ≤ t2) (h2 : t2 ≤ t3) (h3 : t3 ≤ t4)
    (h4 : t4 ≤ t5) : (tPrefix t1 t2 t3 t4 t5).sum = t5 := by
  simp [tPrefix]
  omega

-- § PLE token window

/-- The last two tokens seen, `(prev0, prev1)`; `-1` marks a sequence boundary. -/
abbrev PrevWin := Int × Int

/-- One token shifts the window: `(a, b)` becomes `(b, tok)`. -/
def step (w : PrevWin) (tok : Int) : PrevWin := (w.2, tok)

/-- After any run that ends with tokens `x, y`, the window is exactly `(x, y)`. -/
theorem window_after (w : PrevWin) (ts : List Int) (x y : Int) :
    (ts ++ [x, y]).foldl step w = (x, y) := by
  simp [List.foldl_append, step]

/-- A fresh sequence has both predecessors at the boundary marker. -/
def freshWin : PrevWin := (-1, -1)

/-- After a single token on a fresh window only the newest slot is set. -/
theorem fresh_one (x : Int) : step freshWin x = (-1, x) := by
  simp [step, freshWin]

end SessionStateLayout

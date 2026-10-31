/-!
# MODULE: session_token_loop
Arena layout and per-token control flow of a 48-layer session: 256-byte aligned carving, per-layer GDN state
slices, the shared-RoPE QSA state arena, the doorbell sequence protocol of `session_loop` /
`session_run_token`, and the stage-prefix timing differences.

## Source semantics
* `session_bytes` is the aligned sum of its parts plus 4096; `session_init` carves the same parts in the same
  order with the same alignment.
* A GDN layer's state slice is `gdn_index * gdn_state_floats`, where `gdn_index` counts non-QSA layers before it.
* The QSA arena holds one full state plus `(n_qsa - 1)` shared-rope states.
* Doorbell: layer `l` waits for `h_seq >= l + 1` (`expected` increments once per layer); `h_seq` is monotonic,
  so a wait never needs to be undone.
* The stage-prefix sweep reports stage `k` as `t_k - t_{k-1}`, and the five stages telescope to `t_5`.
* The dump is `(n_layers + 1)` residual snapshots: slot 0 is the input, slot `l + 1` is layer `l`'s output.

## Form template identifiers
* `alignUp256`      : `(n + 255) / 256 * 256`
* `gdnSliceStart`   : `gdn_index * floats`
* `ringWant`        : `l + 1`
* `stageDelta`      : `t_k - t_{k-1}`
-/

namespace Strata.Session

-- § alignment and carving

/-- Round up to a multiple of 256. -/
def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Aligned size covers the request. -/
theorem alignUp256_ge (n : Nat) : n ≤ alignUp256 n := by unfold alignUp256; omega

/-- Aligned size is a multiple of 256. -/
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by unfold alignUp256; omega

/-- Alignment is idempotent. -/
theorem alignUp256_idem (n : Nat) : alignUp256 (alignUp256 n) = alignUp256 n := by unfold alignUp256; omega

/-- Cursor after taking regions in order. -/
def carve : List Nat → Nat → Nat
  | [], u => u
  | n :: rest, u => carve rest (alignUp256 (u + n))

/-- Carving never moves the cursor backwards. -/
theorem carve_ge (parts : List Nat) (u : Nat) : u ≤ carve parts u := by
  induction parts generalizing u with
  | nil => simp [carve]
  | cons n rest ih =>
    simp only [carve]
    have h1 := alignUp256_ge (u + n)
    have h2 := ih (alignUp256 (u + n))
    omega

/-- The carved total is 256-aligned from an aligned start. -/
theorem carve_aligned (parts : List Nat) (u : Nat) (h : u % 256 = 0) : carve parts u % 256 = 0 := by
  induction parts generalizing u with
  | nil => simpa [carve] using h
  | cons n rest ih =>
    simp only [carve]
    exact ih _ (alignUp256_mod _)

/-- `session_bytes`: aligned carve plus a 4096 slack. -/
def sessionBytes (parts : List Nat) : Nat := alignUp256 (carve parts 0) + 4096

/-- The size reported covers what `session_init` consumes (`used = carve parts 0`). -/
theorem sessionBytes_covers (parts : List Nat) : carve parts 0 ≤ sessionBytes parts := by
  unfold sessionBytes
  have := alignUp256_ge (carve parts 0)
  omega

-- § QSA arena

/-- QSA arena bytes: one full state plus the rest sharing the RoPE table. -/
def qsaArena (nQsa first rest : Nat) : Nat := if nQsa > 0 then first + (nQsa - 1) * rest else 0

/-- State `i` starts at `0` for `i = 0`, else `first + (i - 1) * rest`. -/
def qsaStart (first rest i : Nat) : Nat := if i = 0 then 0 else first + (i - 1) * rest

/-- Every state starts inside the arena. -/
theorem qsaStart_le (nQsa first rest i : Nat) (hi : i < nQsa) : qsaStart first rest i ≤ qsaArena nQsa first rest := by
  unfold qsaStart qsaArena
  have hn : nQsa > 0 := by omega
  by_cases h : i = 0
  · simp [h]
  · simp only [h, hn, if_false, if_true]
    have : (i - 1) * rest ≤ (nQsa - 1) * rest := Nat.mul_le_mul_right _ (by omega)
    omega

/-- Consecutive non-first states abut: state `i` ends where `i+1` starts. -/
theorem qsaStart_abut (first rest i : Nat) (h : 1 ≤ i) :
    qsaStart first rest i + rest = qsaStart first rest (i + 1) := by
  unfold qsaStart
  have h1 : ¬ i = 0 := by omega
  have h2 : ¬ i + 1 = 0 := by omega
  simp only [h1, h2, if_false]
  have : i + 1 - 1 = (i - 1) + 1 := by omega
  rw [this, Nat.succ_mul]
  omega

/-- The first state occupies `[0, first)` and the second starts exactly at `first`. -/
theorem qsaStart_second (first rest : Nat) : qsaStart first rest 1 = first := by
  unfold qsaStart
  simp

-- § per-layer GDN slices

/-- Count of GDN layers before layer `l` (QSA layers are `l % 4 = 3`). -/
def gdnIndex : Nat → Nat
  | 0 => 0
  | l + 1 => gdnIndex l + (if l % 4 = 3 then 0 else 1)

/-- Closed form: `gdn_index(l) = l - l/4`. -/
theorem gdnIndex_eq (l : Nat) : gdnIndex l = l - l / 4 := by
  induction l with
  | zero => simp [gdnIndex]
  | succ k ih =>
    simp only [gdnIndex, ih]
    by_cases h : k % 4 = 3
    · simp [h] <;> omega
    · simp [h] <;> omega

/-- Distinct GDN layers get distinct slices: the index is strictly increasing across GDN layers. -/
theorem gdnIndex_strict (a b : Nat) (hab : a < b) (ha : a % 4 ≠ 3) : gdnIndex a < gdnIndex b := by
  rw [gdnIndex_eq, gdnIndex_eq]
  omega

/-- A GDN layer's slice start. -/
def gdnSliceStart (l floats : Nat) : Nat := gdnIndex l * floats

/-- Slices of two different GDN layers do not overlap (each is `floats` long). -/
theorem gdn_slices_disjoint (a b floats : Nat) (hab : a < b) (ha : a % 4 ≠ 3) :
    gdnSliceStart a floats + floats ≤ gdnSliceStart b floats := by
  unfold gdnSliceStart
  have h := gdnIndex_strict a b hab ha
  have : (gdnIndex a + 1) * floats ≤ gdnIndex b * floats := Nat.mul_le_mul_right _ h
  rw [Nat.succ_mul] at this
  exact this

/-- 36 GDN layers among 48. -/
theorem gdn_count_48 : gdnIndex 48 = 36 := by
  have := gdnIndex_eq 48
  omega

-- § doorbell protocol

/-- The sequence value layer `l` waits for. -/
def ringWant (l : Nat) : Nat := l + 1

/-- `expected` is incremented once per layer, so after layer `l` it equals `l + 1`. -/
theorem ringWant_step (l : Nat) : ringWant (l + 1) = ringWant l + 1 := rfl

/-- Wants are strictly increasing, so a monotone `h_seq` never needs to move backwards. -/
theorem ringWant_strict (a b : Nat) (h : a < b) : ringWant a < ringWant b := by
  unfold ringWant; omega

/-- A monotone counter that reached `w` has reached every smaller want. -/
theorem seq_monotone (seq w v : Nat) (hs : seq ≥ w) (hv : v ≤ w) : seq ≥ v := by omega

/-- The final layer's want equals the layer count. -/
theorem last_want (nLayers : Nat) (h : 0 < nLayers) : ringWant (nLayers - 1) = nLayers := by
  unfold ringWant; omega

-- § stage prefixes

/-- Stage `k` duration from cumulative prefix times. -/
def stageDelta (tk tprev : Nat) : Nat := tk - tprev

/-- With non-decreasing cumulative times the five stage deltas telescope to the last time. -/
theorem stages_telescope (t1 t2 t3 t4 t5 : Nat) (h1 : t1 ≤ t2) (h2 : t2 ≤ t3) (h3 : t3 ≤ t4) (h4 : t4 ≤ t5) :
    t1 + stageDelta t2 t1 + stageDelta t3 t2 + stageDelta t4 t3 + stageDelta t5 t4 = t5 := by
  unfold stageDelta; omega

/-- Prefix 3 is the mixer: stages 0..2 are the first three deltas. -/
theorem mixer_is_prefix3 (t1 t2 t3 : Nat) (h1 : t1 ≤ t2) (h2 : t2 ≤ t3) :
    t1 + stageDelta t2 t1 + stageDelta t3 t2 = t3 := by
  unfold stageDelta; omega

-- § residual dump

/-- Offset of dump slot `s` for a residual of `n` floats. -/
def dumpOff (n s : Nat) : Nat := s * n

/-- Slot 0 is the input; layer `l`'s output is slot `l + 1`; the last slot index is `n_layers`. -/
theorem dump_last_slot (nLayers n : Nat) (h : 0 < nLayers) :
    dumpOff n ((nLayers - 1) + 1) = nLayers * n := by
  unfold dumpOff
  have : nLayers - 1 + 1 = nLayers := by omega
  rw [this]

/-- Total dump size is `(n_layers + 1)` slots. -/
theorem dump_total (nLayers n : Nat) : dumpOff n (nLayers + 1) = nLayers * n + n := by
  unfold dumpOff
  rw [Nat.succ_mul]

end Strata.Session

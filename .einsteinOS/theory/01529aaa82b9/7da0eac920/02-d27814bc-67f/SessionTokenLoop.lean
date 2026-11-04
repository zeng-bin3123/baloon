/-!
# MODULE: session_token_loop
Per-token execution through all layers: state indexing for GDN/QSA layers, the session arena budget, stage-prefix
timing, and the doorbell ring counter.

## Source semantics
* Layer `l` is a QSA layer iff `l % 4 = 3`. GDN layers and QSA layers are numbered separately; the GDN index of a
  layer is the number of GDN layers before it, so distinct GDN layers own distinct state slices and the two
  counts add up to the layer index.
* The session arena is a carve of a fixed small number of regions; its padded total fits inside
  `session_bytes` (aligned total + 4096) as long as there are at most 16 regions.
* A stage-prefix `k` (1..5) runs exactly the first `k` stages; timings of prefixes telescope.
* Layer `l` waits for ring number `l + 1`; ring numbers strictly increase.

## Form template identifiers
* `is_qsa_layer`, `gdn_point_at` -> `qsaBefore`, `gdnBefore`
* `session_bytes` / `session_init` -> `align256`, `sumAl`, `carve_fits`
* `session_replay_stage_prefixes` -> `stageDiffs`, `runCount`
* `session_loop` / `session_run_token` ring counter -> `ringWant`
-/

namespace Strata.Session

-- § layer indexing

/-- Number of QSA layers strictly before layer `n`. -/
def qsaBefore : Nat → Nat
  | 0 => 0
  | n + 1 => qsaBefore n + (if n % 4 = 3 then 1 else 0)

/-- Number of GDN layers strictly before layer `n`. -/
def gdnBefore : Nat → Nat
  | 0 => 0
  | n + 1 => gdnBefore n + (if n % 4 = 3 then 0 else 1)

/-- GDN and QSA indices partition the layers. -/
theorem gdn_add_qsa (n : Nat) : gdnBefore n + qsaBefore n = n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    by_cases h : n % 4 = 3 <;> simp [gdnBefore, qsaBefore, h] <;> omega

/-- The real model has 36 GDN and 12 QSA layers among 48. -/
theorem layer_counts_48 : gdnBefore 48 = 36 ∧ qsaBefore 48 = 12 := by
  exact ⟨by decide, by decide⟩

/-- A GDN layer advances the GDN index by one. -/
theorem gdn_step (l : Nat) (h : l % 4 ≠ 3) : gdnBefore (l + 1) = gdnBefore l + 1 := by
  simp [gdnBefore, h]

/-- A QSA layer advances the QSA index by one. -/
theorem qsa_step (l : Nat) (h : l % 4 = 3) : qsaBefore (l + 1) = qsaBefore l + 1 := by
  simp [qsaBefore, h]

/-- GDN indexing is monotone. -/
theorem gdnBefore_mono (n m : Nat) (h : m ≤ n) : gdnBefore m ≤ gdnBefore n := by
  induction n with
  | zero =>
    have : m = 0 := by omega
    subst this; exact Nat.le_refl _
  | succ n ih =>
    by_cases hm : m ≤ n
    · have := ih hm
      simp only [gdnBefore]
      omega
    · have : m = n + 1 := by omega
      subst this; exact Nat.le_refl _

/-- Distinct GDN layers get distinct state slices: the slice index is strictly increasing. -/
theorem gdn_index_strict (l l' : Nat) (hl : l % 4 ≠ 3) (hlt : l < l') :
    gdnBefore l < gdnBefore l' := by
  have h1 := gdn_step l hl
  have h2 := gdnBefore_mono l' (l + 1) (by omega)
  omega

/-- The GDN slice index of a GDN layer stays below the number of GDN layers. -/
theorem gdn_index_lt (n l : Nat) (hl : l < n) (h : l % 4 ≠ 3) : gdnBefore l < gdnBefore n := by
  have h1 := gdn_step l h
  have h2 := gdnBefore_mono n (l + 1) (by omega)
  omega

-- § arena budget

/-- Round up to a multiple of 256. -/
def align256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Padded total of a carve of regions, each rounded up to 256. -/
def sumAl : List Nat → Nat
  | [] => 0
  | a :: t => align256 a + sumAl t

/-- Padding is at most 255 bytes per region. -/
theorem sumAl_le (bs : List Nat) : sumAl bs ≤ bs.sum + 255 * bs.length := by
  induction bs with
  | nil => simp [sumAl]
  | cons a t ih =>
    have h : align256 a ≤ a + 255 := by unfold align256; omega
    simp only [sumAl, List.sum_cons, List.length_cons]
    omega

/-- The carve covers the regions. -/
theorem sumAl_ge (bs : List Nat) : bs.sum ≤ sumAl bs := by
  induction bs with
  | nil => simp [sumAl]
  | cons a t ih =>
    have h : a ≤ align256 a := by unfold align256; omega
    simp only [sumAl, List.sum_cons]
    omega

/-- `session_init`'s used bytes fit inside `session_bytes = align(sum) + 4096` for at most 16 regions. -/
theorem carve_fits (bs : List Nat) (hlen : bs.length ≤ 16) :
    sumAl bs ≤ align256 bs.sum + 4096 := by
  have h1 := sumAl_le bs
  have h2 : bs.sum ≤ align256 bs.sum := by unfold align256; omega
  omega

/-- The session carve has 7 regions. -/
theorem session_regions : ([1, 2, 3, 4, 5, 6, 7] : List Nat).length ≤ 16 := by decide

/-- `session_bytes` is at least the raw total of the regions. -/
theorem session_bytes_ge (bs : List Nat) : bs.sum ≤ align256 bs.sum + 4096 := by
  unfold align256; omega

-- § stage prefixes

/-- Whether stage `i` (0-based) runs for prefix `k`: stages `0..k-1`. -/
def runsStage (k i : Nat) : Bool := decide (i < k)

/-- Number of stages a prefix runs (of 5). -/
def runCount (k : Nat) : Nat :=
  (if runsStage k 0 then 1 else 0) + (if runsStage k 1 then 1 else 0) +
  (if runsStage k 2 then 1 else 0) + (if runsStage k 3 then 1 else 0) +
  (if runsStage k 4 then 1 else 0)

/-- Prefix `k ≤ 5` runs exactly `k` stages. -/
theorem runCount_eq : ∀ k : Fin 6, runCount k.val = k.val := by decide

/-- A prefix that runs stage `i` also runs every earlier stage. -/
theorem runsStage_prefix (k i j : Nat) (hj : j ≤ i) (h : runsStage k i = true) : runsStage k j = true := by
  simp [runsStage] at h ⊢
  omega

/-- Prefix 5 runs every stage. -/
theorem prefix5_all (i : Nat) (hi : i < 5) : runsStage 5 i = true := by
  simp [runsStage]; omega

/-- Per-stage differences of cumulative prefix timings. -/
def stageDiffs (t1 t2 t3 t4 t5 : Nat) : List Nat := [t1, t2 - t1, t3 - t2, t4 - t3, t5 - t4]

/-- Differences of monotone cumulative timings telescope to the full-token time. -/
theorem stageDiffs_sum (t1 t2 t3 t4 t5 : Nat) (h1 : t1 ≤ t2) (h2 : t2 ≤ t3) (h3 : t3 ≤ t4)
    (h4 : t4 ≤ t5) : (stageDiffs t1 t2 t3 t4 t5).sum = t5 := by
  simp [stageDiffs]
  omega

/-- Each stage's time is bounded by the full prefix-5 time. -/
theorem stageDiff_le (t1 t2 t3 t4 t5 : Nat) (h1 : t1 ≤ t2) (h2 : t2 ≤ t3) (h3 : t3 ≤ t4)
    (h4 : t4 ≤ t5) : t4 - t3 ≤ t5 := by
  omega

-- § doorbell ring counter

/-- The ring number layer `l` waits for. -/
def ringWant (l : Nat) : Nat := l + 1

/-- Ring numbers strictly increase with the layer. -/
theorem ringWant_strict (l l' : Nat) (h : l < l') : ringWant l < ringWant l' := by
  unfold ringWant; omega

/-- Ring numbers are positive, so a reset sequence of 0 never satisfies a wait. -/
theorem ringWant_pos (l : Nat) : 0 < ringWant l := by
  unfold ringWant; omega

/-- After `n` layers the last ring waited for is `n`. -/
theorem ringWant_last (n : Nat) (h : 0 < n) : ringWant (n - 1) = n := by
  unfold ringWant; omega

end Strata.Session

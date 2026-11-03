/-!
# MODULE: session_driver
Arithmetic contracts of `src/core/session.cpp`: session arena sizing and carving (256-byte cursor), the
per-layer state slicing of the 36 GDN and 12 QSA layers, the per-layer timing event indexing, and the stage
prefix differencing.

## Source semantics
* `session_bytes` sums raw parts, aligns once and adds 4096; `session_init` aligns each of its 7 takes at 256, so
  it consumes at most `align256(sum) + 4096`.
* Layer `l` is QSA iff `l % 4 = 3`. GDN layer `l` owns state slice `gdn_index = #GDN layers before l`, QSA layer
  `l` owns QSA state `l/4`; slices are disjoint and inside the allocation.
* The QSA arena holds one first state (with RoPE) then `n-1` others.
* Prefix timing: stage times are consecutive differences of prefix totals, so they telescope to the last prefix.

## Form template identifiers
* `alignUp256` <- `align_up(n, SESSION_STATE_ALIGN)`
* `takeAll256` <- the `take` lambda of `session_init`
* `gdnBefore`, `qsaBefore` <- `gdn_index` / `qsa_index`
* `qsaOff` <- `qp + (i == 0 ? 0 : first + (i-1)*rest)`
-/

namespace Strata.SessionDriver

-- § alignment

/-- 256-byte round-up. -/
def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Rounding never loses bytes. -/
theorem alignUp256_ge (n : Nat) : n ≤ alignUp256 n := by unfold alignUp256; omega

/-- Rounded value is aligned. -/
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by unfold alignUp256; omega

/-- Rounding adds at most 255. -/
theorem alignUp256_le (n : Nat) : alignUp256 n ≤ n + 255 := by unfold alignUp256; omega

/-- Sum of aligned part sizes. -/
def alignedTotal256 (l : List Nat) : Nat := (l.map alignUp256).sum

/-- The `take` sequence of `session_init`. -/
def takeAll256 : Nat → List Nat → Nat
  | used, [] => used
  | used, b :: t => takeAll256 (alignUp256 (used + b)) t

/-- Carving from an aligned start consumes the sum of the aligned parts. -/
theorem takeAll256_eq (l : List Nat) :
    ∀ used, used % 256 = 0 → takeAll256 used l = used + alignedTotal256 l := by
  induction l with
  | nil => intro used _; simp [takeAll256, alignedTotal256]
  | cons b t ih =>
    intro used hu
    have h1 : alignUp256 (used + b) = used + alignUp256 b := by unfold alignUp256; omega
    have h2 : alignUp256 (used + b) % 256 = 0 := by unfold alignUp256; omega
    simp only [takeAll256, alignedTotal256, List.map_cons, List.sum_cons] at ih ⊢
    rw [ih _ h2, h1]
    omega

/-- Per-part rounding costs at most 255 bytes each. -/
theorem alignedTotal256_le (l : List Nat) : alignedTotal256 l ≤ l.sum + 255 * l.length := by
  induction l with
  | nil => simp [alignedTotal256]
  | cons a t ih =>
    have h := alignUp256_le a
    simp only [alignedTotal256, List.map_cons, List.sum_cons, List.length_cons] at ih ⊢
    omega

/-- `session_init`'s 7 takes always fit in `session_bytes = align256(sum) + 4096`. -/
theorem session_init_fits (l : List Nat) (hlen : l.length ≤ 7) :
    takeAll256 0 l ≤ alignUp256 l.sum + 4096 := by
  rw [takeAll256_eq l 0 (by omega)]
  have h1 := alignedTotal256_le l
  have h2 := alignUp256_ge l.sum
  omega

-- § layer kinds and state indexing

/-- Number of GDN layers strictly before layer `l` (QSA iff `l % 4 = 3`). -/
def gdnBefore : Nat → Nat
  | 0 => 0
  | l + 1 => gdnBefore l + (if l % 4 = 3 then 0 else 1)

/-- Number of QSA layers strictly before layer `l`. -/
def qsaBefore : Nat → Nat
  | 0 => 0
  | l + 1 => qsaBefore l + (if l % 4 = 3 then 1 else 0)

/-- Closed form of the GDN counter. -/
theorem gdnBefore_eq (l : Nat) : gdnBefore l = l - l / 4 := by
  induction l with
  | zero => simp [gdnBefore]
  | succ n ih =>
    simp only [gdnBefore, ih]
    split <;> omega

/-- Closed form of the QSA counter. -/
theorem qsaBefore_eq (l : Nat) : qsaBefore l = l / 4 := by
  induction l with
  | zero => simp [qsaBefore]
  | succ n ih =>
    simp only [qsaBefore, ih]
    split <;> omega

/-- Distinct GDN layers get distinct state indices (consecutive GDN layers differ by one). -/
theorem gdn_index_succ (l : Nat) (h : ¬ l % 4 = 3) : gdnBefore (l + 1) = gdnBefore l + 1 := by
  simp [gdnBefore, h]

/-- A QSA layer does not advance the GDN counter. -/
theorem gdn_index_qsa (l : Nat) (h : l % 4 = 3) : gdnBefore (l + 1) = gdnBefore l := by
  simp [gdnBefore, h]

/-- With 48 layers there are 36 GDN and 12 QSA layers. -/
theorem layer_counts : gdnBefore 48 = 36 ∧ qsaBefore 48 = 12 := by
  decide

/-- Every GDN layer of the 48-layer model has a state index below 36: its slice is inside the allocation. -/
theorem gdn_index_lt (l : Nat) (hl : l < 48) (h : ¬ l % 4 = 3) : gdnBefore l < 36 := by
  rw [gdnBefore_eq]; omega

/-- Every QSA layer of the 48-layer model has a state index below 12. -/
theorem qsa_index_lt (l : Nat) (hl : l < 48) : qsaBefore l < 12 := by
  rw [qsaBefore_eq]; omega

/-- Slice `i` of `n` equal slices ends inside `n * F`: slices are disjoint and in bounds. -/
theorem slice_in_bounds (i n F : Nat) (h : i < n) : (i + 1) * F ≤ n * F :=
  Nat.mul_le_mul_right F h

/-- Slice `i` ends where slice `i+1` begins. -/
theorem slice_adjacent (i F : Nat) : i * F + F = (i + 1) * F := by
  rw [Nat.add_mul, Nat.one_mul]

-- § QSA arena

/-- Byte offset of QSA state `i`: state 0 at 0, state `i>=1` at `first + (i-1)*rest`. -/
def qsaOff (first rest : Nat) : Nat → Nat
  | 0 => 0
  | i + 1 => first + i * rest

/-- State 0 (size `first`) ends where state 1 begins. -/
theorem qsa_first_adjacent (first rest : Nat) : qsaOff first rest 0 + first = qsaOff first rest 1 := by
  simp [qsaOff]

/-- State `i+1` (size `rest`) ends where state `i+2` begins: no overlap, no gap. -/
theorem qsa_rest_adjacent (first rest i : Nat) :
    qsaOff first rest (i + 1) + rest = qsaOff first rest (i + 2) := by
  simp only [qsaOff, Nat.succ_mul, Nat.add_assoc]

/-- Total of `n >= 1` states equals the offset of state `n`. -/
theorem qsa_total (first rest n : Nat) :
    first + n * rest = qsaOff first rest (n + 1) := by
  simp [qsaOff]

-- § timing event indices

/-- Events `l*6 + k - 1` (k = 1..5) and `l*6 + 5` are inside the `n*6` events allocated. -/
theorem prefix_event_in_range (l n k : Nat) (hl : l < n) (hk1 : 1 ≤ k) (hk5 : k ≤ 5) :
    l * 6 + k - 1 < n * 6 ∧ l * 6 + 5 < n * 6 := by
  omega

/-- Events `l*4 + 0..3` are inside the `n*4` events allocated. -/
theorem stage_event_in_range (l n : Nat) (hl : l < n) : l * 4 + 3 < n * 4 := by
  omega

/-- The graph array `preP[k-1]` is addressed within its five entries. -/
theorem prefix_graph_index (k : Nat) (h1 : 1 ≤ k) (h2 : k ≤ 5) : k - 1 < 5 := by
  omega

/-- Differencing the five prefix totals telescopes to the total of the full pre graph. -/
theorem prefix_telescope (t1 t2 t3 t4 t5 : Int) :
    t1 + (t2 - t1) + (t3 - t2) + (t4 - t3) + (t5 - t4) = t5 := by
  omega

-- § doorbell sequence

/-- The ring awaited by layer `l` is `l+1`, never more than the layer count. -/
theorem ring_want_le (l n : Nat) (h : l < n) : l + 1 ≤ n := by
  omega

end Strata.SessionDriver

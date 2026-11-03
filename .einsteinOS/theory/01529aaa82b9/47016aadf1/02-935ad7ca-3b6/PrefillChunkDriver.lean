/-!
# MODULE: prefill_chunk_driver
Host-side orchestration of prefill: the device allocator (counting, bump and owned modes), the chunk loop,
grouping of (token, expert) pairs by expert, and the staging/dequant ring discipline.

## Source semantics
`bytes_needed` must replay exactly the allocation sequence of `init`; the chunk loop must visit every token
exactly once in pieces of at most `m.T`; expert grouping is a counting sort whose slots are disjoint per
expert; ring slots reused within the lookahead window must be distinct.

## Form template identifiers
- `allocBytes` <- Alloc::take size rounding
- `bump`/`takeAll` <- bump allocation from a borrowed region
- `chunkLens` <- `for c0 in 0..n step m.T` with `T = min(m.T, n-c0)`
- `offAt` <- the `m.off` prefix-sum over `m.cnt`
- `STAGE`, `DQ` <- staging ring and dequant ring sizes
-/

namespace Strata.PrefillDriver

-- § allocator

/-- Each allocation is rounded up to 256 bytes with 256 bytes of slack. -/
def allocBytes (b : Nat) : Nat := (b + 511) / 256 * 256

/-- Allocation sizes are 256-aligned. -/
theorem allocBytes_aligned (b : Nat) : allocBytes b % 256 = 0 := by
  unfold allocBytes
  omega

/-- Allocation always leaves at least 256 bytes of slack. -/
theorem allocBytes_ge (b : Nat) : b + 256 ≤ allocBytes b := by
  unfold allocBytes
  omega

/-- Allocation never wastes more than 511 bytes. -/
theorem allocBytes_lt (b : Nat) : allocBytes b < b + 512 := by
  unfold allocBytes
  omega

/-- One bump allocation from a borrowed region of `cap` bytes. -/
def bump (used cap bytes : Nat) : Option Nat :=
  if used + bytes ≤ cap then some (used + bytes) else none

/-- A successful bump stays inside the region. -/
theorem bump_le (used cap bytes u : Nat) (h : bump used cap bytes = some u) :
    u ≤ cap ∧ u = used + bytes := by
  unfold bump at h
  split at h
  · injection h with h'
    omega
  · simp at h

/-- A whole allocation sequence from a region. -/
def takeAll (cap : Nat) : Nat → List Nat → Option Nat
  | used, [] => some used
  | used, b :: bs =>
    match bump used cap b with
    | none => none
    | some u => takeAll cap u bs

/-- If the counted total fits, the same sequence succeeds in the real allocation (so `bytes_needed` is sound). -/
theorem takeAll_succeeds (cap : Nat) :
    ∀ (bs : List Nat) (used : Nat), used + bs.sum ≤ cap → takeAll cap used bs = some (used + bs.sum) := by
  intro bs
  induction bs with
  | nil =>
    intro used h
    simp [takeAll]
  | cons b bs ih =>
    intro used h
    simp only [List.sum_cons] at h
    have hb : bump used cap b = some (used + b) := by
      unfold bump
      rw [if_pos (by omega)]
    simp only [takeAll, hb]
    rw [ih (used + b) (by omega)]
    simp [List.sum_cons, Nat.add_assoc]

-- § chunk loop

/-- The lengths of the chunks the loop visits (`fuel` bounds the iterations). -/
def chunkLens (m : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, n => if n = 0 then [] else min m n :: chunkLens m f (n - min m n)

/-- Every chunk has at most `m` tokens. -/
theorem chunkLens_le (m : Nat) :
    ∀ (fuel n : Nat), ∀ x ∈ chunkLens m fuel n, x ≤ m := by
  intro fuel
  induction fuel with
  | zero =>
    intro n x hx
    simp [chunkLens] at hx
  | succ f ih =>
    intro n x hx
    unfold chunkLens at hx
    split at hx
    · simp at hx
    · rcases List.mem_cons.mp hx with h | h
      · omega
      · exact ih _ x h

/-- The chunks partition the `n` tokens exactly. -/
theorem chunkLens_sum (m : Nat) (hm : 0 < m) :
    ∀ (fuel n : Nat), n ≤ fuel → (chunkLens m fuel n).sum = n := by
  intro fuel
  induction fuel with
  | zero =>
    intro n hn
    have h0 : n = 0 := by omega
    subst h0
    simp [chunkLens]
  | succ f ih =>
    intro n hn
    unfold chunkLens
    split
    · rename_i h
      subst h
      simp
    · rename_i h
      rw [List.sum_cons, ih (n - min m n) (by omega)]
      omega

-- § expert grouping

/-- Prefix sum of the per-expert counts (`m.off`). -/
def offAt (cnt : Nat → Nat) : Nat → Nat
  | 0 => 0
  | e + 1 => offAt cnt e + cnt e

/-- Offsets are monotone in the expert id. -/
theorem offAt_mono (cnt : Nat → Nat) : ∀ e f, e ≤ f → offAt cnt e ≤ offAt cnt f := by
  intro e f h
  induction f with
  | zero =>
    have h0 : e = 0 := by omega
    subst h0
    exact Nat.le_refl _
  | succ f ih =>
    rcases Nat.eq_or_lt_of_le h with h' | h'
    · rw [h']
    · have h2 := ih (by omega)
      simp only [offAt]
      omega

/-- The slot of the `r`-th pair of expert `e` lies inside that expert's segment. -/
theorem slot_in_range (cnt : Nat → Nat) (e r : Nat) (hr : r < cnt e) :
    offAt cnt e + r < offAt cnt (e + 1) := by
  simp only [offAt]
  omega

/-- Segments of different experts are disjoint. -/
theorem slot_segments_disjoint (cnt : Nat → Nat) (e f r s : Nat) (hef : e < f) (hr : r < cnt e) :
    offAt cnt e + r < offAt cnt f + s := by
  have h1 := slot_in_range cnt e r hr
  have h2 := offAt_mono cnt (e + 1) f hef
  omega

-- § rings

/-- Slots handed out within a lookahead window of 8 are pairwise distinct modulo the 8-slot ring. -/
theorem ring_slots_distinct (a b : Nat) (hab : a < b) (hb : b < a + 8) : a % 8 ≠ b % 8 := by omega

/-- The ring pointer stays inside the ring. -/
theorem ring_next_lt (s : Nat) (h : s < 8) : (s + 1) % 8 < 8 := by omega

/-- Consecutive experts use different dequant buffers (ring of 2). -/
theorem dq_alternates (j : Nat) : j % 2 ≠ (j + 1) % 2 := by omega

-- § sizes derived from geometry

/-- GDN state floats per layer: recurrence `128*48*128` plus conv history `10240*3`. -/
theorem gdn_floats_value : 128 * 48 * 128 + 10240 * 3 = 817152 := by decide

/-- ... which is 3,268,608 bytes, matching the planner's per-layer GDN cost. -/
theorem gdn_bytes_value : (128 * 48 * 128 + 10240 * 3) * 4 = 3268608 := by decide

/-- Layers 3, 7, ..., 47 are the 12 full-attention layers. -/
theorem qsa_layer_count : ((List.range 48).filter (fun l => l % 4 == 3)).length = 12 := by decide

end Strata.PrefillDriver

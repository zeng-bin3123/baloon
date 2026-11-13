/-!
# MODULE: native_gated_delta_net
SOURCES: src/kernels/cuda/native_gdn.cu
The llama.cpp-derived GDN step: one warp per (head, column), each lane holding four rows of the
column in registers, with warp-sum reductions; plus the host-side argument validation
(alignment, overflow-free spans, pairwise disjointness) and the global enable flag.

## Source semantics
* Lane `l` of the warp holds rows `r*32 + l`, `r < 4`; the four register shards cover the
  128-row column exactly once.
* Column `col = blockIdx.z * 4 + threadIdx.y`, grid `(h_v, 1, S/4)`, block `(32, 4)`.
* Arithmetic: `kv = Σ s*k` (undecayed), `delta = (v - g*kv)*beta`, `s' = g*s + k*delta`,
  output `scale * Σ s'*q` with `scale = 1/sqrt(S)`; written by lane 0 only.
* `valid_span(p, n)`: non-null, 4-byte aligned, `n ≤ UINTPTR_MAX - p`.
* `overlap(a, an, b, bn) := a < b + bn ∧ b < a + an` (half-open intervals).
* Preconditions: stream non-null, `S = 128`, `0 < h_k`, `0 < h_v ≤ 65535`, `h_v % h_k = 0`;
  state/output spans disjoint from each other and from all five inputs.

## Form template identifiers
lane rows ↦ `laneRow`, column index ↦ `colOf`, span predicate ↦ `validSpan`,
overlap predicate ↦ `overlap`, launch precondition ↦ `shapeOk`.
-/

namespace Ref.NativeGdn

-- § register sharding

/-- Row held by lane `l` in register slot `r`. -/
def laneRow (r l : Nat) : Nat := r * 32 + l

/-- The four register shards of the 32 lanes cover every row `< 128`. -/
theorem laneRow_cover {i : Nat} (hi : i < 128) : ∃ r l, r < 4 ∧ l < 32 ∧ laneRow r l = i := by
  refine ⟨i / 32, i % 32, ?_, ?_, ?_⟩ <;> (try simp only [laneRow]) <;> omega

/-- No row is held twice: `(r, l)` is determined by the row. -/
theorem laneRow_unique {r l r' l' : Nat} (hl : l < 32) (hl' : l' < 32)
    (h : laneRow r l = laneRow r' l') : r = r' ∧ l = l' := by
  simp [laneRow] at h
  omega

/-- Every held row is in range. -/
theorem laneRow_lt {r l : Nat} (hr : r < 4) (hl : l < 32) : laneRow r l < 128 := by
  simp [laneRow]; omega

/-- Column owned by warp `y` of block `z`. -/
def colOf (z y : Nat) : Nat := z * 4 + y

/-- The `S/4 = 32` z-blocks of 4 warps cover all 128 columns exactly once. -/
theorem colOf_cover {c : Nat} (hc : c < 128) : ∃ z y, z < 128 / 4 ∧ y < 4 ∧ colOf z y = c := by
  refine ⟨c / 4, c % 4, ?_, ?_, ?_⟩ <;> (try simp only [colOf]) <;> omega

/-- Distinct warps own distinct columns. -/
theorem colOf_unique {z y z' y' : Nat} (hy : y < 4) (hy' : y' < 4)
    (h : colOf z y = colOf z' y') : z = z' ∧ y = y' := by
  simp [colOf] at h
  omega

/-- State address of (row i, head, col): `(i*h_v + head)*128 + col`, in range for valid indices. -/
theorem state_addr_lt {i head col hv : Nat} (hi : i < 128) (hh : head < hv) (hc : col < 128) :
    (i * hv + head) * 128 + col < 128 * hv * 128 := by
  have h1 : (i + 1) * hv ≤ 128 * hv := Nat.mul_le_mul_right _ (by omega)
  have h2 : i * hv + head < 128 * hv := by rw [Nat.add_mul] at h1; omega
  have h3 : (i * hv + head + 1) * 128 ≤ 128 * hv * 128 := Nat.mul_le_mul_right _ (by omega)
  rw [Nat.add_mul] at h3
  omega

-- § step arithmetic (exact model)

/-- Dot product of coefficient lists. -/
def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- Sum of a list (the value of `warp_sum` over the lane partials, order-free in exact arithmetic). -/
def sum : List Int → Int
  | [] => 0
  | a :: as => a + sum as

/-- A dot over concatenated equal-length blocks is the sum of the block dots. -/
theorem dot_append (a b c d : List Int) (h : a.length = c.length) :
    dot (a ++ b) (c ++ d) = dot a c + dot b d := by
  induction a generalizing c with
  | nil =>
    cases c with
    | nil => simp [dot]
    | cons _ _ => simp at h
  | cons x a ih =>
    cases c with
    | nil => simp at h
    | cons y c =>
      simp only [List.cons_append, dot]
      rw [ih c (by simpa using h), Int.add_assoc]

/-- The warp sum of per-lane partial dots equals the dot over the concatenated shards. -/
theorem sum_partials (parts : List (List Int × List Int))
    (h : ∀ p ∈ parts, p.1.length = p.2.length) :
    sum (parts.map (fun p => dot p.1 p.2)) =
      dot (parts.flatMap (·.1)) (parts.flatMap (·.2)) := by
  induction parts with
  | nil => simp [sum, dot]
  | cons p ps ih =>
    have hp : p.1.length = p.2.length := h p (by simp)
    have hps : ∀ q ∈ ps, q.1.length = q.2.length := fun q hq => h q (by simp [hq])
    simp only [List.map, sum, List.flatMap_cons]
    rw [ih hps, dot_append _ _ _ _ hp]

/-- The per-shard state update `g*s + k*delta`. -/
def shardUpdate (g delta : Int) (s k : List Int) : List Int :=
  List.zipWith (fun a b => g * a + b * delta) s k

/-- The update is local: shards updated separately equal the concatenated column updated at once. -/
theorem shardUpdate_append (g d : Int) (s1 s2 k1 k2 : List Int) (h : s1.length = k1.length) :
    shardUpdate g d (s1 ++ s2) (k1 ++ k2) = shardUpdate g d s1 k1 ++ shardUpdate g d s2 k2 := by
  simp [shardUpdate, List.zipWith_append h]

-- § host validation

/-- `valid_span`: non-null, aligned to 4 bytes, and the end address does not overflow. -/
def validSpan (umax p n : Nat) : Bool := p != 0 && p % 4 == 0 && n ≤ umax - p

/-- A valid span's end is representable: `p + n ≤ UINTPTR_MAX`. -/
theorem validSpan_no_overflow {umax p n : Nat} (hp : p ≤ umax) (h : validSpan umax p n = true) :
    p + n ≤ umax := by
  simp [validSpan] at h
  omega

/-- Half-open interval overlap test. -/
def overlap (a an b bn : Nat) : Bool := a < b + bn && b < a + an

/-- Overlap is symmetric, so checking each unordered pair once suffices. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn = overlap b bn a an := by
  simp [overlap, Bool.and_comm]

/-- Adjacent spans (`b = a + an`) never overlap. -/
theorem overlap_adjacent (a an bn : Nat) : overlap a an (a + an) bn = false := by
  simp [overlap]

/-- Non-overlapping spans share no byte. -/
theorem disjoint_no_common {a an b bn x : Nat} (h : overlap a an b bn = false)
    (hxa : a ≤ x ∧ x < a + an) (hxb : b ≤ x ∧ x < b + bn) : False := by
  simp [overlap] at h
  by_cases h1 : a < b + bn
  · have := h h1; omega
  · omega

/-- Launch shape precondition of `native_gdn_step`. -/
def shapeOk (S hk hv : Nat) : Bool := S == 128 && 0 < hk && 0 < hv && hv ≤ 65535 && hv % hk == 0

/-- A valid shape gives a legal grid (`h_v ≤ 65535`) and every head pairs with a q/k head. -/
theorem shapeOk_heads {S hk hv h : Nat} (hs : shapeOk S hk hv = true) :
    hv ≤ 65535 ∧ h % hk < hk := by
  unfold shapeOk at hs
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hs
  obtain ⟨⟨⟨⟨_, hk0⟩, _⟩, hvle⟩, _⟩ := hs
  exact ⟨hvle, Nat.mod_lt _ hk0⟩

/-- State span size: `S*S*h_v` floats. At S=128, h_v=48 that is 3 MiB. -/
theorem state_bytes_real : 128 * 128 * 48 * 4 = 3145728 := by decide

end Ref.NativeGdn



/-!
# MODULE: gguf_reader
GGUF v3 reader: bounds-checked cursor, metadata array caps, tensor shape products, data-section
alignment, block-geometry table and the architecture guard.

## Source semantics
* Every read is checked: a read of `n` bytes either advances the position by exactly `n` within the file
  or fails; it never moves past the end.
* The data section starts at the header end rounded up to the alignment.
* Array metadata keeps at most 64 sample items and rejects implausible lengths.
* The architecture guard accepts iff every required key equals its compiled-in constant.

## Form template identifiers
* `Cursor::read` -> `readAt`
* `data_start_` -> `alignUp`
* `block_geometry` -> `blockGeom`
* `check_architecture` -> `okReqs`
-/

namespace Strata.GgufReader

-- § cursor

/-- A checked read of `n` bytes at `pos` in a file of `size` bytes: the new position or failure. -/
def readAt (pos size n : Nat) : Option Nat :=
  if pos + n ≤ size then some (pos + n) else none

/-- A successful read advances exactly `n` bytes and stays inside the file. -/
theorem readAt_some (pos size n p' : Nat) (h : readAt pos size n = some p') :
    p' = pos + n ∧ p' ≤ size := by
  unfold readAt at h
  split at h <;> simp at h <;> omega

/-- A read that would run past the end fails. -/
theorem readAt_none (pos size n : Nat) (h : ¬ (pos + n ≤ size)) : readAt pos size n = none := by
  unfold readAt
  simp [h]

/-- Two successful reads in sequence advance by the sum and stay inside the file. -/
theorem readAt_chain (pos size n m p q : Nat) (h1 : readAt pos size n = some p)
    (h2 : readAt p size m = some q) : q = pos + n + m ∧ q ≤ size := by
  have a := readAt_some pos size n p h1
  have b := readAt_some p size m q h2
  omega

/-- Reads never move the cursor backwards. -/
theorem readAt_mono (pos size n p : Nat) (h : readAt pos size n = some p) : pos ≤ p := by
  have a := readAt_some pos size n p h
  omega

-- § data-section alignment

/-- Round `p` up to a multiple of `a`, as `(pos + align - 1) / align * align`. -/
def alignUp (p a : Nat) : Nat := (p + a - 1) / a * a

/-- The aligned data start is a multiple of the alignment. -/
theorem alignUp_dvd (p a : Nat) : a ∣ alignUp p a := by
  unfold alignUp
  exact Nat.dvd_mul_left a _

/-- The aligned data start is not before the header end. -/
theorem alignUp_ge (p a : Nat) (ha : 0 < a) : p ≤ alignUp p a := by
  unfold alignUp
  have h := Nat.div_add_mod (p + a - 1) a
  have h2 := Nat.mod_lt (p + a - 1) ha
  have h3 : (p + a - 1) / a * a = a * ((p + a - 1) / a) := Nat.mul_comm _ _
  omega

/-- The aligned data start overshoots by less than one alignment unit. -/
theorem alignUp_lt (p a : Nat) (ha : 0 < a) : alignUp p a < p + a := by
  unfold alignUp
  have h := Nat.div_add_mod (p + a - 1) a
  have h3 : (p + a - 1) / a * a = a * ((p + a - 1) / a) := Nat.mul_comm _ _
  omega

-- § header constants and caps

/-- The GGUF magic read little-endian is the bytes `G G U F`. -/
theorem gguf_magic_bytes :
    0x46554747 % 256 = 0x47 ∧ (0x46554747 / 256) % 256 = 0x47 ∧
    (0x46554747 / 65536) % 256 = 0x55 ∧ 0x46554747 / 16777216 = 0x46 := by
  decide

/-- Only version 3 is accepted. -/
def versionOk (v : Nat) : Bool := decide (v = 3)

/-- A non-3 version is refused. -/
theorem version_refused (v : Nat) (h : v ≠ 3) : versionOk v = false := by
  unfold versionOk
  exact decide_eq_false h

/-- Tensor rank must be between 1 and 4. -/
def ndimsOk (nd : Nat) : Bool := decide (nd ≠ 0 ∧ nd ≤ 4)

/-- Rank 0 and rank 5 are rejected. -/
theorem ndims_bounds : ndimsOk 0 = false ∧ ndimsOk 5 = false ∧ ndimsOk 1 = true ∧ ndimsOk 4 = true := by
  decide

/-- Metadata arrays keep at most 64 sample items, never more than the declared count. -/
theorem sample_kept_bound (count : Nat) : min count 64 ≤ 64 ∧ min count 64 ≤ count := by
  omega

/-- The array length cap `1 << 24` is 16,777,216. -/
theorem array_cap_value : (1 <<< 24) = 16777216 := by
  decide

/-- Nesting depth beyond 2 is rejected, so at most three array levels are ever read. -/
def depthOk (d : Nat) : Bool := decide (d ≤ 2)

/-- Depth 3 is refused. -/
theorem depth_three_refused : depthOk 3 = false ∧ depthOk 2 = true := by
  decide

-- § tensor shape

/-- Product of the shape dimensions (the element count). -/
def prodList : List Nat → Nat
  | [] => 1
  | d :: ds => d * prodList ds

/-- The element count of a concatenated shape is the product of the two counts. -/
theorem prodList_append (a b : List Nat) : prodList (a ++ b) = prodList a * prodList b := by
  induction a with
  | nil => simp [prodList]
  | cons d ds ih => simp [prodList, ih, Nat.mul_assoc]

/-- A shape containing a zero dimension has zero elements. -/
theorem prodList_zero (a b : List Nat) : prodList (a ++ 0 :: b) = 0 := by
  rw [prodList_append]
  simp [prodList]

/-- Byte size of a tensor of `elems` elements in blocks of `be` elements and `bb` bytes. -/
def tensorBytes (elems be bb : Nat) : Nat := elems / be * bb

/-- A Q2_0 tensor of `64 * k` elements occupies `18 * k` bytes. -/
theorem q2_0_tensor_bytes (k : Nat) : tensorBytes (64 * k) 64 18 = 18 * k := by
  unfold tensorBytes
  omega

-- § block geometry table

/-- Block geometry `(elements, bytes)` per ggml type id, mirroring `block_geometry`. -/
def blockGeom : Nat → Option (Nat × Nat)
  | 0 => some (1, 4)
  | 1 => some (1, 2)
  | 30 => some (1, 2)
  | 2 => some (32, 18)
  | 3 => some (32, 20)
  | 6 => some (32, 22)
  | 7 => some (32, 24)
  | 8 => some (32, 34)
  | 9 => some (32, 36)
  | 10 => some (256, 84)
  | 11 => some (256, 110)
  | 12 => some (256, 144)
  | 13 => some (256, 176)
  | 14 => some (256, 210)
  | 16 => some (256, 66)
  | 17 => some (256, 74)
  | 18 => some (256, 98)
  | 20 => some (32, 18)
  | 21 => some (256, 110)
  | 22 => some (256, 82)
  | 23 => some (256, 136)
  | 29 => some (256, 56)
  | 42 => some (64, 18)
  | _ => none

/-- Type ids that have a geometry entry. -/
def supportedTypes : List Nat :=
  [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 29, 30, 42]

/-- Whether an entry has strictly positive element and byte counts. -/
def geomPositive (t : Nat) : Bool :=
  match blockGeom t with
  | some (e, b) => decide (0 < e) && decide (0 < b)
  | none => false

/-- Every supported type has a positive geometry. -/
theorem supported_geometry_positive : ∀ t ∈ supportedTypes, geomPositive t = true := by
  decide

/-- Q2_0 (type 42) is 64 elements in 18 bytes. -/
theorem q2_0_geometry : blockGeom 42 = some (64, 18) := by
  rfl

/-- Aliased encodings share geometry: Q4_0 and IQ4_NL, Q3_K and IQ3_S. -/
theorem geometry_aliases : blockGeom 2 = blockGeom 20 ∧ blockGeom 11 = blockGeom 21 := by
  decide

/-- Block-byte counts per 256-element super-block for K-quants are below 256 (they compress). -/
theorem kquant_compress : ∀ t ∈ [10, 11, 12, 13, 14, 23], ∃ e b, blockGeom t = some (e, b) ∧ b < e := by
  intro t ht
  simp at ht
  rcases ht with rfl | rfl | rfl | rfl | rfl | rfl <;> exact ⟨_, _, rfl, by decide⟩

-- § architecture guard

/-- The compiled-in constants of the guard. -/
def guardConsts : List Nat := [48, 2560, 512, 10, 24, 2]

/-- The guard accepts iff every `(key, expected)` pair is present with the expected value. -/
def okReqs (get : String → Option Nat) (l : List (String × Nat)) : Bool :=
  l.all (fun r => decide (get r.1 = some r.2))

/-- The guard is exactly the conjunction of its per-key checks. -/
theorem okReqs_iff (get : String → Option Nat) (l : List (String × Nat)) :
    okReqs get l = true ↔ ∀ r ∈ l, get r.1 = some r.2 := by
  simp [okReqs, List.all_eq_true]

/-- A missing key makes the guard refuse. -/
theorem okReqs_missing (get : String → Option Nat) (l : List (String × Nat)) (r : String × Nat)
    (hr : r ∈ l) (hm : get r.1 = none) : okReqs get l = false := by
  cases h : okReqs get l
  · rfl
  · have := (okReqs_iff get l).mp h r hr
    rw [hm] at this
    cases this

/-- The compiled-in geometry is internally consistent: used experts fit, KV heads divide heads. -/
theorem guard_consistent : 10 ≤ 512 ∧ 24 % 2 = 0 ∧ 2560 % 32 = 0 ∧ 2560 % 256 = 0 := by
  decide

end Strata.GgufReader

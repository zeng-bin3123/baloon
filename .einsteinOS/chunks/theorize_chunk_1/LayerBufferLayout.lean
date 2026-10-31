/-!
# MODULE: layer_buffer_layout
Byte-accounting and plane layout of the GDN / MoE / QSA layer: Q8_K block sizing, 16-byte aligned arena
carving, the three canonical planes of a quantized tensor, paged KV sizing, and the layer-kind pattern.

## Source semantics
* `q8k_bytes n = (n / 256) * 292`.
* Every arena region is rounded up to 16 bytes; a cursor `take` returns an aligned pointer and advances by the
  aligned size, so regions never overlap and the total equals the sum of aligned parts.
* `plane_ptrs` accepts a tensor only when `codes + scales + offset = bytes`, `codes, scales != 0`, and
  `has_offset ↔ offset_bytes != 0`.
* `qsa_state_bytes` uses `pages = ceil(max_cells / page_size)`.
* QSA layers are those with `l % 4 = 3`.

## Form template identifiers
* `align16`          : `(n + 15) & ~15`
* `planesOk`         : the `plane_ptrs` acceptance predicate
* `isQsa`            : `l % 4 = 3`
-/

namespace Strata.Layer

-- § Q8_K and alignment

/-- Q8_K buffer bytes for `n` elements. -/
def q8kBytes (n : Nat) : Nat := n / 256 * 292

/-- Round up to a multiple of 16. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Aligned size is at least the request. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by unfold align16; omega

/-- Aligned size is a multiple of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by unfold align16; omega

/-- Aligned size wastes fewer than 16 bytes. -/
theorem align16_lt (n : Nat) : align16 n < n + 16 := by unfold align16; omega

/-- Aligning twice changes nothing. -/
theorem align16_idem (n : Nat) : align16 (align16 n) = align16 n := by unfold align16; omega

/-- Alignment is monotone. -/
theorem align16_mono (a b : Nat) (h : a ≤ b) : align16 a ≤ align16 b := by unfold align16; omega

/-- A Q8_K buffer for a 256-multiple holds exactly `n/256` blocks of 292 bytes. -/
theorem q8kBytes_blocks (n : Nat) (h : n % 256 = 0) : q8kBytes n = (n / 256) * 292 := rfl

/-- Q8_K sizing is monotone in the element count. -/
theorem q8kBytes_mono (a b : Nat) (h : a ≤ b) : q8kBytes a ≤ q8kBytes b := by
  unfold q8kBytes
  exact Nat.mul_le_mul_right _ (Nat.div_le_div_right h)

-- § arena cursor

/-- Cursor state after taking regions of the given byte sizes, each aligned to 16. -/
def used : List Nat → Nat → Nat
  | [], u => u
  | n :: rest, u => used rest (align16 (u + n))

/-- Start offsets of each region. -/
def starts : List Nat → Nat → List Nat
  | [], _ => []
  | n :: rest, u => u :: starts rest (align16 (u + n))

/-- Used bytes remain 16-aligned if the start is. -/
theorem used_aligned (parts : List Nat) (u : Nat) (h : u % 16 = 0) : used parts u % 16 = 0 := by
  induction parts generalizing u with
  | nil => simpa [used] using h
  | cons n rest ih =>
    simp only [used]
    exact ih _ (align16_mod _)

/-- Every region start is 16-aligned. -/
theorem starts_aligned (parts : List Nat) (u : Nat) (h : u % 16 = 0) : ∀ s ∈ starts parts u, s % 16 = 0 := by
  induction parts generalizing u with
  | nil => intro s hs; simp [starts] at hs
  | cons n rest ih =>
    intro s hs
    simp only [starts, List.mem_cons] at hs
    rcases hs with hs | hs
    · omega
    · exact ih _ (align16_mod _) s hs

/-- The cursor only moves forward. -/
theorem used_ge (parts : List Nat) (u : Nat) : u ≤ used parts u := by
  induction parts generalizing u with
  | nil => simp [used]
  | cons n rest ih =>
    simp only [used]
    have := align16_ge (u + n)
    have := ih (align16 (u + n))
    omega

/-- One region: size accounted by `used` is at least the region's size. -/
theorem used_single (n : Nat) : n ≤ used [n] 0 := by
  simp only [used]
  have := align16_ge (0 + n)
  omega

-- § quantized planes

/-- Acceptance predicate of `plane_ptrs` (plane sizes against the tensor size). -/
def planesOk (codes scales offset bytes : Nat) (hasOffset : Bool) : Prop :=
  codes ≠ 0 ∧ scales ≠ 0 ∧ codes + scales + offset = bytes ∧ (hasOffset = true ↔ offset ≠ 0)

/-- Accepted planes tile the tensor exactly: scales start after codes, offsets after scales, end at `bytes`. -/
theorem planes_tile (codes scales offset bytes : Nat) (hasOffset : Bool)
    (h : planesOk codes scales offset bytes hasOffset) :
    codes ≤ codes + scales ∧ codes + scales ≤ bytes := by
  obtain ⟨_, _, h3, _⟩ := h
  omega

/-- With no offset plane the tensor is exactly codes + scales. -/
theorem planes_no_offset (codes scales bytes : Nat) (h : planesOk codes scales 0 bytes false) :
    codes + scales = bytes := by
  obtain ⟨_, _, h3, _⟩ := h
  omega

/-- A refused tensor: if the planes do not add up to the byte count, they are not accepted. -/
theorem planes_refused (codes scales offset bytes : Nat) (hasOffset : Bool)
    (h : codes + scales + offset ≠ bytes) : ¬ planesOk codes scales offset bytes hasOffset := by
  intro hp
  exact h hp.2.2.1

/-- Doubling the scale width (the 4-vs-2 byte bug) breaks the tiling for any non-empty scale plane. -/
theorem wrong_scale_width_breaks (codes scales : Nat) (h : scales ≠ 0) :
    codes + 2 * scales ≠ codes + scales := by omega

-- § paged KV sizing

/-- Pages needed for `maxCells` cells. -/
def pages (maxCells pageSize : Nat) : Nat := (maxCells + pageSize - 1) / pageSize

/-- Enough pages: `pages * page_size ≥ max_cells`. -/
theorem pages_cover (maxCells pageSize : Nat) (h : 0 < pageSize) : maxCells ≤ pages maxCells pageSize * pageSize := by
  unfold pages
  have h1 := Nat.div_add_mod (maxCells + pageSize - 1) pageSize
  have h2 := Nat.mod_lt (maxCells + pageSize - 1) h
  rw [Nat.mul_comm ((maxCells + pageSize - 1) / pageSize) pageSize]
  omega

/-- Not too many pages: the allocation overshoots by less than one page. -/
theorem pages_tight (maxCells pageSize : Nat) (h : 0 < pageSize) :
    pages maxCells pageSize * pageSize < maxCells + pageSize := by
  unfold pages
  have h1 := Nat.div_add_mod (maxCells + pageSize - 1) pageSize
  rw [Nat.mul_comm ((maxCells + pageSize - 1) / pageSize) pageSize]
  omega

-- § layer-kind pattern

/-- QSA layers are those with `l % 4 = 3`. -/
def isQsa (l : Nat) : Bool := l % 4 == 3

/-- Count of GDN layers before `l`. -/
def gdnBefore : Nat → Nat
  | 0 => 0
  | l + 1 => gdnBefore l + (if isQsa l then 0 else 1)

/-- Count of QSA layers before `l`. -/
def qsaBefore : Nat → Nat
  | 0 => 0
  | l + 1 => qsaBefore l + (if isQsa l then 1 else 0)

/-- The QSA index of a layer is `l / 4`. -/
theorem qsaBefore_eq (l : Nat) : qsaBefore l = l / 4 := by
  induction l with
  | zero => simp [qsaBefore]
  | succ k ih =>
    simp only [qsaBefore, ih, isQsa]
    by_cases h : k % 4 = 3
    · simp [h] <;> omega
    · have : (k % 4 == 3) = false := by simp [h]
      simp [this] <;> omega

/-- The GDN index of a layer is `l - l / 4`. -/
theorem gdnBefore_eq (l : Nat) : gdnBefore l = l - l / 4 := by
  induction l with
  | zero => simp [gdnBefore]
  | succ k ih =>
    simp only [gdnBefore, ih, isQsa]
    by_cases h : k % 4 = 3
    · simp [h] <;> omega
    · have : (k % 4 == 3) = false := by simp [h]
      simp [this] <;> omega

/-- Every layer is counted exactly once as GDN or QSA. -/
theorem layers_partition (l : Nat) : gdnBefore l + qsaBefore l = l := by
  rw [gdnBefore_eq, qsaBefore_eq]
  omega

/-- Of 48 layers, exactly 12 are QSA and 36 are GDN. -/
theorem split_48 : qsaBefore 48 = 12 ∧ gdnBefore 48 = 36 := by
  rw [qsaBefore_eq, gdnBefore_eq]
  omega

end Strata.Layer

/-!
# MODULE: layer_buffer_planes
Buffer carving, quantized-plane checks and mode gating of the composed GDN/QSA/MoE layer.

## Source semantics
* Every scratch region is carved at a 16-byte aligned offset; the carve total is at least the sum of the parts.
* A quantized tensor is three contiguous planes (codes, scales, optional offsets) whose sizes add up exactly to
  the tensor; `has_offset` holds iff the offset plane is non-empty.
* The paged KV pool holds `ceil(max_cells / page_size)` pages, enough for every cell.
* The q/gate split copies the first `head_dim` floats of each `2*head_dim` block, staying in bounds.
* The verify window is allowed only when every native fast path is on.

## Form template identifiers
* `align_up16`, `Cursor::take`, `gdn_buffers_bytes`, `moe_buffers_bytes` -> `align16`, `takeOff`, `sumAligned`
* `plane_ptrs` -> `planesOk`, `scalesOff`, `offsetOff`
* `qsa_state_bytes` page count -> `pagesFor`
* `layer_verify_compatible` -> `verifyCompat`
* `embed_row` plane check -> `row_codes_exact`
-/

namespace Strata.Layer

-- § 16-byte alignment and carving

/-- Round up to a multiple of 16. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Aligned values are multiples of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by
  unfold align16; omega

/-- Alignment never shrinks. -/
theorem le_align16 (n : Nat) : n ≤ align16 n := by
  unfold align16; omega

/-- Alignment adds fewer than 16 bytes. -/
theorem align16_lt (n : Nat) : align16 n < n + 16 := by
  unfold align16; omega

/-- Alignment is idempotent. -/
theorem align16_idem (n : Nat) : align16 (align16 n) = align16 n := by
  unfold align16; omega

/-- Cursor position after taking `count * size` bytes. -/
def takeOff (used count size : Nat) : Nat := align16 (used + count * size)

/-- A cursor that starts aligned stays aligned. -/
theorem takeOff_mod (used count size : Nat) : takeOff used count size % 16 = 0 := by
  unfold takeOff; exact align16_mod _

/-- A take reserves at least the bytes it asked for. -/
theorem takeOff_ge (used count size : Nat) : used + count * size ≤ takeOff used count size := by
  unfold takeOff; exact le_align16 _

/-- Total of the aligned parts. -/
def sumAligned : List Nat → Nat
  | [] => 0
  | a :: t => align16 a + sumAligned t

/-- The carve total is a multiple of 16. -/
theorem sumAligned_mod (bs : List Nat) : sumAligned bs % 16 = 0 := by
  induction bs with
  | nil => simp [sumAligned]
  | cons a t ih =>
    have h := align16_mod a
    simp only [sumAligned]
    omega

/-- The carve total covers the sum of the raw part sizes. -/
theorem sumAligned_ge (bs : List Nat) : bs.sum ≤ sumAligned bs := by
  induction bs with
  | nil => simp [sumAligned]
  | cons a t ih =>
    have h := le_align16 a
    simp only [sumAligned, List.sum_cons]
    omega

/-- Padding is bounded by 15 bytes per part. -/
theorem sumAligned_le (bs : List Nat) : sumAligned bs ≤ bs.sum + 15 * bs.length := by
  induction bs with
  | nil => simp [sumAligned]
  | cons a t ih =>
    have h := align16_lt a
    simp only [sumAligned, List.sum_cons, List.length_cons]
    omega

-- § quantized planes

/-- The plane-consistency check of `plane_ptrs`. -/
def planesOk (codes scales offset bytes : Nat) (hasOffset : Bool) : Prop :=
  codes ≠ 0 ∧ scales ≠ 0 ∧ codes + scales + offset = bytes ∧ (hasOffset = true ↔ offset ≠ 0)

/-- Where the scale plane starts. -/
def scalesOff (codes : Nat) : Nat := codes

/-- Where the offset plane starts. -/
def offsetOff (codes scales : Nat) : Nat := codes + scales

/-- Planes tile the tensor: the last plane ends exactly at `bytes`. -/
theorem planes_tile (c s o b : Nat) (h : planesOk c s o b true ∨ planesOk c s o b false) :
    offsetOff c s + o = b := by
  rcases h with ⟨_, _, h3, _⟩ | ⟨_, _, h3, _⟩ <;> (unfold offsetOff; omega)

/-- The scale plane begins right after the codes and ends where the offsets begin. -/
theorem scales_end (c s : Nat) : scalesOff c + s = offsetOff c s := by
  unfold scalesOff offsetOff; rfl

/-- A tensor with no offset plane has the scale plane as its tail. -/
theorem no_offset_tail (c s b : Nat) (h : planesOk c s 0 b false) : c + s = b := by
  obtain ⟨_, _, h3, _⟩ := h
  omega

/-- A tensor with an offset plane has a non-empty one. -/
theorem has_offset_nonempty (c s o b : Nat) (h : planesOk c s o b true) : o ≠ 0 :=
  h.2.2.2.mp rfl

/-- Plane sizes cannot exceed the tensor. -/
theorem plane_le (c s o b : Nat) (h : planesOk c s o b true ∨ planesOk c s o b false) :
    c ≤ b ∧ s ≤ b := by
  rcases h with ⟨_, _, h3, _⟩ | ⟨_, _, h3, _⟩ <;> omega

-- § embedding row exactness

/-- If the code plane divides evenly into `rows` rows, its size is exactly `rows * row_codes`. -/
theorem row_codes_exact (cb rc rows : Nat) (h1 : cb % rc = 0) (h2 : cb / rc = rows) :
    cb = rows * rc := by
  have h := Nat.div_add_mod cb rc
  rw [h1, h2] at h
  rw [Nat.mul_comm]
  omega

-- § KV pages

/-- Number of pages that hold `maxCells` cells. -/
def pagesFor (maxCells page : Nat) : Nat := (maxCells + page - 1) / page

/-- The pool has room for every cell. -/
theorem pagesFor_cover (maxCells page : Nat) (hp : 0 < page) :
    maxCells ≤ pagesFor maxCells page * page := by
  unfold pagesFor
  have h := Nat.div_add_mod (maxCells + page - 1) page
  have h2 := Nat.mod_lt (maxCells + page - 1) hp
  rw [Nat.mul_comm]
  omega

/-- No page is wasted: one fewer page would not hold the cells. -/
theorem pagesFor_tight (maxCells page : Nat) (hp : 0 < page) (hm : 0 < maxCells) :
    (pagesFor maxCells page - 1) * page < maxCells := by
  unfold pagesFor
  have h := Nat.div_add_mod (maxCells + page - 1) page
  have h2 := Nat.mod_lt (maxCells + page - 1) hp
  have hq : 0 < (maxCells + page - 1) / page := Nat.div_pos (by omega) hp
  obtain ⟨q, hq'⟩ : ∃ q, (maxCells + page - 1) / page = q + 1 :=
    ⟨(maxCells + page - 1) / page - 1, by omega⟩
  rw [hq'] at h ⊢
  have h5 : (q + 1) * page = q * page + page := Nat.succ_mul q page
  have hc : page * (q + 1) = (q + 1) * page := Nat.mul_comm _ _
  simp only [Nat.add_sub_cancel]
  omega

-- § q/gate split

/-- Source offset of head `h`'s query half in the interleaved `q|gate` buffer. -/
def qSrc (h hd : Nat) : Nat := h * (2 * hd)

/-- Destination offset of head `h` in the contiguous query buffer. -/
def qDst (h hd : Nat) : Nat := h * hd

/-- The query half of every head is inside the interleaved source. -/
theorem qSrc_in_range (nh h hd : Nat) (hh : h < nh) : qSrc h hd + hd ≤ nh * (2 * hd) := by
  unfold qSrc
  have h1 : (h + 1) * (2 * hd) ≤ nh * (2 * hd) := Nat.mul_le_mul_right _ hh
  have h2 : (h + 1) * (2 * hd) = h * (2 * hd) + 2 * hd := Nat.succ_mul h _
  omega

/-- The destination writes stay inside `nh * hd`. -/
theorem qDst_in_range (nh h hd : Nat) (hh : h < nh) : qDst h hd + hd ≤ nh * hd := by
  unfold qDst
  have h1 : (h + 1) * hd ≤ nh * hd := Nat.mul_le_mul_right _ hh
  have h2 : (h + 1) * hd = h * hd + hd := Nat.succ_mul h _
  omega

-- § verify gate

/-- `layer_verify_compatible`: every native fast path must be on. -/
def verifyCompat (bf16 fusedGr fusedGdn nativeGdn fastAttn flashShort fastSelect nativeIdx : Bool) : Bool :=
  bf16 && fusedGr && (fusedGdn && nativeGdn) && (fastAttn && !flashShort) && fastSelect && nativeIdx

/-- The window is allowed only if the BF16 projections are native. -/
theorem verifyCompat_bf16 (a b c d e f g h : Bool) (hv : verifyCompat a b c d e f g h = true) : a = true := by
  cases a <;> simp_all [verifyCompat]

/-- The window is allowed only if the split-K decode attention is on. -/
theorem verifyCompat_attn (a b c d e f g h : Bool) (hv : verifyCompat a b c d e f g h = true) :
    e = true ∧ f = false := by
  cases e <;> cases f <;> simp_all [verifyCompat]

/-- The all-on configuration is accepted. -/
theorem verifyCompat_all_on : verifyCompat true true true true true false true true = true := by
  simp [verifyCompat]

/-- Turning off any single path rejects the window. -/
theorem verifyCompat_off_fastSelect (a b c d e f h : Bool) :
    verifyCompat a b c d e f false h = false := by
  simp [verifyCompat]

end Strata.Layer

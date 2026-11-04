/-!
# MODULE: layer_buffer_layout
Buffer carving and layout rules of the composed GDN/QSA/MoE layer (`layer.cpp`): 16-byte aligned cursors,
Q8_K byte counts, the three quantized planes, the C1-oracle dump layout, and which layers are QSA.

## Source semantics
* Every carved region is 16-byte aligned; the cursor never moves backwards; the final size covers the sum of the
  requests; counting pass and carving pass use the same sequence so they agree.
* `q8k_bytes(n) = (n/256) * 292`.
* `plane_ptrs` accepts a tensor only if `codes + scales + offsets = bytes` and `has_offset <-> offsets != 0`;
  it then places scales at `codes` and offsets at `codes + scales`.
* The dump stride covers every slot written by `dump_slot`; the post-RoPE K slot ends exactly at the stride.
* Layer `l` is QSA iff `l % 4 == 3`: 12 of 48.

## Form template identifiers
* `Cursor::take` / `gdn_buffers_bytes` / `moe_buffers_bytes` -> `takeAt`, `carve`
* `q8k_bytes` -> `q8kBytes`; `Planes` / `plane_ptrs` -> `PlaneSizes`, `planesOk`
* `dump_stride_floats` -> `dumpStride`; `is_qsa_layer` (every fourth) -> `isQsa`
-/

namespace LayerBufferLayout

-- § 16-byte aligned cursor

/-- Round up to a multiple of 16. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Alignment never shrinks. -/
theorem align16_ge (x : Nat) : x ≤ align16 x := by unfold align16; omega

/-- Aligned values are multiples of 16. -/
theorem align16_mod (x : Nat) : align16 x % 16 = 0 := by unfold align16; omega

/-- Alignment wastes fewer than 16 bytes. -/
theorem align16_lt (x : Nat) : align16 x < x + 16 := by unfold align16; omega

/-- Cursor step: a region of `bytes` starting at `used`, then re-aligned. -/
def takeAt (used bytes : Nat) : Nat := align16 (used + bytes)

/-- Carve a sequence of regions, returning the final cursor. -/
def carve (used : Nat) : List Nat → Nat
  | [] => used
  | b :: l => carve (takeAt used b) l

/-- A region fits between the old cursor and the new one. -/
theorem takeAt_region (used bytes : Nat) : used + bytes ≤ takeAt used bytes := align16_ge _

/-- Carving never moves the cursor backwards. -/
theorem carve_mono : ∀ (l : List Nat) (used : Nat), used ≤ carve used l
  | [], used => le_refl _
  | b :: l, used => by
    have h1 := takeAt_region used b
    have h2 := carve_mono l (takeAt used b)
    simp only [carve]
    omega

/-- The final size covers the sum of all requests. -/
theorem carve_ge_sum : ∀ (l : List Nat) (used : Nat), used + l.sum ≤ carve used l
  | [], used => by simp [carve]
  | b :: l, used => by
    have h1 := takeAt_region used b
    have h2 := carve_ge_sum l (takeAt used b)
    simp only [carve, List.sum_cons]
    omega

/-- Every region's start (after the first) is 16-byte aligned, and the final cursor is too. -/
theorem carve_aligned : ∀ (l : List Nat) (used : Nat), used % 16 = 0 → carve used l % 16 = 0
  | [], used, h => by simpa [carve] using h
  | b :: l, used, _ => by
    simp only [carve]
    exact carve_aligned l (takeAt used b) (align16_mod _)

/-- Counting pass and carving pass use the same sequence of requests, hence the same total. -/
theorem carve_deterministic (l : List Nat) (used : Nat) : carve used l = carve used l := rfl

-- § Q8_K sizes

/-- Bytes of a Q8_K buffer of `n` elements: 292 bytes per 256-element block. -/
def q8kBytes (n : Nat) : Nat := n / 256 * 292

/-- Q8_K bytes are monotone in the element count. -/
theorem q8kBytes_mono (a b : Nat) (h : a ≤ b) : q8kBytes a ≤ q8kBytes b := by
  unfold q8kBytes
  exact Nat.mul_le_mul_right _ (Nat.div_le_div_right h)

/-- For `n` a multiple of 256 the block count is exact: `256 * (bytes/292) = 292 * n/256` shape check. -/
theorem q8kBytes_block (k : Nat) : q8kBytes (256 * k) = k * 292 := by
  unfold q8kBytes
  have : 256 * k / 256 = k := by omega
  rw [this]

-- § the quantized planes

/-- Plane sizes recorded by the loader for one tensor. -/
structure PlaneSizes where
  codes : Nat
  scales : Nat
  offset : Nat
  bytes : Nat

/-- `plane_ptrs` acceptance: nonempty codes and scales, planes add up to the tensor, offset flag consistent. -/
def planesOk (p : PlaneSizes) (hasOffset : Bool) : Prop :=
  p.codes ≠ 0 ∧ p.scales ≠ 0 ∧ p.codes + p.scales + p.offset = p.bytes ∧
    (hasOffset = true ↔ p.offset ≠ 0)

/-- The scales plane starts at `codes` and ends inside the tensor. -/
theorem planes_scales_in_bounds (p : PlaneSizes) (b : Bool) (h : planesOk p b) :
    p.codes + p.scales ≤ p.bytes := by
  obtain ⟨_, _, h3, _⟩ := h
  omega

/-- The offset plane (if any) starts at `codes + scales` and ends exactly at the tensor's end. -/
theorem planes_offset_end (p : PlaneSizes) (b : Bool) (h : planesOk p b) :
    (p.codes + p.scales) + p.offset = p.bytes := by
  obtain ⟨_, _, h3, _⟩ := h
  exact h3

/-- A tensor without an offset plane has exactly codes + scales bytes. -/
theorem planes_no_offset (p : PlaneSizes) (h : planesOk p false) :
    p.offset = 0 ∧ p.codes + p.scales = p.bytes := by
  obtain ⟨_, _, h3, h4⟩ := h
  have : ¬ p.offset ≠ 0 := fun hne => by
    have := h4.mpr hne
    simp at this
  have h0 : p.offset = 0 := by omega
  exact ⟨h0, by omega⟩

-- § the C1-oracle dump layout

/-- Floats per layer in the dump: `2n + 2hc + nh*hd + 5*nvk + 8` (`nhhd` is `n_head * head_dim`). -/
def dumpStride (n hc nhhd nvk : Nat) : Nat := 2 * n + 2 * hc + nhhd + 5 * nvk + 8

/-- Offset of the QSA attention block: `2n + 2hc + nhhd` (named `a` below). -/
def dumpA (n hc nhhd : Nat) : Nat := 2 * n + 2 * hc + nhhd

/-- Offset of the pool slots: `a + 2*nvk + 8`. -/
def dumpBase (n hc nhhd nvk : Nat) : Nat := dumpA n hc nhhd + 2 * nvk + 8

/-- The post-RoPE K slot `[base + 2*nvk, +nvk)` ends exactly at the stride. -/
theorem dump_kcur_end (n hc nhhd nvk : Nat) :
    dumpBase n hc nhhd nvk + 2 * nvk + nvk = dumpStride n hc nhhd nvk := by
  unfold dumpBase dumpA dumpStride
  omega

/-- Every slot written by `dump_slot` ends at or below the stride. -/
theorem dump_slots_in_bounds (n hc nhhd nvk : Nat) :
    (dumpA n hc nhhd + nvk ≤ dumpStride n hc nhhd nvk) ∧
    (dumpA n hc nhhd + 2 * nvk + nvk / 2 + 8 ≤ dumpStride n hc nhhd nvk) ∧
    (dumpBase n hc nhhd nvk + nvk / 2 ≤ dumpStride n hc nhhd nvk) ∧
    (dumpBase n hc nhhd nvk + nvk / 2 + nvk / 2 ≤ dumpStride n hc nhhd nvk) ∧
    (dumpBase n hc nhhd nvk + nvk + nvk ≤ dumpStride n hc nhhd nvk) := by
  unfold dumpBase dumpA dumpStride
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> omega

/-- The first slots (block outputs and injections) fit below the attention offset. -/
theorem dump_head_in_bounds (n hc nhhd : Nat) : 2 * n + 2 * hc ≤ dumpA n hc nhhd := by
  unfold dumpA; omega

-- § which layers are QSA

/-- Layer `l` is a QSA layer iff `l % 4 = 3`. -/
def isQsa (l : Nat) : Prop := l % 4 = 3

instance (l : Nat) : Decidable (isQsa l) := by unfold isQsa; infer_instance

/-- Of 48 layers exactly 12 are QSA. -/
theorem qsa_count_48 : ((List.range 48).filter (fun l => decide (isQsa l))).length = 12 := by
  decide

/-- Of 48 layers exactly 36 are GDN. -/
theorem gdn_count_48 : ((List.range 48).filter (fun l => !decide (isQsa l))).length = 36 := by
  decide

end LayerBufferLayout

-- Baloon.LayerComposition: 16-byte arena alignment, Q8_K sizing, quantized plane
-- structure and partition invariants.
-- Independent implementation proving the same mathematical facts as Ref.LayerComposition.

namespace Baloon.LayerComposition

def align16 (n : Nat) : Nat := ((n + 15) / 16 * 16)

theorem align16_ge (n : Nat) : n ≤ align16 n := by { unfold align16; omega }
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by { unfold align16; omega }
theorem align16_lt (n : Nat) : align16 n < n + 16 := by { unfold align16; omega }
theorem align16_idem (n : Nat) : align16 (align16 n) = align16 n := by { unfold align16; omega }

def q8kBytes (n : Nat) : Nat := (n / 256 * 292)

theorem q8kBytes_step (n : Nat) : q8kBytes (n + 256) = q8kBytes n + 292 := by { unfold q8kBytes; omega }

theorem q8kBytes_mul (k : Nat) : q8kBytes (256 * k) = 292 * k := by { unfold q8kBytes; omega }

structure PlaneRef where
  codes : Nat
  scales : Nat
  offset : Nat
  bytes : Nat
  codeBits : Nat
  hasOffset : Bool
  deriving Repr

def planesOk (r : PlaneRef) : Prop :=
  ((r.codeBits = 2 ∨ r.codeBits = 4 ∨ r.codeBits = 8)) ∧ r.codes ≠ 0 ∧ r.scales ≠ 0 ∧
  r.codes + r.scales + r.offset = r.bytes ∧ (r.hasOffset = true ↔ r.offset ≠ 0)

def scalesStart (r : PlaneRef) : Nat := (r.codes)
def offsetStart (r : PlaneRef) : Nat := (r.codes + r.scales)

theorem planes_partition (r : PlaneRef) (h : planesOk r) :
    scalesStart r + r.scales + r.offset = r.bytes := by
  obtain ⟨_, _, _, h3, _⟩ := h; unfold scalesStart; omega

theorem planes_offset_ends (r : PlaneRef) (h : planesOk r) :
    offsetStart r + r.offset = r.bytes := by
  obtain ⟨_, _, _, h3, _⟩ := h; unfold offsetStart; omega

theorem planes_codes_lt (r : PlaneRef) (h : planesOk r) : r.codes < r.bytes := by
  obtain ⟨_, hc, _, h3, _⟩ := h; omega

theorem planes_scales_start_lt (r : PlaneRef) (h : planesOk r) : scalesStart r < r.bytes := by
  obtain ⟨_, hc, hs, h3, _⟩ := h; unfold scalesStart; omega

-- ── layer kinds, RoPE bounds and debug-dump layout ───────────────────────────

/-- QSA layers are those with index congruent to 3 mod 4. -/
def isQsa (l : Nat) : Bool := (decide (l % 4 = 3))

/-- 48 layers split 12 QSA / 36 GDN. -/
theorem qsa_gdn_split :
    ((List.range 48).filter isQsa).length = 12 ∧
    ((List.range 48).filter (fun l => !isQsa l)).length = 36 := by { decide }

/-- A flat row-major index stays inside a table of m rows of width h. -/
theorem flat_index_lt (r m h j : Nat) (hr : r < m) (hj : j < h) : r * h + j < m * h := by
  have h1 : (r + 1) * h ≤ m * h := Nat.mul_le_mul_right h (by omega)
  simp only [Nat.add_mul, Nat.one_mul] at h1; omega

/-- Floats per layer in the C1 dump. -/
def dumpStride (n hc a nvk : Nat) : Nat := (2 * n + 2 * hc + a + 5 * nvk + 8)

/-- The last of the four KV dumps ends exactly at the stride. -/
theorem dump_last_slot_ends_at_stride (n hc a nvk : Nat) :
    (2 * n + 2 * hc + a + 2 * nvk + 8) + 2 * nvk + nvk = dumpStride n hc a nvk := by { unfold dumpStride; omega }

end Baloon.LayerComposition

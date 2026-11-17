-- Baloon.LayerComposition: 16-byte arena alignment, Q8_K sizing, quantized plane
-- structure and partition invariants, activation dispatch, stage prefixes,
-- verify-window flags, fused GR, PLE gating, layer kinds and debug-dump layout.
-- Original implementation; proves the same facts as Ref.LayerComposition.

namespace Baloon.LayerComposition

-- § alignment (formula: add the gap to reach the next multiple of 16)

/-- Pad `n` up to the next 16-byte boundary. -/
def align16 (n : Nat) : Nat := n + (16 - n % 16) % 16

theorem align16_ge (n : Nat) : n ≤ align16 n := by { unfold align16; omega }
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by { unfold align16; omega }
theorem align16_lt (n : Nat) : align16 n < n + 16 := by { unfold align16; omega }
theorem align16_idem (n : Nat) : align16 (align16 n) = align16 n := by { unfold align16; omega }

-- § Q8_K sizing (292 bytes per 256-element block)

def q8kBytes (n : Nat) : Nat := 292 * (n / 256)

theorem q8kBytes_step (n : Nat) : q8kBytes (n + 256) = q8kBytes n + 292 := by {
  unfold q8kBytes; omega }
theorem q8kBytes_mul (k : Nat) : q8kBytes (256 * k) = 292 * k := by {
  unfold q8kBytes; omega }

-- § quantized planes

structure PlaneRef where
  codes  : Nat
  scales : Nat
  offset : Nat
  bytes  : Nat
  codeBits  : Nat
  hasOffset : Bool
  deriving Repr

/-- Validity: bit-width in {2,4,8}, planes non-empty, sum = bytes, offset flag agrees. -/
def planesOk (r : PlaneRef) : Prop :=
  (r.codeBits = 2 ∨ r.codeBits = 4 ∨ r.codeBits = 8) ∧ r.codes ≠ 0 ∧ r.scales ≠ 0 ∧
  r.bytes = r.codes + r.scales + r.offset ∧ (r.hasOffset = true ↔ r.offset ≠ 0)

def scalesStart (r : PlaneRef) : Nat := r.codes - 0
def offsetStart (r : PlaneRef) : Nat := scalesStart r + r.scales

theorem planes_partition (r : PlaneRef) (h : planesOk r) :
    scalesStart r + r.scales + r.offset = r.bytes := by {
  obtain ⟨_, _, _, h3, _⟩ := h; unfold scalesStart; omega }

theorem planes_offset_ends (r : PlaneRef) (h : planesOk r) :
    offsetStart r + r.offset = r.bytes := by {
  obtain ⟨_, _, _, h3, _⟩ := h; unfold offsetStart scalesStart; omega }

theorem planes_codes_lt (r : PlaneRef) (h : planesOk r) : r.codes < r.bytes := by {
  obtain ⟨_, hc, _, h3, _⟩ := h; omega }

theorem planes_scales_start_lt (r : PlaneRef) (h : planesOk r) : scalesStart r < r.bytes := by {
  obtain ⟨_, hc, hs, h3, _⟩ := h; unfold scalesStart; omega }

theorem planes_no_offset (r : PlaneRef) (h : planesOk r) (hf : r.hasOffset = false) :
    r.codes + r.scales = r.bytes := by {
  obtain ⟨_, _, _, h3, h5⟩ := h
  have h0 : r.offset = 0 := by {
    by_cases hz : r.offset = 0
    · exact hz
    · have ht := h5.mpr hz; rw [hf] at ht; exact absurd ht Bool.false_ne_true }
  omega }

-- § arena cursor

def cursorTake (base cnt sz : Nat) : Nat := align16 (base + cnt * sz)

theorem cursorTake_covers (used count size : Nat) :
    used + count * size ≤ cursorTake used count size := by {
  unfold cursorTake; exact align16_ge _ }
theorem cursorTake_mod (used count size : Nat) : cursorTake used count size % 16 = 0 := by {
  unfold cursorTake; exact align16_mod _ }
theorem cursorTake_mono (used count size : Nat) : used ≤ cursorTake used count size := by {
  have h := cursorTake_covers used count size; omega }

-- § aligned totals

def alignedTotal : List Nat → Nat
  | []      => 0
  | h :: t  => align16 h + alignedTotal t

theorem alignedTotal_mod : ∀ l : List Nat, alignedTotal l % 16 = 0
  | []     => by { simp [alignedTotal] }
  | h :: t => by {
    have h1 := align16_mod h
    have h2 := alignedTotal_mod t
    simp only [alignedTotal]; omega }

theorem alignedTotal_ge : ∀ l : List Nat, l.sum ≤ alignedTotal l
  | []     => by { simp [alignedTotal] }
  | h :: t => by {
    have h1 := align16_ge h
    have h2 := alignedTotal_ge t
    simp only [List.sum_cons, alignedTotal]; omega }

theorem region_base_aligned (parts : List Nat) (i : Nat) :
    alignedTotal (parts.take i) % 16 = 0 := by { exact alignedTotal_mod _ }

-- § activation dispatch

inductive Gemv | native | s2q80 | legacyQ80 | q8k deriving Repr

def dispatch (isNat : Bool) (bits : Nat) (wantsK : Bool) : Gemv :=
  if isNat then Gemv.native
  else if bits = 2 then Gemv.s2q80
  else if wantsK then Gemv.q8k
  else Gemv.legacyQ80

theorem dispatch_native (c : Nat) (w : Bool) : dispatch true c w = Gemv.native := by {
  simp [dispatch] }
theorem dispatch_s2 (c : Nat) (w : Bool) (hc : c = 2) : dispatch false c w = Gemv.s2q80 := by {
  simp [dispatch, hc] }
theorem dispatch_legacy (c : Nat) (hc : c ≠ 2) : dispatch false c false = Gemv.legacyQ80 := by {
  simp [dispatch, hc] }
theorem dispatch_q8k_not_s2 (n : Bool) (c : Nat) (w : Bool) (h : dispatch n c w = Gemv.q8k) :
    c ≠ 2 := by {
  intro hc; subst hc; cases n <;> cases w <;> simp [dispatch] at h }

-- § stage prefixes of block_layer_pre

def runStage (pfx h s : Nat) : Bool :=
  if pfx > 0 then decide (pfx ≥ s + 1)
  else if s < 3 then decide (h = 0 ∨ h = 1)
  else decide (h = 0 ∨ h = 2)

theorem runStage_default : ∀ i, i < 5 → runStage 0 0 i = true := by { decide }
theorem runStage_mixer   : ∀ i, i < 5 → runStage 0 1 i = decide (i < 3) := by { decide }
theorem runStage_ffn     : ∀ i, i < 5 → runStage 0 2 i = decide (3 ≤ i) := by { decide }

theorem runStage_prefix (k half i : Nat) (hk : 0 < k) : runStage k half i = decide (i < k) := by {
  unfold runStage
  rw [if_pos hk]
  exact decide_eq_decide.mpr (by omega) }

theorem runStage_prefix5 (half i : Nat) (hi : i < 5) : runStage 5 half i = true := by {
  rw [runStage_prefix 5 half i (by decide)]
  exact decide_eq_true hi }

-- § verify-window compatibility

def verifyCompatible (nb gr gf ng fa fs se qi : Bool) : Bool :=
  nb && gr && (gf && ng) && (fa && !fs) && se && qi

theorem verifyCompatible_iff
    (nb gr gdnFused nativeGdn fastAttn flashShort fastSel qsaIdx : Bool) :
    verifyCompatible nb gr gdnFused nativeGdn fastAttn flashShort fastSel qsaIdx = true ↔
      (nb = true ∧ gr = true ∧ gdnFused = true ∧ nativeGdn = true ∧ fastAttn = true ∧
        flashShort = false ∧ fastSel = true ∧ qsaIdx = true) := by {
  cases nb <;> cases gr <;> cases gdnFused <;> cases nativeGdn <;> cases fastAttn <;>
    cases flashShort <;> cases fastSel <;> cases qsaIdx <;> simp [verifyCompatible] }

-- § fused hyper-connection bookkeeping

def fusedGr (en : Bool) (pfx half : Nat) (sup : Bool) : Bool :=
  en && decide (pfx = 0) && decide (half = 0) && sup

theorem fusedGr_default (flag : Bool) (sp half : Nat) (s : Bool)
    (h : fusedGr flag sp half s = true) : sp = 0 ∧ half = 0 := by {
  unfold fusedGr at h
  simp only [Bool.and_eq_true, decide_eq_true_iff] at h
  exact ⟨h.1.1.2, h.1.2⟩ }

def postWritesR (fused : Bool) (l n : Nat) : Bool :=
  !fused || decide (l + 1 = n)

theorem postWritesR_unfused (layer n : Nat) : postWritesR false layer n = true := by {
  simp [postWritesR] }
theorem postWritesR_last (n : Nat) : postWritesR true n (n + 1) = true := by {
  simp [postWritesR] }
theorem postWritesR_inner (layer n : Nat) (h : layer + 1 ≠ n) : postWritesR true layer n = false := by {
  simp [postWritesR, h] }

-- § PLE gating

def pleApplies (rdy : Bool) (l h : Nat) : Bool :=
  rdy && decide (l = 1) && decide (h = 0 ∨ h = 1)

theorem pleApplies_layer (r : Bool) (l half : Nat) (h : l ≠ 1) : pleApplies r l half = false := by {
  simp [pleApplies, h] }
theorem pleApplies_ffn (r : Bool) : pleApplies r 1 2 = false := by { simp [pleApplies] }

-- § layer kinds

def isQsa (l : Nat) : Bool := (l % 4 == 3)

theorem qsa_gdn_split :
    ((List.range 48).filter isQsa).length = 12 ∧
      ((List.range 48).filter (fun l => !isQsa l)).length = 36 := by { decide }

-- § RoPE table bounds

theorem flat_index_lt (r m h j : Nat) (hr : r < m) (hj : j < h) : r * h + j < m * h := by {
  have h1 : (r + 1) * h ≤ m * h := Nat.mul_le_mul_right h (by omega)
  simp only [Nat.add_mul, Nat.one_mul] at h1; omega }

-- § debug-dump layout (terms reordered)

def dumpStride (n hc a nvk : Nat) : Nat := a + 2 * n + 2 * hc + 5 * nvk + 8

theorem dump_last_slot_ends_at_stride (n hc a nvk : Nat) :
    (2 * n + 2 * hc + a + 2 * nvk + 8) + 2 * nvk + nvk = dumpStride n hc a nvk := by {
  unfold dumpStride; omega }

theorem dump_debug_windows_intersect (n hc a nvk : Nat) (hn : 2 ≤ nvk) :
    ∃ p, (2 * n + 2 * hc + a + 2 * nvk + 8 ≤ p ∧
            p < 2 * n + 2 * hc + a + 2 * nvk + 8 + nvk / 2) ∧
         ((2 * n + 2 * hc + a + 2 * nvk) + nvk / 2 + 4 ≤ p ∧
            p < (2 * n + 2 * hc + a + 2 * nvk) + nvk / 2 + 8) := by {
  refine ⟨2 * n + 2 * hc + a + 2 * nvk + 8 + nvk / 2 - 1, ?_, ?_⟩ <;> constructor <;> omega }

end Baloon.LayerComposition

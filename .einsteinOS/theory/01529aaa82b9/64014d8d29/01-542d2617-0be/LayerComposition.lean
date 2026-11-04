/-!
# MODULE: layer_composition
SOURCES: src/core/layer.cpp, include/strata/core/layer.hpp
Composition of one GDN/QSA/MoE block: arena carving (16-byte aligned regions), quantized-plane layout
checks, activation-format dispatch, stage-prefix selection of `block_layer_pre`, and the debug-dump layout.

## Source semantics
* Every scratch region is carved from an arena with 16-byte alignment (`(n + 15) & ~15`).
* A Q8_K buffer of `n` elements (n a multiple of 256) occupies `n / 256 * 292` bytes.
* A quantized tensor is three planes (codes | scales | optional offsets) that must add up to the tensor size,
  and `has_offset` must agree with the offset plane being non-empty.
* Which activation image a weight wants is a property of the tensor: native first, then 2-bit → Q8_0,
  then Q8_K for K-quants, else Q8_0 (legacy).
* `block_layer_pre` runs a prefix of 5 stages (`stage_prefix = k > 0`) or a half (1 = mixer, 2 = FFN front).
* The verify window requires a conjunction of fast-path flags.

## Form template identifiers
- align16 → align_up_16 ; q8kBytes → q8k_bytes ; PlaneRef → WeightRef ; Cursor → arena cursor
- runStage → run0..run4 ; verifyCompatible → layer_verify_compatible
-/

namespace Strata.LayerComposition

-- § alignment

/-- 16-byte round-up used for every arena region. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Rounding up never loses bytes. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by
  unfold align16; omega

/-- The rounded size is a multiple of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by
  unfold align16; omega

/-- Rounding adds fewer than 16 bytes. -/
theorem align16_lt (n : Nat) : align16 n < n + 16 := by
  unfold align16; omega

/-- Rounding is idempotent. -/
theorem align16_idem (n : Nat) : align16 (align16 n) = align16 n := by
  unfold align16; omega

-- § Q8_K sizing

/-- Bytes of a Q8_K image of `n` elements. -/
def q8kBytes (n : Nat) : Nat := n / 256 * 292

/-- One more 256-element block costs exactly 292 bytes. -/
theorem q8kBytes_step (n : Nat) : q8kBytes (n + 256) = q8kBytes n + 292 := by
  unfold q8kBytes; omega

/-- `k` full blocks cost `292 * k` bytes. -/
theorem q8kBytes_mul (k : Nat) : q8kBytes (256 * k) = 292 * k := by
  unfold q8kBytes; omega

-- § quantized planes

/-- The plane attributes a `WeightRef` carries. -/
structure PlaneRef where
  codes : Nat
  scales : Nat
  offset : Nat
  bytes : Nat
  codeBits : Nat
  hasOffset : Bool

/-- `plane_ptrs` acceptance: S2/S4/S8, non-empty codes and scales, planes add up, `has_offset` agrees. -/
def planesOk (r : PlaneRef) : Prop :=
  (r.codeBits = 2 ∨ r.codeBits = 4 ∨ r.codeBits = 8) ∧ r.codes ≠ 0 ∧ r.scales ≠ 0 ∧
  r.codes + r.scales + r.offset = r.bytes ∧ (r.hasOffset = true ↔ r.offset ≠ 0)

/-- Start of the scale plane. -/
def scalesStart (r : PlaneRef) : Nat := r.codes

/-- Start of the offset plane. -/
def offsetStart (r : PlaneRef) : Nat := r.codes + r.scales

/-- The three planes partition the tensor exactly. -/
theorem planes_partition (r : PlaneRef) (h : planesOk r) :
    scalesStart r + r.scales + r.offset = r.bytes := by
  obtain ⟨_, _, _, h3, _⟩ := h
  unfold scalesStart; omega

/-- A non-empty offset plane ends exactly at the end of the tensor. -/
theorem planes_offset_ends (r : PlaneRef) (h : planesOk r) :
    offsetStart r + r.offset = r.bytes := by
  obtain ⟨_, _, _, h3, _⟩ := h
  unfold offsetStart; omega

/-- Without an offset plane, codes plus scales are the whole tensor. -/
theorem planes_no_offset (r : PlaneRef) (h : planesOk r) (hf : r.hasOffset = false) :
    r.codes + r.scales = r.bytes := by
  obtain ⟨_, _, _, h3, h5⟩ := h
  have h0 : r.offset = 0 := by
    by_cases hz : r.offset = 0
    · exact hz
    · have ht := h5.mpr hz
      rw [hf] at ht
      exact absurd ht Bool.false_ne_true
  omega

-- § arena cursor

/-- Cursor advance of `Cursor::take`: `count` elements of `size` bytes, re-aligned to 16. -/
def cursorTake (used count size : Nat) : Nat := align16 (used + count * size)

/-- The taken region fits below the new cursor. -/
theorem cursorTake_covers (used count size : Nat) : used + count * size ≤ cursorTake used count size := by
  unfold cursorTake; exact align16_ge _

/-- The new cursor is 16-byte aligned. -/
theorem cursorTake_mod (used count size : Nat) : cursorTake used count size % 16 = 0 := by
  unfold cursorTake; exact align16_mod _

/-- The cursor never moves backwards. -/
theorem cursorTake_mono (used count size : Nat) : used ≤ cursorTake used count size := by
  have h := cursorTake_covers used count size
  omega

/-- Sum of 16-byte aligned parts (the `*_buffers_bytes` totals). -/
def alignedTotal : List Nat → Nat
  | [] => 0
  | x :: xs => align16 x + alignedTotal xs

/-- Every aligned total is a multiple of 16. -/
theorem alignedTotal_mod : ∀ l : List Nat, alignedTotal l % 16 = 0
  | [] => by simp [alignedTotal]
  | x :: xs => by
    have h1 := align16_mod x
    have h2 := alignedTotal_mod xs
    simp only [alignedTotal]
    omega

/-- The aligned total is at least the plain sum of the parts. -/
theorem alignedTotal_ge : ∀ l : List Nat, l.sum ≤ alignedTotal l
  | [] => by simp [alignedTotal]
  | x :: xs => by
    have h1 := align16_ge x
    have h2 := alignedTotal_ge xs
    simp only [List.sum_cons, alignedTotal]
    omega

/-- Region `i` of `*_buffers_init` starts at an aligned offset. -/
theorem region_base_aligned (parts : List Nat) (i : Nat) : alignedTotal (parts.take i) % 16 = 0 :=
  alignedTotal_mod _

-- § activation dispatch

/-- Which GEMV path `gemv_quantized` takes. -/
inductive Gemv
  | native
  | s2q80
  | legacyQ80
  | q8k

/-- The dispatch rule of `gemv_quantized`. -/
def dispatch (native : Bool) (codeBits : Nat) (wantsQ8k : Bool) : Gemv :=
  if native then Gemv.native
  else if codeBits = 2 then Gemv.s2q80
  else if wantsQ8k then Gemv.q8k
  else Gemv.legacyQ80

/-- A native tensor always takes the native path. -/
theorem dispatch_native (c : Nat) (w : Bool) : dispatch true c w = Gemv.native := by
  simp [dispatch]

/-- A non-native 2-bit tensor always takes the S2/Q8_0 path. -/
theorem dispatch_s2 (c : Nat) (w : Bool) (hc : c = 2) : dispatch false c w = Gemv.s2q80 := by
  simp [dispatch, hc]

/-- A legacy 4/8-bit tensor that does not want Q8_K takes the legacy Q8_0 path. -/
theorem dispatch_legacy (c : Nat) (hc : c ≠ 2) : dispatch false c false = Gemv.legacyQ80 := by
  simp [dispatch, hc]

/-- The Q8_K path is never reached by a 2-bit tensor (which has no Q8_K contract). -/
theorem dispatch_q8k_not_s2 (n : Bool) (c : Nat) (w : Bool) (h : dispatch n c w = Gemv.q8k) :
    c ≠ 2 := by
  intro hc
  subst hc
  cases n <;> cases w <;> simp [dispatch] at h

-- § stage prefixes of block_layer_pre

/-- Whether stage `i` (0..4) runs, given `stage_prefix` and `half`. -/
def runStage (stagePrefix half i : Nat) : Bool :=
  if stagePrefix > 0 then decide (stagePrefix ≥ i + 1)
  else if i < 3 then decide (half = 0 ∨ half = 1)
  else decide (half = 0 ∨ half = 2)

/-- Default call (`half = 0`, no prefix) runs all five stages. -/
theorem runStage_default : ∀ i, i < 5 → runStage 0 0 i = true := by
  decide

/-- `half = 1` is the mixer: stages 0..2. -/
theorem runStage_mixer : ∀ i, i < 5 → runStage 0 1 i = decide (i < 3) := by
  decide

/-- `half = 2` is the FFN front and router: stages 3..4. -/
theorem runStage_ffn : ∀ i, i < 5 → runStage 0 2 i = decide (3 ≤ i) := by
  decide

/-- A positive prefix `k` runs exactly stages `0..k-1`, ignoring `half`. -/
theorem runStage_prefix (k half i : Nat) (hk : 0 < k) : runStage k half i = decide (i < k) := by
  unfold runStage
  rw [if_pos hk]
  exact decide_eq_decide.mpr (by omega)

/-- Prefix 5 runs every stage (it is the full graph). -/
theorem runStage_prefix5 (half i : Nat) (hi : i < 5) : runStage 5 half i = true := by
  rw [runStage_prefix 5 half i (by decide)]
  exact decide_eq_true hi

-- § verify-window compatibility

/-- `layer_verify_compatible` as a boolean conjunction of the fast-path flags. -/
def verifyCompatible (nb gr gdnFused nativeGdn fastAttn flashShort fastSel qsaIdx : Bool) : Bool :=
  nb && gr && (gdnFused && nativeGdn) && (fastAttn && !flashShort) && fastSel && qsaIdx

/-- Compatible exactly when every required flag is on and the short flash adapter is off. -/
theorem verifyCompatible_iff (nb gr gdnFused nativeGdn fastAttn flashShort fastSel qsaIdx : Bool) :
    verifyCompatible nb gr gdnFused nativeGdn fastAttn flashShort fastSel qsaIdx = true ↔
      (nb = true ∧ gr = true ∧ gdnFused = true ∧ nativeGdn = true ∧ fastAttn = true ∧
        flashShort = false ∧ fastSel = true ∧ qsaIdx = true) := by
  cases nb <;> cases gr <;> cases gdnFused <;> cases nativeGdn <;> cases fastAttn <;>
    cases flashShort <;> cases fastSel <;> cases qsaIdx <;> simp [verifyCompatible]

-- § fused hyper-connection bookkeeping

/-- `fused` of `block_layer_pre`: only for the default call. -/
def fusedGr (flag : Bool) (stagePrefix half : Nat) (supported : Bool) : Bool :=
  flag && decide (stagePrefix = 0) && decide (half = 0) && supported

/-- Fusion implies the default call shape. -/
theorem fusedGr_default (flag : Bool) (sp half : Nat) (s : Bool)
    (h : fusedGr flag sp half s = true) : sp = 0 ∧ half = 0 := by
  unfold fusedGr at h
  simp at h
  exact ⟨h.1.1.2, h.1.2⟩

/-- The post half writes the residual itself unless fused, and always on the last layer. -/
def postWritesR (fusedOn : Bool) (layer nLayers : Nat) : Bool :=
  !fusedOn || decide (layer + 1 = nLayers)

/-- Without fusion `gr_write` always runs. -/
theorem postWritesR_unfused (layer n : Nat) : postWritesR false layer n = true := by
  simp [postWritesR]

/-- With fusion the last layer materialises the residual for the head. -/
theorem postWritesR_last (n : Nat) : postWritesR true n (n + 1) = true := by
  simp [postWritesR]

/-- With fusion an inner layer defers its write. -/
theorem postWritesR_inner (layer n : Nat) (h : layer + 1 ≠ n) : postWritesR true layer n = false := by
  simp [postWritesR, h]

-- § PLE gating

/-- The PLE block applies only at layer 1 and only in the mixer-side halves. -/
def pleApplies (ready : Bool) (layer half : Nat) : Bool :=
  ready && decide (layer = 1) && decide (half = 0 ∨ half = 1)

/-- No PLE outside layer 1. -/
theorem pleApplies_layer (r : Bool) (l half : Nat) (h : l ≠ 1) : pleApplies r l half = false := by
  simp [pleApplies, h]

/-- No PLE in the FFN-front half. -/
theorem pleApplies_ffn (r : Bool) : pleApplies r 1 2 = false := by
  simp [pleApplies]

-- § layer kinds

/-- QSA layers are those with index ≡ 3 mod 4. -/
def isQsa (l : Nat) : Bool := decide (l % 4 = 3)

/-- 48 layers split 12 QSA / 36 GDN. -/
theorem qsa_gdn_split :
    ((List.range 48).filter isQsa).length = 12 ∧
      ((List.range 48).filter (fun l => !isQsa l)).length = 36 := by
  decide

-- § RoPE table bounds

/-- A flat row-major index stays inside a table of `m` rows of width `h`. -/
theorem flat_index_lt (r m h j : Nat) (hr : r < m) (hj : j < h) : r * h + j < m * h := by
  have h1 : (r + 1) * h ≤ m * h := Nat.mul_le_mul_right h (by omega)
  simp only [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § debug-dump layout

/-- Floats per layer in the C1 dump (n, hc, a = n_head*head_dim, nvk = n_head_kv*head_dim). -/
def dumpStride (n hc a nvk : Nat) : Nat := 2 * n + 2 * hc + a + 5 * nvk + 8

/-- The last of the four KV dumps ends exactly at the stride. -/
theorem dump_last_slot_ends_at_stride (n hc a nvk : Nat) :
    (2 * n + 2 * hc + a + 2 * nvk + 8) + 2 * nvk + nvk = dumpStride n hc a nvk := by
  unfold dumpStride; omega

/-- Arithmetic fact about the debug layout: the `ids/step` dump window of `qsa_layer` starts 8 floats
below the `k_pool` dump window and so intersects it whenever `nvk ≥ 2`. -/
theorem dump_debug_windows_intersect (n hc a nvk : Nat) (hn : 2 ≤ nvk) :
    ∃ p, (2 * n + 2 * hc + a + 2 * nvk + 8 ≤ p ∧
            p < 2 * n + 2 * hc + a + 2 * nvk + 8 + nvk / 2) ∧
         ((2 * n + 2 * hc + a + 2 * nvk) + nvk / 2 + 4 ≤ p ∧
            p < (2 * n + 2 * hc + a + 2 * nvk) + nvk / 2 + 8) := by
  refine ⟨2 * n + 2 * hc + a + 2 * nvk + 8 + nvk / 2 - 1, ?_, ?_⟩ <;> constructor <;> omega

end Strata.LayerComposition

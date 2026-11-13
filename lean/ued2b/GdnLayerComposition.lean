/-!
# MODULE: gdn_layer_composition
SOURCES: src/core/layer.cpp
Composes one GDN / QSA / MoE layer of the Ref engine from its kernels: carves 16-byte-aligned
scratch arenas, locates the three canonical planes of a quantized tensor, dispatches each quantized
GEMV to the activation format its TENSOR (not its role) wants, and selects which block stages run.

## Source semantics
* Every scratch region is rounded up with `(n + 15) & ~15`; regions are carved back to back, so the
  arena size is the sum of the rounded parts and no two regions overlap.
* A quantized tensor is `codes | scales | offsets`; `plane_ptrs` refuses unless the three plane sizes
  are nonzero (codes, scales), add up to the tensor's byte count, and `has_offset` agrees with the
  offset plane being nonempty.  Recomputing fp16 scales as 4 B/elem breaks that sum (the width bug).
* Activation contract: native weights take the FP32 path; `code_bits == 2` takes Q8_0; otherwise the
  manifest's `wants_q8k` picks Q8_K or Q8_0.  An S2 form must have no offset, group 64, bias -1.
* `block_layer_pre` stage selection: `stage_prefix = k > 0` runs stages `0..k-1`; otherwise `half`
  0 runs all five, 1 runs the mixer (0..2), 2 runs the FFN front and router (3..4).
* The C1 dump stride is `2N + 2hc + NH*HD + 5*NKV*HD + 8` floats per layer; every slot fits, and the
  `k_pool` slot is entirely overwritten by the later `v_scratch`/`ids`/`step` slots.

## Form template identifiers
align16 ↦ align_up16 / `(n+15)&~15`; carveOffsets ↦ gdn_buffers_init / Cursor::take;
carveTotal ↦ gdn_buffers_bytes / moe_buffers_bytes; WeightRef ↦ ref::core::WeightRef;
PlanesOk ↦ plane_ptrs; chooseAct ↦ gemv_quantized; s2Accept ↦ S2 guard in gemv_quantized;
runStage ↦ run0..run4 in block_layer_pre; dumpStride ↦ dump_stride_floats; ropeRowOk ↦ qsa_layer P7 guard.
-/

namespace Ref.Layer

-- § alignment

/-- The 16-byte round-up `(n + 15) & ~15` used for every scratch region. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Rounding up never shrinks a region. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by unfold align16; omega

/-- The rounded size is a multiple of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by unfold align16; omega

/-- Rounding adds strictly less than one alignment unit. -/
theorem align16_lt (n : Nat) : align16 n < n + 16 := by unfold align16; omega

/-- An already aligned size is unchanged. -/
theorem align16_of_aligned (n : Nat) (h : n % 16 = 0) : align16 n = n := by unfold align16; omega

/-- The Q8_K buffer size: `n / 256` blocks of 292 bytes. -/
def q8kBytes (n : Nat) : Nat := n / 256 * 292

/-- A Q8_K buffer of a multiple of 256 elements holds exactly `n/256` blocks. -/
theorem q8kBytes_blocks (m : Nat) : q8kBytes (256 * m) = 292 * m := by
  unfold q8kBytes; omega

-- § arena carving

/-- Total bytes of an arena whose parts are each rounded to 16 (`gdn_buffers_bytes`). -/
def carveTotal : List Nat → Nat
  | [] => 0
  | p :: ps => align16 p + carveTotal ps

/-- The region start offsets produced by carving parts back to back (`gdn_buffers_init`). -/
def carveOffsets (base : Nat) : List Nat → List Nat
  | [] => []
  | p :: ps => base :: carveOffsets (base + align16 p) ps

/-- One offset per part. -/
theorem carveOffsets_length (base : Nat) (ps : List Nat) :
    (carveOffsets base ps).length = ps.length := by
  induction ps generalizing base with
  | nil => rfl
  | cons p ps ih => simp [carveOffsets, ih]

/-- From an aligned base, every region starts 16-byte aligned. -/
theorem carveOffsets_aligned (base : Nat) (ps : List Nat) (hb : base % 16 = 0) :
    ∀ o ∈ carveOffsets base ps, o % 16 = 0 := by
  induction ps generalizing base with
  | nil => intro o h; simp [carveOffsets] at h
  | cons p ps ih =>
    intro o h
    simp only [carveOffsets, List.mem_cons] at h
    rcases h with rfl | h
    · exact hb
    · have ha := align16_mod p
      exact ih (base + align16 p) (by omega) o h

/-- Every region starts at or after the base. -/
theorem carveOffsets_ge (base : Nat) (ps : List Nat) : ∀ o ∈ carveOffsets base ps, base ≤ o := by
  induction ps generalizing base with
  | nil => intro o h; simp [carveOffsets] at h
  | cons p ps ih =>
    intro o h
    simp only [carveOffsets, List.mem_cons] at h
    rcases h with rfl | h
    · exact Nat.le_refl _
    · have := ih (base + align16 p) o h
      omega

/-- Every region `[o, o + p)` ends inside the arena `[base, base + carveTotal ps)`. -/
theorem carve_fits (base : Nat) (ps : List Nat) :
    ∀ x ∈ (carveOffsets base ps).zip ps, x.1 + x.2 ≤ base + carveTotal ps := by
  induction ps generalizing base with
  | nil => intro x h; simp [carveOffsets] at h
  | cons p ps ih =>
    intro x h
    simp only [carveOffsets, List.zip_cons_cons, List.mem_cons] at h
    have hp := align16_ge p
    rcases h with rfl | h
    · simp only [carveTotal]; omega
    · have := ih (base + align16 p) x h
      simp only [carveTotal]; omega

/-- Regions are pairwise disjoint: each one ends before any later one starts. -/
theorem carve_disjoint (base : Nat) (ps : List Nat) :
    List.Pairwise (fun a b : Nat × Nat => a.1 + a.2 ≤ b.1) ((carveOffsets base ps).zip ps) := by
  induction ps generalizing base with
  | nil => simp [carveOffsets]
  | cons p ps ih =>
    simp only [carveOffsets, List.zip_cons_cons, List.pairwise_cons]
    refine ⟨?_, ih (base + align16 p)⟩
    intro b hb
    have hmem : b.1 ∈ carveOffsets (base + align16 p) ps := (List.of_mem_zip hb).1
    have := carveOffsets_ge (base + align16 p) ps b.1 hmem
    have hp := align16_ge p
    omega

-- § the GDN scratch layout

/-- The model geometry fields a layer's buffers depend on. -/
structure Geometry where
  nEmbd : Nat
  convChannels : Nat
  valueDim : Nat
  vHeads : Nat
  stateSize : Nat
  dConv : Nat

/-- The sixteen parts of `gdn_buffers_bytes`, in carve order. -/
def gdnParts (g : Geometry) : List Nat :=
  [q8kBytes g.nEmbd, g.nEmbd / 32 * 34, g.nEmbd * 2, g.convChannels * 4, g.convChannels * 4,
   g.convChannels * 4, g.vHeads * 4, g.vHeads * 4, g.vHeads * 4, g.vHeads * g.stateSize * 4,
   g.valueDim * 4, g.valueDim * 4, q8kBytes g.valueDim, g.valueDim / 32 * 34,
   g.stateSize * g.vHeads * g.stateSize * 4, g.convChannels * (g.dConv - 1) * 4]

/-- `gdn_buffers_init` hands out exactly sixteen pointers, one per part. -/
theorem gdnParts_length (g : Geometry) : (carveOffsets 0 (gdnParts g)).length = 16 := by
  rw [carveOffsets_length]; rfl

/-- All sixteen GDN regions are aligned, disjoint and inside the `gdn_buffers_bytes` arena. -/
theorem gdn_layout_sound (g : Geometry) :
    (∀ o ∈ carveOffsets 0 (gdnParts g), o % 16 = 0) ∧
    (∀ x ∈ (carveOffsets 0 (gdnParts g)).zip (gdnParts g), x.1 + x.2 ≤ carveTotal (gdnParts g)) ∧
    List.Pairwise (fun a b : Nat × Nat => a.1 + a.2 ≤ b.1)
      ((carveOffsets 0 (gdnParts g)).zip (gdnParts g)) := by
  refine ⟨carveOffsets_aligned 0 _ rfl, ?_, carve_disjoint 0 _⟩
  intro x hx
  have := carve_fits 0 (gdnParts g) x hx
  omega

-- § canonical planes of a quantized tensor

/-- The loader's record of a tensor (the fields `plane_ptrs` and `gemv_quantized` read). -/
structure WeightRef where
  codeBits : Nat
  codesBytes : Nat
  scalesBytes : Nat
  offsetBytes : Nat
  bytes : Nat
  hasOffset : Bool
  groupElems : Nat
  codeBias : Int
  wantsQ8k : Bool
  native : Bool

/-- The acceptance condition of `plane_ptrs`. -/
def PlanesOk (w : WeightRef) : Prop :=
  (w.codeBits = 2 ∨ w.codeBits = 4 ∨ w.codeBits = 8) ∧ w.codesBytes ≠ 0 ∧ w.scalesBytes ≠ 0 ∧
  w.codesBytes + w.scalesBytes + w.offsetBytes = w.bytes ∧ (w.hasOffset = true ↔ w.offsetBytes ≠ 0)

/-- Byte offset of the scale plane inside the tensor. -/
def scalesAt (w : WeightRef) : Nat := w.codesBytes

/-- Byte offset of the offset plane inside the tensor. -/
def offsetAt (w : WeightRef) : Nat := w.codesBytes + w.scalesBytes

/-- Accepted planes partition the tensor: codes, then scales, then offsets ending exactly at `bytes`. -/
theorem planes_partition (w : WeightRef) (h : PlanesOk w) :
    w.codesBytes ≤ scalesAt w ∧ scalesAt w + w.scalesBytes = offsetAt w ∧
    offsetAt w + w.offsetBytes = w.bytes := by
  obtain ⟨_, _, _, hsum, _⟩ := h
  unfold scalesAt offsetAt
  omega

/-- A tensor without an offset plane carries `has_offset = false`, and conversely. -/
theorem planes_no_offset (w : WeightRef) (h : PlanesOk w) :
    w.hasOffset = false ↔ w.offsetBytes = 0 := by
  obtain ⟨_, _, _, _, hiff⟩ := h
  cases hb : w.hasOffset <;> rw [hb] at hiff <;> simp at hiff ⊢ <;> omega

/-- **The width bug.**  When the real scale plane is `s` bytes of fp16 and the layout is re-derived
at 4 B/elem (`2 * s`), the three planes no longer add up to the tensor, so `plane_ptrs` refuses. -/
theorem fp16_rederived_rejected (codes s off bytes : Nat) (hs : s ≠ 0)
    (hreal : codes + s + off = bytes) : codes + 2 * s + off ≠ bytes := by
  omega

-- § activation dispatch

/-- The activation image a quantized projection reads. -/
inductive Act where
  | native
  | q8_0
  | q8k
  deriving DecidableEq, Repr

/-- `gemv_quantized`'s choice, a property of the tensor. -/
def chooseAct (w : WeightRef) : Act :=
  if w.native then .native else if w.codeBits = 2 then .q8_0 else if w.wantsQ8k then .q8k else .q8_0

/-- An S2 tensor never gets the Q8_K image (`code_bits 2 has no Q8_K contract`). -/
theorem s2_never_q8k (w : WeightRef) (h : w.codeBits = 2) : chooseAct w ≠ .q8k := by
  unfold chooseAct
  cases w.native <;> simp [h]

/-- A legacy (non-K) form gets Q8_0 even when its code width is 4 or 8. -/
theorem legacy_gets_q8_0 (w : WeightRef) (hn : w.native = false) (hk : w.wantsQ8k = false) :
    chooseAct w = .q8_0 := by
  unfold chooseAct
  by_cases h2 : w.codeBits = 2 <;> simp [hn, hk, h2]

/-- A K-quant of width ≠ 2 gets Q8_K. -/
theorem kquant_gets_q8k (w : WeightRef) (hn : w.native = false) (h2 : w.codeBits ≠ 2)
    (hk : w.wantsQ8k = true) : chooseAct w = .q8k := by
  unfold chooseAct
  simp [hn, h2, hk]

/-- Only native tensors take the native path. -/
theorem native_iff (w : WeightRef) : chooseAct w = .native ↔ w.native = true := by
  unfold chooseAct
  cases w.native <;> simp <;> split <;> (try split) <;> simp

/-- The fixed S2 attributes `s2_gemv_q8` assumes. -/
def s2Accept (w : WeightRef) : Prop := w.hasOffset = false ∧ w.groupElems = 64 ∧ w.codeBias = -1

/-- Two tensors of the same width and attributes but different `wants_q8k` (Q5_0 vs Q5_K, IQ4_NL vs
IQ4_XS) are dispatched differently: `code_bits` alone cannot decide the contract. -/
theorem codeBits_insufficient (w : WeightRef) (hn : w.native = false) (h2 : w.codeBits = 8) :
    chooseAct { w with wantsQ8k := true } ≠ chooseAct { w with wantsQ8k := false } := by
  unfold chooseAct
  simp [hn, h2]

-- § stage selection in block_layer_pre

/-- Whether stage `i` (0..4) runs for a given `stage_prefix` and `half`. -/
def runStage (pre half i : Nat) : Bool :=
  if pre > 0 then decide (i + 1 ≤ pre)
  else if i ≤ 2 then (half == 0 || half == 1) else (half == 0 || half == 2)

/-- With a positive prefix `k`, exactly stages `0..k-1` run, whatever `half` says. -/
theorem runStage_prefix (pre half i : Nat) (hp : 0 < pre) : runStage pre half i = true ↔ i < pre := by
  unfold runStage
  simp [hp]
  omega

/-- `half = 0` with no prefix runs every stage (the default for every older caller). -/
theorem runStage_all (i : Nat) : runStage 0 0 i = true := by
  unfold runStage
  by_cases h : i ≤ 2 <;> simp [h]

/-- `half = 1` runs exactly the mixer stages 0..2. -/
theorem runStage_mixer (i : Nat) : runStage 0 1 i = true ↔ i ≤ 2 := by
  unfold runStage
  by_cases h : i ≤ 2 <;> simp [h]

/-- `half = 2` runs exactly the FFN front and the router (stages ≥ 3). -/
theorem runStage_ffn (i : Nat) : runStage 0 2 i = true ↔ 3 ≤ i := by
  unfold runStage
  by_cases h : i ≤ 2 <;> simp [h] <;> omega

/-- The split capture's two halves together run each of the five stages exactly once. -/
theorem halves_partition (i : Nat) (hi : i < 5) :
    (runStage 0 1 i = true ∨ runStage 0 2 i = true) ∧
    ¬(runStage 0 1 i = true ∧ runStage 0 2 i = true) := by
  rw [runStage_mixer, runStage_ffn]
  omega

-- § QSA position guard

/-- The P7 audit in `qsa_layer`: RoPE reads row `pos_base + pos`, the table holds `max_cells` rows. -/
def ropeRowOk (posBase pos maxCells : Int) : Prop := 0 ≤ posBase ∧ posBase + pos < maxCells

/-- An accepted position reads a row inside the table (given the separate `0 ≤ pos` check). -/
theorem ropeRow_in_table (posBase pos maxCells : Int) (h : ropeRowOk posBase pos maxCells)
    (hp : 0 ≤ pos) : 0 ≤ posBase + pos ∧ posBase + pos < maxCells := by
  obtain ⟨h1, h2⟩ := h
  omega

-- § the C1 dump layout

/-- Floats per layer in the dump: `n` = n_embd, `hc`, `a` = n_head*head_dim, `nvk` = n_head_kv*head_dim. -/
def dumpStride (n hc a nvk : Nat) : Nat := 2 * n + 2 * hc + a + 5 * nvk + 8

/-- Start of the gathered `v_scratch` slot. -/
def vsOff (n hc a nvk : Nat) : Nat := 2 * n + 2 * hc + a + 2 * nvk

/-- Start of the `k_pool` slot (`base` in `qsa_layer`). -/
def kpoolOff (n hc a nvk : Nat) : Nat := 2 * n + 2 * hc + a + 2 * nvk + 8

/-- Every slot `qsa_layer` and `block_layer_*` dump into ends inside one layer's stride. -/
theorem dump_slots_fit (n hc a nvk : Nat) :
    let s := dumpStride n hc a nvk
    let b := kpoolOff n hc a nvk
    let vs := vsOff n hc a nvk
    2 * n + 2 * hc ≤ s ∧ 2 * n + 2 * hc + a ≤ s ∧ 2 * n + 2 * hc + a + nvk ≤ s ∧
    vs + nvk / 2 + 8 ≤ s ∧ b + nvk / 2 + nvk / 2 ≤ s ∧ b + nvk + nvk ≤ s ∧ b + 2 * nvk + nvk ≤ s := by
  simp only [dumpStride, kpoolOff, vsOff]
  omega

/-- **The `k_pool` dump is dead.**  It is enqueued before the later `v_scratch`, `ids` and `step`
copies, and its whole range lies inside theirs, so the host buffer never holds it. -/
theorem kpool_dump_shadowed (n hc a nvk : Nat) :
    vsOff n hc a nvk ≤ kpoolOff n hc a nvk ∧
    kpoolOff n hc a nvk + nvk / 2 = vsOff n hc a nvk + nvk / 2 + 4 + 4 := by
  simp only [kpoolOff, vsOff]
  omega

-- § the doorbell

/-- Mapped-pinned bytes `doorbell_init` allocates: x_f, ids, weights, seq, flag. -/
def doorbellBytes (nEmbd k : Nat) : Nat := nEmbd * 4 + k * 4 + k * 4 + 4 + 4

/-- The doorbell always holds the input row and two k-wide lists. -/
theorem doorbellBytes_ge (nEmbd k : Nat) : nEmbd * 4 + 8 * k ≤ doorbellBytes nEmbd k := by
  unfold doorbellBytes; omega

/-- `moe_route` / `moe_combine_parts` accept `k` only in `1..64`. -/
def kOk (k : Int) : Prop := 1 ≤ k ∧ k ≤ 64

/-- The native router's precondition is a special case of the range check. -/
theorem native_router_k (k : Int) (h : k = 10) : kOk k := by
  unfold kOk; omega

end Ref.Layer



/-!
# MODULE: layer_composition
SOURCES: src/core/layer.cpp
The per-token GDN / QSA / MoE layer: arena carving, canonical plane layout, activation-format dispatch,
RoPE-row bounds, the verify-compatibility predicate and the oracle dump stride.

## Source semantics
* Every buffer region is 16-byte aligned and carved sequentially from one arena (`Cursor`, `*_buffers_init`).
* A quantized tensor is codes ++ scales ++ optional offsets; the three sizes must add up to the tensor size.
* Activation format is a property of the TENSOR: code_bits = 2 → Q8_0; otherwise Q8_K only if `wants_q8k`.
* `qsa_layer` refuses positions outside the RoPE table; pooled-block capacity is `max_cells / idx_block + 2`.

## Form template identifiers
align16 ↦ `align_up16`; carve ↦ `Cursor::take`; WeightRef ↦ `WeightRef`; actFor ↦ `gemv_quantized` dispatch;
Flags ↦ the `g_*` toggles consulted by `layer_verify_compatible`.
-/
namespace Ref.LayerComposition

-- § alignment and cursor carving

/-- Round up to a multiple of 16 (`(n + 15) & ~15`). -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Alignment never shrinks a size. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by unfold align16; omega

/-- Alignment result is a multiple of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by unfold align16; omega

/-- Alignment wastes fewer than 16 bytes. -/
theorem align16_lt (n : Nat) : align16 n < n + 16 := by unfold align16; omega

/-- Alignment is idempotent. -/
theorem align16_idem (n : Nat) : align16 (align16 n) = align16 n := by unfold align16; omega

/-- Total bytes consumed by carving `parts` sequentially from offset `s`. -/
def carve : List Nat → Nat → Nat
  | [], s => s
  | p :: ps, s => carve ps (s + align16 p)

/-- Carving equals start plus the sum of aligned parts (the `*_bytes` and `*_init` totals agree). -/
theorem carve_eq : ∀ (parts : List Nat) (s : Nat), carve parts s = s + (parts.map align16).sum
  | [], s => by simp [carve]
  | p :: ps, s => by
      rw [carve, carve_eq ps]
      simp only [List.map_cons, List.sum_cons]
      omega

/-- Starting aligned, every carve boundary stays aligned. -/
theorem carve_aligned : ∀ (parts : List Nat) (s : Nat), s % 16 = 0 → carve parts s % 16 = 0
  | [], s, h => by simpa [carve] using h
  | p :: ps, s, h => by
      rw [carve]
      apply carve_aligned ps
      have := align16_mod p
      omega

/-- Carving is monotone in the cursor. -/
theorem carve_ge : ∀ (parts : List Nat) (s : Nat), s ≤ carve parts s
  | [], s => Nat.le_refl s
  | p :: ps, s => by
      rw [carve]
      have h := carve_ge ps (s + align16 p)
      omega

/-- Carving splits over concatenation. -/
theorem carve_append : ∀ (xs ys : List Nat) (s : Nat), carve (xs ++ ys) s = carve ys (carve xs s)
  | [], ys, s => by simp [carve]
  | x :: xs, ys, s => by
      simp only [List.cons_append, carve]
      exact carve_append xs ys _

/-- Each region ends inside the arena total: no region overlaps its successors. -/
theorem row_fits (pre post : List Nat) (x : Nat) :
    carve pre 0 + align16 x ≤ carve (pre ++ x :: post) 0 := by
  rw [carve_append]
  have h := carve_ge post (carve pre 0 + align16 x)
  simp only [carve]
  exact h

-- § Q8_K block arithmetic

/-- `q8k_bytes n = n / 256 * 292`. -/
def q8kBytes (n : Nat) : Nat := n / 256 * 292

/-- A whole number of 256-element blocks costs 292 bytes each. -/
theorem q8kBytes_mul (b : Nat) : q8kBytes (256 * b) = b * 292 := by
  unfold q8kBytes
  have h : 256 * b / 256 = b := by omega
  rw [h]

/-- The width of one embedding (2560) is exactly ten blocks. -/
theorem q8kBytes_embd : q8kBytes 2560 = 2920 := by decide

-- § canonical planes

/-- The loader-recorded attributes of a quantized tensor that `plane_ptrs` inspects. -/
structure WeightRef where
  codeBits : Nat
  codesBytes : Nat
  scalesBytes : Nat
  offsetBytes : Nat
  bytes : Nat
  hasOffset : Bool

/-- The acceptance test of `plane_ptrs`. -/
def planesValid (r : WeightRef) : Prop :=
  (r.codeBits = 2 ∨ r.codeBits = 4 ∨ r.codeBits = 8) ∧ r.codesBytes ≠ 0 ∧ r.scalesBytes ≠ 0 ∧
  r.codesBytes + r.scalesBytes + r.offsetBytes = r.bytes ∧ (r.hasOffset = true ↔ r.offsetBytes ≠ 0)

/-- Start of the scale plane. -/
def scalesOff (r : WeightRef) : Nat := r.codesBytes

/-- Start of the offset plane. -/
def offsetOff (r : WeightRef) : Nat := r.codesBytes + r.scalesBytes

/-- Valid planes tile the tensor exactly: the offset plane ends at the tensor end. -/
theorem planes_tile (r : WeightRef) (h : planesValid r) :
    scalesOff r = r.codesBytes ∧ offsetOff r + r.offsetBytes = r.bytes := by
  obtain ⟨_, _, _, hsum, _⟩ := h
  exact ⟨rfl, hsum⟩

/-- No offset flag means an empty offset plane (so the pointer may be null). -/
theorem no_offset_empty (r : WeightRef) (h : planesValid r) (hh : r.hasOffset = false) :
    r.offsetBytes = 0 := by
  obtain ⟨_, _, _, _, hoff⟩ := h
  by_contra hne
  have := hoff.2 hne
  simp [hh] at this

/-- An fp16 plane widened to f32 doubles; with an even f32 size the source size is exact. -/
theorem fp16_src_double (s : Nat) (h : s % 4 = 0) : s / 2 * 2 = s := by omega

/-- The old re-derivation assumed 4 B/scale; for fp16 packs it was twice the real plane. -/
theorem old_derivation_doubles (elems actual derived : Nat)
    (ha : actual = 2 * elems) (hd : derived = 4 * elems) : derived = 2 * actual := by omega

-- § activation contract

/-- The two quantized activation images. -/
inductive Act | q8_0 | q8k
  deriving DecidableEq

/-- `gemv_quantized`: 2-bit forms use Q8_0; others Q8_K only when the manifest source type wants it. -/
def actFor (codeBits : Nat) (wantsQ8k : Bool) : Act :=
  if codeBits = 2 then Act.q8_0 else if wantsQ8k then Act.q8k else Act.q8_0

/-- Two-bit tensors never see Q8_K. -/
theorem actFor_two (w : Bool) : actFor 2 w = Act.q8_0 := by simp [actFor]

/-- A legacy form (not wanting Q8_K) always gets Q8_0. -/
theorem actFor_legacy (b : Nat) : actFor b false = Act.q8_0 := by
  unfold actFor; split <;> rfl

/-- Q8_K is only chosen for non-2-bit tensors that ask for it. -/
theorem actFor_q8k (b : Nat) (w : Bool) (h : actFor b w = Act.q8k) : b ≠ 2 ∧ w = true := by
  unfold actFor at h
  by_cases hb : b = 2
  · simp [hb] at h
  · refine ⟨hb, ?_⟩
    cases w
    · simp [hb] at h
    · rfl

-- § GDN q|k|v split of the conv channels

/-- v starts after q and k, and the three runs fill the channel dimension. -/
theorem qkv_split (qk v C : Nat) (hC : C = 2 * qk + v) : 2 * qk + v = C := hC.symm

/-- The real geometry: 128 * 16 = 2048, 48 * 128 = 6144, total 10240. -/
theorem qkv_real : 2 * (128 * 16) + 48 * 128 = 10240 := by decide

-- § QSA positions

/-- `qsa_layer` accepts only a non-negative base with `pos_base + pos < max_cells`. -/
def ropeRowOk (posBase pos maxCells : Int) : Prop := 0 ≤ posBase ∧ posBase + pos < maxCells

/-- An accepted position reads a RoPE row that exists. -/
theorem ropeRowOk_row (posBase pos maxCells : Int) (h : ropeRowOk posBase pos maxCells) (hp : 0 ≤ pos) :
    0 ≤ posBase + pos ∧ posBase + pos < maxCells := by
  obtain ⟨h1, h2⟩ := h
  exact ⟨by omega, h2⟩

/-- Completed-block count never exceeds the pooled capacity `max_cells / idx_block + 2`. -/
theorem blocks_within_capacity (pos maxCells blk : Nat) (h : pos < maxCells) :
    (pos + 1) / blk ≤ maxCells / blk + 2 := by
  have h1 : (pos + 1) / blk ≤ maxCells / blk := Nat.div_le_div_right (by omega)
  omega

/-- `pages * page_size` covers `max_cells`. -/
theorem pages_cover (m ps : Nat) (hps : 0 < ps) : m ≤ (m + ps - 1) / ps * ps := by
  have h := Nat.div_add_mod (m + ps - 1) ps
  have h2 := Nat.mod_lt (m + ps - 1) hps
  rw [Nat.mul_comm]
  omega

-- § oracle dump layout

/-- Floats per layer in the C1 dump (`nvk = n_head_kv * head_dim`, `ah = n_head * head_dim`). -/
def dumpStride (n hc ah nvk : Nat) : Nat := 2 * n + 2 * hc + ah + 5 * nvk + 8

/-- The last slot (`kcur`) ends exactly at the stride: nothing is written out of the layer's row. -/
theorem dump_last_slot_end (n hc ah nvk : Nat) :
    (2 * n + 2 * hc + ah + 2 * nvk + 8) + 2 * nvk + nvk = dumpStride n hc ah nvk := by
  unfold dumpStride; omega

-- § verify compatibility

/-- The toggles `layer_verify_compatible` consults. -/
structure Flags where
  nativeBf16 : Bool
  fusedGr : Bool
  fusedGdn : Bool
  nativeGdn : Bool
  fastAttn : Bool
  nativeFlashShort : Bool
  fastSelect : Bool
  nativeIndexer : Bool

/-- The window reproduces the default native decode path only if every toggle is in its default state. -/
def verifyCompatible (f : Flags) : Bool :=
  f.nativeBf16 && f.fusedGr && (f.fusedGdn && f.nativeGdn) && (f.fastAttn && !f.nativeFlashShort) &&
    f.fastSelect && f.nativeIndexer

/-- Compatibility forces the fused hyper-connection read on. -/
theorem compat_fusedGr (f : Flags) (h : verifyCompatible f = true) : f.fusedGr = true := by
  cases f; simp_all [verifyCompatible]

/-- Compatibility forces the short flash adapter off. -/
theorem compat_no_flash_short (f : Flags) (h : verifyCompatible f = true) :
    f.nativeFlashShort = false := by
  cases f; simp_all [verifyCompatible]

/-- Compatibility forces the native indexer on. -/
theorem compat_indexer (f : Flags) (h : verifyCompatible f = true) : f.nativeIndexer = true := by
  cases f; simp_all [verifyCompatible]

end Ref.LayerComposition

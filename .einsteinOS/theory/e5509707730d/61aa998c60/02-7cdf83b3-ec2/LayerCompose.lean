/-!
# MODULE: layer_compose
SOURCES: src/core/layer.cpp
The composed GDN / QSA / MoE decoder layer: buffer carving (16-byte aligned cursor), weight-plane validation,
per-tensor activation dispatch, stage selection of `block_layer_pre`, the doorbell order and the debug-dump layout.

## Source semantics
* Every arena region is rounded up to 16 bytes; regions handed out by one cursor never overlap.
* A quantized tensor's three planes (codes, scales, offsets) must add up exactly to the tensor size, and
  `has_offset` must agree with a non-empty offset plane.
* The activation image a projection reads is chosen by the tensor, not by its role: native first, then
  `code_bits == 2` -> Q8_0, a legacy form -> Q8_0, otherwise Q8_K.
* `stage_prefix == 0` means "as `half` says"; `stage_prefix == k > 0` runs exactly stages `0..k-1`.
* The doorbell ring is the LAST operation of the publish sequence.

## Form template identifiers
align16 <- `(n + 15) & ~15`; Cursor.take <- `Cursor::take/take_bytes`; PlanesOk <- `plane_ptrs` guard;
dispatch <- `gemv_quantized`; RunStage <- `run0..run4` in `block_layer_pre`; dumpStride <- `dump_stride_floats`.
-/

namespace Strata.LayerCompose

-- § alignment and the cursor

/-- 16-byte round-up `(n + 15) & ~15`, written as a division-free formula. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Rounding up never loses bytes. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by unfold align16; omega

/-- The result is 16-aligned. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by unfold align16; omega

/-- Rounding up wastes fewer than 16 bytes. -/
theorem align16_lt (n : Nat) : align16 n < n + 16 := by unfold align16; omega

/-- Aligning twice is aligning once. -/
theorem align16_idem (n : Nat) : align16 (align16 n) = align16 n := by unfold align16; omega

/-- Alignment is monotone. -/
theorem align16_mono {a b : Nat} (h : a ≤ b) : align16 a ≤ align16 b := by unfold align16; omega

/-- The arena cursor: only the number of bytes used so far matters. -/
structure Cursor where
  used : Nat

/-- `take bytes` returns the new cursor and the offset of the region just handed out. -/
def Cursor.take (c : Cursor) (bytes : Nat) : Cursor × Nat :=
  (⟨align16 (c.used + bytes)⟩, c.used)

/-- The cursor stays 16-aligned after every take. -/
theorem take_aligned (c : Cursor) (bytes : Nat) : (c.take bytes).1.used % 16 = 0 :=
  align16_mod (c.used + bytes)

/-- The handed-out region `[offset, offset + bytes)` lies entirely below the new cursor (no overlap later). -/
theorem take_covers (c : Cursor) (bytes : Nat) : c.used + bytes ≤ (c.take bytes).1.used :=
  align16_ge (c.used + bytes)

/-- The padding of a take is under 16 bytes. -/
theorem take_tight (c : Cursor) (bytes : Nat) : (c.take bytes).1.used < c.used + bytes + 16 :=
  align16_lt (c.used + bytes)

/-- The returned offset is the old cursor. -/
theorem take_offset (c : Cursor) (bytes : Nat) : (c.take bytes).2 = c.used := rfl

/-- An aligned cursor hands out an aligned offset. -/
theorem take_offset_aligned (c : Cursor) (bytes : Nat) (h : c.used % 16 = 0) : (c.take bytes).2 % 16 = 0 := h

/-- Sum of a list of sizes. -/
def lsum : List Nat → Nat
  | [] => 0
  | a :: l => a + lsum l

/-- Total of the individually aligned parts (`gdn_buffers_bytes`, `moe_buffers_bytes`). -/
def totalAligned : List Nat → Nat
  | [] => 0
  | a :: l => align16 a + totalAligned l

/-- The aligned total covers the raw sum. -/
theorem sum_le_totalAligned (l : List Nat) : lsum l ≤ totalAligned l := by
  induction l with
  | nil => simp [lsum, totalAligned]
  | cons a l ih =>
    have h := align16_ge a
    simp only [lsum, totalAligned]
    omega

/-- Offsets of consecutive aligned parts starting at `start`. -/
def place (start : Nat) : List Nat → List Nat
  | [] => []
  | a :: l => start :: place (start + align16 a) l

/-- One offset per part. -/
theorem place_length (l : List Nat) : ∀ start : Nat, (place start l).length = l.length := by
  induction l with
  | nil => intro start; rfl
  | cons a l ih => intro start; simp [place, ih]

-- § Q8_K sizing

/-- Bytes of a Q8_K buffer for `n` elements (292 bytes per 256-element block). -/
def q8kBytes (n : Nat) : Nat := n / 256 * 292

/-- A whole number of blocks costs `292` bytes each. -/
theorem q8kBytes_blocks (k : Nat) : q8kBytes (k * 256) = k * 292 := by unfold q8kBytes; omega

/-- Q8_K sizing is monotone. -/
theorem q8kBytes_mono {a b : Nat} (h : a ≤ b) : q8kBytes a ≤ q8kBytes b := by unfold q8kBytes; omega

-- § weight planes

/-- The loader-recorded plane sizes of a quantized tensor. -/
structure WRef where
  bits : Nat
  codes : Nat
  scales : Nat
  offset : Nat
  bytes : Nat
  hasOffset : Bool

/-- The guard of `plane_ptrs`. -/
def PlanesOk (r : WRef) : Prop :=
  (r.bits = 2 ∨ r.bits = 4 ∨ r.bits = 8) ∧ r.codes ≠ 0 ∧ r.scales ≠ 0 ∧
  r.codes + r.scales + r.offset = r.bytes ∧ (r.hasOffset = true ↔ r.offset ≠ 0)

/-- Accepted planes tile the tensor: the offset plane starts right after codes and scales and ends at `bytes`. -/
theorem planes_tile (r : WRef) (h : PlanesOk r) :
    r.codes + r.scales ≤ r.bytes ∧ r.codes + r.scales + r.offset = r.bytes := by
  obtain ⟨_, _, _, hs, _⟩ := h
  constructor <;> omega

/-- A tensor with no offset plane has `has_offset = false`. -/
theorem planes_no_offset (r : WRef) (h : PlanesOk r) (h0 : r.offset = 0) : r.hasOffset = false := by
  obtain ⟨_, _, _, _, hi⟩ := h
  cases hh : r.hasOffset with
  | false => rfl
  | true => exact absurd h0 (hi.mp hh)

/-- If planes disagree with the tensor size the guard refuses. -/
theorem planes_refuse (r : WRef) (h : r.codes + r.scales + r.offset ≠ r.bytes) : ¬ PlanesOk r := by
  intro hp
  exact h hp.2.2.2.1

-- § activation dispatch

/-- Which activation image a projection consumes. -/
inductive ActKind
  | native
  | q80
  | q8k
  deriving DecidableEq

/-- `gemv_quantized`: native, then S2 (Q8_0), then legacy (Q8_0), then K-quant (Q8_K). -/
def dispatch (native : Bool) (bits : Nat) (wantsQ8K : Bool) : ActKind :=
  if native then ActKind.native
  else if bits = 2 then ActKind.q80
  else if wantsQ8K then ActKind.q8k
  else ActKind.q80

/-- A 2-bit form always reads the Q8_0 image. -/
theorem dispatch_s2 (bits : Nat) (wants : Bool) (h : bits = 2) : dispatch false bits wants = ActKind.q80 := by
  simp [dispatch, h]

/-- A tensor that does not want Q8_K never reads it. -/
theorem dispatch_legacy (bits : Nat) : dispatch false bits false = ActKind.q80 := by
  unfold dispatch
  by_cases h : bits = 2 <;> simp [h]

/-- A wider-than-2-bit K-quant reads Q8_K. -/
theorem dispatch_kquant (bits : Nat) (h : bits ≠ 2) : dispatch false bits true = ActKind.q8k := by
  simp [dispatch, h]

/-- The native path ignores the code width. -/
theorem dispatch_native (bits : Nat) (w : Bool) : dispatch true bits w = ActKind.native := by
  simp [dispatch]

-- § stage selection of block_layer_pre

/-- Stages 0..2 run in the mixer half (half 0/1), stages 3..4 in the FFN half (half 0/2). -/
def halfAllows (k h : Nat) : Prop :=
  if k ≤ 2 then (h = 0 ∨ h = 1) else (h = 0 ∨ h = 2)

/-- Stage `k` runs for `stage_prefix = p`, `half = h`. -/
def RunStage (k p h : Nat) : Prop := if 0 < p then k < p else halfAllows k h

/-- With a positive prefix exactly the first `p` stages run, whatever `half` is. -/
theorem runStage_prefix {k p h : Nat} (hp : 0 < p) : RunStage k p h ↔ k < p := by
  unfold RunStage
  rw [if_pos hp]

/-- The legacy call (`prefix = 0`, `half = 0`) runs all five stages. -/
theorem runStage_legacy {k : Nat} : RunStage k 0 0 := by
  unfold RunStage halfAllows
  rw [if_neg (Nat.lt_irrefl 0)]
  split <;> simp

/-- Prefixes are downward closed. -/
theorem runStage_mono {k p h : Nat} (hp : 0 < p) (hr : RunStage (k + 1) p h) : RunStage k p h := by
  rw [runStage_prefix hp] at hr ⊢
  omega

/-- `half = 2` skips the mixer. -/
theorem runStage_half2_skips_mixer : ¬ RunStage 0 0 2 := by
  unfold RunStage halfAllows
  rw [if_neg (Nat.lt_irrefl 0)]
  simp

/-- `half = 1` skips the FFN front. -/
theorem runStage_half1_skips_ffn : ¬ RunStage 3 0 1 := by
  unfold RunStage halfAllows
  rw [if_neg (Nat.lt_irrefl 0)]
  simp

-- § verify compatibility, doorbell order, positions, dump layout

/-- `layer_verify_compatible`: all six switches must be on. -/
def verifyCompatible (a b c d e f : Bool) : Bool := a && b && c && d && e && f

/-- Compatible iff every switch is on. -/
theorem verifyCompatible_iff (a b c d e f : Bool) :
    verifyCompatible a b c d e f = true ↔ a = true ∧ b = true ∧ c = true ∧ d = true ∧ e = true ∧ f = true := by
  simp [verifyCompatible, and_assoc]

/-- Operations of the non-kernel doorbell publish. -/
inductive Op
  | copyX
  | copyIds
  | copyWeights
  | ring
  deriving DecidableEq

/-- The publish order. -/
def publishOrder : List Op := [Op.copyX, Op.copyIds, Op.copyWeights, Op.ring]

/-- The ring is last, so a host that sees it sees every byte above. -/
theorem ring_is_last : publishOrder.getLast? = some Op.ring := rfl

/-- RoPE position guard of `qsa_layer`. -/
def qsaPosOk (posBase pos maxCells : Int) : Prop := 0 ≤ posBase ∧ 0 ≤ pos ∧ posBase + pos < maxCells

/-- An accepted position is a valid row of the RoPE table. -/
theorem qsaPosOk_row (posBase pos maxCells : Int) (h : qsaPosOk posBase pos maxCells) :
    0 ≤ posBase + pos ∧ posBase + pos < maxCells := by
  obtain ⟨h1, h2, h3⟩ := h
  constructor <;> omega

/-- Floats per layer of the half-level dump (`A = n_head * head_dim`, `nvk = n_head_kv * head_dim`). -/
def dumpStride (n hc A nvk : Nat) : Nat := 2 * n + 2 * hc + A + 5 * nvk + 8

/-- Offset of the KV slots inside a layer's dump. -/
def dumpKvBase (n hc A nvk : Nat) : Nat := 2 * n + 2 * hc + A + 2 * nvk + 8

/-- The three trailing dump slots end exactly at the stride, so layers never overlap. -/
theorem dump_slots_end_at_stride (n hc A nvk : Nat) : dumpKvBase n hc A nvk + 3 * nvk = dumpStride n hc A nvk := by
  unfold dumpKvBase dumpStride; omega

/-- The two half-pool prefixes fit in the `nvk` slot reserved for them. -/
theorem dump_pool_prefix_fits (nvk : Nat) : nvk / 2 + nvk / 2 ≤ nvk := by omega

/-- Row-bounds of `embed_row`: a row inside the table stays inside the code plane. -/
theorem embed_row_in_bounds (rows rowCodes token codesBytes : Nat) (h : codesBytes = rows * rowCodes)
    (ht : token < rows) : (token + 1) * rowCodes ≤ codesBytes := by
  subst h
  exact Nat.mul_le_mul_right rowCodes (Nat.succ_le_of_lt ht)

end Strata.LayerCompose

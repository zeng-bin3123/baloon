/-!
# MODULE: expert_layout_table
SOURCES: src/kernels/cpu/expert_layout.cpp, src/kernels/cpu/native_expert.cpp
The per-layer expert table loaded from `native_experts.txt` (`expert_layout_load`) and the native GGUF
format sizing (`native_fmt`).  The theory proves that a table accepted by the contiguity check places
every layer at the running prefix sum, that the total is that sum, that `max_blob` bounds every layer's
blob (so one staging buffer fits all), and that a native Q2_0 expert has exactly the legacy blob size.

## Source semantics
* No `native_experts.txt` ⇒ legacy pack: total = n_layers · n_expert · BLOB.
* Native: line l gives (offset, blob); the loop requires `offset[l] == at`, then `at += blob·n_expert`;
  `total = at`.  Any gap or overlap refuses the pack.
* `native_fmt`: up_off = gu_row·n_ff, down_off = 2·up_off, bytes = down_off + d_row·n_embd; a row of
  a block format is n/blk blocks of the type's block bytes; geometry must be whole blocks.
* v2 lines store three GGUF offsets per layer at indices 3l, 3l+1, 3l+2.

## Form template identifiers
checkContig → layerContiguityCheck, totalBytes → packTotal, maxB → maxBlob, NativeFmt → nativeFormat,
fmtBytes → nativeBlobBytes, rowSize → ggmlRowSize, legacyTotal → legacyPackTotal
-/

namespace Strata.Kernels.Cpu.Layout

-- § legacy pack

/-- Legacy total size of a pack: every expert is one 1,382,400-byte blob. -/
def legacyTotal (nLayers nExpert : Nat) : Nat := nLayers * nExpert * 1382400

/-- Legacy layer l starts at l·n_expert·BLOB and the layers tile the pack. -/
theorem legacy_layer_in_pack (nLayers nExpert l : Nat) (hl : l < nLayers) :
    l * nExpert * 1382400 + nExpert * 1382400 ≤ legacyTotal nLayers nExpert := by
  unfold legacyTotal
  have h : l + 1 ≤ nLayers := hl
  have := Nat.mul_le_mul_right (nExpert * 1382400) h
  rw [Nat.add_mul, Nat.one_mul] at this
  rw [Nat.mul_assoc, Nat.mul_assoc]
  exact this

-- § contiguity check of native_experts.txt

/-- The loader's loop: each (offset, blob) must sit at `at`; returns the final `at` (= total). -/
def checkContig (nExpert : Nat) : Nat → List (Nat × Nat) → Option Nat
  | acc, [] => some acc
  | acc, (off, b) :: t => if off = acc then checkContig nExpert (acc + b * nExpert) t else none

/-- Sum of the layers' sizes. -/
def totalBytes (nExpert : Nat) : List (Nat × Nat) → Nat
  | [] => 0
  | (_, b) :: t => b * nExpert + totalBytes nExpert t

/-- An accepted table's total is the start plus the sum of all layer sizes. -/
theorem checkContig_total (nExpert : Nat) (ls : List (Nat × Nat)) (acc tot : Nat)
    (h : checkContig nExpert acc ls = some tot) : tot = acc + totalBytes nExpert ls := by
  induction ls generalizing acc with
  | nil =>
    simp only [checkContig, Option.some.injEq] at h
    simp [totalBytes, h]
  | cons p t ih =>
    obtain ⟨off, b⟩ := p
    simp only [checkContig] at h
    split at h
    · have := ih _ h
      simp only [totalBytes]
      omega
    · contradiction

/-- In an accepted table, layer i starts exactly at the sum of the sizes of layers before it. -/
theorem checkContig_offset (nExpert : Nat) (ls : List (Nat × Nat)) (acc tot : Nat)
    (h : checkContig nExpert acc ls = some tot) (i : Nat) (hi : i < ls.length) :
    (ls[i]).1 = acc + totalBytes nExpert (ls.take i) := by
  induction ls generalizing acc i with
  | nil => simp at hi
  | cons p t ih =>
    obtain ⟨off, b⟩ := p
    simp only [checkContig] at h
    split at h
    · rename_i hoff
      cases i with
      | zero => simp [totalBytes, hoff]
      | succ j =>
        have hj : j < t.length := by simpa using hi
        have := ih _ h j hj
        simp only [List.getElem_cons_succ, List.take_succ_cons, totalBytes]
        omega
    · contradiction

/-- A missing or non-contiguous layer refuses the whole table. -/
theorem checkContig_gap (nExpert acc off b : Nat) (t : List (Nat × Nat)) (h : off ≠ acc) :
    checkContig nExpert acc ((off, b) :: t) = none := by
  simp [checkContig, h]

-- § max_blob bounds every layer (one staging buffer size)

/-- `max_blob` over the layers' blob sizes. -/
def maxB : List Nat → Nat
  | [] => 0
  | b :: t => max b (maxB t)

/-- Every layer's blob fits in a `max_blob`-sized staging buffer. -/
theorem le_maxB (l : List Nat) (b : Nat) (hb : b ∈ l) : b ≤ maxB l := by
  induction l with
  | nil => simp at hb
  | cons a t ih =>
    simp only [maxB]
    rcases List.mem_cons.mp hb with h | h
    · subst h; exact Nat.le_max_left _ _
    · exact Nat.le_trans (ih h) (Nat.le_max_right _ _)

-- § native format sizing

/-- ggml row size of a block format: n/blk blocks of `blockBytes`. -/
def rowSize (blockBytes blk n : Nat) : Nat := n / blk * blockBytes

/-- The fields of `NativeFmt` that `native_fmt` computes. -/
structure NativeFmt where
  guRow : Nat
  dRow : Nat
  nEmbd : Nat
  nFF : Nat

def NativeFmt.upOff (f : NativeFmt) : Nat := f.guRow * f.nFF
def NativeFmt.downOff (f : NativeFmt) : Nat := 2 * f.upOff
def NativeFmt.bytes (f : NativeFmt) : Nat := f.downOff + f.dRow * f.nEmbd

/-- The three matrices tile the native blob: gate rows, up rows, down rows. -/
theorem native_regions (f : NativeFmt) :
    f.upOff + f.guRow * f.nFF = f.downOff ∧ f.downOff + f.dRow * f.nEmbd = f.bytes := by
  simp only [NativeFmt.downOff, NativeFmt.upOff, NativeFmt.bytes]
  generalize f.guRow * f.nFF = A
  exact ⟨by omega, trivial⟩

/-- Up row r (< n_ff) of a native blob lies before the down region. -/
theorem native_up_row_in_region (f : NativeFmt) (r : Nat) (hr : r < f.nFF) :
    f.upOff + r * f.guRow + f.guRow ≤ f.downOff := by
  simp only [NativeFmt.downOff, NativeFmt.upOff]
  have : r * f.guRow + f.guRow ≤ f.nFF * f.guRow := by
    have := Nat.mul_le_mul_right f.guRow (show r + 1 ≤ f.nFF from hr)
    rw [Nat.add_mul, Nat.one_mul] at this; exact this
  rw [Nat.mul_comm f.nFF] at this
  generalize f.guRow * f.nFF = A at *
  generalize r * f.guRow = B at *
  omega

/-- A native pack whose three projections are GGUF Q2_0 (18 B / 64) has exactly the legacy blob size. -/
theorem native_q2_0_is_legacy_size :
    ({ guRow := rowSize 18 64 2560, dRow := rowSize 18 64 640, nEmbd := 2560, nFF := 640 } :
      NativeFmt).bytes = 1382400 := by decide

/-- Whole-block check: H and FF are whole Q2_0 blocks and H is whole 256-blocks, but FF = 640 is not,
so a 256-wide i-quant down projection is refused and those layers keep their down rows in Q2_0. -/
theorem geometry_whole_blocks : 2560 % 64 = 0 ∧ 640 % 64 = 0 ∧ 2560 % 256 = 0 ∧ 640 % 256 ≠ 0 := by
  decide

/-- v2 lines: the three GGUF offsets of layer l are stored in distinct slots inside 3·n_layers. -/
theorem gguf_off_index (l k n : Nat) (hl : l < n) (hk : k < 3) : 3 * l + k < 3 * n := by omega

end Strata.Kernels.Cpu.Layout

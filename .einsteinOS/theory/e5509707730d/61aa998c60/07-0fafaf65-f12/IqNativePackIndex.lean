/-!
# MODULE: iq_native_pack_index
SOURCES: tools/iq_pack.py
Integer model of the native pack for i-quant model files: per-layer expert blob layout, running offsets,
64-byte arena alignment, BF16 router narrowing, extra.bin offsets, layer counting.

## Source semantics
experts.bin: per layer, 512 blobs of [gate | up | down] raw GGUF slices; blob size is per layer. Layer offset
= sum over earlier layers of blob*512. Float tensors are written to dense.bin padded to 64. An F32 router is
narrowed to BF16 only if the low 16 bits of every word are zero (exact), otherwise refused.

## Form template identifiers
`N_EXPERT` -> 512; `ALIGN` -> 64; `layout` -> `expertsOffset`; `kind` map -> `kindOf`;
`index_from_base` extra offsets -> `extraOffset`.
-/

namespace StrataIqPack

-- § constants

/-- Experts per layer. -/
def nExpert : Nat := 512
/-- Arena alignment. -/
def align : Nat := 64

-- § expert layout

/-- Pad to 64. -/
def pad64 (n : Nat) : Nat := (64 - n % 64) % 64

/-- Padding is below 64 and aligns the length. -/
theorem pad64_spec (n : Nat) : pad64 n < 64 ∧ (n + pad64 n) % 64 = 0 := by
  unfold pad64; omega

/-- Offset of layer l in experts.bin, given the per-layer blob size. -/
def expertsOffset (blob : Nat → Nat) : Nat → Nat
  | 0 => 0
  | l + 1 => expertsOffset blob l + blob l * 512

/-- Layer l+1 starts where layer l's 512 blobs end. -/
theorem expertsOffset_succ (blob : Nat → Nat) (l : Nat) :
    expertsOffset blob (l + 1) = expertsOffset blob l + blob l * 512 := rfl

/-- Offsets never decrease with the layer index. -/
theorem expertsOffset_mono (blob : Nat → Nat) (l k : Nat) :
    expertsOffset blob l ≤ expertsOffset blob (l + k) := by
  induction k with
  | zero => exact Nat.le_refl _
  | succ k ih =>
    have h : expertsOffset blob (l + (k + 1)) = expertsOffset blob (l + k) + blob (l + k) * 512 := rfl
    omega

/-- A blob is gate + up + down slices; with each role tensor divisible by 512 the 512 blobs cover the three tensors exactly. -/
theorem blob_cover (g u d : Nat) (hg : g % 512 = 0) (hu : u % 512 = 0) (hd : d % 512 = 0) :
    (g / 512 + u / 512 + d / 512) * 512 = g + u + d := by omega

/-- Layer count is 1 + max layer index: every layer index is below it. -/
theorem le_foldl_max_acc : ∀ (ls : List Nat) (acc : Nat), acc ≤ ls.foldl max acc
  | [], acc => Nat.le_refl _
  | x :: xs, acc => Nat.le_trans (Nat.le_max_left acc x) (le_foldl_max_acc xs (max acc x))

/-- Every listed element is bounded by the fold-max. -/
theorem le_foldl_max : ∀ (ls : List Nat) (acc l : Nat), l ∈ ls → l ≤ ls.foldl max acc
  | [], _, _, h => by simp at h
  | x :: xs, acc, l, h => by
    rcases List.mem_cons.mp h with h1 | h1
    · rw [h1]
      exact Nat.le_trans (Nat.le_max_right acc x) (le_foldl_max_acc xs (max acc x))
    · exact le_foldl_max xs (max acc x) l h1

/-- n_layers = 1 + max index exceeds every layer index present. -/
theorem layer_lt_count (ls : List Nat) (l : Nat) (h : l ∈ ls) : l < 1 + ls.foldl max 0 := by
  have := le_foldl_max ls 0 l h
  omega

-- § dense.bin and routers

/-- Index kind codes: BF16 -> 4, F16 -> 5, F32 -> 2. -/
def kindOf : String → Nat
  | "BF16" => 4
  | "F16" => 5
  | _ => 2

/-- The three float kinds are distinct. -/
theorem kinds_distinct : kindOf "BF16" ≠ kindOf "F16" ∧ kindOf "F16" ≠ kindOf "F32" ∧ kindOf "BF16" ≠ kindOf "F32" := by
  decide

/-- Router F32 -> BF16 narrowing is exact exactly when the low 16 bits are zero; then the high half recovers the word. -/
theorem router_narrowing_exact (u : Nat) (h : u < 4294967296) (hz : u % 65536 = 0) :
    u / 65536 < 65536 ∧ (u / 65536) * 65536 = u := by omega

/-- If the low bits are not zero the narrowing is refused (it would lose information). -/
theorem router_narrowing_lossy (u : Nat) (hz : u % 65536 ≠ 0) : (u / 65536) * 65536 ≠ u := by omega

-- § extra.bin offsets

/-- Offset of the i-th extra tensor: sum of earlier lengths each padded to 64. -/
def extraOffset : List Nat → Nat
  | [] => 0
  | b :: bs => (b + pad64 b) + extraOffset bs

/-- Each padded extra tensor starts on a 64-byte boundary relative to the previous. -/
theorem extra_step_aligned (b : Nat) : (b + pad64 b) % 64 = 0 := (pad64_spec b).2

/-- Padding never shrinks a tensor. -/
theorem extra_step_ge (b : Nat) : b ≤ b + pad64 b := Nat.le_add_right _ _

/-- The total extra offset is a multiple of 64. -/
theorem extraOffset_aligned : ∀ bs : List Nat, extraOffset bs % 64 = 0
  | [] => rfl
  | b :: bs => by
    have h := extra_step_aligned b
    have ih := extraOffset_aligned bs
    unfold extraOffset
    omega

-- § index arena

/-- Index arena offset rounding: each row occupies its size rounded up to 64. -/
def roundRow (n : Nat) : Nat := (n + 63) / 64 * 64

/-- Rounded rows are aligned and wasteful by less than 64. -/
theorem roundRow_spec (n : Nat) : roundRow n % 64 = 0 ∧ n ≤ roundRow n ∧ roundRow n < n + 64 := by
  unfold roundRow; omega

end StrataIqPack

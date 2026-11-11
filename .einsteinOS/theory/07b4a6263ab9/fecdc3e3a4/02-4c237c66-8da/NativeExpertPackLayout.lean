/-!
# MODULE: native_expert_pack_layout
SOURCES: tools/iq_pack.py
Builds a "native" Strata pack for i-quant model files: every quantized tensor stays in its GGUF form;
float tensors go to a 64-byte-aligned dense.bin (or are shared with a base pack); experts are laid out per
layer as 512 blobs of [gate | up | down] rows; F32 routers whose values are exactly BF16 are narrowed.

## Source semantics
* Alignment: each tensor is followed by `(-len) % 64` zero bytes; `write_index` assigns arena offsets as a
  running sum of `ceil(len/64)·64`, so every offset is 64-aligned and no two tensors overlap.
* Router narrowing: an F32 word `u` is accepted only if `u & 0xFFFF == 0`; then `u >> 16` is written as
  BF16. That is exact (lossless) and equals round-to-nearest-even; otherwise the tool refuses.
* Expert layout: layer `l` starts at `offset_l = Σ_{j<l} 512·blob_j`, `blob = gate + up + down` per-expert
  bytes (gate and up equal); expert `e` of layer `l` is the slice `[offset_l + e·blob, offset_l + (e+1)·blob)`.
* `--base`: float tensors must be byte-identical to the base model's; a tensor float here but quantized
  in the base goes to extra.bin as file 3, kind 4 (BF16).

## Form template identifiers
ALIGN → align64, `(-len) % ALIGN` → pad64, write_index offset loop → assignOffsets,
F32 router check and `u >> 16` → routerExact / routerBf16, layout loop → layerOffsets,
per-expert slice → expertStart, N_EXPERT → nExpert
-/

namespace Strata.IqPack

-- § alignment

/-- The arena/file alignment. -/
def align64 : Nat := 64

/-- Python's `(-len) % 64` for a nonnegative length. -/
def pad64 (n : Nat) : Nat := (64 - n % 64) % 64

/-- Padding is less than one alignment unit. -/
theorem pad64_lt (n : Nat) : pad64 n < 64 := by unfold pad64; omega

/-- Padding makes the length a multiple of 64. -/
theorem pad64_aligns (n : Nat) : (n + pad64 n) % 64 = 0 := by unfold pad64; omega

/-- An already-aligned tensor gets no padding. -/
theorem pad64_aligned (n : Nat) (h : n % 64 = 0) : pad64 n = 0 := by unfold pad64; omega

/-- `write_index`'s `(len + 63) // 64 * 64` is the same as `len + pad64 len`, so dense.bin's file layout
and the index's arena layout coincide. -/
theorem roundup_eq_pad (n : Nat) : (n + 63) / 64 * 64 = n + pad64 n := by unfold pad64; omega

-- § index offsets

/-- `r[5] = cur; cur += ceil(len/64)·64`. -/
def assignOffsets : Nat → List Nat → List Nat
  | _, [] => []
  | cur, n :: ns => cur :: assignOffsets (cur + (n + 63) / 64 * 64) ns

/-- Every assigned offset is 64-aligned (starting from 0). -/
theorem assignOffsets_aligned : ∀ (ns : List Nat) (cur : Nat), cur % 64 = 0 →
    ∀ o ∈ assignOffsets cur ns, o % 64 = 0
  | [], _, _, o, h => by simp [assignOffsets] at h
  | n :: ns, cur, hat, o, h => by
    simp only [assignOffsets, List.mem_cons] at h
    cases h with
    | inl he => rw [he]; exact hat
    | inr hm => exact assignOffsets_aligned ns _ (by omega) o hm

/-- Consecutive tensors never overlap: a tensor ends before the next offset. -/
theorem assignOffsets_no_overlap (cur n m : Nat) (ns : List Nat) :
    ∃ nxt, assignOffsets cur (n :: m :: ns) = cur :: nxt :: assignOffsets (nxt + (m + 63) / 64 * 64) ns ∧
      cur + n ≤ nxt := by
  refine ⟨cur + (n + 63) / 64 * 64, rfl, ?_⟩
  omega

/-- One offset per row. -/
theorem assignOffsets_length : ∀ (ns : List Nat) (cur : Nat), (assignOffsets cur ns).length = ns.length
  | [], _ => rfl
  | _ :: ns, cur => by simp [assignOffsets, assignOffsets_length ns]

-- § F32 router narrowing to BF16

/-- The check: low 16 bits zero. -/
def routerExact (u : Nat) : Bool := u % 65536 == 0

/-- `(u >> 16).astype(uint16)`. -/
def routerBf16 (u : Nat) : Nat := (u / 65536) % 65536

/-- Round-to-nearest-even f32 → bf16 (as in bf16_bits.hpp). -/
def bf16Rne (i : Nat) : Nat := ((i + (i / 65536) % 2 + 32767) / 65536) % 65536

/-- An accepted router word is narrowed losslessly: widening the BF16 back gives the same F32 word. -/
theorem router_lossless (u : Nat) (hu : u < 4294967296) (h : routerExact u = true) :
    routerBf16 u * 65536 = u := by
  unfold routerExact at h
  unfold routerBf16
  have h0 : u % 65536 = 0 := by simpa using h
  omega

/-- On accepted words the truncation is the same as the engine's round-to-nearest-even conversion. -/
theorem router_matches_rne (u : Nat) (h : routerExact u = true) : routerBf16 u = bf16Rne u := by
  unfold routerExact at h
  have h0 : u % 65536 = 0 := by simpa using h
  unfold routerBf16 bf16Rne
  have hq : (u + (u / 65536) % 2 + 32767) / 65536 = u / 65536 := by omega
  rw [hq]

/-- A refused word really is not BF16-representable: truncating it would lose bits. -/
theorem router_refusal_justified (u : Nat) (h : routerExact u = false) :
    routerBf16 u * 65536 ≠ u := by
  unfold routerExact at h
  unfold routerBf16
  have h0 : u % 65536 ≠ 0 := by simpa using h
  omega

-- § expert layout

/-- Experts per layer. -/
def nExpert : Nat := 512

/-- `offset += blob * N_EXPERT` per layer. -/
def layerOffsets : Nat → List Nat → List Nat
  | _, [] => []
  | off, b :: bs => off :: layerOffsets (off + b * nExpert) bs

/-- One offset per layer. -/
theorem layerOffsets_length : ∀ (bs : List Nat) (off : Nat), (layerOffsets off bs).length = bs.length
  | [], _ => rfl
  | _ :: bs, off => by simp [layerOffsets, layerOffsets_length bs]

/-- The next layer starts exactly where this layer's 512 blobs end (no gap, no overlap). -/
theorem layers_contiguous (off b c : Nat) (bs : List Nat) :
    layerOffsets off (b :: c :: bs) = off :: (off + b * nExpert) :: layerOffsets (off + b * nExpert + c * nExpert) bs := rfl

/-- Start of expert `e` in a layer cur `off` with blob size `blob`. -/
def expertStart (off blob e : Nat) : Nat := off + e * blob

/-- Every expert of a layer lies inside the layer's region. -/
theorem expert_in_layer (off blob e : Nat) (he : e < nExpert) :
    off ≤ expertStart off blob e ∧ expertStart off blob e + blob ≤ off + blob * nExpert := by
  unfold expertStart
  have h : (e + 1) * blob ≤ nExpert * blob := Nat.mul_le_mul_right blob (by omega)
  rw [Nat.succ_mul] at h
  rw [Nat.mul_comm blob nExpert]
  omega

/-- Distinct experts occupy disjoint slices. -/
theorem experts_disjoint (off blob e1 e2 : Nat) (h : e1 < e2) :
    expertStart off blob e1 + blob ≤ expertStart off blob e2 := by
  unfold expertStart
  have : (e1 + 1) * blob ≤ e2 * blob := Nat.mul_le_mul_right blob h
  rw [Nat.succ_mul] at this
  omega

end Strata.IqPack

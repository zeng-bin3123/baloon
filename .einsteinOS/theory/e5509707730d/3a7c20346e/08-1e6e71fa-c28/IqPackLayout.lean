/-!
# MODULE: iq_pack_layout
SOURCES: tools/iq_pack.py
Native pack for the Q2_0 / IQ2_XS / IQ3_XXS model files: experts kept in GGUF form as per-layer blobs
`[gate | up | down]` x 512, an index table with 64-byte-aligned arena offsets, and BF16 router narrowing.

## Source semantics
* Per layer, blob size = per-expert bytes of gate + up + down, where per-expert bytes = tensor bytes / 512;
  gate and up must agree. The experts arena is the running sum of `blob * 512`.
* The index arena size is the running sum of each tensor's size rounded up to ALIGN = 64.
* An F32 router whose low 16 bits are all zero is exactly a BF16 widened by 16 bits; it is stored as BF16.
* `n_layers = 1 + max layer index` found among `*_exps.weight` tensors.

## Form template identifiers
- N_EXPERT / blob / offset       → `nExpert` / blob sizes list / `totalBytes`
- ALIGN pad                      → `alignUp64`
- (u & 0xFFFF)==0, u>>16         → `routerNarrow`
- write_index at                 → `alignedTotal`
-/

namespace Strata.IqPack

/-- Experts per layer. -/
def nExpert : Nat := 512

-- § experts arena

/-- Total arena bytes for a list of per-layer blob sizes (each layer holds 512 blobs). -/
def totalBytes : List Nat → Nat
  | [] => 0
  | b :: bs => b * 512 + totalBytes bs

/-- The arena is `512 * (sum of blob sizes)`. -/
theorem totalBytes_eq_mul_sum (bs : List Nat) : totalBytes bs = 512 * bs.sum := by
  induction bs with
  | nil => simp [totalBytes]
  | cons b bs ih =>
    simp only [totalBytes, List.sum_cons, ih]
    omega

/-- Per-expert bytes recover exactly when the tensor holds 512 equal experts. -/
theorem per_expert_exact (per : Nat) : (512 * per) / 512 = per := by
  omega

/-- A layer blob is gate + up + down bytes, and the layer's arena slice is 512 of them. -/
theorem layer_slice (g u d : Nat) : (g + u + d) * 512 = g * 512 + u * 512 + d * 512 := by
  omega

-- § 64-byte aligned index

/-- Round up to ALIGN = 64. -/
def alignUp64 (n : Nat) : Nat := (n + 63) / 64 * 64

/-- Alignment is monotone, bounded and lands on a boundary. -/
theorem alignUp64_spec (n : Nat) : n ≤ alignUp64 n ∧ alignUp64 n < n + 64 ∧ alignUp64 n % 64 = 0 := by
  unfold alignUp64
  omega

/-- Running aligned total of a list of tensor sizes (the `at` in `write_index`). -/
def alignedTotal : List Nat → Nat
  | [] => 0
  | n :: ns => alignUp64 n + alignedTotal ns

/-- The arena is never smaller than the raw tensor bytes. -/
theorem sum_le_alignedTotal (ns : List Nat) : ns.sum ≤ alignedTotal ns := by
  induction ns with
  | nil => simp [alignedTotal]
  | cons n ns ih =>
    have h := alignUp64_spec n
    simp only [alignedTotal, List.sum_cons]
    omega

/-- The arena size is itself a multiple of 64. -/
theorem alignedTotal_aligned (ns : List Nat) : alignedTotal ns % 64 = 0 := by
  induction ns with
  | nil => simp [alignedTotal]
  | cons n ns ih =>
    have h := alignUp64_spec n
    simp only [alignedTotal]
    omega

-- § router narrowing

/-- Keep the high 16 bits of an F32 pattern. -/
def routerNarrow (u : Nat) : Nat := u / 65536

/-- When the low 16 bits are zero, narrowing then widening reproduces the F32 pattern exactly. -/
theorem router_narrow_exact (u : Nat) (h : u % 65536 = 0) : routerNarrow u * 65536 = u := by
  unfold routerNarrow
  omega

/-- Widening a BF16 pattern by 16 bits and narrowing back is the identity. -/
theorem bf16_widen_narrow (b : Nat) : routerNarrow (b * 65536) = b := by
  unfold routerNarrow
  omega

/-- A narrowed router fits in 16 bits when the F32 pattern is below 2^32. -/
theorem router_narrow_lt (u : Nat) (h : u < 4294967296) : routerNarrow u < 65536 := by
  unfold routerNarrow
  omega

-- § layer count

/-- `n_layers = 1 + max index`: every layer index found is below the layer count. -/
theorem layer_index_lt (l mx : Nat) (h : l ≤ mx) : l < 1 + mx := by
  omega

end Strata.IqPack

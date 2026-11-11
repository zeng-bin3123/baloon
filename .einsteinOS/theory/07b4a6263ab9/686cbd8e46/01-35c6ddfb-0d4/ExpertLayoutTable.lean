/-!
# MODULE: expert_layout_table
SOURCES: src/kernels/cpu/expert_layout.cpp
The per-layer expert table: legacy (uniform Q2_0 blob) packs, native (IQ) packs described by
`native_experts.txt`, the contiguity validation of their per-layer regions, and the AVX-512
runtime gate that selects the AVX-512 or AVX2 kernels.

## Source semantics
* No `native_experts.txt`: total = n_layers * n_expert * BLOB, every layer uniform.
* Native packs: each line gives (layer, gate/up type, down type, offset, blob bytes [, gguf offsets
  [, shard]]); a layer index outside [0, n_layers) or a blob size disagreeing with its formats rejects.
* After parsing, layer l must start exactly where layer l-1 ended (offset[l] = sum_{k<l} bytes[k] * E);
  a missing layer keeps the sentinel ~0 and fails.  On success total = sum bytes[l] * E and the
  per-layer regions tile [0, total) without gaps or overlap.
* `cpu_avx512_ok` needs CPUID leaf >= 7, OSXSAVE, XCR0 & 0xE6 == 0xE6 (OS saves SSE/AVX/opmask/ZMM
  state) and the five AVX-512 feature bits; STRATA_FORCE_AVX2=1 forces it off.

## Form template identifiers
expert_layout_load -> loadLayout, contiguity loop -> checkContig, ExpertLayout::total -> total,
cpu_avx512_ok -> avx512Ok, q2_rows_any / act_quant_any -> kernelChoice
-/

namespace Strata.ExpertLayout

-- § legacy packs

/-- Size of one legacy Q2_0 expert blob. -/
def BLOB : Nat := 1382400

/-- Legacy total: every layer holds `n_expert` uniform blobs. -/
def legacyTotal (nLayers nExpert : Nat) : Nat := nLayers * nExpert * BLOB

/-- For the qwen4exp shape (48 layers x 512 experts) the legacy pack is 33,973,862,400 bytes. -/
theorem legacy_total_qwen4exp : legacyTotal 48 512 = 33973862400 := by decide

-- § native line validation

/-- A parsed line is accepted only for a layer in range and a blob size matching its formats. -/
def lineOk (l nLayers blob fmtBytes : Int) : Bool := decide (0 ≤ l ∧ l < nLayers ∧ blob = fmtBytes)

/-- An accepted line names a real layer and carries the size its formats imply. -/
theorem lineOk_sound (l n b f : Int) (h : lineOk l n b f = true) : 0 ≤ l ∧ l < n ∧ b = f := by
  unfold lineOk at h; simpa using h

-- § contiguity

/-- The contiguity pass over (offset, bytes) per layer: returns the running end on success. -/
def checkContig (e : Nat) : Nat → List (Nat × Nat) → Option Nat
  | acc, [] => some acc
  | acc, (o, b) :: rest => if o = acc then checkContig e (acc + b * e) rest else none

/-- Sum of `bytes * E` over the layers. -/
def regionSum (e : Nat) : List (Nat × Nat) → Nat
  | [] => 0
  | (_, b) :: rest => b * e + regionSum e rest

/-- On success the total is the start plus the sum of every layer's region. -/
theorem checkContig_total (e : Nat) (xs : List (Nat × Nat)) (start t : Nat)
    (h : checkContig e start xs = some t) : t = start + regionSum e xs := by
  induction xs generalizing start with
  | nil => simp [checkContig] at h; simp [regionSum, h]
  | cons x rest ih =>
    obtain ⟨o, b⟩ := x
    simp only [checkContig] at h
    split at h
    · have := ih _ h; simp only [regionSum]; omega
    · cases h

/-- On success each layer starts at the running end: the first layer starts at `start`, and the
next layer starts exactly where the previous one ends. -/
theorem checkContig_adjacent (e : Nat) (o b : Nat) (rest : List (Nat × Nat)) (start t : Nat)
    (h : checkContig e start ((o, b) :: rest) = some t) :
    o = start ∧ checkContig e (o + b * e) rest = some t := by
  simp only [checkContig] at h
  split at h
  · rename_i ho; subst ho; exact ⟨rfl, h⟩
  · cases h

/-- Regions are pairwise disjoint and in order: for two consecutive accepted layers the second
begins at the first's end. -/
theorem checkContig_consecutive (e o1 b1 o2 b2 : Nat) (rest : List (Nat × Nat)) (start t : Nat)
    (h : checkContig e start ((o1, b1) :: (o2, b2) :: rest) = some t) : o2 = o1 + b1 * e := by
  obtain ⟨h1, h2⟩ := checkContig_adjacent e o1 b1 _ start t h
  exact (checkContig_adjacent e o2 b2 rest _ t h2).1

/-- A layer with no line keeps the sentinel offset `~0ull`; it is rejected whenever the running end
is a real (smaller) offset. -/
theorem missing_layer_rejected (e b : Nat) (rest : List (Nat × Nat)) (acc : Nat)
    (hacc : acc < 18446744073709551615) :
    checkContig e acc ((18446744073709551615, b) :: rest) = none := by
  simp only [checkContig]
  split
  · omega
  · rfl

/-- The legacy layout is the special case of uniform contiguous regions. -/
theorem legacy_is_contiguous (e : Nat) :
    checkContig e 0 [(0, BLOB), (BLOB * e, BLOB)] = some (2 * (BLOB * e)) := by
  simp only [checkContig, if_true, Nat.zero_add]
  congr 1; omega

-- § AVX-512 runtime gate

/-- Bit `k` of a register. -/
def bit (r k : Nat) : Bool := r / 2 ^ k % 2 = 1

/-- `cpu_avx512_ok` as a pure function of the probed values. -/
def avx512Ok (forceAvx2 : Bool) (maxLeaf ecx1 xcr0 ebx7 ecx7 : Nat) : Bool :=
  !forceAvx2 && decide (7 ≤ maxLeaf) && bit ecx1 27 && decide (xcr0 % 256 &&& 0xE6 = 0xE6) &&
    bit ebx7 16 && bit ebx7 30 && bit ebx7 31 && bit ecx7 11 && bit ecx7 1

/-- The forced-AVX2 override always wins. -/
theorem force_avx2_wins (a b c d e : Nat) : avx512Ok true a b c d e = false := by
  simp [avx512Ok]

/-- The XCR0 test `(x & 0xE6) == 0xE6` holds exactly when bits 1, 2, 5, 6, 7 are all set
(SSE, AVX, opmask, ZMM_Hi256, Hi16_ZMM state), checked over every low byte. -/
theorem xcr0_mask_bits : ∀ x ∈ List.range 256,
    (x &&& 0xE6 == 0xE6) = (bit x 1 && bit x 2 && bit x 5 && bit x 6 && bit x 7) := by
  set_option maxRecDepth 100000 in decide

/-- Which kernel family `q2_rows_any` and `act_quant_any` use. -/
def kernelChoice (ok : Bool) : String := if ok then "avx512" else "avx2"

/-- Without the OS saving ZMM state (bit 6 clear) the AVX2 kernels are chosen even if CPUID
advertises AVX-512: the gate cannot select an instruction set that would fault. -/
theorem no_zmm_state_means_avx2 (lf e1 x b7 c7 : Nat) (hx : (x % 256) / 64 % 2 = 0) :
    kernelChoice (avx512Ok false lf e1 x b7 c7) = "avx2" := by
  have hlt : x % 256 < 256 := Nat.mod_lt _ (by decide)
  have hmask : ¬ (x % 256 &&& 0xE6 = 0xE6) := by
    have key := xcr0_mask_bits (x % 256) (List.mem_range.2 hlt)
    intro hm
    rw [hm] at key
    simp [bit, Bool.and_eq_true] at key
    omega
  simp [kernelChoice, avx512Ok, hmask]

end Strata.ExpertLayout

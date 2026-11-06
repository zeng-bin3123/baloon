/-!
# MODULE: expert_layout_table
SOURCES: src/kernels/cpu/expert_layout.cpp
The per-layer expert table loader: contiguity validation of `native_experts.txt`, the resulting total size, the
running maximum blob size, the GGUF-offset indexing, the AVX-512 gating predicate (CPUID bits and XCR0 mask), and
the fixed-size default layout.

## Source semantics
* Layers must tile the pack with no gap: layer `l` starts where layer `l-1` ended (`offset[l] = sum of
  bytes[k]*n_expert for k < l`); a violation is an error. On success `total` is that sum.
* `max_blob` is the maximum layer blob size.
* v2 lines store three GGUF offsets per layer in an array of `3*n_layers`, indexed `3*l + {0,1,2}`.
* AVX-512 is used only if the OS saves the AVX-512 state (`xcr0 & 0xE6 == 0xE6`) and CPUID leaf 7 reports F, BW, VL
  (EBX bits 16, 30, 31), VNNI (ECX bit 11) and VBMI (ECX bit 1).
* With no `native_experts.txt` the layout is `n_layers * n_expert * BLOB` bytes.

## Form template identifiers
contig → contiguity_check, maxOf → max_blob_fold, xcr0Ok → os_state_gate, avx512Ok → isa_gate,
gguf index → gguf_offset_slot.
-/

namespace Strata.Kernels.Cpu.ExpertLayoutTable

-- § contiguity validation

/-- Walk (offset, bytes) pairs: each offset must equal the running total; return the final total, or none. -/
def contig (n : Nat) : List (Nat × Nat) → Nat → Option Nat
  | [], cur => some cur
  | (o, b) :: rest, cur => if o = cur then contig n rest (cur + b * n) else none

/-- Sum of the layers' sizes (`bytes * n_expert`). -/
def layerTotal (n : Nat) (L : List (Nat × Nat)) : Nat := (L.map (fun p => p.2 * n)).sum

/-- A successful validation returns the starting offset plus the sum of all layers' sizes. -/
theorem contig_total (n : Nat) (L : List (Nat × Nat)) :
    ∀ cur t : Nat, contig n L cur = some t → t = cur + layerTotal n L := by
  induction L with
  | nil =>
    intro cur t h
    simp only [contig, Option.some.injEq] at h
    simp [layerTotal, h]
  | cons p rest ih =>
    obtain ⟨o, b⟩ := p
    intro cur t h
    by_cases ho : o = cur
    · simp only [contig, if_pos ho] at h
      have := ih _ _ h
      simp only [layerTotal, List.map_cons, List.sum_cons] at this ⊢
      omega
    · simp [contig, ho] at h

/-- A gap or overlap at the first layer makes validation fail. -/
theorem contig_fails_on_gap (n o b cur : Nat) (rest : List (Nat × Nat)) (h : o ≠ cur) :
    contig n ((o, b) :: rest) cur = none := by
  simp only [contig, if_neg h]

/-- The empty table validates to its starting offset. -/
theorem contig_nil (n cur : Nat) : contig n [] cur = some cur := by
  rfl

/-- An exactly-contiguous single layer starting at 0 has size `bytes * n_expert`. -/
theorem contig_single (n b : Nat) : contig n [(0, b)] 0 = some (0 + b * n) := by
  simp [contig]

-- § running maximum

/-- Fold of `max` over the blob sizes, as `max_blob` is computed. -/
def maxOf : List Nat → Nat → Nat
  | [], m => m
  | b :: bs, m => maxOf bs (max m b)

/-- The fold never goes below its initial value. -/
theorem maxOf_ge_init (l : List Nat) : ∀ m : Nat, m ≤ maxOf l m := by
  induction l with
  | nil => intro m; exact Nat.le_refl m
  | cons b bs ih =>
    intro m
    simp only [maxOf]
    have := ih (max m b)
    omega

/-- Every element is at most the fold result. -/
theorem le_maxOf (l : List Nat) : ∀ m x : Nat, x ∈ l → x ≤ maxOf l m := by
  induction l with
  | nil =>
    intro m x hx
    simp at hx
  | cons b bs ih =>
    intro m x hx
    rcases List.mem_cons.mp hx with h | h
    · rw [h]
      simp only [maxOf]
      have := maxOf_ge_init bs (max m b)
      omega
    · simp only [maxOf]
      exact ih _ x h

-- § GGUF offset slots

/-- Slot of offset `k` (0 = gate, 1 = up, 2 = down) of layer `l` in the `3 * n_layers` array. -/
def gguf_slot (l k : Nat) : Nat := 3 * l + k

/-- Every slot of every valid layer is inside the array. -/
theorem gguf_slot_lt (n_layers l k : Nat) (hl : l < n_layers) (hk : k < 3) :
    gguf_slot l k < 3 * n_layers := by
  unfold gguf_slot
  omega

/-- Different (layer, kind) pairs use different slots. -/
theorem gguf_slot_injective (l1 k1 l2 k2 : Nat) (h1 : k1 < 3) (h2 : k2 < 3)
    (h : gguf_slot l1 k1 = gguf_slot l2 k2) : l1 = l2 ∧ k1 = k2 := by
  unfold gguf_slot at h
  omega

-- § AVX-512 gating

/-- Whether bit `k` of a register value is set. -/
def bitSet (r k : Nat) : Bool := (r >>> k) &&& 1 == 1

/-- The OS-state gate: the SSE, AVX, opmask, ZMM_Hi256 and Hi16_ZMM bits (mask 0xE6) are all enabled. -/
def xcr0Ok (xcr0 : Nat) : Bool := (xcr0 &&& 0xE6) == 0xE6

/-- The ISA gate from CPUID leaf 7: F, BW, VL in EBX; VNNI, VBMI in ECX. -/
def avx512Ok (ebx ecx : Nat) : Bool :=
  bitSet ebx 16 && bitSet ebx 30 && bitSet ebx 31 && bitSet ecx 11 && bitSet ecx 1

/-- A typical AVX-512 OS state (x87+SSE+AVX+opmask+ZMM) passes the gate. -/
theorem xcr0_full_ok : xcr0Ok 0xE7 = true := by
  decide

/-- An OS that only saves AVX (xcr0 = 7) fails the gate even if the CPU has AVX-512. -/
theorem xcr0_avx_only_rejected : xcr0Ok 0x07 = false := by
  decide

/-- A CPU with exactly the five required bits passes. -/
theorem avx512_all_present : avx512Ok (65536 + 1073741824 + 2147483648) (2048 + 2) = true := by
  decide

/-- A CPU missing VNNI (ECX bit 11) is rejected. -/
theorem avx512_missing_vnni : avx512Ok (65536 + 1073741824 + 2147483648) 2 = false := by
  decide

/-- A CPU missing VBMI (ECX bit 1) is rejected. -/
theorem avx512_missing_vbmi : avx512Ok (65536 + 1073741824 + 2147483648) 2048 = false := by
  decide

/-- A CPU that reports nothing is rejected. -/
theorem avx512_none : avx512Ok 0 0 = false := by
  decide

-- § default layout size

/-- Size of the default (no native table) layout. -/
def defaultTotal (n_layers n_expert blob : Nat) : Nat := n_layers * n_expert * blob

/-- With the 1,382,400-byte blob, 512 experts and one layer the default layout is 707,788,800 bytes. -/
theorem default_one_layer : defaultTotal 1 512 1382400 = 707788800 := by
  decide

/-- The default layout equals the contiguous total of identical layers (one layer check). -/
theorem default_matches_contig (n_expert blob : Nat) :
    contig n_expert [(0, blob)] 0 = some (defaultTotal 1 n_expert blob) := by
  rw [contig_single]
  congr 1
  unfold defaultTotal
  rw [Nat.one_mul, Nat.zero_add, Nat.mul_comm]

end Strata.Kernels.Cpu.ExpertLayoutTable

/-!
# MODULE: cpu_expert_layout
SOURCES: src/kernels/cpu/expert_layout.cpp
The per-layer expert table: contiguity validation of `native_experts.txt`, total size, maximum blob size,
GGUF offset slots, and the AVX-512 / AVX2 dispatch (including the OS-saved-state check on XCR0).

## Source semantics
* Layer `l` must start at the sum of the previous layers' `blob * n_expert`; the total is that sum over all
  layers. Without a table every layer has the fixed `BLOB` size.
* `max_blob` is the maximum layer blob size.
* GGUF offsets live in a `3*n_layers` array: slots `3l`, `3l+1`, `3l+2`.
* AVX-512 is used only if not forced off and the CPU and OS both support it.

## Form template identifiers
* `offs`, `endOff`, `check` : contiguous_layer_offsets, layout_total, contiguity_check
* `maxBlob`                 : max_blob
* `gguf slot`               : gguf_offset_slot
* `avx512Ok`, `pickImpl`    : avx512_dispatch
-/

namespace Strata.CpuExpertLayout

-- § contiguous offsets

/-- Expected start of each layer given layer blob sizes, `n_expert = E`, running offset `a`. -/
def offs : List Nat → Nat → Nat → List Nat
  | [], _, _ => []
  | b :: bs, E, a => a :: offs bs E (a + b * E)

/-- The end offset after all layers. -/
def endOff : List Nat → Nat → Nat → Nat
  | [], _, a => a
  | b :: bs, E, a => endOff bs E (a + b * E)

/-- The contiguity check of the loader: every declared offset equals the expected running sum. -/
def check : List Nat → List Nat → Nat → Nat → Bool
  | d :: ds, b :: bs, E, a => (d == a) && check ds bs E (a + b * E)
  | [], [], _, _ => true
  | [], _ :: _, _, _ => false
  | _ :: _, [], _, _ => false

/-- One expected offset per layer. -/
theorem offs_length (bs : List Nat) (E a : Nat) : (offs bs E a).length = bs.length := by
  induction bs generalizing a with
  | nil => rfl
  | cons b bs ih => simp [offs, ih]

/-- The total is the start plus every layer's `blob * n_expert`. -/
theorem endOff_eq (bs : List Nat) (E a : Nat) : endOff bs E a = a + (bs.map (· * E)).sum := by
  induction bs generalizing a with
  | nil => simp [endOff]
  | cons b bs ih =>
    simp only [endOff, ih, List.map_cons, List.sum_cons]
    omega

/-- The expected offsets always pass the contiguity check. -/
theorem check_offs (bs : List Nat) (E a : Nat) : check (offs bs E a) bs E a = true := by
  induction bs generalizing a with
  | nil => rfl
  | cons b bs ih => simp [offs, check, ih]

/-- Without a table the total is `layers * (BLOB * experts)`, one fixed blob per layer. -/
theorem total_uniform (L B E a : Nat) : endOff (List.replicate L B) E a = a + L * (B * E) := by
  induction L generalizing a with
  | zero => simp [endOff]
  | succ L ih =>
    simp only [List.replicate_succ, endOff, ih]
    rw [Nat.add_mul, Nat.one_mul]
    omega

-- § maximum blob

/-- Largest layer blob. -/
def maxBlob (bs : List Nat) : Nat := bs.foldl max 0

/-- The running maximum never drops below its starting value. -/
theorem foldl_max_ge_init (bs : List Nat) : ∀ init, init ≤ bs.foldl max init := by
  induction bs with
  | nil => intro init; exact Nat.le_refl init
  | cons x xs ih =>
    intro init
    exact Nat.le_trans (Nat.le_max_left init x) (ih (max init x))

/-- Every element is at most the fold maximum. -/
theorem le_foldl_max_of_mem (bs : List Nat) : ∀ init b, b ∈ bs → b ≤ bs.foldl max init := by
  induction bs with
  | nil => intro init b hb; simp at hb
  | cons x xs ih =>
    intro init b hb
    rcases List.mem_cons.mp hb with h | h
    · rw [h]
      exact Nat.le_trans (Nat.le_max_right init x) (foldl_max_ge_init xs (max init x))
    · exact ih (max init x) b h

/-- Every layer blob is at most `max_blob`. -/
theorem le_maxBlob (bs : List Nat) (b : Nat) (hb : b ∈ bs) : b ≤ maxBlob bs :=
  le_foldl_max_of_mem bs 0 b hb

-- § GGUF offset slots

/-- Slot of layer `l`, component `k` (0 gate, 1 up, 2 down) in the `3*n_layers` array. -/
def slot (l k : Nat) : Nat := 3 * l + k

/-- Slots of in-range layers stay inside the array. -/
theorem slot_lt (l k n : Nat) (hl : l < n) (hk : k < 3) : slot l k < 3 * n := by
  unfold slot; omega

/-- Distinct (layer, component) pairs use distinct slots. -/
theorem slot_inj (l k l' k' : Nat) (hk : k < 3) (hk' : k' < 3) (h : slot l k = slot l' k') :
    l = l' ∧ k = k' := by
  unfold slot at h
  refine ⟨?_, ?_⟩ <;> omega

-- § feature dispatch

/-- Which implementation the row dispatcher picks. -/
inductive Impl where
  | avx512
  | avx2
  deriving DecidableEq

/-- AVX-512 is allowed only when not forced off and the hardware check passed. -/
def avx512Ok (forcedAvx2 hw : Bool) : Bool := !forcedAvx2 && hw

/-- Dispatch on the result of the check. -/
def pickImpl (ok : Bool) : Impl := if ok then Impl.avx512 else Impl.avx2

/-- `STRATA_FORCE_AVX2=1` always selects the AVX2 implementation. -/
theorem forced_selects_avx2 (hw : Bool) : pickImpl (avx512Ok true hw) = Impl.avx2 := by
  simp [avx512Ok, pickImpl]

/-- Without the force flag the hardware answer decides. -/
theorem unforced_follows_hw (hw : Bool) : pickImpl (avx512Ok false hw) = (if hw then Impl.avx512 else Impl.avx2) := by
  simp [avx512Ok, pickImpl]

set_option maxRecDepth 100000 in
/-- The XCR0 test `(xcr0 & 0xE6) == 0xE6` over the low byte is exactly the five state bits
(SSE/AVX/opmask/ZMM_Hi256/Hi16_ZMM that the OS must save). -/
theorem xcr0_mask_iff : ∀ x : Fin 256, ((x.val &&& 0xE6) = 0xE6) ↔
    ((x.val / 2) % 2 = 1 ∧ (x.val / 4) % 2 = 1 ∧ (x.val / 32) % 2 = 1 ∧
     (x.val / 64) % 2 = 1 ∧ (x.val / 128) % 2 = 1) := by
  decide

end Strata.CpuExpertLayout

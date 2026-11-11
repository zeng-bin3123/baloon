/-!
# MODULE: native_expert_geometry
SOURCES: src/kernels/cpu/native_expert.cpp
Native (GGUF-form) experts on the CPU through ggml-cpu: the `NativeFmt` blob geometry derived from
the gate/up and down types, its validation, and the dispatch between the AVX-512 i-quant kernels
and ggml-cpu's `vec_dot`.

## Source semantics
* A blob is [gate rows | up rows | down rows]: gate rows at r*gu_row, up at up_off + r*gu_row
  (up_off = gu_row * n_ff), down at down_off + r*d_row (down_off = 2*up_off),
  bytes = down_off + d_row * n_embd.  The three regions tile the blob.
* Formats are rejected unless n_embd and n_ff are whole blocks of both the weight types and their
  activation (`vec_dot_type`) types, and unless the quantized activations fit the pool buffers.
* A Q2_0 (type 42) gate/up/down native pack has exactly the 1,382,400-byte size of a Strata blob.
* Gate/up use the AVX-512 kernels only for >= 2 tokens and supported types; one token always takes
  ggml-cpu.  The output is SwiGLU: silu(gate) * up.
* The type-error message names the failing type: gate/up's when its traits are missing, else down's.

## Form template identifiers
native_fmt -> mkFmt, NativeFmt.{gu_row,d_row,up_off,down_off,bytes} -> fields of Fmt,
native_gu_rows dispatch -> guPath, ggml_row_size -> rowSize
-/

namespace Strata.NativeExpert

-- § geometry

/-- `ggml_row_size`: whole blocks times block bytes. -/
def rowSize (blck typeBytes n : Nat) : Nat := n / blck * typeBytes

/-- The derived layout of a native expert blob. -/
structure Fmt where
  guRow : Nat
  dRow : Nat
  nEmbd : Nat
  nFF : Nat

/-- Offset of the up rows. -/
def Fmt.upOff (f : Fmt) : Nat := f.guRow * f.nFF
/-- Offset of the down rows. -/
def Fmt.downOff (f : Fmt) : Nat := 2 * f.upOff
/-- Total blob bytes. -/
def Fmt.bytes (f : Fmt) : Nat := f.downOff + f.dRow * f.nEmbd

/-- Closed form of the blob size: two gate/up matrices plus one down matrix. -/
theorem bytes_closed_form (f : Fmt) : f.bytes = 2 * (f.guRow * f.nFF) + f.dRow * f.nEmbd := by
  simp [Fmt.bytes, Fmt.downOff, Fmt.upOff]

/-- Gate row `r < n_ff` lies wholly before the up region. -/
theorem gate_row_in_region (f : Fmt) (r : Nat) (hr : r < f.nFF) :
    r * f.guRow + f.guRow ≤ f.upOff := by
  unfold Fmt.upOff
  rw [Nat.mul_comm f.guRow f.nFF, ← Nat.succ_mul]
  exact Nat.mul_le_mul_right _ hr

/-- Up row `r < n_ff` lies within `[up_off, down_off)`. -/
theorem up_row_in_region (f : Fmt) (r : Nat) (hr : r < f.nFF) :
    f.upOff ≤ f.upOff + r * f.guRow ∧ f.upOff + r * f.guRow + f.guRow ≤ f.downOff := by
  have := gate_row_in_region f r hr
  unfold Fmt.downOff
  omega

/-- Down row `r < n_embd` lies within `[down_off, bytes)`. -/
theorem down_row_in_region (f : Fmt) (r : Nat) (hr : r < f.nEmbd) :
    f.downOff + r * f.dRow + f.dRow ≤ f.bytes := by
  unfold Fmt.bytes
  have : r * f.dRow + f.dRow ≤ f.dRow * f.nEmbd := by
    rw [Nat.mul_comm f.dRow f.nEmbd, ← Nat.succ_mul]
    exact Nat.mul_le_mul_right _ hr
  omega

-- § validation

/-- `native_fmt`'s acceptance test (types have dot products and quantizers are assumed checked). -/
def geometryOk (nEmbd nFF guBlk dBlk guActBlk dActBlk actBytes hBytes actCap hCap : Nat) : Bool :=
  nEmbd % guBlk == 0 && nFF % dBlk == 0 && nEmbd % guActBlk == 0 && nFF % dActBlk == 0 &&
    decide (actBytes ≤ actCap) && decide (hBytes ≤ hCap)

/-- An accepted format has whole-block rows and activations that fit the pool's buffers. -/
theorem geometryOk_sound (a b c d e g h i j k : Nat) (hok : geometryOk a b c d e g h i j k = true) :
    a % c = 0 ∧ b % d = 0 ∧ a % e = 0 ∧ b % g = 0 ∧ h ≤ j ∧ i ≤ k := by
  unfold geometryOk at hok
  simp [Bool.and_eq_true] at hok
  omega

/-- Whole blocks make the row size exact: `rowSize * blck = n * typeBytes`. -/
theorem rowSize_exact (blck tb n : Nat) (h : n % blck = 0) : rowSize blck tb n * blck = n * tb := by
  unfold rowSize
  have hd := Nat.div_add_mod n blck
  rw [h, Nat.add_zero] at hd
  rw [Nat.mul_right_comm, Nat.mul_comm (n / blck) blck, hd]

/-- A Q2_0 native pack (64-element, 18-byte blocks) at H = 2560, FF = 640: gate/up rows of 720 bytes,
down rows of 180 bytes, blob of 1,382,400 bytes — the Strata blob size. -/
theorem q2_native_pack_matches_blob :
    let f : Fmt := ⟨rowSize 64 18 2560, rowSize 64 18 640, 2560, 640⟩
    f.guRow = 720 ∧ f.dRow = 180 ∧ f.bytes = 1382400 := by decide

-- § dispatch

/-- Which gate/up path `native_gu_rows` takes. -/
def guPath (avx512 : Bool) (nt : Nat) (supported : Bool) : String :=
  if avx512 && decide (nt ≥ 2) && supported then "iq512" else "ggml"

/-- A single token always takes ggml-cpu (the AVX-512 kernels are no faster there). -/
theorem one_token_uses_ggml (a s : Bool) : guPath a 1 s = "ggml" := by
  simp [guPath]

/-- The AVX-512 path requires all three conditions. -/
theorem iq512_requires_all (a s : Bool) (nt : Nat) (h : guPath a nt s = "iq512") :
    a = true ∧ nt ≥ 2 ∧ s = true := by
  unfold guPath at h
  split at h
  · rename_i hc; simp [Bool.and_eq_true] at hc; exact ⟨hc.1.1, hc.1.2, hc.2⟩
  · simp at h

/-- The type reported by the "no dot product" error: `tg && tg->vec_dot ? d_type : gu_type`. -/
def reportedType (guOk : Bool) (guType dType : Nat) : Nat := if guOk then dType else guType

/-- The error names the type that actually lacks a dot product. -/
theorem reportedType_correct (guOk dOk : Bool) (g d : Nat) (hfail : ¬ (guOk = true ∧ dOk = true)) :
    (guOk = false → reportedType guOk g d = g) ∧ (guOk = true → dOk = false ∧ reportedType guOk g d = d) := by
  unfold reportedType
  cases guOk <;> cases dOk <;> simp_all

end Strata.NativeExpert

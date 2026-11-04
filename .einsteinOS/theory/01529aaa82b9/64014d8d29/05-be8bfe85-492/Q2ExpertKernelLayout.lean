/-!
# MODULE: q2_expert_kernel_layout
SOURCES: include/strata/kernels/cpu/expert.hpp, src/kernels/cpu/expert.cpp
The CPU Q2_0 routed-expert kernel: fixed geometry, the 1,382,400-byte expert blob layout, the
activation-quantizer clamp, tail-mask arithmetic of the 512-bit row dot, the Q2_0 offset identity and
the CPU-feature gate.

## Source semantics
* One Q2_0 weight block is QK=64 weights with one fp16 scale; one activation chunk is QKA=32 elements with
  its own scale, so a weight block spans exactly two chunks.
* The blob is: gate/up codes | down codes | gate/up scales | down scales, tiling BLOB exactly.
* `sum (c-1) xhat = sum c xhat - sum xhat` (the code offset is removed through `hx`).
* The multi-token dispatch uses widths 1..8 and falls back to single-token for the oracle contract.

## Form template identifiers
* H, FF, NE, QK, QKA, BB  -> hidden_width, expert_ff_width, n_experts, weight_block, act_chunk, block_bytes
* O_GU_CODES .. O_D_SCALES -> blob section offsets
* ActQ -> quantized_activation, CpuFeatures -> cpu_feature_set
-/

namespace Strata.Q2Expert

-- § geometry (literals, with proofs that they equal the source formulas)
def H : Nat := 2560
def FF : Nat := 640
def NE : Nat := 512
def QK : Nat := 64
def QKA : Nat := 32
def BB : Nat := 18
def MAXC : Nat := 80
def ROW_GU : Nat := 640
def ROW_D : Nat := 160
def SC_GU : Nat := 40
def SC_D : Nat := 10
def BLOB : Nat := 1382400
def O_GU_CODES : Nat := 0
def O_D_CODES : Nat := 819200
def O_GU_SCALES : Nat := 1228800
def O_D_SCALES : Nat := 1331200
def MAXT : Nat := 8

/-- The derived constants equal the formulas written in expert.hpp. -/
theorem geometry_formulas :
    MAXC = H / QKA ∧ ROW_GU = H * 2 / 8 ∧ ROW_D = FF * 2 / 8 ∧ SC_GU = H / QK ∧ SC_D = FF / QK ∧
    BLOB = 3 * (H * FF * BB / QK) := by
  decide

/-- The blob section offsets equal the formulas written in expert.hpp. -/
theorem offset_formulas :
    O_D_CODES = 2 * FF * ROW_GU ∧ O_GU_SCALES = O_D_CODES + 1 * H * ROW_D ∧
    O_D_SCALES = O_GU_SCALES + 2 * FF * SC_GU * 2 := by
  decide

/-- One 64-weight block spans exactly two 32-element activation chunks. -/
theorem weight_block_spans_two_chunks : QK = 2 * QKA := by decide

/-- A Q2_0 block is a 2-byte fp16 scale plus 16 bytes of 2-bit codes. -/
theorem block_bytes_split : BB = 2 + QK / 4 := by decide

/-- Every row width is a whole number of weight blocks. -/
theorem rows_are_whole_blocks : H = QK * SC_GU ∧ FF = QK * SC_D := by decide

/-- The four sections tile the blob exactly. -/
theorem blob_tiles : O_D_SCALES + H * SC_D * 2 = BLOB := by decide

/-- Sections are laid out back to back. -/
theorem sections_contiguous :
    O_GU_CODES + 2 * FF * ROW_GU = O_D_CODES ∧ O_D_CODES + H * ROW_D = O_GU_SCALES ∧
    O_GU_SCALES + 2 * FF * SC_GU * 2 = O_D_SCALES := by
  decide

/-- Gate and up rows are interleaved: row `2r` and `2r+1` of the code plane stay inside the plane. -/
theorem gu_code_rows_in_bounds (r : Nat) (hr : r < FF) :
    O_GU_CODES + (2 * r + 1) * ROW_GU + ROW_GU ≤ O_D_CODES := by
  simp only [O_GU_CODES, ROW_GU, O_D_CODES, FF] at *
  omega

/-- Down code rows stay inside the down-code plane. -/
theorem down_code_rows_in_bounds (r : Nat) (hr : r < H) :
    O_D_CODES + r * ROW_D + ROW_D ≤ O_GU_SCALES := by
  simp only [ROW_D, O_D_CODES, O_GU_SCALES, H] at *
  omega

/-- Gate/up scale rows stay inside their plane. -/
theorem gu_scale_rows_in_bounds (r : Nat) (hr : r < FF) :
    O_GU_SCALES + (2 * r + 1) * SC_GU * 2 + SC_GU * 2 ≤ O_D_SCALES := by
  simp only [SC_GU, O_GU_SCALES, O_D_SCALES, FF] at *
  omega

/-- Down scale rows stay inside the blob. -/
theorem down_scale_rows_in_bounds (r : Nat) (hr : r < H) :
    O_D_SCALES + r * SC_D * 2 + SC_D * 2 ≤ BLOB := by
  simp only [SC_D, O_D_SCALES, BLOB, H] at *
  omega

-- § activation quantizer
/-- Clamp used by `act_quant_q8_1` after rounding. -/
def clamp127 (v : Int) : Int := max (-127) (min v 127)

/-- The clamp lands in [-127, 127]. -/
theorem clamp127_range (v : Int) : -127 ≤ clamp127 v ∧ clamp127 v ≤ 127 := by
  unfold clamp127
  omega

/-- The clamp is the identity on in-range values. -/
theorem clamp127_idem (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clamp127 v = v := by
  unfold clamp127
  omega

/-- Activation chunk counts: 2560 -> 80 (= MAXC), 640 -> 20. -/
theorem chunk_counts : H / QKA = MAXC ∧ FF / QKA = 20 := by decide

-- § the Q2_0 code offset identity
/-- Offset identity: summing `(c-1)*x` equals summing `c*x` minus summing `x`. -/
theorem offset_identity :
    ∀ (cs xs : List Int), cs.length = xs.length →
      (List.zipWith (fun c x => (c - 1) * x) cs xs).sum =
        (List.zipWith (fun c x => c * x) cs xs).sum - xs.sum
  | [], [], _ => by simp
  | c :: cs, x :: xs, h => by
    have h' : cs.length = xs.length := by simpa using h
    have ih := offset_identity cs xs h'
    simp only [List.zipWith_cons_cons, List.sum_cons]
    rw [ih, Int.sub_mul, Int.one_mul]
    generalize c * x = p
    omega
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h

-- § 512-bit tail masks
/-- A mask of `2*nb` bits for `nb ≤ 8` blocks fits a 16-bit lane mask. -/
theorem tail_mask_fits : ∀ nb : Nat, nb < 9 → (1 <<< (2 * nb)) - 1 < 65536 := by
  decide

/-- Gate/up rows (40 blocks) are an exact multiple of the 8-block batch; down rows (10 blocks) leave a 2-block tail. -/
theorem block_batches : SC_GU % 8 = 0 ∧ SC_D % 8 = 2 ∧ (SC_GU + 7) / 8 = 5 ∧ (SC_D + 7) / 8 = 2 := by
  decide

-- § multi-token dispatch
/-- Width chosen by the `switch (n_tokens)` ladder. -/
def dispatchNT (n : Nat) : Nat := if 1 ≤ n ∧ n ≤ 7 then n else 8

/-- For every legal token count 1..MAXT the dispatched width is exactly the token count. -/
theorem dispatch_exact (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ MAXT) : dispatchNT n = n := by
  simp only [dispatchNT, MAXT] at *
  split <;> omega

-- § CPU feature gate
/-- The five AVX-512 feature bits the kernel needs. -/
structure CpuFeatures where
  avx512f : Bool
  avx512bw : Bool
  avx512vl : Bool
  avx512_vnni : Bool
  avx512_vbmi : Bool

/-- `usable()` is the conjunction of all five bits. -/
def CpuFeatures.usable (f : CpuFeatures) : Bool :=
  f.avx512f && f.avx512bw && f.avx512vl && f.avx512_vnni && f.avx512_vbmi

/-- The machine is accepted exactly when every required feature is present. -/
theorem usable_iff (f : CpuFeatures) :
    f.usable = true ↔ (f.avx512f = true ∧ f.avx512bw = true ∧ f.avx512vl = true ∧
      f.avx512_vnni = true ∧ f.avx512_vbmi = true) := by
  unfold CpuFeatures.usable
  simp only [Bool.and_eq_true, and_assoc]

/-- A CPU without VNNI is refused even with every other bit set. -/
theorem no_vnni_refused (f : CpuFeatures) (h : f.avx512_vnni = false) : f.usable = false := by
  unfold CpuFeatures.usable
  simp [h]

end Strata.Q2Expert

/-!
# MODULE: q2_expert_vnni_kernel
The CPU Q2_0 expert kernel: Q8_1-style activation quantisation, 2-bit code unpacking, the offset-corrected
row dot `sum (c-1) * xhat = sum c*xhat - sum xhat`, the gate/up/down row layout of one expert blob, the
tail-mask / multi-token dispatch arithmetic and the CPU feature gate.

## Source semantics
* A weight code `c ∈ {0,1,2,3}` decodes to `(c - 1) * d`; the kernel computes `sum c*xhat` with VNNI and
  subtracts `sum xhat` (carried per 32-chunk as `hx`), so the offset identity must be exact over integers.
* Activation values are clamped to `[-127, 127]`, so integer partial sums cannot overflow.
* One expert blob is exactly 1,382,400 bytes: gate/up rows interleaved (row `2r` gate, `2r+1` up), then down.
* The tail of a block loop uses masks over `nb ≤ 8` blocks; multi-token dispatch handles 1..8 tokens.

## Form template identifiers
H -> hidden_size, FF -> ffn_size, QK -> weight_block, QKA -> act_chunk, row_dot -> row_dot,
act_quant_q8_1 -> act_quant, s2_expert_vnni_q -> expert_forward, CpuFeatures -> cpu_features.
-/

namespace Strata.Q2Expert

-- § geometry

/-- Hidden size of the model, the width of an expert input and output. -/
def H : Nat := 2560
/-- Expert intermediate width. -/
def FF : Nat := 640
/-- Weights per Q2_0 block. -/
def QK : Nat := 64
/-- Activations per quantisation chunk (the unit of the Q8_1 scale). -/
def QKA : Nat := 32
/-- Q2_0 blocks per gate/up row. -/
def SC_GU : Nat := H / QK
/-- Q2_0 blocks per down row. -/
def SC_D : Nat := FF / QK
/-- Code bytes per gate/up row. -/
def ROW_GU : Nat := SC_GU * 16
/-- Code bytes per down row. -/
def ROW_D : Nat := SC_D * 16
/-- Total bytes of one expert blob: codes and scales of gate/up (2*FF rows) plus down (H rows). -/
def BLOB : Nat := 2 * FF * ROW_GU + 2 * FF * SC_GU * 2 + H * ROW_D + H * SC_D * 2

/-- One expert blob is exactly 1,382,400 bytes. -/
theorem blob_bytes : BLOB = 1382400 := by decide

/-- The block counts per row are 40 (gate/up) and 10 (down). -/
theorem block_counts : SC_GU = 40 ∧ SC_D = 10 := by decide

/-- The activation chunk counts are 80 for the input and 20 for the intermediate. -/
theorem chunk_counts : H / QKA = 80 ∧ FF / QKA = 20 := by decide

/-- Each 64-weight block covers exactly two 32-activation chunks. -/
theorem two_chunks_per_block : QK = 2 * QKA := by decide

/-- Both input widths are whole numbers of weight blocks. -/
theorem widths_divisible : H % QK = 0 ∧ FF % QK = 0 := by decide

/-- A block holds 16 code bytes at four codes per byte, which is 64 codes. -/
theorem codes_per_block : 16 * 4 = QK := by decide

/-- Gate and up rows are interleaved: slots `2r` and `2r+1` are distinct and stay below `2*FF`. -/
theorem gate_up_slots (r : Nat) (hr : r < FF) :
    2 * r ≠ 2 * r + 1 ∧ 2 * r + 1 < 2 * FF := by
  unfold FF at *
  omega

/-- Chunk indices `2b` and `2b+1` of block `b` stay inside the row's chunk range. -/
theorem chunk_index_in_row (b nb : Nat) (hb : b < nb) (h : Nat) (hh : h < 2) :
    2 * b + h < 2 * nb := by
  omega

-- § offset-corrected dot product

/-- The weight code offset is removed exactly: `sum (c-1)*x = sum c*x - sum x`. -/
theorem offset_identity : ∀ (c x : List Int), c.length = x.length →
    (List.zipWith (· * ·) (c.map (· - 1)) x).sum = (List.zipWith (· * ·) c x).sum - x.sum
  | [], [], _ => by simp
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | a :: cs, b :: xs, h => by
    have ih := offset_identity cs xs (by simpa using h)
    simp only [List.map_cons, List.zipWith_cons_cons, List.sum_cons]
    rw [ih, Int.sub_mul, Int.one_mul]
    omega

/-- Clamp an activation to the symmetric int8 range used by the quantiser. -/
def clamp127 (v : Int) : Int :=
  if v < -127 then -127 else if v > 127 then 127 else v

/-- The clamp lands in `[-127, 127]`. -/
theorem clamp127_bounds (v : Int) : -127 ≤ clamp127 v ∧ clamp127 v ≤ 127 := by
  unfold clamp127
  split <;> (try split) <;> omega

/-- Clamping is the identity on values already in range. -/
theorem clamp127_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clamp127 v = v := by
  unfold clamp127
  split <;> (try split) <;> omega

/-- A sum of in-range activations is bounded by `127 * length`; int32 accumulation cannot overflow. -/
theorem sum_bound : ∀ (l : List Int), (∀ v ∈ l, -127 ≤ v ∧ v ≤ 127) →
    -(127 * (l.length : Int)) ≤ l.sum ∧ l.sum ≤ 127 * (l.length : Int)
  | [], _ => by simp
  | a :: t, h => by
    have ha := h a (by simp)
    have ht := sum_bound t (fun v hv => h v (by simp [hv]))
    simp only [List.sum_cons, List.length_cons]
    omega

/-- A 64-weight block dot of codes (≤3) against int8 activations fits an int32 with room to spare. -/
theorem block_dot_fits : 3 * 127 * 64 < 2 ^ 31 := by decide

-- § 2-bit unpacking

/-- Code `k` (0..3) of a packed byte, least significant first. -/
def codeOf (byte k : Nat) : Nat := byte / 4 ^ k % 4

/-- Pack four 2-bit codes into a byte, first code in the low bits. -/
def pack4 (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3

/-- Unpacking recovers every packed code. -/
theorem unpack_pack4 (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    codeOf (pack4 c0 c1 c2 c3) 0 = c0 ∧ codeOf (pack4 c0 c1 c2 c3) 1 = c1 ∧
    codeOf (pack4 c0 c1 c2 c3) 2 = c2 ∧ codeOf (pack4 c0 c1 c2 c3) 3 = c3 := by
  simp only [codeOf, pack4]
  omega

/-- A packed byte of four codes is below 256. -/
theorem pack4_lt (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    pack4 c0 c1 c2 c3 < 256 := by
  unfold pack4
  omega

/-- Every unpacked code is a valid 2-bit code. -/
theorem codeOf_lt (byte k : Nat) : codeOf byte k < 4 := by
  unfold codeOf
  exact Nat.mod_lt _ (by decide)

/-- A decoded code `c - 1` lies in the codebook range `[-1, 2]`. -/
theorem decoded_code_range (c : Nat) (h : c < 4) : -1 ≤ (c : Int) - 1 ∧ (c : Int) - 1 ≤ 2 := by
  omega

/-- Weight `j` of a block lives in code byte `j/4` at bit pair `j%4`; the byte stays inside the block. -/
theorem scalar_code_index (b j : Nat) (hj : j < 64) :
    b * 16 + j / 4 < (b + 1) * 16 ∧ j % 4 < 4 := by
  omega

-- § tail masks and block loop

/-- Mask over the low `nb` blocks of an eight-block group. -/
def tailMask (nb : Nat) : Nat := if nb ≥ 8 then 0xFF else 2 ^ nb - 1

/-- Masks over at most eight blocks fit a byte. -/
theorem tailMask_lt (nb : Nat) (h : nb ≤ 8) : tailMask nb < 256 := by
  unfold tailMask
  interval_cases nb <;> decide

/-- Blocks processed in one iteration of the eight-block loop starting at `b0`. -/
def groupSize (nblocks b0 : Nat) : Nat := if nblocks - b0 < 8 then nblocks - b0 else 8

/-- Each iteration processes between one and eight blocks and never runs past the row. -/
theorem group_bounds (nblocks b0 : Nat) (h : b0 < nblocks) :
    1 ≤ groupSize nblocks b0 ∧ groupSize nblocks b0 ≤ 8 ∧ b0 + groupSize nblocks b0 ≤ nblocks := by
  unfold groupSize
  split <;> omega

-- § token dispatch

/-- Template instance selected for `n` tokens: 1..7 exact, anything else the 8-token instance. -/
def dispatchNT (n : Nat) : Nat := if n ≥ 8 then 8 else n

/-- For 1 to 8 tokens the instance matches the token count. -/
theorem dispatchNT_exact (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ 8) : dispatchNT n = n := by
  unfold dispatchNT
  split <;> omega

/-- The multi-token path falls back to per-token execution outside 1..8 tokens. -/
def fallsBack (n : Nat) : Bool := decide (n < 1 ∨ n > 8)

/-- The multi-token fast path is taken exactly for 1..8 tokens. -/
theorem fast_path_range (n : Nat) : fallsBack n = false ↔ 1 ≤ n ∧ n ≤ 8 := by
  unfold fallsBack
  simp only [decide_eq_false_iff_not]
  omega

-- § cpu feature gate

/-- The CPU features the expert kernel needs. -/
structure CpuFeatures where
  avx512f : Bool := false
  avx512bw : Bool := false
  avx512vl : Bool := false
  avx512_vnni : Bool := false
  avx512_vbmi : Bool := false

/-- The kernel is usable iff all five features are present. -/
def CpuFeatures.usable (f : CpuFeatures) : Bool :=
  f.avx512f && f.avx512bw && f.avx512vl && f.avx512_vnni && f.avx512_vbmi

/-- Usability is exactly the conjunction of the five flags. -/
theorem usable_iff (f : CpuFeatures) :
    f.usable = true ↔ (f.avx512f = true ∧ f.avx512bw = true ∧ f.avx512vl = true ∧
      f.avx512_vnni = true ∧ f.avx512_vbmi = true) := by
  simp [CpuFeatures.usable, and_assoc]

/-- Bit `i` of a cpuid register. -/
def bit (x i : Nat) : Bool := (x / 2 ^ i) % 2 == 1

/-- Feature detection: a CPU whose maximum leaf is below 7 reports no features. -/
def detect (maxLeaf ebx ecx : Nat) : CpuFeatures :=
  if maxLeaf < 7 then {}
  else { avx512f := bit ebx 16, avx512bw := bit ebx 30, avx512vl := bit ebx 31,
         avx512_vnni := bit ecx 11, avx512_vbmi := bit ecx 1 }

/-- A CPU without cpuid leaf 7 is never usable. -/
theorem detect_small_leaf (maxLeaf ebx ecx : Nat) (h : maxLeaf < 7) :
    (detect maxLeaf ebx ecx).usable = false := by
  simp [detect, h, CpuFeatures.usable]

/-- An all-zero register reports no feature bit. -/
theorem bit_zero (i : Nat) : bit 0 i = false := by
  simp [bit]

end Strata.Q2Expert

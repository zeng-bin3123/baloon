/-!
# MODULE: q2_row_dot_identity
SOURCES: src/kernels/cpu/expert.cpp, src/kernels/cpu/q2_avx2.cpp, include/strata/kernels/cpu/expert.hpp
The arithmetic of every Q2_0 row dot in the CPU path (`row_dot`, `row_dot_z`, `row_dot_oracle`,
`row_dot_multi`, `q2g_row_multi`, the AVX2 `row_multi`): 2-bit code unpacking, the offset identity
`Σ (c-1)·x = Σ c·x − Σ x`, the two-chunks-per-block scale contract with the precomputed `hx`
correction, and the per-token equivalence of the multi-token kernels.  Exact integer arithmetic stands
in for real arithmetic; FP32 evaluation order is the only thing the kernels change.

## Source semantics
* Code of weight i in a 16-byte block: byte i/4, bits 2·(i%4), masked with 3, so it lies in {0,1,2,3};
  the dequantized weight is (code − 1)·d ∈ {−d, 0, d, 2d}.
* Per chunk, `Σ (c−1)·d_w·x̂·d_x = d_w·(d_x·Σ c·x̂ − d_x·Σ x̂)`; the second term is `hx = d_x·sum`,
  weight-independent, precomputed once per layer.
* A 64-weight block uses TWO activation scales (chunks 2b, 2b+1); one scale per block is a real error.
* `chunk_dot_oracle` subtracts the integer sum before conversion; the result is the same exact value.
* Multi-token kernels unpack codes once and give each token exactly the single-token value.

## Form template identifiers
code → unpackCode, offDot → exactOffsetDot, cDot → codeDot, xSum → actSum, hx → chunkCorrection,
Chunk/Block → actChunk/weightBlock, exactRow → referenceRowDot, kernelAcc/kernelCorr → vnniAccumulator/
correctionAccumulator, oracleRow → oracleRowDot, rowMulti → multiTokenRowDot
-/

namespace Strata.Kernels.Cpu.Q2Dot

-- § 2-bit codes

/-- Code k (0..3) of a byte: `(byte >> 2k) & 3`. -/
def code (byte k : Nat) : Nat := (byte >>> (2 * k)) % 4

/-- Every code is a 2-bit value. -/
theorem code_lt_four (byte k : Nat) : code byte k < 4 := Nat.mod_lt _ (by decide)

/-- Pack four 2-bit codes into a byte, low code first (the GGUF/Strata order). -/
def pack4 (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3

/-- Unpacking inverts packing: code k of `pack4` is the k-th code. -/
theorem code_pack4 (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    code (pack4 c0 c1 c2 c3) 0 = c0 ∧ code (pack4 c0 c1 c2 c3) 1 = c1 ∧
      code (pack4 c0 c1 c2 c3) 2 = c2 ∧ code (pack4 c0 c1 c2 c3) 3 = c3 := by
  simp only [code, pack4, Nat.shiftRight_eq_div_pow]
  refine ⟨?_, ?_, ?_, ?_⟩ <;> simp <;> omega

/-- A packed byte is a byte. -/
theorem pack4_lt (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    pack4 c0 c1 c2 c3 < 256 := by
  simp only [pack4]; omega

/-- Weight i (< 64) of a block is in byte i/4 (< 16) at code position i%4 (< 4). -/
theorem unpack_position (i : Nat) (hi : i < 64) : i / 4 < 16 ∧ i % 4 < 4 ∧ 4 * (i / 4) + i % 4 = i := by
  omega

/-- The dequantized code offset `code − 1` lies in {−1,0,1,2}. -/
theorem code_offset_range (byte k : Nat) :
    -1 ≤ (code byte k : Int) - 1 ∧ (code byte k : Int) - 1 ≤ 2 := by
  have h := code_lt_four byte k
  omega

-- § the offset identity

/-- Exact reference: `Σ (c−1)·x` over (code, activation) pairs. -/
def offDot : List (Int × Int) → Int
  | [] => 0
  | (c, x) :: t => (c - 1) * x + offDot t

/-- The VNNI integer dot `Σ c·x`. -/
def cDot : List (Int × Int) → Int
  | [] => 0
  | (c, x) :: t => c * x + cDot t

/-- The activation sum `Σ x` (the chunk's `sum`). -/
def xSum : List (Int × Int) → Int
  | [] => 0
  | (_, x) :: t => x + xSum t

/-- The Q2_0 identity: the −1 code offset is a weight-independent correction. -/
theorem offDot_eq (l : List (Int × Int)) : offDot l = cDot l - xSum l := by
  induction l with
  | nil => rfl
  | cons p t ih =>
    obtain ⟨c, x⟩ := p
    simp only [offDot, cDot, xSum]
    rw [ih, Int.sub_mul, Int.one_mul]
    generalize c * x = cx
    omega

-- § the two-chunk scale contract

/-- One 32-element activation chunk with its own scale `s` (exact, as an integer stand-in). -/
structure Chunk where
  s : Int
  pairs : List (Int × Int)

/-- `hx = scale · sum`, precomputed once per layer. -/
def hx (ch : Chunk) : Int := ch.s * xSum ch.pairs

/-- Exact chunk contribution `s · Σ (c−1)·x`. -/
def exactChunk (ch : Chunk) : Int := ch.s * offDot ch.pairs

/-- Per chunk: `s·Σ(c−1)x = s·Σ c·x − hx`. -/
theorem exactChunk_eq (ch : Chunk) : exactChunk ch = ch.s * cDot ch.pairs - hx ch := by
  unfold exactChunk hx
  rw [offDot_eq, Int.mul_sub]

/-- One 64-weight block: one weight scale `d`, two chunks with two scales. -/
structure Block where
  d : Int
  lo : Chunk
  hi : Chunk

/-- Reference block value `d · (lo + hi)` with each chunk on its own scale. -/
def exactBlock (b : Block) : Int := b.d * (exactChunk b.lo + exactChunk b.hi)

/-- Reference row: the sum over blocks. -/
def exactRow : List Block → Int
  | [] => 0
  | b :: t => exactBlock b + exactRow t

/-- The kernel's float accumulator: `Σ d·s_lo·cDot_lo + d·s_hi·cDot_hi`. -/
def kernelAcc : List Block → Int
  | [] => 0
  | b :: t => b.d * b.lo.s * cDot b.lo.pairs + b.d * b.hi.s * cDot b.hi.pairs + kernelAcc t

/-- The kernel's correction accumulator: `Σ d·(hx_lo + hx_hi)`. -/
def kernelCorr : List Block → Int
  | [] => 0
  | b :: t => b.d * (hx b.lo + hx b.hi) + kernelCorr t

/-- One block of the kernel equals the reference block. -/
theorem block_kernel_eq (b : Block) :
    exactBlock b = b.d * b.lo.s * cDot b.lo.pairs + b.d * b.hi.s * cDot b.hi.pairs -
      b.d * (hx b.lo + hx b.hi) := by
  unfold exactBlock
  rw [exactChunk_eq, exactChunk_eq, Int.mul_assoc, Int.mul_assoc]
  generalize b.lo.s * cDot b.lo.pairs = A
  generalize b.hi.s * cDot b.hi.pairs = B
  generalize hx b.lo = P
  generalize hx b.hi = Q
  rw [Int.mul_add, Int.mul_sub, Int.mul_sub, Int.mul_add]
  generalize b.d * A = X
  generalize b.d * B = Y
  generalize b.d * P = U
  generalize b.d * Q = V
  omega

/-- `row_dot` correctness: accumulator minus correction, reduced once, is the reference row. -/
theorem row_dot_correct (bs : List Block) : exactRow bs = kernelAcc bs - kernelCorr bs := by
  induction bs with
  | nil => rfl
  | cons b t ih =>
    simp only [exactRow, kernelAcc, kernelCorr]
    rw [ih, block_kernel_eq]
    omega

/-- `row_dot_oracle`: integer offset removed per chunk before scaling, then the per-block sum. -/
def oracleRow : List Block → Int
  | [] => 0
  | b :: t => b.d * (b.lo.s * (cDot b.lo.pairs - xSum b.lo.pairs) +
                     b.hi.s * (cDot b.hi.pairs - xSum b.hi.pairs)) + oracleRow t

/-- The oracle reduction computes the same exact value as the reference. -/
theorem oracle_row_eq (bs : List Block) : oracleRow bs = exactRow bs := by
  induction bs with
  | nil => rfl
  | cons b t ih =>
    simp only [oracleRow, exactRow, exactBlock, exactChunk]
    rw [ih, offDot_eq, offDot_eq]

/-- The collapsed (wrong) contract: one scale for the whole 64-weight block. -/
def oneScaleBlock (b : Block) : Int := b.d * b.lo.s * (offDot b.lo.pairs + offDot b.hi.pairs)

/-- Collapsing the two chunk scales is a real error, not a rounding difference. -/
theorem one_scale_is_wrong :
    ∃ b : Block, oneScaleBlock b ≠ exactBlock b :=
  ⟨{ d := 1, lo := { s := 1, pairs := [(2, 1)] }, hi := { s := 2, pairs := [(2, 1)] } }, by decide⟩

-- § multi-token kernels

/-- One shared code list against one token's activations. -/
def tokenDot (codes acts : List Int) : Int := cDot (codes.zip acts)

/-- `row_dot_multi`: codes unpacked once, every token gets its own dot. -/
def rowMulti (codes : List Int) (tokens : List (List Int)) : List Int :=
  tokens.map (tokenDot codes)

/-- Every token's result is exactly the single-token row dot. -/
theorem rowMulti_get (codes : List Int) (tokens : List (List Int)) (t : Nat)
    (ht : t < (rowMulti codes tokens).length) :
    (rowMulti codes tokens)[t] = tokenDot codes (tokens[t]'(by simpa [rowMulti] using ht)) := by
  simp [rowMulti]

/-- The multi-token kernel returns one value per token. -/
theorem rowMulti_length (codes : List Int) (tokens : List (List Int)) :
    (rowMulti codes tokens).length = tokens.length := by
  simp [rowMulti]

/-- Splitting the token set into groups (the AVX2 path takes ≤ 4 at a time) changes nothing. -/
theorem rowMulti_split (codes : List Int) (tokens : List (List Int)) (k : Nat) :
    rowMulti codes tokens = rowMulti codes (tokens.take k) ++ rowMulti codes (tokens.drop k) := by
  simp only [rowMulti]
  rw [← List.map_append, List.take_append_drop]

end Strata.Kernels.Cpu.Q2Dot

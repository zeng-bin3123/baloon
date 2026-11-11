/-!
# MODULE: bf16_fp32_mmvf_gemv
SOURCES: src/kernels/cuda/native_bf16.cu
The BF16-weight / FP32-activation GEMV adapted from llama.cpp's MMVF: one block per output row,
`BLOCK_SIZE ∈ {32, 64, …, 256}` threads striding over weight PAIRS, a warp xor-butterfly, and a
cross-warp pass through a 32-slot shared array.

## Source semantics
* Preconditions (else `invalid_argument`): `0 < n_in`, `n_in` even, `n_in ≤ INT_MAX`,
  `0 < n_out ≤ INT_MAX`; `x` 8-byte aligned (read as `float2`), `w` and `y` 4-byte aligned.
* `mmvf_block_size(n_in)`: start at `best = 32` with `⌈n_in/64⌉` iterations; for candidates
  `64, 96, …, 256` adopt a candidate iff its iteration count `⌈n_in/(2c)⌉` is STRICTLY smaller.  Hence
  the result minimises iterations and, among ties, is the smallest block.
* Pair `p` of a row is the 32-bit word holding elements `2p` (low half) and `2p+1` (high half).
* After the butterfly, lane `t` of warp `t/32` stores into `partials[t/32]`; `t/32 < 8 ≤ 32`.
* The xor partner of lane `l < off` (off a power of two ≤ 16) is `l + off`, so lane 0 of the
  butterfly follows the same halving tree as a shuffle-down reduction.

## Form template identifiers
* `iters` ↔ `(n_in + 2*candidate - 1) / (2*candidate)`
* `candidates`, `step`, `blockSize` ↔ the loop of `mmvf_block_size`
* `mmvfValid` ↔ the shape guard of `bf16_gemv_fp32_mmvf`
* `unpackLo`, `unpackHi` ↔ `(uint16_t) weight`, `(uint16_t) (weight >> 16)`
-/

namespace Strata.Mmvf

-- § block-size selection

/-- Iterations a block of `c` threads needs over `n` elements taken two at a time. -/
def iters (n c : Nat) : Nat := (n + 2 * c - 1) / (2 * c)

/-- The candidates tried after the initial 32. -/
def candidates : List Nat := [64, 96, 128, 160, 192, 224, 256]

/-- One iteration of the selection loop (strict improvement only). -/
def step (n best c : Nat) : Nat := if iters n c < iters n best then c else best

/-- `mmvf_block_size`. -/
def blockSize (n : Nat) : Nat := candidates.foldl (step n) 32

/-- The source's initial `best_iterations = (n_in + 63) / 64` is `iters n 32`. -/
theorem iters_32 (n : Nat) : iters n 32 = (n + 63) / 64 := by
  unfold iters; omega

/-- Loop invariant: the fold returns one of the values seen, no worse than the start and no worse
than any candidate it passed. -/
theorem foldl_step_spec (n : Nat) : ∀ (l : List Nat) (acc : Nat),
    (l.foldl (step n) acc = acc ∨ l.foldl (step n) acc ∈ l) ∧
    iters n (l.foldl (step n) acc) ≤ iters n acc ∧
    ∀ c ∈ l, iters n (l.foldl (step n) acc) ≤ iters n c
  | [], acc => by simp
  | c :: cs, acc => by
    have ih := foldl_step_spec n cs (step n acc c)
    obtain ⟨hmem, hle, hall⟩ := ih
    have hstep : iters n (step n acc c) ≤ iters n acc ∧ iters n (step n acc c) ≤ iters n c := by
      unfold step
      by_cases h : iters n c < iters n acc <;> simp [h] <;> omega
    simp only [List.foldl_cons, List.mem_cons, forall_eq_or_imp]
    refine ⟨?_, Nat.le_trans hle hstep.1, Nat.le_trans hle hstep.2, hall⟩
    rcases hmem with h | h
    · rw [h]; unfold step
      by_cases hc : iters n c < iters n acc <;> simp [hc]
    · exact Or.inr (Or.inr h)

/-- The selected block size is one of the eight instantiated template sizes. -/
theorem blockSize_mem (n : Nat) : blockSize n ∈ 32 :: candidates := by
  have h := (foldl_step_spec n candidates 32).1
  unfold blockSize
  rcases h with h | h
  · rw [h]; simp
  · exact List.mem_cons_of_mem _ h

/-- Hence it is a multiple of 32 in `[32, 256]` — always a whole number of warps, and always one of the
`STRATA_MMVF_CASE` arms, so the `switch` never falls through without a launch. -/
theorem blockSize_shape (n : Nat) :
    blockSize n % 32 = 0 ∧ 32 ≤ blockSize n ∧ blockSize n ≤ 256 := by
  have hall : ∀ c ∈ 32 :: candidates, c % 32 = 0 ∧ 32 ≤ c ∧ c ≤ 256 := by decide
  exact hall _ (blockSize_mem n)

/-- **Optimality**: no instantiated block size needs fewer iterations than the one selected. -/
theorem blockSize_optimal (n c : Nat) (hc : c ∈ 32 :: candidates) :
    iters n (blockSize n) ≤ iters n c := by
  obtain ⟨_, hle, hall⟩ := foldl_step_spec n candidates 32
  simp only [List.mem_cons] at hc
  rcases hc with rfl | hc
  · exact hle
  · exact hall c hc

/-- Concrete selections: `n_embd = 2560` gets 256 threads; the 640-wide expert axis gets 160
(160, 192, 224 and 256 all need 2 iterations, and the strict comparison keeps the first). -/
theorem blockSize_examples : blockSize 2560 = 256 ∧ blockSize 640 = 160 ∧ blockSize 2 = 32 := by
  decide

-- § preconditions and data layout

/-- `INT_MAX` on the target. -/
def INT_MAX : Nat := 2 ^ 31 - 1

/-- The shape guard of `bf16_gemv_fp32_mmvf`. -/
def mmvfValid (n_in n_out : Nat) : Prop :=
  0 < n_in ∧ n_in % 2 = 0 ∧ n_in ≤ INT_MAX ∧ 0 < n_out ∧ n_out ≤ INT_MAX

/-- Under the guard the `n_in / 2` pairs cover the row exactly — no element is dropped by the pair loop. -/
theorem pairs_cover (n_in n_out : Nat) (h : mmvfValid n_in n_out) : 2 * (n_in / 2) = n_in := by
  obtain ⟨_, heven, _⟩ := h
  omega

/-- An odd row would lose its last element to `n_in / 2`, which is why oddness is refused. -/
theorem odd_row_loses_element : 2 * (2561 / 2) ≠ 2561 := by decide

/-- Pair `p` holds elements `2p` and `2p+1`, and pairs partition the element indices. -/
theorem pair_elements (e : Nat) : e = 2 * (e / 2) + e % 2 ∧ e % 2 < 2 := by omega

/-- Low half of a little-endian 32-bit pair word: `(uint16_t) weight`. -/
def unpackLo (w : Nat) : Nat := w % 65536

/-- High half: `weight >> 16`. -/
def unpackHi (w : Nat) : Nat := w >>> 16

/-- The pair word `e₀ + 2^16·e₁` unpacks to element `2p` then `2p+1`, matching the order of the two
`__fmaf_rn` calls against `input.x` then `input.y`. -/
theorem unpack_pair (e₀ e₁ : Nat) (h₀ : e₀ < 65536) (h₁ : e₁ < 65536) :
    unpackLo (e₀ + 65536 * e₁) = e₀ ∧ unpackHi (e₀ + 65536 * e₁) = e₁ := by
  unfold unpackLo unpackHi
  rw [Nat.shiftRight_eq_div_pow]
  constructor <;> omega

/-- `float2` loads of `x` at pair `p` sit at byte `8p`, aligned whenever `x` is 8-byte aligned. -/
theorem float2_aligned (base p : Nat) (h : base % 8 = 0) : (base + 8 * p) % 8 = 0 := by omega

-- § cross-warp reduction

/-- The cross-warp slot `t / 32` is in range for every thread of the largest block (and the 32-slot
array covers the reduction's 32 lanes, the extra slots being the zeros written first). -/
theorem partials_index (t B : Nat) (hB : B ≤ 256) (ht : t < B) : t / 32 < 8 ∧ 8 ≤ 32 := by omega

/-- For the butterfly offsets, the xor partner of a lane below the offset is `lane + off`: lane 0's
path through `mmvf_warp_sum` is the halving tree. -/
theorem xor_partner : ∀ l : Fin 32, ∀ off ∈ [16, 8, 4, 2, 1], l.val < off → l.val ^^^ off = l.val + off := by
  decide

end Strata.Mmvf

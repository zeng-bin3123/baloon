/-!
# MODULE: s_gemv_and_expert_grouped
SOURCES: include/strata/kernels/s_gemv.hpp, include/strata/kernels/s2_expert_grouped.hpp
Canonical-form GEMV (one decode `cb[code] * scale + offset` for S2/S4/S8 codes, over fp16, Q8_K or Q8_0
activations) and the grouped GPU expert tier (one launch per projection, hit lists passed as device data).

## Source semantics
* A weight row is `n_in / (8 / code_bits)` code bytes and `n_in / group_elems` scales.
* Q8_K activation blocks are 292 bytes per 256 elements (`4 + 256 + 2*16`); Q8_0 blocks are 34 bytes per 32
  elements (`2 + 32`). `n_in = 640` is a multiple of 32 but not of 256, so Q8_K is impossible for it.
* `act_kind` cannot be derived from the other form attributes (IQ4_NL vs IQ4_XS differ in nothing else).
* Grouped expert evaluation: scratch is `n_hits * (2*FF*4 + (FF/32)*34)` bytes; four launches per layer
  instead of five per expert; hit `h` reads slot `slot_index[h]` of the arena and writes row `dst_index[h]`.
* `moe_hit_select` keeps the routed entries whose residency row is non-negative.

## Form template identifiers
`SForm` -> `quant_form`; `code_bits` -> `bits_per_code`; `group_elems` -> `group_size`; `act_kind` -> `act_format`;
`moe_hit_grouped_s2` -> `grouped_expert_eval`; `slot_index` -> `resident_slot`; `dst_index` -> `route_row`.
-/

namespace Strata.SGemv

-- § canonical form

/-- The canonical-form attributes the kernels receive (codebook: 0 affine, 1 IQ4_NL table). -/
structure SForm where
  codeBits : Nat
  codeBias : Int
  groupElems : Nat
  codebook : Nat
  hasOffset : Bool
  actKind : Nat
  deriving DecidableEq

/-- Well-formed forms use 2, 4 or 8 bit codes, a positive group and an activation kind of 0 or 1. -/
def SForm.WellFormed (f : SForm) : Prop :=
  (f.codeBits = 2 ∨ f.codeBits = 4 ∨ f.codeBits = 8) ∧ 0 < f.groupElems ∧ f.actKind ≤ 1

/-- Codes per byte. -/
def perByte (bits : Nat) : Nat := 8 / bits

/-- Two-, four- and eight-bit codes pack 4, 2 and 1 per byte. -/
theorem perByte_values : perByte 2 = 4 ∧ perByte 4 = 2 ∧ perByte 8 = 1 := by decide

/-- Code bytes in one weight row. -/
def rowCodeBytes (nIn bits : Nat) : Nat := nIn / perByte bits

/-- Scales in one weight row. -/
def rowScales (nIn group : Nat) : Nat := nIn / group

/-- The expert gate/up shape: 2560 inputs give 640 code bytes (S2) and 40 scales (group 64) per row. -/
theorem expert_row_sizes : rowCodeBytes 2560 2 = 640 ∧ rowScales 2560 64 = 40 := by decide

/-- The integer-domain decode `code + bias` lies in `[bias, bias + M)` for codes below `M = 2^bits`. -/
theorem decode_range (code M : Nat) (bias : Int) (h : code < M) :
    bias ≤ (code : Int) + bias ∧ (code : Int) + bias < bias + (M : Int) := by
  omega

/-- Q2_0 (bias -1) decodes 2-bit codes to -1..2. -/
theorem q2_range (code : Nat) (h : code < 4) : -1 ≤ (code : Int) + (-1) ∧ (code : Int) + (-1) ≤ 2 := by
  omega

/-- Q5_0 (bias -16) decodes 5-bit values to -16..15. -/
theorem q5_range (code : Nat) (h : code < 32) : -16 ≤ (code : Int) + (-16) ∧ (code : Int) + (-16) ≤ 15 := by
  omega

-- § activation formats

/-- Bytes of a Q8_K activation block: scale, 256 int8 quants, 16 int16 block sums. -/
theorem q8k_block_bytes : 4 + 256 + 2 * 16 = 292 := by decide

/-- Bytes of a Q8_0 block: fp16 scale plus 32 int8 quants. -/
theorem q8_0_block_bytes : 2 + 32 = 34 := by decide

/-- Q8_0 activation image size for `n` elements. -/
def q8_0Bytes (n : Nat) : Nat := n / 32 * 34

/-- The 2560-wide image used by the PLE and expert kernels is 2720 bytes; the 640-wide one is 680. -/
theorem q8_0_sizes : q8_0Bytes 2560 = 2720 ∧ q8_0Bytes 640 = 680 := by decide

/-- `ffn_down_shexp` has `n_in = 640`: a multiple of 32 but not of 256, so Q8_K cannot apply. -/
theorem shexp_needs_q8_0 : 640 % 32 = 0 ∧ 640 % 256 ≠ 0 := by decide

/-- Group sizes 32, 64 and 256 are multiples of four and powers of two, as the split kernels require. -/
theorem groups_ok : (32 % 4 = 0 ∧ ∃ k, 32 = 2 ^ k) ∧ (64 % 4 = 0 ∧ ∃ k, 64 = 2 ^ k) ∧
    (256 % 4 = 0 ∧ ∃ k, 256 = 2 ^ k) :=
  ⟨⟨by decide, 5, by decide⟩, ⟨by decide, 6, by decide⟩, ⟨by decide, 8, by decide⟩⟩

/-- `act_kind` is not derivable: two IQ4 forms identical except for it are different values. -/
theorem act_kind_independent :
    ({ codeBits := 4, codeBias := 0, groupElems := 32, codebook := 1, hasOffset := false, actKind := 0 } : SForm) ≠
    { codeBits := 4, codeBias := 0, groupElems := 32, codebook := 1, hasOffset := false, actKind := 1 } := by
  decide

-- § grouped expert tier

/-- Scratch bytes per hit: gate/up activations (`2*FF` floats) plus the quantised intermediate. -/
def perHitScratch (ff : Nat) : Nat := 2 * ff * 4 + (ff / 32) * 34

/-- Scratch for `n` hits. -/
def hitScratch (n ff : Nat) : Nat := n * perHitScratch ff

/-- Scratch grows by exactly one per-hit amount per extra hit. -/
theorem hitScratch_succ (n ff : Nat) : hitScratch (n + 1) ff = hitScratch n ff + perHitScratch ff := by
  unfold hitScratch
  rw [Nat.add_mul, Nat.one_mul]

/-- Real geometry (FF = 640): 5800 bytes per hit, 58000 for ten hits. -/
theorem hitScratch_real : perHitScratch 640 = 5800 ∧ hitScratch 10 640 = 58000 := by decide

/-- Grouping: 4 launches per layer over 48 layers is 192, versus about 1548 launches per token (x100) for
5 launches per expert at 6.45 hits per layer. -/
theorem launch_counts : 4 * 48 = 192 ∧ 5 * 645 * 48 = 154800 := by decide

/-- Spans of consecutive slots (or consecutive output rows) of `n` bytes/floats are disjoint. -/
theorem slot_spans_disjoint (d d' n : Nat) (h : d < d') : d * n + n ≤ d' * n := by
  have e : (d + 1) * n = d * n + n := by rw [Nat.add_mul, Nat.one_mul]
  have h1 : (d + 1) * n ≤ d' * n := Nat.mul_le_mul_right n h
  omega

/-- Hits selected from `k` routed ids: those whose residency row is non-negative. -/
def hitCount (res : Nat → Int) (ids : List Nat) : Nat := (ids.filter (fun e => decide (0 ≤ res e))).length

/-- There are never more hits than routed entries, so a capacity of `k` hits suffices. -/
theorem hitCount_le (res : Nat → Int) (ids : List Nat) : hitCount res ids ≤ ids.length := by
  unfold hitCount
  exact List.length_filter_le _ _

/-- No routed entries means no hits. -/
theorem hitCount_nil (res : Nat → Int) : hitCount res [] = 0 := by
  simp [hitCount]

end Strata.SGemv

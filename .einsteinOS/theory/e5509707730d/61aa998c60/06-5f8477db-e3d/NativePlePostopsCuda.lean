/-!
# MODULE: native_ple_postops_cuda
SOURCES: src/kernels/cuda/native_ple_postops.cu
The opt-in native PLE post-operations: the gate kernel (512 threads, eight partial lanes per row), the
broadcast, the conv+residual kernel, and the span validation that precedes the launches. This module models
the thread/lane partitions, the tap indexing, the sign identity and the span checks.

## Source semantics
* The gate kernel reads `d = tid + j*512` for `j < 8`, masking `d >= N`; the eight partials are summed in
  a fixed order, then 16 warp partials are reduced.
* `float((s>0) - (s<0))` is the sign function used by the CPU-side formula.
* The conv taps read `history[c*9 + 3k]` for `k < 3` and `normalized[c]` for `k = 3`; weights are
  `weights[c*4 + k]`.
* Spans must be non-null, aligned and bounded; the only permitted output/input overlap is the exact alias
  `result == hidden` (each thread owns one element), plus `query == normalized` between outputs.

## Form template identifiers
* `laneIndex`   : gate_partial_lane_index
* `sgnC`, `sgnI`: sign_function_forms
* `overlaps`    : span_overlap
* `aliasPair`   : permitted_alias_pairs
* `convSum`     : conv_tap_accumulate
-/

namespace Strata.NativePlePostops

-- § constants

/-- `N`. -/
def N : Nat := 2560
/-- `H`. -/
def H : Nat := 4
/-- `D = N*H`. -/
def D : Nat := 10240
/-- History rows per channel. -/
def HISTORY : Nat := 9

/-- `D = N * H`. -/
theorem D_eq : N * H = D := by
  decide

-- § gate kernel thread partition

/-- Element read by thread `tid` in partial lane `j`. -/
def laneIndex (tid j : Nat) : Nat := tid + j * 512

/-- Eight lanes of 512 threads cover `N` elements (4096 >= 2560). -/
theorem gate_lanes_cover : N ≤ 8 * 512 := by
  decide

/-- Every element `d < N` is read by exactly the thread `d % 512` in lane `d / 512 < 8`. -/
theorem gate_element_owner (d : Nat) (hd : d < N) :
    d % 512 < 512 ∧ d / 512 < 8 ∧ laneIndex (d % 512) (d / 512) = d := by
  unfold laneIndex N at *
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- No two (thread, lane) pairs read the same element. -/
theorem gate_element_unique (tid j tid' j' : Nat) (ht : tid < 512) (ht' : tid' < 512)
    (h : laneIndex tid j = laneIndex tid' j') : tid = tid' ∧ j = j' := by
  unfold laneIndex at h
  refine ⟨?_, ?_⟩ <;> omega

/-- 512 threads form 16 warps, so the second-stage load `lane < 16 ? partials[lane] : 0` reads them all. -/
theorem gate_warp_partials : 512 / 32 = 16 := by
  decide

/-- A launch of `H` blocks gives one gate per hidden-channel stream. -/
theorem gate_blocks_eq_streams : H = 4 := by
  decide

-- § sign function

/-- Sign as the three-way comparison used in the source. -/
def sgnC (s : Int) : Int := if 0 < s then 1 else if s < 0 then -1 else 0

/-- Sign as `float((s>0) - (s<0))`. -/
def sgnI (s : Int) : Int := (if 0 < s then 1 else 0) - (if s < 0 then 1 else 0)

/-- Both spellings of the sign agree. -/
theorem sgn_forms_agree (s : Int) : sgnC s = sgnI s := by
  unfold sgnC sgnI
  split_ifs <;> omega

/-- The sign function is odd. -/
theorem sgnC_neg (s : Int) : sgnC (-s) = - sgnC s := by
  unfold sgnC
  split_ifs <;> omega

-- § broadcast and conv-residual launch geometry

/-- The broadcast kernel launches `D/256` blocks with no tail. -/
theorem broadcast_exact : D % 256 = 0 ∧ D / 256 = 40 := by
  decide

/-- Block/thread to element map of the elementwise kernels: `c = blk*256 + tid`. -/
theorem elementwise_roundtrip (c : Nat) (hc : c < D) :
    c / 256 < 40 ∧ c % 256 < 256 ∧ (c / 256) * 256 + c % 256 = c := by
  unfold D at hc
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- Distinct (block, thread) pairs own distinct elements, which is why `result == hidden` is safe. -/
theorem elementwise_unique (b t b' t' : Nat) (ht : t < 256) (ht' : t' < 256)
    (h : b * 256 + t = b' * 256 + t') : b = b' ∧ t = t' := by
  refine ⟨?_, ?_⟩ <;> omega

-- § conv tap indexing

/-- History index read by tap `k < 3`. -/
def tapHist (c k : Nat) : Nat := c * 9 + 3 * k

/-- Taps 0..2 stay inside the channel's history row of 9 and never touch the new row. -/
theorem tapHist_in_row (c k : Nat) (hk : k < 3) :
    c * 9 ≤ tapHist c k ∧ tapHist c k < c * 9 + 9 := by
  unfold tapHist
  refine ⟨?_, ?_⟩ <;> omega

/-- Distinct taps read distinct history rows. -/
theorem tapHist_inj (c k k' : Nat) (h : tapHist c k = tapHist c k') : k = k' := by
  unfold tapHist at h
  omega

/-- Tap history indices stay below `9*D` for `c < D`. -/
theorem tapHist_lt (c k : Nat) (hc : c < D) (hk : k < 3) : tapHist c k < 9 * D := by
  unfold tapHist D at *
  omega

/-- Weight index `c*4 + k` stays below `4*D` for `k < 4`. -/
theorem weight_index_lt (c k : Nat) (hc : c < D) (hk : k < 4) : c * 4 + k < 4 * D := by
  unfold D at *
  omega

/-- The tap accumulation order is `((t0+t1)+t2)+t3`. -/
def convSum : List Int → Int
  | [] => 0
  | t :: rest => rest.foldl (· + ·) t

/-- The four-tap sum is left-associated, exactly the order of the `__fadd_rn` chain. -/
theorem convSum_four (a b c d : Int) : convSum [a, b, c, d] = ((a + b) + c) + d := by
  rfl

-- § span validation

/-- A memory span: start address, byte length, required alignment. -/
structure Span where
  p : Nat
  bytes : Nat
  align : Nat

/-- Largest `uintptr_t`. -/
def MAXU : Nat := 18446744073709551615

/-- Two spans overlap iff each starts before the other ends. -/
def overlaps (a b : Span) : Prop := a.p < b.p + b.bytes ∧ b.p < a.p + a.bytes

/-- Overlap is symmetric. -/
theorem overlaps_symm (a b : Span) : overlaps a b ↔ overlaps b a := by
  unfold overlaps
  omega

/-- A span never overlaps one that starts after it ends. -/
theorem overlaps_disjoint_after (a b : Span) (h : a.p + a.bytes ≤ b.p) : ¬ overlaps a b := by
  unfold overlaps
  omega

/-- A non-null span with `p <= MAXU - bytes` cannot wrap the address space. -/
theorem valid_no_overflow (p bytes : Nat) (hp : p ≠ 0) (h : p ≤ MAXU - bytes) : p + bytes ≤ MAXU := by
  unfold MAXU at *
  omega

/-- The permitted exact alias: output 6 (`result`) against input 1 (`hidden`). -/
def aliasPair (i j : Nat) : Bool := i == 6 && j == 1

/-- Exactly the pair (6,1) is exempt from the output-versus-input check. -/
theorem aliasPair_iff (i j : Nat) : aliasPair i j = true ↔ (i = 6 ∧ j = 1) := by
  simp [aliasPair]

/-- Across the 7 outputs and 8 inputs the exemption applies to exactly one pair. -/
theorem alias_count :
    ((List.range 7).map (fun i => ((List.range 8).filter (fun j => aliasPair i j)).length)).sum = 1 := by
  decide

end Strata.NativePlePostops

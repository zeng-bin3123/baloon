/-!
# MODULE: native_router_lane_layout
Single-warp 512-expert / top-10 router adapted from llama.cpp's topk-moe: each lane holds 16 logits at stride
32, computes softmax in registers, then performs ten warp-wide argmax rounds with xor-shuffle, accumulating the
selected sum per winning lane and renormalising with a `2^-14` clamp. Host side validates alignment, stream and
span disjointness.

## Source semantics
* Lane `l`, register `i` holds expert `l + 32 i`; the map is a bijection `{0..31} × {0..15} ↔ {0..511}` with
  `expert / 32 = i`, `expert % 32 = l`.
* The winning expert's owner lane (`expert & 31`) clears the register `expert / 32` and records the id at
  output rank `rank`; lane `rank` captures the weight.
* Output weights `weights[lane] = selected / max(Σ, 2^-14)` for `lane < 10`.
* Validity: pointer non-null, 4-byte aligned, `address + bytes` does not wrap; spans pairwise disjoint.

## Form template identifiers
- `route`                    → `single_warp_route_kernel`
- `values[16]`               → `lane_registers`
- `native_router_top10`      → `launch_router`
- `valid`, `overlap`         → `span_valid`, `span_overlap`
-/

namespace Strata.NativeRouter

-- § lane / register layout

/-- Expert held by lane `l` in register `i`. -/
def expertOf (l i : Nat) : Nat := l + i * 32

/-- Registers cover 512 experts: 32 lanes × 16 registers. -/
theorem layout_size : 32 * 16 = 512 := by decide

/-- Every (lane, register) pair maps into `[0, 512)`. -/
theorem expertOf_lt (l i : Nat) (hl : l < 32) (hi : i < 16) : expertOf l i < 512 := by
  unfold expertOf
  omega

/-- Register of an expert. -/
theorem expert_register (l i : Nat) (hl : l < 32) : expertOf l i / 32 = i := by
  unfold expertOf
  omega

/-- Owner lane of an expert. -/
theorem expert_lane (l i : Nat) (hl : l < 32) : expertOf l i % 32 = l := by
  unfold expertOf
  omega

/-- Every expert id below 512 is held by exactly the lane/register given by `% 32` and `/ 32`. -/
theorem expert_roundtrip (e : Nat) : expertOf (e % 32) (e / 32) = e := by
  unfold expertOf
  omega

/-- `expert & 31 = expert % 32`, so the owner test `(expert & 31) == lane` is exact. -/
theorem owner_mask (e : Nat) : e &&& 31 = e % 32 := by
  have h := Nat.and_two_pow_sub_one_eq_mod e 5
  simpa using h

/-- Injectivity: two (lane, register) pairs holding the same expert are the same pair. -/
theorem expertOf_injective (l i l' i' : Nat) (hl : l < 32) (hl' : l' < 32)
    (h : expertOf l i = expertOf l' i') : l = l' ∧ i = i' := by
  unfold expertOf at h
  omega

-- § selection ranks

/-- Ten rounds, one output rank each; lanes `0..9` capture weights, `10 ≤ 32` so every rank has a lane. -/
theorem rank_has_lane : 10 ≤ 32 ∧ (10 : Nat) ≤ 16 * 32 := by decide

/-- Combine used by the xor butterfly: higher value wins, ties go to the lower expert id. -/
def comb (v i w j : Nat) : Nat × Nat :=
  if w > v ∨ (w = v ∧ j < i) then (w, j) else (v, i)

/-- The combine is symmetric on distinct experts, so the butterfly pairing order is irrelevant. -/
theorem comb_comm (v i w j : Nat) (hij : i ≠ j) : comb v i w j = comb w j v i := by
  unfold comb
  by_cases h1 : w > v ∨ (w = v ∧ j < i) <;> by_cases h2 : v > w ∨ (v = w ∧ i < j)
  · exfalso
    omega
  · rw [if_pos h1, if_neg h2]
  · rw [if_neg h1, if_pos h2]
  · exfalso
    omega

/-- Xor-partner of a lane under mask `m` is an involution (the butterfly pairs lanes mutually). -/
theorem xor_partner_involutive (l m : Nat) : (l ^^^ m) ^^^ m = l := by
  rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]

/-- Five butterfly masks 16, 8, 4, 2, 1 reduce 32 lanes: `2^5 = 32`. -/
theorem butterfly_depth : 16 + 8 + 4 + 2 + 1 = 31 ∧ 2 ^ 5 = 32 := by decide

-- § renormalisation

/-- Clamped denominator over a scaled numerator domain. -/
def clampDen (s eps : Nat) : Nat := max s eps

/-- The clamp keeps the reciprocal finite. -/
theorem clampDen_pos (s eps : Nat) (h : 0 < eps) : 0 < clampDen s eps := by
  unfold clampDen
  exact Nat.lt_of_lt_of_le h (Nat.le_max_right _ _)

/-- Selected-sum accumulation by winning lane is a regrouping of the same total (over exact integers). -/
theorem lane_grouped_sum (a b c d : Int) : (a + b) + (c + d) = a + b + c + d := by omega

-- § host validation

/-- Pointer validity: aligned and the span does not wrap the address space. -/
def spanValid (address bytes max : Nat) : Prop := address % 4 = 0 ∧ bytes ≤ max - address

/-- Overlap of two byte spans. -/
def spanOverlap (ap an bp bn : Nat) : Prop := ap < bp + bn ∧ bp < ap + an

/-- Overlap is symmetric, so the three pairwise checks are order-independent. -/
theorem spanOverlap_symm (ap an bp bn : Nat) : spanOverlap ap an bp bn ↔ spanOverlap bp bn ap an := by
  unfold spanOverlap
  exact And.comm

/-- Spans that end before the other begins do not overlap. -/
theorem spanOverlap_disjoint (ap an bp bn : Nat) (h : ap + an ≤ bp) : ¬ spanOverlap ap an bp bn := by
  unfold spanOverlap
  omega

/-- Logits 2048 bytes, ids 40 bytes, weights 40 bytes are laid out back to back without overlap. -/
theorem consecutive_spans_disjoint (p : Nat) :
    ¬ spanOverlap p 2048 (p + 2048) 40 ∧ ¬ spanOverlap (p + 2048) 40 (p + 2088) 40 := by
  constructor <;> unfold spanOverlap <;> omega

/-- Byte sizes: 512 floats = 2048, 10 ints / floats = 40. -/
theorem span_sizes : 512 * 4 = 2048 ∧ 10 * 4 = 40 := by decide

end Strata.NativeRouter

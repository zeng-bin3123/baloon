/-!
# MODULE: native_router_top10
Single-row 512-expert softmax / top-10 / renormalise router (llama.cpp topk-moe adaptation) with an
argument-validity guard (alignment, disjoint spans). One 32-lane warp holds 16 experts per lane.

## Source semantics
Expert `e` lives in lane `e % 32`, register `e / 32`. Ties break by lowest index. Winning-expert lane
accumulates the selected sum; the sum is clamped below by 2^-14. Output `weights[rank]` is read from lane `rank`.

## Form template identifiers
native_router_top10 -> router_top10_fixed; logits -> scores; ids -> expert_ids; weights -> expert_weights
-/
namespace Strata.NativeRouter

/-- Half-open span overlap, as in `overlap()` of native_router.cu. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap
  constructor <;> (intro h; omega)

/-- Expert id held in register `i` of lane `lane`. -/
def expertOf (lane i : Nat) : Nat := lane + i * 32

/-- The lane of an expert id is recovered by `% 32`. -/
theorem expert_lane (lane i : Nat) (h : lane < 32) : expertOf lane i % 32 = lane := by
  unfold expertOf; omega

/-- The register slot of an expert id is recovered by `/ 32`. -/
theorem expert_slot (lane i : Nat) (h : lane < 32) : expertOf lane i / 32 = i := by
  unfold expertOf; omega

/-- All 512 experts are covered by 32 lanes x 16 registers. -/
theorem expert_range (lane i : Nat) (hl : lane < 32) (hi : i < 16) : expertOf lane i < 512 := by
  unfold expertOf; omega

/-- Strict preference order: larger value, then smaller index. -/
def better (v i v' i' : Nat) : Prop := v > v' ∨ (v = v' ∧ i < i')

/-- The preference order is asymmetric, so the shuffle tree is well defined. -/
theorem better_asymm (v i v' i' : Nat) (h : better v i v' i') : ¬ better v' i' v i := by
  unfold better at *; omega

/-- The lower clamp of the selected sum. -/
theorem clamp_ge (s m : Nat) : m ≤ max s m := by omega

/-- Ten output ranks fit in the 32 lanes, so `rank == lane` reads every output. -/
theorem rank_lane (r : Nat) (h : r < 10) : r < 32 := by omega

end Strata.NativeRouter
